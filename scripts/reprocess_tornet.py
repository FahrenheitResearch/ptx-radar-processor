"""
Reprocess TorNet-Temporal events into a storm-centered multi-channel chip dataset
using ptx-radar-processor for fast GPU-side L2 -> 518x518 raw float32 extraction.

Important invariants:
  - Output T always matches the TorNet source sequence length and order exactly.
  - Missing products/tilts stay NaN with valid_mask=0.
  - Extra scans rendered inside the padded time window are recorded and ignored.
  - Corrupt or truncated .bin outputs fail the event instead of being skipped.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

TORNET_DIR = Path("/root/data/tornet-temporal")
OUTPUT_DIR = Path("/root/data/tornet-temporal-v2")
WORK_BASE = Path("/root/data/tornet_v2_work")
PTX_BIN = Path("/root/ptx-radar-processor/build/ptx-radar-processor")

# R2 upload (single-node or multi-node resumable mode)
R2_ENDPOINT = "https://de348a5190a7a5676bd2f6f19d506eec.r2.cloudflarestorage.com"
R2_BUCKET = "wrf-era5"
R2_PREFIX = "stormlibre/datasets/tornet-temporal-v2"
_R2_CLIENT = None

TIME_PAD_BEFORE_MIN = 1
TIME_PAD_AFTER_MIN = 2


def get_r2():
    global _R2_CLIENT
    if _R2_CLIENT is None:
        import boto3
        from botocore.client import Config

        _R2_CLIENT = boto3.client(
            "s3",
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=os.environ.get("R2_KEY", ""),
            aws_secret_access_key=os.environ.get("R2_SECRET", ""),
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 5, "mode": "adaptive"},
            ),
            region_name="auto",
        )
    return _R2_CLIENT


def is_not_found_client_error(exc) -> bool:
    code = str(exc.response.get("Error", {}).get("Code", ""))
    return code in {"404", "NoSuchKey", "NotFound"}


def r2_exists(event_name: str) -> bool:
    from botocore.exceptions import ClientError

    try:
        get_r2().head_object(
            Bucket=R2_BUCKET,
            Key=f"{R2_PREFIX}/{event_name}/sequence.npz",
        )
        return True
    except ClientError as e:
        if is_not_found_client_error(e):
            return False
        raise


def r2_upload(event_name: str, local_path: Path) -> None:
    get_r2().upload_file(
        str(local_path),
        R2_BUCKET,
        f"{R2_PREFIX}/{event_name}/sequence.npz",
    )


PRODUCTS = ["REF", "VEL", "SW", "ZDR", "CC", "KDP", "PHI"]
TILTS = [0, 1, 2, 3, 4]
CHANNELS = [f"{product}_T{tilt}" for product in PRODUCTS for tilt in TILTS]
N_CHANNELS = len(CHANNELS)

WIDTH = HEIGHT = 518
ZOOM = 444  # pixels per degree -> ~250 m/pixel in latitude
PIXEL_SIZE_M = 250.0
CHIP_KM = WIDTH * PIXEL_SIZE_M / 1000.0


L2_FN_RE = re.compile(r"^([A-Z]{4})(\d{8})_(\d{6})(?:_V\d{2})?$")

# Filename pattern produced by ptx-radar-processor:
#   <STATION>YYYYMMDD_HHMMSS_V06_<PROD>_T<tilt>.bin
FN_RE = re.compile(
    r"^(?P<stem>[A-Z]{4}\d{8}_\d{6}_V\d{2})_(?P<prod>[A-Z]+)_T(?P<tilt>\d+)\.bin$"
)


def parse_l2_fn_time(name: str):
    import datetime as dt

    match = L2_FN_RE.match(str(name))
    if not match:
        return None
    date_part, time_part = match.group(2), match.group(3)
    return dt.datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")


def parse_scan_time_value(scan_time: str):
    import datetime as dt

    scan_time = str(scan_time)
    parsed = parse_l2_fn_time(scan_time)
    if parsed is not None:
        return parsed
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return dt.datetime.strptime(scan_time, fmt)
        except ValueError:
            continue
    return None


def scan_time_key(scan_time: str) -> str | None:
    parsed = parse_scan_time_value(scan_time)
    if parsed is None:
        return None
    return parsed.strftime("%Y%m%d_%H%M%S")


def scan_time_iso(scan_time: str) -> str | None:
    parsed = parse_scan_time_value(scan_time)
    if parsed is None:
        return None
    return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")


def time_window(scan_times) -> tuple[str, str]:
    import datetime as dt

    parsed = []
    for scan_time in scan_times:
        parsed_time = parse_scan_time_value(scan_time)
        if parsed_time is not None:
            parsed.append(parsed_time)
    if not parsed:
        return None, None
    parsed.sort()
    start = parsed[0] - dt.timedelta(minutes=TIME_PAD_BEFORE_MIN)
    end = parsed[-1] + dt.timedelta(minutes=TIME_PAD_AFTER_MIN)
    return start.strftime("%Y-%m-%dT%H:%M:%SZ"), end.strftime("%Y-%m-%dT%H:%M:%SZ")


def clean_work_dir(work_dir: Path) -> None:
    if not work_dir.exists():
        return
    for child in work_dir.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except OSError:
            pass


def cleanup_output_dir(out_event_dir: Path, out_path: Path | None = None) -> None:
    if out_path is not None and out_path.exists():
        try:
            out_path.unlink()
        except OSError:
            pass
    try:
        out_event_dir.rmdir()
    except OSError:
        pass


def render_event(event_dir: Path, upload_r2: bool = False) -> tuple[str, dict]:
    if upload_r2 and r2_exists(event_dir.name):
        return event_dir.name, {"status": "skip_r2"}

    seq = event_dir / "sequence.npz"
    if not seq.exists():
        return event_dir.name, {"status": "no_seq"}

    with np.load(seq, allow_pickle=True) as z:
        source_scan_times = [str(scan_time) for scan_time in z["scan_times"]]
        radar = str(z["radar"].item())
        lat = float(z["lat"].item())
        lon = float(z["lon"].item())
        center_time = str(z["center_time"].item())
        mag = int(z["mag"].item())
        label = int(z["label"].item())
        category = str(z["category"].item())

    if not source_scan_times:
        return event_dir.name, {"status": "empty_scans"}

    source_scan_keys = []
    source_scan_isos = []
    for scan_time in source_scan_times:
        key = scan_time_key(scan_time)
        iso = scan_time_iso(scan_time)
        if key is None or iso is None:
            return event_dir.name, {"status": "bad_scan_times", "sample": scan_time}
        source_scan_keys.append(key)
        source_scan_isos.append(iso)

    start_iso, end_iso = time_window(source_scan_times)
    if start_iso is None or end_iso is None:
        return event_dir.name, {"status": "bad_scan_times", "sample": source_scan_times[0]}

    work_dir = WORK_BASE / event_dir.name
    work_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(PTX_BIN),
        "--station",
        radar,
        "--start",
        start_iso,
        "--end",
        end_iso,
        "--products",
        ",".join(PRODUCTS),
        "--tilts",
        ",".join(str(tilt) for tilt in TILTS),
        "--width",
        str(WIDTH),
        "--height",
        str(HEIGHT),
        "--zoom",
        str(ZOOM),
        "--center-lat",
        f"{lat:.6f}",
        "--center-lon",
        f"{lon:.6f}",
        "--raw-out",
        "--out",
        str(work_dir),
    ]

    last_err = ""
    last_out = ""
    last_rc = -1
    t0 = time.time()
    for attempt in range(3):
        clean_work_dir(work_dir)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            last_err = "subprocess timeout"
            time.sleep(2 + attempt * 5)
            continue

        last_rc = result.returncode
        last_err = (result.stderr or "")[-1000:]
        last_out = (result.stdout or "")[-500:]
        if last_rc == 0:
            break
        if any(
            transient in last_err
            for transient in (
                "Timeout was reached",
                "List request failed",
                "out of memory",
                "CUDA error",
                "cuModuleLoadData failed",
            )
        ):
            time.sleep(2 + attempt * 5)
            continue
        break
    cmd_dt = time.time() - t0

    bin_files = sorted(work_dir.glob("*.bin"))
    if not bin_files:
        return event_dir.name, {
            "status": "render_fail" if last_rc != 0 else "no_bins",
            "rc": last_rc,
            "stderr_tail": last_err,
            "stdout_tail": last_out,
        }

    by_scan_key: dict[str, dict[str, Path]] = {}
    rendered_stem_for_key: dict[str, str] = {}
    for bin_file in bin_files:
        match = FN_RE.match(bin_file.name)
        if not match:
            continue
        stem = match["stem"]
        key = scan_time_key(stem)
        if key is None:
            continue
        prev_stem = rendered_stem_for_key.get(key)
        if prev_stem is not None and prev_stem != stem:
            return event_dir.name, {"status": "duplicate_scan_key", "key": key}
        channel = f"{match['prod']}_T{match['tilt']}"
        by_scan_key.setdefault(key, {})[channel] = bin_file
        rendered_stem_for_key[key] = stem

    if not by_scan_key:
        return event_dir.name, {"status": "no_stems"}

    source_scan_key_set = set(source_scan_keys)
    T = len(source_scan_keys)
    data = np.full((T, N_CHANNELS, HEIGHT, WIDTH), np.nan, dtype=np.float16)
    valid_mask = np.zeros((T, N_CHANNELS), dtype=np.uint8)
    rendered_scan_mask = np.zeros((T,), dtype=np.uint8)
    corrupt_bins: list[str] = []

    for ti, scan_key in enumerate(source_scan_keys):
        channels_for_scan = by_scan_key.get(scan_key)
        if channels_for_scan is None:
            continue
        rendered_scan_mask[ti] = 1
        for ci, channel in enumerate(CHANNELS):
            bin_file = channels_for_scan.get(channel)
            if bin_file is None:
                continue
            try:
                arr = np.fromfile(bin_file, dtype=np.float32)
            except Exception as e:
                corrupt_bins.append(f"{bin_file.name}:{type(e).__name__}")
                continue
            if arr.size != HEIGHT * WIDTH:
                corrupt_bins.append(bin_file.name)
                continue
            data[ti, ci] = arr.reshape(HEIGHT, WIDTH).astype(np.float16)
            valid_mask[ti, ci] = 1

    if corrupt_bins:
        return event_dir.name, {
            "status": "corrupt_bins",
            "count": len(corrupt_bins),
            "sample": corrupt_bins[:5],
        }

    valid_total = int(valid_mask.sum())
    if valid_total == 0:
        return event_dir.name, {
            "status": "no_requested_bins",
            "requested_scans": T,
            "rendered_scans": len(by_scan_key),
        }

    rendered_scan_times = np.array(
        [scan_time_iso(rendered_stem_for_key[key]) for key in sorted(rendered_stem_for_key.keys())],
        dtype="<U20",
    )
    extra_scan_keys = [
        key for key in sorted(rendered_stem_for_key.keys()) if key not in source_scan_key_set
    ]
    extra_rendered_scan_times = np.array(
        [scan_time_iso(rendered_stem_for_key[key]) for key in extra_scan_keys],
        dtype="<U20",
    )

    out_event_dir = OUTPUT_DIR / event_dir.name
    out_event_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_event_dir / "sequence.npz"
    np.savez_compressed(
        out_path,
        data=data,
        valid_mask=valid_mask,
        scan_times=np.array(source_scan_isos),
        source_scan_times=np.array(source_scan_times),
        rendered_scan_times=rendered_scan_times,
        extra_rendered_scan_times=extra_rendered_scan_times,
        rendered_scan_mask=rendered_scan_mask,
        center_time=center_time,
        lat=lat,
        lon=lon,
        radar=radar,
        mag=mag,
        label=label,
        category=category,
        channels=np.array(CHANNELS),
        pixel_size_m=PIXEL_SIZE_M,
        chip_km=CHIP_KM,
    )

    clean_work_dir(work_dir)
    try:
        work_dir.rmdir()
    except OSError:
        pass

    out_size_mb = out_path.stat().st_size / 1024 / 1024

    if upload_r2:
        try:
            if r2_exists(event_dir.name):
                cleanup_output_dir(out_event_dir, out_path)
                return event_dir.name, {"status": "skip_r2"}
            r2_upload(event_dir.name, out_path)
            cleanup_output_dir(out_event_dir, out_path)
        except Exception as e:
            return event_dir.name, {"status": "r2_upload_fail", "err": str(e)[:200]}

    return event_dir.name, {
        "status": "ok",
        "T": T,
        "valid_channels": valid_total,
        "missing_channels": T * N_CHANNELS - valid_total,
        "matched_scans": int(rendered_scan_mask.sum()),
        "missing_scans": int(T - rendered_scan_mask.sum()),
        "extra_scans_ignored": int(len(extra_scan_keys)),
        "render_seconds": cmd_dt,
        "out_size_mb": out_size_mb,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument(
        "--shard",
        type=int,
        default=0,
        help="this node's shard index (0..num-shards-1)",
    )
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument(
        "--upload-r2",
        action="store_true",
        help="upload sequence.npz to R2 and delete local copy",
    )
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_BASE.mkdir(parents=True, exist_ok=True)

    events = sorted(
        [
            path
            for path in TORNET_DIR.iterdir()
            if path.is_dir() and path.name.startswith("tornet_")
        ]
    )
    if args.shuffle:
        import random

        random.seed(42)
        random.shuffle(events)
    if args.num_shards > 1:
        events = events[args.shard :: args.num_shards]
    if args.limit:
        events = events[: args.limit]

    print(
        f"reprocessing {len(events)} events with {args.workers} workers "
        f"(shard {args.shard}/{args.num_shards}, upload_r2={args.upload_r2})"
    )

    t0 = time.time()
    ok = 0
    fail = 0
    rendered_ok = 0
    total_render = 0.0
    total_size_mb = 0.0
    total_valid_channels = 0
    total_matched_scans = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(render_event, event, args.upload_r2): event for event in events}
        for i, fut in enumerate(as_completed(futs), start=1):
            event_id, stats = fut.result()
            status = stats.get("status")
            if status == "ok":
                ok += 1
                rendered_ok += 1
                total_render += stats["render_seconds"]
                total_size_mb += stats["out_size_mb"]
                total_valid_channels += stats["valid_channels"]
                total_matched_scans += stats["matched_scans"]
            elif status == "skip_r2":
                ok += 1
            else:
                fail += 1
                if fail <= 5:
                    print(f"  FAIL {event_id}: {stats}", flush=True)

            if i % 5 == 0 or i == len(events):
                elapsed = time.time() - t0
                rate = i / max(0.1, elapsed)
                eta = (len(events) - i) / max(0.1, rate)
                print(
                    f"  [{i}/{len(events)}] ok={ok} fail={fail} "
                    f"avg_render={total_render / max(1, rendered_ok):.1f}s "
                    f"avg_size={total_size_mb / max(1, rendered_ok):.1f}MB "
                    f"matched_scans={total_matched_scans // max(1, rendered_ok)} "
                    f"valid_ch={total_valid_channels // max(1, rendered_ok)} "
                    f"rate={rate:.2f}/s eta={eta / 60:.0f}min",
                    flush=True,
                )

    print(f"\nDONE - {ok} ok, {fail} fail in {(time.time() - t0) / 60:.1f} min")
    print(f"  total output: {total_size_mb / 1024:.2f} GB at {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
