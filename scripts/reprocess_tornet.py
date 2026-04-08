"""
Reprocess TorNet-Temporal events into a richer multi-channel chip dataset
using ptx-radar-processor for fast GPU-side L2 → 518×518 raw float32 extraction
centered on the actual storm location.

For each TorNet event:
  1. Read scan_times, radar, lat, lon from sequence.npz
  2. Compute the tightest UTC start/end window covering all scan_times
  3. Invoke ptx-radar-processor once per event:
       --station <radar> --start <iso> --end <iso>
       --products REF,VEL,SW,ZDR,CC,KDP --tilts 0,1,2,3,4
       --width 518 --height 518 --zoom 444 (250 m/pixel at mid-latitudes)
       --center-lat <event lat> --center-lon <event lon>
       --raw-out --out <event work dir>
  4. Group the resulting .bin files by scan timestamp → build a
     (T, 30, 518, 518) float16 array
  5. Save sequence_v2.npz alongside the original

Output schema (sequence_v2.npz):
    data        (T, 30, 518, 518) float16   30 = 6 products × 5 elevations
    valid_mask  (T, 30) uint8                1 if channel was rendered, 0 if missing
    scan_times  (T,) str                     copied from TorNet
    center_time str                          copied
    lat         float64                      event center lat
    lon         float64                      event center lon
    radar       str                          NEXRAD ICAO
    mag         int                          EF magnitude
    label       int                          binary tornado label
    category    str                          TOR / WRN / NUL
    channels    list[str]                    e.g. ['REF_T0','REF_T1',...,'KDP_T4']
    pixel_size_m float                       250.0
    chip_km     float                        129.5 (518 * 250 / 1000)

Run on the Vast node:
    python3 reprocess_tornet.py [--workers 4] [--limit N]
"""
from __future__ import annotations
import argparse, os, re, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

TORNET_DIR = Path("/root/data/tornet-temporal")
OUTPUT_DIR = Path("/root/data/tornet-temporal-v2")
WORK_BASE = Path("/root/data/tornet_v2_work")
PTX_BIN = Path("/root/ptx-radar-processor/build/ptx-radar-processor")

# R2 upload (multi-node sharded mode)
R2_ENDPOINT = "https://de348a5190a7a5676bd2f6f19d506eec.r2.cloudflarestorage.com"
R2_BUCKET = "wrf-era5"
R2_PREFIX = "stormlibre/datasets/tornet-temporal-v2"
_R2_CLIENT = None

def get_r2():
    global _R2_CLIENT
    if _R2_CLIENT is None:
        import boto3
        from botocore.client import Config
        _R2_CLIENT = boto3.client(
            "s3", endpoint_url=R2_ENDPOINT,
            aws_access_key_id=os.environ.get("R2_KEY", ""),
            aws_secret_access_key=os.environ.get("R2_SECRET", ""),
            config=Config(signature_version="s3v4",
                          retries={"max_attempts": 5, "mode": "adaptive"}),
            region_name="auto",
        )
    return _R2_CLIENT

def r2_exists(event_name: str) -> bool:
    from botocore.exceptions import ClientError
    try:
        get_r2().head_object(Bucket=R2_BUCKET, Key=f"{R2_PREFIX}/{event_name}/sequence.npz")
        return True
    except ClientError:
        return False

def r2_upload(event_name: str, local_path: Path) -> None:
    get_r2().upload_file(str(local_path), R2_BUCKET, f"{R2_PREFIX}/{event_name}/sequence.npz")

PRODUCTS = ["REF", "VEL", "SW", "ZDR", "CC", "KDP", "PHI"]
TILTS = [0, 1, 2, 3, 4]
CHANNELS = [f"{p}_T{t}" for p in PRODUCTS for t in TILTS]
N_CHANNELS = len(CHANNELS)  # 35 (target was 40 — all 7 NEXRAD L2 products × 5 elevations)

WIDTH = HEIGHT = 518
ZOOM = 444  # pixels per degree → ~250 m/pixel at 35°N
PIXEL_SIZE_M = 250.0
CHIP_KM = WIDTH * PIXEL_SIZE_M / 1000.0  # 129.5


L2_FN_RE = re.compile(r"^([A-Z]{4})(\d{8})_(\d{6})(?:_V\d{2})?$")


def parse_l2_fn_time(name: str):
    """`KDIX20220218_090700_V06` → datetime(2022, 02, 18, 09, 07, 00)."""
    import datetime as dt
    m = L2_FN_RE.match(str(name))
    if not m:
        return None
    d, t = m.group(2), m.group(3)
    return dt.datetime.strptime(d + t, "%Y%m%d%H%M%S")


def time_window(scan_times) -> tuple[str, str]:
    """Tightest [start, end] window covering all scan_times, padded by ±1 min."""
    import datetime as dt
    parsed = []
    for s in scan_times:
        t = parse_l2_fn_time(s)
        if t is None:
            # Fallback: TorNet sometimes uses 'YYYY-MM-DD HH:MM:SS'
            try:
                t = dt.datetime.strptime(str(s), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
        parsed.append(t)
    if not parsed:
        return None, None
    parsed.sort()
    start = parsed[0] - dt.timedelta(minutes=1)
    end = parsed[-1] + dt.timedelta(minutes=2)
    return start.strftime("%Y-%m-%dT%H:%M:%SZ"), end.strftime("%Y-%m-%dT%H:%M:%SZ")


# Filename pattern produced by ptx-radar-processor:
#   <STATION>YYYYMMDD_HHMMSS_V06_<PROD>_T<tilt>.bin
FN_RE = re.compile(r"^(?P<stem>[A-Z]{4}\d{8}_\d{6}_V\d{2})_(?P<prod>[A-Z]+)_T(?P<tilt>\d+)\.bin$")


def render_event(event_dir: Path, upload_r2: bool = False) -> tuple[str, dict]:
    """Process one TorNet event end-to-end. Returns (event_id, stats_dict)."""
    if upload_r2 and r2_exists(event_dir.name):
        return event_dir.name, {"status": "skip_r2"}
    seq = event_dir / "sequence.npz"
    if not seq.exists():
        return event_dir.name, {"status": "no_seq"}

    z = np.load(seq, allow_pickle=True)
    scan_times = list(z["scan_times"])
    radar = str(z["radar"])
    lat = float(z["lat"])
    lon = float(z["lon"])

    if len(scan_times) == 0:
        return event_dir.name, {"status": "empty_scans"}

    start_iso, end_iso = time_window(scan_times)
    if start_iso is None:
        return event_dir.name, {"status": "bad_scan_times", "sample": str(scan_times[0])}

    work_dir = WORK_BASE / event_dir.name
    work_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(PTX_BIN),
        "--station", radar,
        "--start", start_iso,
        "--end", end_iso,
        "--products", ",".join(PRODUCTS),
        "--tilts", ",".join(str(t) for t in TILTS),
        "--width", str(WIDTH), "--height", str(HEIGHT),
        "--zoom", str(ZOOM),
        "--center-lat", f"{lat:.6f}",
        "--center-lon", f"{lon:.6f}",
        "--raw-out",
        "--out", str(work_dir),
    ]

    # Retry up to 3x for transient S3/CUDA contention failures.
    t0 = time.time()
    last_err = ""
    last_out = ""
    last_rc = -1
    for attempt in range(3):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        except subprocess.TimeoutExpired:
            last_err = "subprocess timeout"
            time.sleep(2 + attempt * 5)
            continue
        last_rc = result.returncode
        last_err = (result.stderr or "")[-1000:]  # tail not head — real error is last
        last_out = (result.stdout or "")[-500:]
        if last_rc == 0:
            break
        # Retry on known transient causes
        if any(s in last_err for s in ("Timeout was reached", "List request failed",
                                        "out of memory", "CUDA error", "cuModuleLoadData failed")):
            time.sleep(2 + attempt * 5)
            continue
        break
    cmd_dt = time.time() - t0

    # Don't trust the exit code: processor exits rc=1 if ANY product/tilt
    # combo failed (e.g., KDP often absent), even when most channels rendered
    # fine. Check the disk instead.
    bin_files = sorted(work_dir.glob("*.bin"))
    if not bin_files:
        return event_dir.name, {
            "status": "render_fail" if last_rc != 0 else "no_bins",
            "rc": last_rc,
            "stderr_tail": last_err,
            "stdout_tail": last_out,
        }

    by_stem: dict[str, dict[str, Path]] = {}
    for f in bin_files:
        m = FN_RE.match(f.name)
        if not m:
            continue
        stem = m["stem"]
        ch = f"{m['prod']}_T{m['tilt']}"
        by_stem.setdefault(stem, {})[ch] = f

    # Sort stems by their embedded timestamp to match scan order
    stems_sorted = sorted(by_stem.keys())  # lex sort = chronological for these names
    T = len(stems_sorted)

    if T == 0:
        return event_dir.name, {"status": "no_stems"}

    # Allocate output: float16 to halve storage
    data = np.full((T, N_CHANNELS, HEIGHT, WIDTH), np.nan, dtype=np.float16)
    valid_mask = np.zeros((T, N_CHANNELS), dtype=np.uint8)
    out_scan_times: list[str] = []

    for ti, stem in enumerate(stems_sorted):
        # Reconstruct ISO timestamp from filename: KTLXyyyymmdd_HHMMSS_V06
        m = re.match(r"^[A-Z]{4}(\d{8})_(\d{6})_V\d{2}$", stem)
        if m:
            d, t = m.group(1), m.group(2)
            iso = f"{d[:4]}-{d[4:6]}-{d[6:]}T{t[:2]}:{t[2:4]}:{t[4:]}Z"
        else:
            iso = stem
        out_scan_times.append(iso)

        for ci, ch in enumerate(CHANNELS):
            f = by_stem[stem].get(ch)
            if f is None:
                continue
            try:
                arr = np.fromfile(f, dtype=np.float32)
                if arr.size != HEIGHT * WIDTH:
                    continue
                data[ti, ci] = arr.reshape(HEIGHT, WIDTH).astype(np.float16)
                valid_mask[ti, ci] = 1
            except Exception:
                pass

    # Save sequence_v2.npz
    out_event_dir = OUTPUT_DIR / event_dir.name
    out_event_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_event_dir / "sequence.npz"
    np.savez_compressed(
        out_path,
        data=data,
        valid_mask=valid_mask,
        scan_times=np.array(out_scan_times),
        center_time=str(z["center_time"]),
        lat=lat, lon=lon,
        radar=radar,
        mag=int(z["mag"]),
        label=int(z["label"]),
        category=str(z["category"]),
        channels=np.array(CHANNELS),
        pixel_size_m=PIXEL_SIZE_M,
        chip_km=CHIP_KM,
    )

    # Clean up the work dir's .bin files
    for f in bin_files:
        try: f.unlink()
        except OSError: pass
    try: work_dir.rmdir()
    except OSError: pass

    valid_total = int(valid_mask.sum())
    out_size_mb = out_path.stat().st_size / 1024 / 1024

    if upload_r2:
        try:
            r2_upload(event_dir.name, out_path)
            out_path.unlink()
            try: out_event_dir.rmdir()
            except OSError: pass
        except Exception as e:
            return event_dir.name, {"status": "r2_upload_fail", "err": str(e)[:200]}

    return event_dir.name, {
        "status": "ok",
        "T": T,
        "valid_channels": valid_total,
        "missing_channels": T * N_CHANNELS - valid_total,
        "render_seconds": cmd_dt,
        "out_size_mb": out_size_mb,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--shard", type=int, default=0, help="this node's shard index (0..num-shards-1)")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--upload-r2", action="store_true",
                    help="upload sequence.npz to R2 and delete local copy")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_BASE.mkdir(parents=True, exist_ok=True)

    events = sorted([p for p in TORNET_DIR.iterdir() if p.is_dir() and p.name.startswith("tornet_")])
    if args.shuffle:
        import random
        random.seed(42)
        random.shuffle(events)
    # Shard slice — deterministic mod stride means each event is owned by exactly one node
    if args.num_shards > 1:
        events = events[args.shard :: args.num_shards]
    if args.limit:
        events = events[: args.limit]
    print(f"reprocessing {len(events)} events with {args.workers} workers "
          f"(shard {args.shard}/{args.num_shards}, upload_r2={args.upload_r2})")

    t0 = time.time()
    ok = fail = 0
    total_render = 0.0
    total_size_mb = 0.0
    total_valid_channels = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(render_event, ev, args.upload_r2): ev for ev in events}
        for i, fut in enumerate(as_completed(futs), start=1):
            ev_id, stats = fut.result()
            status = stats.get("status")
            if status == "ok":
                ok += 1
                total_render += stats["render_seconds"]
                total_size_mb += stats["out_size_mb"]
                total_valid_channels += stats["valid_channels"]
            elif status == "skip_r2":
                ok += 1  # already in R2, count as success
            else:
                fail += 1
                if fail <= 5:
                    print(f"  FAIL {ev_id}: {stats}")
            if i % 5 == 0 or i == len(events):
                el = time.time() - t0
                rate = i / max(0.1, el)
                eta = (len(events) - i) / max(0.1, rate)
                print(
                    f"  [{i}/{len(events)}] ok={ok} fail={fail} "
                    f"avg_render={total_render/max(1,ok):.1f}s "
                    f"avg_size={total_size_mb/max(1,ok):.1f}MB "
                    f"valid_ch={total_valid_channels//max(1,ok)}/{N_CHANNELS*13} "
                    f"rate={rate:.2f}/s eta={eta/60:.0f}min",
                    flush=True,
                )

    print(f"\nDONE — {ok} ok, {fail} fail in {(time.time()-t0)/60:.1f} min")
    print(f"  total output: {total_size_mb/1024:.2f} GB at {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
