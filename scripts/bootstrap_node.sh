#!/bin/bash
# Stand up a fresh Vast 5090 node and start processing its shard of TorNet-Temporal
# directly into R2. Idempotent: re-running picks up where it left off.
#
# Required env vars:
#   R2_KEY, R2_SECRET     - Cloudflare R2 access keys (wrf-era5 bucket)
#   SHARD_IDX             - this node's shard (0..NUM_SHARDS-1)
#   NUM_SHARDS            - total number of nodes
# Optional:
#   WORKERS               - per-node worker count (default 32)
#
# Usage on a fresh Vast node:
#   curl -sSL https://raw.githubusercontent.com/FahrenheitResearch/ptx-radar-processor/main/scripts/bootstrap_node.sh \
#     | R2_KEY=... R2_SECRET=... SHARD_IDX=0 NUM_SHARDS=4 bash
set -euo pipefail

: "${R2_KEY:?must set R2_KEY}"
: "${R2_SECRET:?must set R2_SECRET}"
: "${SHARD_IDX:?must set SHARD_IDX}"
: "${NUM_SHARDS:?must set NUM_SHARDS}"
WORKERS="${WORKERS:-32}"

export R2_KEY R2_SECRET

echo "=== bootstrap shard $SHARD_IDX/$NUM_SHARDS ==="

# 1. system deps
apt-get update -qq
apt-get install -yqq cmake build-essential git zlib1g-dev curl python3-pip
pip3 install -q boto3 numpy huggingface_hub

# 2. build ptx-radar-processor (with the float / batch / exit-code fixes pushed upstream)
if [ ! -x /root/ptx-radar-processor/build/ptx-radar-processor ]; then
    cd /root
    git clone --recursive https://github.com/FahrenheitResearch/ptx-radar-processor.git
    cd ptx-radar-processor
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    cmake --build . -j
fi

# 3. fetch the reprocess script
mkdir -p /root
curl -sSL -o /root/reprocess_tornet.py \
    https://raw.githubusercontent.com/FahrenheitResearch/ptx-radar-processor/main/scripts/reprocess_tornet.py

# 4. fetch this shard's TorNet input npz files from R2 (much faster than HF)
#    if R2 mirror doesn't exist yet, fall back to HF
mkdir -p /root/data/tornet-temporal
cd /root
python3 - <<PY
import os, sys
from huggingface_hub import HfApi
from concurrent.futures import ThreadPoolExecutor
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
import urllib.request

shard = int(os.environ["SHARD_IDX"])
nshards = int(os.environ["NUM_SHARDS"])

r2 = boto3.client("s3",
    endpoint_url="https://de348a5190a7a5676bd2f6f19d506eec.r2.cloudflarestorage.com",
    aws_access_key_id=os.environ["R2_KEY"],
    aws_secret_access_key=os.environ["R2_SECRET"],
    config=Config(signature_version="s3v4", retries={"max_attempts": 5}),
    region_name="auto")

# Always use HF as the canonical shard pool — R2 mirror is incomplete and
# would cause a double-sharding bug if we slice from it.
api = HfApi()
info = api.dataset_info("deepguess/tornet-temporal")
events = sorted(s.rfilename for s in info.siblings if s.rfilename.endswith(".npz"))
use_r2 = False
print(f"HF dataset: {len(events)} events")

events.sort()
my_events = events[shard::nshards]
print(f"shard {shard}/{nshards}: {len(my_events)} events to fetch")

# Skip events already on disk OR already in R2 v2 output
def exists_v2(name):
    try:
        r2.head_object(Bucket="wrf-era5",
                       Key=f"stormlibre/datasets/tornet-temporal-v2/{name}/sequence.npz")
        return True
    except ClientError:
        return False

def fetch(key):
    # HF rfilename format: tornet_<id>_<tag>/sequence.npz  -> name is the first dir
    # R2 key format: stormlibre/datasets/tornet-temporal/tornet_<id>_<tag>/sequence.npz
    name = key.rstrip("/").split("/")[-2]
    # cheap skip
    if exists_v2(name):
        return name, "skip_done"
    dst_dir = f"/root/data/tornet-temporal/{name}"
    dst = f"{dst_dir}/sequence.npz"
    if os.path.exists(dst):
        return name, "cached"
    os.makedirs(dst_dir, exist_ok=True)
    if use_r2:
        r2.download_file("wrf-era5", key, dst)
    else:
        url = f"https://huggingface.co/datasets/deepguess/tornet-temporal/resolve/main/{key}"
        urllib.request.urlretrieve(url, dst)
    return name, "ok"

with ThreadPoolExecutor(max_workers=16) as pool:
    done = 0
    for fut in [pool.submit(fetch, k) for k in my_events]:
        try:
            name, st = fut.result()
        except Exception as e:
            print(f"  fetch err: {e}", flush=True)
        done += 1
        if done % 200 == 0:
            print(f"  fetched {done}/{len(my_events)}", flush=True)
print(f"done fetching shard")
PY

# 5. run reprocess for this shard, streaming straight to R2
cd /root
exec python3 reprocess_tornet.py \
    --workers "$WORKERS" \
    --shard "$SHARD_IDX" --num-shards "$NUM_SHARDS" \
    --upload-r2
