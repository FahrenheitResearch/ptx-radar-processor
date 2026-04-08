# Extraction Plan

## Minimal processor slice

The workstation app in `ptx-radar/` already has the pieces needed for bulk
Level II to PNG conversion. The clean extraction boundary is:

- keep:
  - `src/net/downloader.*`
  - `src/net/aws_nexrad.h`
  - `src/nexrad/level2.*`
  - `src/nexrad/level2_parser.*`
  - `src/nexrad/products.h`
  - `src/nexrad/stations.h`
  - `src/nexrad/sweep_data.h`
  - `src/cuda/renderer.*`
  - `src/cuda/preprocess.*`
  - `src/cuda/gpu_pipeline.*`
- drop:
  - `src/main.cpp`
  - `src/app.*`
  - `src/ui/*`
  - `src/render/gl_*`
  - `src/render/basemap.*`
  - `src/net/warnings.*`
  - `src/historic.*`
  - `src/cuda/volume3d.*`
  - `src/cuda/gpu_detection.*`
  - `src/cuda/gpu_tensor.*`

## End-to-end stages

1. List archive keys for a station/day from AWS.
2. Filter keys to the requested UTC range.
3. Download each Archive II file.
4. Decode BZip2 blocks into the message stream.
5. Build a renderable sweep working set.
6. Upload the selected sweep to the renderer.
7. Render to a device RGBA buffer.
8. Copy back to host and write PNG.

## What this repo now implements

- A top-level processor-only build.
- A headless CLI for station/time-range bulk conversion.
- WIC PNG writing on Windows.
- Fast lowest-sweep ingest when `--tilt 0`.
- CPU fallback for arbitrary tilt selection.

## Remaining speed work

- Add a direct render entrypoint that accepts GPU-ingest device pointers.
  - This removes the current device -> host -> device bounce after `ingestSweepGpu`.
- Overlap network download, BZip2 decode, GPU ingest, render, and PNG write in a staged queue.
- Add local-file manifests so already-downloaded Level II archives can be sharded across runs.
- Add optional raw RGBA output for cases where PNG compression becomes the bottleneck.
- Add multi-GPU sharding if the dataset build is spread across several cards.

## Practical recommendation

For a first dataset build, keep one binary, one station, one product, one tilt,
and render station-centered images. Once the file naming and metadata are locked,
the next pass should optimize the direct GPU ingest to render handoff.
