# ptx-radar-processor

This repo is now the processor-only extraction around [`ptx-radar`](./ptx-radar).
`ptx-radar/` is tracked as a git submodule pointing at the upstream repo.
Everything outside that folder is focused on one job: download NEXRAD Level II
archive volumes and turn them into PNGs with the same CUDA/PTX render path.

Clone with submodules:

```powershell
git clone --recurse-submodules https://github.com/FahrenheitResearch/ptx-radar-processor.git
```

## What stays

- Archive listing and download helpers from `ptx-radar/src/net/`
- Archive II decode from `ptx-radar/src/nexrad/level2_parser.*`
- CUDA/PTX ingest and render kernels from `ptx-radar/src/cuda/`

## What is intentionally cut out

- GLFW / ImGui workstation shell
- OpenGL display textures
- warnings, live polling UI state, historic player UI, basemap UI
- 3D / cross-section / alert workflow features

## Current processor behavior

- downloads one station's archive files for a UTC time range
- filters to one product/tilt selection
- renders one PNG per matching Level II volume
- writes PNGs with a headless Windows WIC path instead of a GL texture
- defaults to the fast path for lowest-sweep renders when `--tilt 0`

## Build

```powershell
.\build.ps1
```

Output binary:

```powershell
.\build\ptx-radar-processor.exe
```

Manual build from a Developer Command Prompt also works:

```powershell
cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER=cl -DCMAKE_CXX_COMPILER=cl
cmake --build build -j 4
```

`build.ps1` is the easier path from normal PowerShell because it bootstraps
`VsDevCmd.bat` before invoking CMake.

## Usage

```powershell
.\build\ptx-radar-processor.exe `
  --station KTLX `
  --start 2025-03-30T20:00:00Z `
  --end   2025-03-30T21:00:00Z `
  --out .\output\ktlx-ref `
  --product REF `
  --tilt 0 `
  --width 1024 `
  --height 1024 `
  --zoom 180
```

Useful flags:

- `--product REF|VEL|SW|ZDR|CC|KDP|PHI`
- `--tilt N`
- `--threshold VALUE`
- `--center-lat LAT --center-lon LON`
- `--limit N`
- `--overwrite`
- `--no-dealias`
- `--cpu-only`

## Notes

- `--tilt 0` uses the fast lowest-sweep ingest path by default, then falls back
  to full CPU parse if the requested product is not present.
- Higher tilts currently use the CPU parse path so tilt selection stays exact.
- The next speed win is removing the temporary host bounce after GPU ingest and
  rendering directly from the GPU-ingest device buffers.

See [`docs/extraction-plan.md`](./docs/extraction-plan.md) for the extraction
breakdown and the remaining work to make this a fully optimized bulk pipeline.
