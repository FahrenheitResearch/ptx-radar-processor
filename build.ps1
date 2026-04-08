param(
    [string]$BuildDir = "build",
    [int]$Jobs = 4
)

$ErrorActionPreference = "Stop"

$vsDevCandidates = @(
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat",
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat",
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\Common7\Tools\VsDevCmd.bat"
)

$vsDevCmd = $vsDevCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $vsDevCmd) {
    throw "Could not find VsDevCmd.bat. Install Visual Studio Build Tools with C++ support."
}

$nvcc = (Get-Command nvcc -ErrorAction Stop).Source

$cmakeConfigure = @(
    "cmake",
    "-S", ".",
    "-B", $BuildDir,
    "-G", "Ninja",
    "-DCMAKE_C_COMPILER=cl",
    "-DCMAKE_CXX_COMPILER=cl",
    "-DCMAKE_CUDA_COMPILER=`"$nvcc`""
) -join " "

$cmakeBuild = @(
    "cmake",
    "--build", $BuildDir,
    "-j", $Jobs
) -join " "

$command = "`"$vsDevCmd`" -arch=amd64 && $cmakeConfigure && $cmakeBuild"
cmd /c $command
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
