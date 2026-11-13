# One command to run memagent at the best settings for this Windows PC.
#
#   run.cmd                    detect CPU, RAM and GPUs, pick a preset, set up, run
#   run.cmd gpu12              force a preset (list: run.cmd -List)
#   run.cmd -DryRun            show the preset and the config, change nothing
#   run.cmd auto -- --offline  anything after -- goes to memagent
#
# Works in a release folder (run.cmd next to memagent.exe) and in a clone
# (scripts\run.cmd, with build\memagent.exe). The Windows release's llama.cpp runs on the
# CPU, so GPU presets answer through memagent's Python engine (PyTorch with CUDA) and
# train with the Python trainer; this installs them when they are missing. Memory lives
# in data\<profile>\ and the config in run\<preset>.ini.
#
# The presets and why: see scripts/run.sh (the same table) and bench/README.md,
# "One-shot presets". Written for Windows PowerShell 5.1, which every Windows 10/11 has.

$ErrorActionPreference = 'Stop'

$Preset = 'auto'; $DryRun = $false; $List = $false; $Extra = @()
for ($i = 0; $i -lt $args.Count; $i++) {
    $a = [string]$args[$i]
    if ($a -eq '--') { if ($i + 1 -lt $args.Count) { $Extra = $args[($i + 1)..($args.Count - 1)] }; break }
    elseif ($a -eq '-DryRun' -or $a -eq '--dry-run') { $DryRun = $true }
    elseif ($a -eq '-List' -or $a -eq '--list') { $List = $true }
    elseif ($a -eq '-h' -or $a -eq '--help' -or $a -eq '-Help') {
        Get-Content $PSCommandPath | Select-Object -Skip 1 -First 14 | ForEach-Object { $_ -replace '^# ?', '' }; exit 0 }
    else { $Preset = $a }
}

$Presets = [ordered]@{
    'cpu-lite' = 'qwen3-0.6b on CPU (42 t/s on 4 cores; 93/96 in the agent benchmark), /sleep trains on 1 core. Under 4 GB RAM: tiny (SmolLM2-135M), a demo.'
    'cpu'      = 'qwen3-0.6b on CPU: 93/96 in the agent benchmark (1.5B: 90, LFM3-1.7B: 89); /sleep trains on a quarter of the cores.'
    'gpu8'     = 'one 8-11 GB GPU: gemma4-e2b (0.90 from the weights, 16 s a deep sleep) with a CUDA llama.cpp; else small (Qwen2.5-1.5B) through Python, retrieval only.'
    'gpu12'    = 'medium (Qwen3-4B) on one 12-39 GB GPU, answering and training on it: the smallest size where trained memory helped.'
    'gpu40'    = 'large (Magistral-Small 24B) on one 40 GB+ GPU, answering and training on it (~43 GB at peak).'
    '2gpu'     = 'two GPUs: answer on GPU 0, train on GPU 1; large if both have 24 GB+, else medium if 12 GB+, else small.'
}
if ($List) { foreach ($k in $Presets.Keys) { '  {0,-9} {1}' -f $k, $Presets[$k] }; exit 0 }

# ---------------------------------------------------------------- where things are
$Here = $PSScriptRoot
if (Test-Path (Join-Path $Here 'memagent.exe')) { $Root = $Here; $Bin = Join-Path $Root 'memagent.exe' }
else { $Root = Split-Path $Here -Parent; $Bin = Join-Path $Root 'build\memagent.exe' }
Set-Location $Root

# ---------------------------------------------------------------- the machine
$Cores = [Environment]::ProcessorCount
$RamGB = 8
try { $RamGB = [int]((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB) } catch { }
$GpuMB = @()
$Smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($Smi) {
    try {
        $GpuMB = @(& $Smi.Source --query-gpu=memory.total --format=csv,noheader,nounits 2>$null |
            Where-Object { $_ -match '\d' } | ForEach-Object { [int]($_.Trim()) })
    } catch { $GpuMB = @() }
}
$NGpu = $GpuMB.Count
$MinGB = 0
if ($NGpu -gt 0) { $MinGB = [int][math]::Floor((($GpuMB | Measure-Object -Minimum).Minimum) / 1024) }
$gpuNote = ''
if ($NGpu -gt 0) { $gpuNote = ", smallest $MinGB GB" }
Write-Host "[run] machine: $Cores cores, $RamGB GB RAM, $NGpu NVIDIA GPU(s)$gpuNote"

if ($Preset -eq 'auto') {
    if ($NGpu -ge 2 -and $MinGB -ge 7) { $Preset = '2gpu' }
    elseif ($NGpu -ge 1 -and $MinGB -ge 40) { $Preset = 'gpu40' }
    elseif ($NGpu -ge 1 -and $MinGB -ge 12) { $Preset = 'gpu12' }
    elseif ($NGpu -ge 1 -and $MinGB -ge 7) { $Preset = 'gpu8' }
    elseif ($Cores -ge 4 -and $RamGB -ge 8) { $Preset = 'cpu' }
    else { $Preset = 'cpu-lite' }
}
if (-not $Presets.Contains($Preset)) {
    Write-Error "unknown preset '$Preset' (have: $($Presets.Keys -join ', '))"; exit 2 }

# ---------------------------------------------------------------- the preset
$TrainThreads = 1; $Infer = -1; $Train = -1; $Trainer = 'llama'; $Engine = 'llama'; $Gpu = $false; $Auto = 1
switch ($Preset) {
    'cpu-lite' { if ($RamGB -ge 4) { $Profile_ = 'qwen3-0.6b'; $Auto = 0 } else { $Profile_ = 'tiny' } }
    'cpu'      {
        $Profile_ = 'qwen3-0.6b'; $Auto = 0
        $TrainThreads = [math]::Max(1, [math]::Floor($Cores / 4))
    }
    'gpu8'     { $Profile_ = 'small';  $Auto = 0; $Infer = 0; $Train = 0; $Gpu = $true }
    'gpu12'    { $Profile_ = 'medium'; $Infer = 0; $Train = 0; $Gpu = $true }
    'gpu40'    { $Profile_ = 'large';  $Infer = 0; $Train = 0; $Gpu = $true }
    '2gpu'     {
        $Infer = 0; $Train = 1; $Gpu = $true
        if ($MinGB -ge 24) { $Profile_ = 'large' } elseif ($MinGB -ge 12) { $Profile_ = 'medium' } else { $Profile_ = 'small' }
    }
}
if ($Gpu) {
    $Trainer = 'python'
    # a CUDA llama.cpp (ggml-cuda.dll next to the exe) answers on the GPU itself; the
    # release's is CPU-only, so the Python engine answers on the GPU instead
    if (-not (Test-Path (Join-Path (Split-Path $Bin -Parent) 'ggml-cuda.dll'))) { $Engine = 'python' }
}
# gpu8 with llama.cpp on the GPU: Gemma 4 E2B (2.8 GB) answers and trains in-process
if ($Preset -eq 'gpu8' -and $Engine -eq 'llama') { $Profile_ = 'gemma4-e2b'; $Auto = 1; $Trainer = 'llama' }
# the judge (OpenJev-Flash-9B, 5.6 GB): only where llama.cpp answers on the GPU and there is room
$Judge = 'off'
if ($Gpu -and $Engine -eq 'llama' -and $MinGB -ge 16) { $Judge = 'openjev-flash-9b' }
$where = 'CPU'; if ($Infer -ge 0) { $where = "GPU $Infer" }
$twhere = "CPU ($TrainThreads thread(s))"; if ($Train -ge 0) { $twhere = "GPU $Train" }
Write-Host "[run] preset ${Preset}: $($Presets[$Preset])"
if ($Auto -eq 0) { Write-Host "[run] background training off for $Profile_ (it scored below retrieval); /sleep trains on demand, auto=1 in run\$Preset.ini turns it on" }
Write-Host "[run] profile $Profile_, answer on $where, train on $twhere with the $Trainer trainer; $Engine engine"

$Python = 'python'
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { $py = Get-Command py -ErrorAction SilentlyContinue }
if ($py) { $Python = $py.Source }

$srcDir = '; src_dir: the converters in cache\llama.cpp'
if (Test-Path (Join-Path $Root 'llama.cpp')) { $srcDir = 'src_dir=..\llama.cpp' }
$AutoNote = '; 1: a fast LoRA after each new fact, a deep sleep after 8'
if ($Auto -eq 0) { $AutoNote = '; 0: trained memory lost to retrieval in the agent at 1.5B and at Qwen3-0.6B (81 against 93 of 96, bench/README.md); /sleep still trains on demand' }
$stamp = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mmZ')
$Ini = @"
; written by run.ps1 for preset $Preset ($stamp); rerun it to regenerate.
; Paths are relative to this file. Settings not listed take memagent's defaults (memagent.ini).
[agent]
profile=$Profile_
engine=$Engine
models=..\models.json
priors=..\priors.md
data_dir=..\data\$Profile_
repo_dir=..
python=$Python
rag_k=3
auto_provision=1

[llama]
threads=0
gpu_layers=-1
$srcDir

[gpu]
infer=$Infer
train=$Train

[consolidate]
trainer=$Trainer
threads=$TrainThreads
$AutoNote
auto=$Auto
tiers=auto

[retrieval]
cache_dir=..\cache
; retrieve again for each name in the first hits: two-fact questions get both facts
hops=2

[judge]
; a decision model for the answer guard and fact merges (bench/README.md): off | openjev-flash-9b
; off unless llama.cpp runs on the GPU: on a CPU it adds ~11 s to each guarded answer
model=$Judge
guard_min=0.9
merge_min=0.9
"@

$IniPath = Join-Path $Root "run\$Preset.ini"
if ($DryRun) {
    Write-Host "[run] -DryRun: would write $IniPath :"
    $Ini -split "`n" | ForEach-Object { '    ' + $_.TrimEnd("`r") }
    exit 0
}

# ---------------------------------------------------------------- set up what is missing
if (-not (Test-Path $Bin)) {
    Write-Error ("memagent.exe not found next to this script or in build\. Download the Windows release " +
        "(https://github.com/FMXExpress/Short-Long-Memory-LLM/releases) and run run.cmd from its folder.")
    exit 1
}
if ($Trainer -eq 'python' -or $Engine -eq 'python') {
    if (-not $py) {
        Write-Error "Python is missing; the $Profile_ preset trains in Python. Install Python 3.10+ from python.org (tick 'Add to PATH')."
        exit 1
    }
    & $Python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>$null
    if ($LASTEXITCODE -ne 0) {
        # pip's default torch on Windows is CPU-only: take the CUDA build
        Write-Host '[run] installing PyTorch with CUDA (one time, about 3 GB)'
        & $Python -m pip install -q torch --index-url https://download.pytorch.org/whl/cu128
        if ($LASTEXITCODE -ne 0) { Write-Error 'installing PyTorch with CUDA failed'; exit 1 }
    }
    & $Python -c "import transformers, peft, accelerate" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host '[run] installing the Python trainer (transformers, peft, accelerate)'
        & $Python -m pip install -q -r (Join-Path $Root 'trainer\requirements.txt')
        if ($LASTEXITCODE -ne 0) { Write-Error 'installing the Python trainer failed'; exit 1 }
    }
    if ($Profile_ -ne 'small') {
        & $Python -c "import bitsandbytes" 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "[run] installing bitsandbytes ($Profile_ runs 4-bit)"
            & $Python -m pip install -q bitsandbytes
        }
    }
    & $Python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>$null
    if ($LASTEXITCODE -ne 0) { Write-Error 'PyTorch cannot see the GPU: update the NVIDIA driver, then rerun.'; exit 1 }
}

New-Item -ItemType Directory -Force -Path (Join-Path $Root 'run'), (Join-Path $Root "data\$Profile_") | Out-Null
# no BOM: memagent's ini reader would take it as part of the first section name
[IO.File]::WriteAllText($IniPath, ($Ini -replace "`r?`n", "`r`n"), (New-Object Text.UTF8Encoding($false)))

Write-Host "[run] starting: $Bin --config $IniPath $($Extra -join ' ')"
& $Bin --config $IniPath @Extra
exit $LASTEXITCODE
