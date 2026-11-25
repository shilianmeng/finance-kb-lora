# 本机 transformers 部署的一次性准备：下载 torch(CUDA) + 依赖 + Qwen2.5-1.5B-Instruct
# 支持断点续传，中断后重新运行即可接着下。
#   powershell -ExecutionPolicy Bypass -File scripts\setup_local_model.ps1
$ErrorActionPreference = 'Continue'
$root  = Split-Path -Parent $PSScriptRoot
$dl    = Join-Path (Split-Path -Parent $root) '_downloads'
$model = Join-Path $root 'models\Qwen2.5-1.5B-Instruct'
$wheel = Join-Path $dl 'torch-2.6.0+cu124-cp312-cp312-win_amd64.whl'
$wheelUrl = 'https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp312-cp312-win_amd64.whl'
$pypi = 'https://pypi.tuna.tsinghua.edu.cn/simple'
New-Item -ItemType Directory -Force $dl, $model | Out-Null
Start-Transcript -Path (Join-Path $dl 'setup.log') -Append | Out-Null

function Get-File($url, $out) {
    for ($i = 1; $i -le 20; $i++) {
        curl.exe -L -C - --retry 5 --retry-delay 3 --speed-limit 1000 --speed-time 60 -# -o $out $url
        if ($LASTEXITCODE -eq 0 -or $LASTEXITCODE -eq 33) { return $true }
        Write-Host "中断（$LASTEXITCODE），第 $i 次重试..."
        Start-Sleep 3
    }
    return $false
}

# 1. 模型（ModelScope 国内源）
Write-Host "== 下载模型到 $model"
foreach ($f in 'config.json','generation_config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','LICENSE','model.safetensors') {
    Write-Host "-- $f"
    Get-File "https://www.modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct/resolve/master/$f" (Join-Path $model $f) | Out-Null
}

# 2. torch CUDA 版
python -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "== 下载 torch 2.6.0+cu124（约 2.5GB）"
    Get-File $wheelUrl $wheel | Out-Null
    pip install --no-cache-dir $wheel -i $pypi
}

# 3. 其他依赖
pip install --no-cache-dir "transformers>=4.44,<5" "peft>=0.12" "accelerate>=0.33" "modelscope>=1.17" sentencepiece -i $pypi

# 4. 自检
python -c "import torch,transformers;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'transformers',transformers.__version__)"
Get-ChildItem $model | Select-Object Name, @{n='MB';e={[math]::Round($_.Length/1MB,1)}} | Format-Table -AutoSize
Write-Host "完成后运行：python scripts\serve_hf.py   （或 VS Code 里选「电商风控 + 本地模型」）"
