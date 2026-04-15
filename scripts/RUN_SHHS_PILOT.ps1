$ErrorActionPreference = 'Stop'

$ROOT = 'C:\Projects\aiot\sleepfm-repro'
$REPO = 'C:\Projects\aiot\sleepfm-clinical'
$CONDA = 'C:\Users\Jae\anaconda3\Scripts\conda.exe'
$MODEL_LOCAL = "$ROOT\artifacts\embeddings\model_base_local"

Write-Host '== Step 1: preprocess SHHS pilot EDF -> HDF5 =='
cmd /c "$CONDA run -n sleepfm_env python $REPO\sleepfm\preprocessing\preprocessing.py --root_dir $ROOT\raw\shhs_pilot --target_dir $ROOT\artifacts\processed\shhs --num_threads 8 --num_files -1 --resample_rate 128"

Write-Host '== Step 2: prepare local model_base checkpoint config =='
if (Test-Path $MODEL_LOCAL) { Remove-Item $MODEL_LOCAL -Recurse -Force }
New-Item -ItemType Directory -Force -Path $MODEL_LOCAL | Out-Null
Copy-Item "$REPO\sleepfm\checkpoints\model_base\best.pt" "$MODEL_LOCAL\best.pt" -Force
Copy-Item "$REPO\sleepfm\checkpoints\model_base\config.json" "$MODEL_LOCAL\config.json" -Force

$config = Get-Content "$MODEL_LOCAL\config.json" | ConvertFrom-Json
$config.data_path = 'C:/Projects/aiot/sleepfm-repro/artifacts/processed/shhs'
$config.split_path = 'C:/Projects/aiot/sleepfm-repro/artifacts/splits/shhs_downstream_pilot_split.json'
$config.use_wandb = $false
$config.num_workers = 4
$config | ConvertTo-Json -Depth 10 | Set-Content "$MODEL_LOCAL\config.json"

Write-Host '== Step 3: generate embeddings =='
cmd /c "$CONDA run -n sleepfm_env python $REPO\sleepfm\pipeline\generate_embeddings.py --model_path $MODEL_LOCAL --dataset_name shhs --split_path $ROOT\artifacts\splits\shhs_downstream_pilot_split.json --num_workers 4 --batch_size 32"

Write-Host '== Step 4: finetune CoxPH downstream =='
cmd /c "$CONDA run -n sleepfm_env python $REPO\sleepfm\pipeline\finetune_diagnosis_coxph.py --config_path $ROOT\configs\config_finetune_diagnosis_coxph_shhs_pilot.yaml"
