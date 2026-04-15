param(
    [string]$RepoRoot = "C:\Projects\aiot\sleepfm-repro",
    [string]$EmbeddingDir = "C:\Projects\aiot\sleepfm-repro\artifacts\embeddings\model_base_paperexact\shhs_5min_agg",
    [string]$AuditDir = "C:\Projects\aiot\sleepfm-repro\artifacts\audit\loo_geometry_model_base_paperexact_shhs",
    [string]$EvalDir = "C:\Projects\aiot\sleepfm-repro\artifacts\embeddings\model_base_paperexact\DiagnosisFinetuneFullLSTMCOXPHWithDemo_shhs_shhs_6cvd_paperexact_3291_496_2000_demo_labels_BAS_RESP_EKG_EMG__ep_5_bs_8\shhs_downstream_paperlike_3291_496_2000\test",
    [string]$LabelsDir = "C:\Projects\aiot\sleepfm-repro\artifacts\labels\shhs_6cvd_paperexact_3291_496_2000",
    [string]$FocusTarget = "chf"
)

$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $RepoRoot

python scripts\analyze_loo_geometry.py `
  --embeddings-dir $EmbeddingDir `
  --output-dir $AuditDir

python scripts\make_loo_diagnosis_figures.py `
  --geometry-dir $AuditDir `
  --embedding-dir $EmbeddingDir `
  --eval-dir $EvalDir `
  --labels-dir $LabelsDir `
  --output-dir (Join-Path $AuditDir "diagnosis_6y") `
  --focus-target $FocusTarget

python scripts\export_loo_cl_github_bundle.py `
  --audit-dir $AuditDir `
  --output-dir (Join-Path $RepoRoot "reports\loo_cl_audit")
