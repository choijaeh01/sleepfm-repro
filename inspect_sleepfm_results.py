from pathlib import Path
base = Path(r"C:\Projects\aiot\sleepfm-repro\artifacts\embeddings\model_base_local\DiagnosisFinetuneFullLSTMCOXPHWithDemo_shhs_shhs_6cvd_localpilot_demo_labels_BAS_RESP_EKG_EMG__ep_5_bs_8")
print('BASE_EXISTS', base.exists())
if base.exists():
    for p in sorted(base.rglob('*')):
        print(('DIR ' if p.is_dir() else 'FILE'), p)

log = Path(r"C:\Projects\aiot\sleepfm-repro\run_shhs_downstream_localpilot.log")
print('LOG_EXISTS', log.exists())
if log.exists():
    text = log.read_text(encoding='utf-8', errors='ignore')
    print('--- FILTERED LOG ---')
    keys = ['Validation Loss after Epoch', 'Best model saved', 'test', 'c-index', 'concordance', 'auc', 'Epoch [5/5] Loss']
    for line in text.splitlines():
        low = line.lower()
        if any(k.lower() in low for k in keys):
            print(line)

src = Path(r"C:\Projects\aiot\sleepfm-clinical\sleepfm\pipeline\finetune_diagnosis_coxph.py")
print('SRC_EXISTS', src.exists())
if src.exists():
    print('--- SOURCE 250-340 ---')
    lines = src.read_text(encoding='utf-8', errors='ignore').splitlines()
    for i, line in enumerate(lines, 1):
        if 250 <= i <= 340:
            print(f'{i}: {line}')
