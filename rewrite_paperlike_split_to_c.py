import json
from pathlib import Path
split_path = Path(r'C:\Projects\aiot\sleepfm-repro\artifacts\splits\shhs_downstream_paperlike_3291_496_2000.json')
text = split_path.read_text(encoding='utf-8')
text = text.replace('G:/sleepfm-repro/processed/shhs_paperlike', 'C:/Projects/aiot/sleepfm-repro/artifacts/processed/shhs_paperlike')
split_path.write_text(text, encoding='utf-8')
summary_path = Path(r'C:\Projects\aiot\sleepfm-repro\artifacts\labels\shhs_6cvd_paperlike_3291_496_2000\summary.json')
if summary_path.exists():
    data = json.loads(summary_path.read_text(encoding='utf-8'))
    data['processed_root'] = r'C:\Projects\aiot\sleepfm-repro\artifacts\processed\shhs_paperlike'
    summary_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
print('rewritten')
