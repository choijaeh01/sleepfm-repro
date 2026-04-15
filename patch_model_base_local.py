import json
p = r'C:\Projects\aiot\sleepfm-repro\artifacts\embeddings\model_base_local\config.json'
with open(p, 'r', encoding='utf-8') as f:
    cfg = json.load(f)
cfg['data_path'] = r'C:/Projects/aiot/sleepfm-repro/raw/shhs_pilot/shhs1'
cfg['split_path'] = r'C:/Projects/aiot/sleepfm-repro/artifacts/splits/shhs_downstream_pilot_split.json'
cfg['use_wandb'] = False
cfg['num_workers'] = 4
with open(p, 'w', encoding='utf-8') as f:
    json.dump(cfg, f, indent=2)
print('PATCHED')
