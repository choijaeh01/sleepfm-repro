from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

SEED = 20
N_TRAIN = 3291
N_TEST = 2000

base = Path(r'C:\Projects\aiot\sleepfm-repro\artifacts\labels\shhs_6cvd')
out_base = Path(r'C:\Projects\aiot\sleepfm-repro\artifacts\labels\shhs_6cvd_paperlike_3291_496_2000')
out_base.mkdir(parents=True, exist_ok=True)
split_path = Path(r'C:\Projects\aiot\sleepfm-repro\artifacts\splits\shhs_downstream_paperlike_3291_496_2000.json')
split_path.parent.mkdir(parents=True, exist_ok=True)
processed_root = Path(r'G:\sleepfm-repro\processed\shhs_paperlike')

def read_csv(path: Path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

is_event_rows = read_csv(base / 'is_event.csv')
time_rows = read_csv(base / 'time_to_event.csv')
demo_rows = read_csv(base / 'demo_labels.csv')

study_ids = [row['Study ID'] for row in is_event_rows]
assert len(study_ids) == len(set(study_ids)), 'duplicate study ids in is_event.csv'
assert len(study_ids) == len(time_rows) == len(demo_rows)

n = len(study_ids)
assert N_TRAIN + N_TEST <= n
n_val = n - N_TRAIN - N_TEST

rng = random.Random(SEED)
ids = study_ids[:]
rng.shuffle(ids)
train_ids = ids[:N_TRAIN]
val_ids = ids[N_TRAIN:N_TRAIN+n_val]
test_ids = ids[N_TRAIN+n_val:]
assert len(test_ids) == N_TEST

splits = {
    'train': train_ids,
    'validation': val_ids,
    'test': test_ids,
}

# write split json with processed hdf5 paths
split_json = {
    name: [str(processed_root / f'{sid}.hdf5').replace('\\', '/') for sid in sid_list]
    for name, sid_list in splits.items()
}
split_path.write_text(json.dumps(split_json, indent=2), encoding='utf-8')

# write label csvs filtered/reordered to this universe
row_by_id_event = {row['Study ID']: row for row in is_event_rows}
row_by_id_time = {row['Study ID']: row for row in time_rows}
row_by_id_demo = {row['Study ID']: row for row in demo_rows}
ordered_ids = train_ids + val_ids + test_ids

for name, row_map, fields in [
    ('is_event.csv', row_by_id_event, list(is_event_rows[0].keys())),
    ('time_to_event.csv', row_by_id_time, list(time_rows[0].keys())),
]:
    with (out_base / name).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sid in ordered_ids:
            w.writerow(row_map[sid])

# model currently expects two demo columns: age_z, male
with (out_base / 'demo_labels.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['Study ID', 'age_z', 'male'])
    for sid in ordered_ids:
        row = row_by_id_demo[sid]
        w.writerow([sid, row['age_z'], row['male']])

# summarize overall + split-specific event counts
summary = {
    'seed': SEED,
    'n_subjects': n,
    'n_train': len(train_ids),
    'n_validation': len(val_ids),
    'n_test': len(test_ids),
    'processed_root': str(processed_root),
    'split_path': str(split_path),
    'targets_overall': {},
    'targets_by_split': defaultdict(dict),
}

targets = [k for k in is_event_rows[0].keys() if k != 'Study ID']
for target in targets:
    summary['targets_overall'][target] = int(sum(int(row[target]) for row in is_event_rows))
    for split_name, sid_list in splits.items():
        summary['targets_by_split'][split_name][target] = int(sum(int(row_by_id_event[sid][target]) for sid in sid_list))

(out_base / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
