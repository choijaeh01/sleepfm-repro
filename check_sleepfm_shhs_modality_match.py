import glob, json, os
import h5py
from collections import Counter

CHANNEL_GROUPS = r'C:\Projects\aiot\sleepfm-clinical\sleepfm\configs\channel_groups.json'
ROOT = r'C:\Projects\aiot\sleepfm-repro\raw\shhs_pilot\shhs1'

with open(CHANNEL_GROUPS, 'r', encoding='utf-8') as f:
    groups = json.load(f)

files = sorted(glob.glob(os.path.join(ROOT, '*.hdf5')))
print('n_files', len(files))
missing_counter = Counter()
examples = {}
all_ok = []

for p in files:
    with h5py.File(p, 'r') as hf:
        names = list(hf.keys())
    present = {}
    for modality in ['BAS', 'RESP', 'EKG', 'EMG']:
        matched = [n for n in names if n in groups[modality]]
        present[modality] = matched
    missing = [m for m, matched in present.items() if len(matched) == 0]
    if missing:
        for m in missing:
            missing_counter[m] += 1
            examples.setdefault(m, (os.path.basename(p), names))
    else:
        all_ok.append(os.path.basename(p))

print('all_modalities_ok', len(all_ok))
print('missing_counter', dict(missing_counter))
for m, ex in examples.items():
    fname, names = ex
    print(f'EXAMPLE_MISSING_{m}', fname)
    print('CHANNELS', names)
