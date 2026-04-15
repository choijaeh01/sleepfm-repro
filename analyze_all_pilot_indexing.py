import glob, json, os
from collections import Counter
import h5py

CHANNEL_GROUPS = r'C:\Projects\aiot\sleepfm-clinical\sleepfm\configs\channel_groups.json'
ROOT = r'C:\Projects\aiot\sleepfm-repro\raw\shhs_pilot'
SAMPLES_PER_CHUNK = 5 * 60 * 128

with open(CHANNEL_GROUPS, 'r', encoding='utf-8') as f:
    groups = json.load(f)

files = sorted(glob.glob(os.path.join(ROOT, '**', '*.hdf5'), recursive=True))
print('n_files', len(files))
qualifying = []
reasons = Counter()
by_folder = Counter()

for p in files:
    with h5py.File(p, 'r') as hf:
        names = [k for k in hf.keys() if isinstance(hf[k], h5py.Dataset)]
        mods = {
            'BAS': [n for n in names if n in groups['BAS']],
            'RESP': [n for n in names if n in groups['RESP']],
            'EKG': [n for n in names if n in groups['EKG']],
            'EMG': [n for n in names if n in groups['EMG']],
        }
        missing = [m for m,v in mods.items() if len(v) == 0]
        if missing:
            reasons['missing_' + '_'.join(missing)] += 1
            continue
        dset_name = names[-1]
        num_samples = hf[dset_name].shape[0]
        num_chunks = num_samples // SAMPLES_PER_CHUNK
        if num_chunks == 0:
            reasons['zero_chunks'] += 1
            continue
        qualifying.append((os.path.basename(p), num_chunks, os.path.relpath(os.path.dirname(p), ROOT)))
        by_folder[os.path.relpath(os.path.dirname(p), ROOT)] += 1

print('qualifying_files', len(qualifying))
print('by_folder', dict(by_folder))
print('reasons', dict(reasons))
print('sample', qualifying[:20])
print('total_chunks', sum(x[1] for x in qualifying))
