import glob, json, os
from collections import Counter
import h5py

CHANNEL_GROUPS = r'C:\Projects\aiot\sleepfm-clinical\sleepfm\configs\channel_groups.json'
ROOT = r'C:\Projects\aiot\sleepfm-repro\raw\shhs_pilot\shhs1'
SAMPLES_PER_CHUNK = 5 * 60 * 128

with open(CHANNEL_GROUPS, 'r', encoding='utf-8') as f:
    groups = json.load(f)

files = sorted(glob.glob(os.path.join(ROOT, '*.hdf5')))
print('n_files', len(files))
qualifying = []
nonqual = []
chunks_total = 0
reasons = Counter()

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
            nonqual.append((os.path.basename(p), names, mods))
            continue
        dset_name = names[-1]
        num_samples = hf[dset_name].shape[0]
        num_chunks = num_samples // SAMPLES_PER_CHUNK
        if num_chunks == 0:
            reasons['zero_chunks'] += 1
            nonqual.append((os.path.basename(p), names, mods))
            continue
        qualifying.append((os.path.basename(p), num_samples, num_chunks, mods))
        chunks_total += num_chunks

print('qualifying_files', len(qualifying))
print('chunks_total', chunks_total)
print('reasons', dict(reasons))
print('first_10_qualifying')
for item in qualifying[:10]:
    print(item[0], 'samples=', item[1], 'chunks=', item[2], {k: len(v) for k,v in item[3].items()})
print('first_5_nonqual')
for item in nonqual[:5]:
    print(item[0], item[1], {k: len(v) for k,v in item[2].items()})
