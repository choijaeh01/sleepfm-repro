import os, glob
from collections import Counter
root = r'C:\Projects\aiot\sleepfm-repro\raw\shhs_pilot'
for ext in ('*.edf','*.hdf5'):
    files = glob.glob(os.path.join(root, '**', ext), recursive=True)
    print(ext, 'count', len(files))
    ctr = Counter(os.path.relpath(os.path.dirname(p), root) for p in files)
    print(dict(ctr))
    print(files[:10])
