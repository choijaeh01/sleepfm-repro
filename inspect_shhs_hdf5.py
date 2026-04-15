import h5py, glob, os
paths = glob.glob(r'C:\Projects\aiot\sleepfm-repro\raw\shhs_pilot\shhs1\*.hdf5')
print('n_hdf5', len(paths))
for p in paths[:5]:
    with h5py.File(p, 'r') as f:
        print('FILE', os.path.basename(p))
        print(list(f.keys())[:80])
        print()
