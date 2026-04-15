import h5py, glob, os
paths = glob.glob(r'C:\Projects\aiot\sleepfm-repro\artifacts\embeddings\model_base_local\shhs\*.hdf5')
print('n', len(paths))
for p in paths[:2]:
    print('FILE', os.path.basename(p))
    with h5py.File(p,'r') as f:
        for k in f.keys():
            print(k, f[k].shape, f[k].dtype)
    print()
