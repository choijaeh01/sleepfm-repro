import json, os, sys
sys.path.append(r'C:\Projects\aiot\sleepfm-clinical\sleepfm')
from utils import load_config, load_data
from models.dataset import SetTransformerDataset

config = load_config(r'C:\Projects\aiot\sleepfm-repro\artifacts\embeddings\model_base_local\config.json')
channel_groups = load_data(r'C:\Projects\aiot\sleepfm-clinical\sleepfm\configs\channel_groups.json')
split = load_data(r'C:\Projects\aiot\sleepfm-repro\artifacts\splits\shhs_downstream_pilot_split.json')

hdf5_paths = []
for s in ['train','validation','test']:
    filtered_files = [fp for fp in split[s] if 'shhs' in fp.lower()]
    hdf5_paths += filtered_files
hdf5_paths = [os.path.join(config['data_path'], f) for f in hdf5_paths]
print('input_paths', len(hdf5_paths))
print('first3', hdf5_paths[:3])

dataset = SetTransformerDataset(config, channel_groups, hdf5_paths=hdf5_paths, split='test')
print('dataset_len', len(dataset))
print('unique_files', len(set(x[0] for x in dataset.index_map)))
print('first10_index', dataset.index_map[:10])
