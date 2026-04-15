import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

import sys
sys.path.append(r'C:\Projects\aiot\sleepfm-clinical\sleepfm')
from models.models import SetTransformer  # noqa: E402

MODEL_DIR = Path(r'C:\Projects\aiot\sleepfm-repro\artifacts\embeddings\model_base_paperexact')
CONFIG_PATH = MODEL_DIR / 'config.json'
CHECKPOINT_PATH = MODEL_DIR / 'best.pt'
CHANNEL_GROUPS_PATH = Path(r'C:\Projects\aiot\sleepfm-clinical\sleepfm\configs\channel_groups.json')
SPLIT_PATH = Path(r'C:\Projects\aiot\sleepfm-repro\artifacts\splits\shhs_downstream_paperlike_3291_496_2000.json')
OUT_SEQ = MODEL_DIR / 'shhs'
OUT_AGG = MODEL_DIR / 'shhs_5min_agg'
SUMMARY_PATH = MODEL_DIR / 'shhs_paperexact_embedding_summary.json'

OUT_SEQ.mkdir(parents=True, exist_ok=True)
OUT_AGG.mkdir(parents=True, exist_ok=True)

with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
    config = json.load(f)
with open(CHANNEL_GROUPS_PATH, 'r', encoding='utf-8') as f:
    channel_groups = json.load(f)
with open(SPLIT_PATH, 'r', encoding='utf-8') as f:
    split = json.load(f)

file_paths = []
for key in ['train', 'validation', 'test']:
    file_paths.extend(split[key])
file_paths = sorted(dict.fromkeys(file_paths))

modalities = config['modality_types']
samples_per_chunk = config['sampling_duration'] * 60 * config['sampling_freq']

model = SetTransformer(
    config['in_channels'],
    config['patch_size'],
    config['embed_dim'],
    config['num_heads'],
    config['num_layers'],
    pooling_head=config['pooling_head'],
    dropout=0.0,
)
model = nn.DataParallel(model)
checkpoint = torch.load(CHECKPOINT_PATH)
model.load_state_dict(checkpoint['state_dict'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

summary = {
    'n_files_requested': len(file_paths),
    'processed': 0,
    'skipped_missing_modality': 0,
    'skipped_zero_chunk': 0,
    'errors': [],
}

for file_path in tqdm(file_paths, desc='Embedding SHHS paperexact'):
    subject_id = Path(file_path).stem
    seq_out = OUT_SEQ / f'{subject_id}.hdf5'
    agg_out = OUT_AGG / f'{subject_id}.hdf5'
    if seq_out.exists() and agg_out.exists():
        summary['processed'] += 1
        continue
    try:
        with h5py.File(file_path, 'r') as hf:
            names = [k for k in hf.keys() if isinstance(hf[k], h5py.Dataset)]
            modality_to_channels = {m: [n for n in names if n in channel_groups[m]] for m in modalities}
            if any(len(v) == 0 for v in modality_to_channels.values()):
                summary['skipped_missing_modality'] += 1
                continue
            min_len = min(hf[ch].shape[0] for m in modalities for ch in modality_to_channels[m])
            num_chunks = min_len // samples_per_chunk
            if num_chunks == 0:
                summary['skipped_zero_chunk'] += 1
                continue

            if seq_out.exists():
                seq_out.unlink()
            if agg_out.exists():
                agg_out.unlink()

            with h5py.File(seq_out, 'w') as f_seq, h5py.File(agg_out, 'w') as f_agg:
                for chunk_idx in range(num_chunks):
                    start = chunk_idx * samples_per_chunk
                    end = start + samples_per_chunk
                    for modality in modalities:
                        channels = modality_to_channels[modality]
                        arr = np.stack([hf[ch][start:end] for ch in channels], axis=0).astype(np.float32)
                        x = torch.from_numpy(arr).unsqueeze(0).to(device)
                        mask = torch.zeros((1, arr.shape[0]), dtype=torch.bool, device=device)
                        with torch.no_grad():
                            pooled, seq = model(x, mask)
                        pooled = pooled[0].detach().cpu().numpy().astype(np.float32)
                        seq = seq[0].detach().cpu().numpy().astype(np.float32)

                        if modality not in f_seq:
                            f_seq.create_dataset(modality, data=seq, maxshape=(None, seq.shape[1]), chunks=(min(256, seq.shape[0]), seq.shape[1]))
                        else:
                            d = f_seq[modality]
                            old = d.shape[0]
                            d.resize((old + seq.shape[0], seq.shape[1]))
                            d[old:old + seq.shape[0]] = seq

                        pooled2 = pooled.reshape(1, -1)
                        if modality not in f_agg:
                            f_agg.create_dataset(modality, data=pooled2, maxshape=(None, pooled2.shape[1]), chunks=(1, pooled2.shape[1]))
                        else:
                            d = f_agg[modality]
                            old = d.shape[0]
                            d.resize((old + 1, pooled2.shape[1]))
                            d[old:old + 1] = pooled2
        summary['processed'] += 1
    except Exception as e:
        summary['errors'].append({'file': file_path, 'error': str(e)})
        with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
