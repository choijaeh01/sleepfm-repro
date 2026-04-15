import json, math, pickle
from pathlib import Path
import numpy as np

TARGETS = ['mi', 'stroke', 'chf', 'cvd_death', 'chd_death', 'revasc_proc']
base = Path(r'C:\Projects\aiot\sleepfm-repro\artifacts\embeddings\model_base_local\DiagnosisFinetuneFullLSTMCOXPHWithDemo_shhs_shhs_6cvd_localpilot_demo_labels_BAS_RESP_EKG_EMG__ep_5_bs_8\shhs_downstream_localpilot_split\test')

with open(base / 'all_outputs.pickle', 'rb') as f:
    outputs = np.array(pickle.load(f), dtype=float)
with open(base / 'all_event_times.pickle', 'rb') as f:
    event_times = np.array(pickle.load(f), dtype=float)
with open(base / 'all_is_event.pickle', 'rb') as f:
    is_event = np.array(pickle.load(f), dtype=float)
with open(base / 'all_paths.pickle', 'rb') as f:
    paths = np.array(pickle.load(f))


def harrell_c_index(times, events, risks):
    n = len(times)
    comparable = 0
    concordant = 0.0
    tied_risk = 0.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if events[i] == 1 and times[i] < times[j]:
                comparable += 1
                if risks[i] > risks[j]:
                    concordant += 1
                elif risks[i] == risks[j]:
                    tied_risk += 1
    if comparable == 0:
        return None, 0
    return (concordant + 0.5 * tied_risk) / comparable, comparable


def simple_auc(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    better = 0.0
    total = 0
    for i in pos:
        for j in neg:
            total += 1
            if scores[i] > scores[j]:
                better += 1
            elif scores[i] == scores[j]:
                better += 0.5
    return better / total if total else None

# Try multiple plausible horizons because label units are not yet guaranteed from this script.
HORIZONS = [6.0, 6 * 365.25]

report = {
    'n_test': int(len(paths)),
    'paths_head': [str(x) for x in paths[:5]],
    'outputs_shape': list(outputs.shape),
    'event_times_shape': list(event_times.shape),
    'is_event_shape': list(is_event.shape),
    'targets': {}
}

for idx, target in enumerate(TARGETS):
    t = event_times[:, idx]
    e = is_event[:, idx].astype(int)
    r = outputs[:, idx]
    cidx, comparable = harrell_c_index(t, e, r)
    target_report = {
        'n_positive': int(e.sum()),
        'n_negative_or_censored': int((e == 0).sum()),
        'event_time_min': float(np.min(t)) if len(t) else None,
        'event_time_median': float(np.median(t)) if len(t) else None,
        'event_time_max': float(np.max(t)) if len(t) else None,
        'risk_min': float(np.min(r)) if len(r) else None,
        'risk_median': float(np.median(r)) if len(r) else None,
        'risk_max': float(np.max(r)) if len(r) else None,
        'harrell_c_index': None if cidx is None else float(cidx),
        'comparable_pairs': int(comparable),
        'auroc_by_horizon': {}
    }
    for h in HORIZONS:
        y = ((e == 1) & (t <= h)).astype(int)
        auc = simple_auc(y, r)
        target_report['auroc_by_horizon'][str(h)] = {
            'n_positive_within_horizon': int(y.sum()),
            'auc': None if auc is None else float(auc)
        }
    report['targets'][target] = target_report

print(json.dumps(report, indent=2))
