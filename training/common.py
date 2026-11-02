"""Shared loaders for OWL experiments (features, tiles, labels, splits).

All paths derive from one root, KNOX_ROOT (default: the current directory); KNOX_MODEL_DIR optionally
overrides where trained MIL weights are read and written."""
import json, os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

KNOX_ROOT = os.environ.get("KNOX_ROOT", ".")
ROOT = os.path.join(KNOX_ROOT, "data")            # labels, tiles, features, tissue masks, experiment outputs
LOGS = os.path.join(KNOX_ROOT, "logs")            # result JSONs (logs/exp/)
MODEL_DIR = os.environ.get("KNOX_MODEL_DIR", os.path.join(KNOX_ROOT, "model"))   # owl/ (v1), owl_v2/ (v2)
JOBS = os.path.join(ROOT, "jobs_all.tsv")         # slide list: cohort, case, WSI source, mask source
DIM = {"hoptimus0": 1536, "midnight": 3072}


def labels(ds):
    """DataFrame indexed by case with columns event (bool), time (years)."""
    if ds == "owl":
        d = pd.read_csv(f"{ROOT}/labels/training_labels.csv")
        d["case"] = d.case_id.str.replace(".tif", "", regex=False)
        d = d.rename(columns={"follow_up_years": "time"})
    else:
        d = pd.read_csv(f"{ROOT}/labels/tcga_bcr.csv").rename(columns={"case_id": "case", "follow_up_y": "time"})
    d = d[["case", "event", "time"]].set_index("case").sort_index()
    d["event"] = d.event.astype(bool)
    return d


def _load(path, key):
    with np.load(path) as z:
        return z[key]


def feats(enc, ds, cases, key="x20", threads=32, tissue=False):
    """List of [n_i, D] float16 arrays in the order of `cases`; tissue=True keeps only E0 tissue tiles."""
    paths = [f"{ROOT}/feats/{enc}/{ds}/{c}.npz" for c in cases]
    with ThreadPoolExecutor(threads) as ex:
        X = list(ex.map(lambda p: _load(p, key), paths))
    return [x[np.load(f"{ROOT}/tissue/{ds}/{c}.npz")["keep"]] for x, c in zip(X, cases)] if tissue else X


def tiles(ds, case):
    z = np.load(f"{ROOT}/tiles/{ds}/{case}.npz")
    return {k: z[k] for k in z.files if k != "meta"} | {"meta": json.loads(str(z["meta"]))}


def folds(event, n_splits=5, repeats=1, seed=0):
    """Event-stratified patient folds: list over repeats of fold-id arrays."""
    from sklearn.model_selection import StratifiedKFold
    out = []
    for r in range(repeats):
        f = np.zeros(len(event), int)
        for k, (_, te) in enumerate(StratifiedKFold(n_splits, shuffle=True, random_state=seed + r).split(event, event)):
            f[te] = k
        out.append(f)
    return out


def cindex(event, time, risk):
    from sksurv.metrics import concordance_index_censored
    return concordance_index_censored(np.asarray(event, bool), np.asarray(time, float), np.asarray(risk, float))[0]
