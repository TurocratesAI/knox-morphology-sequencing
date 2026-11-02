"""E12: does the number of tiles per slide matter? (decides whether full-coverage re-extraction is worth it)

Midnight ABMIL, E0 tissue tiles. For each budget N, every slide's bag is a uniform-stride subsample of its
(already 16k-capped) tiles to <= N, used for training (2048 random tiles/step as E1) and for prediction.
RUMC 5-fold x 2 repeats OOF; TCGA from repeat-0 fold models. A curve still rising at 16k -> re-extract all tiles.
Writes logs/exp/e12_tile_budget.json
"""
import json, os, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "training")]
import common as C
from e1_baselines import train_mil, predict_mil

BUDGETS = (500, 1000, 2000, 4000, 8000, 16000)


def stride(x, n):
    return x if len(x) <= n else x[np.arange(0, len(x), int(np.ceil(len(x) / n)))[:n]]


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    L = {ds: C.labels(ds) for ds in ("owl", "tcga")}
    X = {ds: C.feats("midnight", ds, list(L[ds].index), tissue=True) for ds in L}
    e, t = L["owl"].event.values, L["owl"].time.values; et, tt = L["tcga"].event.values, L["tcga"].time.values
    res = {}
    for n in BUDGETS:
        Xo = [stride(x, n) for x in X["owl"]]; Xt = [stride(x, n) for x in X["tcga"]]
        oc, tp = [], []
        for r, fold in enumerate(C.folds(e, 5, 2, seed=0)):
            oof = np.zeros(len(e))
            for k in range(5):
                tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
                m = train_mil("abmil", [Xo[i] for i in tr], t[tr], e[tr], Xo[0].shape[1], dev, seed=r * 10 + k)
                oof[te] = predict_mil(m, [Xo[i] for i in te], dev)
                if r == 0:
                    p = predict_mil(m, Xt, dev); tp.append((p - p.mean()) / p.std())
            oc.append(C.cindex(e, t, oof))
        res[n] = dict(oof=float(np.mean(oc)), oof_sd=float(np.std(oc)), tcga=C.cindex(et, tt, np.mean(tp, 0)),
                      mean_tiles=float(np.mean([len(x) for x in Xo])))
        print(n, res[n], flush=True)
        json.dump(res, open(f"{C.LOGS}/exp/e12_tile_budget.json", "w"), indent=1)


if __name__ == "__main__":
    main()
