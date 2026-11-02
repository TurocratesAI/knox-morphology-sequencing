"""E0: per-tile tissue filter. The tile grid used the organiser's low-resolution tissue mask (>=25% of the tile
inside the mask), which lets through tiles that are mostly glass, lumen, cracks or fat.

Ground truth: real pixels of the 20k stain-pilot tiles (200 per slide, 100 slides, both cohorts); tissue pixel
= saturation > 0.08 and max channel < 0.92. A ridge regression per encoder predicts the tissue fraction from the
tile embedding (slide-grouped CV: R2 0.95, AUC 0.99 for <25% tissue, both cohorts). The final predictors are fit
on all pilot tiles; a tile is kept if the mean of the two encoders' predictions is >= THR.
Writes data/tissue/<ds>/<case>.npz (pred float16, keep bool) and data/tissue/summary.json
"""
import glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
import common as C
from sklearn.linear_model import RidgeCV

THR = 0.25
ENCS = ("hoptimus0", "midnight")


def main():
    X = {e: [] for e in ENCS}; y = []
    for f in sorted(glob.glob(f"{C.ROOT}/dict/stain_pilot/*/*.npz")):
        ds, case = f.split("/")[-2], os.path.basename(f)[:-4]; z = np.load(f)
        t = z["tiles"].astype(np.float32) / 255; mx, mn = t.max(-1), t.min(-1)
        y.append((((mx - mn) / (mx + 1e-6) > 0.08) & (mx < 0.92)).mean((1, 2)))
        for e in ENCS:
            X[e].append(C.feats(e, ds, [case])[0][z["idx"]].astype(np.float32))
    y = np.concatenate(y)
    M = {e: RidgeCV(alphas=[1, 10, 100, 1000]).fit(np.concatenate(X[e]), y) for e in ENCS}
    os.makedirs(f"{C.ROOT}/tissue", exist_ok=True)
    np.savez(f"{C.ROOT}/tissue/predictors.npz", **{f"{e}_coef": M[e].coef_ for e in ENCS},
             **{f"{e}_intercept": M[e].intercept_ for e in ENCS}, thr=THR)
    summ = {}
    for ds in ("owl", "tcga"):
        os.makedirs(f"{C.ROOT}/tissue/{ds}", exist_ok=True); kept = []
        cases = list(C.labels(ds).index)
        F = {e: C.feats(e, ds, cases) for e in ENCS}
        for i, c in enumerate(cases):
            p = np.mean([M[e].predict(F[e][i].astype(np.float32)) for e in ENCS], 0)
            keep = p >= THR
            if keep.sum() < 16:                       # never leave a slide (almost) empty
                keep = p >= np.sort(p)[-min(16, len(p))]
            np.savez_compressed(f"{C.ROOT}/tissue/{ds}/{c}.npz", pred=p.astype(np.float16), keep=keep)
            kept.append(keep.mean())
        summ[ds] = dict(slides=len(cases), kept_mean=float(np.mean(kept)), kept_min=float(np.min(kept)),
                        kept_p05=float(np.percentile(kept, 5)))
        print(ds, json.dumps(summ[ds]), flush=True)
    json.dump(dict(thr=THR, n_pilot=int(len(y)), **summ), open(f"{C.ROOT}/tissue/summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
