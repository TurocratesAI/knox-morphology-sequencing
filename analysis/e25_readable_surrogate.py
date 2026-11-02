"""E25: a readable surrogate of the model (out of fold), and a fixed blend with the model.

From E24's cache (data/exp/e24): per slide, OOF fold-model tile values v_i and attention a_i, nuclear reads F,
compartment g and 448 um context. The surrogate keeps the model's attention and replaces each v_i with its prediction
from readable inputs (nuclear reads + compartment + context; 'paired' in E24), fit on other folds' slides per cohort:
  y_read = sum a_i vhat_i
Compared with the model score y by paired patient bootstrap of the difference in Harrell's C within each cohort.
Blend: z(y) + z(y_read), z-scored within cohort, weights fixed at 1:1 (no tuning on outcome).
Writes logs/exp/e25_surrogate.json and data/exp/e25_slides.csv
"""
import json, os, sys
import numpy as np, pandas as pd
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "training"), os.path.join(_HERE, "..", "container")]
import common as C
from e24_paired_knockout import CACHE

DESIGNS = ("nuc", "both", "paired")


def X_of(s, nm, i=slice(None)):
    oh = np.eye(4, dtype=np.float32)[s["g"][i].astype(int)]
    return {"nuc": s["F"][i], "both": np.c_[s["F"][i], oh], "paired": np.c_[s["F"][i], oh, s["ctx"][i]]}[nm]


def boot_diff(e, t, a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(e), len(e)); d.append(C.cindex(e[i] > 0, t[i], a[i]) - C.cindex(e[i] > 0, t[i], b[i]))
    d = np.array(d); return [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)], round(float((d <= 0).mean()), 3)


def main():
    from sklearn.model_selection import StratifiedKFold
    from sklearn.ensemble import HistGradientBoostingRegressor
    L = {ds: C.labels(ds) for ds in ("owl", "tcga")}
    cases = [(ds, c) for ds in L for c in L[ds].index]
    coh = np.array([ds for ds, _ in cases]); ev = np.concatenate([L[ds].event.values for ds in L]); tm = np.concatenate([L[ds].time.values for ds in L])
    strat = np.array([f"{c}{int(v)}" for c, v in zip(coh, ev)]); fold = np.zeros(len(cases), int)
    for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=2026).split(strat, strat)):
        fold[te] = k
    S = [dict(np.load(f"{CACHE}/{ds}/{c}.npz")) if os.path.exists(f"{CACHE}/{ds}/{c}.npz") else None for ds, c in cases]
    rows, res = [], {}
    for ds in ("owl", "tcga"):
        J = [j for j in range(len(cases)) if S[j] is not None and coh[j] == ds]; rng = np.random.default_rng(0)
        sub = {j: rng.choice(len(S[j]["v"]), min(400, len(S[j]["v"])), replace=False) for j in J}
        ys = {nm: {} for nm in DESIGNS}
        for k in range(5):
            tr = [j for j in J if fold[j] != k]; te = [j for j in J if fold[j] == k]; vtr = np.concatenate([S[j]["v"][sub[j]] for j in tr])
            for nm in DESIGNS:
                m = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(np.concatenate([X_of(S[j], nm, sub[j]) for j in tr]), vtr)
                for j in te:
                    ys[nm][j] = float((S[j]["a"] * m.predict(X_of(S[j], nm))).sum())
        d = pd.DataFrame(dict(j=J, case=[cases[j][1] for j in J], cohort=ds, event=ev[J], time=tm[J],
                              y=[float((S[j]["a"] * S[j]["v"]).sum()) for j in J], **{f"y_{nm}": [ys[nm][j] for j in J] for nm in DESIGNS}))
        z = lambda s: (s - s.mean()) / s.std()
        d["blend"] = z(d.y) + z(d.y_paired)
        e, t = d.event.values, d.time.values; out = {"n": len(d), "c_model": round(float(C.cindex(e > 0, t, d.y.values)), 3)}
        for col in [f"y_{nm}" for nm in DESIGNS] + ["blend"]:
            ci, p = boot_diff(e, t, d[col].values, d.y.values)
            out[col] = dict(c=round(float(C.cindex(e > 0, t, d[col].values)), 3), diff_vs_model_ci=ci, p_not_better=p,
                            r_with_model=round(float(np.corrcoef(d[col], d.y)[0, 1]), 3))
        res[ds] = out; rows.append(d)
    pd.concat(rows).to_csv(f"{C.ROOT}/exp/e25_slides.csv", index=False)
    json.dump(res, open(f"{C.LOGS}/exp/e25_surrogate.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
