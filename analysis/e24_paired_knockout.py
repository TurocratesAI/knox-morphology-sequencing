"""E24: paired-end reads and compartment knockouts (all out of fold).

1. Compartment knockouts. For each slide and compartment S (tumour proxy / gland / stroma / unassigned; H-opt K16
   dictionary), with the v1 fold model that did not train on the slide:
     knockout  y_{-S} = (y - sum_S a_i v_i) / (1 - A_S)      (exact: attention renormalises over the remaining tiles)
     read-only y_S    = sum_S a_i v_i / A_S                   (the score if the slide held only that compartment)
   Reported: mean drop y - y_{-S}, and Harrell's C of y_{-S} and y_S within each cohort.
2. Paired-end reads. Each tile is paired with its 448 um surroundings: compartment mix
   of tissue tiles within 224 um of its centre, number of such tiles, and distance to the nearest tumour-proxy, gland
   and stroma tile. Tile values v_i are mapped from nuclear reads + compartment ('both', as E23) and from those plus
   context ('paired'); slide-grouped OOF per cohort; mapped slide score keeps the model's attention.
Caches per-slide arrays in data/exp/e24/<ds>/<case>.npz (v, a, g, F, ctx, xy_um) for later analyses.
Writes logs/exp/e24_paired_knockout.json
"""
import json, os, sys
import numpy as np, pandas as pd, torch
from scipy.spatial import cKDTree
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "training"), os.path.join(_HERE, "..", "container")]
import common as C
from e23_mapping_slide import MD, DICT, FEATS, GRP, compartment, boot_c

CACHE = f"{C.ROOT}/exp/e24"
CTX = ["ctx_tumour", "ctx_gland", "ctx_stroma", "ctx_unassigned", "ctx_n", "d_tumour", "d_gland", "d_stroma"]


def context(xy, g, r=224.0):
    tr = cKDTree(xy); nb = tr.query_ball_point(xy, r)
    out = np.zeros((len(g), 8), np.float32)
    for i, n in enumerate(nb):
        h = np.bincount(g[n], minlength=4); out[i, :4] = h / max(1, h.sum()); out[i, 4] = len(n)
    for q, col in ((0, 5), (1, 6), (2, 7)):
        m = g == q
        out[:, col] = cKDTree(xy[m]).query(xy)[0] if m.any() else 5000.0
    out[:, 5:] = np.minimum(out[:, 5:], 5000.0)
    return out


def slide_arrays(dev, nets, fold, cases):
    from inference import ABMIL  # noqa: F401 (nets already built)
    S = []
    with torch.no_grad():
        for j, (ds, c) in enumerate(cases):
            p = f"{CACHE}/{ds}/{c}.npz"
            if os.path.exists(p):
                z = np.load(p); S.append({k: z[k] for k in z.files}); continue
            cf = f"{C.ROOT}/cellfeat/{ds}/{c}.npz"
            if not os.path.exists(cf):
                S.append(None); continue
            keep = np.load(f"{C.ROOT}/tissue/{ds}/{c}.npz")["keep"]; idx = np.where(keep)[0]
            net, mu, sd = nets[fold[j]]
            X = torch.tensor(C.feats("midnight", ds, [c])[0][keep].astype(np.float32), device=dev)
            a, s = net.parts(X); v = ((s - mu) / sd).cpu().numpy(); a = a.cpu().numpy()
            z = np.load(cf); names = [str(x) for x in z["names"]]; F = z["X"][idx][:, [names.index(f) for f in FEATS]].astype(np.float32)
            g = compartment(np.load(f"{DICT}/{ds}/{c}.npz")["label"][idx]).astype(np.int8)
            T = C.tiles(ds, c); s20, mpp0 = T["meta"]["tile20_l0"], T["meta"]["mpp0"]
            xy = ((T["coords20"][idx] + s20 / 2) * mpp0).astype(np.float32)
            d = dict(v=v, a=a, g=g, F=F, ctx=context(xy, g.astype(int)), xy_um=xy, idx=idx.astype(np.int32))
            os.makedirs(os.path.dirname(p), exist_ok=True); np.savez(p, **d); S.append(d)
            if j % 100 == 0:
                print(j, c, len(v), flush=True)
    return S


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    from sklearn.model_selection import StratifiedKFold
    from sklearn.ensemble import HistGradientBoostingRegressor
    from inference import ABMIL
    cal = json.load(open(f"{MD}/calib.json")); nets = []
    for m in cal["models"]:
        n = ABMIL(cal["dim"]); n.load_state_dict(torch.load(f"{MD}/{m['file']}", map_location="cpu")); nets.append((n.eval().to(dev), m["mu"], m["sd"]))
    L = {ds: C.labels(ds) for ds in ("owl", "tcga")}
    cases = [(ds, c) for ds in L for c in L[ds].index]
    coh = np.array([ds for ds, _ in cases]); ev = np.concatenate([L[ds].event.values for ds in L])
    tm = np.concatenate([L[ds].time.values for ds in L])
    strat = np.array([f"{c}{int(v)}" for c, v in zip(coh, ev)]); fold = np.zeros(len(cases), int)
    for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=2026).split(strat, strat)):
        fold[te] = k
    S = slide_arrays(dev, nets, fold, cases); ok = np.array([s is not None for s in S])
    res = dict(knockout={}, paired={})
    # ---- 1. compartment knockouts and read-only scores
    rows = []
    for j in np.where(ok)[0]:
        v, a, g = S[j]["v"], S[j]["a"], S[j]["g"]; y = float((a * v).sum()); r = dict(j=j, y=y)
        for q in range(4):
            m = g == q; A = float(a[m].sum()); sv = float((a[m] * v[m]).sum())
            r[f"ko_{GRP[q]}"] = (y - sv) / (1 - A) if A < 1 - 1e-6 else np.nan
            r[f"only_{GRP[q]}"] = sv / A if A > 1e-9 else np.nan
            r[f"att_{GRP[q]}"] = A
        rows.append(r)
    K = pd.DataFrame(rows).set_index("j")
    for ds in ("owl", "tcga"):
        d = K[coh[K.index] == ds]; e, t = ev[d.index], tm[d.index]; out = dict(y=dict(c=round(float(C.cindex(e > 0, t, d.y.values)), 3)))
        for q in GRP:
            for kind in ("ko", "only"):
                col = f"{kind}_{q}"; m = d[col].notna().values
                out[col] = dict(n=int(m.sum()), c=round(float(C.cindex(e[m] > 0, t[m], d[col].values[m])), 3),
                                ci=boot_c(e[m], t[m], d[col].values[m]),
                                mean_drop=round(float((d.y - d[col])[m].mean()), 3) if kind == "ko" else None)
            out[f"att_{q}"] = round(float(d[f"att_{q}"].mean()), 3)
        res["knockout"][ds] = out
    # ---- 2. paired-end mapping
    for ds in ("owl", "tcga"):
        J = np.where(ok & (coh == ds))[0]; rng = np.random.default_rng(0)
        sub = {j: rng.choice(len(S[j]["v"]), min(400, len(S[j]["v"])), replace=False) for j in J}
        oh = lambda g: np.eye(4, dtype=np.float32)[g.astype(int)]
        des = {"both": lambda s, i=slice(None): np.c_[s["F"][i], oh(s["g"][i])],
               "paired": lambda s, i=slice(None): np.c_[s["F"][i], oh(s["g"][i]), s["ctx"][i]],
               "context_only": lambda s, i=slice(None): np.c_[oh(s["g"][i]), s["ctx"][i]]}
        pred = {nm: {} for nm in des}
        for k in range(5):
            tr = [j for j in J if fold[j] != k]; te = [j for j in J if fold[j] == k]
            vtr = np.concatenate([S[j]["v"][sub[j]] for j in tr])
            for nm, f in des.items():
                m = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(np.concatenate([f(S[j], sub[j]) for j in tr]), vtr)
                for j in te:
                    pred[nm][j] = m.predict(f(S[j]))
        allv = np.concatenate([S[j]["v"] for j in J]); allg = np.concatenate([S[j]["g"] for j in J]); out = {}
        e, t = ev[J], tm[J]; y = np.array([(S[j]["a"] * S[j]["v"]).sum() for j in J])
        for nm in des:
            pv = np.concatenate([pred[nm][j] for j in J]); ym = np.array([(S[j]["a"] * pred[nm][j]).sum() for j in J]); u = y - ym
            out[nm] = dict(tile_r2=round(float(1 - ((allv - pv) ** 2).sum() / ((allv - allv.mean()) ** 2).sum()), 3),
                           within_corr2={GRP[q]: round(float(np.corrcoef(allv[allg == q], pv[allg == q])[0, 1] ** 2), 3) for q in range(4)},
                           slide_r=round(float(np.corrcoef(ym, y)[0, 1]), 3), c_mapped=round(float(C.cindex(e > 0, t, ym)), 3),
                           c_unmapped=round(float(C.cindex(e > 0, t, u)), 3), c_unmapped_ci=boot_c(e, t, u))
        res["paired"][ds] = out
    json.dump(res, open(f"{C.LOGS}/exp/e24_paired_knockout.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
