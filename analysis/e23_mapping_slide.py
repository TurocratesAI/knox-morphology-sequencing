"""E23: does the nuclear mapping hold where the model decides, and is the unmapped part prognostic? (all out of fold)

Motivation: E17 tile R2 turned out to match what tissue compartment alone gives.

Per slide (806), the v1 fold model that did not train on it (folds as train_final / E21) gives, for every E0 tissue
tile, the tile value v_i and attention a_i; slide score y = sum a_i v_i. Tile values are then predicted from
  nuc  : E9 nuclear reads (15 features, as E17)
  cmp  : compartment only (H-opt K16 dictionary: tumour proxy / gland / stroma / unassigned)
  both : compartment + nuclear reads
with gradient boosting fit on <=400 sampled tiles per slide of the OTHER folds' slides, per cohort. Mapped slide scores
keep the model's own attention: y_x = sum a_i vhat_i. Reported per cohort: tile R2 (all, and within compartment),
slide-level r(y_x, y), Harrell's C of y, y_x and of the unmapped part y - y_x, and a two-term Cox (y_nuc, y - y_nuc).
Also the compartment of the top 1% tiles by exact deletion effect D_i = a_i (v_i - y)/(1 - a_i).
Writes logs/exp/e23_mapping_slide.json
"""
import json, os, sys
import numpy as np, pandas as pd, torch
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "training"), os.path.join(_HERE, "..", "container")]
import common as C
from inference import ABMIL

MD = f"{C.MODEL_DIR}/owl"
DICT = f"{C.ROOT}/dict/hoptimus0_t/K16/assign"
BEN, STR = {1, 8, 10, 11, 13}, {2, 3, 5, 6, 7, 9, 12, 14, 15}
FEATS = ["n_dens", "area_mu", "area_cv", "elong_mu", "hema_mu", "hema_sd_mu", "eos_ring_mu", "small_dark", "nn_mu", "nn_cv",
         "clark", "align", "moran_area", "moran_hema", "deg20"]
GRP = ["tumour", "gland", "stroma", "unassigned"]


def compartment(lab):
    return np.where(lab == 0, 0, np.where(np.isin(lab, list(BEN)), 1, np.where(np.isin(lab, list(STR)), 2, 3)))


def boot_c(e, t, s, n=1000, seed=0):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        i = rng.integers(0, len(e), len(e)); out.append(C.cindex(e[i] > 0, t[i], s[i]))
    return [round(float(np.percentile(out, 2.5)), 3), round(float(np.percentile(out, 97.5)), 3)]


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    from sklearn.model_selection import StratifiedKFold
    from sklearn.ensemble import HistGradientBoostingRegressor
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
    rng = np.random.default_rng(0); S = []
    with torch.no_grad():
        for j, (ds, c) in enumerate(cases):
            cf = f"{C.ROOT}/cellfeat/{ds}/{c}.npz"
            if not os.path.exists(cf):
                S.append(None); continue
            keep = np.load(f"{C.ROOT}/tissue/{ds}/{c}.npz")["keep"]; idx = np.where(keep)[0]
            net, mu, sd = nets[fold[j]]
            X = torch.tensor(C.feats("midnight", ds, [c])[0][keep].astype(np.float32), device=dev)
            a, s = net.parts(X); v = ((s - mu) / sd).cpu().numpy(); a = a.cpu().numpy(); y = float((a * v).sum())
            z = np.load(cf); names = [str(x) for x in z["names"]]; F = z["X"][idx][:, [names.index(f) for f in FEATS]].astype(np.float32)
            g = compartment(np.load(f"{DICT}/{ds}/{c}.npz")["label"][idx])
            sub = rng.choice(len(v), min(400, len(v)), replace=False)
            S.append(dict(v=v, a=a, y=y, F=F, g=g, sub=sub, D=a * (v - y) / np.clip(1 - a, 1e-6, None)))
            if j % 100 == 0:
                print(j, c, len(v), flush=True)
    ok = np.array([s is not None for s in S]); print("slides with cell features", ok.sum(), flush=True)
    res = {}; slide = pd.DataFrame(dict(case=[c for _, c in cases], cohort=coh, event=ev, time=tm, fold=fold, ok=ok))
    slide["y"] = [s["y"] if s else np.nan for s in S]
    for name in ("nuc", "cmp", "both"):
        slide[f"y_{name}"] = np.nan
    tile_r2 = {}
    for ds in ("owl", "tcga"):
        J = np.where(ok & (coh == ds))[0]; tile_r2[ds] = {}
        allv = np.concatenate([S[j]["v"] for j in J]); allg = np.concatenate([S[j]["g"] for j in J])
        pred = {nm: np.zeros(len(allv)) for nm in ("nuc", "cmp", "both")}
        off = np.cumsum([0] + [len(S[j]["v"]) for j in J])
        for k in range(5):
            tr = [j for j in J if fold[j] != k]; te_pos = [p for p, j in enumerate(J) if fold[j] == k]
            Ftr = np.concatenate([S[j]["F"][S[j]["sub"]] for j in tr]); gtr = np.concatenate([S[j]["g"][S[j]["sub"]] for j in tr])
            vtr = np.concatenate([S[j]["v"][S[j]["sub"]] for j in tr])
            oh = lambda g: np.eye(4, dtype=np.float32)[g]
            gm = np.array([vtr[gtr == q].mean() if (gtr == q).any() else vtr.mean() for q in range(4)])
            mn = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(Ftr, vtr)
            mb = HistGradientBoostingRegressor(max_iter=300, random_state=0).fit(np.c_[Ftr, oh(gtr)], vtr)
            for p in te_pos:
                j = J[p]; sl = slice(off[p], off[p + 1]); Fj, gj = S[j]["F"], S[j]["g"]
                pred["nuc"][sl] = mn.predict(Fj); pred["cmp"][sl] = gm[gj]; pred["both"][sl] = mb.predict(np.c_[Fj, oh(gj)])
        tot = ((allv - allv.mean()) ** 2).sum()
        for nm, pv in pred.items():
            within = {GRP[q]: round(float(np.corrcoef(allv[allg == q], pv[allg == q])[0, 1] ** 2), 3) for q in range(4) if nm != "cmp"}
            tile_r2[ds][nm] = dict(r2=round(float(1 - ((allv - pv) ** 2).sum() / tot), 3), within_compartment_corr2=within)
            for p, j in enumerate(J):
                slide.loc[j, f"y_{nm}"] = float((S[j]["a"] * pv[off[p]:off[p + 1]]).sum())
        # where the top 1% tiles by deletion effect sit, and attention share per compartment
        top = []; att = np.zeros(4); n_t = np.zeros(4)
        for j in J:
            D, g, a = S[j]["D"], S[j]["g"], S[j]["a"]; kk = max(1, int(round(0.01 * len(D))))
            top.append(g[np.argsort(-D)[:kk]]); att += np.bincount(g, weights=a, minlength=4) / len(J); n_t += np.bincount(g, minlength=4)
        top = np.concatenate(top)
        tile_r2[ds]["top1pct_deletion_compartment"] = {GRP[q]: round(float((top == q).mean()), 3) for q in range(4)}
        tile_r2[ds]["attention_share"] = {GRP[q]: round(float(att[q]), 3) for q in range(4)}
        tile_r2[ds]["tile_share"] = {GRP[q]: round(float(n_t[q] / n_t.sum()), 3) for q in range(4)}
    res["tiles"] = tile_r2
    from lifelines import CoxPHFitter
    res["slides"] = {}
    for ds in ("owl", "tcga"):
        d = slide[slide.ok & (slide.cohort == ds)].copy(); e, t = d.event.values, d.time.values; out = {}
        for col in ("y", "y_nuc", "y_cmp", "y_both"):
            out[col] = dict(c=round(float(C.cindex(e > 0, t, d[col].values)), 3), ci=boot_c(e, t, d[col].values),
                            r_with_y=round(float(np.corrcoef(d[col], d.y)[0, 1]), 3))
        for base in ("y_nuc", "y_cmp", "y_both"):
            u = (d.y - d[base]).values
            cx = CoxPHFitter().fit(pd.DataFrame(dict(mapped=(d[base] - d[base].mean()) / d[base].std(), unmapped=(u - u.mean()) / u.std(),
                                                     time=t, event=e)), "time", "event")
            out[f"unmapped_vs_{base}"] = dict(c=round(float(C.cindex(e > 0, t, u)), 3), ci=boot_c(e, t, u),
                                              hr_mapped=round(float(cx.hazard_ratios_["mapped"]), 3), p_mapped=float(cx.summary.p["mapped"]),
                                              hr_unmapped=round(float(cx.hazard_ratios_["unmapped"]), 3), p_unmapped=float(cx.summary.p["unmapped"]))
        res["slides"][ds] = out
    slide.to_csv(f"{C.ROOT}/exp/e23_slides.csv", index=False)
    json.dump(res, open(f"{C.LOGS}/exp/e23_mapping_slide.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
