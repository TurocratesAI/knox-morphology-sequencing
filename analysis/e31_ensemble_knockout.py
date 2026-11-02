"""E31: knockouts for the deployed three-encoder model (v2), out of fold, plus a mask-edit (re-selection) check.

Deployed model: $KNOX_MODEL_DIR/owl_v2 (5 fold models per encoder; folds StratifiedKFold(5, seed 2026) over
cohort x event, as train_final_v2). Each slide is scored only by the fold models that did not train on it, on its E26
tiles (virchow2 idx[::2], ~2,000), exactly as the container scores: per encoder e, y_e = sum a v,
z_e = (y_e - enc_mu)/enc_sd; ensemble Z = mean_e z_e. All encoders read the same tiles, so removing a set S is exact:
y_e,-S = (y_e - sum_S a v)/(1 - A_S). Single-tile effect (the map the container writes): D_i = mean_e D_e,i / enc_sd.

1. Deletion curves (paper Fig. 2): remove tiles by D (map), by mean attention, by mean share a v, or at random
   (3 permutations); drop Z - Z_-S in units of the SD of held-out Z.
2. Compartments (paper Table 2): Harrell's C of the ensemble when it reads only one compartment, or all but it.
   Share of the top 1% of tiles by D in each compartment and of tissue.
3. Mask edit, Midnight member only (the only encoder with features for every lattice tile): tissue tiles are ranked by
   the container's heatmap (D on analysed tiles, half the nearest analysed value elsewhere) and removed; the
   container then re-runs select() on what is left. Compare its Midnight score drop with the fixed-read exact drop,
   and with random removal under re-selection (noise floor).
Writes logs/exp/e31_ensemble_knockout.json. Usage: python e31_ensemble_knockout.py [cuda:1]
"""
import json, os, sys
import numpy as np, torch
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "training"), os.path.join(_HERE, "..", "container")]
import common as C
from e23_mapping_slide import DICT, compartment, GRP
from inference import MIL, select, spread

MD = f"{C.MODEL_DIR}/owl_v2"
ENCS = ("midnight", "uni2h", "phikon2")
SUB = 5                                                    # mask-edit simulation on every 5th slide
FRAC = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)


def feats(enc, ds, c, keep):
    if enc == "midnight":
        return C._load(f"{C.ROOT}/feats/midnight/{ds}/{c}.npz", "x20")[keep]
    z = np.load(f"{C.ROOT}/feats/{enc}/{ds}/{c}.npz"); m = np.isin(z["idx"], keep)
    assert (z["idx"][m] == keep).all(), (enc, c)
    return z["x20"][m]


def main():
    global LIM
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"; LIM = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    from sklearn.model_selection import StratifiedKFold
    cal = json.load(open(f"{MD}/calib.json")); cfg = cal["cfg"]; nets = {}
    for e in ENCS:
        ce = cal["encoders"][e]; nets[e] = []
        for m in ce["models"]:
            n = MIL(ce["dim"], cfg["H"], cfg["A"], cfg["heads"], cfg["p"], cfg["pool"], cfg["topk"])
            n.load_state_dict(torch.load(f"{MD}/{m['file']}", map_location="cpu")); nets[e].append((n.eval().to(dev), m["mu"], m["sd"]))
    L = {ds: C.labels(ds) for ds in ("owl", "tcga")}
    cases = [(ds, c) for ds in L for c in L[ds].index]
    coh = np.array([ds for ds, _ in cases]); ev = np.concatenate([L[ds].event.values for ds in L]); tm = np.concatenate([L[ds].time.values for ds in L])
    y = np.array([f"{c}{int(v)}" for c, v in zip(coh, ev)]); fold = np.zeros(len(cases), int)
    for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=2026).split(y, y)):
        fold[te] = k
    rng = np.random.default_rng(0)
    Z = np.zeros(len(cases)); curves = {o: [] for o in ("map", "attention", "share", "random")}
    comp = {f"{w}_{g}": np.full(len(cases), np.nan) for w in ("only", "ko") for g in GRP}
    top = np.zeros((len(cases), 4)); area = np.zeros((len(cases), 4))
    me = {o: [] for o in ("fixed", "select_fill05", "select_fill1", "lattice", "random_select", "random_lattice")}
    mid_z, mid_z2, n2 = np.zeros(len(cases)), np.zeros(len(cases)), np.zeros(len(cases)); n_same = 0; sub_idx = []
    with torch.no_grad():
        for j, (ds, c) in enumerate(cases):
            if LIM and j >= LIM:
                print("test stop", flush=True); return
            sel = np.load(f"{C.ROOT}/feats/virchow2/{ds}/{c}.npz")["idx"][::2]
            P = {}
            for e in ENCS:
                net, mu, sd = nets[e][fold[j]]
                a, s = net.parts(torch.tensor(feats(e, ds, c, sel).astype(np.float32), device=dev))
                P[e] = (a.double().cpu().numpy(), ((s - mu) / sd).double().cpu().numpy())
            emu = {e: cal["encoders"][e]["enc_mu"] for e in ENCS}; esd = {e: cal["encoders"][e]["enc_sd"] for e in ENCS}

            def ens(S=None, only=False):
                zs = []
                for e in ENCS:
                    a, v = P[e]; ye = (a * v).sum()
                    if S is not None:
                        A, sv = a[S].sum(), (a[S] * v[S]).sum()
                        ye = sv / A if only else (ye - sv) / (1 - A)
                    zs.append((ye - emu[e]) / esd[e])
                return float(np.mean(zs))
            Z[j] = ens(); n = len(sel)
            D = np.mean([P[e][0] * (P[e][1] - (P[e][0] * P[e][1]).sum()) / np.clip(1 - P[e][0], 1e-9, None) / esd[e] for e in ENCS], 0)
            att = np.mean([P[e][0] for e in ENCS], 0); share = np.mean([P[e][0] * P[e][1] / esd[e] for e in ENCS], 0)

            def drop(order):
                out = []
                for f in FRAC:
                    S = np.zeros(n, bool); S[order[:max(1, int(round(f * n)))]] = True; out.append(Z[j] - ens(S))
                return out
            curves["map"].append(drop(np.argsort(-D))); curves["attention"].append(drop(np.argsort(-att)))
            curves["share"].append(drop(np.argsort(-share))); curves["random"].append(np.mean([drop(rng.permutation(n)) for _ in range(3)], 0).tolist())
            g = compartment(np.load(f"{DICT}/{ds}/{c}.npz")["label"][sel])
            k1 = np.argsort(-D)[:max(1, int(round(0.01 * n)))]
            for q in range(4):
                m = g == q; top[j, q] = (g[k1] == q).mean(); area[j, q] = m.mean()
                if m.any():
                    comp[f"only_{GRP[q]}"][j] = ens(m, only=True)
                    comp[f"ko_{GRP[q]}"][j] = ens(m) if (~m).any() else np.nan
            # 3. mask edit, Midnight member (v2 fold model). Every slide: Midnight z on the sub-lattice tiles (for C).
            keep = np.load(f"{C.ROOT}/tissue/{ds}/{c}.npz")["keep"]; tis = np.where(keep)[0]
            n_same += int(np.array_equal(select(keep), sel))
            T = C.tiles(ds, c); xy = T["coords20"].astype(float); gq = T["coords20"] // T["meta"]["tile20_l0"]
            lat = (gq[:, 0] + 3 * gq[:, 1]) % 6 == 0                 # mask-independent sub-lattice, 1 in 6 positions
            sel2 = np.where(keep & lat)[0]
            Xall = C._load(f"{C.ROOT}/feats/midnight/{ds}/{c}.npz", "x20"); net, mu, sd = nets["midnight"][fold[j]]

            def zmid(rows, parts=False):
                a, s = net.parts(torch.tensor(Xall[rows].astype(np.float32), device=dev)); v = (s - mu) / sd
                z = (float((a * v).sum()) - emu["midnight"]) / esd["midnight"]
                return (z, a.double().cpu().numpy(), v.double().cpu().numpy()) if parts else z
            a0, v0 = P["midnight"]; y0 = (a0 * v0).sum(); mid_z[j] = (y0 - emu["midnight"]) / esd["midnight"]
            z2, a2, w2 = zmid(sel2, True); mid_z2[j] = z2; n2[j] = len(sel2)
            if j % SUB == 0:
                z_sel0 = zmid(select(keep))
                D2 = a2 * (w2 - (a2 * w2).sum()) / np.clip(1 - a2, 1e-9, None) / esd["midnight"]

                def edit(order, rule):
                    out = []
                    for f in FRAC:
                        k2 = keep.copy(); k2[tis[order[:max(1, int(round(f * len(tis))))]]] = False
                        if rule == "fixed":
                            S = ~np.isin(sel, np.where(k2)[0]); A, sv = a0[S].sum(), (a0[S] * v0[S]).sum()
                            out.append(mid_z[j] - ((y0 - sv) / (1 - A) - emu["midnight"]) / esd["midnight"])
                        elif rule == "select":
                            out.append(z_sel0 - zmid(select(k2)))
                        else:
                            out.append(z2 - zmid(np.where(k2 & lat)[0]))
                    return out
                o0 = np.argsort(-spread(D, xy, sel, keep.copy(), 0.5)[tis], kind="stable")
                o1 = np.argsort(-spread(D, xy, sel, keep.copy(), 1.0)[tis], kind="stable")
                o2 = np.argsort(-spread(D2, xy, sel2, keep.copy(), 0.5)[tis], kind="stable"); r = rng.permutation(len(tis))
                for k_, v_ in (("fixed", edit(o0, "fixed")), ("select_fill05", edit(o0, "select")), ("select_fill1", edit(o1, "select")),
                               ("lattice", edit(o2, "lattice")), ("random_select", edit(r, "select")), ("random_lattice", edit(r, "lattice"))):
                    me[k_].append(v_)
                sub_idx.append(j)
            if j % 100 == 0 or LIM:
                print(j, c, round(Z[j], 3), "D[:3]", np.round(curves["map"][-1][:3], 3), "me", {k: np.round(v[-1][:3], 2).tolist() for k, v in me.items() if v}, flush=True)
    sdZ = {ds: float(Z[coh == ds].std()) for ds in L} | {"all": float(Z.std())}
    sdM = {ds: float(mid_z[coh == ds].std()) for ds in L} | {"all": float(mid_z.std())}
    cm = lambda r: {ds: round(float(C.cindex(ev[coh == ds] > 0, tm[coh == ds], r[coh == ds])), 4) for ds in L}
    res = dict(fractions=FRAC, sd_Z=sdZ, sd_midnight=sdM, select_matches_e26=n_same, n=len(cases),
               c_whole=cm(Z), c_midnight_e26=cm(mid_z), c_midnight_lattice=cm(mid_z2), n_lattice_median=float(np.median(n2)),
               z_midnight_lattice_vs_e26=dict(r=float(np.corrcoef(mid_z, mid_z2)[0, 1]), sd_diff=float((mid_z2 - mid_z).std()), mean_diff=float((mid_z2 - mid_z).mean())))
    sub = np.array(sub_idx)
    for name, dd, sdd, ch in (("deletion", curves, sdZ, coh), ("mask_edit", me, sdM, coh[sub])):
        res[name] = {}
        for o, v in dd.items():
            v = np.array(v); res[name][o] = {}
            for grp, m in (("all", np.ones(len(ch), bool)), ("owl", ch == "owl"), ("tcga", ch == "tcga")):
                w = v[m] / sdd[grp]
                res[name][o][grp] = dict(mean=w.mean(0).tolist(), sem=(w.std(0) / np.sqrt(m.sum())).tolist())
            print(name, o, np.round(v.mean(0) / sdd["all"], 3).tolist(), flush=True)
    res["compartments"] = {}
    for ds in L:
        m = coh == ds; r = {}
        for k, v in comp.items():
            ok = m & ~np.isnan(v); r[k] = dict(n=int(ok.sum()), c=round(float(C.cindex(ev[ok] > 0, tm[ok], v[ok])), 3))
        r["top1_share"] = dict(zip(GRP, np.round(top[m].mean(0), 3).tolist())); r["area_share"] = dict(zip(GRP, np.round(area[m].mean(0), 3).tolist()))
        res["compartments"][ds] = r
    print(json.dumps(res["compartments"], indent=0), res["c_whole"], res["c_midnight_e26"], res["c_midnight_lattice"], res["z_midnight_lattice_vs_e26"], "select==e26:", n_same, flush=True)
    json.dump(res, open(f"{C.ROOT}/../logs/exp/e31_ensemble_knockout.json", "w"), indent=1)


if __name__ == "__main__":
    main()
