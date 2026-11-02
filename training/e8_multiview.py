"""E8: multi-view tiles - each 112 um tile seen with its surroundings, like a pathologist toggling objectives.

Per 20x tissue tile (E0 filter): concat of
  t  its own 112 um embedding
  c  embedding of the 448 um context tile it sits in (nearest context-tile centre, full-coverage grid)
  r  mean embedding of the context tiles in the 3x3 block around that parent (~1.3 mm field, parent included)
Views: t | tc | tcr. Models as E1 (meanpool ridge Cox; addmil = per-tile MLP risk averaged, stays additive; abmil).
Same folds/protocol as E1 (5 x R RUMC OOF; TCGA = repeat-0 fold models averaged).
Writes logs/exp/e8_<enc>_<model>_<view>.json
"""
import argparse, json, os, sys, time
import numpy as np, torch
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(__file__))
import common as C
from e1_baselines import cox_loss, ABMIL, AddMIL, fit_meanpool


def load(enc, ds, case, ring=True):
    z = np.load(f"{C.ROOT}/feats/{enc}/{ds}/{case}.npz"); T = C.tiles(ds, case)
    keep = np.load(f"{C.ROOT}/tissue/{ds}/{case}.npz")["keep"]
    x20, xc = z["x20"][keep], z["xctx"]
    s20, sc = T["meta"]["tile20_l0"], T["meta"]["tilectx_l0"]
    cc = T["coordsctx"] + sc / 2
    par = cKDTree(cc).query(T["coords20"][keep] + s20 / 2)[1]
    if not ring:                                   # memory saver when no view uses r
        return x20, xc, par.astype(np.int32)
    g = np.round(T["coordsctx"] / sc).astype(int); g -= g.min(0) - 1
    G = -np.ones(g.max(0) + 2, int); G[g[:, 0], g[:, 1]] = np.arange(len(g))
    ring = np.zeros(xc.shape, np.float32); cnt = np.zeros(len(g), np.float32)
    for da in (-1, 0, 1):
        for db in (-1, 0, 1):
            j = G[g[:, 0] + da, g[:, 1] + db]; ok = j >= 0
            ring[ok] += xc[j[ok]]; cnt[ok] += 1
    ring /= cnt[:, None]
    return x20, np.concatenate([xc, ring.astype(np.float16)], 1), par.astype(np.int32)


def bag(item, view, idx=None):
    x20, ctx, par = item
    if idx is not None:
        x20, par = x20[idx], par[idx]
    D = x20.shape[1]
    parts = [x20] + ([ctx[par, :D]] if "c" in view else []) + ([ctx[par, D:]] if "r" in view else [])
    return np.concatenate(parts, 1)


def train(kind, items, view, ttr, etr, D, dev, epochs=30, bs=32, ntile=2048, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = (ABMIL if kind == "abmil" else AddMIL)(D).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    T, E = torch.tensor(ttr, device=dev).float(), torch.tensor(etr, device=dev).float()
    for ep in range(epochs):
        m.train()
        for b in np.array_split(rng.permutation(len(items)), max(1, len(items) // bs)):
            rs = []
            for i in b:
                n = len(items[i][0]); idx = rng.choice(n, ntile, replace=False) if n > ntile else None
                rs.append(m(torch.from_numpy(bag(items[i], view, idx)).to(dev).float()))
            loss = cox_loss(torch.stack(rs), T[b], E[b]); opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return m


@torch.no_grad()
def predict(m, items, view, dev):
    m.eval()
    return np.array([m(torch.from_numpy(bag(it, view)).to(dev).float()).item() for it in items])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="hoptimus0"); ap.add_argument("--models", default="meanpool,addmil,abmil")
    ap.add_argument("--views", default="t,tc,tcr"); ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(); dev = a.device; t0 = time.time()
    L = {ds: C.labels(ds) for ds in ("owl", "tcga")}
    I = {}
    for ds in L:
        with ThreadPoolExecutor(16) as ex:
            I[ds] = list(ex.map(lambda c: load(a.enc, ds, c, ring="r" in a.views), L[ds].index))
    print(f"loaded {time.time()-t0:.0f}s", flush=True)
    e, t = L["owl"].event.values, L["owl"].time.values; et, tt = L["tcga"].event.values, L["tcga"].time.values
    D1 = I["owl"][0][0].shape[1]
    for model in a.models.split(","):
        for view in a.views.split(","):
            D = D1 * len(view); res = dict(enc=a.enc, model=model, view=view, oof_c=[], fold_c=[]); tp = []
            if model == "meanpool":
                M = {ds: np.stack([bag(it, view).astype(np.float32).mean(0) for it in I[ds]]) for ds in L}
            for r, fold in enumerate(C.folds(e, 5, a.repeats, seed=0)):
                oof = np.zeros(len(e))
                for k in range(5):
                    tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
                    if model == "meanpool":
                        m, _ = fit_meanpool(M["owl"][tr], t[tr], e[tr]); oof[te] = m.predict(M["owl"][te])
                        pt = m.predict(M["tcga"]) if r == 0 else None
                    else:
                        m = train(model, [I["owl"][i] for i in tr], view, t[tr], e[tr], D, dev, seed=r * 10 + k)
                        oof[te] = predict(m, [I["owl"][i] for i in te], view, dev)
                        pt = predict(m, I["tcga"], view, dev) if r == 0 else None
                    res["fold_c"].append(C.cindex(e[te], t[te], oof[te]))
                    if pt is not None:
                        tp.append((pt - pt.mean()) / (pt.std() + 1e-8))
                res["oof_c"].append(C.cindex(e, t, oof))
            res["tcga_c"] = C.cindex(et, tt, np.mean(tp, 0))
            res["oof_c_mean"], res["oof_c_sd"] = float(np.mean(res["oof_c"])), float(np.std(res["oof_c"]))
            os.makedirs(f"{C.ROOT}/exp/e8", exist_ok=True)
            np.save(f"{C.ROOT}/exp/e8/{a.enc}_{model}_{view}_tcga.npy", np.mean(tp, 0))
            json.dump(res, open(f"{C.LOGS}/exp/e8_{a.enc}_{model}_{view}.json", "w"), indent=1)
            print(f"{model} {view}: OOF {res['oof_c_mean']:.3f}±{res['oof_c_sd']:.3f} TCGA {res['tcga_c']:.3f} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
