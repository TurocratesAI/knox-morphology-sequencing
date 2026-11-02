"""Joint-cohort runner: train on RUMC + TCGA together, evaluate WITHIN each cohort.

Folds: 5-fold x R over all 806 patients, stratified by cohort x event (fold_seed). Each OOF prediction is scored
with Harrell's C within RUMC and within TCGA separately (never pooled across cohorts); 'mean' = average of the two.
Losses:
  cox       Cox partial likelihood with risk sets pooled over cohorts (as v0)
  stratcox  Cox with risk sets restricted to the same cohort (stratified partial likelihood)
  disc      discrete-time proportional-odds hazard: logit h_k = r + b_k on 4 intervals (training event-time
            quartiles), censoring-aware NLL; r is the slide risk used for ranking
Other keys as runner.py (MIL architecture, pooling, tiles, seeds).
Saves data/exp/runner_joint/<id>.npz (oof [R, 806], cases, cohort). Usage: python runner_joint.py --ids j00,j01
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
import common as C
from runner import MIL, BASE

OUT = f"{C.ROOT}/exp/runner_joint"
EXPS = {
    "j00": dict(loss="cox"), "j01": dict(loss="stratcox"), "j02": dict(loss="disc"),
    "j03": dict(loss="stratcox", pool="topk", topk=0.05), "j04": dict(loss="stratcox", heads=2),
    "j05": dict(loss="stratcox", enc="hoptimus0"), "j06": dict(loss="disc", enc="hoptimus0"),
    # Virchow2 (E19: <=4000 stride-subsampled E0 tiles per slide); Midnight on the same tiles; both concatenated
    "j10": dict(loss="cox", enc="virchow2"), "j11": dict(loss="cox", enc="midnight@v2"), "j12": dict(loss="cox", enc="mid+v2"),
    # same with per-dimension standardisation of tile features (label-free stats from 200k random tiles; Virchow2 has
    # ~6x larger feature scale than Midnight and one dimension with SD 12.7, which the Midnight-tuned head handles badly)
    "j13": dict(loss="cox", enc="virchow2", std=True), "j14": dict(loss="cox", enc="midnight@v2", std=True),
    "j15": dict(loss="cox", enc="mid+v2", std=True), "j16": dict(loss="cox", enc="hoptimus0", std=True),
    # E26 encoders on every second Virchow2 tile (~2,000 per slide); Midnight on exactly the same tiles for a fair comparison
    "j20": dict(loss="cox", enc="uni2h"), "j21": dict(loss="cox", enc="gigapath"), "j22": dict(loss="cox", enc="phikon2"),
    "j23": dict(loss="cox", enc="midnight@e26"), "j24": dict(loss="cox", enc="hoptimus0@e26"),
}
E26 = ("uni2h", "gigapath", "phikon2")


def standardise(X, seed=0, n=200_000):
    rng = np.random.default_rng(seed); per = max(1, n // len(X))
    S = np.concatenate([x[rng.choice(len(x), min(per, len(x)), replace=False)] for x in X]).astype(np.float32)
    mu, sd = S.mean(0), S.std(0) + 1e-6
    return [((x.astype(np.float32) - mu) / sd).astype(np.float16) for x in X]


def load_enc(enc, ds, cases):
    """Tile features per case. 'virchow2' files are already E0-filtered and subsampled (idx into coords20);
    'midnight@v2' takes the Midnight rows at those idx; 'mid+v2' concatenates the two."""
    if enc in E26 or enc.endswith("@e26"):                    # rows at virchow2 idx[::2]; a few early slides were run at stride 1
        def one(c):
            keep = np.load(f"{C.ROOT}/feats/virchow2/{ds}/{c}.npz")["idx"][::2]
            if enc.endswith("@e26"):                          # midnight / hoptimus0 files hold every coords20 row
                return C._load(f"{C.ROOT}/feats/{enc[:-4]}/{ds}/{c}.npz", "x20")[keep]
            z = np.load(f"{C.ROOT}/feats/{enc}/{ds}/{c}.npz"); return z["x20"][np.isin(z["idx"], keep)]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(16) as ex:
            return list(ex.map(one, cases))
    if enc not in ("virchow2", "midnight@v2", "mid+v2"):
        return C.feats(enc, ds, cases, tissue=True)
    V = [np.load(f"{C.ROOT}/feats/virchow2/{ds}/{c}.npz") for c in cases]
    if enc == "virchow2":
        return [v["x20"] for v in V]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(16) as ex:                       # index per slide so full Midnight bags are never all in RAM
        M = list(ex.map(lambda cv: C._load(f"{C.ROOT}/feats/midnight/{ds}/{cv[0]}.npz", "x20")[cv[1]["idx"]], zip(cases, V)))
    return M if enc == "midnight@v2" else [np.concatenate([m, v["x20"]], 1) for m, v in zip(M, V)]


def cox(r, t, e):
    o = torch.argsort(t, descending=True); r, e = r[o], e[o]
    return -((r - torch.logcumsumexp(r, 0)) * e).sum() / e.sum().clamp(min=1)


def strat_cox(r, t, e, g):
    tot, ne = 0, 0
    for k in torch.unique(g):
        m = g == k
        if e[m].sum() > 0:
            tot = tot + cox(r[m], t[m], e[m]) * e[m].sum(); ne = ne + e[m].sum()
    return tot / ne if ne > 0 else r.sum() * 0


def disc_nll(r, t, e, cuts, b):
    """intervals [0,c1),[c1,c2),[c2,c3),[c3,inf); hazard in interval k: sigmoid(r + b_k)"""
    k = torch.bucketize(t, cuts)                          # interval index of the event/censoring time
    h = torch.sigmoid(r[:, None] + b[None, :]).clamp(1e-6, 1 - 1e-6)
    K = len(b); idx = torch.arange(K, device=r.device)[None, :]
    surv = (torch.log1p(-h) * (idx < k[:, None])).sum(1)  # survived all earlier intervals
    ev = torch.log(h.gather(1, k[:, None].clamp(max=K - 1)).squeeze(1))
    cen_last = torch.log1p(-h.gather(1, k[:, None].clamp(max=K - 1)).squeeze(1))
    return -(surv + torch.where(e > 0, ev, cen_last * 0)).mean()


def train(cfg, X, t, e, g, dev, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = MIL(X[0].shape[1], cfg["H"], cfg["A"], cfg["heads"], cfg["p"], cfg["pool"], cfg["topk"]).to(dev)
    b = nn.Parameter(torch.zeros(4, device=dev))
    opt = torch.optim.AdamW(list(m.parameters()) + [b], lr=cfg["lr"], weight_decay=cfg["wd"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    T, E, G = (torch.tensor(v, device=dev).float() for v in (t, e, g))
    cuts = torch.tensor(np.quantile(t[e > 0], [0.25, 0.5, 0.75]), device=dev).float()
    for _ in range(cfg["epochs"]):
        m.train()
        for bi in np.array_split(rng.permutation(len(X)), max(1, len(X) // 32)):
            rs = []
            for i in bi:
                x = X[i]
                if len(x) > cfg["ntile"]:
                    x = x[rng.choice(len(x), cfg["ntile"], replace=False)]
                rs.append(m(torch.from_numpy(x).to(dev).float()))
            r = torch.stack(rs)
            loss = (cox(r, T[bi], E[bi]) if cfg["loss"] == "cox" else strat_cox(r, T[bi], E[bi], G[bi])
                    if cfg["loss"] == "stratcox" else disc_nll(r, T[bi], E[bi], cuts, b))
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    return m


@torch.no_grad()
def predict(m, X, dev):
    m.eval(); return np.array([m(torch.from_numpy(x).to(dev).float()).item() for x in X])


def folds(e, g, repeats, seed):
    from sklearn.model_selection import StratifiedKFold
    y = np.array([f"{a}{int(b)}" for a, b in zip(g, e)]); out = []
    for r in range(repeats):
        f = np.zeros(len(e), int)
        for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed + r).split(y, y)):
            f[te] = k
        out.append(f)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ids", required=True); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--repeats", type=int, default=2); ap.add_argument("--fold_seed", type=int, default=200)
    ap.add_argument("--tag", default="", help="output suffix, e.g. _f300 for a fresh-fold confirmation run"); a = ap.parse_args()
    cache = {}
    for eid in a.ids.split(","):
        if os.path.exists(f"{OUT}/{eid}{a.tag}.npz"):
            print(eid, "done already", flush=True); continue
        cfg = {**BASE, **EXPS[eid]}; t0 = time.time()
        if cfg["enc"] not in cache:
            cache.clear(); L = {ds: C.labels(ds) for ds in ("owl", "tcga")}
            X = sum((load_enc(cfg["enc"], ds, list(L[ds].index)) for ds in L), [])
            cases = np.concatenate([L[ds].index.values for ds in L]); g = np.concatenate([[k] * len(L[ds]) for k, ds in enumerate(L)])
            t = np.concatenate([L[ds].time.values for ds in L]); e = np.concatenate([L[ds].event.values for ds in L]).astype(float)
            cache[cfg["enc"]] = (X, cases, g, t, e)
        X, cases, g, t, e = cache[cfg["enc"]]
        if cfg.get("std"):
            X = standardise(X)
        oofs = []
        for r, fold in enumerate(folds(e, g, a.repeats, a.fold_seed)):
            oof = np.zeros(len(e))
            for k in range(5):
                tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
                ps = []
                for s in range(cfg["seeds"]):
                    m = train(cfg, [X[i] for i in tr], t[tr], e[tr], g[tr], a.device, seed=r * 100 + k * 10 + s)
                    ps.append(predict(m, [X[i] for i in te], a.device))
                oof[te] = np.mean(ps, 0)
            oofs.append(oof)
        os.makedirs(OUT, exist_ok=True)
        np.savez(f"{OUT}/{eid}{a.tag}.npz", oof=np.array(oofs), cases=cases, cohort=g, cfg=json.dumps(cfg))
        o = np.mean(oofs, 0); cr = [C.cindex(e[g == k] > 0, t[g == k], o[g == k]) for k in (0, 1)]
        print(f"{eid}{a.tag} RUMC {cr[0]:.3f} TCGA {cr[1]:.3f} mean {np.mean(cr):.3f} ({time.time() - t0:.0f}s) {json.dumps(EXPS[eid])}", flush=True)


if __name__ == "__main__":
    main()
