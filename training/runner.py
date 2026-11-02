"""Experiment runner: screen many slide-level recipes on stored features under one protocol.

Protocol (screening): RUMC only, event-stratified 5-fold x R (default 1 repeat, seed 0 = E1 folds).
Saves per experiment: data/exp/runner/<id>.npz (oof [R, 508], cases) and a SEALED TCGA prediction (mean of the
repeat-0 fold models, z-scored) in data/exp/runner/sealed/<id>.npy - never printed or read by the leaderboard.
Leaderboard (paired patient bootstrap vs the parent): separate script, not included.

Config keys: enc (midnight|hoptimus0), view (t|tc|tcr), model (mil|meanpool), H, A (attention dim), heads,
p (dropout), pool (att|topk), topk, tdrop (extra tile dropout), wd, lr, epochs, ntile, loss (cox|coxpair),
pair_w, seeds (models averaged per fold).
Usage: python runner.py --ids m02,m03 --device cuda:0 [--repeats 1]
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
import common as C
from e1_baselines import cox_loss, fit_meanpool

OUT = f"{C.ROOT}/exp/runner"
BASE = dict(enc="midnight", view="t", model="mil", H=256, A=128, heads=1, p=0.25, pool="att", topk=0.1, tdrop=0.0,
            wd=1e-2, lr=2e-4, epochs=30, ntile=2048, loss="cox", pair_w=0.2, seeds=1, fold_seed=0)
EXPS = {
    "m00": {},                                             # parent = E1 Midnight ABMIL tissue-only
    "m01": dict(model="meanpool"),
    "m02": dict(H=64), "m03": dict(A=32), "m04": dict(heads=2), "m05": dict(pool="topk", topk=0.1),
    "m06": dict(tdrop=0.25), "m08": dict(loss="coxpair"), "m10": dict(wd=1e-1), "m13": dict(seeds=3),
    "m41": dict(p=0.5), "m42": dict(epochs=15), "m43": dict(epochs=60, lr=1e-4), "m44": dict(ntile=4096),
    "m45": dict(ntile=1024), "m46": dict(H=512),
    "h00": dict(enc="hoptimus0"), "h12": dict(enc="hoptimus0", view="tc"), "h15": dict(enc="hoptimus0", view="tcr"),
    "m12": dict(view="tc"),
    "m50": dict(pool="topk", topk=0.05), "m51": dict(pool="topk", topk=0.2), "m52": dict(pool="topk", topk=0.02),
    "m53": dict(pool="topk", topk=0.1, seeds=3), "h05": dict(enc="hoptimus0", pool="topk", topk=0.1),
    "h55": dict(enc="hoptimus0", view="tc", pool="topk", topk=0.1),
}
# confirmation round: finalists on fresh folds (fold_seed 100), run with --repeats 3
for _k in ("m00", "m50", "m05", "m04", "m44", "h05", "m53", "m13"):
    EXPS[_k + "c"] = {**EXPS[_k], "fold_seed": 100}


class MIL(nn.Module):
    def __init__(self, D, H, A, heads, p, pool, topk):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(D, H), nn.GELU(), nn.Dropout(p))
        self.a, self.b = nn.Linear(H, A * heads), nn.Linear(H, A * heads)
        self.w = nn.ModuleList([nn.Linear(A, 1) for _ in range(heads)])
        self.head = nn.ModuleList([nn.Linear(H, 1) for _ in range(heads)])
        self.heads, self.A, self.pool, self.topk = heads, A, pool, topk

    def forward(self, x):
        h = self.enc(x)
        if self.pool == "topk":                                # soft top-k: mean of the k highest tile scores
            s = self.head[0](h).squeeze(-1); k = max(1, int(self.topk * len(s)))
            return s.topk(k).values.mean()
        g = (torch.tanh(self.a(h)) * torch.sigmoid(self.b(h))).view(len(h), self.heads, self.A)
        r = []
        for j in range(self.heads):
            att = torch.softmax(self.w[j](g[:, j]).squeeze(-1), 0)
            r.append(self.head[j]((att[:, None] * h).sum(0)))
        return torch.stack(r).mean()


def pair_loss(risk, t, e):
    """pairwise logistic over comparable pairs (i had the event earlier than j's time)"""
    d = risk[:, None] - risk[None, :]; comp = (e[:, None] > 0) & (t[:, None] < t[None, :])
    return nn.functional.softplus(-d[comp]).mean() if comp.any() else risk.sum() * 0


def train(cfg, X, t, e, dev, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = MIL(X[0].shape[1], cfg["H"], cfg["A"], cfg["heads"], cfg["p"], cfg["pool"], cfg["topk"]).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg["epochs"])
    T, E = torch.tensor(t, device=dev).float(), torch.tensor(e, device=dev).float()
    for _ in range(cfg["epochs"]):
        m.train()
        for b in np.array_split(rng.permutation(len(X)), max(1, len(X) // 32)):
            rs = []
            for i in b:
                x = X[i]; n = len(x)
                keep = n if cfg["tdrop"] == 0 else max(16, int(n * (1 - cfg["tdrop"])))
                k = min(cfg["ntile"], keep)
                if k < n:
                    x = x[rng.choice(n, k, replace=False)]
                rs.append(m(torch.from_numpy(x).to(dev).float()))
            r = torch.stack(rs); loss = cox_loss(r, T[b], E[b])
            if cfg["loss"] == "coxpair":
                loss = (1 - cfg["pair_w"]) * loss + cfg["pair_w"] * pair_loss(r, T[b], E[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()
    return m


@torch.no_grad()
def predict(m, X, dev):
    m.eval(); return np.array([m(torch.from_numpy(x).to(dev).float()).item() for x in X])


_CACHE = {}


def data(enc, view):
    key = (enc, view)
    if key in _CACHE:
        return _CACHE[key]
    _CACHE.clear()                                         # keep one feature set in memory
    L = {ds: C.labels(ds) for ds in ("owl", "tcga")}
    if view == "t":
        X = {ds: C.feats(enc, ds, list(L[ds].index), tissue=True) for ds in L}
    else:
        from e8_multiview import load, bag
        def one(ds, c):
            it = load(enc, ds, c, ring="r" in view); return bag(it, view).astype(np.float16)
        X = {}
        for ds in L:
            with ThreadPoolExecutor(16) as ex:
                X[ds] = list(ex.map(lambda c: one(ds, c), L[ds].index))
    _CACHE[key] = (L, X); return L, X


def run(eid, dev, repeats):
    cfg = {**BASE, **EXPS[eid]}; t0 = time.time()
    L, X = data(cfg["enc"], cfg["view"])
    e, t = L["owl"].event.values, L["owl"].time.values
    if cfg["model"] == "meanpool":
        M = {ds: np.stack([x.astype(np.float32).mean(0) for x in X[ds]]) for ds in X}
    oofs, tp = [], []
    for r, fold in enumerate(C.folds(e, 5, repeats, seed=cfg["fold_seed"])):
        oof = np.zeros(len(e))
        for k in range(5):
            tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
            if cfg["model"] == "meanpool":
                m, _ = fit_meanpool(M["owl"][tr], t[tr], e[tr]); oof[te] = m.predict(M["owl"][te])
                pt = m.predict(M["tcga"]) if r == 0 else None
            else:
                ps, pts = [], []
                for s in range(cfg["seeds"]):
                    m = train(cfg, [X["owl"][i] for i in tr], t[tr], e[tr], dev, seed=r * 100 + k * 10 + s)
                    ps.append(predict(m, [X["owl"][i] for i in te], dev))
                    if r == 0:
                        pts.append(predict(m, X["tcga"], dev))
                oof[te] = np.mean(ps, 0); pt = np.mean(pts, 0) if r == 0 else None
            if pt is not None:
                tp.append((pt - pt.mean()) / (pt.std() + 1e-8))
        oofs.append(oof)
    os.makedirs(f"{OUT}/sealed", exist_ok=True)
    np.savez(f"{OUT}/{eid}.npz", oof=np.array(oofs), cases=np.array(L["owl"].index), cfg=json.dumps(cfg))
    np.save(f"{OUT}/sealed/{eid}.npy", np.mean(tp, 0))
    c = [C.cindex(e, t, o) for o in oofs]
    print(f"{eid} OOF {np.mean(c):.3f} ({time.time() - t0:.0f}s) {json.dumps(EXPS[eid])}", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ids", required=True); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--repeats", type=int, default=1); a = ap.parse_args()
    for eid in a.ids.split(","):
        if os.path.exists(f"{OUT}/{eid}.npz"):
            print(eid, "done already", flush=True); continue
        try:
            run(eid, a.device, a.repeats)
        except Exception as ex:
            print(eid, "FAILED", type(ex).__name__, ex, flush=True)


if __name__ == "__main__":
    main()
