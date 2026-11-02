"""E1: capacity baselines on raw tile embeddings (Model A setting: train on RUMC only).

models:
  meanpool  slide mean embedding -> scaler -> PCA-64 -> ridge Cox (alpha by inner 3-fold CV)
  abmil     gated-attention MIL, Cox loss
  addmil    additive MIL: slide risk = mean of per-tile MLP risks (= PAC with residual only), Cox loss
Evaluation: event-stratified 5-fold x R repeats on RUMC (out-of-fold C per repeat), and the
5 fold-models of repeat 0 averaged (z-scored risks) on all TCGA patients (external C).
--tissue: E0 tissue tiles only (outputs get suffix _t).
Writes logs/exp/e1_<enc>_<model>.json and data/exp/e1/<enc>_<model>_oof.npz
"""
import argparse, json, os, sys, time
import numpy as np
import torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
import common as C


def cox_loss(risk, time, event):
    o = torch.argsort(time, descending=True)
    r, e = risk[o], event[o]
    if e.sum() == 0:
        return risk.sum() * 0
    return -((r - torch.logcumsumexp(r, 0)) * e).sum() / e.sum()


class ABMIL(nn.Module):
    def __init__(self, D, H=256, p=0.25):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(D, H), nn.GELU(), nn.Dropout(p))
        self.a, self.b, self.w = nn.Linear(H, 128), nn.Linear(H, 128), nn.Linear(128, 1)
        self.head = nn.Linear(H, 1)

    def forward(self, x):
        h = self.enc(x)
        att = torch.softmax(self.w(torch.tanh(self.a(h)) * torch.sigmoid(self.b(h))).squeeze(-1), 0)
        return self.head((att[:, None] * h).sum(0)).squeeze()


class AddMIL(nn.Module):
    def __init__(self, D, H=256, p=0.25):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(D, H), nn.GELU(), nn.Dropout(p), nn.Linear(H, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1).mean()


def train_mil(kind, Xtr, ttr, etr, D, dev, epochs=30, bs=32, ntile=2048, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = (ABMIL if kind == "abmil" else AddMIL)(D).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-4, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    T, E = torch.tensor(ttr, device=dev).float(), torch.tensor(etr, device=dev).float()
    for ep in range(epochs):
        m.train()
        for b in np.array_split(rng.permutation(len(Xtr)), max(1, len(Xtr) // bs)):
            rs = []
            for i in b:
                x = Xtr[i]
                if len(x) > ntile:
                    x = x[rng.choice(len(x), ntile, replace=False)]
                rs.append(m(torch.from_numpy(x).to(dev, non_blocking=True).float()))
            loss = cox_loss(torch.stack(rs), T[b], E[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return m


@torch.no_grad()
def predict_mil(m, X, dev):
    m.eval()
    return np.array([m(torch.from_numpy(x).to(dev).float()).item() for x in X])


def fit_meanpool(Xtr, ttr, etr):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    y = np.array(list(zip(etr.astype(bool), ttr)), dtype=[("e", bool), ("t", float)])
    best, best_c = None, -1
    inner = C.folds(etr, 3, 1, seed=99)[0]
    for alpha in (1, 10, 100, 1000, 10000):
        cs = []
        for k in range(3):
            tr, va = inner != k, inner == k
            p = make_pipeline(StandardScaler(), PCA(64, random_state=0), CoxPHSurvivalAnalysis(alpha=alpha))
            p.fit(Xtr[tr], y[tr]); cs.append(C.cindex(etr[va], ttr[va], p.predict(Xtr[va])))
        if np.mean(cs) > best_c:
            best, best_c = alpha, np.mean(cs)
    p = make_pipeline(StandardScaler(), PCA(64, random_state=0), CoxPHSurvivalAnalysis(alpha=best))
    p.fit(Xtr, y)
    return p, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="hoptimus0"); ap.add_argument("--model", default="meanpool")
    ap.add_argument("--repeats", type=int, default=3); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--tissue", action="store_true")
    a = ap.parse_args(); sfx = "_t" if a.tissue else ""; dev = a.device; t0 = time.time()
    L = {ds: C.labels(ds) for ds in ("owl", "tcga")}
    F = {ds: C.feats(a.enc, ds, list(L[ds].index), tissue=a.tissue) for ds in ("owl", "tcga")}
    D = F["owl"][0].shape[1]
    if a.model == "meanpool":
        F = {ds: np.stack([x.astype(np.float32).mean(0) for x in F[ds]]) for ds in F}
    e, t = L["owl"].event.values, L["owl"].time.values
    print(f"{a.enc}/{a.model}: loaded {len(e)} RUMC + {len(L['tcga'])} TCGA in {time.time()-t0:.0f}s", flush=True)

    res = dict(enc=a.enc, model=a.model, fold_c=[], oof_c=[], alphas=[])
    oof_all, tcga_preds = [], []
    for r, fold in enumerate(C.folds(e, 5, a.repeats, seed=0)):
        oof = np.zeros(len(e))
        for k in range(5):
            tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
            if a.model == "meanpool":
                m, alpha = fit_meanpool(F["owl"][tr], t[tr], e[tr]); res["alphas"].append(alpha)
                oof[te] = m.predict(F["owl"][te]); pt = m.predict(F["tcga"]) if r == 0 else None
            else:
                m = train_mil(a.model, [F["owl"][i] for i in tr], t[tr], e[tr], D, dev, epochs=a.epochs, seed=r * 10 + k)
                oof[te] = predict_mil(m, [F["owl"][i] for i in te], dev)
                pt = predict_mil(m, F["tcga"], dev) if r == 0 else None
            res["fold_c"].append(C.cindex(e[te], t[te], oof[te]))
            if pt is not None:
                tcga_preds.append((pt - pt.mean()) / (pt.std() + 1e-8))
            print(f"  rep{r} fold{k} C={res['fold_c'][-1]:.3f} ({time.time()-t0:.0f}s)", flush=True)
        res["oof_c"].append(C.cindex(e, t, oof)); oof_all.append(oof)
        print(f" rep{r} OOF C={res['oof_c'][-1]:.3f}", flush=True)
    tp = np.mean(tcga_preds, 0)
    res["tcga_c"] = C.cindex(L["tcga"].event.values, L["tcga"].time.values, tp)
    res["oof_c_mean"], res["oof_c_sd"] = float(np.mean(res["oof_c"])), float(np.std(res["oof_c"]))
    res["fold_c_mean"], res["fold_c_sd"] = float(np.mean(res["fold_c"])), float(np.std(res["fold_c"]))
    res["secs"] = round(time.time() - t0)
    os.makedirs(f"{C.ROOT}/exp/e1", exist_ok=True)
    np.savez(f"{C.ROOT}/exp/e1/{a.enc}_{a.model}{sfx}_oof.npz", oof=np.array(oof_all), tcga=tp,
             cases=np.array(L["owl"].index), tcga_cases=np.array(L["tcga"].index))
    json.dump(res, open(f"{C.LOGS}/exp/e1_{a.enc}_{a.model}{sfx}.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("oof_c_mean", "oof_c_sd", "fold_c_mean", "tcga_c", "secs")}), flush=True)


if __name__ == "__main__":
    main()
