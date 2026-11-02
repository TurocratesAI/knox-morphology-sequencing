"""E2: outcome-blind phenotype dictionary (spherical k-means on PCA-256 of H-optimus-0 tile features).

- Fit sample: up to 512 tiles per RUMC patient (equal weight per patient).
- K in {16, 32, 64}; stability = ARI between patient-bootstrap refits on a fixed probe set.
- Final dictionaries (fit on all RUMC = Model A) are applied to every tile of all 806 slides.
  Per slide we save hard labels, max cosine and K-fractions (+ an 'unassigned' bucket: max cos below
  the 1st percentile of the fit sample).
Outputs: data/dict/<enc>/K{K}/{centroids.npz, stability.json, assign/<ds>/<case>.npz, fractions_<ds>.csv}
"""
import argparse, itertools, json, os, sys, time
import numpy as np, pandas as pd
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
import common as C


def spherical_kmeans(X, K, iters=100, n_init=4, seed=0):
    """X: [N,d] unit-norm cuda tensor. Returns unit-norm centroids [K,d] with best objective."""
    best, best_obj = None, -1
    g = torch.Generator(device=X.device).manual_seed(seed)
    for r in range(n_init):
        # k-means++ style init on cosine distance
        idx = [torch.randint(len(X), (1,), generator=g, device=X.device).item()]
        d = 1 - X @ X[idx[0]]
        for _ in range(K - 1):
            p = d.clamp_min(0) ** 2
            idx.append(torch.multinomial(p / p.sum(), 1, generator=g).item())
            d = torch.minimum(d, 1 - X @ X[idx[-1]])
        Cn = X[idx].clone()
        for _ in range(iters):
            sim = X @ Cn.T
            lab = sim.argmax(1)
            newC = torch.zeros_like(Cn).index_add_(0, lab, X)
            empty = newC.norm(dim=1) < 1e-8
            if empty.any():                                   # re-seed empty clusters at worst-fit points
                worst = sim.max(1).values.argsort()[: int(empty.sum())]
                newC[empty] = X[worst]
            newC = torch.nn.functional.normalize(newC, dim=1)
            if torch.allclose(newC, Cn, atol=1e-6):
                break
            Cn = newC
        obj = (X @ Cn.T).max(1).values.mean().item()
        if obj > best_obj:
            best, best_obj = Cn, obj
    return best, best_obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", default="hoptimus0"); ap.add_argument("--per_patient", type=int, default=512)
    ap.add_argument("--Ks", default="16,32,64"); ap.add_argument("--boots", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fit", default="owl", choices=["owl", "both"])   # both = RUMC+TCGA (reference dictionary)
    ap.add_argument("--cohort_balance", action="store_true")           # equal total fit tiles per cohort
    ap.add_argument("--tag", default="")                                # output dir suffix, e.g. _ref
    ap.add_argument("--exclude", default="")                            # csv with column case: never used for fitting
    ap.add_argument("--boot_only", action="store_true")                 # stability only, keep existing dictionary
    ap.add_argument("--tissue", action="store_true")                    # use data/tissue keep masks (E0 filter)
    a = ap.parse_args()
    dev = a.device; rng = np.random.default_rng(0)
    out = f"{C.ROOT}/dict/{a.enc}{a.tag}"; os.makedirs(out, exist_ok=True)

    lab = {ds: C.labels(ds) for ds in ("owl", "tcga")}
    t0 = time.time()
    F = {ds: C.feats(a.enc, ds, list(lab[ds].index)) for ds in ("owl", "tcga")}
    print(f"loaded feats in {time.time()-t0:.0f}s", flush=True)
    # tissue keep-mask per slide (all True without --tissue); non-tissue tiles get label K+1 and never enter fits
    keep = {ds: [np.load(f"{C.ROOT}/tissue/{ds}/{c}.npz")["keep"] if a.tissue else np.ones(len(x), bool)
                 for c, x in zip(lab[ds].index, F[ds])] for ds in F}

    # fit sample: equal tiles per patient; remember patient (and cohort) of each row for bootstraps
    fit_ds = ["owl"] + (["tcga"] if a.fit == "both" else [])
    excl = set(pd.read_csv(a.exclude).case) if a.exclude else set()
    rows, pid, coh = [], [], []
    for ci, ds in enumerate(fit_ds):
        npp = a.per_patient
        if a.cohort_balance and ds == "tcga":
            npp = int(round(a.per_patient * len(F["owl"]) / len(F["tcga"])))
        for case, x, kp in zip(lab[ds].index, F[ds], keep[ds]):
            if case in excl:
                continue
            x = x[kp]
            k = min(npp, len(x)); sel = rng.choice(len(x), k, replace=False)
            rows.append(x[sel]); pid.append(np.full(k, len(pid))); coh.append(np.full(k, ci))
    S = torch.from_numpy(np.concatenate(rows).astype(np.float32)); pid = np.concatenate(pid); coh = np.concatenate(coh)
    mu = S.mean(0)
    U, Sv, V = torch.pca_lowrank((S - mu).to(dev), q=256, center=False, niter=4)
    P = V[:, :256]                                                       # [D,256]
    proj = lambda x: torch.nn.functional.normalize((x.to(dev).float() - mu.to(dev)) @ P, dim=1)
    Z = proj(S)
    evr = (Sv[:256] ** 2).sum().item() / ((S - mu) ** 2).sum().item()
    print(f"fit sample {tuple(S.shape)}  PCA-256 explained var {evr:.3f}", flush=True)
    if not a.boot_only:
        np.savez(f"{out}/pca.npz", mu=mu.numpy(), P=P.cpu().numpy(), evr=evr)

    probe = torch.from_numpy(rng.choice(len(Z), 50000, replace=False)).to(dev)
    from sklearn.metrics import adjusted_rand_score
    n_pat = int(pid.max()) + 1
    rows_of = np.split(np.arange(len(pid)), np.flatnonzero(np.diff(pid)) + 1)
    for K in map(int, a.Ks.split(",")):
        kd = f"{out}/K{K}"; os.makedirs(kd, exist_ok=True)
        # stability: full-pipeline patient bootstrap (patients drawn with multiplicity, PCA refit too),
        # ARI of hard labels on the fixed probe set
        labs = []
        for b in range(a.boots):
            bp = rng.choice(n_pat, n_pat, replace=True)
            ib = torch.from_numpy(np.concatenate([rows_of[p] for p in bp])).to(dev)
            Sb = S.to(dev)[ib] if S.numel() < 2e9 else S[ib.cpu()].to(dev)
            mub = Sb.mean(0); _, _, Vb = torch.pca_lowrank(Sb - mub, q=256, center=False, niter=4); del Sb
            Zb = torch.nn.functional.normalize((S.to(dev) - mub) @ Vb[:, :256], dim=1) if S.numel() < 2e9 else \
                torch.cat([torch.nn.functional.normalize((s.to(dev) - mub) @ Vb[:, :256], dim=1) for s in S.split(200000)])
            Cb, _ = spherical_kmeans(Zb[ib], K, seed=100 + b)
            labs.append((Zb[probe] @ Cb.T).argmax(1).cpu().numpy()); del Zb
        aris = [adjusted_rand_score(labs[i], labs[j]) for i, j in itertools.combinations(range(len(labs)), 2)]
        if a.boot_only:
            st = dict(K=K, ari_mean=float(np.mean(aris)), ari_min=float(np.min(aris)), ari_sd=float(np.std(aris)),
                      bootstrap="full-pipeline (multiplicity + PCA refit)")
            json.dump(st, open(f"{kd}/stability_full.json", "w"), indent=1); print(json.dumps(st), flush=True)
            continue
        # final dictionary on all RUMC
        Cf, obj = spherical_kmeans(Z, K, n_init=8, seed=K)
        maxcos_fit = (Z @ Cf.T).max(1).values
        thr = torch.quantile(maxcos_fit, 0.01).item()
        np.savez(f"{kd}/centroids.npz", C=Cf.cpu().numpy(), unassigned_thr=thr)
        st = dict(K=K, fit=a.fit, cohort_balance=a.cohort_balance, n_fit=len(Z),
                  ari_mean=float(np.mean(aris)), ari_min=float(np.min(aris)), ari_sd=float(np.std(aris)),
                  objective=obj, unassigned_thr=thr)
        json.dump(st, open(f"{kd}/stability.json", "w"), indent=1)
        print(json.dumps(st), flush=True)

        # assign every tile of every slide
        for ds in ("owl", "tcga"):
            os.makedirs(f"{kd}/assign/{ds}", exist_ok=True)
            fr = []
            for case, x, kp in zip(lab[ds].index, F[ds], keep[ds]):
                z = proj(torch.from_numpy(x)); sim = z @ Cf.T
                mc, hl = sim.max(1)
                hl = torch.where(mc < thr, torch.full_like(hl, K), hl)          # K = unassigned bucket
                hl, mc = hl.cpu().numpy().astype(np.int16), mc.cpu().numpy().astype(np.float16)
                hl[~kp] = K + 1                                                 # K+1 = not tissue (E0 filter)
                np.savez_compressed(f"{kd}/assign/{ds}/{case}.npz", label=hl, maxcos=mc)
                fr.append(np.r_[np.bincount(hl[kp], minlength=K + 1) / max(kp.sum(), 1), 1 - kp.mean()])
            pd.DataFrame(fr, index=lab[ds].index, columns=[f"p{k}" for k in range(K)] + ["unassigned", "nontissue"]) \
                .to_csv(f"{kd}/fractions_{ds}.csv")
        print(f"K={K} assigned all slides ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
