"""E27: cross-encoder agreement per tile, an adversarial check on what the model relies on.

Idea (sequencing analogy: a variant seen on only one platform is suspect). For each slide, on the tiles shared by all
encoders (E19/E26 idx), each encoder defines the tile's k nearest neighbours within the slide (cosine on per-dimension
standardised features). Agreement of a tile = mean Jaccard overlap of its neighbour sets over encoder pairs. Low
agreement = encoders disagree about what the tile resembles (candidate artefact, stain or encoder-specific shortcut,
or unusual morphology; pathologists decide which).
Joined with E24's cache (Midnight v1 OOF knockout effect D_i, compartment, nuclear reads) to ask: are the tiles the
model relies on (top 1% D) ones the encoders agree on? Writes logs/exp/e27_agreement.json, data/exp/e27/<ds>/<case>.npz
Usage: python e27_encoder_agreement.py --encs midnight,virchow2,hoptimus0 --n 120
"""
import argparse, itertools, json, os, sys
import numpy as np, torch
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "training")]
import common as C

E24 = f"{C.ROOT}/exp/e24"


FULL = ("midnight", "hoptimus0")                          # files hold every coords20 row
E26 = ("uni2h", "gigapath", "phikon2")                     # files hold their own idx (every second Virchow2 tile)


def load(enc, ds, c, idx):
    if enc in FULL:
        return C._load(f"{C.ROOT}/feats/{enc}/{ds}/{c}.npz", "x20")[idx]
    z = np.load(f"{C.ROOT}/feats/{enc}/{ds}/{c}.npz"); pos = {t: i for i, t in enumerate(z["idx"])}
    return z["x20"][[pos[t] for t in idx]]


def base_idx(encs, ds, c):
    idx = np.load(f"{C.ROOT}/feats/virchow2/{ds}/{c}.npz")["idx"].astype(np.int64)
    if any(e in E26 for e in encs):
        idx = idx[::2]
        for e in encs:
            if e in E26:
                idx = idx[np.isin(idx, np.load(f"{C.ROOT}/feats/{e}/{ds}/{c}.npz")["idx"])]
    return idx


def knn(x, k, dev):
    x = torch.tensor(x, dtype=torch.float32, device=dev); x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    x = torch.nn.functional.normalize(x, dim=1); s = x @ x.T; s.fill_diagonal_(-2)
    return s.topk(k, dim=1).indices.cpu().numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--encs", default="midnight,virchow2,hoptimus0")
    ap.add_argument("--n", type=int, default=120); ap.add_argument("--k", type=int, default=10); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--out", default="")
    a = ap.parse_args(); encs = a.encs.split(","); rng = np.random.default_rng(0)
    cases = [(ds, c) for ds in ("owl", "tcga") for c in C.labels(ds).index
             if os.path.exists(f"{E24}/{ds}/{c}.npz") and all(os.path.exists(f"{C.ROOT}/feats/{e}/{ds}/{c}.npz") for e in encs if e not in FULL)]
    cases = [cases[i] for i in sorted(rng.choice(len(cases), min(a.n, len(cases)), replace=False))]
    rows = []
    for j, (ds, c) in enumerate(cases):
        idx = base_idx(encs, ds, c)
        E = np.load(f"{E24}/{ds}/{c}.npz"); pos = {t: i for i, t in enumerate(E["idx"])}
        keep = np.array([t in pos for t in idx]); idx = idx[keep]; p = np.array([pos[t] for t in idx])
        N = {e: knn(load(e, ds, c, idx), a.k, a.device) for e in encs}
        agr = np.zeros(len(idx))
        for e1, e2 in itertools.combinations(encs, 2):
            agr += np.array([len(set(u) & set(v)) / len(set(u) | set(v)) for u, v in zip(N[e1], N[e2])])
        agr /= len(list(itertools.combinations(encs, 2)))
        v, at, g = E["v"][p], E["a"][p], E["g"][p]; y = float((E["a"] * E["v"]).sum())
        D = at * (v - y) / np.clip(1 - at, 1e-6, None); dens = E["F"][p, 0]
        top = np.zeros(len(D), bool); top[np.argsort(-D)[:max(1, int(round(0.01 * len(D))))]] = True
        low = agr <= np.quantile(agr, 0.1)
        os.makedirs(f"{C.ROOT}/exp/e27{a.out}/{ds}", exist_ok=True)
        np.savez(f"{C.ROOT}/exp/e27{a.out}/{ds}/{c}.npz", idx=idx, agr=agr.astype(np.float32), D=D.astype(np.float32), g=g)
        rows.append(dict(ds=ds, case=c, n=len(idx), agr_mean=float(agr.mean()), agr_top=float(agr[top].mean()),
                         rho_D=float(np.corrcoef(np.argsort(np.argsort(agr)), np.argsort(np.argsort(D)))[0, 1]),
                         top_in_low=float(low[top].mean()), dens_low=float(np.nanmedian(dens[low])), dens_rest=float(np.nanmedian(dens[~low])),
                         **{f"agr_{q}": float(agr[g == q].mean()) if (g == q).any() else np.nan for q in range(4)}))
        if j % 20 == 0:
            print(j, c, round(rows[-1]["agr_mean"], 3), flush=True)
    import pandas as pd
    R = pd.DataFrame(rows); out = {"encs": encs, "k": a.k, "n_slides": len(R)}
    for ds in ("owl", "tcga"):
        d = R[R.ds == ds]
        out[ds] = dict(agreement_mean=round(d.agr_mean.mean(), 3), agreement_top1pct_knockout=round(d.agr_top.mean(), 3),
                       spearman_agreement_vs_knockout=round(d.rho_D.mean(), 3), share_top1pct_in_lowest_decile=round(d.top_in_low.mean(), 3),
                       nuclear_density_low_vs_rest=[round(d.dens_low.median(), 1), round(d.dens_rest.median(), 1)],
                       agreement_by_compartment={n: round(d[f"agr_{q}"].mean(), 3) for q, n in enumerate(["tumour", "gland", "stroma", "unassigned"])})
    json.dump(out, open(f"{C.LOGS}/exp/e27_agreement{a.out}.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
