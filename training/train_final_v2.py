"""Submission v2: three-encoder ensemble (Midnight, UNI2-h, Phikon-v2), chosen by the pre-set rule in
training/e28_ensemble.py (OOF 0.728/0.731; fresh folds 0.710/0.723 vs Midnight 0.705/0.706).

Tiles: E26 set = every second tile of the Virchow2 set (<= 4,000 stride-subsampled E0 tissue tiles), ~2,000 per slide.
Per encoder: 5 fold models (folds stratified by cohort x event, seed 2026, as v1), runner_joint recipe (joint pooled
Cox, gated-attention MIL, BASE config). Each fold model's risk is z-scored with its own training-set mean/SD; the
encoder score is the mean over its 5 fold models; encoder scores are z-scored with their OOF mean/SD and averaged
with equal weights. Held-out predictions give honest within-cohort OOF C for each encoder and for the ensemble.
Writes $KNOX_MODEL_DIR/owl_v2/{<enc>_f<k>.pt, calib.json}. v1 ($KNOX_MODEL_DIR/owl) is left untouched.
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import runner_joint as RJ
from runner import BASE

OUT = os.path.join(C.MODEL_DIR, "owl_v2")
ENCS = {"midnight": "midnight@e26", "uni2h": "uni2h", "phikon2": "phikon2"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    from sklearn.model_selection import StratifiedKFold
    L = {ds: C.labels(ds) for ds in ("owl", "tcga")}
    t = np.concatenate([L[ds].time.values for ds in L]).astype(float); e = np.concatenate([L[ds].event.values for ds in L]).astype(float)
    g = np.concatenate([[k] * len(L[ds]) for k, ds in enumerate(L)]); coh = np.array(["owl", "tcga"])[g]
    y = np.array([f"{c}{int(v)}" for c, v in zip(coh, e)]); folds = list(StratifiedKFold(5, shuffle=True, random_state=2026).split(y, y))
    cfg = {**BASE, "loss": "cox"}; calib = dict(version="v2-3enc-fold5", cfg=cfg, encoders={})
    cidx = lambda r: {ds: round(float(C.cindex(e[g == k] > 0, t[g == k], r[g == k])), 4) for k, ds in enumerate(("owl", "tcga"))}
    S = {}
    for enc, src in ENCS.items():
        X = sum((RJ.load_enc(src, ds, list(L[ds].index)) for ds in L), []); oof = np.zeros(len(t)); models = []
        for k, (tr, te) in enumerate(folds):
            m = RJ.train(cfg, [X[i] for i in tr], t[tr], e[tr], g[tr], a.device, seed=k)
            rtr = RJ.predict(m, [X[i] for i in tr], a.device); mu, sd = float(rtr.mean()), float(rtr.std())
            oof[te] = (RJ.predict(m, [X[i] for i in te], a.device) - mu) / sd
            torch.save(m.state_dict(), f"{OUT}/{enc}_f{k}.pt"); models.append(dict(file=f"{enc}_f{k}.pt", mu=mu, sd=sd))
        S[enc] = oof; calib["encoders"][enc] = dict(dim=int(X[0].shape[1]), models=models, enc_mu=float(oof.mean()), enc_sd=float(oof.std()), oof=cidx(oof))
        print(enc, calib["encoders"][enc]["oof"], flush=True); del X
    ens = np.mean([(S[n] - S[n].mean()) / S[n].std() for n in ENCS], 0)
    calib.update(oof=cidx(ens), q50=float(np.quantile(ens, 0.5)), q80=float(np.quantile(ens, 0.8)), z_sd=float(ens.std()),
                 median_followup_y=float(np.median(t)),
                 tiles="E0 tissue tiles; if more than 4000, idx[arange(0, n, ceil(n/4000))][:4000]; then every second tile")
    print("ensemble", calib["oof"], flush=True)
    np.savez(f"{OUT}/oof.npz", **S, ens=ens, cohort=g)
    json.dump(calib, open(f"{OUT}/calib.json", "w"), indent=1)


if __name__ == "__main__":
    main()
