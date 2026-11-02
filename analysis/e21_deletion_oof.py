"""E21: deletion-faithfulness curves for the deployed v1 model, out of fold (paper Fig. 1).

v1 = 5 Midnight ABMIL fold models (v1 training script, not included; folds StratifiedKFold(5, seed 2026) over cohort x event).
Each slide is scored only by the fold model that did not train on it. For one model, slide score y = sum_i a_i v_i
with v_i = (w . h_i + b - mu) / sd; the exact single-tile deletion effect is D_i = a_i (v_i - y) / (1 - a_i).
Tiles are removed in three orders: by D (our map, largest first), by attention a (largest first), and at random
(3 permutations); after each removal fraction the bag is re-scored (attention renormalised), giving y - y_{-S}.
Writes logs/exp/e21_deletion_oof.json (mean and 95% CI over slides, per cohort).
"""
import json, os, sys
import numpy as np, torch
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "training"), os.path.join(_HERE, "..", "container")]
import common as C
from inference import ABMIL

MD = f"{C.MODEL_DIR}/owl"
FRAC = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    from sklearn.model_selection import StratifiedKFold
    cal = json.load(open(f"{MD}/calib.json")); nets = []
    for m in cal["models"]:
        n = ABMIL(cal["dim"]); n.load_state_dict(torch.load(f"{MD}/{m['file']}", map_location="cpu")); nets.append((n.eval().to(dev), m["mu"], m["sd"]))
    L = {ds: C.labels(ds) for ds in ("owl", "tcga")}
    cases = [(ds, c) for ds in L for c in L[ds].index]
    coh = np.array([ds for ds, _ in cases]); e = np.concatenate([L[ds].event.values for ds in L])
    y = np.array([f"{c}{int(v)}" for c, v in zip(coh, e)]); fold = np.zeros(len(cases), int)
    for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=2026).split(y, y)):
        fold[te] = k
    rng = np.random.default_rng(0); out = {o: [] for o in ("map", "attention", "random")}
    with torch.no_grad():
        for j, (ds, c) in enumerate(cases):
            net, mu, sd = nets[fold[j]]
            X = torch.tensor(C.feats("midnight", ds, [c], tissue=True)[0].astype(np.float32), device=dev)
            a, s = net.parts(X); v = (s - mu) / sd; z = float((a * v).sum())
            D = (a * (v - z) / (1 - a).clamp(min=1e-6)).cpu().numpy(); an = a.cpu().numpy(); n = len(D)
            def drop(order):
                r = []
                for f in FRAC:
                    keep = np.ones(n, bool); keep[order[:max(1, int(round(f * n)))]] = False
                    kt = torch.tensor(keep, device=dev); a2, s2 = net.parts(X[kt]); r.append(z - float((a2 * (s2 - mu) / sd).sum()))
                return r
            out["map"].append(drop(np.argsort(-D))); out["attention"].append(drop(np.argsort(-an)))
            out["random"].append(np.mean([drop(rng.permutation(n)) for _ in range(3)], 0).tolist())
            if j % 100 == 0:
                print(j, c, flush=True)
    res = {"fractions": FRAC}
    for o, v in out.items():
        v = np.array(v); res[o] = {}
        for name, m in (("all", np.ones(len(cases), bool)), ("owl", coh == "owl"), ("tcga", coh == "tcga")):
            res[o][name] = dict(mean=v[m].mean(0).tolist(), lo=np.percentile(v[m], 2.5, 0).tolist(), hi=np.percentile(v[m], 97.5, 0).tolist(),
                                sem=(v[m].std(0) / np.sqrt(m.sum())).tolist())
        print(o, np.round(v.mean(0), 3).tolist(), flush=True)
    json.dump(res, open(f"{C.LOGS}/exp/e21_deletion_oof.json", "w"), indent=1)


if __name__ == "__main__":
    main()
