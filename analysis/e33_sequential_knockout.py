"""E33: sequential knockout ordering for the deployed three-encoder model (follow-up to E31).

E31 found that ranking tiles by single-tile knockout effect D_i (exact for one removal) is matched by the share a_i v_i
up to 2% removed and beaten beyond it, because attention renormalises as tiles are removed. Sequential knockout:
remove the tile whose removal lowers the ensemble score most, recompute every effect exactly on what is left, repeat.
Per model m (unnormalised): P_m = sum_rest a v, Q_m = sum_rest a, y_m = P_m / Q_m; removing i gives (P_m - a_i v_i)/(Q_m - a_i).
Orders compared on held-out slides (fold models that did not train on the slide, as E31): single D, share, sequential,
attention, random; drop of ensemble Z in SD of held-out Z. Caches per-slide (a, v) of the held-out models in
data/exp/e33/<ds>/<case>.npz. Writes logs/exp/e33_sequential_knockout.json. Usage: python e33_sequential_knockout.py [cuda:1]
"""
import json, os, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "training"), os.path.join(_HERE, "..", "container")]
import common as C
from inference import MIL

MD = f"{C.MODEL_DIR}/owl_v2"
ENCS = ("midnight", "uni2h", "phikon2")
FRAC = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
CACHE = f"{C.ROOT}/exp/e33"


def feats(enc, ds, c, keep):
    if enc == "midnight":
        return C._load(f"{C.ROOT}/feats/midnight/{ds}/{c}.npz", "x20")[keep]
    z = np.load(f"{C.ROOT}/feats/{enc}/{ds}/{c}.npz"); m = np.isin(z["idx"], keep)
    return z["x20"][m]


def sequential(A, V, w, frac_max=1.0):
    """A, V: [M, n] attention and standardised value per model; w: [M] weight of each model in the ensemble score
    (ensemble Z = sum_m w_m y_m + const). Greedy order of removal and the marginal drop at each step."""
    M, n = A.shape; P = (A * V).sum(1); Q = A.sum(1); alive = np.ones(n, bool); order, gain = [], []
    AV = A * V
    for _ in range(int(np.ceil(frac_max * n)) - 1):
        y = P / Q; y2 = (P[:, None] - AV) / np.clip(Q[:, None] - A, 1e-12, None)
        d = (w[:, None] * (y[:, None] - y2)).sum(0); d[~alive] = -np.inf; i = int(np.argmax(d))
        order.append(i); gain.append(float(d[i])); alive[i] = False; P -= AV[:, i]; Q -= A[:, i]
    rest = np.where(alive)[0]; order += rest.tolist(); gain += [0.0] * len(rest)
    return np.array(order), np.array(gain)


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cuda:1"
    from sklearn.model_selection import StratifiedKFold
    cal = json.load(open(f"{MD}/calib.json")); cfg = cal["cfg"]; nets = {}
    for e in ENCS:
        ce = cal["encoders"][e]; nets[e] = []
        for m in ce["models"]:
            n_ = MIL(ce["dim"], cfg["H"], cfg["A"], cfg["heads"], cfg["p"], cfg["pool"], cfg["topk"])
            n_.load_state_dict(torch.load(f"{MD}/{m['file']}", map_location="cpu")); nets[e].append((n_.eval().to(dev), m["mu"], m["sd"]))
    L = {ds: C.labels(ds) for ds in ("owl", "tcga")}
    cases = [(ds, c) for ds in L for c in L[ds].index]
    coh = np.array([ds for ds, _ in cases]); ev = np.concatenate([L[ds].event.values for ds in L])
    y = np.array([f"{c}{int(v)}" for c, v in zip(coh, ev)]); fold = np.zeros(len(cases), int)
    for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=2026).split(y, y)):
        fold[te] = k
    esd = np.array([cal["encoders"][e]["enc_sd"] for e in ENCS]); emu = np.array([cal["encoders"][e]["enc_mu"] for e in ENCS])
    w = 1.0 / (len(ENCS) * esd)

    def load(j):
        ds, c = cases[j]; p = f"{CACHE}/{ds}/{c}.npz"
        if os.path.exists(p):
            z = np.load(p); return z["A"], z["V"]
        sel = np.load(f"{C.ROOT}/feats/virchow2/{ds}/{c}.npz")["idx"][::2]
        return {e: feats(e, ds, c, sel) for e in ENCS}
    rng = np.random.default_rng(0); Z = np.zeros(len(cases)); curves = {o: [] for o in ("single", "share", "sequential", "attention", "random")}
    with ThreadPoolExecutor(4) as ex, torch.no_grad():
        futs = [ex.submit(load, j) for j in range(min(8, len(cases)))]
        for j in range(len(cases)):
            r = futs[j].result()
            if j + 8 < len(cases):
                futs.append(ex.submit(load, j + 8))
            if isinstance(r, dict):
                A, V = [], []
                for e in ENCS:
                    net, mu, sd = nets[e][fold[j]]; a, s = net.parts(torch.tensor(r[e].astype(np.float32), device=dev))
                    A.append(a.double().cpu().numpy()); V.append(((s - mu) / sd).double().cpu().numpy())
                A, V = np.stack(A), np.stack(V); ds, c = cases[j]; os.makedirs(f"{CACHE}/{ds}", exist_ok=True)
                np.savez(f"{CACHE}/{ds}/{c}.npz", A=A.astype(np.float32), V=V.astype(np.float32)); A, V = A.astype(float), V.astype(float)
            else:
                A, V = r[0].astype(float), r[1].astype(float)
            n = A.shape[1]; yv = (A * V).sum(1); Z[j] = float(((yv - emu) / esd).mean())

            def drop(order):
                out = []
                for f in FRAC:
                    S = order[:max(1, int(round(f * n)))]; P = yv - (A[:, S] * V[:, S]).sum(1); Q = 1 - A[:, S].sum(1)
                    out.append(float((w * (yv - P / Q)).sum()))
                return out
            single = (w[:, None] * A * (V - yv[:, None]) / np.clip(1 - A, 1e-12, None)).sum(0)
            seq, _ = sequential(A, V, w, frac_max=max(FRAC) + 0.01)
            curves["single"].append(drop(np.argsort(-single))); curves["share"].append(drop(np.argsort(-(w[:, None] * A * V).sum(0))))
            curves["sequential"].append(drop(seq)); curves["attention"].append(drop(np.argsort(-A.mean(0))))
            curves["random"].append(np.mean([drop(rng.permutation(n)) for _ in range(3)], 0).tolist())
            if j % 100 == 0:
                print(j, cases[j][1], {k: np.round(v[-1][:4], 3).tolist() for k, v in curves.items()}, flush=True)
    sdZ = {ds: float(Z[coh == ds].std()) for ds in L} | {"all": float(Z.std())}
    res = dict(fractions=FRAC, sd_Z=sdZ, deletion={})
    for o, v in curves.items():
        v = np.array(v); res["deletion"][o] = {}
        for grp, m in (("all", np.ones(len(cases), bool)), ("owl", coh == "owl"), ("tcga", coh == "tcga")):
            q = v[m] / sdZ[grp]; res["deletion"][o][grp] = dict(mean=q.mean(0).tolist(), sem=(q.std(0) / np.sqrt(m.sum())).tolist())
        print(o, np.round(v.mean(0) / sdZ["all"], 3).tolist(), flush=True)
    json.dump(res, open(f"{C.ROOT}/../logs/exp/e33_sequential_knockout.json", "w"), indent=1)


if __name__ == "__main__":
    main()
