"""E28: multi-encoder ensemble with a rule fixed before the encoder results were seen (written 2026-10-28, while
j20-j24 were still running).

Pool: every permitted encoder the team has access to, all under one protocol (joint-cohort Cox MIL, raw features,
the same ~2,000 tiles per slide = every second Virchow2 tile):
  midnight j23, hoptimus0 j24, uni2h j20, gigapath j21, phikon2 j22.
  Virchow2 is left out: the team has no Hugging Face access, and the FAQ allows gated encoders only with own access.
Rule:
  1. Score per encoder = OOF risk averaged over the 2 fold repeats; C within each cohort; summary = mean of the two.
  2. Include every encoder whose summary C is within 0.02 of the best single encoder.
  3. Ensemble = equal-weight mean of each included encoder's risk, z-scored over all 806 OOF predictions
     (at test time the same mean/SD from training predictions are used, so this is deployable).
  4. Confirm on fresh folds: rerun the included encoders with --fold_seed 300 --tag _f300 and run this script with
     --tag _f300; the inclusion set is read from the first run, never re-selected.
Reports ensemble and single-encoder C, and paired patient-bootstrap CIs of (ensemble - best single) and
(ensemble - Midnight full tiles j00) within each cohort. Writes logs/exp/e28_ensemble{tag}.json
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import common as C

RJ = f"{C.ROOT}/exp/runner_joint"
POOL = {"midnight": "j23", "hoptimus0": "j24", "uni2h": "j20", "gigapath": "j21", "phikon2": "j22"}
MARGIN = 0.02
LOG = C.LOGS + "/exp/e28_ensemble{}.json"


def boot_diff(e, t, a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(e), len(e)); d.append(C.cindex(e[i] > 0, t[i], a[i]) - C.cindex(e[i] > 0, t[i], b[i]))
    return [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default=""); a = ap.parse_args()
    L = {ds: C.labels(ds) for ds in ("owl", "tcga")}
    R, cases, g = {}, None, None
    for name, jid in list(POOL.items()) + [("midnight_full_j00", "j00")]:
        f = f"{RJ}/{jid}{a.tag if jid != 'j00' else ''}.npz"
        if os.path.exists(f):
            z = np.load(f, allow_pickle=True); R[name] = z["oof"].mean(0)
            assert cases is None or (z["cases"] == cases).all(); cases, g = z["cases"], z["cohort"]
    lab = [L["owl" if k == 0 else "tcga"].loc[c] for c, k in zip(cases, g)]
    t = np.array([x.time for x in lab], float); e = np.array([x.event for x in lab], float)
    cidx = lambda r: [round(float(C.cindex(e[g == k] > 0, t[g == k], r[g == k])), 3) for k in (0, 1)]
    single = {n: cidx(r) for n, r in R.items()}
    pool = {n: c for n, c in single.items() if n in POOL}
    if a.tag:                                                  # confirmation: same set as the first run
        inc = json.load(open(LOG.format("")))["included"]
    else:
        best = max(np.mean(c) for c in pool.values()); inc = [n for n, c in pool.items() if np.mean(c) >= best - MARGIN]
    z = lambda r: (r - r.mean()) / r.std()
    ens = np.mean([z(R[n]) for n in inc], 0); allfive = np.mean([z(R[n]) for n in pool], 0)
    best_name = max(pool, key=lambda n: np.mean(pool[n]))
    out = dict(tag=a.tag, single=single, included=inc, best_single=best_name, ensemble=cidx(ens), all_pool_equal=cidx(allfive))
    for ref in (best_name, "midnight_full_j00"):
        if ref in R:
            out[f"ens_minus_{ref}_ci"] = {ds: boot_diff(e[g == k], t[g == k], ens[g == k], R[ref][g == k]) for k, ds in enumerate(("rumc", "tcga"))}
    np.savez(f"{RJ}/e28_ensemble{a.tag}.npz", ens=ens, cases=cases, cohort=g, included=inc)
    json.dump(out, open(LOG.format(a.tag), "w"), indent=1); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
