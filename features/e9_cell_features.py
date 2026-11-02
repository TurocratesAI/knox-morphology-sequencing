"""E9: per-tile cell-arrangement features from the whole-slide nucleus maps (E7).

For every 20x tile (112 um; same index as the feature/assign files) from the nuclei whose centroid lies in it,
with neighbours taken from the whole slide (no tile-edge truncation). Non-spatial ("what cells"):
  n_dens     nuclei per mm^2 of masked tissue
  area_mu, area_cv, elong_mu (major/minor), hema_mu, hema_sd_mu, eos_ring_mu, small_dark (lymphocyte-like share)
Spatial ("how arranged"):
  nn_mu      mean nearest-neighbour distance (um)
  nn_cv      CV of nearest-neighbour distance (spacing regularity)
  clark      Clark-Evans ratio (1 = random, <1 clustered, >1 regular)
  align      mean cos(2*dtheta) between each elongated nucleus and its 6 nearest neighbours (orientation coherence)
  moran_area within-tile Moran's I of log-area on the 6-NN graph (size-position coupling)
  moran_hema same for hematoxylin OD (chromatin-position coupling)
  deg20      mean number of nuclei within 20 um (local crowding graph degree)
Null (same file, suffix _null): moran_*/align recomputed after shuffling the marks among nuclei of the same tile,
which keeps every tile's cell inventory and positions but destroys the pairing of marks with positions.
Writes data/cellfeat/<ds>/<case>.npz: X float32 [n20, F], names, n_nuc per tile.
"""
import argparse, glob, json, os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
import common as C

NAMES = ["n_dens", "area_mu", "area_cv", "elong_mu", "hema_mu", "hema_sd_mu", "eos_ring_mu", "small_dark",
         "nn_mu", "nn_cv", "clark", "align", "moran_area", "moran_hema", "deg20",
         "align_null", "moran_area_null", "moran_hema_null"]
K = 6
MIN_NUC = 8


def tile_index(T, xy0):
    s = T["meta"]["tile20_l0"]; g = T["coords20"] // s
    G = -np.ones(g.max(0) + 2, np.int64); G[g[:, 0], g[:, 1]] = np.arange(len(g))
    q = xy0 // s; ok = (q[:, 0] < G.shape[0]) & (q[:, 1] < G.shape[1])
    ti = -np.ones(len(xy0), np.int64); ti[ok] = G[q[ok, 0], q[ok, 1]]
    return ti


def moran(v, vn, ti, n_t):
    """within-tile Moran's I of v on the kNN graph (neighbours may lie outside the tile; centred on the tile mean)"""
    cnt = np.bincount(ti, minlength=n_t).astype(float)
    mu = np.bincount(ti, v, minlength=n_t) / np.maximum(cnt, 1)
    z = v - mu[ti]; zn = vn.mean(1) - mu[ti]           # vn: neighbour values [n, K]
    return np.bincount(ti, z * zn, minlength=n_t) / np.maximum(np.bincount(ti, z * z, minlength=n_t), 1e-9)


def one(job):
    ds, case = job; out = f"{C.ROOT}/cellfeat/{ds}/{case}.npz"
    if os.path.exists(out):
        return case, "skip"
    z = np.load(f"{C.ROOT}/nuclei/{ds}/{case}.npz"); meta = json.loads(str(z["meta"])); mpp0 = meta["mpp0"]
    T = C.tiles(ds, case); n_t = len(T["coords20"])
    xy = z["xy"].astype(np.float64) * mpp0; area = z["area"].astype(float); el = (z["major"] / np.maximum(z["minor"], 1e-3)).astype(float)
    th = z["angle"].astype(float); hema = z["hema"].astype(float)
    ok = np.isfinite(area) & (area > 4) & (area < 400); xy, area, el, th, hema = xy[ok], area[ok], el[ok], th[ok], hema[ok]
    hsd, eos = z["hema_sd"].astype(float)[ok], z["eos_ring"].astype(float)[ok]
    tree = cKDTree(xy); d, nb = tree.query(xy, K + 1); d, nb = d[:, 1:], nb[:, 1:]
    deg = tree.query_ball_point(xy, 20, return_length=True) - 1
    ti_all = tile_index(T, (xy / mpp0).astype(np.int64)); m = ti_all >= 0
    ti = ti_all[m]; idx = np.where(m)[0]
    cnt = np.bincount(ti, minlength=n_t).astype(float); den = np.maximum(cnt, 1)
    mean = lambda v: np.bincount(ti, v[idx], minlength=n_t) / den
    la = np.log(area); hq = np.quantile(hema, 0.75)
    small_dark = ((area < 30) & (el < 1.4) & (hema > hq)).astype(float)
    tissue_mm2 = (T["meta"]["tile20_l0"] * mpp0 / 1000) ** 2 * np.maximum(T["mask_frac20"], 0.05)
    nn = d[:, 0]; nn_mu = mean(nn); nn_sd = np.sqrt(np.maximum(mean(nn ** 2) - nn_mu ** 2, 0))
    dens_um2 = cnt / (tissue_mm2 * 1e6)
    area_mu = mean(area); area_sd = np.sqrt(np.maximum(mean(area ** 2) - area_mu ** 2, 0))
    rng = np.random.default_rng(0)
    perm = idx.copy()                                  # shuffle marks within tile: permute idx inside each tile group
    order = np.argsort(ti, kind="stable"); starts = np.r_[0, np.cumsum(np.bincount(ti, minlength=n_t))]
    for a_, b_ in zip(starts[:-1], starts[1:]):
        if b_ - a_ > 1:
            perm[order[a_:b_]] = rng.permutation(idx[order[a_:b_]])
    def marks_shuffled(v):
        w = v.copy(); w[idx] = v[perm]; return w
    nbm = nb[idx]
    X = np.c_[cnt / (tissue_mm2), area_mu, area_sd / np.maximum(area_mu, 1e-6), mean(el), mean(hema), mean(hsd), mean(eos),
              mean(small_dark), nn_mu, nn_sd / np.maximum(nn_mu, 1e-6), nn_mu / np.maximum(0.5 / np.sqrt(np.maximum(dens_um2, 1e-9)), 1e-6),
              _align(th, el, nbm, idx, ti, n_t),
              moran(la[idx], la[nbm], ti, n_t), moran(hema[idx], hema[nbm], ti, n_t), mean(deg.astype(float)),
              _align(marks_shuffled(th), marks_shuffled(el), nbm, idx, ti, n_t),
              moran(la[idx], marks_shuffled(la)[nbm], ti, n_t), moran(hema[idx], marks_shuffled(hema)[nbm], ti, n_t)]
    X[cnt < MIN_NUC] = np.nan
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, X=X.astype(np.float32), names=np.array(NAMES), n_nuc=cnt.astype(np.int32))
    return case, int(m.sum())


def _align(th, el, nbm, idx, ti, n_t):
    ok = el > 1.3
    c = np.cos(2 * (th[idx][:, None] - th[nbm])); w = ok[nbm] & ok[idx][:, None]
    a = (c * w).sum(1) / np.maximum(w.sum(1), 1); has = w.sum(1) > 0
    return np.bincount(ti[has], a[has], minlength=n_t) / np.maximum(np.bincount(ti[has], minlength=n_t), 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    jobs = [(ds, os.path.basename(f)[:-4]) for ds in ("owl", "tcga") for f in sorted(glob.glob(f"{C.ROOT}/nuclei/{ds}/*.npz")) if not f.endswith(".part.npz")]
    with ProcessPoolExecutor(a.workers) as ex:
        for case, r in ex.map(one, jobs):
            print(case, r, flush=True)


if __name__ == "__main__":
    main()
