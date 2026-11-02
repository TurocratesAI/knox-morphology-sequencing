"""E32: end-to-end mask-edit check of the challenge container (verify deletion through the real container).

Step 1 (make): from a full run of the container on one case (debug.npz with per-tile attention/value of every fold
model, plus the written heatmap), build edited cases whose tissue mask has the top f of tissue pixels by heat removed
(as the organisers' progressive deletion does), WSI hard-linked. Also predicts the score after each edit exactly from
the full run, by dropping the analysed tiles the edited mask no longer covers (same tissue rule, >= 25% inside).
Step 2 (compare): reads the container's outputs on the edited cases and prints observed vs predicted drops.
Usage: python e32_mask_edit_e2e.py make <full_output_case_dir> <input_case_dir> <out_root> <tag>
       python e32_mask_edit_e2e.py compare <out_root>
"""
import json, os, sys
import numpy as np

FRACS = (0.05, 0.2)
ENCS = ("midnight", "uni2h", "phikon2")
CAL = os.path.join(os.environ.get("KNOX_MODEL_DIR", os.path.join(os.environ.get("KNOX_ROOT", "."), "model")),
                   "owl_v2", "calib.json")


def read_lvl0(path):
    import openslide
    s = openslide.OpenSlide(path); a = np.asarray(s.read_region((0, 0), 0, s.level_dimensions[0]))[..., 0]; s.close(); return a


def write_mask(path, m, spacing_um):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "container"))
    from inference import write_heatmap
    from pathlib import Path
    write_heatmap(Path(path), m.astype(np.uint8), spacing_um)


def predict(dbg, keep_tile):
    """Exact ensemble z after dropping analysed tiles with keep_tile False (container's score_v2 on the rest)."""
    cal = json.load(open(CAL)); zs = []
    for e in ENCS:
        ce = cal["encoders"][e]; ys = []
        for k in range(len(ce["models"])):
            a, v = dbg[f"{e}_{k}_a"].astype(float), dbg[f"{e}_{k}_v"].astype(float); a2 = a[keep_tile]
            ys.append((a2 * v[keep_tile]).sum() / a2.sum())
        zs.append((np.mean(ys) - ce["enc_mu"]) / ce["enc_sd"])
    return float(np.mean(zs))


def make(full, inp, root, tag):
    import openslide
    dbg = np.load(f"{full}/debug.npz"); mpath = [f"{inp}/images/prostatectomy-tissue-mask/{f}" for f in os.listdir(f"{inp}/images/prostatectomy-tissue-mask")][0]
    wsi = [f"{inp}/images/prostatectomy-wsi/{f}" for f in os.listdir(f"{inp}/images/prostatectomy-wsi")][0]
    mask = read_lvl0(mpath) > 0; heat = read_lvl0(f"{full}/images/prostatectomy-overall-survival-attribution-heatmap/attribution.tif")
    assert heat.shape == mask.shape, (heat.shape, mask.shape)
    W = openslide.OpenSlide(wsi).level_dimensions[0][0]; ms = W / mask.shape[1]
    sp = float(openslide.OpenSlide(wsi).properties.get("openslide.mpp-x", 0.2425)) * ms
    s20 = int(round(224 * 0.5 / float(openslide.OpenSlide(wsi).properties.get("openslide.mpp-x", 0.2425))))
    ys, xs = np.nonzero(mask); h = heat[ys, xs].astype(int); order = np.lexsort((np.arange(len(h)), -h))
    pred = {"full": float(dbg["z"])}; xy = dbg["xy"]
    for f in FRACS:
        m2 = mask.copy(); r = order[:int(round(f * len(order)))]; m2[ys[r], xs[r]] = False
        name = f"{tag}_del{int(f * 100):02d}"; d = f"{root}/{name}"
        os.makedirs(f"{d}/images/prostatectomy-tissue-mask", exist_ok=True); os.makedirs(f"{d}/images/prostatectomy-wsi", exist_ok=True)
        write_mask(f"{d}/images/prostatectomy-tissue-mask/mask.tif", m2, sp)
        dst = f"{d}/images/prostatectomy-wsi/{os.path.basename(wsi)}"
        if not os.path.exists(dst):
            os.link(wsi, dst)
        os.system(f"cp {inp}/inputs.json {d}/inputs.json")
        # tissue rule on the edited mask for each analysed tile (fraction of its footprint inside the mask >= 0.25)
        x0, y0 = (xy[:, 0] / ms).astype(int), (xy[:, 1] / ms).astype(int); x1, y1 = ((xy[:, 0] + s20) / ms).astype(int), ((xy[:, 1] + s20) / ms).astype(int)
        frac = np.array([m2[b:max(dd, b + 1), a:max(c, a + 1)].mean() for a, b, c, dd in zip(x0, y0, x1, y1)])
        pred[name] = dict(z_pred=predict(dbg, frac >= 0.25), removed_tiles=int((frac < 0.25).sum()), n_tiles=len(xy))
        print(name, pred[name], flush=True)
    json.dump(pred, open(f"{root}/{tag}_pred.json", "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make(*sys.argv[2:6])
