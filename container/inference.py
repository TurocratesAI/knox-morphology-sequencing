"""OWL submission v2: three frozen encoders (Midnight-12K, UNI2-h, Phikon-v2) -> gated-attention MIL fold models
(joint pooled Cox) -> equal-weight ensemble, with an exact per-tile deletion effect used for the attribution map and
the report. An earlier v1 (Midnight only, all tissue tiles) is not included.

v2 changes to the pipeline below: after the Midnight pass and E0 tissue filter, the analysed set is the training
tile rule (at most 4,000 tissue tiles by uniform stride, then every second one, ~2,000); UNI2-h and Phikon-v2 embed
only those (fp16 weights and ImageNet normalisation, as for training). Per encoder e and fold model k:
z_ek = (sum_i a_i s_i - mu_ek) / sd_ek; z_e = (mean_k z_ek - mu_e) / sd_e; z = mean_e z_e. The deletion effect of
tile i is the same average of each model's exact a_i (v_i - z_ek) / (1 - a_i).

Pipeline per case (mirrors features/owl_extract.py, features/e0_tissue_filter.py, training/e1_baselines.py):
  organiser mask -> 112 um tile grid (>= 25% mask) -> fixed lattice subsample (~16k; mask edits only remove tiles)
  -> 224 px @ 0.5 um/px
  -> Midnight [CLS | mean patch] (3072) -> E0 tissue predictor (Midnight ridge, >= 0.25)
  -> 5 ABMIL models: risk_m = sum_i a_i (w . h_i) + b ; z_m = (risk_m - mu_m) / sd_m ; z = mean_m z_m
  -> tile deletion effect D_i = mean_m a_i (v_i - z_m) / (1 - a_i), v_i = (w . h_i + b - mu_m) / sd_m: the exact
     change in the score when tile i is removed (softmax renormalised), as OWL's perturbation test measures.
Outputs: overall-survival-years.json (monotone in -z), prostate-cancer-risk-classification.json,
the attribution TIFF on the mask grid (uint8, 128 = neutral, >128 pushes towards recurrence) and a 2-page PDF.
Env (local sandbox only): OWL_GPU_MEM_GB caps GPU memory to emulate a T4.
"""
import glob, json, os, queue, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
INPUT_PATH, OUTPUT_PATH = Path("/input"), Path("/output")
MODEL_DIR = Path(os.environ.get("OWL_MODEL_DIR", "/opt/ml/model"))
TILE_PX, MPP20, MAX_TILES, GRID_THR, NMAX = 224, 0.5, 16000, 0.25, 4000
SELECT = os.environ.get("OWL_SELECT", "lattice")         # "list" = training rule; lattice is mask-stable (e31)
MN_MEAN, MN_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
IN_MEAN, IN_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)        # UNI2-h, Phikon-v2 (as features/e26)
ENCS = ("midnight", "uni2h", "phikon2")
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


class ABMIL(nn.Module):                                   # identical to training/e1_baselines.ABMIL
    def __init__(self, D, H=256, p=0.25):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(D, H), nn.GELU(), nn.Dropout(p))
        self.a, self.b, self.w = nn.Linear(H, 128), nn.Linear(H, 128), nn.Linear(128, 1)
        self.head = nn.Linear(H, 1)

    def parts(self, x):
        h = self.enc(x)
        att = torch.softmax(self.w(torch.tanh(self.a(h)) * torch.sigmoid(self.b(h))).squeeze(-1), 0)
        s = self.head(h).squeeze(-1)                     # w . h_i + b ; risk = sum_i att_i s_i (att sums to 1)
        return att, s


# ------------------------------------------------------------------ model
def init_model():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(0); log(f"GPU {p.name} {p.total_memory / 2**30:.1f} GB")
        cap = float(os.environ.get("OWL_GPU_MEM_GB", 0))
        if cap:
            torch.cuda.set_per_process_memory_fraction(min(1.0, cap * 2**30 / p.total_memory)); log(f"GPU memory capped at {cap} GB")
    from transformers import AutoModel
    import timm
    from safetensors.torch import load_file
    mn = AutoModel.from_pretrained(str(MODEL_DIR / "midnight"), local_files_only=True).eval().to(dev)
    kw = json.load(open(MODEL_DIR / "uni2h/timm_kwargs.json"))
    uni = timm.create_model("vit_huge_patch14_224", pretrained=False, mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU, **kw)
    uni.load_state_dict(load_file(str(MODEL_DIR / "uni2h/model.safetensors"))); uni = uni.eval().half().to(dev)
    phk = AutoModel.from_pretrained(str(MODEL_DIR / "phikon2"), local_files_only=True, dtype=torch.float16).eval().to(dev)
    cal = json.load(open(MODEL_DIR / "owl_v2/calib.json")); cfg = cal["cfg"]; mils = {}
    for e in ENCS:
        ce = cal["encoders"][e]; mils[e] = []
        for m in ce["models"]:
            net = MIL(ce["dim"], cfg["H"], cfg["A"], cfg["heads"], cfg["p"], cfg["pool"], cfg["topk"])
            net.load_state_dict(torch.load(MODEL_DIR / "owl_v2" / m["file"], map_location="cpu")); mils[e].append((net.eval().to(dev), m["mu"], m["sd"]))
    t = lambda v: torch.tensor(v, device=dev).view(1, 3, 1, 1)
    log("model loaded")
    return dict(dev=dev, mn=mn, uni=uni, phk=phk, mils=mils, cal=cal, mean=t(MN_MEAN), std=t(MN_STD), imean=t(IN_MEAN), istd=t(IN_STD),
                tcoef=torch.tensor(cal["tissue"]["coef"], dtype=torch.float32, device=dev))


# ------------------------------------------------------------------ slide geometry (as owl_extract.py)
def tiff_mpp(path):
    import tifffile
    with tifffile.TiffFile(path) as t:
        p = t.pages[0]; xr = p.tags["XResolution"].value; unit = int(p.tags["ResolutionUnit"].value)
        ppu = xr[0] / xr[1]; per = {3: 1e4, 2: 25400.0}.get(unit)
        if per is None or ppu <= 1:
            raise ValueError(f"no usable resolution tag (unit={unit}, ppu={ppu})")
        return per / ppu


def slide_mpp(sl, path):
    try:
        return tiff_mpp(path)
    except Exception:
        v = sl.properties.get("openslide.mpp-x")
        if v:
            return float(v)
        log("WARNING: no resolution tag; assuming 0.2425 um/px"); return 0.2425


def grid(W, H, size0, mask, mscale, thr=GRID_THR):
    xs = np.arange(0, W - size0 + 1, size0); ys = np.arange(0, H - size0 + 1, size0)
    gx, gy = np.meshgrid(xs, ys); c = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.int64)
    S = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1)); S[1:, 1:] = mask.astype(np.float64).cumsum(0).cumsum(1)
    h, w = mask.shape
    x0 = np.clip((c[:, 0] / mscale).astype(int), 0, w); y0 = np.clip((c[:, 1] / mscale).astype(int), 0, h)
    x1 = np.clip(((c[:, 0] + size0) / mscale).astype(int), 0, w); y1 = np.clip(((c[:, 1] + size0) / mscale).astype(int), 0, h)
    x1 = np.maximum(x1, np.minimum(x0 + 1, w)); y1 = np.maximum(y1, np.minimum(y0 + 1, h))
    f = (S[y1, x1] - S[y0, x1] - S[y1, x0] + S[y0, x0]) / np.maximum((x1 - x0) * (y1 - y0), 1)
    return c[f >= thr]


def cap(c, W, H, size0, n=MAX_TILES):
    """Fixed spatial lattice (every k-th grid column and row), with k set from the slide extent only, so editing or
    perturbing the mask can only remove tiles from the bag, never resample it. k assumes ~50% tissue: bags are
    ~n tiles on typical slides and at most ~2n."""
    k = max(1, int(np.ceil(np.sqrt((W // size0) * (H // size0) * 0.5 / n))))
    g = c // size0
    return c[(g[:, 0] % k == 0) & (g[:, 1] % k == 0)]


def read_mask(path):
    import openslide
    ms = openslide.OpenSlide(path)
    m = np.asarray(ms.read_region((0, 0), 0, ms.level_dimensions[0]))[..., 0] > 0
    ms.close(); return m


# ------------------------------------------------------------------ tiles -> embeddings
class Reader:
    def __init__(self, path, threads):
        import openslide
        self.path, self.local, self.os = path, threading.local(), openslide
        self.pool = ThreadPoolExecutor(threads)

    def one(self, xy, size):
        if not hasattr(self.local, "s"):
            self.local.s = self.os.OpenSlide(self.path)
        try:
            r = self.local.s.read_region((int(xy[0]), int(xy[1])), 0, (size, size))
        except self.os.OpenSlideError:
            del self.local.s; return None
        bg = Image.new("RGB", r.size, (255, 255, 255)); bg.paste(r, mask=r.split()[3])
        return np.asarray(bg.resize((TILE_PX, TILE_PX), Image.BILINEAR))


@torch.inference_mode()
def embed(M, wsi, coords, size, bs=64, fn=None):
    rd = Reader(wsi, threads=min(16, os.cpu_count() or 4)); q = queue.Queue(maxsize=4)

    def prod():
        for i in range(0, len(coords), bs):
            q.put(list(rd.pool.map(lambda c: rd.one(c, size), coords[i:i + bs])))
        q.put(None)
    threading.Thread(target=prod, daemon=True).start()
    out, ok = [], []
    while (b := q.get()) is not None:
        good = [t is not None for t in b]; ok += good
        if any(good):
            x = torch.from_numpy(np.stack([t for t in b if t is not None])).to(M["dev"]).permute(0, 3, 1, 2).float().div_(255)
            out.append((fn or f_midnight)(M, x))
    rd.pool.shutdown()
    if isinstance(out[0], dict):
        return {k: torch.cat([o[k] for o in out]) for k in out[0]}, np.array(ok)
    return torch.cat(out), np.array(ok)


def f_midnight(M, x):
    with torch.autocast(M["dev"].type, dtype=torch.float16, enabled=M["dev"].type == "cuda"):
        hs = M["mn"](pixel_values=(x - M["mean"]) / M["std"]).last_hidden_state
    return torch.cat([hs[:, 0], hs[:, 1:].mean(1)], -1).float()


def f_second(M, x):
    """UNI2-h [CLS | mean patch after 8 registers] and Phikon-v2 CLS, rounded to fp16 as the stored training features."""
    xi = ((x - M["imean"]) / M["istd"]).half()
    with torch.autocast(M["dev"].type, dtype=torch.float16, enabled=M["dev"].type == "cuda"):
        o = M["uni"].forward_features(xi); u = torch.cat([o[:, 0], o[:, 9:].mean(1)], -1)
        p = M["phk"](pixel_values=xi).last_hidden_state[:, 0]
    return dict(uni2h=u.half().float(), phikon2=p.half().float())


@torch.inference_mode()
def tissue_keep(M, X):
    cal = M["cal"]; tp = X @ M["tcoef"] + cal["tissue"]["intercept"]; keep = tp >= cal["tissue"]["thr"]
    if keep.sum() < 16:
        keep = tp >= torch.sort(tp).values[-min(16, len(tp))]
    return keep.cpu().numpy()


@torch.inference_mode()
def score_v2(M, F):
    """F: encoder -> [n, D] features of the analysed tiles. Returns ensemble z, per-tile deletion effect, per-encoder z."""
    zs, cs, per, parts = [], [], {}, []
    for e in ENCS:
        ce = M["cal"]["encoders"][e]; ze, ce_ = [], []
        for net, mu, sd in M["mils"][e]:
            att, s = net.parts(F[e]); v = (s - mu) / sd; y = (att * v).sum()
            ze.append(y); ce_.append(att * (v - y) / (1 - att).clamp(min=1e-6))
            parts.append((att.double().cpu().numpy(), v.double().cpu().numpy(), 1.0 / (len(ENCS) * len(M["mils"][e]) * ce["enc_sd"])))
        z_e = (torch.stack(ze).mean() - ce["enc_mu"]) / ce["enc_sd"]; per[e] = float(z_e)
        zs.append(z_e); cs.append(torch.stack(ce_).mean(0) / ce["enc_sd"])
    A, V, w = (np.stack([p[0] for p in parts]), np.stack([p[1] for p in parts]), np.array([p[2] for p in parts]))
    return float(torch.stack(zs).mean()), torch.stack(cs).mean(0).cpu().numpy(), per, (A, V, w)


def sequential_effects(A, V, w):
    """Sequential knockout over the analysed tiles (analysis/e33). Each step removes the tile whose removal lowers
    the ensemble score most given the tiles already removed, with every effect recomputed exactly (Eq. 1 per model:
    y_m = P_m / Q_m over remaining tiles). Stops when no removal lowers the score. Returns per-tile effect (the drop at
    the step it was removed, made non-increasing along the order; for tiles never removed, their effect at the stop,
    <= 0) and the full order (removed tiles first, then the rest from least to most negative)."""
    M, n = A.shape; AV = A * V; P = AV.sum(1); Q = A.sum(1); alive = np.ones(n, bool); order, gain = [], []
    d = np.zeros(n)
    while alive.sum() > 1:
        y = P / Q; y2 = (P[:, None] - AV) / np.clip(Q[:, None] - A, 1e-12, None)
        d = (w[:, None] * (y[:, None] - y2)).sum(0); d[~alive] = -np.inf; i = int(np.argmax(d))
        if d[i] <= 0:
            break
        order.append(i); gain.append(float(d[i])); alive[i] = False; P -= AV[:, i]; Q -= A[:, i]
    eff = np.zeros(n)
    if order:
        eff[order] = np.minimum.accumulate(np.array(gain))
    rest = np.where(alive)[0]
    if len(rest) and np.isfinite(d[rest]).all():
        eff[rest] = np.minimum(d[rest], 0.0)
    rest = rest[np.argsort(-eff[rest], kind="stable")]
    return eff, np.concatenate([np.array(order, int), rest]).astype(int)


def heat_rank(order, k_up, xy, sel, tissue, all_xy):
    """uint8 attribution map on the grid, by rank so the 8-bit map keeps the sequential-knockout order.
    Analysed tiles in order: the first k_up (as many as raise the score on their own) 255..129 (red), the rest
    127..1 (blue, last = most protective). Unread tissue takes the value of its nearest analysed tile, so a region is
    removed together with the field that stands for it and removing the top x% of tissue removes about the top x% of
    analysed fields in order. Non-tissue 128; outside the mask 0 (paint)."""
    from scipy.spatial import cKDTree
    n = len(order); k_up = int(min(max(k_up, 0), n)); r = np.arange(n); val = np.empty(n, np.int32)
    up = 255 - np.floor(126 * r[:k_up] / max(k_up, 1)).astype(int)
    dn = 127 - np.floor(126 * (r[k_up:] - k_up) / max(n - k_up, 1)).astype(int)
    val[order] = np.concatenate([up, dn])
    lat = np.full(len(xy), 128, np.int32); lat[sel] = val
    rest = tissue.copy(); rest[sel] = False
    if rest.any():
        lat[rest] = val[cKDTree(xy[sel]).query(xy[rest])[1]]
    dist, nn = cKDTree(xy).query(all_xy)
    return np.clip(lat[nn], 1, 255).astype(np.uint8)


@torch.inference_mode()
def score(M, X):
    cal = M["cal"]; tp = X @ M["tcoef"] + cal["tissue"]["intercept"]
    keep = tp >= cal["tissue"]["thr"]
    if keep.sum() < 16:
        keep = tp >= torch.sort(tp).values[-min(16, len(tp))]
    Xk = X[keep]; zs, cs = [], []
    for net, mu, sd in M["mils"]:
        att, s = net.parts(Xk)
        v = (s - mu) / sd; y = (att * v).sum()
        zs.append(y); cs.append(att * (v - y) / (1 - att).clamp(min=1e-6))   # exact deletion effect y - y_{-i}
    return float(torch.stack(zs).mean()), torch.stack(cs).mean(0).cpu().numpy(), keep.cpu().numpy()


# ------------------------------------------------------------------ outputs
def write_heatmap(path, heat, spacing_um):
    import tifffile
    levels = [heat]
    while max(levels[-1].shape) > 512:
        levels.append(levels[-1][::2, ::2])
    res = (1e4 / spacing_um, 1e4 / spacing_um)
    try:
        import imagecodecs  # noqa: F401  (LZW like the organiser masks)
        comp = "lzw"
    except ImportError:
        comp = "zlib"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffWriter(path, bigtiff=heat.size > 2**31) as tw:
        for i, lv in enumerate(levels):
            tw.write(np.ascontiguousarray(lv), tile=(512, 512), compression=comp, photometric="minisblack",
                     resolution=(res[0] / 2**i, res[1] / 2**i), resolutionunit="CENTIMETER",
                     subfiletype=1 if i else 0, metadata=None)


class MIL(nn.Module):                                     # training/runner.MIL with heads=1, pool="att"
    def __init__(self, D, H, A, heads, p, pool, topk):
        super().__init__()
        assert heads == 1 and pool == "att"
        self.enc = nn.Sequential(nn.Linear(D, H), nn.GELU(), nn.Dropout(p))
        self.a, self.b = nn.Linear(H, A), nn.Linear(H, A)
        self.w = nn.ModuleList([nn.Linear(A, 1)]); self.head = nn.ModuleList([nn.Linear(H, 1)])

    def parts(self, x):
        h = self.enc(x)
        att = torch.softmax(self.w[0](torch.tanh(self.a(h)) * torch.sigmoid(self.b(h))).squeeze(-1), 0)
        return att, self.head[0](h).squeeze(-1)          # risk = head(sum_i att_i h_i) = sum_i att_i s_i


def lattice_k(W, H, size0, n=MAX_TILES):
    return max(1, int(np.ceil(np.sqrt((W // size0) * (H // size0) * 0.5 / n))))   # as cap()


def select_lattice(keep, xy, size0, k):
    """Mask-stable subsample: tissue tiles at one lattice position in five, (gx + 2 gy) mod 5 == 0 on lattice indices
    (an evenly spread sub-lattice; ~1,940 tiles on the median slide, close to the ~2,000 the models were trained on).
    Whether a tile is read depends only on its own position, so editing the mask removes reads and never redraws them
    (analysis/e31: the list-order rule below redraws the sample after an edit and weakens deletion)."""
    g = xy // size0 // k
    return np.where(keep & ((g[:, 0] + 2 * g[:, 1]) % 5 == 0))[0]


def select(keep):
    """Training tile rule (E19 tile index + features/e26): tissue tiles, <= NMAX by uniform stride, then every second."""
    idx = np.where(keep)[0]
    if len(idx) > NMAX:
        idx = idx[np.arange(0, len(idx), int(np.ceil(len(idx) / NMAX)))[:NMAX]]
    return idx[::2]


def spread(c, xy, sel, tissue, fill=0.5):
    """Per embedded tile: exact effect on analysed tiles, fill x nearest analysed tile on other tissue tiles, 0 elsewhere."""
    from scipy.spatial import cKDTree
    full = np.zeros(len(xy)); full[sel] = c; rest = tissue.copy(); rest[sel] = False
    if rest.any():
        full[rest] = fill * c[cKDTree(xy[sel]).query(xy[rest])[1]]
    return full


def heat_values(c, keep, kept_xy, all_xy, fill=0.5, scale=None):
    """uint8 per grid tile: 128 + 127 * effect / scale. Analysed tiles carry their exact deletion effect; tissue tiles
    off the lattice (not analysed) take `fill` x the nearest analysed tile, so they never outrank the real evidence."""
    from scipy.spatial import cKDTree
    scale = (np.quantile(np.abs(c), 0.99) if scale is None else scale) + 1e-12
    full = np.zeros(len(keep)); full[keep] = c                       # filtered (non-tissue) tiles stay neutral
    d, nn = cKDTree(kept_xy).query(all_xy)
    v = np.where(d == 0, full[nn], fill * full[nn])
    return np.clip(np.round(128 + 127 * v / scale), 1, 255).astype(np.uint8), scale


def paint(vals, all_xy, size0, mscale, mask):
    """grid-tile values -> mask-grid image (0 outside tissue)."""
    gx, gy = all_xy[:, 0] // size0, all_xy[:, 1] // size0
    G = np.zeros((gy.max() + 2, gx.max() + 2), np.uint8); G[gy, gx] = vals
    h, w = mask.shape
    rows = np.minimum((np.arange(h) * mscale // size0).astype(int), G.shape[0] - 1)
    cols = np.minimum((np.arange(w) * mscale // size0).astype(int), G.shape[1] - 1)
    out = G[rows][:, cols]; out[~mask] = 0
    return out


def write_report(path, sl, info, tiles_up, tiles_down, heat_small):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list("rb", ["#2166ac", "#f7f7f7", "#b2182b"])
    thumb = np.asarray(sl.get_thumbnail((1400, 1400)).convert("RGB"))
    hs = np.asarray(Image.fromarray(heat_small).resize((thumb.shape[1], thumb.shape[0]), Image.NEAREST)).astype(float)
    over = np.ma.masked_where(hs == 0, (hs - 128) / 127)
    with PdfPages(path) as pdf:
        f = plt.figure(figsize=(8.27, 11.69))
        f.text(0.07, 0.955, "Computational recurrence-risk report", size=17, weight="bold")
        f.text(0.07, 0.935, "Research use only. Not a diagnostic test. Grading was not used by this model.", size=8.5, color="#555")
        lines = [f"Estimated risk group: {info['group'].upper()}",
                 f"Relative risk score: {info['z']:+.2f} SD (training-cohort percentile {info['pct']:.0f})",
                 f"Predicted time to biochemical recurrence (ranking value): {info['years']:.1f} years",
                 f"Tissue: about {info['mm2']:.0f} mm2 sampled; {info['n_tiles']:,} fields of 112 x 112 um analysed"]
        for i, s in enumerate(lines):
            f.text(0.07, 0.90 - i * 0.022, s, size=10.5, weight="bold" if i == 0 else None)
        a1 = f.add_axes([0.05, 0.42, 0.44, 0.38]); a1.imshow(thumb); a1.set_title("Specimen overview", size=9); a1.axis("off")
        a2 = f.add_axes([0.51, 0.42, 0.44, 0.38]); a2.imshow(thumb); a2.imshow(over, cmap=cm, vmin=-1, vmax=1, alpha=0.6)
        a2.set_title("Where the estimate comes from", size=9); a2.axis("off")
        txt = ("How to read the map. Fields are ordered by sequential knockout: the field whose removal lowers the "
               "estimate most comes first, every effect is recomputed exactly, and so on. Red regions raise the "
               "estimate, blue regions lower it; the strongest colours are the first and last in that order. Each "
               "region takes the value of the analysed field it surrounds. The model saw the tissue as "
               "112 um fields at 20x and learned which patterns accompanied recurrence in 806 prostatectomy cases "
               "(Radboudumc and TCGA). Page 2 shows the fields with the largest contributions in each direction for "
               "the pathologist to review.\n\nLimitations. Trained on Radboudumc and TCGA cohorts; scanner and "
               "staining differences can shift scores. Contributions show association with the model output, not "
               "causation. Ink, folds and out-of-focus areas are filtered automatically but may remain.")
        f.text(0.07, 0.36, txt, size=9, wrap=True, va="top", ha="left")
        pdf.savefig(f); plt.close(f)
        f = plt.figure(figsize=(8.27, 11.69))
        f.text(0.07, 0.955, "Fields with the largest contributions (112 um, 20x)", size=13, weight="bold")
        for r, (title, T) in enumerate((("Pushing towards recurrence", tiles_up), ("Pushing towards no recurrence", tiles_down))):
            f.text(0.07, 0.915 - r * 0.45, title, size=10.5, color="#b2182b" if r == 0 else "#2166ac", weight="bold")
            for k, (im, xy_mm, cv) in enumerate(T[:6]):
                ax = f.add_axes([0.05 + (k % 3) * 0.31, 0.71 - r * 0.45 - (k // 3) * 0.2, 0.28, 0.18])
                ax.imshow(im); ax.axis("off"); ax.set_title(f"x {xy_mm[0]:.1f} mm, y {xy_mm[1]:.1f} mm  ({cv:+.3f})", size=7)
        pdf.savefig(f); plt.close(f)


# ------------------------------------------------------------------ main
def run(model):
    keyset = tuple(sorted(sv["socket"]["slug"] for sv in json.load(open(INPUT_PATH / "inputs.json"))))
    assert keyset == ("prostatectomy-tissue-mask", "prostatectomy-tissue-whole-slide-image"), keyset
    return handle(model)


def find(d):
    fs = sorted(sum((glob.glob(str(INPUT_PATH / d / f"*.{e}")) for e in ("tif", "tiff", "mha")), []))
    assert fs, f"no input in {d}"; return fs[0]


def handle(M):
    import openslide
    t0 = time.time(); wsi, mpath = find("images/prostatectomy-wsi"), find("images/prostatectomy-tissue-mask")
    sl = openslide.OpenSlide(wsi); W, H = sl.level_dimensions[0]; mpp0 = slide_mpp(sl, wsi)
    mask = read_mask(mpath); mscale = W / mask.shape[1]; mspacing = mpp0 * mscale
    s20 = int(round(TILE_PX * MPP20 / mpp0))
    all_xy = grid(W, H, s20, mask, mscale); xy = cap(all_xy, W, H, s20)
    log(f"slide {W}x{H} mpp {mpp0:.4f}; mask {mask.shape} ({mspacing:.2f} um/px); tiles {len(all_xy)} -> {len(xy)}")
    if len(xy) == 0:                                    # empty mask: fall back to the whole slide
        mask = np.ones_like(mask); all_xy = grid(W, H, s20, mask, mscale); xy = cap(all_xy, W, H, s20)
    X, ok = embed(M, wsi, xy, s20); xy = xy[ok]
    tissue = tissue_keep(M, X)
    sel = select_lattice(tissue, xy, s20, lattice_k(W, H, s20)) if SELECT == "lattice" else select(tissue)
    if len(sel) < 16:                                   # tiny tissue: fall back to the list rule
        sel = select(tissue)
    log(f"Midnight embedded {len(xy)} tiles; tissue {tissue.sum()}; analysed {len(sel)}")
    F2, ok2 = embed(M, wsi, xy[sel], s20, fn=f_second); sel = sel[ok2]
    F = dict(midnight=X[sel].half().float(), **F2)
    log(f"UNI2-h + Phikon-v2 embedded {len(sel)} tiles")
    z, c_single, per, (A_, V_, w_) = score_v2(M, F)
    c, order = sequential_effects(A_, V_, w_)
    log(f"sequential knockout: {int((c > 0).sum())} of {len(c)} tiles lower the score when removed")
    if os.environ.get("OWL_DEBUG_DUMP"):                # per-tile parts for the mask-edit check (not used in scoring)
        dump = dict(xy=xy[sel], z=z, c=c, c_single=c_single, order=order)
        with torch.no_grad():
          for e in ENCS:
            for k_, (net, mu, sd) in enumerate(M["mils"][e]):
                att, s_ = net.parts(F[e]); dump[f"{e}_{k_}_a"] = att.float().cpu().numpy(); dump[f"{e}_{k_}_v"] = ((s_ - mu) / sd).float().cpu().numpy()
        np.savez(OUTPUT_PATH / "debug.npz", **dump)
    keep = np.zeros(len(xy), bool); keep[sel] = True
    cal = M["cal"]; kxy = xy[keep]
    years = float(np.clip(cal["median_followup_y"] * np.exp(-z / max(cal["z_sd"], 1e-6)), 0.1, 50.0))
    group = "low" if z < cal["q50"] else "medium" if z < cal["q80"] else "high"
    from scipy.stats import norm
    pct = float(100 * norm.cdf(z / max(cal["z_sd"], 1e-6)))
    log(f"z {z:+.3f} group {group} years {years:.2f}; per encoder " + " ".join(f"{e} {v:+.2f}" for e, v in per.items()))

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    json.dump(years, open(OUTPUT_PATH / "overall-survival-years.json", "w"))
    json.dump(group, open(OUTPUT_PATH / "prostate-cancer-risk-classification.json", "w"))
    heat = paint(heat_rank(order, int((c_single > 0).sum()), xy, sel, tissue, all_xy), all_xy, s20, mscale, mask)
    heat_view = heat                                     # the report shows the submitted map
    if os.environ.get("OWL_DEBUG_DUMP"):
        np.save(OUTPUT_PATH / "heat_view.npy", heat_view)
    write_heatmap(OUTPUT_PATH / "images/prostatectomy-overall-survival-attribution-heatmap/attribution.tif", heat, mspacing)
    log("heatmap written")
    try:
        up = [int(i) for i in order[:6]]; dn = [int(i) for i in np.argsort(c_single) if c_single[i] < 0][:6]; rd = Reader(wsi, 4)
        grab = lambda ids: [(rd.one(kxy[i], s20), (kxy[i] + s20 / 2) * mpp0 / 1000, float(c_single[i])) for i in ids]
        f = max(1, max(heat.shape) // 2048)
        write_report(OUTPUT_PATH / "pathology-report.pdf", sl,
                     dict(group=group, z=z, pct=pct, years=years, n_tiles=int(keep.sum()),
                          mm2=tissue.sum() * (s20 * mpp0 / 1000) ** 2), grab(up), grab(dn), heat_view[::f, ::f])
    except Exception as e:                                # the report must never cost us the prediction
        log(f"report failed ({type(e).__name__}: {e}); writing minimal PDF")
        write_min_pdf(OUTPUT_PATH / "pathology-report.pdf", group, years)
    sl.close()
    log(f"case done in {time.time() - t0:.0f}s")
    return 0


def write_min_pdf(path, group, years):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = plt.figure(figsize=(8.27, 11.69)); f.text(0.1, 0.9, f"Risk group: {group}; ranking time {years:.1f} y", size=12)
    f.savefig(path, format="pdf"); plt.close(f)


if __name__ == "__main__":
    raise SystemExit(run(model=init_model()))
