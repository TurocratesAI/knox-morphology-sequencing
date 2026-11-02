"""OWL submission v2: three frozen encoders (Midnight-12K, UNI2-h, Phikon-v2) -> gated-attention MIL fold models
(joint pooled Cox) -> equal-weight ensemble, with an exact per-tile deletion effect used for the attribution map and
the report. v1 (Midnight only, all tissue tiles) is kept as inference_v1.py.

v2 changes to the pipeline below: after the Midnight pass and E0 tissue filter, the analysed set is the training
tile rule (at most 4,000 tissue tiles by uniform stride, then every second one, ~2,000); UNI2-h and Phikon-v2 embed
only those (fp16 weights and ImageNet normalisation, as for training). Per encoder e and fold model k:
z_ek = (sum_i a_i s_i - mu_ek) / sd_ek; z_e = (mean_k z_ek - mu_e) / sd_e; z = mean_e z_e. The deletion effect of
tile i is the same average of each model's exact a_i (v_i - z_ek) / (1 - a_i).

Pipeline per case (mirrors pipeline/owl_extract.py + experiments/e0/e1 used in training):
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
IN_MEAN, IN_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)        # UNI2-h, Phikon-v2 (as experiments/e26)
ENCS = ("midnight", "uni2h", "phikon2")
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


class ABMIL(nn.Module):                                   # identical to experiments/e1_baselines.ABMIL
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
    cp = MODEL_DIR / "concepts" / "concepts.npz"
    concepts = {k: v for k, v in np.load(cp).items()} if cp.exists() else None
    log("model loaded" + ("" if concepts is not None else " (no concept read-outs: report without field estimates)"))
    return dict(dev=dev, mn=mn, uni=uni, phk=phk, mils=mils, cal=cal, concepts=concepts, mean=t(MN_MEAN), std=t(MN_STD), imean=t(IN_MEAN), istd=t(IN_STD),
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
    """Sequential knockout over the analysed tiles (experiments/e33). Each step removes the tile whose removal lowers
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


class MIL(nn.Module):                                     # experiments/runner.MIL with heads=1, pool="att"
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
    (experiments/e31: the list-order rule below redraws the sample after an edit and weakens deletion)."""
    g = xy // size0 // k
    return np.where(keep & ((g[:, 0] + 2 * g[:, 1]) % 5 == 0))[0]


def select(keep):
    """Training tile rule (experiments/e19 + e26): tissue tiles, <= NMAX by uniform stride, then every second."""
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


# 101 quantiles of development out-of-fold risk scores (deployed folds; model/owl_v2/oof.npz 'ens'), for the report's percentile
REF_Q = [-1.6560, -1.4729, -1.4002, -1.3542, -1.2884, -1.2465, -1.1966, -1.1535, -1.1287, -1.1096, -1.0577, -1.0303, -1.0032, -0.9781, -0.9542, -0.9320, -0.9121, -0.8915, -0.8629, -0.8318, -0.8121, -0.7777, -0.7696, -0.7549, -0.7277, -0.7033, -0.6935, -0.6639, -0.6513, -0.6164, -0.5941, -0.5755, -0.5538, -0.5343, -0.5064, -0.4809, -0.4697, -0.4420, -0.4130, -0.3978, -0.3818, -0.3661, -0.3439, -0.3265, -0.3029, -0.2850, -0.2585, -0.2302, -0.2042, -0.1936, -0.1777, -0.1489, -0.1369, -0.1072, -0.0819, -0.0619, -0.0292, -0.0189, 0.0124, 0.0422, 0.0648, 0.0945, 0.1217, 0.1423, 0.1782, 0.2110, 0.2389, 0.2879, 0.3340, 0.3683, 0.4074, 0.4502, 0.4868, 0.5178, 0.5536, 0.6097, 0.6507, 0.6864, 0.7147, 0.7615, 0.8163, 0.8661, 0.9317, 0.9972, 1.0664, 1.1253, 1.1786, 1.2361, 1.2890, 1.3345, 1.3734, 1.4475, 1.5121, 1.5667, 1.6180, 1.6927, 1.8097, 1.9148, 2.0418, 2.1693, 3.0853]


# ------------------------------------------------------------------ report (pathologist- and oncologist-facing)
# Three kinds of statement, labelled on the page: computed model response (exact score change when fields are excluded),
# automated image estimate (model-generated tissue label or measurement), development outcome evidence (earlier patients).
# Sign convention everywhere: score change = score after exclusion - original score (negative = lower score).
REF = json.load(open(Path(__file__).with_name("report_ref.json")))
COMP_LABEL = ["Epithelium-enriched glands", "Benign-type glands", "Stroma", "Other tissue"]
NUC_SHOW = [(0, "estimated nuclear density"), (1, "estimated nuclear size"), (3, "nuclear staining (haematoxylin)"), (4, "estimated nuclear crowding")]
REFS = [
    "Ilse M, Tomczak JM, Welling M. Attention-based deep multiple instance learning. Proc Mach Learn Res. 2018;80:2127-36.",
    "Cox DR. Regression models and life-tables. J R Stat Soc Series B. 1972;34(2):187-220.",
    "Harrell FE Jr, Califf RM, Pryor DB, Lee KL, Rosati RA. Evaluating the yield of medical tests. JAMA. 1982;247(18):2543-6.",
    "Karasikov M, van Doorn J, Kanzig N, et al. Training state-of-the-art pathology foundation models with orders of magnitude less data. arXiv:2504.05186. 2025.",
    "Chen RJ, Ding T, Lu MY, Williamson DFK, et al. Towards a general-purpose foundation model for computational pathology. Nat Med. 2024;30:850-62.",
    "Filiot A, Jacob P, Mac Kain A, Saillard C. Phikon-v2, a large and public feature extractor for biomarker prediction. arXiv:2409.09173. 2024.",
    "Pachitariu M, Rariden M, Stringer C. Cellpose-SAM: superhuman generalization for cellular segmentation. bioRxiv. 2025. doi:10.1101/2025.04.28.651001.",
    "Veltri RW, Christudass CS, Isharwal S. Nuclear morphometry, nucleomics and prostate cancer progression. Asian J Androl. 2012;14(3):375-84. doi:10.1038/aja.2011.148.",
    "Howard FM, Dolezal J, Kochanny S, et al. The impact of site-specific digital histology signatures on deep learning model accuracy and bias. Nat Commun. 2021;12:4423. doi:10.1038/s41467-021-24698-1.",
]


def pct_of(z, q):
    q = np.asarray(q, float); return float(np.interp(z, q, np.linspace(0.0, 100.0, len(q))))


def ref_percentile(z):
    return pct_of(z, REF_Q)


def concept_readouts(M, Xm):
    """Compartment (argmax) and nuclear measurement estimates from the Midnight embeddings (experiments/e34)."""
    C_ = M.get("concepts")
    if C_ is None:
        return None, None
    Xs = (Xm - C_["mu"]) / C_["sd"]
    comp = np.argmax(Xs @ C_["comp_W"].T + C_["comp_b"], 1)
    nuc = (Xs @ C_["nuc_W"].T + C_["nuc_b"]) * C_["ysd"] + C_["ymu"]
    return comp, nuc


def excl_change(A, V, w, S):
    """Exact change in the risk score (score units), after minus before, when the analysed fields in S are excluded:
    per member y_-S = (y - sum_S a v) / (1 - A_S), combined with the risk score's fixed member weights."""
    AV = A * V; y = AV.sum(1); y2 = (y - AV[:, S].sum(1)) / np.clip(1 - A[:, S].sum(1), 1e-12, None)
    return float((w * (y2 - y)).sum())


def report_data(M, Xm, A, V, w, c_single, order, kxy, s20, mpp0, per):
    sd = M["cal"]["z_sd"]; n = len(kxy); comp, nuc = concept_readouts(M, Xm); d = dict(n=n, sd=sd, comp=comp)
    up = [int(i) for i in order[:6]]; dn = [int(i) for i in np.argsort(c_single, kind="stable") if c_single[i] < 0][:3]
    pct = None if nuc is None else {k: (np.argsort(np.argsort(nuc[:, k], kind="stable"), kind="stable") + 0.5) / n * 100 for k, _ in NUC_SHOW}

    def describe(i):
        if comp is None:
            return ""
        parts = [f"{name} near the {pct[k][i]:.0f}th percentile of this slide's sampled fields"
                 for k, name in sorted(NUC_SHOW, key=lambda t: -abs(pct[t[0]][i] - 50))[:2]]
        return f"{COMP_LABEL[comp[i]].lower()}; " + "; ".join(parts)
    field = lambda i: dict(i=i, mm=(kxy[i] + s20 / 2) * mpp0 / 1000, chg=-float(c_single[i]) / sd, desc=describe(i),
                           comp=None if comp is None else int(comp[i]))
    d["up"], d["dn"] = [field(i) for i in up], [field(i) for i in dn]
    d["enc"] = {e: pct_of(per[e], REF["enc_q"][e]) for e in per}
    if comp is not None:
        rows = []
        for q in range(4):
            S = comp == q
            if 0 < S.sum() < n:
                rows.append((COMP_LABEL[q], int(S.sum()), 100 * S.mean(), excl_change(A, V, w, S) / sd))
        d["comp_rows"] = rows
        P = np.where(comp == 0)[0]
        if len(P) >= 40:
            o = P[np.argsort(nuc[P, 0], kind="stable")]; k = len(P) // 4; mk = lambda ids: np.isin(np.arange(n), ids)
            d["dens"] = dict(n=len(P), k=k, hi=excl_change(A, V, w, mk(o[-k:])) / sd, lo=excl_change(A, V, w, mk(o[:k])) / sd,
                             d_hi=float(np.mean(nuc[o[-k:], 0])), d_lo=float(np.mean(nuc[o[:k], 0])))
        else:
            d["dens"] = dict(n=len(P))
    return d


def write_report_v4(path, sl, info, d, crops, heat_small):
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import LinearSegmentedColormap
    INK, MUTED, RED, BLUE, GRID, ACC = "#1f1b2e", "#5d5872", "#b2182b", "#2166ac", "#e4e1ea", "#3b5b92"
    BANDC = {"low": "#7a8fb5", "medium": "#c08a3e", "high": "#9c3d54"}
    cm = LinearSegmentedColormap.from_list("rb", ["#2166ac", "#f7f7f7", "#b2182b"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.linewidth": 0.6})
    W_ = lambda s, n=112: textwrap.fill(s, n)
    thumb = np.asarray(sl.get_thumbnail((1400, 1400)).convert("RGB")); W0 = sl.level_dimensions[0][0]; fx = thumb.shape[1] / W0
    hs = np.asarray(Image.fromarray(heat_small).resize((thumb.shape[1], thumb.shape[0]), Image.NEAREST)).astype(float)
    over = np.ma.masked_where(hs == 0, (hs - 128) / 127)
    band, P = info["group"], info["pct"]; case = info.get("case", "")

    def frame(f, title, page, sub=None):
        f.text(0.07, 0.962, title, size=14.5, weight="bold", color=INK)
        if sub:
            f.text(0.07, 0.945, sub, size=8, color=MUTED, va="top")
        f.add_artist(plt.Line2D([0.07, 0.93], [0.035, 0.035], color=GRID, lw=0.8, transform=f.transFigure))
        f.text(0.07, 0.022, f"KNOX research recurrence-ranking report  |  {case}  |  development-stage research test; clinical utility "
               "has not been established", size=6.6, color=MUTED)
        f.text(0.93, 0.022, f"page {page} of 5", size=6.6, color=MUTED, ha="right")

    def para(f, x, y, s, size=8.4, n=112, color=INK, lh=0.0128):
        t = W_(s, n); f.text(x, y, t, size=size, va="top", color=color); return y - lh * (t.count("\n") + 1)

    def marks(ax, fields, color, labels):
        for lab, fd in zip(labels, fields):
            cx, cy = fd["mm"][0] * 1000 / info["mpp0"] * fx, fd["mm"][1] * 1000 / info["mpp0"] * fx
            ax.plot(cx, cy, "o", ms=5, mfc="none", mec=color, mew=1.1); ax.text(cx + 9, cy - 9, lab, color=color, size=7.5, weight="bold")

    with PdfPages(path) as pdf:
        # ---------------- page 1: the result and what it means
        f = plt.figure(figsize=(8.27, 11.69))
        frame(f, f"{case}: {P:.0f}th percentile of the development reference", 1)
        y = 0.93
        y = para(f, 0.07, y, f"This slide's model score is higher than about {P:.0f}% of scores in the development reference and falls in the "
                 f"study-defined {band} band. It is a relative ranking, not a probability of recurrence and not a time to recurrence. The model "
                 "is in development: results on the challenge's hidden test centres are not yet available, and this report does not show how "
                 "the result should change follow-up or treatment.", size=9.2, n=104, lh=0.0138) - 0.012
        ax = f.add_axes([0.10, y - 0.045, 0.80, 0.022]); ax.set_xlim(0, 100); ax.set_ylim(0, 1); ax.set_yticks([])
        for lo_, hi_, b in ((0, 50, "low"), (50, 80, "medium"), (80, 100, "high")):
            ax.axvspan(lo_, hi_, color=BANDC[b], alpha=0.35 if b != band else 0.75, lw=0)
            ax.text((lo_ + hi_) / 2, 1.35, f"{b}  ({lo_}th to {hi_}th)" if b != "high" else "high  (80th and above)", ha="center", size=7, color=INK)
        ax.plot([P, P], [-0.2, 1.2], color=INK, lw=2); ax.text(P, -1.6, f"this case: {P:.0f}", ha="right" if P > 88 else "left" if P < 12 else "center", size=7.5, weight="bold", color=INK)
        ax.set_xticks([0, 50, 80, 100]); ax.tick_params(labelsize=6.5, length=2); [s_.set_visible(False) for s_ in ax.spines.values()]
        y -= 0.10
        hgt = 0.30; a1 = f.add_axes([0.05, y - hgt, 0.44, hgt]); a1.imshow(thumb); a1.axis("off"); a1.set_title("H&E overview", size=8.5)
        a2 = f.add_axes([0.51, y - hgt, 0.44, hgt]); a2.imshow(thumb); im = a2.imshow(over, cmap=cm, vmin=-1, vmax=1, alpha=0.6)
        a2.axis("off"); a2.set_title("Removal-order map (the submitted attribution map)", size=8.5)
        for a_ in (a1, a2):
            marks(a_, d["up"], RED, [str(k + 1) for k in range(len(d["up"]))]); marks(a_, d["dn"], BLUE, "abc"[:len(d["dn"])])
        y -= hgt + 0.012
        cb = f.add_axes([0.58, y, 0.30, 0.008]); bar = f.colorbar(im, cax=cb, orientation="horizontal", ticks=[-1, 1])
        bar.ax.set_xticklabels(["removed later", "removed earlier"], size=6.5); bar.outline.set_linewidth(0.4)
        y = para(f, 0.07, y + 0.004, "Colour shows the order of sequential removal, not whether a field raises or lowers the score on its own.",
                 size=7.2, n=60, color=MUTED) - 0.02
        f.text(0.07, y, "Key findings for review", size=10.5, weight="bold", color=INK); y -= 0.022
        kf = []
        ep = [r for r in d.get("comp_rows", []) if r[0] == COMP_LABEL[0]]
        if ep:
            kf.append(f"Computed model response. Excluding all {ep[0][1]:,} epithelium-enriched gland fields changes the score by {ep[0][3]:+.2f} "
                      "development-reference SD (page 3).")
        if d["up"]:
            f1 = d["up"][0]; lab = COMP_LABEL[f1["comp"]].lower() if f1["comp"] is not None else "unlabelled"
            kf.append(f"The field selected first (field 1, page 2) carries the automated label \"{lab}\". Please confirm its tissue identity "
                      "and check for artefact before relying on the result.")
        dn_ = d.get("dens", {})
        if "hi" in dn_:
            kf.append(f"Computed model response. Within epithelium-enriched fields, excluding the densest quarter changes the score by "
                      f"{dn_['hi']:+.2f} SD and excluding the least dense quarter by {dn_['lo']:+.2f} SD (page 3).")
        for s in kf:
            f.text(0.08, y, "•", size=9, va="top", color=INK); y = para(f, 0.10, y, s, size=8.4, n=106) - 0.008
        y -= 0.008
        bx = f.add_axes([0.07, 0.05, 0.86, y - 0.055]); bx.set_xticks([]); bx.set_yticks([])
        [s_.set_color(GRID) for s_ in bx.spines.values()]
        enc = d["enc"]; bands_e = {e: ("low" if p_ < 50 else "medium" if p_ < 80 else "high") for e, p_ in enc.items()}
        agree = len(set(bands_e.values())) == 1
        dist = min(abs(P - 50), abs(P - 80)); near = 50 if abs(P - 50) < abs(P - 80) else 80
        qc = [f"Tissue-mask area: about {info['mm2']:.0f} mm². Analysed input: {d['n']:,} sampled fields of 112 x 112 µm on a fixed grid, "
              f"about {d['n'] * 0.012544:.0f} mm².",
              f"Fields excluded before analysis as mostly glass, lumen or fat: {info['filt']:.0f}% of {info['n_grid']:,} grid fields.",
              "Encoder results (each against its own development reference): " + ", ".join(
                  f"{nm} {enc[e]:.0f}th ({bands_e[e]})" for e, nm in (("midnight", "Midnight"), ("uni2h", "UNI2-h"), ("phikon2", "Phikon-v2"))) +
              (". All three fall in the same band." if agree else ". They do not all fall in the same band."),
              f"The case percentile is about {dist:.0f} points from the {near}th-percentile band boundary.",
              "Encoder agreement and boundary distance describe consistency within this model; they are not a calibrated confidence estimate."]
        bx.text(0.015, 0.93, "Sampling, image QC and model agreement", size=9, weight="bold", color=INK, transform=bx.transAxes, va="top")
        yy = 0.80
        for s in qc:
            t = W_(s, 118); bx.text(0.015, yy, t, size=7.6, color=INK if not s.startswith("Encoder agreement") else MUTED, va="top", transform=bx.transAxes)
            yy -= 0.13 * (t.count("\n") + 1) * (0.18 / max(y - 0.055, 0.1))
        pdf.savefig(f); plt.close(f)

        # ---------------- page 2: field review cards
        f = plt.figure(figsize=(8.27, 11.69))
        frame(f, "Inspect the fields selected first by sequential knockout", 2,
              W_("Cards are in sequential-knockout order, which recomputes every effect after each removal. The score change on each card is "
                 "for excluding that field alone from the complete input. Left: 448 µm context with the field outlined; right: the 112 µm "
                 "field as the model read it. Coordinates are in mm from the top-left corner of the scanned image.", 118))
        for k, fd in enumerate(d["up"]):
            r_, c_ = divmod(k, 2); x0 = 0.06 + c_ * 0.46; sq = 0.2 * 8.27 / 11.69; yb = 0.715 - r_ * 0.272
            ctx, fld = crops["up"][k]
            a = f.add_axes([x0, yb, 0.2, sq]); a.imshow(ctx); a.axis("off"); h, w_ = ctx.shape[:2]
            a.add_patch(plt.Rectangle((w_ * 0.375, h * 0.375), w_ * 0.25, h * 0.25, fill=False, ec=RED, lw=1.1))
            px100 = w_ / 448 * 100; a.plot([w_ * 0.05, w_ * 0.05 + px100], [h * 0.93, h * 0.93], color="k", lw=2); a.text(w_ * 0.05, h * 0.89, "100 µm", size=5.5)
            b = f.add_axes([x0 + 0.21, yb, 0.2, sq]); b.imshow(fld); b.axis("off")
            f.text(x0, yb + sq + 0.006, f"Field {k + 1}   x {fd['mm'][0]:.1f} mm, y {fd['mm'][1]:.1f} mm", size=8.3, weight="bold", color=RED)
            yy = para(f, x0, yb - 0.006, f"Computed model response: excluding this field alone changes the score by {fd['chg']:+.3f} SD.", size=7.1, n=60)
            if fd["desc"]:
                yy = para(f, x0, yy - 0.002, "Automated image estimate: " + fd["desc"] + ".", size=6.9, n=62, color=MUTED, lh=0.0118)
            para(f, x0, yy - 0.002, "Review: confirm tissue identity and assess artefact.", size=6.9, n=62, color=ACC, lh=0.0118)
        pdf.savefig(f); plt.close(f)

        # ---------------- page 3: exclusion tests
        f = plt.figure(figsize=(8.27, 11.69))
        frame(f, "How the score changes when selected fields are excluded", 3,
              "Score change = score after exclusion minus original score, in development-reference SD. Negative values mean a lower model score.")
        y = 0.915
        if d.get("comp_rows"):
            f.text(0.07, y, "Excluding each estimated tissue compartment", size=10, weight="bold", color=INK); y -= 0.02
            rows = d["comp_rows"]; ax = f.add_axes([0.32, y - 0.035 * len(rows), 0.55, 0.035 * len(rows) - 0.005])
            vals = [r[3] for r in rows]; ax.barh(range(len(rows)), vals, color=[RED if v < 0 else BLUE for v in vals], height=0.55)
            ax.axvline(0, color=INK, lw=0.6); ax.set_yticks(range(len(rows)))
            ax.set_yticklabels([f"{r[0]}  ({r[1]:,} fields, {r[2]:.0f}%)" for r in rows], size=7.2); ax.invert_yaxis()
            m_ = max(0.1, max(abs(v) for v in vals)) * 1.25; ax.set_xlim(-m_, m_); ax.tick_params(labelsize=6.5)
            for i_, v in enumerate(vals):
                ax.text(v + (0.02 * m_ if v >= 0 else -0.02 * m_), i_, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", size=6.8, color=INK)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            y -= 0.035 * len(rows) + 0.015
            y = para(f, 0.07, y, "Each test excludes one group of sampled fields and recomputes the score, so the model's attention over the remaining "
                     "fields changes. The effects overlap and are not shares of the score. Compartments are automated estimates; in a separate "
                     "reviewed development sample (40 fields), 93% of fields in the epithelium-enriched cluster were epithelium and 65% carcinoma, "
                     "which describes that sample, not the chance that a field here is carcinoma.", size=7.6, n=118, color=MUTED) - 0.02
        f.text(0.07, y, "Nuclear density within epithelium-enriched fields", size=10, weight="bold", color=INK); y -= 0.02
        dn_ = d.get("dens", {})
        if "hi" in dn_:
            ax = f.add_axes([0.10, y - 0.13, 0.30, 0.12])
            ax.bar([0, 1], [dn_["hi"], dn_["lo"]], color=[RED, "#9a96a8"], width=0.6); ax.axhline(0, color=INK, lw=0.6)
            ax.set_xticks([0, 1]); ax.set_xticklabels([f"densest {dn_['k']:,}", f"least dense {dn_['k']:,}"], size=6.8)
            ax.set_ylabel("score change (SD)", size=6.8); ax.tick_params(labelsize=6.3)
            for i_, v in enumerate([dn_["hi"], dn_["lo"]]):
                ax.text(i_, v, f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", size=6.8)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            diff = dn_["lo"] - dn_["hi"]
            ax2 = f.add_axes([0.52, y - 0.13, 0.38, 0.12]); bins = np.asarray(REF["dens_bins"]); ctr = (bins[1:] + bins[:-1]) / 2
            for ds, col, nm in (("owl", "#7a8fb5", "Radboudumc"), ("tcga", "#c08a3e", "TCGA")):
                hh = np.asarray(REF["dens_hist"][ds], float); ax2.plot(ctr, hh / hh.sum(), color=col, lw=1.3, label=nm)
            ax2.axvline(diff, color=INK, lw=1.6); ax2.text(diff, ax2.get_ylim()[1] * 0.95, " this case", size=6.6, va="top")
            ax2.set_xlabel("difference between the two tests (SD)", size=6.8); ax2.set_yticks([]); ax2.tick_params(labelsize=6.3)
            ax2.legend(frameon=False, fontsize=6.3); [ax2.spines[s_].set_visible(False) for s_ in ("top", "right", "left")]
            ax2.set_title("development patients with enough of this tissue", size=6.8)
            y -= 0.16
            y = para(f, 0.07, y, f"Of {dn_['n']:,} epithelium-enriched fields, excluding the {dn_['k']:,} with the highest estimated nuclear density "
                     f"(about {dn_['d_hi']:,.0f} nuclei per mm²) changes the score by {dn_['hi']:+.2f} SD; excluding the {dn_['k']:,} with the lowest "
                     f"(about {dn_['d_lo']:,.0f} per mm²) changes it by {dn_['lo']:+.2f} SD, a difference of {diff:.2f} SD. In development patients "
                     "eligible for this analysis (371 Radboudumc, 280 TCGA), excluding the densest quarter lowered the score more than excluding the "
                     "least dense quarter in 98% and 96%; the mean differences were 0.22 and 0.18 SD. The two groups also differ in features other "
                     "than density, so this test shows what the score is sensitive to, not a cause. Prior nuclear-morphometry research gives biological "
                     "context [8]; it does not validate this measurement or case.", size=7.9, n=116) - 0.02
        else:
            y = para(f, 0.07, y, f"Not computed: only {dn_.get('n', 0)} epithelium-enriched fields were analysed (at least 40 are needed).") - 0.02
        f.text(0.07, y, "Fields whose exclusion raises the score most", size=10, weight="bold", color=BLUE); y -= 0.015
        for k, fd in enumerate(d["dn"]):
            x0 = 0.07 + k * 0.30; ctx, fld = crops["dn"][k]; sq = 0.14 * 8.27 / 11.69
            b = f.add_axes([x0, y - sq - 0.018, 0.14, sq]); b.imshow(fld); b.axis("off")
            f.text(x0, y - 0.012, f"{'abc'[k]}   x {fd['mm'][0]:.1f}, y {fd['mm'][1]:.1f} mm", size=7.6, weight="bold", color=BLUE)
            yy = para(f, x0, y - sq - 0.024, f"Excluding it changes the score by {fd['chg']:+.3f} SD.", size=6.9, n=38)
            if fd["desc"]:
                para(f, x0, yy - 0.002, "Automated estimate: " + fd["desc"] + ".", size=6.6, n=40, color=MUTED, lh=0.0112)
        if d["up"] and d["dn"] and max(abs(x["chg"]) for x in d["dn"]) < 0.5 * max(abs(x["chg"]) for x in d["up"]):
            f.text(0.07, 0.06, "These single-field effects are much smaller than those of the fields on page 2.", size=7.4, color=MUTED)
        pdf.savefig(f); plt.close(f)

        # ---------------- page 4: development outcomes by band
        f = plt.figure(figsize=(8.27, 11.69))
        frame(f, "Observed outcomes in the development cohorts", 4,
              W_("Biochemical-recurrence-free curves (Kaplan-Meier, 95% confidence bands) for development patients grouped by band using out-of-fold "
                 "scores, that is, each patient scored by models that did not train on that patient. Band cut-offs are fixed percentiles, not tuned "
                 f"to outcomes. This case's band ({band}) is drawn heavier.", 118))
        for j_, (ds, nm, fu) in enumerate((("owl", "Radboudumc", "508 patients, 87 events, median follow-up 5.1 years"),
                                            ("tcga", "TCGA-PRAD", "298 patients, 62 events, median follow-up 2.1 years"))):
            K = REF["km"][ds]; g_ = np.asarray(K["grid"]); y_top = 0.88 - j_ * 0.41
            ax = f.add_axes([0.12, y_top - 0.27, 0.80, 0.26])
            for b in ("low", "medium", "high"):
                B = K["bands"][b]; lw = 2.4 if b == band else 1.2
                ax.step(g_, B["s"], where="post", color=BANDC[b], lw=lw, label=f"{b} ({B['n']} patients, {B['events']} events)")
                ax.fill_between(g_, B["lo"], B["hi"], step="post", color=BANDC[b], alpha=0.12 if b == band else 0.06, lw=0)
            ax.set_ylim(0, 1.02); ax.set_xlim(0, K["tmax"]); ax.set_ylabel("recurrence-free", size=7.5); ax.set_xlabel("years of follow-up", size=7.5)
            ax.grid(True, color=GRID, lw=0.5); ax.tick_params(labelsize=7); ax.legend(frameon=False, fontsize=7, loc="lower left")
            [ax.spines[s_].set_visible(False) for s_ in ("top", "right")]
            ax.set_title(f"{nm}: {fu}", size=8.5, loc="left", color=INK)
            ty = y_top - 0.30
            f.text(0.03, ty, "At risk", size=6.8, color=MUTED, weight="bold")
            for b_i, b in enumerate(("low", "medium", "high")):
                f.text(0.03, ty - 0.013 * (b_i + 1), b, size=6.6, color=BANDC[b])
                for tk, nr in zip(K["ticks"], K["bands"][b]["at_risk"]):
                    f.text(0.12 + 0.80 * tk / K["tmax"], ty - 0.013 * (b_i + 1), str(nr), size=6.6, ha="center", color=INK)
        para(f, 0.07, 0.085, "These curves describe what happened to development patients in each band. They are not this patient's predicted "
             "recurrence-free probability, and band separation in development data does not establish performance in new patients or centres. "
             "Recurrence events and follow-up are as defined and curated by the challenge organisers; curves stop where fewer than 10 patients "
             "in any band remain at risk.", size=7.5, n=120, color=MUTED)
        pdf.savefig(f); plt.close(f)

        # ---------------- page 5: how it was produced, how to question it
        f = plt.figure(figsize=(8.27, 11.69))
        frame(f, "How this result was produced, and how to question it", 5)
        blocks = [("Workflow",
                   "H&E slide → 112 µm fields sampled on a fixed grid inside the tissue mask → three frozen pathology foundation models "
                   "(Midnight-12K, UNI2-h, Phikon-v2) [4-6] → attention-based multiple-instance models [1] trained on recurrence outcomes with the "
                   "Cox partial likelihood [2] → risk score (mean of the three encoders' standardised scores, 15 models in all) → percentile in the "
                   "development reference → exclusion tests. Inputs are H&E image fields only; training targets were recurrence outcomes."),
                  ("What an exclusion test computes",
                   "Each model scores a slide as an attention-weighted sum of field values. Excluding fields renormalises the attention of the rest, "
                   "and the new score is computed exactly for the fixed model and field representations. Exact computation does not establish that "
                   "tissue labels, outcome predictions or biological interpretations are correct."),
                  ("Automated image estimates",
                   "Tissue compartments and nuclear measurements are predicted from the Midnight field representation by linear read-outs trained on "
                   "202 development slides and validated with 5-fold cross-validation grouped by slide: compartment accuracy 0.92; variance explained "
                   "0.94 for nuclear density, 0.86 for nuclear size, 0.93 for nuclear staining and 0.90 for crowding. Reference nuclei came from a "
                   "network trained to reproduce Cellpose-SAM [7], whose counts correlated with Cellpose-SAM at r = 0.97 on 100 reviewed fields."),
                  ("Development performance",
                   "Ranking quality, out-of-fold, by Harrell's C [3]: 0.71 (95% CI 0.64 to 0.77) at Radboudumc and 0.70 (0.64 to 0.77) in TCGA. "
                   "C is the share of comparable patient pairs ranked correctly, not an accuracy. Results on the challenge's hidden test centres are "
                   "pending; scanner, stain and site differences can shift deep models [9]."),
                  ("Review checklist",
                   "1. Confirm sampled-tissue coverage on page 1.  2. Inspect field 1 and the other selected fields for tissue identity and artefact.  "
                   "3. Check whether the automated labels and nuclear measurements fit the images.  4. Inspect the contrasting fields and any "
                   "encoder disagreement.  5. Record any discrepancy before interpreting the result. Inspecting selected fields cannot establish that "
                   "all analysed tissue is free of artefact.")]
        y = 0.92
        for h, s in blocks:
            f.text(0.07, y, h, size=9.8, weight="bold", color=INK); y = para(f, 0.07, y - 0.018, s, size=8.1, n=116) - 0.016
        f.text(0.07, y, "References", size=9.8, weight="bold", color=INK); y -= 0.018
        for k, r in enumerate(REFS):
            y = para(f, 0.07, y, f"{k + 1}. {r}", size=6.9, n=128, lh=0.0108) - 0.003
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
    pct = ref_percentile(z)
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
        info = dict(group=group, z=z, pct=pct, mpp0=mpp0, mm2=tissue.sum() * (s20 * mpp0 / 1000) ** 2, case=f"Case {Path(wsi).stem}",
                    n_grid=int(len(xy)), filt=100.0 * (1 - tissue.mean()))
        d = report_data(M, F["midnight"].float().cpu().numpy(), A_, V_, w_, c_single, order, kxy, s20, mpp0, per)
        rd = Reader(wsi, 4)

        def ctx(xy_l0):
            S = 4 * s20; lv = sl.get_best_level_for_downsample(S / 336); ds = sl.level_downsamples[lv]
            x0, y0 = int(xy_l0[0] + s20 / 2 - S / 2), int(xy_l0[1] + s20 / 2 - S / 2)
            r = sl.read_region((max(x0, 0), max(y0, 0)), lv, (int(np.ceil(S / ds)), int(np.ceil(S / ds))))
            bg = Image.new("RGB", r.size, (255, 255, 255)); bg.paste(r, mask=r.split()[3]); return np.asarray(bg.resize((336, 336)))
        crops = {k: [(ctx(kxy[fd["i"]]), rd.one(kxy[fd["i"]], s20)) for fd in d[k]] for k in ("up", "dn")}
        f = max(1, max(heat.shape) // 2048)
        write_report_v4(OUTPUT_PATH / "pathology-report.pdf", sl, info, d, crops, heat_view[::f, ::f])
    except Exception as e:                                # the report must never cost us the prediction
        import traceback; log(f"report failed ({type(e).__name__}: {e}); writing minimal PDF\n" + traceback.format_exc())
        write_min_pdf(OUTPUT_PATH / "pathology-report.pdf", group, pct)
    sl.close()
    log(f"case done in {time.time() - t0:.0f}s")
    return 0


def write_min_pdf(path, group, pct):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = plt.figure(figsize=(8.27, 11.69)); f.text(0.1, 0.9, f"Reference band: {group}; reference percentile {pct:.0f}. Research use only.", size=12)
    f.savefig(path, format="pdf"); plt.close(f)


if __name__ == "__main__":
    raise SystemExit(run(model=init_model()))
