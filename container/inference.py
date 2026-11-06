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
the attribution TIFF on the mask grid and a 5-page PDF. The attribution map is drawn on the patch grid of the
organisers' faithfulness test (256 px at 0.5 um/px from the mask origin): every patch that overlaps a field removed
by sequential knockout carries that field's rank (earliest 255 ... 130); tissue the model did not read takes 3 ... 129
from the order spread from its nearest read field (heat_rank), so it always ranks below every read field; patches that
touch only fields whose removal would raise the score 1; outside the mask 0 (heat_patches). The report shows the
spread order itself, which is easier to read at slide scale.
Faithfulness notebook: get_args_parser() and run(args) load the model on each call (see run()).
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
DIRS = {}                                                   # folder overrides for run(args) (faithfulness notebook)
REPORT = True                                               # False: no PDF (prediction and map unchanged)
ENSEMBLE = "15 fold models, 5 per image model"            # how the score is combined (printed in the report)
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
MPP_OK = (0.08, 2.0)                                       # plausible slide pixel sizes (um); outside: tag is wrong


def tiff_mpp(path):
    import tifffile
    with tifffile.TiffFile(path) as t:
        p = t.pages[0]; xr = p.tags["XResolution"].value; unit = int(p.tags["ResolutionUnit"].value)
        ppu = xr[0] / xr[1]; per = {3: 1e4, 2: 25400.0}.get(unit)
        if per is None or ppu <= 1:
            raise ValueError(f"no usable resolution tag (unit={unit}, ppu={ppu})")
        return per / ppu


def slide_mpp(sl, path, mask_spacing=None, mscale=None):
    """Slide pixel size in um: the TIFF resolution tag, then openslide's mpp, then the mask's own spacing divided by
    the slide-to-mask scale, then 0.25; each only if it lies in MPP_OK."""
    cands = []
    try:
        cands.append(("tiff tag", tiff_mpp(path)))
    except Exception:
        pass
    try:
        cands.append(("openslide mpp", float(sl.properties.get("openslide.mpp-x"))))
    except Exception:
        pass
    if mask_spacing and mscale:
        cands.append(("mask spacing", mask_spacing / mscale))
    for src, v in cands:
        if MPP_OK[0] <= v <= MPP_OK[1]:
            if src != "tiff tag":
                log(f"WARNING: slide pixel size from {src}: {v:.4f} um")
            return float(v)
    log(f"WARNING: no plausible pixel size ({cands}); assuming 0.25 um/px"); return 0.25


def grid(W, H, size0, mask, mscale, thr=GRID_THR):
    xs = np.arange(0, W - size0 + 1, size0); ys = np.arange(0, H - size0 + 1, size0)
    gx, gy = np.meshgrid(xs, ys); c = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.int64)
    dt = np.int32 if mask.size < 2**31 else np.int64         # integral image of the mask (exact, a quarter of float64 memory)
    S = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dt); S[1:, 1:] = mask.cumsum(0, dtype=dt).cumsum(1, dtype=dt)
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


def _read_mha(path):
    """Minimal MetaImage reader (local data, raw or zlib): returns the 2-D array."""
    import zlib
    raw = open(path, "rb").read(); i = raw.index(b"ElementDataFile"); j = raw.index(b"\n", i) + 1
    hdr = dict(l.split("=", 1) for l in raw[:j].decode("latin-1").splitlines() if "=" in l); hdr = {k.strip(): v.strip() for k, v in hdr.items()}
    assert hdr.get("ElementDataFile", "LOCAL") == "LOCAL", "detached MetaImage data"
    dims = [int(v) for v in hdr["DimSize"].split()]
    dt = {"MET_UCHAR": np.uint8, "MET_CHAR": np.int8, "MET_USHORT": np.uint16, "MET_SHORT": np.int16, "MET_UINT": np.uint32,
          "MET_INT": np.int32, "MET_FLOAT": np.float32, "MET_DOUBLE": np.float64}[hdr["ElementType"]]
    data = raw[j:]
    if hdr.get("CompressedData", "False").lower() == "true":
        data = zlib.decompress(data)
    ch = int(hdr.get("ElementNumberOfChannels", 1))
    a = np.frombuffer(data, dt)[: int(np.prod(dims)) * ch].reshape(list(reversed(dims)) + ([ch] if ch > 1 else []))
    while a.ndim > 2 and a.shape[0] == 1:
        a = a[0]
    return a[..., 0] if a.ndim == 3 else a


def read_mask(path):
    """Tissue mask at full resolution as bool, plus its pixel spacing in um (None if unknown). Tissue is value 1 where
    the mask uses 1, as in the organisers' faithfulness notebook, otherwise any value > 0. TIFF page 0 is read with
    tifffile (any tiling, bit depth or photometric), .mha with a small MetaImage reader, anything else with openslide."""
    spacing = None
    if str(path).lower().endswith(".mha"):
        m = _read_mha(path)
    else:
        try:
            import tifffile
            with tifffile.TiffFile(path) as t:
                m = t.pages[0].asarray()
            try:
                spacing = tiff_mpp(path)
            except Exception:
                pass
        except Exception as e:
            log(f"mask: tifffile failed ({type(e).__name__}: {e}); reading with openslide")
            import openslide
            ms = openslide.OpenSlide(path)
            m = np.asarray(ms.read_region((0, 0), 0, ms.level_dimensions[0]))[..., 0]; ms.close()
    if m.ndim == 3:
        m = m[..., 0]
    one = m == 1
    return (one if one.any() else m > 0), spacing


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
        except Exception:                                     # a bad tile is skipped, never fatal
            try:
                del self.local.s
            except Exception:
                pass
            return None
        bg = Image.new("RGB", r.size, (255, 255, 255)); bg.paste(r, mask=r.split()[3])
        return np.asarray(bg.resize((TILE_PX, TILE_PX), Image.BILINEAR))


def ink_fraction(tiles):
    """Share of tissue pixels that look like marking ink: green, blue or yellow at clear saturation, or near-black.
    Calibrated on 162 report fields (ink 0.71 to 1.00; tissue, dark TCGA carcinoma included, at most 0.004)."""
    a = np.stack(tiles)[:, ::4, ::4].astype(np.float32) / 255; r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = a.max(-1), a.min(-1); d = mx - mn + 1e-6
    h = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    sat = (mx - mn) / (mx + 1e-6); tissue = (mx < 0.92) | (sat > 0.12); n = np.maximum(tissue.sum((1, 2)), 1)
    col = (((h >= 70) & (h <= 200) & (sat > 0.20)) | ((h > 200) & (h <= 235) & (sat > 0.30)) | ((h >= 35) & (h <= 65) & (sat > 0.30))) & tissue
    blk = (mx < 0.18) & tissue
    return np.maximum(col.sum((1, 2)) / n, blk.sum((1, 2)) / n / 2)      # >= 0.25: colour >= 25% or black >= 50%


INK_THR = 0.25


@torch.inference_mode()
def embed(M, wsi, coords, size, bs=64, fn=None, stats=None):
    rd = Reader(wsi, threads=min(16, os.cpu_count() or 4)); q = queue.Queue(maxsize=4)

    err = []

    def prod():
        try:
            for i in range(0, len(coords), bs):
                q.put(list(rd.pool.map(lambda c: rd.one(c, size), coords[i:i + bs])))
        except BaseException as e:                            # never leave the consumer waiting
            err.append(e)
        finally:
            q.put(None)
    threading.Thread(target=prod, daemon=True).start()
    out, ok = [], []
    while (b := q.get()) is not None:
        good = [t is not None for t in b]; ok += good
        if any(good):
            if stats is not None:                             # report-only colour check, aligned with the tiles kept
                stats.extend(ink_fraction([t for t in b if t is not None]).tolist())
            x = torch.from_numpy(np.stack([t for t in b if t is not None])).to(M["dev"]).permute(0, 3, 1, 2).float().div_(255)
            out.append((fn or f_midnight)(M, x))
    rd.pool.shutdown()
    if err:
        raise err[0]
    if not out:
        return None, np.array(ok + [False] * (len(coords) - len(ok)), bool)
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


def faith_patch_px(sl, W, mask_w, mpp0, size=256, spacing=0.5):
    """The faithfulness notebook's patch (mask_tile_size: `size` px at the slide level closest to `spacing` um/px) in
    mask pixels; None if it is not a whole number (the notebook then stops for that slide)."""
    ds = [W / lw for lw, _ in sl.level_dimensions]
    lev = int(np.argmin([abs(d - spacing / mpp0) for d in ds]))
    t = size * int(round(spacing / (mpp0 * ds[lev]))) * ds[lev] * mask_w / W
    return int(round(t)) if t >= 1 and abs(t - round(t)) < 0.01 else None


def heat_patches(c, order, kxy, s20, mscale, mask, tile, view=None):
    """uint8 map on the faithfulness patch grid (patches of `tile` mask px from (0, 0)). The fields that sequential
    knockout removes (c > 0) are taken in order and each patch their mask window overlaps gets the next rank
    (255 ... 130, earliest first), so removing the top patches removes those fields whole and in knockout order (a
    field leaves the bag once < 25% of its window is tissue). Tissue the model did not read (removing it changes
    nothing) takes 3 ... 129 from `view`, the order spread from the nearest read field, so the map reads as a
    continuous picture without changing what removing the top patches does; patches touching only fields whose
    removal would raise the score get 1 (removed last); outside the mask 0."""
    h, w = mask.shape; gh, gw = -(-h // tile), -(-w // tile)
    x0 = np.clip((kxy[:, 0] / mscale).astype(int), 0, w); y0 = np.clip((kxy[:, 1] / mscale).astype(int), 0, h)
    x1 = np.clip(((kxy[:, 0] + s20) / mscale).astype(int), 0, w); y1 = np.clip(((kxy[:, 1] + s20) / mscale).astype(int), 0, h)
    x1 = np.maximum(x1, np.minimum(x0 + 1, w)); y1 = np.maximum(y1, np.minimum(y0 + 1, h))
    r0, r1, q0, q1 = y0 // tile, (y1 - 1) // tile, x0 // tile, (x1 - 1) // tile
    G = np.full((gh, gw), 2, np.uint8)
    if view is not None:                                    # unread tissue: the spread order, below every read field
        rr_, qq_ = np.minimum(np.arange(gh) * tile + tile // 2, h - 1), np.minimum(np.arange(gw) * tile + tile // 2, w - 1)
        Vc = view[rr_][:, qq_].astype(np.int32)
        G = np.clip(3 + (126 * np.maximum(Vc - 1, 0)) // 254, 3, 129).astype(np.uint8)
    for t in order:
        if c[t] <= 0:
            G[r0[t]:r1[t] + 1, q0[t]:q1[t] + 1] = 1
    seen = np.zeros((gh, gw), bool); seq = []
    for t in order:
        if c[t] > 0:
            for r in range(r0[t], r1[t] + 1):
                for q in range(q0[t], q1[t] + 1):
                    if not seen[r, q]:
                        seen[r, q] = True; seq.append((r, q))
    if seq:
        rr, qq = np.array(seq).T
        G[rr, qq] = (255 - np.floor(126 * np.arange(len(seq)) / len(seq))).astype(np.uint8)
    out = np.repeat(np.repeat(G, tile, 0), tile, 1)[:h, :w]; out[~mask] = 0
    return out


REF_Q = [-1.6560, -1.4729, -1.4002, -1.3542, -1.2884, -1.2465, -1.1966, -1.1535, -1.1287, -1.1096, -1.0577, -1.0303, -1.0032, -0.9781, -0.9542, -0.9320, -0.9121, -0.8915, -0.8629, -0.8318, -0.8121, -0.7777, -0.7696, -0.7549, -0.7277, -0.7033, -0.6935, -0.6639, -0.6513, -0.6164, -0.5941, -0.5755, -0.5538, -0.5343, -0.5064, -0.4809, -0.4697, -0.4420, -0.4130, -0.3978, -0.3818, -0.3661, -0.3439, -0.3265, -0.3029, -0.2850, -0.2585, -0.2302, -0.2042, -0.1936, -0.1777, -0.1489, -0.1369, -0.1072, -0.0819, -0.0619, -0.0292, -0.0189, 0.0124, 0.0422, 0.0648, 0.0945, 0.1217, 0.1423, 0.1782, 0.2110, 0.2389, 0.2879, 0.3340, 0.3683, 0.4074, 0.4502, 0.4868, 0.5178, 0.5536, 0.6097, 0.6507, 0.6864, 0.7147, 0.7615, 0.8163, 0.8661, 0.9317, 0.9972, 1.0664, 1.1253, 1.1786, 1.2361, 1.2890, 1.3345, 1.3734, 1.4475, 1.5121, 1.5667, 1.6180, 1.6927, 1.8097, 1.9148, 2.0418, 2.1693, 3.0853]


# ------------------------------------------------------------------ report (pathologist- and oncologist-facing)
# Three kinds of statement, labelled on the page: computed model response (exact score change when fields are excluded),
# automated image estimate (model-generated tissue label or measurement), development outcome evidence (earlier patients).
# Sign convention everywhere: score change = score after exclusion - original score (negative = lower score).
try:
    REF = json.load(open(Path(__file__).with_name("report_ref.json")))
except Exception:                                           # report reference missing: the report falls back, the prediction does not need it
    REF = {}
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


def ordinal(x):
    """93 -> '93rd'; rounds first, so the printed number and its suffix always agree."""
    n = int(round(float(x))); suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def pct_ord(x):
    """A percentile for display: 1st to 99th (a field or case cannot sit at the 0th or 100th of a reference)."""
    return ordinal(min(99, max(1, int(round(float(x))))))


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


def report_data(M, Xm, A, V, w, c_single, order, kxy, s20, mpp0, per, ink=None):
    sd = M["cal"]["z_sd"]; n = len(kxy); comp, nuc = concept_readouts(M, Xm); d = dict(n=n, sd=sd, comp=comp)
    up = [int(i) for i in order[:6]]; dn = [int(i) for i in np.argsort(c_single, kind="stable") if c_single[i] < 0][:3]
    pct = None if nuc is None else {k: (np.argsort(np.argsort(nuc[:, k], kind="stable"), kind="stable") + 0.5) / n * 100 for k, _ in NUC_SHOW}

    def describe(i):
        if comp is None:
            return ""
        parts = [f"{name} near the {pct_ord(pct[k][i])} percentile of this slide's sampled fields"
                 for k, name in sorted(NUC_SHOW, key=lambda t: -abs(pct[t[0]][i] - 50))[:2]]
        return f"{COMP_LABEL[comp[i]].lower()}; " + "; ".join(parts)
    field = lambda i: dict(i=i, mm=(kxy[i] + s20 / 2) * mpp0 / 1000, chg=-float(c_single[i]) / sd, desc=describe(i),
                           comp=None if comp is None else int(comp[i]),
                           nuc=None if pct is None else {name: float(pct[k][i]) for k, name in NUC_SHOW},
                           ink=bool(ink is not None and ink[i] >= INK_THR))
    d["up"], d["dn"] = [field(i) for i in up], [field(i) for i in dn]
    sel_ = lambda ids: np.isin(np.arange(n), ids)
    d["up6"] = excl_change(A, V, w, sel_(up)) / sd              # the listed fields removed together
    if ink is not None:                                         # fields that look like marking ink, and the result without them
        fl = np.asarray(ink) >= INK_THR
        d["ink"] = dict(n=int(fl.sum()), share=float(fl.mean()), chg=(excl_change(A, V, w, fl) / sd) if 0 < fl.sum() < n else None)
    d["top"] = [(fr, int(max(1, round(fr * n))), excl_change(A, V, w, sel_(order[:max(1, round(fr * n))])) / sd) for fr in (0.01, 0.05, 0.10)]
    d["enc"] = {e: pct_of(per[e], REF["enc_q"][e]) for e in per}
    if comp is not None:                                  # slide-level morphology measures, as in the development reference (E55)
        bio = dict(epi_share=float((comp == 0).mean()), stroma_share=float((comp == 2).mean()), n_epi=int((comp == 0).sum()))
        if bio["n_epi"] >= 40:
            for k, nm in enumerate(("n_dens", "area_mu", "area_cv", "hema_mu", "deg20", "small_dark")):
                bio[nm] = float(np.median(nuc[comp == 0, k]))
        d["bio"] = bio
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


# report v5 colours and units
INK, MUTED, LINE, PANEL, HEAD, NAVY = "#1b2232", "#5a6273", "#cdd3dc", "#f4f6f9", "#e7ebf1", "#17325c"
BAND_C = {"low": "#2c7a5e", "medium": "#b8740f", "high": "#ad2c45"}
OK_C, WARN_C, NA_C = "#2c7a5e", "#b8740f", "#8a92a1"
RED, BLUE = "#ad2c45", "#2a5fa3"
PT = 1 / 842                                              # one point in figure height units (A4)


def write_report_v5(path, info, d, crops, heat_small, thumb, W0, ensemble, K):
    """Morphology report laid out like a genomic (NGS) test report: page 1 report highlights (specimen and test, the
    result, morphology findings against development reference ranges with their effect on the result, interpretation,
    pathologist verification, quality metrics); page 2 evidence regions; page 3 findings in detail; page 4 outcomes by
    band in the development cohorts; page 5 methods, quality definitions, limitations and references."""
    import datetime, textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
    REF, REFS, COMP_LABEL, ordinal, ref_pct = K["REF"], K["REFS"], K["COMP_LABEL"], K["ordinal"], K["ref_percentile"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    cm = LinearSegmentedColormap.from_list("rb", ["#2a5fa3", "#f5f5f5", "#ad2c45"])
    band, P, z, sd = info["group"], float(info["pct"]), float(info["z"]), float(d["sd"])
    case = info.get("case", "").replace("Case ", "")
    po = lambda x: ordinal(min(99, max(1, int(round(float(x))))))
    Pd, Pn = po(P), min(99, max(1, int(round(P))))
    band_of = lambda p: "low" if p < 50 else "medium" if p < 80 else "high"
    after = lambda dz_sd: ref_pct(z + dz_sd * sd)              # percentile after an exclusion (change in SD)
    fx = thumb.shape[1] / W0
    hs = np.asarray(Image.fromarray(heat_small).resize((thumb.shape[1], thumb.shape[0]), Image.NEAREST)).astype(float)
    over = np.ma.masked_where(hs == 0, (hs - 128) / 127)
    BIO = REF.get("bio", {}).get("measures", {}); bio = d.get("bio", {})
    ref_of = lambda nm, v: float(np.interp(v, BIO[nm]["q"], np.linspace(0, 100, len(BIO[nm]["q"])))) if nm in BIO and v is not None else None

    def wrap(s, w, size):
        return textwrap.fill(s, max(18, int(w * 595 / (0.54 * size))))

    def text(f, x, y, s, size=8.0, w=0.86, color=INK, weight="normal", lh=1.36, ha="left"):
        t = wrap(s, w, size); f.text(x, y, t, size=size, va="top", ha=ha, color=color, weight=weight, linespacing=lh)
        return y - (t.count("\n") + 1) * size * lh * PT - 0.003

    def rbox(f, x, y0, w, h, fc="white", ec=LINE, lw=0.8):
        f.add_artist(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0,rounding_size=0.005", transform=f.transFigure, fc=fc, ec=ec, lw=lw, zorder=-2))

    def section(f, x, y_top, w, h, title, accent=NAVY):
        rbox(f, x, y_top - h, w, h)
        f.add_artist(Rectangle((x, y_top - 0.022), w, 0.022, transform=f.transFigure, fc=HEAD, ec="none", zorder=-1))
        f.add_artist(Rectangle((x, y_top - 0.022), 0.004, 0.022, transform=f.transFigure, fc=accent, ec="none", zorder=-1))
        f.text(x + 0.012, y_top - 0.011, title.upper(), size=7.4, weight="bold", color=accent, va="center")
        return y_top - 0.032

    def chip(f, x, y, label, color, size=6.8):
        wdt = 0.014 + 0.0058 * len(label) * size / 6.8
        f.add_artist(FancyBboxPatch((x, y - 0.0075), wdt, 0.015, boxstyle="round,pad=0,rounding_size=0.004", transform=f.transFigure, fc=color, ec="none"))
        f.text(x + wdt / 2, y, label, size=size, color="white", weight="bold", va="center", ha="center")

    def checkbox(f, x, y, label, size=7.0):
        sw = 0.0095; shh = sw * 8.27 / 11.69                  # square on the page
        f.add_artist(Rectangle((x, y - shh / 2), sw, shh, transform=f.transFigure, fc="white", ec=INK, lw=0.7))
        f.text(x + sw + 0.006, y, label, size=size, va="center", color=INK)
        return x + sw + 0.006 + len(label) * size * 0.56 / 595 + 0.022

    def header(f, title, page, sub=None):
        f.add_artist(Rectangle((0, 0.952), 1, 0.048, transform=f.transFigure, fc=NAVY, ec="none"))
        f.text(0.06, 0.981, "KNOX  MORPHOLOGY SEQUENCING REPORT", size=11, weight="bold", color="white", va="center")
        f.text(0.06, 0.962, "Recurrence-risk ranking from radical prostatectomy H&E", size=7.6, color="#c9d6ea", va="center")
        f.text(0.94, 0.976, "RESEARCH USE ONLY", size=7.8, weight="bold", color="#ffd2da", va="center", ha="right")
        f.text(0.94, 0.961, f"Specimen {case}", size=7.2, color="#c9d6ea", va="center", ha="right")
        f.text(0.06, 0.930, title, size=12.5, weight="bold", color=INK, va="center")
        y = 0.914
        if sub:
            y = text(f, 0.06, 0.914, sub, size=7.5, w=0.88, color=MUTED)
        f.add_artist(plt.Line2D([0.06, 0.94], [0.036, 0.036], color=LINE, lw=0.8, transform=f.transFigure))
        f.text(0.06, 0.024, "KNOX 1.0 (development). Research use only: not for diagnosis or treatment decisions; clinical utility has not "
               "been established.", size=6.3, color=MUTED)
        f.text(0.94, 0.024, f"Page {page} of 5", size=6.3, color=MUTED, ha="right")
        return y - 0.006

    def marks(ax, fields, color, labels, ms=6):
        for lab, fd in zip(labels, fields):
            cx, cy = fd["mm"][0] * 1000 / info["mpp0"] * fx, fd["mm"][1] * 1000 / info["mpp0"] * fx
            ax.plot(cx, cy, "o", ms=ms, mfc="none", mec=color, mew=1.3); ax.text(cx + 10, cy - 10, lab, color=color, size=7.2, weight="bold")

    def refbar(f, x, y, w, nm, v, h=0.011):
        """Reference-range bar: development 10th to 90th percentile shaded, median tick, this slide's value marked."""
        q = BIO[nm]["q"]; lo, hi = q[0], q[100]; span = max(hi - lo, 1e-9); X = lambda t: x + w * (min(max(t, lo), hi) - lo) / span
        f.add_artist(Rectangle((x, y - h / 2), w, h, transform=f.transFigure, fc=PANEL, ec=LINE, lw=0.5))
        f.add_artist(Rectangle((X(q[10]), y - h / 2), X(q[90]) - X(q[10]), h, transform=f.transFigure, fc="#d9e1ee", ec="none"))
        f.add_artist(plt.Line2D([X(q[50])] * 2, [y - h / 2, y + h / 2], color=MUTED, lw=0.8, transform=f.transFigure))
        f.add_artist(plt.Line2D([X(v)] * 2, [y - h * 0.9, y + h * 0.9], color=INK, lw=2.0, transform=f.transFigure))

    # ---- numbers used on several pages
    enc = d["enc"]; bands_e = {e: band_of(p_) for e, p_ in enc.items()}; n_agree = sum(b_ == band for b_ in bands_e.values())
    dist = min(abs(P - 50), abs(P - 80)); near = 50 if abs(P - 50) < abs(P - 80) else 80
    km = REF["km"]["owl"]; g_ = np.asarray(km["grid"]); i5 = int(np.searchsorted(g_, 5.0, side="right") - 1)
    bcr5 = {b: 100 * (1 - km["bands"][b]["s"][i5]) for b in ("low", "medium", "high")}
    kmt = REF["km"]["tcga"]; gt = np.asarray(kmt["grid"]); i3 = int(np.searchsorted(gt, 3.0, side="right") - 1)
    bcr3t = {b: 100 * (1 - kmt["bands"][b]["s"][i3]) for b in ("low", "medium", "high")}
    up6 = after(d["up6"]) if "up6" in d else None
    top = [(fr, k_, after(v)) for fr, k_, v in d.get("top", [])]
    comp_after = {r[0]: (r[1], r[2], after(r[3])) for r in d.get("comp_rows", [])}
    EPI, STR = COMP_LABEL[0], COMP_LABEL[2]
    dn_ = d.get("dens", {})
    n_epi = bio.get("n_epi", 0)
    tissue_ok = "Adequate" if d["n"] >= 500 else "Limited" if d["n"] >= 100 else "Insufficient"
    epi_ok = "Adequate" if n_epi >= 40 else "Low"

    # morphology findings (rows: label, this slide, development reference percentile, effect on result)
    FIND = []
    if "epi_share" in bio:
        eff = f"removing it: {Pd} to {po(comp_after[EPI][2])}" if EPI in comp_after else "not tested"
        FIND.append(("Epithelium-enriched glands", f"{100 * bio['epi_share']:.0f}% of tissue read", ref_of("epi_share", bio["epi_share"]), eff, "epi_share"))
    if "n_dens" in bio:
        eff = (f"densest quarter removed: to {po(after(dn_['hi']))}; least dense: to {po(after(dn_['lo']))}") if "hi" in dn_ else "not tested"
        FIND.append(("Nuclear density in gland epithelium", f"{bio['n_dens']:,.0f} per mm²", ref_of("n_dens", bio["n_dens"]), eff, "n_dens"))
        FIND.append(("Nuclear size in gland epithelium", f"{bio['area_mu']:.0f} µm²", ref_of("area_mu", bio["area_mu"]), "", "area_mu"))
        FIND.append(("Chromatin density (haematoxylin)", f"{bio['hema_mu']:.2f} (a.u.)", ref_of("hema_mu", bio["hema_mu"]), "", "hema_mu"))
        FIND.append(("Nuclear crowding (neighbours in 20 µm)", f"{bio['deg20']:.1f}", ref_of("deg20", bio["deg20"]), "", "deg20"))
    if "stroma_share" in bio:
        eff = f"removing it: {Pd} to {po(comp_after[STR][2])}" if STR in comp_after else "not tested"
        FIND.append(("Stroma", f"{100 * bio['stroma_share']:.0f}% of tissue read", ref_of("stroma_share", bio["stroma_share"]), eff, "stroma_share"))

    # interpretation (rule-based, plain words)
    moves = sorted(((abs(v[2] - P), k_) for k_, v in comp_after.items()), reverse=True)
    interp = [f"This slide ranks in the {band} band, at the {Pd} percentile of 806 development patients."]
    if moves:
        interp.append(f"The ranking rests mainly on {moves[0][1].lower()}: without it the result would move from the {Pd} to the "
                      f"{po(comp_after[moves[0][1]][2])} percentile.")
    if "hi" in dn_:
        mh, ml = abs(after(dn_["hi"]) - P), abs(after(dn_["lo"]) - P)
        pts = lambda v: "less than 1 percentile point" if v < 1 else f"{v:.0f} percentile point{'s' if round(v) != 1 else ''}"
        interp.append(f"Within the gland epithelium, the ranking leans on fields with {'dense' if mh > ml else 'sparse'} nuclei: removing the "
                      f"densest quarter moves it by {pts(mh)}, removing the least dense quarter by {pts(ml)}.")
    n_ep = sum(1 for fd in d["up"][:6] if fd.get("comp") == 0); n_up = min(6, len(d["up"]))
    if d["up"]:
        interp.append(f"{'All ' + str(n_up) if n_ep == n_up else f'{n_ep} of the {n_up}'} evidence regions on page 2 "
                      f"{'is an epithelium-enriched gland' if n_ep == 1 and n_up > 1 else 'are epithelium-enriched glands'} by automated "
                      "estimate; their identity as carcinoma needs the pathologist's confirmation.")
    ink = d.get("ink") or {}
    if ink.get("n"):
        interp.append(f"{ink['n']:,} of the fields read {'looks' if ink['n'] == 1 else 'look'} like marking ink (automated colour check)"
                      + (f"; without them the result would be the {po(after(ink['chg']))} percentile ({band_of(after(ink['chg']))} band)."
                         if ink.get("chg") is not None else "."))
    interp.append(f"In the Radboudumc development cohort, {bcr5[band]:.0f}% of patients in the {band} band had biochemical recurrence within "
                  f"5 years (low {bcr5['low']:.0f}%, medium {bcr5['medium']:.0f}%, high {bcr5['high']:.0f}%).")

    with PdfPages(path) as pdf:
        # ============================================================ page 1: report highlights
        f = plt.figure(figsize=(8.27, 11.69))
        header(f, "Report highlights", 1)
        # specimen | test
        rbox(f, 0.06, 0.835, 0.43, 0.068); rbox(f, 0.51, 0.835, 0.43, 0.068)
        for x0, title, rows in ((0.06, "SPECIMEN", [("Specimen ID", case), ("Type", "Radical prostatectomy"), ("Material", "H&E whole-slide image"),
                                                    ("Scan", f"{info['mpp0']:.3f} µm per pixel")]),
                                (0.51, "TEST", [("Test", "KNOX 1.0 (development)"), ("Reference", "806 development patients"),
                                                ("Models", ensemble), ("Report date", datetime.date.today().isoformat())])):
            f.text(x0 + 0.012, 0.894, title, size=6.8, weight="bold", color=NAVY, va="top")
            for k_, (lab, val) in enumerate(rows):
                yy = 0.879 - k_ * 0.0105
                f.text(x0 + 0.012, yy, lab, size=6.8, color=MUTED, va="top"); f.text(x0 + 0.105, yy, wrap(val, 0.31, 6.8).split("\n")[0], size=6.9, color=INK, va="top")
        # result
        rbox(f, 0.06, 0.72, 0.88, 0.105, ec=BAND_C[band], lw=1.4)
        f.add_artist(Rectangle((0.06, 0.72), 0.24, 0.105, transform=f.transFigure, fc=BAND_C[band], ec="none", zorder=-1))
        f.text(0.18, 0.806, "RISK BAND", size=7.2, color="white", ha="center", va="center", weight="bold")
        f.text(0.18, 0.772, band.upper(), size=27, color="white", ha="center", va="center", weight="bold")
        f.text(0.18, 0.737, f"{Pd} percentile", size=9, color="white", ha="center", va="center", weight="bold")
        f.text(0.325, 0.810, "Morphology recurrence-risk ranking", size=9.2, weight="bold", color=INK, va="center")
        ax = f.add_axes([0.33, 0.770, 0.58, 0.018]); ax.set_xlim(0, 100); ax.set_ylim(0, 1); ax.set_yticks([])
        for lo_, hi_, b in ((0, 50, "low"), (50, 80, "medium"), (80, 100, "high")):
            ax.axvspan(lo_, hi_, color=BAND_C[b], alpha=0.9 if b == band else 0.22, lw=0)
            ax.text((lo_ + hi_) / 2, 1.5, b.capitalize(), ha="center", size=7, color=INK, weight="bold" if b == band else "normal")
        ax.plot([P, P], [-0.25, 1.25], color=INK, lw=2.2, clip_on=False)
        ax.set_xticks([0, 50, 80, 100]); ax.set_xticklabels(["0", "50th", "80th", "100th"], size=6.3); ax.tick_params(length=2)
        [s_.set_visible(False) for s_ in ax.spines.values()]
        text(f, 0.325, 0.750, f"Higher than about {Pn}% of 806 development patients. A ranking, not a probability of recurrence for this "
             "patient. Gleason grade, stage, margins and PSA were not used.", size=6.7, w=0.6, color=MUTED, lh=1.25)
        # morphology findings table
        RH = 0.0215; nrow = len(FIND); th = 0.032 + 0.038 + nrow * RH + 0.010
        y = section(f, 0.06, 0.705, 0.88, th, "Morphology findings")
        cols = [0.075, 0.34, 0.47, 0.70]
        for x_, h_ in zip(cols, ("Finding", "This slide", "Development reference", "Effect on the result")):
            f.text(x_, y, h_, size=6.8, color=MUTED, weight="bold", va="top")
        f.text(0.47, y - 0.012, "10th to 90th percentile shaded, median tick", size=5.8, color=MUTED, va="top")
        y -= 0.030
        for lab, val, pr, eff, nm in FIND:
            f.add_artist(plt.Line2D([0.07, 0.93], [y + 0.008, y + 0.008], color=LINE, lw=0.5, transform=f.transFigure))
            f.text(cols[0], y, lab, size=7.3, color=INK, va="top", weight="bold" if nm in ("epi_share", "n_dens") else "normal")
            f.text(cols[1], y, val, size=7.3, color=INK, va="top")
            if pr is not None:
                refbar(f, cols[2], y - 0.0045, 0.14, nm, bio[nm] if nm in bio else 0)
                f.text(cols[2] + 0.148, y, po(pr), size=6.8, color=INK, va="top")
            if eff:
                text(f, cols[3], y + 0.001, eff, size=6.6, w=0.23, color=INK, lh=1.2)
            y -= RH
        text(f, 0.075, y + 0.004, "Effect: where the result would move if that tissue were removed from the analysis (exact recomputation). "
             "Findings are automated estimates from the image; reference = the same measures on 806 development slides.", size=6.0, w=0.85, color=MUTED, lh=1.2)
        y_int = 0.705 - th - 0.007
        # interpretation
        ih = 0.030 + 0.0127 * sum(len(wrap(s, 0.82, 7.8).split("\n")) for s in interp) + 0.0045 * len(interp) + 0.005
        y = section(f, 0.06, y_int, 0.88, ih, "Interpretation")
        for s in interp:
            f.text(0.078, y, "•", size=8.5, va="top", color=INK); y = text(f, 0.093, y, s, size=7.8, w=0.82) - 0.003
        y_act = y_int - ih - 0.007
        # pathologist verification
        y = section(f, 0.06, y_act, 0.88, 0.088, "Pathologist verification before use", accent=RED)
        y = text(f, 0.075, y, "The result depends most on the six regions shown on page 2. Confirm that they are carcinoma and free of artefact. "
                 "If most are not carcinoma, or are artefact, treat the result as unreliable and do not report it.", size=7.7, w=0.85) - 0.006
        xx = 0.08
        for lab in ("Regions reviewed", "Mostly carcinoma", "No major artefact", "Result reportable"):
            xx = checkbox(f, xx, y - 0.004, lab, 7.4)
        f.text(0.08, y - 0.022, "Pathologist: ______________________     Date: ____________", size=7.2, color=INK, va="center")
        y_q = y_act - 0.088 - 0.007
        # quality metrics
        qrows = [("Tissue read by the model", f"{d['n']:,} fields of 112 µm ({d['n'] * 0.012544:.0f} mm²)", tissue_ok, OK_C if tissue_ok == "Adequate" else WARN_C),
                 ("Gland-epithelium fields", f"{n_epi:,}", epi_ok, OK_C if epi_ok == "Adequate" else WARN_C),
                 ("Agreement of the three image models", f"{n_agree} of 3 in the {band} band", "Consistent" if n_agree == 3 else "Partial" if n_agree == 2 else "Inconsistent",
                  OK_C if n_agree == 3 else WARN_C),
                 ("Distance from the nearest band cut-off", f"{round(dist)} points from the {ordinal(near)} percentile", "Clear" if dist >= 5 else "Borderline",
                  OK_C if dist >= 5 else WARN_C)]
        if "ink" in d:
            ik = d["ink"]; qrows.append(("Marking ink among the fields read", "none detected" if not ik["n"] else
                                         f"{ik['n']:,} fields ({100 * ik['share']:.1f}%)" + (f"; without them: {po(after(ik['chg']))}" if ik.get("chg") is not None else ""),
                                         "None" if not ik["n"] else "Present", OK_C if not ik["n"] else WARN_C))
        qh = 0.030 + 0.0165 * len(qrows) + 0.003
        y = section(f, 0.06, y_q, 0.88, qh, "Quality metrics")
        for lab, val, st, col in qrows:
            f.text(0.075, y, lab, size=7.3, color=INK, va="top"); f.text(0.42, y, val, size=7.2, color=INK, va="top"); chip(f, 0.80, y - 0.0055, st, col)
            y -= 0.0165
        # evidence overview (fills the rest of page 1)
        y_e = y_q - qh - 0.010
        if y_e - 0.05 > 0.12:
            eh = y_e - 0.05
            y = section(f, 0.06, y_e, 0.88, eh, "Where the evidence is")
            ih_ = eh - 0.045; a1 = f.add_axes([0.075, 0.05 + 0.010, 0.40, ih_]); a1.imshow(thumb); a1.axis("off")
            marks(a1, d["up"], RED, [str(k + 1) for k in range(len(d["up"]))], ms=5)
            a2 = f.add_axes([0.525, 0.05 + 0.010, 0.40, ih_]); a2.imshow(thumb); a2.imshow(over, cmap=cm, vmin=-1, vmax=1, alpha=0.6); a2.axis("off")
            f.text(0.275, y + 0.002, "H&E, evidence regions 1 to 6 (page 2)", size=6.6, color=MUTED, ha="center", va="top")
            f.text(0.725, y + 0.002, "Removal order: red first, blue last", size=6.6, color=MUTED, ha="center", va="top")
        pdf.savefig(f); plt.close(f)

        # ============================================================ page 2: evidence regions
        f = plt.figure(figsize=(8.27, 11.69))
        y = header(f, "Evidence regions: what the result depends on", 2,
                   "Regions are ranked by sequential removal: region 1 is the one whose removal lowers the result most, region 2 the next once "
                   "region 1 is gone, and so on, each change computed exactly for this model. Left image: 448 µm surroundings with the region "
                   "boxed; right image: the 112 µm region as read. Positions are in mm from the top-left corner of the scan.")
        y -= 0.012; hgt = 0.150; a1 = f.add_axes([0.06, y - hgt, 0.43, hgt]); a1.imshow(thumb); a1.axis("off"); a1.set_title("H&E overview", size=7.3, color=MUTED)
        a2 = f.add_axes([0.51, y - hgt, 0.43, hgt]); a2.imshow(thumb); a2.imshow(over, cmap=cm, vmin=-1, vmax=1, alpha=0.6); a2.axis("off")
        a2.set_title("Removal order (red removed first, blue last)", size=7.3, color=MUTED)
        for a_ in (a1, a2):
            marks(a_, d["up"], RED, [str(k + 1) for k in range(len(d["up"]))], ms=5)
        y -= hgt + 0.008
        if up6 is not None:
            y = text(f, 0.06, y, f"Removing these six regions together would move the result from the {Pd} to the {po(up6)} percentile; "
                     + (f"removing the top {top[1][0] * 100:.0f}% of fields in this order ({top[1][1]:,} fields) would move it to the "
                        f"{po(top[1][2])} ({band_of(top[1][2])} band)." if len(top) > 1 else ""), size=7.4, w=0.88) - 0.006
        WORD = lambda v: "high for this slide" if v >= 75 else "low for this slide" if v <= 25 else "typical for this slide"
        NICE = {"estimated nuclear density": "nuclear density", "estimated nuclear size": "nuclear size",
                "nuclear staining (haematoxylin)": "chromatin", "estimated nuclear crowding": "crowding"}
        ch = (y - 0.045) / 3
        for k, fd in enumerate(d["up"][:6]):
            r_, c_ = divmod(k, 2); x0 = 0.06 + c_ * 0.45; yt = y - r_ * ch
            rbox(f, x0, yt - ch + 0.008, 0.43, ch - 0.008)
            f.add_artist(Rectangle((x0, yt - 0.020), 0.43, 0.020, transform=f.transFigure, fc=HEAD, ec="none", zorder=-1))
            f.text(x0 + 0.010, yt - 0.010, f"REGION {k + 1}", size=7.8, weight="bold", color=RED, va="center")
            f.text(x0 + 0.42, yt - 0.010, f"x {fd['mm'][0]:.1f} mm   y {fd['mm'][1]:.1f} mm", size=6.8, color=MUTED, va="center", ha="right")
            sq = 0.205; sh = min(sq * 8.27 / 11.69, ch - 0.074); sq = sh * 11.69 / 8.27; iy = yt - 0.024 - sh
            ctx, fld = crops["up"][k]
            a = f.add_axes([x0 + 0.008, iy, sq, sh]); a.imshow(ctx); a.axis("off"); h_, w_ = ctx.shape[:2]
            a.add_patch(plt.Rectangle((w_ * 0.375, h_ * 0.375), w_ * 0.25, h_ * 0.25, fill=False, ec=RED, lw=1.1))
            px100 = w_ / 448 * 100; a.plot([w_ * 0.05, w_ * 0.05 + px100], [h_ * 0.93, h_ * 0.93], color="k", lw=2); a.text(w_ * 0.05, h_ * 0.88, "100 µm", size=5.3)
            b = f.add_axes([x0 + 0.43 - 0.008 - sq, iy, sq, sh]); b.imshow(fld); b.axis("off")
            lab = COMP_LABEL[fd["comp"]] if fd.get("comp") is not None else "not available"
            nuc_w = [f"{NICE.get(nm, nm)} {WORD(v).replace(' for this slide', '')}" for nm, v in
                     sorted((fd.get("nuc") or {}).items(), key=lambda t: -abs(t[1] - 50)) if abs(v - 50) >= 25][:3]
            est = f"Automated estimate: {lab.lower()}" + (f"; {', '.join(nuc_w)} for this slide" if nuc_w else "") + "."
            if fd.get("ink"):
                est = "Possible marking ink. " + est
            ty = text(f, x0 + 0.010, iy - 0.004, est, size=6.5, w=0.41, color=RED if fd.get("ink") else MUTED, lh=1.2)
            cy = ty - 0.008; xx = x0 + 0.010
            f.text(xx, cy, "Review:", size=6.6, color=INK, va="center", weight="bold"); xx += 0.060
            for lab_ in ("Carcinoma", "Benign", "Stroma/other", "Artefact"):
                xx = checkbox(f, xx, cy, lab_, 6.4) - 0.008
        pdf.savefig(f); plt.close(f)

        # ============================================================ page 3: findings in detail
        f = plt.figure(figsize=(8.27, 11.69))
        y = header(f, "Morphology findings in detail", 3,
                   "Top: what the result rests on. Each row removes one group of the fields the model read and recomputes the result exactly; "
                   "the dot is this slide's result and the arrow shows where it would move. Groups overlap, so the moves do not add up.")
        scen = [(f"Without {k_.lower()} ({v[0]:,} fields, {v[1]:.0f}%)", v[2]) for k_, v in comp_after.items()]
        if "hi" in dn_:
            scen += [(f"Without the densest quarter of gland fields ({dn_['k']:,})", after(dn_["hi"])),
                     (f"Without the least dense quarter of gland fields ({dn_['k']:,})", after(dn_["lo"]))]
        scen += [(f"Without the top {fr * 100:.0f}% of fields in removal order ({k_:,})", pa) for fr, k_, pa in top]
        hh = 0.026 * len(scen) + 0.03
        ax = f.add_axes([0.42, y - hh, 0.50, hh]); ax.set_xlim(0, 100); ax.set_ylim(len(scen) - 0.4, -0.9)
        for lo_, hi_, b in ((0, 50, "low"), (50, 80, "medium"), (80, 100, "high")):
            ax.axvspan(lo_, hi_, color=BAND_C[b], alpha=0.09, lw=0); ax.text((lo_ + hi_) / 2, -0.62, b.capitalize(), ha="center", size=6.8, color=BAND_C[b])
        for i_, (lab, pa) in enumerate(scen):
            if abs(pa - P) >= 0.6:
                ax.annotate("", xy=(pa, i_), xytext=(P, i_), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.0, shrinkA=3, shrinkB=0))
            ax.plot(P, i_, "o", color=BAND_C[band], ms=5.5, zorder=3)
            ax.text(pa + (1.6 if pa >= P else -1.6), i_ - 0.25, po(pa), size=6.5, ha="left" if pa >= P else "right", color=INK)
        ax.set_yticks(range(len(scen))); ax.set_yticklabels([s[0] for s in scen], size=6.9); ax.tick_params(axis="y", length=0)
        ax.set_xticks([0, 50, 80, 100]); ax.set_xticklabels(["0", "50th", "80th", "100th"], size=6.5)
        ax.set_xlabel("percentile of the development cohort", size=6.8); [ax.spines[s_].set_visible(False) for s_ in ("top", "right", "left")]
        y -= hh + 0.045
        # reference ranges in detail
        rows = [r for r in FIND if r[2] is not None]
        y = section(f, 0.06, y, 0.88, 0.032 + 0.030 * len(rows) + 0.02, "This slide against the development cohort")
        for lab, val, pr, eff, nm in rows:
            q = BIO[nm]["q"]
            f.text(0.075, y, lab, size=7.2, color=INK, va="top"); f.text(0.34, y, val, size=7.2, color=INK, va="top")
            refbar(f, 0.47, y - 0.005, 0.30, nm, bio[nm], h=0.012)
            unit = "%" if nm.endswith("share") else ""
            fmt = (lambda t: f"{100 * t:.0f}%") if nm.endswith("share") else (lambda t: f"{t:,.0f}" if t > 100 else f"{t:.2f}" if t < 10 else f"{t:.1f}")
            f.text(0.47, y - 0.016, f"10th {fmt(q[10])}   median {fmt(q[50])}   90th {fmt(q[90])}", size=5.8, color=MUTED, va="top")
            f.text(0.79, y, f"{po(pr)} percentile", size=7.0, color=INK, va="top")
            y -= 0.030
        text(f, 0.075, y - 0.002, "Reference: the same automated measures on 806 development slides (nuclear measures within gland epithelium, "
             "slides with at least 40 such fields). Estimated by linear read-outs from the image model (cross-validated on development slides: "
             "tissue group accuracy 0.92; nuclear density R² 0.94, size 0.86, chromatin 0.93, crowding 0.90).", size=6.3, w=0.85, color=MUTED, lh=1.2)
        y -= 0.06
        f.text(0.06, y, "Regions that lower the result most", size=9, weight="bold", color=BLUE, va="top"); y -= 0.017
        y = text(f, 0.06, y, "Removing one of these on its own raises the result the most; they are shown so that the evidence on both sides "
                 "can be checked.", size=7.1, w=0.88, color=MUTED) - 0.004
        sq = 0.15; sh = sq * 8.27 / 11.69
        for k, fd in enumerate(d["dn"][:3]):
            x0 = 0.06 + k * 0.30; ctx, fld = crops["dn"][k]
            a = f.add_axes([x0, y - sh, sq, sh]); a.imshow(fld); a.axis("off")
            f.text(x0, y - sh - 0.008, f"{'abc'[k]}   x {fd['mm'][0]:.1f}, y {fd['mm'][1]:.1f} mm", size=7, weight="bold", color=BLUE, va="top")
            lab = COMP_LABEL[fd["comp"]].lower() if fd.get("comp") is not None else ""
            f.text(x0, y - sh - 0.021, "Possible marking ink" if fd.get("ink") else f"Automated estimate: {lab}", size=6.5,
                   color=RED if fd.get("ink") else MUTED, va="top")
        pdf.savefig(f); plt.close(f)

        # ============================================================ page 4: outcomes by band
        f = plt.figure(figsize=(8.27, 11.69))
        y = header(f, "Clinical validity in the development cohorts", 4,
                   "Each development patient was scored by models that had not seen that patient (out-of-fold) and grouped by band. "
                   f"Curves show the share free of biochemical recurrence, with 95% bands; this slide's band ({band}) is drawn heavier.")
        y_tab = y; y = section(f, 0.06, y, 0.88, 0.112, "Outcome by band")
        tab = [("Band", "Radboudumc patients", "Recurrence by 5 years", "TCGA-PRAD patients", "Recurrence by 3 years")]
        for b in ("low", "medium", "high"):
            tab.append((b.capitalize(), f"{km['bands'][b]['n']}", f"{bcr5[b]:.0f}%", f"{kmt['bands'][b]['n']}", f"{bcr3t[b]:.0f}%"))
        xs = [0.08, 0.22, 0.41, 0.60, 0.77]
        for i_, row in enumerate(tab):
            for j_, cell in enumerate(row):
                f.text(xs[j_], y - i_ * 0.0145, cell, size=7.4 if i_ else 6.8, va="top",
                       color=(BAND_C[row[0].lower()] if (j_ == 0 and i_) else MUTED if not i_ else INK),
                       weight="bold" if (i_ and row[0].lower() == band) else "normal")
        y -= 0.0145 * len(tab)
        text(f, 0.08, y, "Ranking accuracy (Harrell's C, out-of-fold): 0.71 at Radboudumc and 0.70 in TCGA-PRAD (0.5 is chance, 1 is perfect).",
             size=7.1, w=0.84)
        y0 = y_tab - 0.112 - 0.035
        for j_, (ds, nm, fu) in enumerate((("owl", "Radboudumc", "508 patients, 87 recurrences, median follow-up 5.1 years"),
                                            ("tcga", "TCGA-PRAD", "298 patients, 62 recurrences, median follow-up 2.1 years"))):
            Kd = REF["km"][ds]; gg = np.asarray(Kd["grid"]); y_top = y0 - j_ * 0.335
            ax = f.add_axes([0.12, y_top - 0.235, 0.80, 0.225])
            for b in ("low", "medium", "high"):
                B = Kd["bands"][b]; lw = 2.6 if b == band else 1.2
                ax.step(gg, B["s"], where="post", color=BAND_C[b], lw=lw, label=f"{b} ({B['n']} patients, {B['events']} recurrences)")
                ax.fill_between(gg, B["lo"], B["hi"], step="post", color=BAND_C[b], alpha=0.12 if b == band else 0.05, lw=0)
            ax.set_ylim(0, 1.02); ax.set_xlim(0, Kd["tmax"]); ax.set_ylabel("free of recurrence", size=7.3); ax.set_xlabel("years after surgery", size=7.3)
            ax.grid(True, color=HEAD, lw=0.6); ax.tick_params(labelsize=6.8); ax.legend(frameon=False, fontsize=6.8, loc="lower left")
            [ax.spines[s_].set_visible(False) for s_ in ("top", "right")]; ax.set_title(f"{nm}: {fu}", size=8, loc="left", color=INK)
            ty = y_top - 0.262; f.text(0.03, ty, "At risk", size=6.4, color=MUTED, weight="bold")
            for b_i, b in enumerate(("low", "medium", "high")):
                f.text(0.03, ty - 0.0115 * (b_i + 1), b, size=6.2, color=BAND_C[b])
                for tk, nr in zip(Kd["ticks"], Kd["bands"][b]["at_risk"]):
                    f.text(0.12 + 0.80 * tk / Kd["tmax"], ty - 0.0115 * (b_i + 1), str(nr), size=6.2, ha="center", color=INK)
        text(f, 0.06, 0.072, "These figures describe development patients; they are not this patient's predicted probability, and results at "
             "new centres may differ. Recurrence events and follow-up are as curated by the challenge organisers.", size=6.8, w=0.88, color=MUTED)
        pdf.savefig(f); plt.close(f)

        # ============================================================ page 5: methods
        f = plt.figure(figsize=(8.27, 11.69))
        y = header(f, "Methods, quality definitions and limitations", 5)
        y = section(f, 0.06, y, 0.88, 0.098, "Workflow")
        steps = ["H&E slide and\ntissue mask", "112 µm fields on\na fixed grid", "Three pathology\nimage models", "Recurrence model\ntrained on outcomes",
                 "Percentile, band\nand evidence regions"]
        bw = 0.152; gap = (0.84 - 5 * bw) / 4
        for i_, s in enumerate(steps):
            x0 = 0.08 + i_ * (bw + gap); rbox(f, x0, y - 0.05, bw, 0.044, fc=PANEL)
            f.text(x0 + bw / 2, y - 0.028, s, size=6.8, ha="center", va="center", color=INK)
            if i_ < 4:
                f.add_artist(FancyArrowPatch((x0 + bw + 0.002, y - 0.028), (x0 + bw + gap - 0.002, y - 0.028), transform=f.transFigure,
                                             arrowstyle="-|>", mutation_scale=8, color=MUTED, lw=0.9))
        y -= 0.075
        blocks = [("Intended use", "A research adjunct to routine reporting of radical prostatectomy specimens. It ranks the slide's "
                   "recurrence risk against development patients and shows the tissue the ranking depends on, for the pathologist to verify. "
                   "It does not replace grading, staging or margin assessment and is not for treatment decisions."),
                  ("Input and training", "Only the H&E image inside the tissue mask. The models were trained on time to biochemical recurrence "
                   "in 806 development patients (Radboudumc and TCGA-PRAD). Gleason grade, ISUP group, stage, margins and PSA were not used as "
                   "inputs, targets or intermediate steps."),
                  ("Model and evidence regions", "Frozen image models (Midnight-12K, UNI2-h, Phikon-v2) [4-6] describe each field; "
                   "attention-based multiple-instance models [1] trained with the Cox partial likelihood [2] combine the fields into one "
                   f"score ({ensemble}). The score is an attention-weighted average of field values, so removing fields re-weights the rest "
                   "and the new score is computed exactly. Evidence regions are ordered by sequential removal."),
                  ("Morphology findings", "Tissue groups and nuclear measures are estimated from the image model by linear read-outs trained on "
                   "202 development slides against nuclei segmented with a network distilled from Cellpose-SAM [7] (counts r = 0.97 on 100 "
                   "reviewed fields). They describe the tissue the model read; the model itself does not use them."),
                  ("Quality metrics", "Tissue read: adequate at 500 fields or more, limited at 100 to 499. Gland-epithelium fields: at least 40 "
                   "for the density finding. Agreement: whether the three image models, each scored alone, fall in the same band. Borderline: "
                   "within 5 percentile points of a band cut-off. Thresholds are provisional."),
                  ("Limitations", "Development-stage test. Results at the challenge's external test centres are not yet available; scanner, "
                   "stain and site differences can shift deep models [9]. Six regions cannot show that all analysed tissue is free of artefact. "
                   "Morphology findings are estimates and are shown for review, not as measurements for reporting.")]
        for h, s in blocks:
            f.text(0.06, y, h, size=8.4, weight="bold", color=NAVY, va="top"); y = text(f, 0.06, y - 0.015, s, size=7.5, w=0.88) - 0.008
        f.text(0.06, y, "References", size=8.4, weight="bold", color=NAVY, va="top"); y -= 0.016
        for k, r in enumerate(REFS):
            y = text(f, 0.06, y, f"{k + 1}. {r}", size=6.3, w=0.88, lh=1.22) - 0.0005
        pdf.savefig(f); plt.close(f)


# ------------------------------------------------------------------ main
def get_args_parser():
    """Arguments for run(args), as the organisers' faithfulness notebook calls it (one slide and one mask per call)."""
    import argparse
    p = argparse.ArgumentParser("KNOX inference", add_help=False)
    p.add_argument("--input", default=str(INPUT_PATH), help="folder holding images/prostatectomy-wsi and images/prostatectomy-tissue-mask")
    p.add_argument("--output", default=str(OUTPUT_PATH), help="folder for overall-survival-years.json and the other outputs")
    p.add_argument("--wsi-dir", default=None, help="slide folder (default: <input>/images/prostatectomy-wsi)")
    p.add_argument("--mask-dir", default=None, help="mask folder (default: <input>/images/prostatectomy-tissue-mask)")
    p.add_argument("--no-report", action="store_true", help="skip the PDF report; the prediction and the map are unchanged")
    return p


def run(model=None):
    """Grand Challenge: run(model) with model = init_model(), inputs under /input (inputs.json lists the sockets).
    Faithfulness notebook: run(args) with args from get_args_parser(); the model is loaded on each call and no
    inputs.json is needed. The prediction is written to <output>/overall-survival-years.json in both cases."""
    import argparse
    global INPUT_PATH, OUTPUT_PATH, REPORT
    if model is None or isinstance(model, argparse.Namespace):
        args = model if model is not None else get_args_parser().parse_args([])
        INPUT_PATH, OUTPUT_PATH, REPORT = Path(args.input), Path(args.output), not args.no_report
        DIRS.clear()
        if args.wsi_dir:
            DIRS["images/prostatectomy-wsi"] = Path(args.wsi_dir)
        if args.mask_dir:
            DIRS["images/prostatectomy-tissue-mask"] = Path(args.mask_dir)
        M = init_model()
        try:
            return handle(M)
        finally:
            del M
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    keyset = tuple(sorted(sv["socket"]["slug"] for sv in json.load(open(INPUT_PATH / "inputs.json"))))
    assert keyset == ("prostatectomy-tissue-mask", "prostatectomy-tissue-whole-slide-image"), keyset
    return handle(model)


def find(d):
    base = DIRS.get(d) or INPUT_PATH / d
    fs = sorted(sum((glob.glob(str(Path(base) / f"*.{e}")) for e in ("tif", "tiff", "mha")), []))
    assert fs, f"no input in {d}"; return fs[0]


def handle(M):
    import openslide
    t0 = time.time(); wsi, mpath = find("images/prostatectomy-wsi"), find("images/prostatectomy-tissue-mask")
    sl = openslide.OpenSlide(wsi); W, H = sl.level_dimensions[0]
    mask, mask_sp = read_mask(mpath); mscale = W / mask.shape[1]
    if abs((H / mask.shape[0]) / mscale - 1) > 0.01:
        log(f"WARNING: slide-to-mask scale differs by axis ({mscale:.3f} in x, {H / mask.shape[0]:.3f} in y)")
    mpp0 = slide_mpp(sl, wsi, mask_sp, mscale)
    mspacing = mask_sp if (mask_sp and 0.5 <= mask_sp <= 100) else mpp0 * mscale       # the map takes the mask's spacing
    s20 = max(1, int(round(TILE_PX * MPP20 / mpp0)))
    all_xy = grid(W, H, s20, mask, mscale); xy = cap(all_xy, W, H, s20)
    log(f"slide {W}x{H} mpp {mpp0:.4f}; mask {mask.shape} ({mspacing:.2f} um/px); tiles {len(all_xy)} -> {len(xy)}")
    if len(xy) == 0:                                    # empty mask (e.g. every field removed): nothing to read
        return write_empty(M, mask, mspacing, sl, t0)
    ink_all = []
    X, ok = embed(M, wsi, xy, s20, stats=ink_all)
    if X is None:
        return write_empty(M, mask, mspacing, sl, t0, note="No field could be read from the slide image, so the development median is reported.")
    xy = xy[ok]
    tissue = tissue_keep(M, X)
    sel = select_lattice(tissue, xy, s20, lattice_k(W, H, s20)) if SELECT == "lattice" else select(tissue)
    if len(sel) < 16 and tissue.sum() < 80:             # tiny tissue (the lattice cannot hold 16 fields): list rule
        sel = select(tissue)
    if len(sel) == 0:                                   # every read field removed by a mask edit: nothing left to read
        return write_empty(M, mask, mspacing, sl, t0, note="No analysed field remained inside the tissue mask, so the "
                           "development median is reported.")
    log(f"Midnight embedded {len(xy)} tiles; tissue {tissue.sum()}; analysed {len(sel)}")
    F2, ok2 = embed(M, wsi, xy[sel], s20, fn=f_second)
    if F2 is None:
        return write_empty(M, mask, mspacing, sl, t0, note="No analysed field could be read again, so the development median is reported.")
    sel = sel[ok2]
    ink_sel = np.asarray(ink_all, float)[sel] if len(ink_all) == len(xy) else np.zeros(len(sel))
    F = dict(midnight=X[sel].half().float(), **F2)
    log(f"UNI2-h + Phikon-v2 embedded {len(sel)} tiles")
    z, c_single, per, (A_, V_, w_) = score_v2(M, F)
    if not np.isfinite(z):                              # never write a non-finite prediction
        log(f"WARNING: non-finite score ({z}); reporting the development median")
        return write_empty(M, mask, mspacing, sl, t0, note="The score could not be computed, so the development median is reported.")
    c, order = sequential_effects(A_, V_, w_)
    log(f"sequential knockout: {int((c > 0).sum())} of {len(c)} tiles lower the score when removed")
    if os.environ.get("OWL_DEBUG_DUMP"):                # per-tile parts for the mask-edit check (not used in scoring)
        dump = dict(xy=xy[sel], z=z, c=c, c_single=c_single, order=order)
        with torch.no_grad():
          for e in ENCS:
            for k_, (net, mu, sd) in enumerate(M["mils"][e]):
                att, s_ = net.parts(F[e]); dump[f"{e}_{k_}_a"] = att.float().cpu().numpy(); dump[f"{e}_{k_}_v"] = ((s_ - mu) / sd).float().cpu().numpy()
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True); np.savez(OUTPUT_PATH / "debug.npz", **dump)
    keep = np.zeros(len(xy), bool); keep[sel] = True
    cal = M["cal"]; kxy = xy[keep]
    years = float(np.clip(cal["median_followup_y"] * np.exp(-z / max(cal["z_sd"], 1e-6)), 0.01, 50.0))
    group = "low" if z < cal["q50"] else "medium" if z < cal["q80"] else "high"
    pct = ref_percentile(z)
    log(f"z {z:+.3f} group {group} years {years:.2f}; per encoder " + " ".join(f"{e} {v:+.2f}" for e, v in per.items()))

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    json.dump(years, open(OUTPUT_PATH / "overall-survival-years.json", "w"))
    json.dump(group, open(OUTPUT_PATH / "prostate-cancer-risk-classification.json", "w"))
    heat_view = paint(heat_rank(order, int((c_single > 0).sum()), xy, sel, tissue, all_xy), all_xy, s20, mscale, mask)
    tile = faith_patch_px(sl, W, mask.shape[1], mpp0)
    heat = heat_patches(c, order, kxy, s20, mscale, mask, tile, view=heat_view) if tile else heat_view   # submitted map
    log(f"attribution map on the faithfulness grid ({tile} mask px patches)" if tile else "attribution map: grid fallback (no whole-number patch size)")
    if os.environ.get("OWL_DEBUG_DUMP"):
        np.save(OUTPUT_PATH / "heat_view.npy", heat_view)
    write_heatmap(OUTPUT_PATH / "images/prostatectomy-overall-survival-attribution-heatmap/attribution.tif", heat, mspacing)
    log("heatmap written")
    if not REPORT:                                      # faithfulness runs: the prediction and the map are unchanged
        sl.close(); log(f"case done in {time.time() - t0:.0f}s (no report)")
        return 0
    try:
        info = dict(group=group, z=z, pct=pct, mpp0=mpp0, mm2=tissue.sum() * (s20 * mpp0 / 1000) ** 2, case=f"Case {Path(wsi).stem}",
                    n_grid=int(len(xy)), filt=100.0 * (1 - tissue.mean()))
        d = report_data(M, F["midnight"].float().cpu().numpy(), A_, V_, w_, c_single, order, kxy, s20, mpp0, per, ink=ink_sel)
        rd = Reader(wsi, 4)

        def ctx(xy_l0):
            S = 4 * s20; lv = sl.get_best_level_for_downsample(S / 336); ds = sl.level_downsamples[lv]
            x0, y0 = int(xy_l0[0] + s20 / 2 - S / 2), int(xy_l0[1] + s20 / 2 - S / 2)
            r = sl.read_region((max(x0, 0), max(y0, 0)), lv, (int(np.ceil(S / ds)), int(np.ceil(S / ds))))
            bg = Image.new("RGB", r.size, (255, 255, 255)); bg.paste(r, mask=r.split()[3]); return np.asarray(bg.resize((336, 336)))
        crops = {k: [(ctx(kxy[fd["i"]]), rd.one(kxy[fd["i"]], s20)) for fd in d[k]] for k in ("up", "dn")}
        f = max(1, max(heat.shape) // 2048)
        if os.environ.get("OWL_REPORT_DUMP"):              # report inputs for offline layout work (not used in scoring)
            import pickle
            pickle.dump(dict(info=info, d=d, crops=crops, heat_small=heat_view[::f, ::f], W0=sl.level_dimensions[0][0],
                             thumb=np.asarray(sl.get_thumbnail((1400, 1400)).convert("RGB"))), open(os.environ["OWL_REPORT_DUMP"], "wb"))
        K = dict(REF=REF, REFS=REFS, COMP_LABEL=COMP_LABEL, ordinal=ordinal, ref_percentile=ref_percentile)
        write_report_v5(OUTPUT_PATH / "pathology-report.pdf", info, d, crops, heat_view[::f, ::f],
                        np.asarray(sl.get_thumbnail((1400, 1400)).convert("RGB")), sl.level_dimensions[0][0], ENSEMBLE, K)
    except Exception as e:                                # the report must never cost us the prediction
        import traceback; log(f"report failed ({type(e).__name__}: {e}); writing minimal PDF\n" + traceback.format_exc())
        write_min_pdf(OUTPUT_PATH / "pathology-report.pdf", group, pct)
    sl.close()
    if torch.cuda.is_available():
        log(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    log(f"case done in {time.time() - t0:.0f}s")
    return 0


def write_empty(M, mask, mspacing, sl, t0, note="The tissue mask was empty, so no tissue was analysed and the development median is reported."):
    """No tissue inside the mask: report the development median (the score with no evidence either way), an empty map
    and a one-page report, so that a fully removed mask gives a fixed baseline instead of a whole-slide reading."""
    cal = M["cal"]; z = float(cal["q50"]); pct = 50.0
    years = float(np.clip(cal["median_followup_y"] * np.exp(-z / max(cal["z_sd"], 1e-6)), 0.01, 50.0))
    group = "low" if z < cal["q50"] else "medium" if z < cal["q80"] else "high"
    log(f"empty tissue mask: development-median score z {z:+.3f} ({group}), empty map")
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    json.dump(years, open(OUTPUT_PATH / "overall-survival-years.json", "w"))
    json.dump(group, open(OUTPUT_PATH / "prostate-cancer-risk-classification.json", "w"))
    write_heatmap(OUTPUT_PATH / "images/prostatectomy-overall-survival-attribution-heatmap/attribution.tif", np.zeros(mask.shape, np.uint8), mspacing)
    write_min_pdf(OUTPUT_PATH / "pathology-report.pdf", group, pct, note=note)
    sl.close(); log(f"case done in {time.time() - t0:.0f}s")
    return 0


def write_min_pdf(path, group, pct, note=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = plt.figure(figsize=(8.27, 11.69)); f.text(0.1, 0.9, f"Reference band: {group}; reference percentile {pct:.0f}. Research use only.", size=12)
    if note:
        f.text(0.1, 0.87, note, size=10)
    f.savefig(path, format="pdf"); plt.close(f)


if __name__ == "__main__":
    raise SystemExit(run(model=init_model()))
