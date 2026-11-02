#!/usr/bin/env python3
"""OWL / TCGA-PRAD streaming feature extractor.

Per slide:
  download WSI (+ mask)
  -> tile grid from the organizer mask (>= 25% tissue)
  -> H-optimus-0 + Midnight-12K features at 20x (224 px @ 0.5 um/px, cap 16k tiles)
     and context tiles (224 px @ 2.0 um/px)
  -> save -> TCGA WSI to the upload queue / OWL WSI deleted (archived separately from S3).

Resumable: a slide is skipped if its tiles/<ds>/<case>.npz exists.
Run one worker per GPU: --shard i/N --device cuda:k
"""
import argparse, os, sys, json, time, shutil, subprocess, threading, traceback, queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = os.path.join(os.environ.get("KNOX_ROOT", "."), "data")   # as training/common.py
HF_REPO = "waticlems/tcga-prad-bcr"
TILE_PX = 224
MPP20, MPPCTX = 0.5, 2.0
MAX_TILES = 16000
MIN_FREE_GB = 20   # wait for this much free disk before each download

H0_MEAN, H0_STD = (0.707223, 0.578729, 0.703617), (0.211883, 0.230117, 0.177517)
MN_MEAN, MN_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def free_gb(path=ROOT):
    return shutil.disk_usage(path).free / 2**30


# ------------------------------------------------------------------ download
def fetch(job, tmp):
    ds, case, wsrc, msrc = job
    os.makedirs(tmp, exist_ok=True)
    while free_gb() < MIN_FREE_GB:
        log(f"[disk] {free_gb():.0f} GB free < {MIN_FREE_GB}; waiting for uploads"); time.sleep(300)
    wdst = os.path.join(tmp, f"{case}.tif"); mdst = os.path.join(tmp, f"{case}_mask.tif")
    if ds == "owl":
        local = os.path.join(ROOT, "wsi", f"{case}.tif")                  # pilot copies
        if os.path.exists(local):
            shutil.move(local, wdst)
        else:
            subprocess.run(["s5cmd", "--no-sign-request", "--endpoint-url", "https://s3.us-west-2.amazonaws.com",
                            "cp", wsrc, wdst], check=True, stdout=subprocess.DEVNULL)
        shutil.copy(msrc, mdst)
    else:
        from huggingface_hub import hf_hub_download
        for src, dst in ((wsrc, wdst), (msrc, mdst)):
            p = hf_hub_download(HF_REPO, src, repo_type="dataset", local_dir=os.path.join(tmp, "hf"))
            shutil.move(p, dst)
        shutil.rmtree(os.path.join(tmp, "hf"), ignore_errors=True)
    return wdst, mdst


# ------------------------------------------------------------------ geometry
def tiff_mpp(path):
    import tifffile
    with tifffile.TiffFile(path) as t:
        p = t.pages[0]
        xr = p.tags["XResolution"].value; unit = p.tags["ResolutionUnit"].value
        ppu = xr[0] / xr[1]
        unit = int(unit)
        per = {3: 1e4, 2: 25400.0}.get(unit)
        if per is None or ppu <= 1:
            raise ValueError(f"no usable resolution tag (unit={unit}, ppu={ppu})")
        return per / ppu


def read_level_downsampled(slide, level, factor, block=4096):
    """Read a whole openslide level and shrink it by `factor` block-wise -> uint8 RGB."""
    W, H = slide.level_dimensions[level]; ds = slide.level_downsamples[level]
    ow, oh = max(1, int(W / factor)), max(1, int(H / factor))
    out = np.full((oh, ow, 3), 255, np.uint8)
    for y in range(0, H, block):
        for x in range(0, W, block):
            bw, bh = min(block, W - x), min(block, H - y)
            reg = slide.read_region((int(x * ds), int(y * ds)), level, (bw, bh))
            bg = Image.new("RGB", reg.size, (255, 255, 255)); bg.paste(reg, mask=reg.split()[3])
            tw, th = max(1, round(bw / factor)), max(1, round(bh / factor))
            ox, oy = int(x / factor), int(y / factor)
            tw, th = min(tw, ow - ox), min(th, oh - oy)
            if tw > 0 and th > 0:
                out[oy:oy + th, ox:ox + tw] = np.asarray(bg.resize((tw, th), Image.BILINEAR))
    return out


def overview(slide, mpp0, target_mpp, max_dim=14000):
    """RGB image of the slide at ~target_mpp (coarser if it would exceed max_dim). Returns (img, eff_mpp)."""
    W, H = slide.level_dimensions[0]
    want_ds = max(target_mpp / mpp0, max(W, H) / max_dim)
    lvl = 0
    for i, d in enumerate(slide.level_downsamples):
        if d <= want_ds:
            lvl = i
    img = read_level_downsampled(slide, lvl, want_ds / slide.level_downsamples[lvl])
    return img, mpp0 * W / img.shape[1]


def integral(a):
    s = np.zeros((a.shape[0] + 1, a.shape[1] + 1), np.float64)
    s[1:, 1:] = a.astype(np.float64).cumsum(0).cumsum(1)
    return s


def box_frac(S, x0, y0, x1, y1):
    """mean of the map over boxes [x0,x1) x [y0,y1) (vectorised, coords already clipped, in map px)."""
    area = np.maximum((x1 - x0) * (y1 - y0), 1)
    return (S[y1, x1] - S[y0, x1] - S[y1, x0] + S[y0, x0]) / area


def map_boxes(coords, size0, scale, shape):
    """level-0 boxes -> clipped integer boxes on a map whose px = `scale` level-0 px."""
    h, w = shape
    x0 = np.clip((coords[:, 0] / scale).astype(int), 0, w); y0 = np.clip((coords[:, 1] / scale).astype(int), 0, h)
    x1 = np.clip(((coords[:, 0] + size0) / scale).astype(int), 0, w); y1 = np.clip(((coords[:, 1] + size0) / scale).astype(int), 0, h)
    x1 = np.maximum(x1, np.minimum(x0 + 1, w)); y1 = np.maximum(y1, np.minimum(y0 + 1, h))
    return x0, y0, x1, y1


def grid(W, H, size0, mask, mscale, thr=0.25):
    xs = np.arange(0, W - size0 + 1, size0); ys = np.arange(0, H - size0 + 1, size0)
    gx, gy = np.meshgrid(xs, ys); c = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.int64)
    S = integral(mask)
    f = box_frac(S, *map_boxes(c, size0, mscale, mask.shape))
    keep = f >= thr
    return c[keep], f[keep].astype(np.float32)


def cap(coords, *arrs, n=MAX_TILES):
    if len(coords) <= n:
        return (coords,) + arrs
    step = int(np.ceil(len(coords) / n)); idx = np.arange(0, len(coords), step)[:n]
    return (coords[idx],) + tuple(a[idx] for a in arrs)


# ------------------------------------------------------------------ encoders
class Encoders:
    def __init__(self, device):
        import torch, timm
        from transformers import AutoModel
        os.environ["HF_HUB_OFFLINE"] = "1"
        self.torch, self.device = torch, device
        # load the cached weights directly (offline): without access to the gated repo, any hub lookup
        # (even for a cached model) returns 403
        import glob
        h0_file = glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--bioptimus--H-optimus-0/snapshots/*/pytorch_model.bin"))[0]
        # same architecture/kwargs as the hub config.json (vit_giant_patch14_reg4_dinov2, CLS pooling)
        self.h0 = timm.create_model("vit_giant_patch14_reg4_dinov2", pretrained=False, num_classes=0,
                                    global_pool="token", img_size=224, init_values=1e-5,
                                    dynamic_img_size=False)
        sd = torch.load(h0_file, map_location="cpu")
        self.h0.load_state_dict(sd, strict=True)
        self.h0 = self.h0.eval().to(device)
        self.mn = AutoModel.from_pretrained("kaiko-ai/midnight", local_files_only=True).eval().to(device)
        t = lambda v: torch.tensor(v, device=device).view(1, 3, 1, 1)
        self.h0m, self.h0s, self.mnm, self.mns = t(H0_MEAN), t(H0_STD), t(MN_MEAN), t(MN_STD)

    def __call__(self, tiles_u8):
        torch = self.torch
        x = torch.from_numpy(tiles_u8).to(self.device).permute(0, 3, 1, 2).float().div_(255.0)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            a = self.h0((x - self.h0m) / self.h0s)                                # [B,1536] CLS
            hs = self.mn(pixel_values=(x - self.mnm) / self.mns).last_hidden_state
            b = torch.cat([hs[:, 0], hs[:, 1:].mean(1)], -1)                      # [B,3072]
        return a.float().cpu().numpy().astype(np.float16), b.float().cpu().numpy().astype(np.float16)


# ------------------------------------------------------------------ tile reading
class Reader:
    def __init__(self, path, threads=16):
        import openslide
        self.path, self.local, self.pool, self.openslide = path, threading.local(), ThreadPoolExecutor(threads), openslide

    def _h(self):
        if not hasattr(self.local, "s"):
            self.local.s = self.openslide.OpenSlide(self.path)
        return self.local.s

    def _one(self, xy, level, size_lvl):
        """Tile as uint8 [224,224,3], or None if the slide data there is corrupt (e.g. bad JPEG)."""
        try:
            r = self._h().read_region((int(xy[0]), int(xy[1])), level, (size_lvl, size_lvl))
        except self.openslide.OpenSlideError:
            del self.local.s                      # handle is unusable after an error; reopen next time
            return None
        bg = Image.new("RGB", r.size, (255, 255, 255)); bg.paste(r, mask=r.split()[3])
        return np.asarray(bg.resize((TILE_PX, TILE_PX), Image.BILINEAR))

    def batches(self, coords, level, size_lvl, bs):
        """Yields (tiles of the readable positions, bool mask of readable positions in this batch)."""
        for i in range(0, len(coords), bs):
            ts = list(self.pool.map(lambda c: self._one(c, level, size_lvl), coords[i:i + bs]))
            ok = np.array([t is not None for t in ts])
            yield (np.stack([t for t in ts if t is not None]) if ok.any() else None), ok

    def close(self):
        self.pool.shutdown()


def embed_all(reader, enc, coords, level, size_lvl, bs=64):
    A, B = [], []
    q = queue.Queue(maxsize=3)
    err = []

    def producer():
        try:
            for b in reader.batches(coords, level, size_lvl, bs):
                q.put(b)
        except BaseException as e:            # surface read errors: a swallowed error once wrote a slide
            err.append(e)                     # with 5,312 of 13,830 tiles and no context tiles
        finally:
            q.put(None)
    th = threading.Thread(target=producer, daemon=True); th.start()
    OK = []
    while (item := q.get()) is not None:
        b, ok = item; OK.append(ok)
        if b is not None:
            a, m = enc(b); A.append(a); B.append(m)
    th.join()
    if err:
        raise RuntimeError(f"tile read failed: {err[0]!r}") from err[0]
    ok = np.concatenate(OK) if OK else np.zeros(0, bool)
    if len(ok) != len(coords) or sum(len(a) for a in A) != ok.sum():
        raise RuntimeError(f"embedded {sum(len(a) for a in A)} of {len(coords)} tiles")
    if not A:
        return np.zeros((0, 1536), np.float16), np.zeros((0, 3072), np.float16), ok
    return np.concatenate(A), np.concatenate(B), ok


# ------------------------------------------------------------------ per slide
def process(job, wpath, mpath, enc, device):
    import openslide, tifffile
    ds, case = job[0], job[1]
    t0 = time.time()
    mpp0 = tiff_mpp(wpath)
    s = openslide.OpenSlide(wpath); W, H = s.level_dimensions[0]
    ms = openslide.OpenSlide(mpath)                                          # tifffile.asarray breaks on this numpy
    mask = np.asarray(ms.read_region((0, 0), 0, ms.level_dimensions[0]))[..., 0] > 0
    ms.close()
    mscale = W / mask.shape[1]                                              # level-0 px per mask px

    # Organizer tissue mask is provided for every slide -> GrandQC is not used.
    timg, _ = overview(s, mpp0, 16.0)
    thumb = Image.fromarray(timg); thumb.thumbnail((2048, 2048))

    # --- tile grids from the organizer mask
    s20 = int(round(TILE_PX * MPP20 / mpp0))
    c20, f20 = grid(W, H, s20, mask, mscale)
    n20_all = len(c20)
    sctx = int(round(TILE_PX * MPPCTX / mpp0))
    cctx, fctx = grid(W, H, sctx, mask, mscale)
    c20, f20 = cap(c20, f20)
    s.close()

    # --- read + embed (20x from level 0; context from the best level with downsample <= needed)
    rd = Reader(wpath)
    h20, m20, ok20 = embed_all(rd, enc, c20, 0, s20)
    sl = openslide.OpenSlide(wpath)
    want = sctx; lvl = max(i for i, d in enumerate(sl.level_downsamples) if d <= want / TILE_PX + 1e-6)
    lds = sl.level_downsamples[lvl]; sl.close()
    hctx, mctx, okctx = embed_all(rd, enc, cctx, lvl, int(round(sctx / lds)))
    rd.close()
    n_bad20, n_badctx = int((~ok20).sum()), int((~okctx).sum())      # corrupt tiles are dropped, not filled
    c20, f20, cctx, fctx = c20[ok20], f20[ok20], cctx[okctx], fctx[okctx]

    meta = dict(ds=ds, case=case, mpp0=mpp0, W=W, H=H, mask_scale=mscale, tile20_l0=s20, tilectx_l0=sctx,
                ctx_level=lvl, n20=len(c20), n20_before_cap=int(n20_all), nctx=len(cctx),
                n_corrupt20=n_bad20, n_corruptctx=n_badctx,
                qc="organizer_mask", secs=round(time.time() - t0, 1))
    for d in ("tiles", "feats/hoptimus0", "feats/midnight", "thumbs"):
        os.makedirs(f"{ROOT}/{d}/{ds}", exist_ok=True)
    np.savez_compressed(f"{ROOT}/feats/hoptimus0/{ds}/{case}.npz", x20=h20, xctx=hctx)
    np.savez_compressed(f"{ROOT}/feats/midnight/{ds}/{case}.npz", x20=m20, xctx=mctx)
    thumb.save(f"{ROOT}/thumbs/{ds}/{case}.jpg", quality=85)
    # tiles file written LAST = completion marker
    np.savez_compressed(f"{ROOT}/tiles/{ds}/{case}.npz", coords20=c20.astype(np.int32), mask_frac20=f20,
                        coordsctx=cctx.astype(np.int32), mask_fracctx=fctx, meta=json.dumps(meta))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    i, N = map(int, args.shard.split("/"))
    jobs = [l.rstrip("\n").split("\t") for l in open(args.jobs) if l.strip()]
    jobs = [j for k, j in enumerate(jobs) if k % N == i]
    todo = [j for j in jobs if not os.path.exists(f"{ROOT}/tiles/{j[0]}/{j[1]}.npz")]
    if args.limit:
        todo = todo[:args.limit]
    log(f"shard {args.shard}: {len(jobs)} jobs, {len(todo)} to do")
    enc = Encoders(args.device)
    tmp_root = f"{ROOT}/_tmp/shard{i}"
    # prefetch next slide while the current one is processed
    pre = ThreadPoolExecutor(1)
    fut = pre.submit(fetch, todo[0], f"{tmp_root}/a") if todo else None
    for k, job in enumerate(todo):
        tmp = f"{tmp_root}/{'a' if k % 2 == 0 else 'b'}"
        try:
            wpath, mpath = fut.result()
        except Exception as e:
            log(f"[FAIL-download] {job[1]}: {e}")
            with open(f"{ROOT}/../logs/failed.tsv", "a") as fh:
                fh.write("\t".join(job) + f"\tdownload\t{e}\n")
            wpath = None
        fut = pre.submit(fetch, todo[k + 1], f"{tmp_root}/{'b' if k % 2 == 0 else 'a'}") if k + 1 < len(todo) else None
        if wpath is None:
            shutil.rmtree(tmp, ignore_errors=True); continue
        try:
            m = process(job, wpath, mpath, enc, args.device)
            log(f"[ok] {job[0]} {job[1]} n20={m['n20']}/{m['n20_before_cap']} nctx={m['nctx']} "
                f"mpp={m['mpp0']:.4f} {m['secs']}s")
            if job[0] == "tcga":                                           # queue raw TCGA files for Drive
                q = f"{ROOT}/upload_pending/tcga_prad_bcr"
                os.makedirs(f"{q}/whole-slide-images", exist_ok=True); os.makedirs(f"{q}/tissue-masks", exist_ok=True)
                shutil.move(wpath, f"{q}/whole-slide-images/{job[1]}.tif")
                shutil.move(mpath, f"{q}/tissue-masks/{job[1]}.tif")
        except Exception as e:
            log(f"[FAIL] {job[1]}: {e}\n{traceback.format_exc()}")
            with open(f"{ROOT}/../logs/failed.tsv", "a") as fh:
                fh.write("\t".join(job) + f"\tprocess\t{e}\n")
        shutil.rmtree(tmp, ignore_errors=True)
    log("shard done")


if __name__ == "__main__":
    main()
