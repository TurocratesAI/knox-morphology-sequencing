"""E7: whole-slide nucleus maps (full coverage, no sampling) with a distilled Cellpose-SAM (cpsam) U-Net
(module `seg`, not included; see README).

Per slide: fetch the WSI, read ~2 mm blocks (4096 px + 48 px halo) at the pyramid level nearest 0.5 um/px wherever
the full-coverage context grid has tissue, segment, keep nuclei whose centroid lies in the block core (no duplicates),
and compute per-nucleus geometry/stain with vectorised bincounts.
Output data/nuclei/<ds>/<case>.npz:
  xy      int32 [n,2]  centroid, level-0 px
  area    float16      um^2
  major, minor float16 um (2*sqrt(2*eigenvalue) of the pixel second moments ~ ellipse axes)
  angle   float16      radians, major-axis orientation in [-pi/2, pi/2)
  hema, hema_sd float16 mean / sd hematoxylin OD inside the nucleus (fixed Ruifrok vectors)
  eos_ring float16     mean eosin OD in a 2-px ring around the nucleus (cytoplasm/stroma context)
  meta json: mpp0, level, mpp, n_blocks, secs
No outcome information is touched. Slides are deleted after use.
"""
import argparse, json, os, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import torch
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "training")]
sys.path.insert(0, os.environ.get("KNOX_SEG_DIR", "."))   # directory holding the nucleus segmentation module seg.py
import common as C
import owl_extract as X
import seg

BLOCK, HALO, WMPP = 4096, 48, 0.5           # working grid 0.5 um/px; block = 2.05 mm
# Ruifrok & Johnston H&E(DAB) OD vectors; columns of the inverse give H and E concentrations
RGB_FROM_HED = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]])
HED_FROM_RGB = np.linalg.inv(RGB_FROM_HED)


def nuclei_props(rgb, lab, mpp, dev):
    """Per-label properties on the GPU (labels 1..n on the working grid). Returns dict of numpy arrays."""
    import torch.nn.functional as Fn
    L = torch.as_tensor(lab, device=dev).long(); n = int(L.max())
    if n == 0:
        return None
    x = torch.as_tensor(rgb, device=dev).float()
    od = -torch.log((x + 1) / 256); hed = od.reshape(-1, 3) @ torch.as_tensor(HED_FROM_RGB, device=dev).float()
    H, E = hed[:, 0], hed[:, 1]
    l = L.reshape(-1); fg = l > 0; li = l[fg] - 1
    pos = torch.nonzero(fg).squeeze(1); yy = (pos // L.shape[1]).double(); xx = (pos % L.shape[1]).double()
    bc = lambda w: torch.bincount(li, w, minlength=n).double()
    a = torch.bincount(li, minlength=n).double().clamp(min=1); cy, cx = bc(yy) / a, bc(xx) / a
    syy = bc(yy * yy) / a - cy ** 2 + 1 / 12; sxx = bc(xx * xx) / a - cx ** 2 + 1 / 12; sxy = bc(xx * yy) / a - cx * cy
    tr, det = sxx + syy, sxx * syy - sxy ** 2; disc = torch.sqrt((tr ** 2 / 4 - det).clamp(min=0))
    l1, l2 = tr / 2 + disc, (tr / 2 - disc).clamp(min=1e-6)
    Hf = H[fg].double(); hm = bc(Hf) / a; hs = torch.sqrt((bc(Hf ** 2) / a - hm ** 2).clamp(min=0))
    dil = Fn.max_pool2d(L[None, None].float(), 5, 1, 2)[0, 0].long().reshape(-1); ring = (l == 0) & (dil > 0)
    ri = dil[ring] - 1
    er = torch.bincount(ri, E[ring].double(), minlength=n) / torch.bincount(ri, minlength=n).double().clamp(min=1)
    out = dict(cy=cy, cx=cx, area=a * mpp ** 2, major=2 * torch.sqrt(2 * l1) * mpp, minor=2 * torch.sqrt(2 * l2) * mpp,
               angle=0.5 * torch.atan2(2 * sxy, sxx - syy), hema=hm, hema_sd=hs, eos_ring=er)
    keep = torch.bincount(li, minlength=n) > 0
    return {k: v[keep].cpu().numpy() for k, v in out.items()}


def run_slide(ds, case, wsi, device, out):
    import openslide
    t0 = time.time(); T = C.tiles(ds, case); mpp0 = T["meta"]["mpp0"]
    sl = openslide.OpenSlide(wsi)
    lvl = max(i for i, d in enumerate(sl.level_downsamples) if d * mpp0 <= WMPP * 1.08)
    mpp_l = mpp0 * sl.level_downsamples[lvl]; dsf = WMPP / mpp0; mpp = WMPP      # dsf: level-0 px per working px
    # tissue blocks from the full-coverage context grid (tile centres, level-0 px)
    s = T["meta"]["tilectx_l0"]; c = T["coordsctx"][T["mask_fracctx"] >= 0.1] + s // 2
    B0 = BLOCK * dsf; blocks = sorted({(int(x // B0), int(y // B0)) for x, y in c})
    acc = {k: [] for k in ("cy", "cx", "area", "major", "minor", "angle", "hema", "hema_sd", "eos_ring")}

    def read(b):
        bx, by = b; x0, y0 = int(bx * B0 - HALO * dsf), int(by * B0 - HALO * dsf)
        S = int(round((BLOCK + 2 * HALO) * WMPP / mpp_l))
        im = sl.read_region((x0, y0), lvl, (S, S)).convert("RGB")
        if S != BLOCK + 2 * HALO:
            im = im.resize((BLOCK + 2 * HALO, BLOCK + 2 * HALO), Image.BOX)
        im = np.asarray(im)
        return b, x0, y0, im
    with ThreadPoolExecutor(8) as ex:
        for b, x0, y0, im in ex.map(read, blocks):
            lab = seg.segment(im, mpp, device=device)
            p = nuclei_props(im, lab, mpp, device)
            if p is None:
                continue
            core = (p["cy"] >= HALO) & (p["cy"] < HALO + BLOCK) & (p["cx"] >= HALO) & (p["cx"] < HALO + BLOCK)
            p["cy"], p["cx"] = y0 + p["cy"] * dsf, x0 + p["cx"] * dsf
            for k in acc:
                acc[k].append(p[k][core])
    sl.close()
    A = {k: np.concatenate(v) for k, v in acc.items()}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp_out = out[:-4] + ".part.npz"
    np.savez_compressed(tmp_out, xy=np.c_[A["cx"], A["cy"]].astype(np.int32),
                        **{k: A[k].astype(np.float16) for k in ("area", "major", "minor", "angle", "hema", "hema_sd", "eos_ring")},
                        meta=json.dumps(dict(mpp0=mpp0, level=lvl, mpp=mpp, n_blocks=len(blocks), secs=round(time.time() - t0))))
    os.replace(tmp_out, out)
    return len(A["area"]), len(blocks), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="")            # comma list; default = all jobs (resumable)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--prefetch", type=int, default=3)
    a = ap.parse_args()
    jobs = [l.rstrip("\n").split("\t") for l in open(C.JOBS)]
    if a.cases:
        want = set(a.cases.split(",")); jobs = [j for j in jobs if j[1] in want]
    i, n = map(int, a.shard.split("/")); jobs = jobs[i::n]
    torch.cuda.set_device(a.device)

    def get(job):
        tmp = f"{C.ROOT}/_tmp/nuclei/{job[1]}"
        for attempt in range(3):
            try:
                return job, X.fetch(job, tmp)[0], tmp
            except Exception as e:
                print(f"  fetch {job[1]} failed ({type(e).__name__}); retry", flush=True); shutil.rmtree(tmp, ignore_errors=True)
        return job, None, tmp
    todo = [j for j in jobs if not os.path.exists(f"{C.ROOT}/nuclei/{j[0]}/{j[1]}.npz")]
    print(f"{len(todo)} slides to do", flush=True)
    from collections import deque
    with ThreadPoolExecutor(a.prefetch) as pre:      # keep `prefetch` downloads in flight while segmenting
        q = deque(pre.submit(get, j) for j in todo[:a.prefetch]); nxt = a.prefetch
        while q:
            job, wsi, tmp = q.popleft().result()
            if nxt < len(todo):
                q.append(pre.submit(get, todo[nxt])); nxt += 1
            if wsi is None:
                continue
            try:
                nn, nb, dt = run_slide(job[0], job[1], wsi, a.device, f"{C.ROOT}/nuclei/{job[0]}/{job[1]}.npz")
                print(f"[ok] {job[1]} nuclei={nn} blocks={nb} {dt:.0f}s", flush=True)
            except Exception as e:
                print(f"[fail] {job[1]} {type(e).__name__}: {e}", flush=True)
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
