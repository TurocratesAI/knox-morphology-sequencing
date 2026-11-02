"""E26: more permitted encoders for an ensemble, one slide download feeding several encoders.

Same tiles as E19 (--stride 2 = every second E19 tile, ~2,000 per slide, used from 2026-10-27 to fit the GPU budget; Virchow2 idx: <= 4,000 stride-subsampled E0 tissue tiles, 224 px at 0.5 um/px), so rows align
across encoders. Encoders (all on the OWL permitted list; team HF access checked 2026-10-27):
  uni2h     MahmoodLab/UNI2-h (ViT-H/14, 8 registers)  -> [CLS | mean patch] 3072-d
  gigapath  prov-gigapath/prov-gigapath tile encoder   -> CLS 1536-d
  phikon2   owkin/phikon-v2 (ViT-L/16)                 -> CLS 1024-d
ImageNet mean/std, fp16 autocast. Output data/feats/<enc>/<ds>/<case>.npz: x20, idx (into coords20).
Slides: by default only the needed byte ranges are fetched (sparse_fetch.py, pixel-identical; checked on one RUMC and one TCGA slide 2026-10-28).
Needs HF_TOKEN in the environment and HF_HUB_DISABLE_XET=1. Usage: python e26_multi_encoder.py --shard 0/2 --device cuda:0
"""
import argparse, os, shutil, sys, time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [_HERE, os.path.join(_HERE, "..", "training")]
import common as C
import owl_extract as X
import sparse_fetch as SF

ENCS = ("uni2h", "gigapath", "phikon2")
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
NMAX = 4000


def build(name, dev):
    import timm
    if name == "uni2h":
        m = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, img_size=224, patch_size=14, depth=24, num_heads=24,
                              init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
                              mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True)
        f = lambda x: (lambda o: torch.cat([o[:, 0], o[:, 9:].mean(1)], -1))(m.forward_features(x))
    elif name == "gigapath":
        m = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        f = lambda x: m(x)
    elif name == "phikon2":
        from transformers import AutoModel
        m = AutoModel.from_pretrained("owkin/phikon-v2")
        f = lambda x: m(pixel_values=x).last_hidden_state[:, 0]
    m.eval().half().to(dev)                                     # fp16 weights: halves memory; matmuls are fp16 under autocast anyway
    return f


class Encs:
    def __init__(self, names, dev):
        self.f = {n: build(n, dev) for n in names}; self.dev = dev
        self.mu = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1); self.sd = torch.tensor(STD, device=dev).view(1, 3, 1, 1)

    @torch.inference_mode()
    def __call__(self, tiles):
        x = ((torch.from_numpy(tiles).to(self.dev).permute(0, 3, 1, 2).float().div_(255) - self.mu) / self.sd).half()
        with torch.autocast("cuda", dtype=torch.float16):
            return {n: f(x).float().cpu().numpy().astype(np.float16) for n, f in self.f.items()}


LAT = False                                                     # --tiles lattice: mask-stable subsample (e31), feats/<enc>_lat


def tile_idx(ds, case, stride=1):
    if LAT:                                                     # tissue tiles at (gx + 3 gy) mod 6 == 0 of the 20x grid
        T = C.tiles(ds, case); g = T["coords20"] // T["meta"]["tile20_l0"]
        keep = np.load(f"{C.ROOT}/tissue/{ds}/{case}.npz")["keep"]
        return np.where(keep & ((g[:, 0] + 3 * g[:, 1]) % 6 == 0))[0].astype(np.int64)
    v = f"{C.ROOT}/feats/virchow2/{ds}/{case}.npz"
    if os.path.exists(v):
        return np.load(v)["idx"].astype(np.int64)[::stride]
    idx = np.where(np.load(f"{C.ROOT}/tissue/{ds}/{case}.npz")["keep"])[0]
    return idx[np.arange(0, len(idx), int(np.ceil(len(idx) / NMAX)))[:NMAX]] if len(idx) > NMAX else idx


def one(enc, names, ds, case, wsi, bs=64, stride=1):
    T = C.tiles(ds, case); idx = tile_idx(ds, case, stride)
    rd = X.Reader(wsi); s20 = T["meta"]["tile20_l0"]; out = {n: [] for n in names}; ok = []
    for tiles, good in rd.batches(T["coords20"][idx], 0, s20, bs):
        ok.append(good)
        if tiles is not None:
            for n, z in enc(tiles).items():
                out[n].append(z)
    rd.close(); ok = np.concatenate(ok)
    for n in names:
        dst = f"{C.ROOT}/feats/{n}{'_lat' if LAT else ''}/{ds}/{case}.npz"; os.makedirs(os.path.dirname(dst), exist_ok=True)
        np.savez_compressed(dst[:-4] + ".part.npz", x20=np.concatenate(out[n]), idx=idx[ok].astype(np.int32))
        os.replace(dst[:-4] + ".part.npz", dst)
    return int(ok.sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--shard", default="0/1"); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--encs", default=",".join(ENCS)); ap.add_argument("--prefetch", type=int, default=3)
    ap.add_argument("--only", default=""); ap.add_argument("--batch", type=int, default=64); ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--sparse", type=int, default=1); ap.add_argument("--tiles", default="e26"); ap.add_argument("--every", type=int, default=1); a = ap.parse_args(); names = a.encs.split(",")
    global LAT; LAT = a.tiles == "lattice"
    jobs = [l.rstrip("\n").split("\t") for l in open(C.JOBS)]
    have = {(ds, c) for ds in ("owl", "tcga") for c in C.labels(ds).index}
    jobs = [j for j in jobs if (j[0], j[1]) in have and (not a.only or j[1] == a.only)]
    jobs = jobs[::a.every]                                      # --every 4: validation subset (both cohorts)
    i, n = map(int, a.shard.split("/")); jobs = jobs[i::n]
    todo = [j for j in jobs if not all(os.path.exists(f"{C.ROOT}/feats/{e}{'_lat' if LAT else ''}/{j[0]}/{j[1]}.npz") for e in names)]
    torch.cuda.set_device(a.device); enc = Encs(names, a.device); print(f"{len(todo)} slides to do", flush=True)

    def get(j):
        tmp = f"{C.ROOT}/_tmp/e26{'_lat' if LAT else ''}/{j[1]}"; os.makedirs(tmp, exist_ok=True); full = f"{tmp}/{j[1]}.tif"
        if os.path.exists(full):                                # complete download left by an earlier run (downloads land under temp names)
            return j, full, tmp
        if a.sparse:                                            # only the bytes openslide needs for these tiles; pixel-identical
            try:
                T = C.tiles(j[0], j[1]); dst = f"{tmp}/{j[1]}.sparse.tif"
                SF.fetch_sparse(j, dst, T["coords20"][tile_idx(j[0], j[1], a.stride)], T["meta"]["tile20_l0"]); return j, dst, tmp
            except Exception as e:
                print(f"  sparse {j[1]} failed ({type(e).__name__}: {e}); full download", flush=True)
        for _ in range(3):
            try:
                return j, X.fetch(j, tmp)[0], tmp
            except Exception as e:
                print(f"  fetch {j[1]} failed ({type(e).__name__})", flush=True); shutil.rmtree(tmp, ignore_errors=True)
        return j, None, tmp
    with ThreadPoolExecutor(a.prefetch) as pre:
        q = deque(pre.submit(get, j) for j in todo[:a.prefetch]); nxt = a.prefetch
        while q:
            j, wsi, tmp = q.popleft().result()
            if nxt < len(todo):
                q.append(pre.submit(get, todo[nxt])); nxt += 1
            if wsi:
                t0 = time.time()
                try:
                    print(f"[ok] {j[1]} tiles={one(enc, names, j[0], j[1], wsi, a.batch, a.stride)} {time.time() - t0:.0f}s", flush=True)
                except Exception as e:
                    print(f"[fail] {j[1]} {type(e).__name__}: {e}", flush=True)
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
