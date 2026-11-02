"""Fetch only the bytes of a remote slide that openslide needs for a given set of regions.

The link here averages ~22 MB/s, and a whole RUMC slide is ~4 GB, while ~2,000 sampled tiles touch ~10% of it.
So: parse the TIFF structure over HTTP range requests (tifffile), then write the structure blocks plus the level-0
JPEG tiles covering the requested regions into a sparse local file at their original offsets. openslide reads the
same bytes it would read from a full download, so tiles are pixel-identical; regions outside the requested set
would read as zeros (never touched).
Sources: s3://leopard-challenge (public, unsigned HTTPS) and the HF dataset (token from HF_TOKEN, sent only to huggingface.co).
"""
import http.client, io, os, threading, time, urllib.parse
from concurrent.futures import ThreadPoolExecutor
import numpy as np, tifffile

HF_REPO = "waticlems/tcga-prad-bcr"
BLOCK = 1 << 19                      # 512 KB blocks for structure reads
GAP = 1 << 15                        # merge tile ranges closer than 32 KB
MAXRUN = 2 << 20                     # cap merged range size: one connection only gets ~1.4 MB/s here
_tl = threading.local()


def _conn(host):
    c = getattr(_tl, "c", {})
    if host not in c:
        c[host] = http.client.HTTPSConnection(host, timeout=60); _tl.c = c
    return c[host]


def _drop(host):
    c = getattr(_tl, "c", {}); x = c.pop(host, None)
    if x:
        x.close()


def resolve(job):
    """(https url for range requests, size in bytes)."""
    ds, case, src = job[:3]
    if ds == "owl":
        b, k = src[5:].split("/", 1); url = f"https://{b}.s3.us-west-2.amazonaws.com/{k}"
    else:
        url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{src}"
        for _ in range(6):                                      # follow redirects by hand; auth only for huggingface.co
            u = urllib.parse.urlsplit(url); h = {"Range": "bytes=0-0"}
            if u.hostname == "huggingface.co" and os.environ.get("HF_TOKEN"):
                h["Authorization"] = "Bearer " + os.environ["HF_TOKEN"]
            c = http.client.HTTPSConnection(u.hostname, timeout=60)
            c.request("GET", u.path + ("?" + u.query if u.query else ""), headers=h); r = c.getresponse(); r.read(); c.close()
            if r.status in (301, 302, 303, 307, 308):
                url = urllib.parse.urljoin(url, r.getheader("Location")); continue
            break
    return url, _get(url, 0, 1)[1]


def _get(url, start, n, tries=6):
    """Bytes [start, start+n) of url, and the total object size."""
    u = urllib.parse.urlsplit(url); path = u.path + ("?" + u.query if u.query else "")
    for k in range(tries):
        try:
            c = _conn(u.hostname); c.request("GET", path, headers={"Range": f"bytes={start}-{start + n - 1}"})
            r = c.getresponse(); d = r.read()
            if r.status != 206 or len(d) != n:
                raise IOError(f"status {r.status} len {len(d)} want {n}")
            return d, int(r.getheader("Content-Range").rsplit("/", 1)[1])
        except Exception:
            _drop(u.hostname)
            if k == tries - 1:
                raise
            time.sleep(1 + 2 * k)


class RangeFile(io.RawIOBase):
    """Read-only seekable file over HTTP range requests, block cached; remembers which blocks were read."""

    def __init__(self, url, size):
        self.url, self.size, self.pos, self.blocks = url, size, 0, {}

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off; return self.pos

    def _block(self, b):
        if b not in self.blocks:
            s = b * BLOCK; self.blocks[b] = _get(self.url, s, min(BLOCK, self.size - s))[0]
        return self.blocks[b]

    def readinto(self, buf):
        n = min(len(buf), max(0, self.size - self.pos)); out, p = memoryview(buf), 0
        while p < n:
            b, o = divmod(self.pos + p, BLOCK); d = self._block(b)[o:o + n - p]; out[p:p + len(d)] = d; p += len(d)
        self.pos += n; return n


def needed_tiles(page, xy, size):
    """Level-0 tile indices covering squares [x, x+size) x [y, y+size)."""
    H, W = page.shape[:2]; tw, tl = page.tilewidth, page.tilelength; ntx = -(-W // tw); need = set()
    for x, y in np.asarray(xy, dtype=np.int64):
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + size, W), min(y + size, H)
        if x1 <= x0 or y1 <= y0:
            continue
        for r in range(y0 // tl, (y1 - 1) // tl + 1):
            for c in range(x0 // tw, (x1 - 1) // tw + 1):
                need.add(r * ntx + c)
    return need


def merge(ranges):
    out = []
    for s, n in sorted(r for r in ranges if r[1] > 0):
        if out and s - (out[-1][0] + out[-1][1]) <= GAP and s + n - out[-1][0] <= MAXRUN:
            out[-1][1] = max(out[-1][1], s + n - out[-1][0])
        else:
            out.append([s, n])
    return out


def fetch_sparse(job, dst, xy, size, threads=32, small_page_mb=4):
    """Write a sparse copy of the job's slide to dst holding what openslide needs for regions xy (level 0, size px)."""
    url, total = resolve(job); rf = RangeFile(url, total)
    with tifffile.TiffFile(rf) as t:
        pages = list(t.pages)
        for p in pages:                                        # force every tag value and strile array to be read
            for tag in p.tags:
                tag.value
            p.dataoffsets, p.databytecounts
        p0 = pages[0]; assert p0.is_tiled, "level 0 is not tiled"
        rng = [(int(p0.dataoffsets[i]), int(p0.databytecounts[i])) for i in needed_tiles(p0, xy, size)]
        for p in pages[1:]:                                     # small pages whole (lowest level for openslide's hash, label, thumbnail)
            if np.sum(p.databytecounts) < small_page_mb * 1e6:
                rng += [(int(o), int(n)) for o, n in zip(p.dataoffsets, p.databytecounts)]
    have = set(rf.blocks); runs = [r for r in merge(rng) if not all(b in have for b in range(r[0] // BLOCK, (r[0] + r[1] - 1) // BLOCK + 1))]
    part = dst + ".part"; fd = os.open(part, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
    try:
        os.ftruncate(fd, total)
        for b, d in rf.blocks.items():
            os.pwrite(fd, d, b * BLOCK)
        with ThreadPoolExecutor(threads) as ex:
            for (s, n), d in zip(runs, ex.map(lambda r: _get(url, r[0], r[1])[0], runs)):
                os.pwrite(fd, d, s)
    finally:
        os.close(fd)
    os.replace(part, dst)
    return dict(total_mb=total / 1e6, fetched_mb=(sum(n for _, n in runs) + len(rf.blocks) * BLOCK) / 1e6, requests=len(runs) + len(rf.blocks))
