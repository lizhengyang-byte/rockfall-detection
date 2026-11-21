"""
rockfall_dataset.py
===================
Dataset builder for the **Rockfall / Landslide object-detection** benchmark.

Source file
-----------
``rockfall.ndjson``: one JSON record per line.
  * line 0            -> ``{"type": "dataset", ...}`` metadata (classes, url, bytes ...)
  * every other line  -> ``{"type": "image", "file", "url", "width", "height",
                            "split", "annotations": {"boxes": [[cls, cx, cy, w, h], ...]}}``

Boxes are already **YOLO-normalised** (cx, cy, w, h in [0, 1]) relative to the
image shown in ``width`` x ``height``.  ``width``/``height`` describe the image
*as delivered by the CDN url* (the CDN may return a resized file, but the boxes
were normalised to it, so mixing them would mis-align labels).  We therefore
simply copy the downloaded image bytes unchanged.

Output layout (Ultralytics-ready)
---------------------------------
::

  datasets/rockfall/
    images/{train,val,test}/*.jpg
    labels/{train,val,test}/*.txt        # "cls cx cy w h" normalised
    data.yaml                            # names + paths
    dataset_stats.json                   # counts, class dist., per-split boxes

The builder is *idempotent*: files already on disk are skipped.
"""
from __future__ import annotations

import json
import os
import sys
import time
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from http.client import IncompleteRead
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
NDJSON_PATH = Path("rockfall.ndjson")
DATASET_ROOT = Path("datasets") / "rockfall"
SPLITS = ("train", "val", "test")
MAX_WORKERS = 16          # parallel downloads
RETRIES = 4               # per-URL retries
TIMEOUT = 45              # seconds / request


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _read_records(path: Path):
    """Return (meta_dict, list_of_image_dicts)."""
    with open(path, "r", encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip()]
    records = [json.loads(ln) for ln in lines]
    meta = next((r for r in records if r.get("type") == "dataset"), {})
    images = [r for r in records if r.get("type") == "image"]
    return meta, images


def _download(url: str, dst: Path) -> int:
    """Download ``url`` to ``dst`` atomically. Returns byte count (0 if skipped)."""
    if dst.exists() and dst.stat().st_size > 0:
        return 0  # idempotent skip
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0 rockfall-builder"})
            chunks = []
            with urlopen(req, timeout=TIMEOUT) as resp:
                while True:
                    chunk = resp.read(1 << 16)   # stream; raises IncompleteRead on truncation
                    if not chunk:
                        break
                    chunks.append(chunk)
            data = b"".join(chunks)
            if not data:
                raise IOError("empty body")
            tmp.write_bytes(data)
            os.replace(tmp, dst)          # atomic
            return len(data)
        except (IncompleteRead, URLError, HTTPError, IOError, OSError) as exc:  # noqa: PERF203
            last_err = exc
            time.sleep(min(2 ** attempt * 0.5, 8))
    if tmp.exists():
        tmp.unlink(missing_ok=True)
    raise RuntimeError(f"failed to download {url!r}: {last_err}")


def _write_label(boxes, dst: Path) -> None:
    """Write a YOLO label file from [[cls,cx,cy,w,h], ...]."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for b in boxes:
        cls = int(b[0])
        cx, cy, w, h = (float(b[1]), float(b[2]), float(b[3]), float(b[4]))
        # clamp to valid range; skip degenerate boxes
        if w <= 0 or h <= 0:
            continue
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        w = min(w, 1.0)
        h = min(h, 1.0)
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _safe_stem(file_name: str) -> str:
    """Unique-ish stem keeping original name but sanitised."""
    stem = Path(file_name).stem
    for ch in ' <>:"/\\|?*':
        stem = stem.replace(ch, "_")
    return stem


# --------------------------------------------------------------------------- #
# Main build routine
# --------------------------------------------------------------------------- #
def build_dataset(ndjson_path: Path = NDJSON_PATH,
                  root: Path = DATASET_ROOT,
                  workers: int = MAX_WORKERS,
                  verbose: bool = True) -> dict:
    """Download images + write labels and data.yaml. Returns a stats dict."""
    ndjson_path = Path(ndjson_path)
    root = Path(root)
    meta, images = _read_records(ndjson_path)
    class_names = {int(k): v for k, v in meta.get("class_names", {}).items()}
    if not class_names:
        raise ValueError("No class_names found in ndjson metadata")

    for split in SPLITS:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = {
        "class_names": class_names,
        "num_classes": len(class_names),
        "splits": {s: {"images": 0, "boxes": 0, "per_class": {}} for s in SPLITS},
        "downloaded_bytes": 0,
        "failed": [],
    }

    # --- write label files first (fast, serial) ---
    jobs = []  # (split, stem, url, boxes)
    for rec in images:
        split = rec.get("split", "train")
        if split not in SPLITS:
            split = "train"
        stem = _safe_stem(rec["file"])
        boxes = rec.get("annotations", {}).get("boxes", [])
        _write_label(boxes, root / "labels" / split / f"{stem}.txt")
        jobs.append((split, stem, rec["url"], boxes))

    # --- download images in parallel ---
    lock = threading.Lock()
    done = 0
    total = len(jobs)

    def _task(job):
        split, stem, url, boxes = job
        dst = root / "images" / split / f"{stem}.jpg"
        nbytes = _download(url, dst)
        return split, boxes, nbytes

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_task, j): j for j in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                split, boxes, nbytes = fut.result()
                with lock:
                    s = stats["splits"][split]
                    s["images"] += 1
                    s["boxes"] += len(boxes)
                    for b in boxes:
                        c = str(int(b[0]))
                        s["per_class"][c] = s["per_class"].get(c, 0) + 1
                    stats["downloaded_bytes"] += nbytes
                    done += 1
                    if verbose and (done % 100 == 0 or done == total):
                        print(f"  [{done}/{total}] downloaded/verified")
            except Exception as exc:  # noqa: BLE001
                stats["failed"].append({"file": job[1], "split": job[0], "error": str(exc)})
                if verbose:
                    print(f"  !! FAILED {job[1]}: {exc}")

    # --- data.yaml ---
    yaml_path = root / "data.yaml"
    names_block = "\n".join(f"  {i}: {n}" for i, n in sorted(class_names.items()))
    yaml_text = (
        f"# Rockfall/Landslide detection dataset (auto-generated)\n"
        f"path: {root.resolve().as_posix()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {len(class_names)}\n"
        f"names:\n{names_block}\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")

    (root / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    if verbose:
        print("\n=== dataset build summary ===")
        for s in SPLITS:
            d = stats["splits"][s]
            print(f"  {s:5s}: {d['images']:5d} images, {d['boxes']:6d} boxes, per-class {d['per_class']}")
        print(f"  downloaded: {stats['downloaded_bytes'] / 1e6:.1f} MB, failed: {len(stats['failed'])}")
        print(f"  data.yaml : {yaml_path.resolve()}")
    return stats


if __name__ == "__main__":
    build_dataset()
