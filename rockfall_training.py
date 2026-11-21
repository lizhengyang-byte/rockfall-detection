"""
rockfall_training.py
====================
Reusable training-runtime helpers for the rockfall notebook, kept out of the
.ipynb so the notebook cells stay short and legible.

Contents
--------
* ``TrainingHistory``      – per-batch / per-epoch metric recorder
* ``make_callbacks``       – Ultralytics callback factory that fills a history
* ``plot_training_curves`` – loss / mAP / LR / gradient-norm / epoch-time curves
* ``plot_gradient_norms_by_layer`` – per-layer gradient-flow figure
* ``plot_lr_schedule``     – warm-up + schedule visualisation
* ``evaluate_model``       – full metric report (P/R/F1, per-class, COCO mAP)
* ``plot_confusion_matrix``– normalised confusion matrix coloured by class
* ``plot_pr_f1_curves``    – precision-recall & F1-confidence curves
* ``plot_detection_gallery`` – prediction gallery with GT vs Pred
* ``summarise_detection_sizes`` – bbox size distribution (authoring aid)
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# SCI-friendly defaults ------------------------------------------------------
matplotlib.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": True,
})

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf", "#8c564b"]


# --------------------------------------------------------------------------- #
# History recorder
# --------------------------------------------------------------------------- #
class TrainingHistory:
    """Accumulates everything a training-curve figure needs."""

    def __init__(self):
        # per-batch (gradient descent dynamics)
        self.it_loss = []          # mean of (box+cls+dfl) loss per optimizer step
        self.it_box = []
        self.it_cls = []
        self.it_dfl = []
        self.it_grad_norm = []     # total L2 norm of all parameter grads
        self.it_lr = []            # learning rate of param group 0
        self.it_epoch = []         # fractional epoch x
        # per-layer gradient norm of the last recorded step
        self.layer_grad_norms = defaultdict(float)
        # per-epoch (validation metrics from trainer.metrics)
        self.epochs = []
        self.ep_train_box = []
        self.ep_train_cls = []
        self.ep_train_dfl = []
        self.ep_precision = []
        self.ep_recall = []
        self.ep_map50 = []
        self.ep_map5095 = []
        self.ep_lr = []
        self.ep_time = []

    def __len__(self):
        return len(self.epochs)


def _grad_norm(model):
    """Return (total L2 grad-norm, {layer_name: L2 grad-norm})."""
    total_sq = 0.0
    per_layer = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        n = float(g.norm().item())
        total_sq += n * n
        # aggregate to top-level module group
        layer = name.split(".")[0] + "." + (name.split(".")[1] if "." in name else "")
        per_layer[layer] = per_layer.get(layer, 0.0) + n * n
    per_layer = {k: math.sqrt(v) for k, v in per_layer.items()}
    return math.sqrt(total_sq), per_layer


def make_callbacks(history: TrainingHistory):
    """Build a dict of Ultralytics callbacks that fill ``history``."""

    def on_train_start(trainer):
        history.n_start = trainer.epochs
        history.n_batches = len(trainer.train_loader)

    def on_train_batch_end(trainer):
        # loss items average over batches (Ultralytics keeps running mean in tloss)
        tl = trainer.tloss or {}
        box = float(tl.get("box_loss", 0.0))
        cls = float(tl.get("cls_loss", 0.0))
        dfl = float(tl.get("dfl_loss", 0.0))
        history.it_box.append(box)
        history.it_cls.append(cls)
        history.it_dfl.append(dfl)
        history.it_loss.append(box + cls + dfl)
        history.it_epoch.append(float(trainer.epoch))
        try:
            history.it_lr.append(trainer.optimizer.param_groups[0]["lr"])
        except Exception:
            history.it_lr.append(float("nan"))
        # Gradient norm: Ultralytics steps + zero_grads inside the loop (after this
        # callback), so .grad here is the per-step accumulated gradient. To remove
        # the saw-tooth left by micro-batch gradient accumulation we divide the
        # measured norm by sqrt(accumulate) -> a per-step-equivalent norm.
        try:
            total, per_layer = _grad_norm(trainer.model)
            acc = max(int(getattr(trainer, "accumulate", 1)), 1)
            history.it_grad_norm.append(total / math.sqrt(acc))
            history.layer_grad_norms = {k: v / math.sqrt(acc) for k, v in per_layer.items()}
        except Exception:
            history.it_grad_norm.append(float("nan"))

    def on_fit_epoch_end(trainer):
        m = trainer.metrics or {}
        history.epochs.append(trainer.epoch + 1)
        tl = trainer.tloss or {}
        history.ep_train_box.append(float(tl.get("box_loss", 0.0)))
        history.ep_train_cls.append(float(tl.get("cls_loss", 0.0)))
        history.ep_train_dfl.append(float(tl.get("dfl_loss", 0.0)))
        history.ep_precision.append(float(m.get("metrics/precision(B)", 0.0)))
        history.ep_recall.append(float(m.get("metrics/recall(B)", 0.0)))
        history.ep_map50.append(float(m.get("metrics/mAP50(B)", 0.0)))
        history.ep_map5095.append(float(m.get("metrics/mAP50-95(B)", 0.0)))
        try:
            history.ep_lr.append(float(trainer.optimizer.param_groups[0]["lr"]))
        except Exception:
            history.ep_lr.append(float("nan"))
        history.ep_time.append(getattr(trainer, "epoch_time", float("nan")))

    return {
        "on_train_start": on_train_start,
        "on_train_batch_end": on_train_batch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
    }


# --------------------------------------------------------------------------- #
# Training-process figures
# --------------------------------------------------------------------------- #
def _smooth(x, k=50):
    x = np.asarray(x, dtype=float)
    if len(x) < k or k < 2:
        return x
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="valid")


def plot_training_curves(history: TrainingHistory, out_dir: str | Path, class_names=None):
    """Create and save the core training-process figures."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved = {}

    # ---- 1. per-iteration loss (gradient descent dynamics) -----------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    it = np.arange(len(history.it_loss))
    ax.plot(it, history.it_loss, color=PALETTE[0], lw=0.6, alpha=0.35, label="raw")
    if len(it) > 50:
        sm = _smooth(history.it_loss, 50)
        ax.plot(np.arange(49, 49 + len(sm)), sm, color=PALETTE[1], lw=1.8, label="moving avg (50)")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("total loss (box+cls+dfl)")
    ax.set_title("Per-step Training Loss (gradient-descent dynamics)")
    ax.legend()
    p = out / "curve_step_loss.png"; fig.savefig(p); saved["step_loss"] = p; plt.show(); plt.close(fig)

    # ---- 2. decomposed losses (box / cls / dfl), smoothed ----------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(len(history.it_box))
    for i, (name, arr) in enumerate([("box_loss", history.it_box),
                                     ("cls_loss", history.it_cls),
                                     ("dfl_loss", history.it_dfl)]):
        sm = _smooth(arr, 50)
        xx = np.arange(49, 49 + len(sm)) if len(arr) > 50 else xs
        ax.plot(xx, sm, color=PALETTE[i], lw=1.6, label=name)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss (moving avg)")
    ax.set_title("Loss Decomposition (Box / Class / DFL)")
    ax.legend()
    p = out / "curve_step_loss_decomposed.png"; fig.savefig(p); saved["loss_decomp"] = p; plt.show(); plt.close(fig)

    # ---- 3. per-epoch metrics ---------------------------------------------
    ep = history.epochs
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(ep, history.ep_map50, "-o", ms=3, color=PALETTE[2], label="mAP@50")
    axes[0].plot(ep, history.ep_map5095, "-s", ms=3, color=PALETTE[3], label="mAP@50-95")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("mAP"); axes[0].legend()
    axes[0].set_title("Validation mAP per epoch"); axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].plot(ep, history.ep_precision, "-o", ms=3, color=PALETTE[0], label="Precision")
    axes[1].plot(ep, history.ep_recall, "-s", ms=3, color=PALETTE[1], label="Recall")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("score"); axes[1].legend()
    axes[1].set_title("Validation P / R per epoch"); axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    p = out / "curve_epoch_metrics.png"; fig.savefig(p); saved["epoch_metrics"] = p; plt.show(); plt.close(fig)

    # ---- 4. epoch losses ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ep, history.ep_train_box, "-o", ms=3, color=PALETTE[0], label="box")
    ax.plot(ep, history.ep_train_cls, "-s", ms=3, color=PALETTE[1], label="cls")
    ax.plot(ep, history.ep_train_dfl, "-^", ms=3, color=PALETTE[2], label="dfl")
    ax.set_xlabel("epoch"); ax.set_ylabel("train loss (epoch mean)"); ax.legend()
    ax.set_title("Training Loss Convergence per Epoch"); ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    p = out / "curve_epoch_loss.png"; fig.savefig(p); saved["epoch_loss"] = p; plt.show(); plt.close(fig)

    # ---- 5. gradient norm --------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(len(history.it_grad_norm))
    ax.plot(xs, history.it_grad_norm, color=PALETTE[4], lw=0.6, alpha=0.35)
    sm = _smooth(history.it_grad_norm, 50)
    if len(sm):
        xx = np.arange(49, 49 + len(sm)) if len(history.it_grad_norm) > 50 else xs
        ax.plot(xx, sm, color=PALETTE[1], lw=1.8, label="moving avg (50)")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("L2 grad norm")
    ax.set_title("Gradient-Norm Evolution (training stability)"); ax.legend()
    p = out / "curve_grad_norm.png"; fig.savefig(p); saved["grad_norm"] = p; plt.show(); plt.close(fig)

    # ---- 6. LR schedule ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(np.arange(len(history.it_lr)), history.it_lr, color=PALETTE[0], lw=1.4)
    ax.set_xlabel("optimizer step"); ax.set_ylabel("learning rate")
    ax.set_title("Learning-Rate Schedule (warm-up + decay)")
    p = out / "curve_lr_schedule.png"; fig.savefig(p); saved["lr"] = p; plt.show(); plt.close(fig)

    # ---- 7. epoch time -----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.0))
    ax.bar(ep, history.ep_time, color=PALETTE[5], alpha=0.85)
    ax.set_xlabel("epoch"); ax.set_ylabel("seconds")
    ax.set_title("Epoch Wall-Time")
    p = out / "bars_epoch_time.png"; fig.savefig(p); saved["epoch_time"] = p; plt.show(); plt.close(fig)

    return saved


def plot_gradient_norms_by_layer(history: TrainingHistory, out_dir: str | Path):
    """Horizontal bar chart of the last recorded per-layer gradient norms."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    norms = dict(sorted(history.layer_grad_norms.items(), key=lambda kv: kv[1], reverse=True))
    if not norms:
        return None
    names = list(norms.keys())
    vals = list(norms.values())
    fig, ax = plt.subplots(figsize=(9, max(3.5, 0.32 * len(names))))
    y = np.arange(len(names))
    ax.barh(y, vals, color=PALETTE[0], alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("L2 gradient norm")
    ax.set_title("Gradient Flow by Module Group (last step)")
    p = out / "bar_grad_norm_by_layer.png"; fig.savefig(p); plt.show(); plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def evaluate_model(model, data: str | Path, split: str = "val", imgsz: int = 640,
                   conf: float = 0.001, iou: float = 0.6, device: int | str = 0,
                   plots: bool = True):
    """Run the Ultralytics validator; return (results, metrics_dict)."""
    res = model.val(data=str(data), split=split, imgsz=imgsz, conf=conf, iou=iou,
                    device=device, plots=plots, verbose=True)
    m = res.box
    report = {
        "overall": {
            "precision": float(np.mean(m.p)),
            "recall": float(np.mean(m.r)),
            "f1": float(np.mean(2 * m.p * m.r / (m.p + m.r + 1e-16))),
            "map50": float(m.map50),
            "map": float(m.map),
            "mp": float(m.mp),
            "mr": float(m.mr),
        },
        "per_class": {},
    }
    for c in m.ap_class_index:
        c = int(c)
        cls_name = res.names.get(c, str(c)) if isinstance(res.names, dict) else str(c)
        report["per_class"][cls_name] = {
            "precision": float(m.p[c]), "recall": float(m.r[c]),
            "ap50": float(m.ap50[c]), "ap": float(m.ap[c]),
        }
    return res, report


def plot_confusion_matrix(cm, class_names, out_dir: str | Path, normalize: bool = True):
    """Plot a confusion matrix (numpy array). Returns saved path."""
    from ultralytics.utils.metrics import ConfusionMatrix  # not needed, kept for parity
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cm = np.asarray(cm, dtype=float)
    if normalize and cm.sum() > 0:
        cm = cm / (cm.sum(axis=0, keepdims=True) + 1e-16)  # column-normalised (per true class)
    labels = list(class_names.values()) if isinstance(class_names, dict) else list(class_names)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() if cm.max() > 0 else 1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels) + 1 if cm.shape[0] == len(labels) + 1 else cm.shape[0]))
    ylabels = labels + ["background"] if cm.shape[0] == len(labels) + 1 else labels
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (column-normalised)" if normalize else ""))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("proportion" if normalize else "count")
    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.2f}" if normalize else f"{int(v)}",
                        ha="center", va="center", fontsize=8,
                        color="white" if v > cm.max() * 0.5 else "black")
    ax.grid(False)
    p = out / "confusion_matrix.png"; fig.savefig(p); plt.show(); plt.close(fig)
    return p


def plot_pr_f1_curves(results, out_dir: str | Path):
    """Precision-Recall curves and F1-vs-confidence from validator results."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    eps = np.finfo(np.float64).eps
    px = results.box.px if hasattr(results.box, "px") else None

    # PR curve: precision as a function of confidence, per class
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    try:
        conf = np.linspace(0, 1, len(results.box.p[0]))
        for k in range(len(results.box.p)):
            cls = results.names[k] if k in results.names else str(k)
            ax.plot(conf, results.box.p[k], color=PALETTE[k % len(PALETTE)], lw=1.6,
                    label=f"{cls} (AP50={results.box.ap50[k]:.3f})")
        ax.set_xlabel("Confidence"); ax.set_ylabel("Precision")
        ax.set_title("Precision vs Confidence (per class)"); ax.legend(loc="lower left", fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        p1 = out / "pr_curves.png"; fig.savefig(p1); plt.show(); plt.close(fig)
    except Exception:
        p1 = None; plt.close(fig)

    # F1 vs confidence
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    try:
        conf = np.linspace(0, 1, results.box.p.shape[1])
        f1 = 2 * results.box.p * results.box.r / (results.box.p + results.box.r + eps)
        f1c = f1.mean(axis=0)
        ax.plot(conf, f1c, color=PALETTE[1], lw=1.8)
        best = int(np.argmax(f1c))
        ax.axvline(conf[best], ls="--", color="gray", alpha=0.7)
        ax.set_xlabel("Confidence"); ax.set_ylabel("F1"); ax.set_title("F1 vs Confidence")
        p2 = out / "f1_confidence.png"; fig.savefig(p2); plt.show(); plt.close(fig)
    except Exception:
        p2 = None; plt.close(fig)
    return p1, p2


def plot_detection_gallery(model, source, out_dir: str | Path, names=None,
                           imgsz=640, conf=0.25, max_images=8, seed=0):
    """Grid of test images with predictions. Saves a gallery + returns predictions."""
    import cv2
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    src = Path(source)
    files = sorted(src.glob("*.jpg"))
    rng = np.random.default_rng(seed)
    if len(files) > max_images:
        files = list(rng.choice(files, size=max_images, replace=False))
    results = model.predict([str(f) for f in files], imgsz=imgsz, conf=conf, verbose=False)
    n = len(results); cols = min(4, n); rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, r in zip(axes, results):
        im = r.plot(labels=True, conf=True)  # BGR with drawn boxes
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(Path(r.path).name, fontsize=8); ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    p = out / "detection_gallery.png"; fig.savefig(p); plt.show(); plt.close(fig)
    return p, results


def summarise_detection_sizes(data_yaml: str | Path, imgsz: int = 640):
    """Return bbox width/height (px at imgsz) lists for a data.yaml (authoring aid)."""
    import yaml
    cfg = yaml.safe_load(open(data_yaml, encoding="utf-8"))
    root = Path(cfg["path"])
    widths, heights = [], []
    for split in ("train", "val", "test"):
        lbl = root / "labels" / split
        if not lbl.exists():
            continue
        for f in lbl.glob("*.txt"):
            for line in f.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) == 5:
                    widths.append(float(parts[3]) * imgsz)
                    heights.append(float(parts[4]) * imgsz)
    return np.array(widths), np.array(heights)
