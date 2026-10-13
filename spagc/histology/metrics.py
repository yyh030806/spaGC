"""Segmentation confusion matrices, agreement and tile evaluation."""

from __future__ import annotations

import numpy as np
import torch


def confusion_matrix(
    pred: np.ndarray, label: np.ndarray, num_classes: int, ignore_idx: int = 0
) -> np.ndarray:
    """Build `cm[ground_truth, prediction]` while excluding ignored ground-truth pixels."""
    valid = label != ignore_idx
    predicted = pred[valid].astype(np.int64)
    truth = label[valid].astype(np.int64)
    in_range = (
        (predicted >= 0)
        & (predicted < num_classes)
        & (truth >= 0)
        & (truth < num_classes)
    )
    (predicted, truth) = (predicted[in_range], truth[in_range])
    cm = np.bincount(
        truth * num_classes + predicted, minlength=num_classes * num_classes
    )
    return cm.reshape(num_classes, num_classes)


def metrics_from_cm(cm: np.ndarray, ignore_idx: int = 0) -> dict:
    """Calculate foreground segmentation metrics and per-class IoU from a confusion matrix."""
    num_classes = cm.shape[0]
    gt = cm.sum(axis=1)
    pr = cm.sum(axis=0)
    tp = np.diag(cm).astype(np.float64)
    denom_iou = gt + pr - tp
    total = gt.sum()
    (ious, dices, recalls, fwiou) = ([], [], [], 0.0)
    per_class = {}
    for c in range(num_classes):
        if c == ignore_idx or gt[c] == 0:
            continue
        iou = tp[c] / denom_iou[c] if denom_iou[c] > 0 else 0.0
        dice = 2 * tp[c] / (gt[c] + pr[c]) if gt[c] + pr[c] > 0 else 0.0
        recall = tp[c] / gt[c] if gt[c] > 0 else 0.0
        ious.append(iou)
        dices.append(dice)
        recalls.append(recall)
        fwiou += gt[c] / total * iou if total > 0 else 0.0
        per_class[c] = (float(iou), float(dice), int(gt[c]))
    accuracy = float(tp.sum() / total) if total > 0 else 0.0
    agree = agreement_metrics_from_cm(cm)
    return {
        "miou": float(np.mean(ious)) if ious else 0.0,
        "fwiou": float(fwiou),
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else 0.0,
        "mean_acc": float(np.mean(recalls)) if recalls else 0.0,
        "dice": float(np.mean(dices)) if dices else 0.0,
        "kappa": agree["kappa"],
        "mcc": agree["mcc"],
        "per_class_iou": per_class,
    }


def agreement_metrics_from_cm(cm: np.ndarray) -> dict:
    """Calculate Cohen's kappa and multiclass MCC from a foreground confusion matrix."""
    c = cm.astype(np.float64)
    n = c.sum()
    if n < 1:
        return {"kappa": 0.0, "mcc": 0.0}
    t = c.sum(axis=1)
    p = c.sum(axis=0)
    correct = np.trace(c)
    po = correct / n
    pe = float((t * p).sum()) / (n * n)
    kappa = 0.0 if 1 - pe == 0 else (po - pe) / (1 - pe)
    cov_xy = correct * n - float((t * p).sum())
    cov_xx = n * n - float((p * p).sum())
    cov_yy = n * n - float((t * t).sum())
    denom = (cov_xx * cov_yy) ** 0.5
    mcc = 0.0 if denom == 0 else cov_xy / denom
    return {"kappa": float(kappa), "mcc": float(mcc)}


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    ignore_idx=0,
    num_niche_classes=None,
    num_domain_classes=None,
    criterion_domain=None,
    criterion_niche=None,
    loss_weights=None,
):
    """Evaluate both heads and optionally calculate validation loss.

    The loss arguments are optional so existing inference/evaluation callers
    keep the original metrics-only behavior.  When supplied, ``val_loss`` and
    the two head losses are averaged over validation batches.
    """
    model.eval()
    matrices = {"domain": None, "niche": None}
    sizes = {"domain": num_domain_classes, "niche": num_niche_classes}
    calculate_loss = criterion_domain is not None or criterion_niche is not None
    if calculate_loss and (criterion_domain is None or criterion_niche is None):
        raise ValueError(
            "criterion_domain and criterion_niche must be supplied together"
        )
    loss_weights = loss_weights or {"domain": 1.0, "niche": 1.0}
    loss_sums = {"domain": 0.0, "niche": 0.0, "total": 0.0}
    loss_batches = 0
    for batch in loader:
        if isinstance(batch, dict):
            (feat, rgb) = (batch["feat"], batch["rgb"])
            labels = (
                batch.get("domain_label"),
                batch.get("niche_label", batch.get("label")),
            )
        else:
            (feat, rgb, niche, domain) = batch[:4]
            labels = (domain, niche)
        outputs = model(feat.to(device).float(), rgb.to(device).float())
        if calculate_loss:
            targets = tuple(
                (
                    label.to(device, non_blocking=True)
                    if isinstance(label, torch.Tensor)
                    else label
                )
                for label in labels
            )
            loss_domain = (
                criterion_domain(outputs[0], targets[0])
                if targets[0] is not None
                else outputs[0].sum() * 0.0
            )
            loss_niche = (
                criterion_niche(outputs[1], targets[1])
                if targets[1] is not None
                else outputs[1].sum() * 0.0
            )
            loss_sums["domain"] += float(loss_domain.detach().cpu())
            loss_sums["niche"] += float(loss_niche.detach().cpu())
            loss_sums["total"] += float(
                loss_weights["domain"] * loss_domain.detach().cpu()
                + loss_weights["niche"] * loss_niche.detach().cpu()
            )
            loss_batches += 1
        for head, logits, label in zip(("domain", "niche"), outputs, labels):
            classes = sizes[head] or logits.shape[1]
            if classes != logits.shape[1]:
                raise ValueError(f"{head} class count disagrees with model")
            if matrices[head] is None:
                matrices[head] = np.zeros((classes, classes), dtype=np.int64)
            if label is not None:
                target = np.asarray(
                    label.cpu() if isinstance(label, torch.Tensor) else label
                )
                if np.any(
                    (target != ignore_idx) & ((target < 0) | (target >= classes))
                ):
                    raise ValueError(f"{head} label ID out of range")
                matrices[head] += confusion_matrix(
                    logits.argmax(1).cpu().numpy(), target, classes, ignore_idx
                )
    result = {}
    for head, cm in matrices.items():
        if cm is None:
            cm = np.zeros((sizes[head] or 1, sizes[head] or 1), dtype=np.int64)
        result.update(
            {f"{head}_{k}": v for (k, v) in metrics_from_cm(cm, ignore_idx).items()}
        )
    result.update({key: result[f"niche_{key}"] for key in ("miou", "dice", "accuracy")})
    if calculate_loss:
        if loss_batches:
            result.update(
                {
                    "val_loss": loss_sums["total"] / loss_batches,
                    "val_domain_loss": loss_sums["domain"] / loss_batches,
                    "val_niche_loss": loss_sums["niche"] / loss_batches,
                }
            )
        else:
            result.update(
                {
                    "val_loss": float("nan"),
                    "val_domain_loss": float("nan"),
                    "val_niche_loss": float("nan"),
                }
            )
    return result
