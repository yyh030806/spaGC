"""Segmentation losses, training and fixed-budget leave-out evaluation."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm

from spagc.histology.dataset import CTransPathPatchDataset, build_slide_dataset
from spagc.histology.metadata import (
    counts_with_background,
    validate_patient_partition,
    validate_training_slides,
)
from spagc.histology.metrics import evaluate
from spagc.histology.model import CTransPathRGBFusionSeg


class FocalLoss(nn.Module):
    """Multiclass focal loss with optional class weights and an ignored label index."""

    def __init__(self, gamma: float = 2.0, weight=None, ignore_index: int = 0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        unweighted = F.cross_entropy(
            logits, target, ignore_index=self.ignore_index, reduction="none"
        )
        pt = torch.exp(-unweighted)
        loss = (1.0 - pt) ** self.gamma * ce
        mask = target != self.ignore_index
        if mask.any():
            return loss[mask].mean()
        return (loss * 0.0).sum()


class SafeCrossEntropyLoss(nn.CrossEntropyLoss):
    """An absent head contributes differentiable zero rather than NaN."""

    def forward(self, logits, target):
        if not torch.any(target != self.ignore_index):
            return logits.sum() * 0.0
        return super().forward(logits, target)


def spatial_train_val_split(
    dataset, h5_path, val_ratio: float = 0.4, split_axis: str = "x"
):
    """Split observations spatially to prevent leakage from random neighboring tiles."""
    if not 0 < val_ratio < 1 or split_axis not in ("x", "y"):
        raise ValueError("Require 0 < val_ratio < 1 and axis x/y")
    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:]
        sizes = f["valid_size"][:]
    axis = 0 if split_axis == "x" else 1
    starts = coords[:, axis]
    extent = sizes[:, 1 - axis]
    boundary = np.quantile(starts, 1 - val_ratio, method="higher")
    train_indices = np.flatnonzero(starts + extent <= boundary).tolist()
    val_indices = np.flatnonzero(starts >= boundary).tolist()
    if not train_indices or not val_indices:
        raise ValueError("Slide is too small for a nonoverlapping spatial split")
    return (Subset(dataset, train_indices), Subset(dataset, val_indices))


def slide_balanced_sample_weights(dataset):
    """Weight samples inversely by slide size so slides have equal sampling probability."""
    if isinstance(dataset, ConcatDataset):
        lens = [len(d) for d in dataset.datasets]
    else:
        lens = [len(dataset)]
    weights = []
    for length in lens:
        weights.extend([1.0 / max(length, 1)] * length)
    return torch.tensor(weights, dtype=torch.double)


def compute_class_weights(
    slides,
    label_key: str,
    num_classes: int,
    ignore_idx: int = 0,
    clip: tuple[float, float] = (0.2, 8.0),
):
    """Calculate median-frequency class weights with background weight set to zero."""
    counts = np.zeros(num_classes, dtype=np.float64)
    for s in slides:
        p = s.get(label_key)
        if not p or not os.path.exists(p):
            continue
        a = np.fromfile(p, dtype=np.uint16)
        if a.size and a.max() >= num_classes:
            raise ValueError(
                f"{p}: class count must include background and every label ID"
            )
        counts += np.bincount(a, minlength=num_classes)
    counts[ignore_idx] = 0
    present = counts > 0
    weights = np.zeros(num_classes, dtype=np.float32)
    if present.sum() == 0:
        weights[:] = 1.0
        weights[ignore_idx] = 0.0
        return torch.tensor(weights)
    freq = counts / counts.sum()
    med = np.median(freq[present])
    weights[present] = np.clip(med / freq[present], clip[0], clip[1])
    weights[ignore_idx] = 0.0
    return torch.tensor(weights, dtype=torch.float32)


def build_spatial_split_datasets(
    slides,
    num_niche_classes: int = 23,
    ignore_idx: int = 0,
    region_size: int = 28,
    use_rgb_region: bool = True,
    load_h5_RAM: str | bool = "Auto",
    val_ratio: float = 0.2,
    split_axis: str = "x",
    standardize_features: bool = False,
    augment: bool = False,
    feature_noise_std: float = 0.05,
    rgb_jitter: float = 0.1,
):
    """Construct per-slide spatial train/validation splits and combine them across slides."""
    (train_parts, val_parts) = ([], [])
    for slide in slides:
        dataset = CTransPathPatchDataset(
            h5_path=slide["h5_path"],
            label_path=slide.get("niche_label_path", slide.get("label_path")),
            label_shape=slide["label_shape"],
            num_clusters=num_niche_classes,
            he_tif_path=slide["he_tif_path"],
            ignore_idx=ignore_idx,
            load_h5_RAM=load_h5_RAM,
            region_size=region_size,
            use_rgb_region=use_rgb_region,
            return_meta=False,
            domain_label_path=slide.get("domain_label_path"),
            standardize_features=standardize_features,
            augment=augment,
            feature_noise_std=feature_noise_std,
            rgb_jitter=rgb_jitter,
        )
        (train_sub, val_sub) = spatial_train_val_split(
            dataset=dataset,
            h5_path=slide["h5_path"],
            val_ratio=val_ratio,
            split_axis=split_axis,
        )
        if augment:
            dataset.augment_index_set = set(int(i) for i in train_sub.indices)
        train_parts.append(train_sub)
        val_parts.append(val_sub)
    train_set = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    val_set = val_parts[0] if len(val_parts) == 1 else ConcatDataset(val_parts)
    return (train_set, val_set)


def _save_training_history(history, out_dir):
    """Persist the complete history and refresh a compact training plot."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "history.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    fields = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_domain_loss",
        "val_niche_loss",
        "niche_accuracy",
        "domain_accuracy",
        "learning_rate",
    ]
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "[WARN] matplotlib is unavailable; history.json and metrics.csv were saved, but no plot was generated."
        )
        return
    epochs = [row["epoch"] for row in history]

    def values(key):
        return [row.get(key) for row in history]

    (fig, axes) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    panels = [
        (("train_loss", "val_loss"), "Loss", "Loss"),
        (("niche_accuracy", "domain_accuracy"), "Validation accuracy", "Accuracy"),
    ]
    labels = {
        "train_loss": "train loss",
        "val_loss": "validation loss",
        "niche_accuracy": "niche accuracy",
        "domain_accuracy": "domain accuracy",
    }
    for axis, (keys, title, ylabel) in zip(axes.flat, panels):
        for key in keys:
            y = values(key)
            if any(item is not None for item in y):
                axis.plot(epochs, y, marker="o", markersize=3, label=labels[key])
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        (handles, _) = axis.get_legend_handles_labels()
        if handles:
            axis.legend()
    fig.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close(fig)


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    scaler=None,
    use_amp=True,
    max_grad_norm=1.0,
    loss_weights=None,
    criterion_domain=None,
    criterion_niche=None,
):
    criterion_domain = criterion_domain or criterion
    criterion_niche = criterion_niche or criterion
    model.train()
    total_loss = 0.0
    valid_steps = 0
    skipped_empty = 0
    skipped_nan = 0
    amp = bool(use_amp and torch.device(device).type == "cuda")
    loss_weights = loss_weights or {"domain": 1.0, "niche": 1.0}
    for feat, rgb, niche_label, domain_label in loader:
        feat = feat.to(device, non_blocking=True)
        rgb = rgb.to(device, non_blocking=True, memory_format=torch.channels_last)
        niche_label = niche_label.to(device, non_blocking=True)
        domain_label = domain_label.to(device, non_blocking=True)
        if not amp:
            feat = feat.float()
        if (niche_label != 0).sum().item() == 0 and (
            domain_label != 0
        ).sum().item() == 0:
            skipped_empty += 1
            continue
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            (domain_pred, niche_pred) = model(feat, rgb)
            loss_domain = criterion_domain(domain_pred, domain_label)
            loss_niche = criterion_niche(niche_pred, niche_label)
            loss = (
                loss_weights["domain"] * loss_domain
                + loss_weights["niche"] * loss_niche
            )
        if not torch.isfinite(loss):
            skipped_nan += 1
            print("[WARN] non-finite loss detected, skip this batch")
            continue
        if scaler is not None and amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        total_loss += float(loss.detach().cpu())
        valid_steps += 1
    if valid_steps == 0:
        return float("nan")
    return total_loss / valid_steps


def train_mul_slices(
    out_dir: str,
    device: torch.device | str | None = None,
    train_slides: list[dict] | None = None,
    val_slides: list[dict] | None = None,
    num_domain_classes: int = 5,
    num_niche_classes: int = 21,
    batch_size: int = 64,
    ignore_idx: int = 0,
    lr: float = 0.0001,
    weight_decay: float = 0.0001,
    epochs: int = 30,
    num_workers: int = 8,
    load_h5_RAM: str | bool = "Auto",
    cache_h5: bool | None = None,
    loss_weights: dict | None = None,
    standardize_features: bool = True,
    augment: bool = True,
    dropout: float = 0.1,
    class_balanced_loss: bool = True,
    slide_balanced_sampler: bool = True,
    val_interval: int = 2,
    spatial_split: dict | None = None,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
    lr_schedule: str = "constant",
    warmup_epochs: int = 3,
    feature_noise_std: float = 0.05,
    rgb_jitter: float = 0.1,
    model_config: dict | None = None,
    use_amp: bool = True,
    max_grad_norm: float = 1.0,
    save_interval: int = 5,
    checkpoint_selection: str = "validation",
):
    """Train joint segmentation; choose validation or fixed-budget final weights explicitly."""
    from torch.utils.data import WeightedRandomSampler

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if train_slides is None or val_slides is None:
        raise ValueError("train_slides and val_slides are required")
    if min(epochs, batch_size, val_interval, save_interval) <= 0:
        raise ValueError("epochs, batch_size and intervals must be positive")
    if checkpoint_selection not in {"validation", "final"}:
        raise ValueError("checkpoint_selection must be validation or final")
    if checkpoint_selection == "final" and spatial_split:
        raise ValueError("Final-checkpoint evaluation requires disjoint whole slides")
    if (
        checkpoint_selection == "final"
        and os.path.isdir(out_dir)
        and os.listdir(out_dir)
    ):
        raise FileExistsError("Use a fresh directory for final-checkpoint evaluation")
    if cache_h5 is not None:
        load_h5_RAM = "Yes" if cache_h5 else "No"
    validate_training_slides(
        train_slides,
        val_slides,
        num_domain_classes,
        num_niche_classes,
        bool(spatial_split),
    )
    (num_domain_output_classes, num_niche_output_classes) = counts_with_background(
        num_domain_classes, num_niche_classes
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"Using {device}")
    loss_weights = loss_weights or {"domain": 1.0, "niche": 1.0}
    if spatial_split:
        all_slides = list({s["h5_path"]: s for s in train_slides + val_slides}.values())
        weight_slides = all_slides
        print(
            f"[spatial-split] {len(all_slides)} slides, val_ratio={spatial_split.get('val_ratio', 0.2)}, axis={spatial_split.get('split_axis', 'x')}"
        )
        (train_set, val_set) = build_spatial_split_datasets(
            slides=all_slides,
            num_niche_classes=num_niche_output_classes,
            ignore_idx=ignore_idx,
            region_size=28,
            use_rgb_region=True,
            load_h5_RAM=load_h5_RAM,
            val_ratio=spatial_split.get("val_ratio", 0.2),
            split_axis=spatial_split.get("split_axis", "x"),
            standardize_features=standardize_features,
            augment=augment,
            feature_noise_std=feature_noise_std,
            rgb_jitter=rgb_jitter,
        )
    else:
        weight_slides = train_slides
        train_set = build_slide_dataset(
            slides=train_slides,
            num_niche_classes=num_niche_output_classes,
            ignore_idx=ignore_idx,
            region_size=28,
            use_rgb_region=True,
            load_h5_RAM=load_h5_RAM,
            standardize_features=standardize_features,
            augment=augment,
            feature_noise_std=feature_noise_std,
            rgb_jitter=rgb_jitter,
        )
        val_set = build_slide_dataset(
            slides=val_slides,
            num_niche_classes=num_niche_output_classes,
            ignore_idx=ignore_idx,
            region_size=28,
            use_rgb_region=True,
            load_h5_RAM=load_h5_RAM,
            standardize_features=standardize_features,
            augment=False,
        )
    print(f"Datasets ready: train={len(train_set)}, val={len(val_set)}")
    train_loader_kwargs = dict(
        dataset=train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.device(device).type == "cuda",
        drop_last=False,
    )
    if slide_balanced_sampler:
        sample_weights = slide_balanced_sample_weights(train_set)
        train_loader_kwargs["sampler"] = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(train_set), replacement=True
        )
    else:
        train_loader_kwargs["shuffle"] = True
    val_loader_kwargs = dict(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.device(device).type == "cuda",
        drop_last=False,
    )
    if num_workers > 0:
        train_loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
        val_loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(**train_loader_kwargs)
    val_loader = DataLoader(**val_loader_kwargs)
    architecture = dict(model_config or {})
    architecture.update(
        num_domain_classes=num_domain_output_classes,
        num_niche_classes=num_niche_output_classes,
        region_grid=8,
        dropout=dropout,
    )
    model = CTransPathRGBFusionSeg(**architecture).to(device)
    if class_balanced_loss and spatial_split:
        raise ValueError(
            "For spatial split, disable class balancing to avoid using validation label frequencies"
        )
    if class_balanced_loss:
        niche_w = compute_class_weights(
            weight_slides, "niche_label_path", num_niche_output_classes, ignore_idx
        ).to(device)
        domain_w = compute_class_weights(
            weight_slides, "domain_label_path", num_domain_output_classes, ignore_idx
        ).to(device)
        print(
            f"[class weights] niche range [{niche_w.min():.2f},{niche_w.max():.2f}], domain range [{domain_w.min():.2f},{domain_w.max():.2f}]"
        )
    else:
        niche_w = domain_w = None
    if use_focal:
        criterion_niche = FocalLoss(
            gamma=focal_gamma, weight=niche_w, ignore_index=ignore_idx
        )
        criterion_domain = FocalLoss(
            gamma=focal_gamma, weight=domain_w, ignore_index=ignore_idx
        )
        print(f"[loss] Focal (gamma={focal_gamma}) + class weights")
    else:
        criterion_niche = SafeCrossEntropyLoss(ignore_index=ignore_idx, weight=niche_w)
        criterion_domain = SafeCrossEntropyLoss(
            ignore_index=ignore_idx, weight=domain_w
        )
    criterion = criterion_niche
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if lr_schedule == "cosine":

        def lr_lambda(ep):
            if ep < warmup_epochs:
                return (ep + 1) / max(1, warmup_epochs)
            progress = (ep - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        print(f"[lr] cosine schedule, warmup={warmup_epochs} epochs")
    else:
        scheduler = None
    scaler = torch.amp.GradScaler("cuda", enabled=torch.device(device).type == "cuda")
    pbar = tqdm(range(1, epochs + 1), desc="Training...", leave=True)
    best_niche_miou = -1.0
    best_metrics = {}
    history = []
    for epoch in pbar:
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            use_amp=use_amp,
            max_grad_norm=max_grad_norm,
            loss_weights=loss_weights,
            criterion_domain=criterion_domain,
            criterion_niche=criterion_niche,
        )
        eval_metrics = {}
        if checkpoint_selection == "validation" and (
            epoch % val_interval == 0 or epoch == epochs
        ):
            eval_metrics = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                ignore_idx=ignore_idx,
                num_niche_classes=num_niche_output_classes,
                num_domain_classes=num_domain_output_classes,
                criterion_domain=criterion_domain,
                criterion_niche=criterion_niche,
                loss_weights=loss_weights,
            )
            ckpt = {
                "model_config": architecture,
                "real_class_counts": {
                    "domain": int(num_domain_classes),
                    "niche": int(num_niche_classes),
                    "background_id": 0,
                },
                "standardize_features": standardize_features,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(ckpt, os.path.join(out_dir, "model_latest.pth"))
            if eval_metrics.get("niche_miou", 0) > best_niche_miou:
                best_niche_miou = eval_metrics["niche_miou"]
                best_metrics = {**eval_metrics, "epoch": epoch}
                torch.save(ckpt, os.path.join(out_dir, "model_best.pth"))
        if epoch % save_interval == 0 or epoch == epochs:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_config": architecture,
                    "real_class_counts": {
                        "domain": int(num_domain_classes),
                        "niche": int(num_niche_classes),
                        "background_id": 0,
                    },
                    "standardize_features": standardize_features,
                },
                os.path.join(out_dir, "model_latest.pth"),
            )

        def metric_or_none(name):
            value = eval_metrics.get(name)
            if value is None:
                return None
            value = float(value)
            return value if math.isfinite(value) else None

        history.append(
            {
                "epoch": epoch,
                "train_loss": (
                    float(train_loss) if math.isfinite(float(train_loss)) else None
                ),
                "val_loss": metric_or_none("val_loss"),
                "val_domain_loss": metric_or_none("val_domain_loss"),
                "val_niche_loss": metric_or_none("val_niche_loss"),
                "niche_accuracy": metric_or_none("niche_accuracy"),
                "domain_accuracy": metric_or_none("domain_accuracy"),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        _save_training_history(history, out_dir)
        with open(os.path.join(out_dir, "run.log"), "a", encoding="utf-8") as f:
            print(
                f"epoch {epoch}: train loss = {train_loss:.8f} | val loss={eval_metrics.get('val_loss', float('nan')):.8f} | niche Acc={eval_metrics.get('niche_accuracy', float('nan')):.4f} | domain Acc={eval_metrics.get('domain_accuracy', float('nan')):.4f}",
                file=f,
            )
        if scheduler is not None:
            scheduler.step()
        pbar.set_postfix(
            {
                "train loss": f"{train_loss:.4f}",
                "val loss": f"{eval_metrics.get('val_loss', float('nan')):.4f}",
                "niche Acc": f"{eval_metrics.get('niche_accuracy', float('nan')):.4f}",
                "domain Acc": f"{eval_metrics.get('domain_accuracy', float('nan')):.4f}",
            }
        )
    if checkpoint_selection == "final":
        checkpoint = {
            "model_config": architecture,
            "real_class_counts": {
                "domain": int(num_domain_classes),
                "niche": int(num_niche_classes),
                "background_id": 0,
            },
            "standardize_features": standardize_features,
            "epoch": epochs,
            "planned_epochs": epochs,
            "checkpoint_selection": "final",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(out_dir, "model_final.pth"))
        held_out_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            ignore_idx=ignore_idx,
            num_niche_classes=num_niche_output_classes,
            num_domain_classes=num_domain_output_classes,
            criterion_domain=criterion_domain,
            criterion_niche=criterion_niche,
            loss_weights=loss_weights,
        )
        final_metrics = {
            "test_" + key[4:] if key.startswith("val_") else key: (
                None
                if isinstance(value, float) and (not math.isfinite(value))
                else value
            )
            for (key, value) in held_out_metrics.items()
        }
        final_metrics.update(
            epoch=epochs,
            checkpoint="model_final.pth",
            checkpoint_selection="final",
            held_out_evaluated_after_checkpoint_save=True,
        )
        Path(out_dir, "held_out_metrics.json").write_text(
            json.dumps(final_metrics, indent=2, allow_nan=False), encoding="utf-8"
        )
        return final_metrics
    return best_metrics


def leave_one_slide_out(slides, out_dir, group_key=None, **options):
    """Train final-epoch folds, optionally leaving all slides from a patient out."""
    options = dict(options)
    if options.pop("checkpoint_selection", "final") != "final":
        raise ValueError(
            "Held-out data cannot select checkpoints; use final selection."
        )
    if options.get("spatial_split"):
        raise ValueError(
            "Spatial splitting cannot mix the omitted group into training."
        )
    if int(options.get("epochs", 30)) <= 0:
        raise ValueError("Declare a positive epoch budget before fitting folds.")
    unique = {}
    for slide in slides:
        name = slide["name"]
        if (
            not isinstance(name, str)
            or name in {"", ".", ".."}
            or Path(name).name != name
        ):
            raise ValueError("Slide names must be safe single directory names.")
        if name in unique and slide["h5_path"] != unique[name]["h5_path"]:
            raise ValueError("Conflicting slide names.")
        unique[name] = slide
    records = list(unique.values())
    paths = [str(Path(s["h5_path"]).resolve()) for s in records]
    if len(set(paths)) != len(paths):
        raise ValueError("One feature file is declared under multiple slide names.")
    has_patient = [bool(s.get("patient_id")) for s in records]
    if any(has_patient) and (not all(has_patient)):
        raise ValueError("Supply patient IDs for every slide or none.")
    if group_key is None:
        group_key = "patient_id" if all(has_patient) else "name"
    groups = {}
    for slide in records:
        group = slide.get(group_key)
        if (
            not group
            or not isinstance(group, str)
            or group in {".", ".."}
            or (Path(group).name != group)
        ):
            raise ValueError("Each fold needs a nonmissing safe group identifier.")
        groups.setdefault(group, []).append(slide)
    if len(groups) < 2:
        raise ValueError("At least two independent evaluation groups are required.")
    root = Path(out_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            "Use a fresh fold output directory; previous results are preserved."
        )
    root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for group, held_out in groups.items():
        training = [s for (key, items) in groups.items() if key != group for s in items]
        verified = validate_patient_partition(training, held_out)
        summary[group] = train_mul_slices(
            out_dir=str(root / group),
            train_slides=training,
            val_slides=held_out,
            checkpoint_selection="final",
            **options,
        )
        summary[group].update(
            evaluation_unit=group_key,
            held_out_slides=[s["name"] for s in held_out],
            patient_independence_verified=verified,
        )
        (root / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False)
        )
    return summary
