"""Training lifecycle and physical-coordinate volume reconstruction."""

from __future__ import annotations

import json
import os

import anndata as ad
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy.sparse
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader

from spagc.reconstruction.dataset import DeepSpatialDataset
from spagc.reconstruction.histology import HistologyConfig, HistologyRuntime
from spagc.reconstruction.model import GiT
from spagc.reconstruction.training import DeepSpatialModule

DEFAULT_PROPAGATED_LABEL_KEYS = ("pred_domain", "pred_niche")


class _AnchorPreparation:
    def __init__(self):
        """
        Initializes an empty DeepSpatial instance.
        Configurations and paths are injected during functional method calls.
        """
        self.dataset = None
        self.train_loader = None
        self.gene_dim = None
        self.gene_names = None
        self.num_classes = None
        self.model = None
        self.module = None
        self.categories = None
        self.spatial_stats = None
        self.spatial_key = None
        self.z_key = None
        self.label_key = None
        self.propagated_label_keys = ()
        self.model_config = {}
        self.train_config = {}
        self.histology_config = None
        self.histology_runtime = None
        self.use_celltype = True
        self.uot_config = {
            "solver": "dense",
            "top_k": None,
            "bidirectional": True,
            "candidate_radius": None,
        }

    def _normalize_spatial(self, adata_list: list[ad.AnnData]) -> None:
        """
        Extracts coordinates from user keys and computes normalization stats.
        Stores normalized data in internal 'spatial_norm' and 'z_norm' keys.
        """
        z_raw_list = []
        all_spatial = []
        for i, adata in enumerate(adata_list):
            if self.spatial_key not in adata.obsm:
                raise KeyError(
                    f"Spatial key '{self.spatial_key}' not found in slice {i}."
                )
            if self.z_key not in adata.obs:
                raise KeyError(f"Z-coord key '{self.z_key}' not found in slice {i}.")
            all_spatial.append(adata.obsm[self.spatial_key])
            z_raw_list.append(adata.obs[self.z_key].iloc[0])
        all_spatial = np.vstack(all_spatial)
        z_raw_arr = np.array(z_raw_list)
        self.spatial_stats = {
            "x_min": float(all_spatial[:, 0].min()),
            "x_range": float(all_spatial[:, 0].max() - all_spatial[:, 0].min() + 1e-08),
            "y_min": float(all_spatial[:, 1].min()),
            "y_range": float(all_spatial[:, 1].max() - all_spatial[:, 1].min() + 1e-08),
            "z_min": float(z_raw_arr.min()),
            "z_range": float(z_raw_arr.max() - z_raw_arr.min() + 1e-08),
        }
        norm_z_arr = (z_raw_arr - self.spatial_stats["z_min"]) / self.spatial_stats[
            "z_range"
        ]
        for i, adata in enumerate(adata_list):
            coords = adata.obsm[self.spatial_key].copy()
            if not np.issubdtype(coords.dtype, np.floating):
                coords = coords.astype(float)
            coords[:, 0] = (
                coords[:, 0] - self.spatial_stats["x_min"]
            ) / self.spatial_stats["x_range"]
            coords[:, 1] = (
                coords[:, 1] - self.spatial_stats["y_min"]
            ) / self.spatial_stats["y_range"]
            adata.obsm["spatial_norm"] = coords
            adata.obs["z_norm"] = norm_z_arr[i]

    def setup_data(
        self,
        adata_list: list[ad.AnnData],
        spatial_key: str = "spatial",
        z_key: str = "z_coord",
        label_key: str = "cell_class",
        batch_size: int = 128,
        num_workers: int = 4,
        n_samples_base: int = 50000,
        alpha_spatial: float = 0.5,
        uot_reg: float = 0.8,
        uot_tau: float = 0.05,
        uot_solver: str = "dense",
        uot_top_k: int | None = None,
        uot_bidirectional: bool = True,
        uot_candidate_radius: float | None = None,
        mode: str = "fit",
        use_celltype: bool = True,
        histology=None,
    ):
        """
        Prepares the data pipeline and calculates physical normalization statistics.

        Args:
            adata_list: List of AnnData objects (slices).
            spatial_key: Key in `.obsm` for XY coordinates.
            z_key: Key in `.obs` for the physical Z coordinate.
            label_key: Key in `.obs` for cell type annotations.
            batch_size: Number of samples per training batch.
            num_workers: Multi-process data loading workers.
            n_samples_base: Base number of cell pairs to sample per slice pair.
            alpha_spatial: UOT spatial distance weight.
            uot_reg: Entropy regularization for UOT.
            uot_tau: Marginal relaxation for UOT.
            uot_solver: 'dense' or 'sparse_topk' UOT backend.
            uot_top_k: Number of spatial candidates for sparse UOT.
            uot_bidirectional: Include reverse spatial kNN candidates.
            uot_candidate_radius: Optional candidate radius in solver coordinates.
            mode: Dataset mode ('fit' for training, 'predict' for inference).
        """
        self.spatial_key = spatial_key
        self.z_key = z_key
        self.label_key = label_key
        self.use_celltype = use_celltype
        self.uot_config = {
            "solver": uot_solver,
            "top_k": uot_top_k,
            "bidirectional": uot_bidirectional,
            "candidate_radius": uot_candidate_radius,
        }
        self.histology_config = (
            histology
            if isinstance(histology, HistologyConfig)
            else HistologyConfig(**histology or {})
        )
        if self.histology_config.use_histology:
            pl.seed_everything(self.histology_config.seed, workers=True)
        if len(adata_list) < 2 or any(a.n_obs == 0 for a in adata_list):
            raise ValueError("At least two nonempty anchor slices required")
        if any(not a.var_names.equals(adata_list[0].var_names) for a in adata_list):
            raise ValueError(
                "Anchor gene identities/order must agree; align var_names first"
            )
        self._resolve_propagated_label_keys_for_anchors(adata_list)
        self.gene_names = adata_list[0].var_names.tolist()
        if self.histology_config.use_histology:
            zs = []
            for a in adata_list:
                z = np.asarray(a.obs[z_key], float)
                if not np.isfinite(z).all() or not np.all(z == z[0]):
                    raise ValueError("One finite physical Z per anchor required")
                if not np.isfinite(a.obsm[spatial_key]).all():
                    raise ValueError("Invalid physical XY")
                zs.append(z[0])
            if not np.all(np.diff(zs) > 0):
                raise ValueError("Sort anchors by increasing physical Z")
        self._normalize_spatial(adata_list)
        self.histology_runtime = None
        if self.histology_config.use_histology:
            self.histology_runtime = HistologyRuntime(
                self.histology_config, self.spatial_stats
            )
            for a in adata_list:
                self.histology_runtime.section(a)
        self.dataset = DeepSpatialDataset(
            adata_list=adata_list,
            spatial_key="spatial_norm",
            z_key="z_norm",
            label_key=label_key,
            n_samples_base=n_samples_base,
            alpha_spatial=alpha_spatial,
            uot_reg=uot_reg,
            uot_tau=uot_tau,
            uot_solver=uot_solver,
            uot_top_k=uot_top_k,
            uot_bidirectional=uot_bidirectional,
            uot_candidate_radius=uot_candidate_radius,
            mode=mode,
            use_celltype=use_celltype,
            histology_runtime=self.histology_runtime,
        )
        self.categories = pd.Index(self.dataset.label_encoder.classes_)
        if not len(self.dataset):
            raise ValueError(
                "No UOT pairs sampled; inspect coupling or increase n_samples_base"
            )
        self.train_loader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        batch = next(iter(self.train_loader))
        self.gene_dim = batch["g0"].shape[1]
        self.num_classes = batch["c0"].shape[1]


class _ModelLifecycle:
    def build_model(
        self,
        patch_size: int = 8,
        hidden_size: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        path_type: str = "Linear",
        lr: float = 0.0002,
        weight_decay: float = 1e-05,
        lambda_g: float = 0.1,
        lambda_c: float = 10.0,
        sampling_method: str = "dopri5",
        atol: float = 1e-05,
        rtol: float = 1e-05,
    ):
        """
        Instantiates the GiT network architecture and Flow Matching logic.

        Args:
            patch_size: Tokenization patch size for spatial coordinates.
            hidden_size: Transformer embedding dimension.
            depth: Number of transformer layers.
            num_heads: Number of attention heads.
            mlp_ratio: Expansion ratio for MLP layers.
            path_type: Probability path type for Flow Matching.
            lr: Learning rate for training.
            weight_decay: L2 regularization weight.
            lambda_g: Loss weight for gene expression reconstruction.
            lambda_c: Loss weight for cell type classification.
            sampling_method: ODE solver for inference (e.g., 'dopri5', 'euler').
            atol: Absolute tolerance for ODE solver.
            rtol: Relative tolerance for ODE solver.
        """
        if self.gene_dim is None or self.num_classes is None:
            raise ValueError("Dimensions unknown. Call `setup_data()` first.")
        self.model_config = {
            "patch_size": patch_size,
            "hidden_size": hidden_size,
            "depth": depth,
            "num_heads": num_heads,
            "mlp_ratio": mlp_ratio,
        }
        self.train_config = {
            "path_type": path_type,
            "prediction": "velocity",
            "train_eps": 0.02,
            "sample_eps": 0.02,
            "ema_decay": 0.999,
            "lr": lr,
            "weight_decay": weight_decay,
            "lambda_g": lambda_g,
            "lambda_c": lambda_c,
            "sampling_method": sampling_method,
            "atol": atol,
            "rtol": rtol,
        }
        self.train_config["use_celltype"] = self.use_celltype
        self.model = GiT(
            gene_dim=self.gene_dim, num_classes=self.num_classes, **self.model_config
        )
        self.module = DeepSpatialModule(
            self.train_config, self.model, self.histology_runtime
        )

    def _save_config(self, save_dir: str):
        """Internal helper to persist metadata and configurations."""
        os.makedirs(save_dir, exist_ok=True)
        config = {
            "gene_dim": self.gene_dim,
            "gene_names": self.gene_names,
            "num_classes": self.num_classes,
            "spatial_stats": self.spatial_stats,
            "categories": (
                self.categories.tolist() if self.categories is not None else None
            ),
            "spatial_key": self.spatial_key,
            "z_key": self.z_key,
            "label_key": self.label_key,
            "propagated_label_keys": list(self.propagated_label_keys),
            "model_config": self.model_config,
            "train_config": self.train_config,
            "use_celltype": self.use_celltype,
            "uot_config": self.uot_config,
            "histology_config": (
                self.histology_config.to_dict() if self.histology_config else None
            ),
            "feature_revision": (
                self.histology_runtime.store.revision
                if self.histology_runtime
                else None
            ),
        }
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config, f)

    def load_checkpoint(
        self, ckpt_path: str, config_path: str = None, sampling_method: str = "dopri5"
    ):
        """
        Loads model weights and metadata for inference or resuming.

        Args:
            ckpt_path: Path to the `.ckpt` file.
            config_path: Path to `config.json`. Defaults to the same folder as ckpt_path.
            sampling_method: Overrides the ODE solver for this inference session.
        """
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        if config_path is None:
            config_path = os.path.join(os.path.dirname(ckpt_path), "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            self.gene_dim = config["gene_dim"]
            self.gene_names = config.get("gene_names")
            self.num_classes = config["num_classes"]
            self.spatial_stats = config["spatial_stats"]
            self.spatial_key = config.get("spatial_key", "spatial")
            self.z_key = config.get("z_key", "z_coord")
            self.label_key = config["label_key"]
            self.propagated_label_keys = tuple(config.get("propagated_label_keys", ()))
            self.categories = pd.Index(config["categories"])
            self.uot_config = config.get(
                "uot_config",
                {
                    "solver": "dense",
                    "top_k": None,
                    "bidirectional": True,
                    "candidate_radius": None,
                },
            )
            self.model_config = config.get("model_config", {})
            self.train_config = config.get("train_config", {})
            self.use_celltype = config.get("use_celltype", True)
            self.histology_config = HistologyConfig(
                **config.get("histology_config") or {}
            )
            self.histology_runtime = None
            if self.histology_config.use_histology:
                self.histology_runtime = HistologyRuntime(
                    self.histology_config, self.spatial_stats
                )
                if (
                    config.get("feature_revision")
                    != self.histology_runtime.store.revision
                ):
                    raise ValueError(
                        "Feature cache changed since checkpoint; use the training feature revision"
                    )
        else:
            raise ValueError("Metadata 'config.json' not found. Cannot rebuild model.")
        self.train_config["sampling_method"] = sampling_method
        self.model = GiT(
            gene_dim=self.gene_dim, num_classes=self.num_classes, **self.model_config
        )
        self.module = DeepSpatialModule(
            self.train_config, self.model, self.histology_runtime
        )
        checkpoint = torch.load(
            ckpt_path, map_location=torch.device("cpu"), weights_only=True
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        self.module.load_state_dict(state_dict)

    def fit(
        self,
        max_epochs: int = 100,
        save_dir: str = "./checkpoints",
        accelerator: str = "auto",
        devices: str = "auto",
        save_ckpt: bool = False,
        resume_ckpt_path: str = None,
    ):
        """
        Executes the training loop using PyTorch Lightning.

        Args:
            max_epochs: Max training epochs.
            save_dir: Directory to save checkpoints and metadata.
            accelerator: Hardware accelerator ('auto', 'gpu', 'cpu', 'mps').
            devices: Number of devices or indices ('auto', 1, [0, 1]).
            save_ckpt: Whether to save progress.
            resume_ckpt_path: Path to resume full training state.
        """
        if save_ckpt:
            self._save_config(save_dir)
        callbacks = []
        if save_ckpt:
            callbacks.append(
                ModelCheckpoint(
                    dirpath=save_dir,
                    monitor="loss",
                    filename="deepspatial-{epoch:02d}-{loss:.4f}",
                    save_top_k=1,
                    mode="min",
                )
            )
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            callbacks=callbacks,
            enable_checkpointing=save_ckpt,
            logger=False,
        )
        trainer.fit(self.module, self.train_loader, ckpt_path=resume_ckpt_path)


class _PhysicalCoordinates:
    def _restore_3d_physical_coords(self, adata_3d: ad.AnnData) -> ad.AnnData:
        """Internal helper to map [0,1] coordinates back to physical scale."""
        if self.spatial_stats is None:
            raise ValueError(
                "Physical stats missing. Ensure setup_data or load_checkpoint was called."
            )
        adata_3d.obsm[self.spatial_key][:, 0] = (
            adata_3d.obsm[self.spatial_key][:, 0] * self.spatial_stats["x_range"]
            + self.spatial_stats["x_min"]
        )
        adata_3d.obsm[self.spatial_key][:, 1] = (
            adata_3d.obsm[self.spatial_key][:, 1] * self.spatial_stats["y_range"]
            + self.spatial_stats["y_min"]
        )
        adata_3d.obs[self.z_key] = (
            adata_3d.obs[self.z_key] * self.spatial_stats["z_range"]
            + self.spatial_stats["z_min"]
        )
        adata_3d.obs_names = [
            f"cell_3d_z{z:.4f}_{i}" for (i, z) in enumerate(adata_3d.obs[self.z_key])
        ]
        return adata_3d


class _GapReconstruction:
    @torch.no_grad()
    def reconstruct_between_slices(
        self,
        adata0: ad.AnnData,
        adata1: ad.AnnData,
        thickness: float,
        steps: int = 20,
        chunk_size: int = 2048,
        device: str = "auto",
        cache_celltype_trajectory: bool = False,
    ) -> ad.AnnData:
        """
        Generates a 3D volume segment between two specific AnnData slices.

        Args:
            adata0: Source slice AnnData (must contain 'spatial_norm' and 'z_norm').
            adata1: Target slice AnnData (must contain 'spatial_norm' and 'z_norm').
            steps: Number of integration steps for the ODE solver.
            thickness: Physical distance (um) between generated cells, controlling density.
            chunk_size: Batch size for ODE integration to manage VRAM usage.
            device: Computing device ('auto', 'cuda', 'cpu', or specific 'cuda:n').

        Returns:
            An AnnData object containing the interpolated 3D segment in physical coordinates.
        """
        if self.module is None or self.spatial_stats is None:
            raise ValueError("Build or load a model before reconstruction")
        if steps < 2 or chunk_size < 1:
            raise ValueError("steps>=2 and chunk_size>=1 required")
        if cache_celltype_trajectory and (not self.use_celltype):
            raise ValueError("cache_celltype_trajectory requires use_celltype=True")
        self._prepare_reconstruction_inputs(adata0, adata1)
        if device == "auto":
            dev = (
                self.module.device
                if self.module.device.type != "cpu"
                else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        else:
            dev = torch.device(device)
        if self.module.device != dev:
            self.module.to(dev)
        self.module.eval()
        (x0, z0, g0, c0, x1, z1, g1, c1, target_cells, total_cells) = (
            self._setup_and_extract(adata0, adata1, thickness, dev)
        )
        mix_data = self._generate_and_prune_optimized(
            x0,
            g0,
            c0,
            x1,
            g1,
            c1,
            z0,
            z1,
            steps,
            target_cells,
            total_cells,
            dev,
            chunk_size,
            cache_celltype_trajectory=cache_celltype_trajectory,
        )
        adata_segment = self._assemble_fast_anndata(adata0, adata1, mix_data)
        return self._restore_3d_physical_coords(adata_segment)


class _HECellInference:
    @torch.no_grad()
    def predict_on_he_cells(
        self,
        he_cells: ad.AnnData,
        adata0: ad.AnnData,
        adata1: ad.AnnData,
        steps: int = 20,
        chunk_size: int = 512,
        device: str = "auto",
        trajectory_cache: dict | None = None,
    ) -> ad.AnnData:
        """Predict one molecular state for every supplied H&E cell.

        The source anchor cells are integrated from ``adata0`` toward ``adata1``.
        For each H&E query plane, every query coordinate is assigned the nearest
        propagated source trajectory in registered physical XY.  The returned
        AnnData keeps the H&E coordinates as its spatial coordinates and stores
        the propagated coordinate and nearest-neighbour distance for QC.

        This is a cell-level inference interface, not a one-to-one lineage
        tracker: several H&E cells may use the same nearby propagated anchor
        trajectory.  ``chunk_size`` controls ODE memory usage only.  When
        ``trajectory_cache`` is supplied, the continuous cell-type trajectory
        selected for each H&E query cell is stored during this same ODE pass;
        callers can materialize labels later without integrating again.
        """
        if self.module is None or self.spatial_stats is None:
            raise ValueError("Build or load a model before H&E-cell prediction")
        if steps < 2 or chunk_size < 1:
            raise ValueError("steps>=2 and chunk_size>=1 required")
        if self.spatial_key not in he_cells.obsm:
            raise KeyError(f"H&E cells must contain obsm[{self.spatial_key!r}]")
        if self.z_key not in he_cells.obs:
            raise KeyError(f"H&E cells must contain obs[{self.z_key!r}]")
        if not adata0.n_obs or not adata1.n_obs:
            raise ValueError("Both anchor slices must be nonempty")
        if trajectory_cache is not None:
            trajectory_cache.clear()
            trajectory_cache.update({"version": 1, "steps": int(steps), "planes": {}})
        self._prepare_reconstruction_inputs(adata0, adata1)
        query_xy = np.asarray(he_cells.obsm[self.spatial_key], dtype=np.float64)
        query_z = np.asarray(he_cells.obs[self.z_key], dtype=np.float64)
        if query_xy.shape != (he_cells.n_obs, 2) or not np.isfinite(query_xy).all():
            raise ValueError(
                "H&E cells must provide finite physical XY coordinates [N,2]"
            )
        if query_z.shape != (he_cells.n_obs,) or not np.isfinite(query_z).all():
            raise ValueError("H&E cells must provide finite physical Z coordinates")
        z0_phys = float(adata0.obs[self.z_key].iloc[0])
        z1_phys = float(adata1.obs[self.z_key].iloc[0])
        if z1_phys <= z0_phys:
            raise ValueError("Anchor slices must have increasing physical Z")
        if (query_z < z0_phys - 1e-06).any() or (query_z > z1_phys + 1e-06).any():
            raise ValueError("H&E query Z coordinates must lie between the two anchors")
        if device == "auto":
            dev = (
                self.module.device
                if self.module.device.type != "cpu"
                else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        else:
            dev = torch.device(device)
        if self.module.device != dev:
            self.module.to(dev)
        self.module.eval()

        def extract_anchor(adata):
            x = torch.tensor(
                adata.obsm["spatial_norm"], dtype=torch.float32, device=dev
            )
            g_arr = (
                adata.X.toarray()
                if scipy.sparse.issparse(adata.X)
                else np.asarray(adata.X)
            )
            g = torch.tensor(g_arr, dtype=torch.float32, device=dev)
            if self.use_celltype and self.label_key in adata.obs:
                codes = pd.Categorical(
                    adata.obs[self.label_key], categories=self.categories
                ).codes.astype(np.int64)
                if (codes < 0).any():
                    raise ValueError(
                        "Unknown/missing source cell type at H&E prediction"
                    )
                c = torch.nn.functional.one_hot(
                    torch.tensor(codes, dtype=torch.long, device=dev),
                    num_classes=self.num_classes,
                ).float()
            else:
                c = torch.zeros(
                    (adata.n_obs, self.num_classes), dtype=torch.float32, device=dev
                )
            return (x, g, c)

        (x0, g0, c0) = extract_anchor(adata0)
        z0_norm = float(adata0.obs["z_norm"].iloc[0])
        z1_norm = float(adata1.obs["z_norm"].iloc[0])
        delta_z = z1_norm - z0_norm
        scale = np.array(
            [self.spatial_stats["x_range"], self.spatial_stats["y_range"]],
            dtype=np.float64,
        )
        origin = np.array(
            [self.spatial_stats["x_min"], self.spatial_stats["y_min"]], dtype=np.float64
        )
        predicted_xy = np.empty_like(query_xy, dtype=np.float64)
        assigned_xy = np.empty_like(query_xy, dtype=np.float64)
        predicted_g = np.empty((he_cells.n_obs, g0.shape[1]), dtype=np.float32)
        predicted_c = np.zeros(he_cells.n_obs, dtype=np.int64)
        prediction_distance = np.empty(he_cells.n_obs, dtype=np.float32)
        for z_phys in np.unique(query_z):
            query_ids = np.flatnonzero(np.isclose(query_z, z_phys, rtol=0, atol=1e-06))
            t_value = float((z_phys - z0_phys) / (z1_phys - z0_phys))
            propagated_xy = []
            propagated_g = []
            propagated_c = []
            propagated_c_cont = []
            for start in range(0, x0.shape[0], chunk_size):
                stop = min(start + chunk_size, x0.shape[0])
                batch = {
                    "x0": x0[start:stop],
                    "g0": g0[start:stop],
                    "c0": c0[start:stop],
                    "z0": torch.full((stop - start, 1), z0_norm, device=dev),
                    "z1": torch.full((stop - start, 1), z1_norm, device=dev),
                    "delta_z": torch.full((stop - start, 1), delta_z, device=dev),
                }
                if t_value <= 1e-08:
                    x_t = batch["x0"]
                    g_t = batch["g0"]
                    c_t = torch.argmax(batch["c0"], dim=-1)
                    c_traj = batch["c0"].unsqueeze(0).expand(steps, -1, -1)
                else:
                    result = self.module.sample(batch, mode="ODE", steps=steps)
                    scaled_t = t_value * (steps - 1)
                    low = min(int(np.floor(scaled_t)), steps - 1)
                    high = min(low + 1, steps - 1)
                    weight = float(scaled_t - low)
                    x_t = (1 - weight) * result["x_traj"][low] + weight * result[
                        "x_traj"
                    ][high]
                    g_t = (1 - weight) * result["g_traj"][low] + weight * result[
                        "g_traj"
                    ][high]
                    c_index = min(int(np.round(scaled_t)), steps - 1)
                    c_t = result["c_traj_discrete"][c_index]
                    c_traj = result["c_traj_cont"]
                    del result
                propagated_xy.append(x_t.detach().cpu().numpy() * scale + origin)
                propagated_g.append(
                    torch.clamp(g_t, min=0).detach().cpu().numpy().astype(np.float32)
                )
                propagated_c.append(c_t.detach().cpu().numpy().astype(np.int64))
                propagated_c_cont.append(
                    c_traj.detach().cpu().numpy().astype(np.float32)
                )
            propagated_xy = np.concatenate(propagated_xy, axis=0)
            propagated_g = np.concatenate(propagated_g, axis=0)
            propagated_c = np.concatenate(propagated_c, axis=0)
            propagated_c_cont = np.concatenate(propagated_c_cont, axis=1)
            (distances, nearest) = cKDTree(propagated_xy).query(
                query_xy[query_ids], k=1
            )
            predicted_xy[query_ids] = query_xy[query_ids]
            assigned_xy[query_ids] = propagated_xy[nearest]
            predicted_g[query_ids] = propagated_g[nearest]
            predicted_c[query_ids] = propagated_c[nearest]
            prediction_distance[query_ids] = distances.astype(np.float32)
            if trajectory_cache is not None:
                trajectory_cache["planes"][str(float(z_phys))] = {
                    "z_um": float(z_phys),
                    "t_values": np.full(len(query_ids), t_value, dtype=np.float32),
                    "query_ids": query_ids.astype(np.int64),
                    "source_indices": nearest.astype(np.int64),
                    "c_traj_cont": propagated_c_cont[:, nearest, :].astype(
                        np.float32, copy=False
                    ),
                }
        obs = he_cells.obs.copy()
        obs["prediction_distance_um"] = prediction_distance
        obs["prediction_source"] = "anchor0_forward_nearest"
        if self.categories is not None:
            obs[self.label_key] = pd.Categorical.from_codes(
                predicted_c, categories=self.categories
            )
        result = ad.AnnData(
            X=scipy.sparse.csr_matrix(predicted_g),
            obs=obs,
            var=adata0.var.copy(),
            obsm={self.spatial_key: predicted_xy, "spatial_flow": assigned_xy},
        )
        return result


class _VolumeAssembly:
    @torch.no_grad()
    def reconstruct_full_volume(
        self,
        adata_list: list,
        thickness: float,
        steps: int = 100,
        chunk_size: int = 2048,
        device: str = "auto",
        cache_celltype_trajectory: bool = False,
    ) -> ad.AnnData:
        """
        High-level API to reconstruct the entire 3D volume from a list of slices.

        Args:
            adata_list: Ordered list of AnnData slices.
            thickness: Target physical distance (um) between cells in the Z-axis.
            steps: Number of ODE integration steps per gap.
            chunk_size: Processing batch size to prevent OOM on large datasets.
            device: Target device for the entire reconstruction process.

        Returns:
            A single merged AnnData object representing the continuous 3D volume.
        """
        from tqdm import tqdm

        segment_list = []
        num_pairs = len(adata_list) - 1
        if num_pairs < 1:
            raise ValueError("adata_list must contain at least 2 slices.")
        pbar = tqdm(range(num_pairs), desc="DeepSpatial: 3D Reconstruct", unit="gap")
        for i in pbar:
            (ad0, ad1) = (adata_list[i], adata_list[i + 1])
            z_start = ad0.obs[self.z_key].iloc[0]
            z_end = ad1.obs[self.z_key].iloc[0]
            pbar.set_postfix({"range": f"{z_start:.1f}-{z_end:.1f}um"})
            segment = self.reconstruct_between_slices(
                adata0=ad0,
                adata1=ad1,
                steps=steps,
                thickness=thickness,
                chunk_size=chunk_size,
                device=device,
                cache_celltype_trajectory=cache_celltype_trajectory,
            )
            segment_list.append(segment)
        full_volume = ad.concat(
            segment_list, join="outer", uns_merge="first", index_unique="-"
        )
        if hasattr(adata_list[0], "uns"):
            full_volume.uns.update(adata_list[0].uns)
        if cache_celltype_trajectory:
            full_volume.uns["deepspatial_celltype_trajectory_cache"] = {
                "enabled": True,
                "storage": "obsm['celltype_flow_state']",
                "semantics": "continuous cell-type state at sampled virtual Z",
                "one_pass_ode": True,
            }
        return full_volume


class _ReconstructionInputs:
    def _prepare_reconstruction_inputs(self, adata0, adata1):
        """Fresh H5AD inputs use TRAINING coordinate statistics, never refit them."""
        self._resolve_propagated_label_keys(adata0, adata1)
        if not adata0.var_names.equals(adata1.var_names):
            raise ValueError("Reconstruction gene identities/order must agree")
        stats = self.spatial_stats
        zs = []
        for a in (adata0, adata1):
            if not a.n_obs or a.n_vars != self.gene_dim:
                raise ValueError(
                    "Nonempty anchors with training gene dimension required"
                )
            if self.gene_names is not None and a.var_names.tolist() != self.gene_names:
                raise ValueError(
                    "Reconstruction gene order differs from training checkpoint"
                )
            xy = np.asarray(a.obsm[self.spatial_key])
            z = np.asarray(a.obs[self.z_key], dtype=float)
            if (
                xy.shape != (a.n_obs, 2)
                or not np.isfinite(xy).all()
                or (not np.isfinite(z).all())
                or (not np.all(z == z[0]))
            ):
                raise ValueError(
                    "Finite XY [N,2] and one physical Z per anchor required"
                )
            if self.use_celltype and (
                self.label_key not in a.obs or a.obs[self.label_key].isna().any()
            ):
                raise ValueError(
                    "Missing reconstruction labels; use a model trained with use_celltype=False"
                )
            coords = xy.copy()
            if not np.issubdtype(coords.dtype, np.floating):
                coords = coords.astype(float)
            coords[:, 0] = (coords[:, 0] - stats["x_min"]) / stats["x_range"]
            coords[:, 1] = (coords[:, 1] - stats["y_min"]) / stats["y_range"]
            a.obsm["spatial_norm"] = coords
            a.obs["z_norm"] = (z[0] - stats["z_min"]) / stats["z_range"]
            zs.append(z[0])
            if self.histology_runtime:
                self.histology_runtime.section(a)
        if self.histology_runtime and zs[1] <= zs[0]:
            raise ValueError("Reconstruction anchors must have increasing physical Z")

    def _resolve_propagated_label_keys_for_anchors(self, adata_list):
        """Validate metadata labels shared by all training anchors.

        The model only predicts ``cell_class``.  ``pred_domain`` and
        ``pred_niche`` are carried as metadata, so silently training with
        only a subset of anchors carrying them would create incomplete
        reconstruction outputs later.
        """
        present_any = {
            key: any(key in adata.obs for adata in adata_list)
            for key in DEFAULT_PROPAGATED_LABEL_KEYS
        }
        incomplete = [
            key
            for (key, seen) in present_any.items()
            if seen and (not all(key in adata.obs for adata in adata_list))
        ]
        if incomplete:
            raise ValueError(
                "Propagated labels must be present on every anchor: "
                + ", ".join(incomplete)
            )
        self.propagated_label_keys = tuple(
            key for key in DEFAULT_PROPAGATED_LABEL_KEYS if present_any[key]
        )
        for key in self.propagated_label_keys:
            if any(adata.obs[key].isna().any() for adata in adata_list):
                raise ValueError(f"Missing values in propagated label {key!r}")
        return self.propagated_label_keys

    def _resolve_propagated_label_keys(self, adata0, adata1):
        """Resolve and validate metadata labels for one reconstruction gap."""
        if self.propagated_label_keys:
            keys = tuple(self.propagated_label_keys)
        else:
            present_any = {
                key: key in adata0.obs or key in adata1.obs
                for key in DEFAULT_PROPAGATED_LABEL_KEYS
            }
            incomplete = [
                key
                for (key, seen) in present_any.items()
                if seen and (not (key in adata0.obs and key in adata1.obs))
            ]
            if incomplete:
                raise ValueError(
                    "Propagated labels must be present on both reconstruction anchors: "
                    + ", ".join(incomplete)
                )
            keys = tuple(
                key for key in DEFAULT_PROPAGATED_LABEL_KEYS if present_any[key]
            )
        for key in keys:
            for anchor_name, anchor in (("adata0", adata0), ("adata1", adata1)):
                if key not in anchor.obs:
                    raise ValueError(
                        f"{anchor_name} is missing propagated label {key!r}"
                    )
                if anchor.obs[key].isna().any():
                    raise ValueError(
                        f"{anchor_name} contains missing values in propagated label {key!r}"
                    )
        self.propagated_label_keys = keys
        return keys

    def _setup_and_extract(self, adata0, adata1, thickness, dev):
        """Calculates sampling density and maps normalized input to device."""
        if not np.isfinite(thickness) or thickness <= 0:
            raise ValueError("thickness must be positive")
        avg_n_ref = (adata0.n_obs + adata1.n_obs) / 2
        (z0, z1) = (adata0.obs[self.z_key].iloc[0], adata1.obs[self.z_key].iloc[0])
        physical_gap = abs(z1 - z0)
        target_cells = max(1, int(avg_n_ref * (physical_gap / thickness)))
        total_cells = int(target_cells)

        def extract(adata):
            x = torch.tensor(
                adata.obsm["spatial_norm"], dtype=torch.float32, device=dev
            )
            z = adata.obs["z_norm"].iloc[0]
            g_arr = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
            g = torch.tensor(g_arr, dtype=torch.float32, device=dev)
            if self.use_celltype and self.label_key in adata.obs:
                codes = pd.Categorical(
                    adata.obs[self.label_key], categories=self.categories
                ).codes.astype(np.int64)
                if (codes < 0).any():
                    raise ValueError("Unknown/missing cell type at reconstruction")
                c_idx = torch.tensor(codes, device=dev)
            else:
                c_idx = torch.zeros(adata.n_obs, device=dev, dtype=torch.long)
            return (x, z, g, c_idx)

        (x0, z0, g0, c0) = extract(adata0)
        (x1, z1, g1, c1) = extract(adata1)
        return (x0, z0, g0, c0, x1, z1, g1, c1, target_cells, total_cells)


class _ReconstructionSampler:
    def _generate_and_prune_optimized(
        self,
        x0,
        g0,
        c0,
        x1,
        g1,
        c1,
        z0,
        z1,
        steps,
        target_cells,
        total_cells,
        dev,
        chunk_size,
        cache_celltype_trajectory=False,
    ):
        """Memory-efficient ODE integration and spatial density pruning."""
        (N0, N1) = (float(x0.shape[0]), float(x1.shape[0]))
        u = torch.rand(total_cells, device=dev)
        if abs(N0 - N1) < 1e-05:
            t_vals = u
        else:
            t_vals = (-N0 + torch.sqrt(N0**2 * (1 - u) + N1**2 * u)) / (N1 - N0)
        target_zs = t_vals * (z1 - z0) + z0
        is_fwd = torch.rand(total_cells, device=dev) > t_vals
        (fwd_ids, bwd_ids) = (torch.where(is_fwd)[0], torch.where(~is_fwd)[0])
        src_fwd = torch.randint(0, x0.shape[0], (len(fwd_ids),), device=dev)
        src_bwd = torch.randint(0, x1.shape[0], (len(bwd_ids),), device=dev)
        final_x = torch.zeros((total_cells, 2), device=dev)
        final_g = torch.zeros((total_cells, g0.shape[1]), device=dev)
        final_c = torch.zeros(total_cells, device=dev, dtype=torch.long)
        final_c_cont = (
            torch.zeros((total_cells, self.num_classes), device=dev)
            if cache_celltype_trajectory
            else None
        )

        def process_direction(
            is_forward, src_indices, child_ids, x_ref, g_ref, c_ref, z_start, z_end
        ):
            if len(child_ids) == 0:
                return
            (unique_parents, inverse_indices) = torch.unique(
                src_indices, return_inverse=True
            )
            c_onehot_ref = torch.nn.functional.one_hot(
                c_ref, num_classes=self.num_classes
            ).float()
            if not self.use_celltype:
                c_onehot_ref.zero_()
            for i in range(0, len(unique_parents), chunk_size):
                end_idx = min(i + chunk_size, len(unique_parents))
                chunk_parents = unique_parents[i:end_idx]
                mask_in_chunk = (inverse_indices >= i) & (inverse_indices < end_idx)
                chunk_child_ids = child_ids[mask_in_chunk]
                if len(chunk_child_ids) == 0:
                    continue
                local_parent_idx = inverse_indices[mask_in_chunk] - i
                batch = {
                    "x0": x_ref[chunk_parents],
                    "g0": g_ref[chunk_parents],
                    "c0": c_onehot_ref[chunk_parents],
                    "z0": torch.full((len(chunk_parents), 1), z_start, device=dev),
                    "z1": torch.full((len(chunk_parents), 1), z_end, device=dev),
                    "delta_z": torch.full(
                        (len(chunk_parents), 1), z_end - z_start, device=dev
                    ),
                }
                if self.histology_runtime:
                    batch["reverse"] = not is_forward
                res = self.module.sample(batch, mode="ODE", steps=steps)
                if not is_forward:
                    res["x_traj"] = torch.flip(res["x_traj"], dims=[0])
                    res["g_traj"] = torch.flip(res["g_traj"], dims=[0])
                    res["c_traj_cont"] = torch.flip(res["c_traj_cont"], dims=[0])
                    res["c_traj_discrete"] = torch.flip(
                        res["c_traj_discrete"], dims=[0]
                    )
                chunk_t_vals = t_vals[chunk_child_ids] * (steps - 1)
                idx_low = chunk_t_vals.long()
                idx_high = torch.clamp(idx_low + 1, max=steps - 1)
                w_high = (chunk_t_vals - idx_low.float()).unsqueeze(1)
                final_x[chunk_child_ids] = (1 - w_high) * res["x_traj"][
                    idx_low, local_parent_idx
                ] + w_high * res["x_traj"][idx_high, local_parent_idx]
                final_g[chunk_child_ids] = (1 - w_high) * res["g_traj"][
                    idx_low, local_parent_idx
                ] + w_high * res["g_traj"][idx_high, local_parent_idx]
                final_c[chunk_child_ids] = res["c_traj_discrete"][
                    torch.clamp(chunk_t_vals.round().long(), max=steps - 1),
                    local_parent_idx,
                ]
                if final_c_cont is not None:
                    c_low = res["c_traj_cont"][idx_low, local_parent_idx]
                    c_high = res["c_traj_cont"][idx_high, local_parent_idx]
                    final_c_cont[chunk_child_ids] = (
                        1 - w_high
                    ) * c_low + w_high * c_high
                del res, batch
                torch.cuda.empty_cache()

        process_direction(True, src_fwd, fwd_ids, x0, g0, c0, z0, z1)
        process_direction(False, src_bwd, bwd_ids, x1, g1, c1, z1, z0)
        (fwd_src_keep, bwd_src_keep) = (src_fwd, src_bwd)
        (fwd_keep_indices, bwd_keep_indices) = (fwd_ids, bwd_ids)
        fwd_mask = is_fwd
        nnz_parent = torch.zeros(target_cells, device=dev, dtype=torch.long)
        nnz_parent[fwd_mask] = (g0 > 0).sum(dim=1)[fwd_src_keep]
        nnz_parent[~fwd_mask] = (g1 > 0).sum(dim=1)[bwd_src_keep]
        max_k = int(nnz_parent.max().item())
        if max_k > 0:
            for i in range(0, target_cells, chunk_size):
                end_idx = min(i + chunk_size, target_cells)
                (g_chunk, nnz_chunk) = (final_g[i:end_idx], nnz_parent[i:end_idx])
                (_, sorted_indices) = torch.sort(g_chunk, dim=1, descending=True)
                top_indices = sorted_indices[:, :max_k]
                mask = torch.arange(max_k, device=dev).unsqueeze(
                    0
                ) < nnz_chunk.unsqueeze(1)
                final_mask = torch.zeros_like(g_chunk, dtype=torch.bool).scatter_(
                    1, top_indices, mask
                )
                g_chunk[~final_mask] = 0.0
                final_g[i:end_idx] = g_chunk
        result = {
            "x": final_x.cpu().numpy(),
            "g": final_g,
            "c": final_c.cpu().numpy(),
            "zs": target_zs.cpu().numpy(),
            "cells": target_cells,
            "fwd_src": fwd_src_keep.cpu().numpy(),
            "bwd_src": bwd_src_keep.cpu().numpy(),
            "fwd_idx": fwd_keep_indices.cpu().numpy(),
            "bwd_idx": bwd_keep_indices.cpu().numpy(),
        }
        if final_c_cont is not None:
            result["c_cont"] = final_c_cont.cpu().numpy()
        return result

    def _assemble_fast_anndata(self, adata0, adata1, mix_data):
        """Constructs the resulting AnnData with sparse expression matrix."""
        nz_indices = mix_data["g"].nonzero(as_tuple=True)
        res_g_sparse = scipy.sparse.csr_matrix(
            (
                mix_data["g"][nz_indices].cpu().numpy(),
                (nz_indices[0].cpu().numpy(), nz_indices[1].cpu().numpy()),
            ),
            shape=(mix_data["cells"], mix_data["g"].shape[1]),
        )
        obs_fwd = adata0.obs.iloc[mix_data["fwd_src"]].copy()
        obs_fwd.index = mix_data["fwd_idx"]
        obs_bwd = adata1.obs.iloc[mix_data["bwd_src"]].copy()
        obs_bwd.index = mix_data["bwd_idx"]
        mixed_obs = pd.concat([obs_fwd, obs_bwd]).sort_index()
        mixed_obs[self.z_key] = mix_data["zs"]
        mixed_obs[self.label_key] = pd.Categorical.from_codes(
            mix_data["c"], categories=self.categories
        )
        for key in self.propagated_label_keys:
            if key not in mixed_obs.columns:
                raise RuntimeError(f"Reconstruction lost propagated label {key!r}")
            if mixed_obs[key].isna().any():
                raise RuntimeError(
                    f"Reconstruction produced missing propagated label values in {key!r}"
                )
        mixed_obs.index = mixed_obs.index.astype(str)
        obsm = {self.spatial_key: mix_data["x"]}
        if "c_cont" in mix_data:
            obsm["celltype_flow_state"] = np.asarray(
                mix_data["c_cont"], dtype=np.float32
            )
        return ad.AnnData(
            X=res_g_sparse, obs=mixed_obs, var=adata0.var.copy(), obsm=obsm
        )


class DeepSpatial(
    _AnchorPreparation,
    _ModelLifecycle,
    _PhysicalCoordinates,
    _GapReconstruction,
    _HECellInference,
    _VolumeAssembly,
    _ReconstructionInputs,
    _ReconstructionSampler,
):
    """Fit joint molecular/spatial flows and reconstruct physical tissue volumes."""
