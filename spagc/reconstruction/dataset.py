"""Transport-guided endpoint pairs for joint flow matching."""

from __future__ import annotations

import gc

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from tqdm import tqdm

from spagc.reconstruction.uot import SparseCoupling, compute_uot_coupling


class DeepSpatialDataset(Dataset):
    """
    DeepSpatial Global Trajectory Dataset.
    Constructs cross-slice cell pairs using Unbalanced Optimal Transport (UOT)
    to provide continuous training samples for Flow Matching.
    """

    def __init__(
        self,
        adata_list: list,
        spatial_key: str = "spatial_norm",
        z_key: str = "z_norm",
        label_key: str = "cell_class",
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
        histology_runtime=None,
    ):
        """
        Args:
            adata_list: List of AnnData objects.
            spatial_key: Key in .obsm containing normalized XY coordinates.
            z_key: Key in .obs containing normalized Z coordinates.
            label_key: Key in .obs for cell labels/types.
            n_samples_base: Target total number of trajectory pairs.
            alpha_spatial: Balance between spatial and gene distances for UOT.
            uot_reg: Entropy regularization for UOT solver.
            uot_tau: Marginal relaxation for UOT solver.
            uot_solver: 'dense' for the release solver or 'sparse_topk'.
            uot_top_k: Number of spatial candidates per cell for sparse UOT.
            uot_bidirectional: Add reverse kNN candidates to preserve target support.
            uot_candidate_radius: Optional candidate radius in solver coordinate units.
            mode: 'fit' for full multi-slice training, 'predict' for limited slice pairs.
        """
        self.adata_list = adata_list
        self.spatial_key = spatial_key
        self.z_key = z_key
        self.label_key = label_key
        self.n_samples_base = n_samples_base
        self.alpha_spatial = alpha_spatial
        self.uot_reg = uot_reg
        self.uot_tau = uot_tau
        if uot_solver not in ("dense", "sparse_topk"):
            raise ValueError("uot_solver must be 'dense' or 'sparse_topk'")
        if uot_solver == "sparse_topk" and (uot_top_k is None or int(uot_top_k) < 1):
            raise ValueError("uot_top_k must be a positive integer for sparse_topk")
        self.uot_solver = uot_solver
        self.uot_top_k = None if uot_top_k is None else int(uot_top_k)
        self.uot_bidirectional = bool(uot_bidirectional)
        self.uot_candidate_radius = uot_candidate_radius
        self.use_celltype = use_celltype
        self.histology = histology_runtime
        self.rng = np.random.default_rng(
            histology_runtime.config.seed if histology_runtime else 0
        )
        if use_celltype and any(
            label_key not in a.obs or a.obs[label_key].isna().any() for a in adata_list
        ):
            raise ValueError(
                "Missing cell-type labels; annotate all anchors or choose use_celltype=False"
            )
        self.label_encoder = LabelEncoder()
        all_labels = []
        for adata in adata_list:
            if use_celltype and label_key in adata.obs:
                all_labels.extend(adata.obs[label_key].astype(str).tolist())
        if all_labels:
            self.label_encoder.fit(all_labels)
            self.num_classes = len(self.label_encoder.classes_)
            self.id2label = {
                i: label for i, label in enumerate(self.label_encoder.classes_)
            }
        else:
            self.label_encoder.fit(["unknown"])
            self.num_classes = 1
            self.id2label = {0: "unknown"}
        if mode != "fit":
            self.adata_list = self.adata_list[:2]
        self.trajectory_pairs = {
            "x0": [],
            "g0": [],
            "c0": [],
            "z0": [],
            "x1": [],
            "g1": [],
            "c1": [],
            "z1": [],
            "delta_z": [],
        }
        if self.histology and self.histology.paths:
            self.trajectory_pairs["path_id"] = []
        self._build_trajectory_dataset()
        self._convert_to_tensors()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tensors.items()}

    def decode_label(self, one_hot_vec):
        """Converts model one-hot output back to original string label."""
        idx = torch.argmax(one_hot_vec, dim=-1).item()
        return self.id2label[idx]

    def _get_data_arrays(self, adata):
        """Extracts spatial, gene expression, and one-hot labels from AnnData."""
        x = adata.obsm[self.spatial_key].astype(np.float32)
        g = (
            adata.X.toarray().astype(np.float32)
            if sp.issparse(adata.X)
            else adata.X.astype(np.float32)
        )
        if self.use_celltype and self.label_key in adata.obs:
            raw_labels = adata.obs[self.label_key].astype(str).values
            indices = self.label_encoder.transform(raw_labels)
            c_onehot = np.eye(self.num_classes)[indices].astype(np.float32)
        else:
            c_onehot = np.zeros((adata.n_obs, self.num_classes), dtype=np.float32)
        z = float(adata.obs[self.z_key].iloc[0])
        return (x, g, c_onehot, z)

    def _build_trajectory_dataset(self):
        """Pairs cells across adjacent slices using UOT coupling."""
        num_slices = len(self.adata_list)
        pair_sizes = [
            self.adata_list[k].n_obs * self.adata_list[k + 1].n_obs
            for k in range(num_slices - 1)
        ]
        total_weight = sum(pair_sizes)
        sampling_counts = [
            int(self.n_samples_base * (w / total_weight)) for w in pair_sizes
        ]
        for k in tqdm(range(num_slices - 1), desc="DeepSpatial: Building Trajectories"):
            n_to_sample = sampling_counts[k]
            if n_to_sample <= 0:
                continue
            if (
                self.histology
                and self.uot_solver == "dense"
                and (pair_sizes[k] > self.histology.config.max_uot_entries)
            ):
                raise MemoryError(
                    "Dense UOT exceeds max_uot_entries before expression densification; subset/aggregate anchors explicitly"
                )
            x0, g0, c0, z0 = self._get_data_arrays(self.adata_list[k])
            x1, g1, c1, z1 = self._get_data_arrays(self.adata_list[k + 1])
            delta_z = z1 - z0
            kwargs = {}
            if not self.use_celltype:
                kwargs["use_celltype"] = False
            kwargs.update(
                solver=self.uot_solver,
                top_k=self.uot_top_k,
                bidirectional=self.uot_bidirectional,
                candidate_radius=self.uot_candidate_radius,
            )
            if self.histology:
                config = self.histology.config
                s0 = self.histology.section(self.adata_list[k])
                s1 = self.histology.section(self.adata_list[k + 1])
                kwargs["max_uot_entries"] = config.max_uot_entries
                if config.use_morphology_uot:
                    kwargs.update(
                        spatial_weight=config.spatial_weight,
                        gene_weight=config.gene_weight,
                        histology_weight=config.histology_weight,
                        class_weight=config.class_weight,
                    )
                    if config.histology_weight:
                        h0, h0_valid = self.histology.store.get_feature(
                            s0, self.histology.xy_um(x0), return_valid=True
                        )
                        h1, h1_valid = self.histology.store.get_feature(
                            s1, self.histology.xy_um(x1), return_valid=True
                        )
                        kwargs["h0"] = h0.numpy()
                        kwargs["h1"] = h1.numpy()
                        kwargs["h0_valid"] = h0_valid.numpy()
                        kwargs["h1_valid"] = h1_valid.numpy()
            physical_cost = bool(
                self.histology
                and self.histology.config.use_morphology_uot
                and (self.histology.config.spatial_metric == "physical_um")
            )
            cost_x0 = self.histology.xy_um(x0) if physical_cost else x0
            cost_x1 = self.histology.xy_um(x1) if physical_cost else x1
            candidate_x0 = self.histology.xy_um(x0) if self.histology else cost_x0
            candidate_x1 = self.histology.xy_um(x1) if self.histology else cost_x1
            kwargs["candidate_x0"] = candidate_x0
            kwargs["candidate_x1"] = candidate_x1
            pi = compute_uot_coupling(
                cost_x0,
                g0,
                c0,
                cost_x1,
                g1,
                c1,
                alpha_spatial=self.alpha_spatial,
                uot_reg=self.uot_reg,
                uot_tau=self.uot_tau,
                **kwargs,
            )
            if isinstance(pi, SparseCoupling):
                idx0, idx1 = pi.sample(n_to_sample, rng=self.rng)
            else:
                pi_flat = pi.ravel()
                pi_sum = pi_flat.sum()
                if pi_sum > 0:
                    pi_prob = pi_flat / pi_sum
                    idx_flat = self.rng.choice(
                        len(pi_flat), size=n_to_sample, p=pi_prob, replace=True
                    )
                    idx0, idx1 = np.unravel_index(idx_flat, pi.shape)
                else:
                    idx0 = np.array([], dtype=np.int64)
                    idx1 = np.array([], dtype=np.int64)
            if len(idx0):
                if self.histology and self.histology.paths:
                    keys = [
                        f"{s0}:{self.adata_list[k].obs_names[i]}->{s1}:{self.adata_list[k + 1].obs_names[j]}"
                        for i, j in zip(idx0, idx1)
                    ]
                    ids = self.histology.paths.prepare(
                        s0,
                        s1,
                        self.histology.xy_um(x0[idx0]),
                        self.histology.xy_um(x1[idx1]),
                        keys,
                    )
                    offset = len(self.histology.path_ids)
                    self.histology.path_ids.extend(ids)
                    self.trajectory_pairs["path_id"].append(
                        np.arange(offset, offset + n_to_sample, dtype=np.int64)
                    )
                self.trajectory_pairs["x0"].append(x0[idx0])
                self.trajectory_pairs["g0"].append(g0[idx0])
                self.trajectory_pairs["c0"].append(c0[idx0])
                self.trajectory_pairs["z0"].append(
                    np.full((n_to_sample, 1), z0, dtype=np.float32)
                )
                self.trajectory_pairs["x1"].append(x1[idx1])
                self.trajectory_pairs["g1"].append(g1[idx1])
                self.trajectory_pairs["c1"].append(c1[idx1])
                self.trajectory_pairs["z1"].append(
                    np.full((n_to_sample, 1), z1, dtype=np.float32)
                )
                self.trajectory_pairs["delta_z"].append(
                    np.full((n_to_sample, 1), delta_z, dtype=np.float32)
                )
            del pi, x0, g0, c0, x1, g1, c1
            gc.collect()

    def _convert_to_tensors(self):
        """Aggregates list of arrays into final PyTorch tensors."""
        self.tensors = {}
        for key in self.trajectory_pairs:
            if self.trajectory_pairs[key]:
                concatenated = np.concatenate(self.trajectory_pairs[key], axis=0)
                self.tensors[key] = torch.from_numpy(concatenated)
        self.trajectory_pairs.clear()
        gc.collect()
        if "x0" in self.tensors:
            self.num_samples = self.tensors["x0"].shape[0]
        else:
            self.num_samples = 0
