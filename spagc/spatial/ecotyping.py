"""Contrastive graph representations and specimen ecotyping."""

from __future__ import annotations

import anndata as ad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import dropout_edge


def validate_cells(
    data: ad.AnnData, obs_keys: tuple[str, ...], feature_key: str | None = None
) -> None:
    """Validate axes, required metadata and finite XY without changing the input."""
    missing = set(obs_keys).difference(data.obs.columns)
    if missing:
        raise KeyError(f"Missing cell metadata: {sorted(missing)}")
    if not data.obs_names.is_unique or not data.var_names.is_unique:
        raise ValueError("Cell and gene identifiers must be unique.")
    xy = np.asarray(data.obsm.get("spatial", []))
    if xy.shape != (data.n_obs, 2) or not np.isfinite(xy).all():
        raise ValueError("spatial must contain finite N x 2 physical coordinates.")
    if any(data.obs[key].isna().any() for key in obs_keys):
        raise ValueError("Required grouping/annotation fields contain missing values.")
    if feature_key is not None:
        value = np.asarray(data.obsm[feature_key])
        if value.shape[0] != data.n_obs or not np.isfinite(value).all():
            raise ValueError(f"Invalid feature matrix: {feature_key}")


def build_sample_graphs(
    data: ad.AnnData, sample_key: str, feature_key: str, n_neighbors: int
) -> tuple[list[Data], list[str]]:
    """Build the Figure 3 nearest-neighbour graph separately for each specimen."""
    import squidpy as sq

    validate_cells(data, (sample_key, "niche"), feature_key)
    graphs, names = [], []
    for name in data.obs[sample_key].astype(str).unique():
        subset = data[
            (data.obs[sample_key].astype(str) == name)
            & (data.obs["niche"].astype(str) != "Low quality")
        ].copy()
        if subset.n_obs <= n_neighbors - 1:
            continue
        sq.gr.spatial_neighbors(
            subset, coord_type="generic", n_neighs=n_neighbors, spatial_key="spatial"
        )
        row, col = subset.obsp["spatial_connectivities"].nonzero()
        graphs.append(
            Data(
                x=torch.as_tensor(
                    np.asarray(subset.obsm[feature_key]), dtype=torch.float32
                ),
                edge_index=torch.as_tensor(np.vstack([row, col]), dtype=torch.long),
            )
        )
        names.append(str(name))
    if len(graphs) < 3:
        raise ValueError("At least three eligible specimens are needed for ecotyping.")
    return graphs, names


class SpatialGraphEncoder(nn.Module):
    """Figure 3 encoder: two ReLU GCNs, mean pooling and a separate projection."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 64):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.proj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x, edge_index, batch):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        graph_embedding = global_mean_pool(h, batch)
        return graph_embedding, self.proj_head(graph_embedding)


def augment_graph(
    data: Data, edge_drop_prob: float = 0.1, feat_drop_prob: float = 0.2
) -> Data:
    """Drop directed edges and mask channels across every node in a view."""
    x = data.x.clone()
    mask = torch.rand(x.shape[1], device=x.device) < feat_drop_prob
    x[:, mask] = 0
    edges, _ = dropout_edge(
        data.edge_index, p=edge_drop_prob, force_undirected=False, training=True
    )
    return Data(x=x, edge_index=edges, batch=data.batch)


def nt_xent_loss(
    z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5
) -> torch.Tensor:
    """Full 2B-view NT-Xent; paired positives remain in the denominator."""
    if z1.shape != z2.shape or len(z1) < 2 or temperature <= 0:
        raise ValueError(
            "Contrastive loss needs paired batches of at least two specimens and positive temperature."
        )
    n = len(z1)
    views = F.normalize(torch.cat([z1, z2]), dim=1)
    similarities = views @ views.T
    similarities = similarities.masked_fill(
        torch.eye(2 * n, dtype=torch.bool, device=z1.device), -9e15
    )
    targets = torch.cat(
        [torch.arange(n, 2 * n, device=z1.device), torch.arange(n, device=z1.device)]
    )
    return F.cross_entropy(similarities / temperature, targets)


def train_ecotyper(
    graphs: list[Data],
    *,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    projection_dim: int,
    learning_rate: float,
    device: str,
) -> tuple[nn.Module, list[dict]]:
    """Fit one encoder and retain a complete finite training-loss history."""
    model = SpatialGraphEncoder(
        graphs[0].num_node_features, hidden_dim, projection_dim
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    # Avoid a singleton tail, whose contrastive objective has no negatives.
    sizes = [
        list(range(i, min(i + batch_size, len(graphs))))
        for i in range(0, len(graphs), batch_size)
    ]
    if len(sizes[-1]) == 1 and len(sizes) > 1:
        sizes[-2].extend(sizes.pop())
    for epoch in range(epochs):
        order = torch.randperm(len(graphs)).tolist()
        total, seen = 0.0, 0
        model.train()
        for indices in sizes:
            batch = next(
                iter(
                    DataLoader(
                        [graphs[order[i]] for i in indices], batch_size=len(indices)
                    )
                )
            ).to(device)
            a, b = augment_graph(batch), augment_graph(batch)
            loss = nt_xent_loss(
                model(a.x, a.edge_index, a.batch)[1],
                model(b.x, b.edge_index, b.batch)[1],
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite specimen contrastive loss.")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
            seen += len(indices)
        history.append({"epoch": epoch + 1, "loss": total / seen})
    return model, history


def projected_embeddings(
    model: nn.Module, graphs: list[Data], device: str
) -> np.ndarray:
    """Use the projected vectors specified by the Figure 3 analysis, not pooled h."""
    model.eval()
    rows = []
    with torch.no_grad():
        for graph in graphs:
            batch = graph.to(device)
            membership = torch.zeros(batch.num_nodes, dtype=torch.long, device=device)
            rows.append(
                model(batch.x, batch.edge_index, membership)[1].cpu().numpy()[0]
            )
    return np.stack(rows)


def cluster_ecotypes(
    vectors: np.ndarray, names: list[str], *, neighbors: int, resolution: float, seed=42
) -> ad.AnnData:
    """Standardize/L2-normalize projections and run cosine-neighbour Leiden."""
    import scanpy as sc
    from sklearn.preprocessing import StandardScaler, normalize

    result = ad.AnnData(normalize(StandardScaler().fit_transform(vectors), norm="l2"))
    result.obs_names = names
    result.obsm["X_graph_embedding"] = result.X.copy()
    sc.pp.neighbors(
        result,
        n_neighbors=min(neighbors, len(names) - 1),
        use_rep="X_graph_embedding",
        metric="cosine",
    )
    sc.tl.leiden(result, resolution=resolution, key_added="ecotype", random_state=seed)
    return result
