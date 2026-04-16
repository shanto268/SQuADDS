"""Embedding-space analysis and visualization helpers for the universal stack."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from squadds.ml.universal.features.arithmetic import embedding_cosine_similarity


@dataclass(frozen=True)
class NeighborResult:
    """A nearest-neighbor match in embedding space."""

    index: int
    similarity: float
    label: str | None = None
    identifier: str | None = None


@dataclass(frozen=True)
class DifferenceMatch:
    """A ranked similarity match for a difference vector."""

    label: str
    similarity: float


def _as_2d_embeddings(embeddings: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {array.shape}.")
    return array


def _validate_labels(labels: Sequence[str], num_rows: int) -> list[str]:
    labels_list = list(labels)
    if len(labels_list) != num_rows:
        raise ValueError(f"Expected {num_rows} labels, got {len(labels_list)}.")
    return labels_list


def compute_embedding_projection(
    embeddings: np.ndarray | Sequence[Sequence[float]],
    *,
    method: str = "pca",
    n_components: int = 2,
    random_state: int = 42,
    **kwargs,
) -> np.ndarray:
    """Project embeddings to a lower-dimensional space.

    Supported methods:
    - `pca`: implemented with NumPy SVD and always available
    - `kernel_pca`: requires scikit-learn
    - `tsne`: requires scikit-learn
    - `umap`: requires `umap-learn`
    """

    x = _as_2d_embeddings(embeddings)
    method_normalized = method.lower()

    if method_normalized == "pca":
        centered = x - x.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        return (centered @ vt[:n_components].T).astype(np.float32)

    if method_normalized == "kernel_pca":
        from sklearn.decomposition import KernelPCA

        return KernelPCA(n_components=n_components, kernel="rbf", random_state=random_state, **kwargs).fit_transform(x)

    if method_normalized == "tsne":
        from sklearn.manifold import TSNE

        return TSNE(n_components=n_components, random_state=random_state, init="pca", learning_rate="auto", **kwargs).fit_transform(x)

    if method_normalized == "umap":
        import umap

        return umap.UMAP(n_components=n_components, random_state=random_state, **kwargs).fit_transform(x)

    raise ValueError(f"Unsupported projection method: {method!r}")


def compute_embedding_projections(
    embeddings: np.ndarray | Sequence[Sequence[float]],
    *,
    methods: Sequence[str] = ("pca",),
    random_state: int = 42,
    method_kwargs: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, np.ndarray]:
    """Compute multiple named projections for the same embedding matrix."""

    kwargs_map = dict(method_kwargs or {})
    return {
        method: compute_embedding_projection(
            embeddings,
            method=method,
            random_state=random_state,
            **dict(kwargs_map.get(method, {})),
        )
        for method in methods
    }


def compute_cosine_similarity_matrix(
    embeddings: np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    """Compute the full pairwise cosine-similarity matrix."""

    x = _as_2d_embeddings(embeddings)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    normalized = x / norms
    return (normalized @ normalized.T).astype(np.float32)


def find_nearest_neighbors(
    query_embedding: np.ndarray | Sequence[float],
    reference_embeddings: np.ndarray | Sequence[Sequence[float]],
    *,
    labels: Sequence[str] | None = None,
    identifiers: Sequence[str] | None = None,
    top_k: int = 5,
    exclude_index: int | None = None,
) -> list[NeighborResult]:
    """Find nearest embeddings by cosine similarity."""

    refs = _as_2d_embeddings(reference_embeddings)
    query = np.asarray(query_embedding, dtype=np.float32).ravel()
    if refs.shape[1] != query.shape[0]:
        raise ValueError(f"Query dim {query.shape[0]} does not match reference dim {refs.shape[1]}.")

    label_list = _validate_labels(labels, refs.shape[0]) if labels is not None else [None] * refs.shape[0]
    id_list = _validate_labels(identifiers, refs.shape[0]) if identifiers is not None else [None] * refs.shape[0]

    scores = np.array([embedding_cosine_similarity(query, ref) for ref in refs], dtype=np.float32)
    order = np.argsort(scores)[::-1]

    results: list[NeighborResult] = []
    for index in order:
        if exclude_index is not None and int(index) == exclude_index:
            continue
        results.append(
            NeighborResult(
                index=int(index),
                similarity=float(scores[index]),
                label=label_list[index],
                identifier=id_list[index],
            )
        )
        if len(results) >= top_k:
            break
    return results


def compute_label_centroids(
    embeddings: np.ndarray | Sequence[Sequence[float]],
    labels: Sequence[str],
) -> dict[str, np.ndarray]:
    """Compute mean embeddings for each label."""

    x = _as_2d_embeddings(embeddings)
    label_list = _validate_labels(labels, x.shape[0])

    centroids: dict[str, list[np.ndarray]] = {}
    for embedding, label in zip(x, label_list):
        centroids.setdefault(label, []).append(embedding)

    return {
        label: np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32)
        for label, vectors in centroids.items()
    }


def rank_difference_vector(
    difference_embedding: np.ndarray | Sequence[float],
    reference_embeddings: np.ndarray | Sequence[Sequence[float]],
    reference_labels: Sequence[str],
) -> list[DifferenceMatch]:
    """Rank a difference vector against label centroids by cosine similarity."""

    centroids = compute_label_centroids(reference_embeddings, reference_labels)
    query = np.asarray(difference_embedding, dtype=np.float32).ravel()

    matches = [
        DifferenceMatch(label=label, similarity=embedding_cosine_similarity(query, centroid))
        for label, centroid in centroids.items()
    ]
    return sorted(matches, key=lambda item: item.similarity, reverse=True)


def plot_embedding_projection(
    projection: np.ndarray | Sequence[Sequence[float]],
    labels: Sequence[str],
    *,
    ax=None,
    title: str | None = None,
    palette: Mapping[str, str] | None = None,
    highlight_indices: Sequence[int] | None = None,
    highlight_kwargs: Mapping[str, object] | None = None,
):
    """Plot a 2D embedding projection grouped by label."""

    coords = _as_2d_embeddings(projection)
    if coords.shape[1] != 2:
        raise ValueError(f"plot_embedding_projection expects an (N, 2) array, got {coords.shape}.")
    label_list = _validate_labels(labels, coords.shape[0])
    unique_labels = list(dict.fromkeys(label_list))

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    palette_map = dict(palette or {})
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    for idx, label in enumerate(unique_labels):
        indices = [i for i, value in enumerate(label_list) if value == label]
        color = palette_map.get(label, default_colors[idx % len(default_colors)])
        ax.scatter(
            coords[indices, 0],
            coords[indices, 1],
            label=label,
            color=color,
            s=55,
            alpha=0.82,
            edgecolors="white",
            linewidths=0.4,
        )

    if highlight_indices:
        options = {"color": "#d62828", "marker": "*", "s": 240, "zorder": 10}
        options.update(dict(highlight_kwargs or {}))
        highlight = np.array(list(highlight_indices), dtype=int)
        ax.scatter(coords[highlight, 0], coords[highlight, 1], **options)

    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.grid(True, alpha=0.3)
    ax.legend(framealpha=0.9)
    if title:
        ax.set_title(title)
    return ax


def plot_projection_grid(
    projections: Mapping[str, np.ndarray],
    labels: Sequence[str],
    *,
    palette: Mapping[str, str] | None = None,
    figsize: tuple[float, float] | None = None,
    suptitle: str | None = None,
):
    """Plot several named projections in a horizontal grid."""

    items = list(projections.items())
    if not items:
        raise ValueError("projections must contain at least one projection.")

    if figsize is None:
        figsize = (6 * len(items), 5)
    fig, axes = plt.subplots(1, len(items), figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    for ax, (name, projection) in zip(axes_flat, items):
        plot_embedding_projection(projection, labels, ax=ax, title=name, palette=palette)

    if suptitle:
        fig.suptitle(suptitle, y=1.02)
    fig.tight_layout()
    return fig, axes_flat


def plot_similarity_bars(
    matches: Sequence[DifferenceMatch] | Mapping[str, float],
    *,
    ax=None,
    title: str | None = None,
    color: str = "#4C72B0",
):
    """Plot a simple similarity ranking bar chart."""

    if isinstance(matches, Mapping):
        labels = list(matches.keys())
        values = [float(value) for value in matches.values()]
    else:
        labels = [match.label for match in matches]
        values = [float(match.similarity) for match in matches]

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    ax.bar(labels, values, color=color)
    ax.set_ylabel("Cosine similarity")
    ax.set_ylim(min(0.0, min(values, default=0.0)), 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    if title:
        ax.set_title(title)
    return ax


__all__ = [
    "DifferenceMatch",
    "NeighborResult",
    "compute_cosine_similarity_matrix",
    "compute_embedding_projection",
    "compute_embedding_projections",
    "compute_label_centroids",
    "find_nearest_neighbors",
    "plot_embedding_projection",
    "plot_projection_grid",
    "plot_similarity_bars",
    "rank_difference_vector",
]
