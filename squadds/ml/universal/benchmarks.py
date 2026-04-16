"""Embedding benchmark helpers for the universal geometry/graph stack."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from shapely.ops import unary_union

from squadds.ml.universal.features.arithmetic import (
    compute_shared_frame_shape_embedding,
    embedding_cosine_similarity,
)
from squadds.ml.universal.features.node_encoder import get_polygon_for_component
from squadds.ml.universal.features.protocol import (
    DEFAULT_EMBEDDING_CONFIG,
    EmbeddingConfig,
    EmbeddingMode,
    compute_component_embedding,
)
from squadds.ml.universal.visualization import (
    DifferenceMatch,
    compute_cosine_similarity_matrix,
)
from squadds.ml.universal.workflows import (
    STANDARD_SQUADDS_ROW_SCHEMA,
    UniversalRowSchema,
    build_layout_from_row,
)

STANDARD_COMPONENT_LABELS = {
    "qubit": "Qubit",
    "claw": "Claw",
    "resonator": "Resonator",
    "feedline": "Feedline",
}


@dataclass(frozen=True)
class EmbeddingCollection:
    """A flat collection of component embeddings and their metadata."""

    embeddings: np.ndarray
    labels: list[str]
    identifiers: list[str]
    component_names: list[str]
    row_indices: list[int]


@dataclass(frozen=True)
class ClusterLabelStats:
    """Per-label summary for the clustering benchmark."""

    label: str
    count: int
    mean_self_centroid_similarity: float
    mean_nearest_other_centroid_similarity: float
    separation_margin: float


@dataclass(frozen=True)
class ClusteringBenchmarkResult:
    """Summary metrics for component-family clustering."""

    num_embeddings: int
    num_labels: int
    centroid_top1_accuracy: float
    nearest_neighbor_top1_accuracy: float
    mean_intra_label_similarity: float
    mean_inter_label_similarity: float
    separation_gap: float
    per_label: list[ClusterLabelStats]


@dataclass(frozen=True)
class ArithmeticSpec:
    """A compositional arithmetic identity to test in embedding space."""

    name: str
    minuend_components: tuple[str, ...]
    subtrahend_components: tuple[str, ...]
    expected_component: str


@dataclass(frozen=True)
class ArithmeticTrialResult:
    """The result of one arithmetic trial on one row."""

    case_name: str
    row_index: int | None
    expected_component: str
    expected_label: str
    predicted_label: str
    expected_rank: int
    expected_similarity: float
    predicted_similarity: float
    margin: float
    top1_success: bool
    top2_success: bool
    matches: tuple[DifferenceMatch, ...]


@dataclass(frozen=True)
class ArithmeticCaseSummary:
    """Aggregate summary for one arithmetic identity across rows."""

    case_name: str
    expected_label: str
    num_trials: int
    top1_accuracy: float
    top2_accuracy: float
    mean_expected_rank: float
    mean_expected_similarity: float
    mean_margin: float


@dataclass(frozen=True)
class ArithmeticBenchmarkResult:
    """Aggregate arithmetic benchmark across all cases and rows."""

    num_trials: int
    top1_accuracy: float
    top2_accuracy: float
    mean_expected_rank: float
    mean_expected_similarity: float
    mean_margin: float
    per_case: list[ArithmeticCaseSummary]
    trials: list[ArithmeticTrialResult]


STANDARD_ARITHMETIC_SPECS = (
    ArithmeticSpec(
        name="(qubit + claw) - claw -> qubit",
        minuend_components=("qubit", "claw"),
        subtrahend_components=("claw",),
        expected_component="qubit",
    ),
    ArithmeticSpec(
        name="(claw + resonator) - resonator -> claw",
        minuend_components=("claw", "resonator"),
        subtrahend_components=("resonator",),
        expected_component="claw",
    ),
    ArithmeticSpec(
        name="(resonator + feedline) - feedline -> resonator",
        minuend_components=("resonator", "feedline"),
        subtrahend_components=("feedline",),
        expected_component="resonator",
    ),
)


def _safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def build_component_embedding_collection(
    rows: Iterable[Mapping[str, object] | object],
    *,
    embedding_config: EmbeddingConfig = DEFAULT_EMBEDDING_CONFIG,
    row_schema: UniversalRowSchema = STANDARD_SQUADDS_ROW_SCHEMA,
    component_names: Sequence[str] = tuple(STANDARD_COMPONENT_LABELS.keys()),
    component_label_map: Mapping[str, str] = STANDARD_COMPONENT_LABELS,
    layout_builder=None,
) -> EmbeddingCollection:
    """Build a flat dataset of component embeddings from row-like objects."""

    embeddings: list[np.ndarray] = []
    labels: list[str] = []
    identifiers: list[str] = []
    names: list[str] = []
    row_indices: list[int] = []

    if layout_builder is None:
        from squadds.ml.universal.geometry.layout import build_layout as layout_builder

    for row_index, row in enumerate(rows):
        layout = build_layout_from_row(row, row_schema=row_schema, layout_builder=layout_builder)
        polygons = {
            component_name: get_polygon_for_component(layout[component_name])
            for component_name in component_names
        }
        reference_polygons = list(polygons.values()) if embedding_config.mode != EmbeddingMode.GEOMETRY_ONLY else None

        for component_name in component_names:
            component = layout[component_name]
            embedding = compute_component_embedding(
                polygons[component_name],
                params=component.get("params", {}),
                config=embedding_config,
                reference_polygons=reference_polygons,
            )
            embeddings.append(embedding)
            labels.append(component_label_map.get(component_name, component_name.title()))
            identifiers.append(f"row-{row_index}:{component_name}")
            names.append(component_name)
            row_indices.append(row_index)

    if not embeddings:
        raise ValueError("No embeddings were generated from the provided rows.")

    return EmbeddingCollection(
        embeddings=np.stack(embeddings, axis=0).astype(np.float32),
        labels=labels,
        identifiers=identifiers,
        component_names=names,
        row_indices=row_indices,
    )


def benchmark_component_family_clustering(
    embeddings: np.ndarray | Sequence[Sequence[float]],
    labels: Sequence[str],
) -> ClusteringBenchmarkResult:
    """Measure how cleanly component families cluster in embedding space."""

    x = np.asarray(embeddings, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {x.shape}.")
    label_list = list(labels)
    if len(label_list) != x.shape[0]:
        raise ValueError(f"Expected {x.shape[0]} labels, got {len(label_list)}.")

    similarity = compute_cosine_similarity_matrix(x)
    unique_labels = list(dict.fromkeys(label_list))

    label_to_centroid: dict[str, np.ndarray] = {}
    for label in unique_labels:
        indices = [idx for idx, value in enumerate(label_list) if value == label]
        label_to_centroid[label] = np.mean(x[indices], axis=0).astype(np.float32)

    centroid_hits = 0
    neighbor_hits = 0
    per_label_stats: list[ClusterLabelStats] = []
    intra_scores: list[float] = []
    inter_scores: list[float] = []

    for row_index, label in enumerate(label_list):
        centroid_scores = {
            centroid_label: embedding_cosine_similarity(x[row_index], centroid)
            for centroid_label, centroid in label_to_centroid.items()
        }
        ranked_centroids = sorted(centroid_scores.items(), key=lambda item: item[1], reverse=True)
        if ranked_centroids[0][0] == label:
            centroid_hits += 1

        neighbor_order = np.argsort(similarity[row_index])[::-1]
        for neighbor_index in neighbor_order:
            if int(neighbor_index) == row_index:
                continue
            if label_list[int(neighbor_index)] == label:
                neighbor_hits += 1
            break

    for i in range(x.shape[0]):
        for j in range(i + 1, x.shape[0]):
            score = float(similarity[i, j])
            if label_list[i] == label_list[j]:
                intra_scores.append(score)
            else:
                inter_scores.append(score)

    for label in unique_labels:
        indices = [idx for idx, value in enumerate(label_list) if value == label]
        self_scores = [
            embedding_cosine_similarity(x[idx], label_to_centroid[label])
            for idx in indices
        ]
        other_labels = [other for other in unique_labels if other != label]
        nearest_other_scores = []
        for idx in indices:
            if other_labels:
                nearest_other_scores.append(
                    max(embedding_cosine_similarity(x[idx], label_to_centroid[other]) for other in other_labels)
                )
            else:
                nearest_other_scores.append(float("nan"))

        self_mean = _safe_mean(self_scores)
        other_mean = _safe_mean(nearest_other_scores)
        per_label_stats.append(
            ClusterLabelStats(
                label=label,
                count=len(indices),
                mean_self_centroid_similarity=self_mean,
                mean_nearest_other_centroid_similarity=other_mean,
                separation_margin=self_mean - other_mean,
            )
        )

    return ClusteringBenchmarkResult(
        num_embeddings=x.shape[0],
        num_labels=len(unique_labels),
        centroid_top1_accuracy=centroid_hits / x.shape[0],
        nearest_neighbor_top1_accuracy=neighbor_hits / x.shape[0],
        mean_intra_label_similarity=_safe_mean(intra_scores),
        mean_inter_label_similarity=_safe_mean(inter_scores),
        separation_gap=_safe_mean(intra_scores) - _safe_mean(inter_scores),
        per_label=per_label_stats,
    )


def evaluate_standard_arithmetic_case(
    row: Mapping[str, object] | object,
    spec: ArithmeticSpec,
    *,
    row_index: int | None = None,
    row_schema: UniversalRowSchema = STANDARD_SQUADDS_ROW_SCHEMA,
    component_label_map: Mapping[str, str] = STANDARD_COMPONENT_LABELS,
    component_names: Sequence[str] = tuple(STANDARD_COMPONENT_LABELS.keys()),
    shape_resolution: int = 32,
    padding: float = 0.0,
    padding_fraction: float = 0.05,
    layout_builder=None,
) -> ArithmeticTrialResult:
    """Evaluate one arithmetic identity on a single standard row."""

    if layout_builder is None:
        from squadds.ml.universal.geometry.layout import build_layout as layout_builder

    layout = build_layout_from_row(row, row_schema=row_schema, layout_builder=layout_builder)
    polygons = {
        component_name: get_polygon_for_component(layout[component_name])
        for component_name in component_names
    }

    bounds_reference = [polygons[name] for name in component_names]
    candidate_embeddings = {
        component_name: compute_shared_frame_shape_embedding(
            polygons[component_name],
            reference_polygons=bounds_reference,
            shape_resolution=shape_resolution,
            padding=padding,
            padding_fraction=padding_fraction,
        )
        for component_name in component_names
    }

    minuend = unary_union([polygons[name] for name in spec.minuend_components])
    subtrahend = unary_union([polygons[name] for name in spec.subtrahend_components])
    difference = minuend.difference(subtrahend)

    difference_embedding = compute_shared_frame_shape_embedding(
        difference,
        reference_polygons=bounds_reference,
        shape_resolution=shape_resolution,
        padding=padding,
        padding_fraction=padding_fraction,
    )

    aggregated_by_label: dict[str, list[np.ndarray]] = {}
    for component_name, embedding in candidate_embeddings.items():
        label = component_label_map.get(component_name, component_name.title())
        aggregated_by_label.setdefault(label, []).append(embedding)

    matches = [
        DifferenceMatch(
            label=label,
            similarity=embedding_cosine_similarity(
                difference_embedding,
                np.mean(np.stack(vectors, axis=0), axis=0),
            ),
        )
        for label, vectors in aggregated_by_label.items()
    ]
    ranked_matches = tuple(sorted(matches, key=lambda item: item.similarity, reverse=True))

    expected_label = component_label_map.get(spec.expected_component, spec.expected_component.title())
    expected_rank = next(index for index, match in enumerate(ranked_matches, start=1) if match.label == expected_label)
    expected_similarity = next(match.similarity for match in ranked_matches if match.label == expected_label)
    predicted = ranked_matches[0]
    best_other_similarity = max(
        (match.similarity for match in ranked_matches if match.label != expected_label),
        default=expected_similarity,
    )

    return ArithmeticTrialResult(
        case_name=spec.name,
        row_index=row_index,
        expected_component=spec.expected_component,
        expected_label=expected_label,
        predicted_label=predicted.label,
        expected_rank=expected_rank,
        expected_similarity=expected_similarity,
        predicted_similarity=predicted.similarity,
        margin=expected_similarity - best_other_similarity,
        top1_success=expected_rank == 1,
        top2_success=expected_rank <= 2,
        matches=ranked_matches,
    )


def benchmark_standard_embedding_arithmetic(
    rows: Iterable[Mapping[str, object] | object],
    *,
    arithmetic_specs: Sequence[ArithmeticSpec] = STANDARD_ARITHMETIC_SPECS,
    row_schema: UniversalRowSchema = STANDARD_SQUADDS_ROW_SCHEMA,
    component_label_map: Mapping[str, str] = STANDARD_COMPONENT_LABELS,
    component_names: Sequence[str] = tuple(STANDARD_COMPONENT_LABELS.keys()),
    shape_resolution: int = 32,
    padding: float = 0.0,
    padding_fraction: float = 0.05,
    layout_builder=None,
) -> ArithmeticBenchmarkResult:
    """Benchmark embedding arithmetic across rows and standard component families."""

    trials: list[ArithmeticTrialResult] = []
    for row_index, row in enumerate(rows):
        for spec in arithmetic_specs:
            trials.append(
                evaluate_standard_arithmetic_case(
                    row,
                    spec,
                    row_index=row_index,
                    row_schema=row_schema,
                    component_label_map=component_label_map,
                    component_names=component_names,
                    shape_resolution=shape_resolution,
                    padding=padding,
                    padding_fraction=padding_fraction,
                    layout_builder=layout_builder,
                )
            )

    if not trials:
        raise ValueError("No arithmetic trials were generated from the provided rows.")

    per_case: list[ArithmeticCaseSummary] = []
    for spec in arithmetic_specs:
        case_trials = [trial for trial in trials if trial.case_name == spec.name]
        per_case.append(
            ArithmeticCaseSummary(
                case_name=spec.name,
                expected_label=component_label_map.get(spec.expected_component, spec.expected_component.title()),
                num_trials=len(case_trials),
                top1_accuracy=_safe_mean([1.0 if trial.top1_success else 0.0 for trial in case_trials]),
                top2_accuracy=_safe_mean([1.0 if trial.top2_success else 0.0 for trial in case_trials]),
                mean_expected_rank=_safe_mean([float(trial.expected_rank) for trial in case_trials]),
                mean_expected_similarity=_safe_mean([trial.expected_similarity for trial in case_trials]),
                mean_margin=_safe_mean([trial.margin for trial in case_trials]),
            )
        )

    return ArithmeticBenchmarkResult(
        num_trials=len(trials),
        top1_accuracy=_safe_mean([1.0 if trial.top1_success else 0.0 for trial in trials]),
        top2_accuracy=_safe_mean([1.0 if trial.top2_success else 0.0 for trial in trials]),
        mean_expected_rank=_safe_mean([float(trial.expected_rank) for trial in trials]),
        mean_expected_similarity=_safe_mean([trial.expected_similarity for trial in trials]),
        mean_margin=_safe_mean([trial.margin for trial in trials]),
        per_case=per_case,
        trials=trials,
    )


__all__ = [
    "ArithmeticBenchmarkResult",
    "ArithmeticCaseSummary",
    "ArithmeticSpec",
    "ArithmeticTrialResult",
    "ClusterLabelStats",
    "ClusteringBenchmarkResult",
    "EmbeddingCollection",
    "STANDARD_ARITHMETIC_SPECS",
    "STANDARD_COMPONENT_LABELS",
    "benchmark_component_family_clustering",
    "benchmark_standard_embedding_arithmetic",
    "build_component_embedding_collection",
    "evaluate_standard_arithmetic_case",
]
