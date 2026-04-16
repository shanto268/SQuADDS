"""Reusable workflow helpers for the universal graph-ML stack.

This module defines the stable internal interfaces that notebook code should
prefer over ad hoc cell logic:

1. Map a tabular dataset row into standard layout parameters
2. Build labeled ``HeteroData`` graphs from row-like objects
3. Read structured node/edge predictions from model outputs

The first target for these helpers is the quarter-wave qubit/claw/resonator/
feedline workflow used in Tutorial 12.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import torch
from torch_geometric.data import HeteroData

from squadds.ml.universal.geometry.layout import build_layout
from squadds.ml.universal.graph.builder import UniversalGraphBuilder
from squadds.ml.universal.graph.netlist import CircuitNetlist, ComponentSpec, EdgeSpec
from squadds.ml.universal.model.gat_model import (
    EDGE_INFERENCE_READOUT,
    EDGE_TARGET_NAMES,
    NODE_INFERENCE_READOUT,
    NODE_TARGET_NAMES,
    UniversalGNN,
)

DEFAULT_LAYOUT_PARAM_COLUMNS = {
    "cross_length": "cross_length",
    "cross_gap": "cross_gap",
    "claw_length": "claw_length",
    "ground_spacing": "ground_spacing",
    "coupling_length": "coupling_length",
    "total_length": "total_length",
}

DEFAULT_NODE_TARGET_COLUMNS = {
    "qubit_freq_GHz": "qubit_frequency_GHz",
    "anharmonicity_MHz": "anharmonicity_MHz",
    "cavity_freq_GHz": "cavity_frequency_GHz",
}

DEFAULT_EDGE_TARGET_COLUMNS = {
    "g_MHz": "g_MHz",
    "kappa_kHz": "kappa_kHz",
}

DEFAULT_NODE_TARGET_SCALES = {
    "qubit_freq_GHz": 1.0,
    "anharmonicity_MHz": 100.0,
    "cavity_freq_GHz": 1.0,
}

DEFAULT_EDGE_TARGET_SCALES = {
    "g_MHz": 100.0,
    "kappa_kHz": 100.0,
}


@dataclass(frozen=True)
class UniversalRowSchema:
    """Map dataset columns into layout parameters and training targets."""

    layout_params: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_LAYOUT_PARAM_COLUMNS))
    node_targets: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_NODE_TARGET_COLUMNS))
    edge_targets: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_EDGE_TARGET_COLUMNS))


@dataclass(frozen=True)
class TargetScales:
    """Normalization factors used during training and inference readout."""

    node: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_NODE_TARGET_SCALES))
    edge: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_EDGE_TARGET_SCALES))


@dataclass(frozen=True)
class NodePrediction:
    """Structured node-level inference readout."""

    component_name: str
    component_type: str
    target_name: str
    raw_value: float
    value: float


@dataclass(frozen=True)
class EdgePrediction:
    """Structured edge-level inference readout."""

    component_a: str
    component_b: str
    component_types: tuple[str, str]
    target_name: str
    raw_value: float
    value: float
    directions_aggregated: int


@dataclass(frozen=True)
class PredictionSummary:
    """Structured node and edge predictions for a single graph."""

    nodes: list[NodePrediction]
    edges: list[EdgePrediction]


@dataclass(frozen=True)
class UniversalModelDims:
    """Feature dimensions needed to instantiate :class:`UniversalGNN`."""

    comp_dim: int
    virt_dim: int
    phys_edge_dim: int
    spat_edge_dim: int


STANDARD_SQUADDS_ROW_SCHEMA = UniversalRowSchema()
STANDARD_TARGET_SCALES = TargetScales()


def make_standard_qubit_cavity_netlist() -> CircuitNetlist:
    """Return the 4-component quarter-wave training netlist from Tutorial 12."""

    return CircuitNetlist(
        components=[
            ComponentSpec(name="qubit", component_type="TransmonCross"),
            ComponentSpec(name="claw", component_type="Claw"),
            ComponentSpec(name="resonator", component_type="RouteMeander"),
            ComponentSpec(name="feedline", component_type="CoupledLineTee"),
        ],
        edges=[
            EdgeSpec(src="qubit", dst="claw", coupling_type="capacitive"),
            EdgeSpec(src="claw", dst="resonator", coupling_type="galvanic"),
            EdgeSpec(src="resonator", dst="feedline", coupling_type="capacitive"),
        ],
    )


def _as_row_mapping(row: Mapping[str, object] | object) -> Mapping[str, object]:
    if isinstance(row, Mapping):
        return row
    if hasattr(row, "to_dict"):
        return row.to_dict()
    if hasattr(row, "_asdict"):
        return row._asdict()
    raise TypeError("Row must be a mapping or expose to_dict()/ _asdict().")


def _read_required_float(row: Mapping[str, object], column_name: str) -> float:
    if column_name not in row:
        raise KeyError(f"Required column {column_name!r} not found in row.")
    return float(row[column_name])


def extract_layout_params(
    row: Mapping[str, object] | object,
    *,
    row_schema: UniversalRowSchema = STANDARD_SQUADDS_ROW_SCHEMA,
) -> dict[str, float]:
    """Extract the standard layout kwargs expected by :func:`build_layout`."""

    row_map = _as_row_mapping(row)
    return {
        layout_key: _read_required_float(row_map, column_name)
        for layout_key, column_name in row_schema.layout_params.items()
    }


def build_layout_from_row(
    row: Mapping[str, object] | object,
    *,
    row_schema: UniversalRowSchema = STANDARD_SQUADDS_ROW_SCHEMA,
    layout_builder=build_layout,
) -> dict:
    """Build a standard quarter-wave layout directly from a row-like object."""

    return layout_builder(**extract_layout_params(row, row_schema=row_schema))


def attach_targets_from_row(
    data: HeteroData,
    row: Mapping[str, object] | object,
    *,
    row_schema: UniversalRowSchema = STANDARD_SQUADDS_ROW_SCHEMA,
    target_scales: TargetScales = STANDARD_TARGET_SCALES,
    copy_data: bool = True,
) -> HeteroData:
    """Fill node/edge targets on an existing graph from a row-like object."""

    row_map = _as_row_mapping(row)
    graph = copy.deepcopy(data) if copy_data else data

    node_dtype = graph["component"].x.dtype
    node_target_values = torch.tensor(
        [
            _read_required_float(row_map, row_schema.node_targets[target_name]) / target_scales.node.get(target_name, 1.0)
            for target_name in NODE_TARGET_NAMES
        ],
        dtype=node_dtype,
    )
    num_components = graph["component"].x.size(0)
    graph["component"].y = node_target_values.repeat(num_components, 1)

    phys_key = ("component", "physical", "component")
    edge_dtype = node_dtype
    if graph[phys_key].edge_attr.numel() > 0:
        edge_dtype = graph[phys_key].edge_attr.dtype
    edge_target_values = torch.tensor(
        [
            _read_required_float(row_map, row_schema.edge_targets[target_name]) / target_scales.edge.get(target_name, 1.0)
            for target_name in EDGE_TARGET_NAMES
        ],
        dtype=edge_dtype,
    )
    num_edges = graph[phys_key].edge_index.size(1)
    graph[phys_key].y = edge_target_values.repeat(num_edges, 1)
    return graph


def build_graph_from_row(
    row: Mapping[str, object] | object,
    *,
    netlist: CircuitNetlist,
    builder: UniversalGraphBuilder,
    global_features: dict[str, float] | None = None,
    row_schema: UniversalRowSchema = STANDARD_SQUADDS_ROW_SCHEMA,
    target_scales: TargetScales = STANDARD_TARGET_SCALES,
    include_targets: bool = True,
    layout_builder=build_layout,
) -> HeteroData:
    """Build a graph from a standard row, optionally attaching training targets."""

    layout = build_layout_from_row(row, row_schema=row_schema, layout_builder=layout_builder)
    graph = builder.build(layout, netlist, global_features=global_features)
    if not include_targets:
        return graph
    return attach_targets_from_row(graph, row, row_schema=row_schema, target_scales=target_scales, copy_data=False)


def build_graph_dataset(
    rows: Iterable[Mapping[str, object] | object],
    *,
    netlist: CircuitNetlist,
    builder: UniversalGraphBuilder,
    global_features: dict[str, float] | None = None,
    row_schema: UniversalRowSchema = STANDARD_SQUADDS_ROW_SCHEMA,
    target_scales: TargetScales = STANDARD_TARGET_SCALES,
    include_targets: bool = True,
    layout_builder=build_layout,
) -> list[HeteroData]:
    """Build a list of graphs from an iterable of row-like objects."""

    return [
        build_graph_from_row(
            row,
            netlist=netlist,
            builder=builder,
            global_features=global_features,
            row_schema=row_schema,
            target_scales=target_scales,
            include_targets=include_targets,
            layout_builder=layout_builder,
        )
        for row in rows
    ]


def infer_model_dims(data: HeteroData) -> UniversalModelDims:
    """Infer ``UniversalGNN`` input dimensions from a sample graph."""

    phys_key = ("component", "physical", "component")
    spat_key = ("component", "spatial_in", "virtual")

    return UniversalModelDims(
        comp_dim=int(data["component"].x.size(-1)),
        virt_dim=int(data["virtual"].x.size(-1)),
        phys_edge_dim=int(data[phys_key].edge_attr.size(-1)),
        spat_edge_dim=int(data[spat_key].edge_attr.size(-1)),
    )


def build_model_from_graph(
    data: HeteroData,
    *,
    hidden_dim: int = 128,
    edge_hidden: int = 32,
    num_layers: int = 3,
    num_heads: int = 4,
) -> UniversalGNN:
    """Instantiate :class:`UniversalGNN` directly from a sample graph."""

    dims = infer_model_dims(data)
    return UniversalGNN(
        comp_dim=dims.comp_dim,
        virt_dim=dims.virt_dim,
        phys_edge_dim=dims.phys_edge_dim,
        spat_edge_dim=dims.spat_edge_dim,
        hidden_dim=hidden_dim,
        edge_hidden=edge_hidden,
        num_layers=num_layers,
        num_heads=num_heads,
    )


def read_node_predictions(
    data: HeteroData,
    out: Mapping[str, torch.Tensor],
    *,
    target_scales: TargetScales = STANDARD_TARGET_SCALES,
) -> list[NodePrediction]:
    """Extract structured node predictions using the inference readout map."""

    node_preds = out["node_preds"].detach().cpu()
    results: list[NodePrediction] = []

    for idx, (name, component_type) in enumerate(zip(data["component"].component_name, data["component"].component_type)):
        readout = NODE_INFERENCE_READOUT.get(component_type, [])
        for target_index, target_name in enumerate(NODE_TARGET_NAMES):
            if target_name not in readout:
                continue
            raw_value = float(node_preds[idx, target_index].item())
            scale = float(target_scales.node.get(target_name, 1.0))
            results.append(
                NodePrediction(
                    component_name=name,
                    component_type=component_type,
                    target_name=target_name,
                    raw_value=raw_value,
                    value=raw_value * scale,
                )
            )
    return results


def read_edge_predictions(
    data: HeteroData,
    out: Mapping[str, torch.Tensor],
    *,
    target_scales: TargetScales = STANDARD_TARGET_SCALES,
) -> list[EdgePrediction]:
    """Extract structured undirected edge predictions using the inference readout map."""

    phys_key = ("component", "physical", "component")
    if phys_key not in data.edge_types or out["edge_preds"].numel() == 0:
        return []

    edge_index = data[phys_key].edge_index.detach().cpu()
    edge_preds = out["edge_preds"].detach().cpu()
    component_names = data["component"].component_name
    component_types = data["component"].component_type

    undirected_groups: dict[tuple[int, int], list[int]] = {}
    for edge_idx in range(edge_index.size(1)):
        src = int(edge_index[0, edge_idx].item())
        dst = int(edge_index[1, edge_idx].item())
        key = (min(src, dst), max(src, dst))
        undirected_groups.setdefault(key, []).append(edge_idx)

    results: list[EdgePrediction] = []
    for (src_idx, dst_idx), indices in undirected_groups.items():
        src_type = component_types[src_idx]
        dst_type = component_types[dst_idx]
        readout = EDGE_INFERENCE_READOUT.get((src_type, dst_type), []) or EDGE_INFERENCE_READOUT.get(
            (dst_type, src_type), []
        )
        if not readout:
            continue

        avg_pred = edge_preds[indices].mean(dim=0)
        for target_index, target_name in enumerate(EDGE_TARGET_NAMES):
            if target_name not in readout:
                continue
            raw_value = float(avg_pred[target_index].item())
            scale = float(target_scales.edge.get(target_name, 1.0))
            results.append(
                EdgePrediction(
                    component_a=component_names[src_idx],
                    component_b=component_names[dst_idx],
                    component_types=(src_type, dst_type),
                    target_name=target_name,
                    raw_value=raw_value,
                    value=raw_value * scale,
                    directions_aggregated=len(indices),
                )
            )
    return results


def read_prediction_summary(
    data: HeteroData,
    out: Mapping[str, torch.Tensor],
    *,
    target_scales: TargetScales = STANDARD_TARGET_SCALES,
) -> PredictionSummary:
    """Return the structured node and edge readout for one graph prediction."""

    return PredictionSummary(
        nodes=read_node_predictions(data, out, target_scales=target_scales),
        edges=read_edge_predictions(data, out, target_scales=target_scales),
    )


__all__ = [
    "DEFAULT_EDGE_TARGET_COLUMNS",
    "DEFAULT_EDGE_TARGET_SCALES",
    "DEFAULT_LAYOUT_PARAM_COLUMNS",
    "DEFAULT_NODE_TARGET_COLUMNS",
    "DEFAULT_NODE_TARGET_SCALES",
    "EdgePrediction",
    "NodePrediction",
    "PredictionSummary",
    "STANDARD_SQUADDS_ROW_SCHEMA",
    "STANDARD_TARGET_SCALES",
    "TargetScales",
    "UniversalModelDims",
    "UniversalRowSchema",
    "attach_targets_from_row",
    "build_graph_dataset",
    "build_graph_from_row",
    "build_layout_from_row",
    "build_model_from_graph",
    "extract_layout_params",
    "infer_model_dims",
    "make_standard_qubit_cavity_netlist",
    "read_edge_predictions",
    "read_node_predictions",
    "read_prediction_summary",
]
