"""Tests for the current heterogeneous universal graph builder."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from squadds.ml.universal.geometry.layout import build_layout
from squadds.ml.universal.graph.builder import UniversalGraphBuilder
from squadds.ml.universal.graph.netlist import CircuitNetlist, ComponentSpec, EdgeSpec
from squadds.ml.universal.features.protocol import EmbeddingConfig, EmbeddingMode, EmbeddingVersion, embedding_dim
from squadds.ml.universal.workflows import (
    STANDARD_TARGET_SCALES,
    attach_targets_from_row,
    build_graph_dataset,
    build_graph_from_row,
    build_layout_from_row,
    extract_layout_params,
    make_standard_qubit_cavity_netlist,
    read_prediction_summary,
)


@pytest.fixture
def layout():
    return build_layout(
        cross_length=200,
        cross_gap=20,
        claw_length=50,
        ground_spacing=6,
        coupling_length=200,
        total_length=4000,
    )


@pytest.fixture
def netlist():
    return CircuitNetlist(
        components=[
            ComponentSpec(name="qubit", component_type="TransmonCross"),
            ComponentSpec(name="claw", component_type="Claw"),
            ComponentSpec(name="resonator", component_type="RouteMeander"),
            ComponentSpec(name="feedline", component_type="CoupledLineTee"),
        ],
        edges=[
            EdgeSpec(src="qubit.E", dst="claw.readout", coupling_type="capacitive"),
            EdgeSpec(src="claw.readout", dst="resonator.start", coupling_type="galvanic"),
            EdgeSpec(src="resonator.end", dst="feedline.prime_end", coupling_type="capacitive"),
        ],
    )


@pytest.fixture
def row():
    return {
        "cross_length": 200.0,
        "cross_gap": 20.0,
        "claw_length": 50.0,
        "ground_spacing": 6.0,
        "coupling_length": 200.0,
        "total_length": 4000.0,
        "qubit_frequency_GHz": 5.1,
        "anharmonicity_MHz": -250.0,
        "cavity_frequency_GHz": 7.2,
        "g_MHz": 80.0,
        "kappa_kHz": 150.0,
    }


def test_netlist_validation(netlist):
    netlist.validate()

    invalid = CircuitNetlist(
        components=[ComponentSpec(name="qubit", component_type="TransmonCross")],
        edges=[EdgeSpec(src="qubit.E", dst="claw.pin", coupling_type="capacitive")],
    )
    with pytest.raises(ValueError, match="Edge destination component 'claw' not found"):
        invalid.validate()


def test_netlist_to_pyg_edge_index(netlist):
    edge_index = netlist.to_pyg_edge_index()
    assert edge_index.shape == (2, 6)


def test_graph_builder_outputs_heterodata(layout, netlist, tmp_path):
    builder = UniversalGraphBuilder(shape_resolution=8, cache_dir=tmp_path / "graph_cache")
    data = builder.build(layout, netlist, global_features={"dielectric_constant": 11.45})

    assert set(data.node_types) == {"component", "virtual"}
    assert ("component", "physical", "component") in data.edge_types
    assert ("component", "spatial_in", "virtual") in data.edge_types
    assert ("virtual", "spatial_out", "component") in data.edge_types

    assert data["component"].x.shape[0] == 4
    assert data["virtual"].x.shape[0] == 1

    # 3 undirected logical edges -> 6 directed physical edges
    assert data["component", "physical", "component"].edge_index.shape[1] == 6
    assert data["component", "physical", "component"].edge_attr.shape[0] == 6
    assert data["component", "physical", "component"].y.shape == (6, 2)

    # Spatial edges connect every component to the virtual node in both directions
    assert data["component", "spatial_in", "virtual"].edge_index.shape[1] == 4
    assert data["virtual", "spatial_out", "component"].edge_index.shape[1] == 4

    assert len(data["component"].component_name) == 4
    assert len(data["component"].component_type) == 4
    assert len(data["component"].inference_readout) == 4


def test_graph_builder_cache_roundtrip(layout, netlist, tmp_path):
    cache_dir = tmp_path / "graph_cache"
    builder = UniversalGraphBuilder(shape_resolution=8, cache_dir=cache_dir)
    first = builder.build(layout, netlist)
    second = builder.build(layout, netlist)

    assert first["component"].x.shape == second["component"].x.shape
    assert any(cache_dir.iterdir())


def test_graph_builder_supports_versioned_context_embeddings(layout, netlist, tmp_path):
    config = EmbeddingConfig(
        version=EmbeddingVersion.V2_HASHED_PARAMS,
        mode=EmbeddingMode.DEVICE_CONTEXT,
        shape_resolution=8,
        param_hash_dim=4,
    )
    builder = UniversalGraphBuilder(
        shape_resolution=8,
        cache_dir=tmp_path / "graph_cache_v2",
        embedding_config=config,
    )
    data = builder.build(layout, netlist)

    assert data["component"].x.shape[1] == embedding_dim(config)
    assert data["component"].embedding_config["version"] == config.version.value
    assert data["component"].embedding_config["mode"] == config.mode.value


def test_graph_builder_cache_separates_embedding_configs(layout, netlist, tmp_path):
    cache_dir = tmp_path / "shared_cache"

    legacy_builder = UniversalGraphBuilder(shape_resolution=8, cache_dir=cache_dir)
    legacy = legacy_builder.build(layout, netlist)

    v2_config = EmbeddingConfig(
        version=EmbeddingVersion.V2_HASHED_PARAMS,
        mode=EmbeddingMode.DEVICE_CONTEXT,
        shape_resolution=8,
        param_hash_dim=4,
    )
    v2_builder = UniversalGraphBuilder(shape_resolution=8, cache_dir=cache_dir, embedding_config=v2_config)
    upgraded = v2_builder.build(layout, netlist)

    assert legacy["component"].x.shape[1] != upgraded["component"].x.shape[1]
    assert len(list(cache_dir.iterdir())) == 2


def test_make_standard_qubit_cavity_netlist():
    standard = make_standard_qubit_cavity_netlist()

    assert [component.name for component in standard.components] == ["qubit", "claw", "resonator", "feedline"]
    assert [edge.coupling_type for edge in standard.edges] == ["capacitive", "galvanic", "capacitive"]


def test_extract_layout_params_from_row(row):
    params = extract_layout_params(row)

    assert params == {
        "cross_length": 200.0,
        "cross_gap": 20.0,
        "claw_length": 50.0,
        "ground_spacing": 6.0,
        "coupling_length": 200.0,
        "total_length": 4000.0,
    }


def test_build_layout_from_row(row):
    built = build_layout_from_row(row)

    assert set(built) >= {"qubit", "claw", "resonator", "feedline", "design_params"}
    assert built["design_params"]["cross_length"] == 200.0


def test_attach_targets_from_row_copies_by_default(row, tmp_path):
    builder = UniversalGraphBuilder(shape_resolution=8, cache_dir=tmp_path / "graph_cache")
    graph = builder.build(build_layout_from_row(row), make_standard_qubit_cavity_netlist())

    labeled = attach_targets_from_row(graph, row)

    assert torch.all(graph["component"].y == 0)
    assert labeled["component"].y[0, 0].item() == pytest.approx(5.1)
    assert labeled["component"].y[0, 1].item() == pytest.approx(-2.5)
    assert labeled["component", "physical", "component"].y[0, 0].item() == pytest.approx(0.8)
    assert labeled["component", "physical", "component"].y[0, 1].item() == pytest.approx(1.5)


def test_build_graph_from_row_attaches_targets(row, tmp_path):
    builder = UniversalGraphBuilder(shape_resolution=8, cache_dir=tmp_path / "graph_cache")
    graph = build_graph_from_row(
        row,
        netlist=make_standard_qubit_cavity_netlist(),
        builder=builder,
        global_features={"dielectric_constant": 11.45},
    )

    assert graph["component"].y.shape == (4, 3)
    assert graph["component", "physical", "component"].y.shape == (6, 2)
    assert graph["component"].y[:, 2].tolist() == pytest.approx([7.2, 7.2, 7.2, 7.2])


def test_build_graph_dataset_from_rows(row, tmp_path):
    builder = UniversalGraphBuilder(shape_resolution=8, cache_dir=tmp_path / "graph_cache")
    second_row = dict(row)
    second_row["qubit_frequency_GHz"] = 5.4

    dataset = build_graph_dataset(
        [row, second_row],
        netlist=make_standard_qubit_cavity_netlist(),
        builder=builder,
    )

    assert len(dataset) == 2
    assert dataset[1]["component"].y[0, 0].item() == pytest.approx(5.4)


def test_read_prediction_summary_scales_and_filters_readout(row, tmp_path):
    builder = UniversalGraphBuilder(shape_resolution=8, cache_dir=tmp_path / "graph_cache")
    graph = build_graph_from_row(
        row,
        netlist=make_standard_qubit_cavity_netlist(),
        builder=builder,
    )

    out = {
        "node_preds": torch.tensor(
            [
                [5.1, -2.5, 7.2],
                [9.9, 9.9, 9.9],
                [3.3, 3.3, 7.4],
                [8.8, 8.8, 8.8],
            ],
            dtype=torch.float32,
        ),
        "edge_preds": torch.tensor(
            [
                [0.8, 0.1],
                [0.8, 0.2],
                [0.1, 0.1],
                [0.1, 0.1],
                [0.2, 1.5],
                [0.3, 1.5],
            ],
            dtype=torch.float32,
        ),
    }

    summary = read_prediction_summary(graph, out, target_scales=STANDARD_TARGET_SCALES)

    assert len(summary.nodes) == 3
    assert summary.nodes[0].component_name == "qubit"
    assert summary.nodes[0].target_name == "qubit_freq_GHz"
    assert summary.nodes[0].value == pytest.approx(5.1)
    assert summary.nodes[1].target_name == "anharmonicity_MHz"
    assert summary.nodes[1].value == pytest.approx(-250.0)
    assert summary.nodes[2].component_name == "resonator"
    assert summary.nodes[2].target_name == "cavity_freq_GHz"
    assert summary.nodes[2].value == pytest.approx(7.4)

    assert len(summary.edges) == 2
    assert summary.edges[0].component_a == "qubit"
    assert summary.edges[0].component_b == "claw"
    assert summary.edges[0].target_name == "g_MHz"
    assert summary.edges[0].value == pytest.approx(80.0)
    assert summary.edges[1].component_a == "resonator"
    assert summary.edges[1].component_b == "feedline"
    assert summary.edges[1].target_name == "kappa_kHz"
    assert summary.edges[1].value == pytest.approx(150.0)
    assert all(item.directions_aggregated == 2 for item in summary.edges)
