"""Tests for the graph builder module (Milestone 3)."""

import pytest
import torch
from torch_geometric.data import Data

from squadds.ml.universal.graph.builder import UniversalGraphBuilder
from squadds.ml.universal.graph.netlist import CircuitNetlist, ComponentSpec, EdgeSpec
from squadds.ml.universal.graph.virtual_hub import VirtualHubInjector


class MockNodeEncoder:
    def __call__(self, comp_name, comp_data):
        # Return a dummy 10-dimensional node feature vector
        return torch.ones(10, dtype=torch.float)


class MockEdgeExtractor:
    def extract(self, poly_a, poly_b):
        # Return dummy edge features
        return {
            "shortest_gap": 1.0,
            "overlap_length": 2.0,
            "metal_area": 3.0,
            "void_area": 4.0,
        }


@pytest.fixture
def dummy_layout():
    return {
        "qubit": {"trace": "poly1"},
        "claw": {"trace": "poly2"},
        "resonator": {"trace": "poly3"},
        "feedline": {"trace": "poly4"},
    }


@pytest.fixture
def dummy_netlist():
    return CircuitNetlist(
        components=[
            ComponentSpec(name="qubit", component_type="TransmonCross"),
            ComponentSpec(name="claw", component_type="Claw"),
            ComponentSpec(name="resonator", component_type="RouteMeander"),
            ComponentSpec(name="feedline", component_type="CoupledLineTee"),
        ],
        edges=[
            EdgeSpec(src="qubit.E", dst="claw.pin", coupling_type="capacitive"),
            EdgeSpec(src="claw.arm", dst="resonator.start", coupling_type="galvanic"),
            EdgeSpec(src="resonator.end", dst="feedline.prime", coupling_type="capacitive"),
        ],
    )


def test_netlist_validation(dummy_netlist):
    # Should not raise
    dummy_netlist.validate()

    # Create invalid netlist
    invalid_netlist = CircuitNetlist(
        components=[ComponentSpec(name="qubit", component_type="TransmonCross")],
        edges=[EdgeSpec(src="qubit.E", dst="claw.pin", coupling_type="capacitive")],
    )
    with pytest.raises(ValueError, match="Edge destination component 'claw' not found"):
        invalid_netlist.validate()


def test_virtual_hub_injection():
    injector = VirtualHubInjector(edge_dim=4)

    # Dummy data with 3 nodes, 2 undirected edges (4 directed)
    x = torch.ones((3, 10))
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_attr = torch.ones((4, 4))
    y = torch.ones((3, 2))

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.y_edge = torch.ones((4, 1))

    global_features = torch.zeros((1, 10))

    new_data = injector.inject(data, global_features)

    # Check node features (3 real + 1 hub)
    assert new_data.x.size(0) == 4
    assert torch.all(new_data.x[-1] == 0)

    # Check targets (3 real + 1 hub NaN)
    assert new_data.y.size(0) == 4
    assert torch.isnan(new_data.y[-1]).all()

    # Check edges (4 original + 3*2 spatial = 10)
    assert new_data.edge_index.size(1) == 10
    assert new_data.edge_attr.size(0) == 10

    # Check edge targets (4 original + 6 spatial NaN)
    assert new_data.y_edge.size(0) == 10
    assert torch.isnan(new_data.y_edge[4:]).all()


def test_graph_builder_without_hub(dummy_layout, dummy_netlist):
    builder = UniversalGraphBuilder(
        node_encoder=MockNodeEncoder(),
        edge_extractor=MockEdgeExtractor(),
        hub_injector=None,
    )

    data = builder.build(dummy_layout, dummy_netlist)

    # 4 components
    assert data.x.size(0) == 4
    assert data.x.size(1) == 10

    # 3 edges in netlist -> 6 directed edges in PyG
    assert data.edge_index.size(1) == 6
    assert data.edge_attr.size(0) == 6
    assert data.edge_attr.size(1) == 4


def test_graph_builder_with_hub(dummy_layout, dummy_netlist):
    injector = VirtualHubInjector(edge_dim=4)
    builder = UniversalGraphBuilder(
        node_encoder=MockNodeEncoder(),
        edge_extractor=MockEdgeExtractor(),
        hub_injector=injector,
    )

    data = builder.build(dummy_layout, dummy_netlist)

    # 4 components + 1 hub = 5 nodes
    assert data.x.size(0) == 5
    assert data.x.size(1) == 10

    # 3 edges (6 directed) + 4 spatial (8 directed) = 14 edges
    assert data.edge_index.size(1) == 14
    assert data.edge_attr.size(0) == 14
    assert data.edge_attr.size(1) == 4
