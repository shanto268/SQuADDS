"""Tests for squadds.ml.graph.featurizer."""

from __future__ import annotations

import json

import numpy as np
import pytest

from squadds.ml.graph.featurizer import (
    PAD_TOKEN,
    CircuitGraphBuilder,
    ComponentFeaturizer,
    SQuADDSGraphDataset,
    _flatten_dict,
    _parse_value,
    build_vocab,
)

# ---------------------------------------------------------------------------
# _parse_value
# ---------------------------------------------------------------------------


class TestParseValue:
    def test_float_passthrough(self):
        assert _parse_value(3.14) == pytest.approx(3.14)

    def test_int_passthrough(self):
        assert _parse_value(42) == pytest.approx(42.0)

    def test_um_string(self):
        assert _parse_value("20um") == pytest.approx(20.0)

    def test_mm_string(self):
        assert _parse_value("5mm") == pytest.approx(5000.0)

    def test_nm_string(self):
        assert _parse_value("100nm") == pytest.approx(0.1)

    def test_nonsense_returns_none(self):
        assert _parse_value("main") is None

    def test_empty_string_returns_none(self):
        assert _parse_value("") is None


# ---------------------------------------------------------------------------
# _flatten_dict
# ---------------------------------------------------------------------------


class TestFlattenDict:
    def test_flat_dict(self):
        d = {"a": 1, "b": 2}
        assert _flatten_dict(d) == {"a": 1, "b": 2}

    def test_nested_dict(self):
        d = {"x": {"y": 3, "z": 4}, "w": 5}
        flat = _flatten_dict(d)
        assert flat == {"x.y": 3, "x.z": 4, "w": 5}


# ---------------------------------------------------------------------------
# build_vocab
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_json_dir(tmp_path):
    """Create a temp directory with two minimal component JSONs."""
    comp1 = {
        "component_name": "Comp1",
        "design_parameters": [
            {"parameter_name": "alpha"},
            {"parameter_name": "beta"},
        ],
    }
    comp2 = {
        "component_name": "Comp2",
        "design_parameters": [
            {"parameter_name": "beta"},
            {"parameter_name": "gamma"},
        ],
    }
    (tmp_path / "Comp1.json").write_text(json.dumps(comp1))
    (tmp_path / "Comp2.json").write_text(json.dumps(comp2))
    return tmp_path


class TestBuildVocab:
    def test_basic(self, tmp_json_dir):
        vocab = build_vocab(json_dir=tmp_json_dir)
        assert vocab[PAD_TOKEN] == 0
        assert "alpha" in vocab
        assert "beta" in vocab
        assert "gamma" in vocab
        # alphabetical after PAD
        assert vocab["alpha"] == 1
        assert vocab["beta"] == 2
        assert vocab["gamma"] == 3

    def test_save_and_reload(self, tmp_json_dir, tmp_path):
        save_path = tmp_path / "vocab.json"
        vocab = build_vocab(json_dir=tmp_json_dir, save_path=save_path)
        with open(save_path) as f:
            loaded = json.load(f)
        assert loaded == {k: v for k, v in vocab.items()}


# ---------------------------------------------------------------------------
# ComponentFeaturizer
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_vocab():
    return {PAD_TOKEN: 0, "cross_width": 1, "cross_length": 2, "cross_gap": 3}


class TestComponentFeaturizer:
    def test_featurize_shape(self, simple_vocab, tmp_path):
        # No JSON dir → still works (area/perimeter = 0, ports = 0)
        feat = ComponentFeaturizer(vocab=simple_vocab, json_dir=tmp_path)
        result = feat.featurize(
            "TransmonCross",
            {"cross_width": "20um", "cross_length": "200um", "cross_gap": "20um"},
        )
        assert result["layer_stack"].shape == (5, 3)
        assert isinstance(result["design_params"], list)
        assert len(result["design_params"]) == 3
        assert result["ports"].shape == (4,)

    def test_key_ids_in_vocab(self, simple_vocab, tmp_path):
        feat = ComponentFeaturizer(vocab=simple_vocab, json_dir=tmp_path)
        result = feat.featurize("X", {"cross_width": "10um"})
        key_ids = [kid for kid, _ in result["design_params"]]
        assert 1 in key_ids  # cross_width → 1

    def test_unknown_key_maps_to_pad(self, simple_vocab, tmp_path):
        feat = ComponentFeaturizer(vocab=simple_vocab, json_dir=tmp_path)
        result = feat.featurize("X", {"unknown_param": "5um"})
        key_ids = [kid for kid, _ in result["design_params"]]
        assert key_ids == [0]  # PAD


# ---------------------------------------------------------------------------
# CircuitGraphBuilder
# ---------------------------------------------------------------------------


class TestCircuitGraphBuilder:
    def test_single_node(self, simple_vocab, tmp_path):
        pytest.importorskip("spektral")
        builder = CircuitGraphBuilder(vocab=simple_vocab, k_max=5, json_dir=tmp_path)
        g = builder.build(
            components=[("TransmonCross", {"cross_width": "20um"})],
            edges=[],
            targets=[5.0, -300.0],
        )
        assert g.x.shape[0] == 1  # 1 node
        assert g.a.shape == (1, 1)
        np.testing.assert_array_equal(g.y, np.array([5.0, -300.0], dtype=np.float32))

    def test_two_nodes_one_edge(self, simple_vocab, tmp_path):
        pytest.importorskip("spektral")
        builder = CircuitGraphBuilder(vocab=simple_vocab, k_max=5, json_dir=tmp_path)
        g = builder.build(
            components=[
                ("TransmonCross", {"cross_width": "20um"}),
                ("CavityClaw", {"cross_length": "100um"}),
            ],
            edges=[(0, 1)],
        )
        assert g.x.shape[0] == 2
        # adjacency should be symmetric
        a_dense = g.a.toarray()
        assert a_dense[0, 1] == 1.0
        assert a_dense[1, 0] == 1.0


# ---------------------------------------------------------------------------
# SQuADDSGraphDataset
# ---------------------------------------------------------------------------


class TestSQuADDSGraphDataset:
    def test_split_sizes(self, simple_vocab, tmp_path):
        pytest.importorskip("spektral")
        builder = CircuitGraphBuilder(vocab=simple_vocab, k_max=5, json_dir=tmp_path)
        graphs = [
            builder.build(
                components=[("X", {"cross_width": f"{i}um"})],
                edges=[],
                targets=[float(i)],
            )
            for i in range(20)
        ]
        ds = SQuADDSGraphDataset(graphs, val_split=0.1, test_split=0.1)
        assert len(ds) == 20
        total = len(ds.train_graphs) + len(ds.val_graphs) + len(ds.test_graphs)
        assert total == 20
