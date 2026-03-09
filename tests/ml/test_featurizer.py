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
        assert vocab["alpha"] == 1
        assert vocab["beta"] == 2
        assert vocab["gamma"] == 3

    def test_save_and_reload(self, tmp_json_dir, tmp_path):
        save_path = tmp_path / "vocab.json"
        vocab = build_vocab(json_dir=tmp_json_dir, save_path=save_path)
        with open(save_path) as f:
            loaded = json.load(f)
        assert loaded == {k: v for k, v in vocab.items()}

    def test_extra_jsons(self, tmp_json_dir, tmp_path):
        extra_json = tmp_path / "Extra.json"
        extra_json.write_text(
            json.dumps(
                {
                    "component_name": "Extra",
                    "design_parameters": [{"parameter_name": "delta"}],
                }
            )
        )
        vocab = build_vocab(json_dir=tmp_json_dir, extra_jsons=[str(extra_json)])
        assert "delta" in vocab


# ---------------------------------------------------------------------------
# ComponentFeaturizer
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_vocab():
    return {PAD_TOKEN: 0, "cross_width": 1, "cross_length": 2, "cross_gap": 3}


class TestComponentFeaturizer:
    def test_featurize_shape(self, simple_vocab, tmp_path):
        feat = ComponentFeaturizer(vocab=simple_vocab, json_dir=tmp_path)
        result = feat.featurize(
            "TransmonCross",
            {"cross_width": "20um", "cross_length": "200um", "cross_gap": "20um"},
        )
        # layer_stack is now a list of tuples, not an ndarray
        assert isinstance(result["layer_stack"], list)
        assert isinstance(result["design_params"], list)
        assert len(result["design_params"]) == 3
        assert result["ports"].shape == (5,)  # 5-element port vector

    def test_custom_layer_stack(self, simple_vocab, tmp_path):
        feat = ComponentFeaturizer(vocab=simple_vocab, json_dir=tmp_path)
        custom_ls = [(500.0, 11.45), (0.1, 0.0), (0.0, 1.0)]
        result = feat.featurize(
            "X",
            {"cross_width": "10um"},
            layer_stack=custom_ls,
        )
        assert result["layer_stack"] == custom_ls

    def test_custom_ports_vector(self, simple_vocab, tmp_path):
        feat = ComponentFeaturizer(vocab=simple_vocab, json_dir=tmp_path)
        result = feat.featurize(
            "X",
            {"cross_width": "10um"},
            ports_vector=[1, 2, 0, 0, 1],
        )
        np.testing.assert_array_equal(result["ports"], [1, 2, 0, 0, 1])

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
# CircuitGraphBuilder  —  new dict-based component format
# ---------------------------------------------------------------------------


class TestCircuitGraphBuilder:
    def test_single_node(self, simple_vocab, tmp_path):
        pytest.importorskip("spektral")
        builder = CircuitGraphBuilder(vocab=simple_vocab, k_max=5, json_dir=tmp_path)
        g = builder.build(
            components=[{"type": "TransmonCross", "design_overrides": {"cross_width": "20um"}}],
            edges=[],
            targets=[5.0, -300.0],
        )
        assert g.x.shape[0] == 1
        assert g.a.shape == (1, 1)
        np.testing.assert_array_equal(g.y, np.array([5.0, -300.0], dtype=np.float32))

    def test_two_nodes_one_edge(self, simple_vocab, tmp_path):
        pytest.importorskip("spektral")
        builder = CircuitGraphBuilder(vocab=simple_vocab, k_max=5, json_dir=tmp_path)
        g = builder.build(
            components=[
                {"type": "TransmonCross", "design_overrides": {"cross_width": "20um"}},
                {"type": "CavityClaw", "design_overrides": {"cross_length": "100um"}},
            ],
            edges=[(0, 1)],
        )
        assert g.x.shape[0] == 2
        a_dense = g.a.toarray()
        assert a_dense[0, 1] == 1.0
        assert a_dense[1, 0] == 1.0

    def test_custom_ports_and_layer_stack(self, simple_vocab, tmp_path):
        pytest.importorskip("spektral")
        builder = CircuitGraphBuilder(vocab=simple_vocab, k_max=5, n_ls=3, json_dir=tmp_path)
        g = builder.build(
            components=[
                {
                    "type": "X",
                    "design_overrides": {"cross_width": "10um"},
                    "layer_stack": [(350.0, 11.45), (0.25, 0.0)],
                    "ports_vector": [1, 0, 2, 0, 1],
                },
            ],
            edges=[],
            targets=[1.0],
        )
        # n_ls=3, so feature starts with 3*2=6 floats for layer stack
        assert g.x.shape[1] == 3 * 2 + 5 * 2 + 2 + 5  # 6+10+2+5=23


# ---------------------------------------------------------------------------
# SQuADDSGraphDataset
# ---------------------------------------------------------------------------


class TestSQuADDSGraphDataset:
    def test_split_sizes(self, simple_vocab, tmp_path):
        pytest.importorskip("spektral")
        builder = CircuitGraphBuilder(vocab=simple_vocab, k_max=5, json_dir=tmp_path)
        graphs = [
            builder.build(
                components=[{"type": "X", "design_overrides": {"cross_width": f"{i}um"}}],
                edges=[],
                targets=[float(i)],
            )
            for i in range(20)
        ]
        ds = SQuADDSGraphDataset(graphs, val_split=0.1, test_split=0.1)
        assert len(ds) == 20
        total = len(ds.train_graphs) + len(ds.val_graphs) + len(ds.test_graphs)
        assert total == 20
