"""Tests for the current universal feature extractors."""

from __future__ import annotations

import numpy as np
import pytest
from shapely import affinity
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

torch = pytest.importorskip("torch")

from squadds.ml.universal.features.arithmetic import (
    compute_shared_frame_embedding,
    compute_shared_frame_shape_embedding,
    embedding_cosine_similarity,
)
from squadds.ml.universal.features.cnn_encoder import CNNEncoder
from squadds.ml.universal.features.deepsets import DeepSetsEncoder
from squadds.ml.universal.features.edge_extractor import (
    EdgeFeatureExtractor,
    edge_feature_dim,
    extract_edge_features,
)
from squadds.ml.universal.features.moments import compute_moments, moment_names
from squadds.ml.universal.features.node_encoder import (
    compute_static_embedding,
    get_polygon_for_component,
    static_embedding_dim,
)
from squadds.ml.universal.features.protocol import (
    EmbeddingConfig,
    EmbeddingMode,
    EmbeddingVersion,
    compute_component_embedding,
    embedding_dim,
    encode_params,
)
from squadds.ml.universal.features.rasterizer import compute_shared_bounds, rasterize_fast, rasterize_in_bounds
from squadds.ml.universal.benchmarks import (
    STANDARD_ARITHMETIC_SPECS,
    benchmark_component_family_clustering,
    benchmark_standard_embedding_arithmetic,
    build_component_embedding_collection,
    evaluate_standard_arithmetic_case,
)
from squadds.ml.universal.geometry.claw import make_claw
from squadds.ml.universal.geometry.layout import build_layout
from squadds.ml.universal.visualization import (
    compute_cosine_similarity_matrix,
    compute_embedding_projection,
    compute_embedding_projections,
    compute_label_centroids,
    find_nearest_neighbors,
    plot_embedding_projection,
    plot_projection_grid,
    plot_similarity_bars,
    rank_difference_vector,
)


class TestMoments:
    def test_square_area_perimeter(self):
        square = box(0, 0, 10, 10)
        m = compute_moments(square)
        assert abs(m[0] - 100.0) < 0.01
        assert abs(m[1] - 40.0) < 0.01

    def test_aspect_ratio(self):
        rect = box(0, 0, 20, 10)
        m = compute_moments(rect)
        assert abs(m[6] - 2.0) < 0.01

    def test_fill_factor_square(self):
        square = box(0, 0, 10, 10)
        m = compute_moments(square)
        assert abs(m[4] - 1.0) < 0.01

    def test_output_shape(self):
        square = box(0, 0, 10, 10)
        m = compute_moments(square)
        assert m.shape == (8,)
        assert m.dtype == np.float32

    def test_moment_names_length(self):
        assert len(moment_names()) == 8

    def test_circularity_positive(self):
        square = box(0, 0, 10, 10)
        m = compute_moments(square)
        assert m[7] > 0


class TestRasterizer:
    def test_output_shape(self):
        square = box(0, 0, 100, 100)
        mask = rasterize_fast(square, resolution=64)
        assert mask.shape == (64, 64)
        assert mask.dtype == np.float32

    def test_nonempty(self):
        square = box(0, 0, 100, 100)
        mask = rasterize_fast(square, resolution=64)
        assert mask.sum() > 0

    def test_fill_ratio(self):
        square = box(0, 0, 100, 100)
        mask = rasterize_fast(square, resolution=64)
        fill = mask.sum() / (64 * 64)
        assert fill > 0.5

    def test_empty_polygon(self):
        empty = Polygon()
        mask = rasterize_fast(empty, resolution=64)
        assert mask.sum() == 0

    def test_different_resolution(self):
        square = box(0, 0, 100, 100)
        mask = rasterize_fast(square, resolution=32)
        assert mask.shape == (32, 32)

    def test_compute_shared_bounds_contains_all_polygons(self):
        a = box(0, 0, 10, 10)
        b = box(15, -5, 25, 5)
        bounds = compute_shared_bounds([a, b], padding=1.0)
        assert bounds == pytest.approx((-1.0, -6.0, 26.0, 11.0))

    def test_rasterize_in_bounds_shape(self):
        square = box(0, 0, 10, 10)
        bounds = compute_shared_bounds([square], padding=2.0)
        mask = rasterize_in_bounds(square, bounds, resolution=32)
        assert mask.shape == (32, 32)
        assert mask.sum() > 0


class TestCNNEncoder:
    def test_output_shape(self):
        cnn = CNNEncoder(out_dim=128)
        x = torch.randn(2, 1, 64, 64)
        out = cnn(x)
        assert out.shape == (2, 128)

    def test_3d_input(self):
        cnn = CNNEncoder(out_dim=64)
        x = torch.randn(3, 64, 64)
        out = cnn(x)
        assert out.shape == (3, 64)

    def test_gradient_flow(self):
        cnn = CNNEncoder(out_dim=32)
        x = torch.randn(1, 1, 64, 64, requires_grad=True)
        out = cnn(x)
        out.sum().backward()
        assert x.grad is not None


class TestDeepSets:
    @pytest.fixture
    def encoder(self):
        return DeepSetsEncoder(vocab_size=10, embed_dim=16, hidden_dim=32, out_dim=32)

    def test_output_shape(self, encoder):
        keys = torch.tensor([[0, 1, 2]], dtype=torch.long)
        vals = torch.tensor([[1.0, 2.0, 3.0]])
        out = encoder(keys, vals)
        assert out.shape == (1, 32)

    def test_permutation_invariance(self, encoder):
        keys_a = torch.tensor([[0, 1, 2]], dtype=torch.long)
        vals_a = torch.tensor([[1.0, 2.0, 3.0]])

        keys_b = torch.tensor([[2, 0, 1]], dtype=torch.long)
        vals_b = torch.tensor([[3.0, 1.0, 2.0]])

        encoder.eval()
        with torch.no_grad():
            out_a = encoder(keys_a, vals_a)
            out_b = encoder(keys_b, vals_b)

        assert torch.allclose(out_a, out_b, atol=1e-5)

    def test_masking(self, encoder):
        keys = torch.tensor([[0, 1, 2]], dtype=torch.long)
        vals = torch.tensor([[1.0, 2.0, 999.0]])
        mask = torch.tensor([[True, True, False]])

        keys2 = torch.tensor([[0, 1, 0]], dtype=torch.long)
        vals2 = torch.tensor([[1.0, 2.0, 0.0]])
        mask2 = torch.tensor([[True, True, False]])

        encoder.eval()
        with torch.no_grad():
            out1 = encoder(keys, vals, mask)
            out2 = encoder(keys2, vals2, mask2)

        assert torch.allclose(out1, out2, atol=1e-5)

    def test_build_vocab(self):
        vocab = DeepSetsEncoder.build_vocab(["cross_length", "cross_gap", "claw_length", "cross_length"])
        assert len(vocab) == 3
        assert "cross_length" in vocab

    def test_encode_params(self):
        vocab = {"a": 0, "b": 1, "c": 2}
        keys, vals, mask = DeepSetsEncoder.encode_params({"a": 1.0, "c": 3.0}, vocab, max_params=5)
        assert keys.shape == (5,)
        assert vals.shape == (5,)
        assert mask.sum().item() == 2


class TestEdgeExtractor:
    def test_feature_shape(self):
        a = box(0, 0, 10, 10)
        b = box(15, 0, 25, 10)
        features = extract_edge_features(a, b, coupling_type="capacitive", shape_resolution=16)
        assert features.shape == (edge_feature_dim(16),)

    def test_coupling_one_hot(self):
        a = box(0, 0, 10, 10)
        b = box(15, 0, 25, 10)
        features = extract_edge_features(a, b, coupling_type="capacitive", shape_resolution=16)
        np.testing.assert_array_equal(features[:3], np.array([1.0, 0.0, 0.0], dtype=np.float32))

    def test_relative_center_direction(self):
        a = box(0, 0, 10, 10)
        b = box(15, 0, 25, 10)
        features = extract_edge_features(a, b, coupling_type="capacitive", shape_resolution=16)
        assert features[3] > 0  # dx

    def test_touching_polygons_still_embed(self):
        a = box(0, 0, 10, 10)
        b = box(10, 0, 20, 10)
        features = extract_edge_features(a, b, coupling_type="galvanic", shape_resolution=16)
        assert np.isfinite(features).all()

    def test_extractor_dim(self):
        extractor = EdgeFeatureExtractor(shape_resolution=16)
        assert extractor.dim == edge_feature_dim(16)


class TestStaticEmbedding:
    def test_static_embedding_shape(self):
        square = box(0, 0, 100, 100)
        emb = compute_static_embedding(square, params={"a": 1.0, "b": 2.0}, shape_resolution=16)
        assert emb.shape == (static_embedding_dim(16),)

    def test_param_sum_is_first_value(self):
        square = box(0, 0, 100, 100)
        emb = compute_static_embedding(square, params={"a": 1.0, "b": 2.0}, shape_resolution=16)
        assert emb[0] == pytest.approx(3.0)

    def test_get_polygon_for_component_trace(self):
        square = box(0, 0, 100, 100)
        out = get_polygon_for_component({"trace": square})
        assert out.equals(square)

    def test_get_polygon_for_component_polygon_key(self):
        square = box(0, 0, 100, 100)
        out = get_polygon_for_component({"polygon": square})
        assert out.equals(square)

    def test_get_polygon_for_component_raises(self):
        with pytest.raises(ValueError, match="No polygon found"):
            get_polygon_for_component({"params": {"a": 1}})


class TestEmbeddingArithmetic:
    def test_shared_frame_shape_subtraction_is_exact_for_disjoint_polygons(self):
        a = box(0, 0, 10, 10)
        b = box(15, 0, 25, 10)
        union = unary_union([a, b])
        bounds = compute_shared_bounds([a, b], padding_fraction=0.05)

        e_union = compute_shared_frame_shape_embedding(union, bounds=bounds, shape_resolution=32)
        e_b = compute_shared_frame_shape_embedding(b, bounds=bounds, shape_resolution=32)
        e_a = compute_shared_frame_shape_embedding(a, bounds=bounds, shape_resolution=32)

        np.testing.assert_allclose(e_union - e_b, e_a)
        assert embedding_cosine_similarity(e_union - e_b, e_a) == pytest.approx(1.0)

    def test_shared_frame_full_embedding_matches_standard_dim(self):
        square = box(0, 0, 100, 100)
        emb = compute_shared_frame_embedding(
            square,
            reference_polygons=[square],
            params={"a": 1.0, "b": 2.0},
            shape_resolution=16,
        )
        assert emb.shape == (static_embedding_dim(16),)
        assert emb[0] == pytest.approx(3.0)

    def test_layout_qubit_difference_stays_close_in_shared_frame(self):
        layout = build_layout(
            cross_length=200,
            cross_gap=20,
            claw_length=50,
            ground_spacing=6,
            coupling_length=200,
            total_length=4000,
        )
        qubit = get_polygon_for_component(layout["qubit"])
        claw = get_polygon_for_component(layout["claw"])
        union = unary_union([qubit, claw])
        diff = union.difference(claw)
        bounds = compute_shared_bounds([qubit, claw], padding_fraction=0.05)

        e_diff = compute_shared_frame_shape_embedding(diff, bounds=bounds, shape_resolution=48)
        e_qubit = compute_shared_frame_shape_embedding(qubit, bounds=bounds, shape_resolution=48)

        assert embedding_cosine_similarity(e_diff, e_qubit) > 0.99


class TestVersionedEmbeddingProtocol:
    def test_v2_param_encoding_distinguishes_same_sum(self):
        config = EmbeddingConfig(version=EmbeddingVersion.V2_HASHED_PARAMS, param_hash_dim=8)
        enc_a = encode_params({"cross_length": 300.0, "cross_gap": 30.0}, config=config)
        enc_b = encode_params({"cross_length": 280.0, "cross_gap": 50.0}, config=config)

        assert enc_a.shape == enc_b.shape == (16,)
        assert not np.allclose(enc_a, enc_b)

    def test_component_embedding_dim_matches_config(self):
        polygon = box(0, 0, 10, 10)
        config = EmbeddingConfig(
            version=EmbeddingVersion.V2_HASHED_PARAMS,
            mode=EmbeddingMode.GEOMETRY_ONLY,
            shape_resolution=16,
            param_hash_dim=4,
        )
        emb = compute_component_embedding(polygon, params={"a": 1.0, "b": 2.0}, config=config)
        assert emb.shape == (embedding_dim(config),)


class TestEmbeddingRobustness:
    def test_geometry_only_translation_invariance(self):
        polygon = box(0, 0, 10, 20)
        moved = affinity.translate(polygon, xoff=123.0, yoff=-77.0)
        config = EmbeddingConfig(version=EmbeddingVersion.V2_HASHED_PARAMS, mode=EmbeddingMode.GEOMETRY_ONLY)

        emb_a = compute_component_embedding(polygon, params={"a": 1.0, "b": 2.0}, config=config)
        emb_b = compute_component_embedding(moved, params={"a": 1.0, "b": 2.0}, config=config)

        np.testing.assert_allclose(emb_a, emb_b)

    def test_geometry_only_rotation_is_not_invariant_for_asymmetric_shape(self):
        shape = Polygon([(0, 0), (20, 0), (20, 5), (8, 5), (8, 15), (0, 15)])
        rotated = affinity.rotate(shape, 90, origin="centroid")
        config = EmbeddingConfig(mode=EmbeddingMode.GEOMETRY_ONLY, shape_resolution=32)

        emb_a = compute_component_embedding(shape, config=config)
        emb_b = compute_component_embedding(rotated, config=config)

        assert embedding_cosine_similarity(emb_a, emb_b) < 0.999

    def test_geometry_only_mirroring_is_not_invariant_for_asymmetric_shape(self):
        shape = Polygon([(0, 0), (20, 0), (20, 5), (8, 5), (8, 15), (0, 15)])
        mirrored = affinity.scale(shape, xfact=-1.0, yfact=1.0, origin="centroid")
        config = EmbeddingConfig(mode=EmbeddingMode.GEOMETRY_ONLY, shape_resolution=32)

        emb_a = compute_component_embedding(shape, config=config)
        emb_b = compute_component_embedding(mirrored, config=config)

        assert embedding_cosine_similarity(emb_a, emb_b) < 0.999

    def test_device_context_padding_changes_embedding_but_keeps_it_close(self):
        polygon = box(0, 0, 10, 10)
        refs = [polygon, affinity.translate(polygon, xoff=15)]
        config_a = EmbeddingConfig(
            mode=EmbeddingMode.DEVICE_CONTEXT,
            version=EmbeddingVersion.V2_HASHED_PARAMS,
            shape_resolution=32,
            param_hash_dim=4,
            shared_bounds_padding_fraction=0.05,
        )
        config_b = EmbeddingConfig(
            mode=EmbeddingMode.DEVICE_CONTEXT,
            version=EmbeddingVersion.V2_HASHED_PARAMS,
            shape_resolution=32,
            param_hash_dim=4,
            shared_bounds_padding_fraction=0.25,
        )

        emb_a = compute_component_embedding(polygon, params={"a": 1.0}, config=config_a, reference_polygons=refs)
        emb_b = compute_component_embedding(polygon, params={"a": 1.0}, config=config_b, reference_polygons=refs)

        assert not np.allclose(emb_a, emb_b)
        assert embedding_cosine_similarity(emb_a, emb_b) > 0.6

    def test_exporter_noise_collinear_vertices_do_not_change_embedding(self):
        clean = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        noisy = Polygon([(0, 0), (5, 0), (10, 0), (10, 5), (10, 10), (5, 10), (0, 10), (0, 5)])
        config = EmbeddingConfig(version=EmbeddingVersion.V2_HASHED_PARAMS, param_hash_dim=4)

        emb_clean = compute_component_embedding(clean, params={"a": 1.0}, config=config)
        emb_noisy = compute_component_embedding(noisy, params={"a": 1.0}, config=config)

        np.testing.assert_allclose(emb_clean, emb_noisy)


class TestEmbeddingVisualization:
    def test_pca_projection_shape(self):
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.9, 0.1],
            ],
            dtype=np.float32,
        )
        projection = compute_embedding_projection(embeddings, method="pca")
        assert projection.shape == (4, 2)

    def test_compute_embedding_projections_multiple(self):
        embeddings = np.eye(4, dtype=np.float32)
        projections = compute_embedding_projections(embeddings, methods=("pca",))
        assert set(projections) == {"pca"}
        assert projections["pca"].shape == (4, 2)

    def test_cosine_similarity_matrix_is_symmetric(self):
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        sim = compute_cosine_similarity_matrix(embeddings)
        np.testing.assert_allclose(sim, sim.T)
        np.testing.assert_allclose(np.diag(sim), np.ones(3))

    def test_find_nearest_neighbors_orders_results(self):
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.99, 0.01, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        neighbors = find_nearest_neighbors(
            embeddings[0],
            embeddings,
            labels=["A", "A-like", "B"],
            identifiers=["zero", "close", "far"],
            top_k=2,
            exclude_index=0,
        )

        assert [n.identifier for n in neighbors] == ["close", "far"]
        assert neighbors[0].similarity > neighbors[1].similarity

    def test_rank_difference_vector_prefers_matching_centroid(self):
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.95, 0.05, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.95, 0.05],
            ],
            dtype=np.float32,
        )
        labels = ["Qubit", "Qubit", "Claw", "Claw"]
        matches = rank_difference_vector(np.array([1.0, 0.0, 0.0], dtype=np.float32), embeddings, labels)
        assert matches[0].label == "Qubit"
        assert matches[0].similarity > matches[1].similarity

    def test_compute_label_centroids(self):
        embeddings = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        centroids = compute_label_centroids(embeddings, ["A", "A", "B"])
        np.testing.assert_allclose(centroids["A"], np.array([2.0, 0.0], dtype=np.float32))
        np.testing.assert_allclose(centroids["B"], np.array([0.0, 2.0], dtype=np.float32))

    def test_plot_helpers_return_axes_and_figure(self):
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.9, 0.1],
            ],
            dtype=np.float32,
        )
        labels = ["Qubit", "Qubit", "Claw", "Claw"]
        projection = compute_embedding_projection(embeddings, method="pca")

        ax = plot_embedding_projection(projection, labels, title="Projection")
        assert ax.get_title() == "Projection"
        fig, axes = plot_projection_grid({"PCA": projection}, labels, suptitle="Grid")
        assert len(axes) == 1
        assert fig._suptitle.get_text() == "Grid"
        ax_bar = plot_similarity_bars({"Qubit": 0.9, "Claw": 0.1}, title="Similarity")
        assert ax_bar.get_title() == "Similarity"


class TestEmbeddingBenchmarks:
    def test_build_component_embedding_collection_shape(self):
        rows = [
            {
                "cross_length": 220.0,
                "cross_gap": 22.0,
                "claw_length": 52.0,
                "ground_spacing": 6.0,
                "coupling_length": 210.0,
                "total_length": 3900.0,
            },
            {
                "cross_length": 260.0,
                "cross_gap": 28.0,
                "claw_length": 58.0,
                "ground_spacing": 8.0,
                "coupling_length": 240.0,
                "total_length": 4300.0,
            },
        ]

        collection = build_component_embedding_collection(rows)

        assert collection.embeddings.shape[0] == 8
        assert collection.embeddings.shape[1] > 0
        assert collection.labels[:4] == ["Qubit", "Claw", "Resonator", "Feedline"]
        assert collection.identifiers[0] == "row-0:qubit"

    def test_clustering_benchmark_reports_strong_separation(self):
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.98, 0.02, 0.0],
                [0.0, 1.0, 0.0],
                [0.02, 0.98, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.02, 0.98],
            ],
            dtype=np.float32,
        )
        labels = ["Qubit", "Qubit", "Claw", "Claw", "Resonator", "Resonator"]

        result = benchmark_component_family_clustering(embeddings, labels)

        assert result.centroid_top1_accuracy == pytest.approx(1.0)
        assert result.nearest_neighbor_top1_accuracy == pytest.approx(1.0)
        assert result.mean_intra_label_similarity > result.mean_inter_label_similarity
        assert result.separation_gap > 0.8
        assert {item.label for item in result.per_label} == {"Qubit", "Claw", "Resonator"}

    def test_single_arithmetic_case_prefers_expected_component(self):
        row = {
            "cross_length": 240.0,
            "cross_gap": 24.0,
            "claw_length": 55.0,
            "ground_spacing": 7.0,
            "coupling_length": 230.0,
            "total_length": 4100.0,
        }

        trial = evaluate_standard_arithmetic_case(
            row,
            STANDARD_ARITHMETIC_SPECS[0],
            shape_resolution=48,
        )

        assert trial.expected_label == "Qubit"
        assert trial.top1_success
        assert trial.matches[0].label == "Qubit"
        assert trial.expected_similarity > 0.95

    def test_arithmetic_benchmark_runs_across_standard_cases(self):
        rows = [
            {
                "cross_length": 230.0,
                "cross_gap": 23.0,
                "claw_length": 50.0,
                "ground_spacing": 6.0,
                "coupling_length": 220.0,
                "total_length": 4000.0,
            },
            {
                "cross_length": 270.0,
                "cross_gap": 27.0,
                "claw_length": 60.0,
                "ground_spacing": 9.0,
                "coupling_length": 250.0,
                "total_length": 4400.0,
            },
        ]

        result = benchmark_standard_embedding_arithmetic(rows, shape_resolution=48)

        assert result.num_trials == len(rows) * len(STANDARD_ARITHMETIC_SPECS)
        assert result.top1_accuracy >= 2.0 / 3.0
        assert result.top2_accuracy >= result.top1_accuracy
        assert len(result.per_case) == len(STANDARD_ARITHMETIC_SPECS)
