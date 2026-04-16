# Universal Embedding Protocol

Date: 2026-04-15
Repo: `/Users/shanto/LFL/SQuADDS_Refactor`
Working branch: `codex/universal-foundation-pass1`

## Purpose

This note captures the current answer to the C2 questions for the universal embedding track:

- `C2.2` invariance / equivariance desiderata
- `C2.3` robustness testing strategy
- `C2.4` improved parameter encoding
- `C2.5` embedding mode taxonomy
- `C2.6` explicit embedding versioning

It is intentionally narrower than the large branch context doc. This file is the protocol note for the embedding layer itself.

## Protocol Summary

The embedding protocol now has two explicit axes:

1. `version`
2. `mode`

Both are encoded in [`squadds/ml/universal/features/protocol.py`](squadds/ml/universal/features/protocol.py).

## Versions

### `v1_legacy`

This is the existing branch behavior:

- parameter features: `param_sum` only
- moments: 8 deterministic geometric moments
- shape tensor: rasterized mask in the component's own bounding box

Use this for:

- backward compatibility
- reproducing current Tutorial 12 behavior
- keeping the current graph builder default stable

### `v2_hashed_params`

This is the first improved deterministic parameter encoder.

Parameter features are now:

- 8 global parameter statistics:
  - count
  - sum
  - mean
  - std
  - min
  - max
  - L1 norm
  - L2 norm
- plus `param_hash_dim` hashed bins accumulated by parameter name

Why this exists:

- it distinguishes parameter dictionaries that share the same sum
- it remains deterministic and lightweight
- it avoids introducing learned parameter encoders too early

This is the current answer to `C2.4`.

## Modes

### `geometry_only`

Component-local embedding.

Behavior:

- shape tensor uses the component's own bounding box
- translation invariant
- intentionally context-free
- best when the goal is component identity / family clustering

### `component_context`

Shared-frame embedding over a caller-provided local context set.

Behavior:

- shape tensor uses shared bounds over the provided reference polygons
- lets arithmetic / composition become meaningful
- useful when nearby context matters

### `device_context`

Shared-frame embedding over the full device context.

Behavior:

- shape tensor uses shared bounds over the full device polygon set
- preserves relative placement within the device frame
- most relevant for transfer/generalization or later GDS-native workflows

Current implementation note:

- inside `UniversalGraphBuilder`, both non-`geometry_only` modes currently use the full component list as the shared reference set
- so today `component_context` and `device_context` are semantically distinct but operationally the same in the builder
- this is acceptable for now; finer local-context selection can be added later

This is the current answer to `C2.5`.

## Invariance / Equivariance Matrix

These are the current intended behaviors.

| Transformation | `geometry_only` | `component_context` / `device_context` | Notes |
|---|---|---|---|
| Translation | invariant | variant if shared frame changes; invariant if frame translates with geometry | component-local identity should not depend on absolute position |
| Rotation | not invariant by default | not invariant by default | orientation may matter physically |
| Mirroring | not invariant by default | not invariant by default | mirrored couplers can encode different semantics |
| Added shared-frame padding | not applicable | should stay close but not identical | current robustness target is neighborhood stability, not exact invariance |
| Collinear-vertex / tessellation noise | invariant | invariant | exporter noise should not change the embedding materially |
| Multipart geometry representation | invariant | invariant | this required fixing `MultiPolygon` handling in moments/rasterization |
| Layer-number remapping across groups | future canonicalization target | future canonicalization target | full solution depends on GDS-native layer ontology |

This is the current answer to `C2.2`.

## Robustness Coverage Added Tonight

Current automated checks live mainly in [`tests/ml/universal/test_features.py`](tests/ml/universal/test_features.py) and [`tests/ml/universal/test_graph.py`](tests/ml/universal/test_graph.py).

Covered now:

- exact shared-frame subtraction on disjoint shapes
- shared-frame arithmetic closeness on real qubit/claw geometry
- translation invariance for `geometry_only`
- non-invariance to rotation for asymmetric shapes
- non-invariance to mirroring for asymmetric shapes
- padding sensitivity but neighborhood closeness for context-aware modes
- exporter-noise robustness for collinear-vertex polygon variants
- builder support for versioned/context-aware embeddings

Important correctness fix that made this possible:

- rasterization and moment extraction no longer silently drop all but the largest polygon in a `MultiPolygon`

## What Is Still Only Spec-Level

The layer-remapping part of robustness is defined here, but not fully executable yet.

Reason:

- the real layer-remapping problem depends on the future GDS-native pipeline
- that requires:
  - canonical layer ontology
  - importer/exporter mapping adapters
  - actual layer/datatype-bearing data

So the protocol answer today is:

- the requirement is accepted and versioning-aware
- the implementation is deferred to the C3 GDS work

## Versioning Boundary

The embedding identity should be treated as a function of:

- `version`
- `mode`
- `shape_resolution`
- `shared_bounds_padding`
- `shared_bounds_padding_fraction`
- `param_hash_dim`

These values now belong in cache keys and metadata. They should also appear in any future serialized embedding artifacts.

This is the current answer to `C2.6`.

## Practical Guidance

Use:

- `v1_legacy + geometry_only` for reproducing existing results
- `v2_hashed_params + geometry_only` when you want a stronger deterministic component embedding without changing coordinate frame semantics
- `v2_hashed_params + device_context` when you want arithmetic / composition / placement-aware behavior

## Likely Next Step

The next natural step after this protocol pass is to build nicer embedding-space utilities and benchmarks:

- projection plots
- cosine-similarity views
- arithmetic benchmark scorecards
- invariance benchmark dashboards
