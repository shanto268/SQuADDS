# SQuADDS ML Taskboard

Date: 2026-04-15
Repo: `/Users/shanto/LFL/SQuADDS_Refactor`
Primary context doc: [`BRANCH_CONTEXT_ml-embedding_2026-04-15.md`](BRANCH_CONTEXT_ml-embedding_2026-04-15.md)
Embedding protocol note: [`UNIVERSAL_EMBEDDING_PROTOCOL_2026-04-15.md`](UNIVERSAL_EMBEDDING_PROTOCOL_2026-04-15.md)

## Active Execution Tactic

- Active Codex working branch: `codex/universal-foundation-pass1`
- Active Codex working clone: `/Users/shanto/Documents/New project/SQuADDS_Refactor_work`
- Why: keep the original `ml-embedding` worktree untouched while foundation-hardening the universal stack
- First-pass scope:
  - refresh stale universal tests against the current hetero-graph API
  - add optional dependency packaging for the universal torch stack
  - expose a cleaner lazy import surface for universal package symbols
  - verify both:
    - plain `dev` environment behavior
    - explicit `universal`-extra behavior
- Second-pass scope:
  - define a reusable workflow API for standard quarter-wave rows
  - move core Tutorial 12 row-to-graph and prediction-readout logic into package code
  - add tests around the new workflow layer

## Purpose

This file is the execution board for the three ML tracks:

- Track A: MLP productionization
- Track B: symbolic regression / physics-discovery paper
- Track C: universal embeddings + graph ML

The goal is to make work asynchronous and resumable across future sessions, collaborators, and AI agents.

## Status Legend

- `todo` - not started
- `ready` - can be started now
- `blocked` - waiting on dependency or external input
- `in_progress` - actively being worked
- `review` - implemented / prepared, needs review
- `done` - complete
- `deferred` - intentionally not in the current phase

## Priority Legend

- `P0` - current top priority
- `P1` - next highest priority
- `P2` - important, but depends on earlier work
- `P3` - longer horizon

## Board Rules

- Keep task IDs stable once created.
- Prefer updating status and notes rather than rewriting tasks.
- Add links to notebooks, PRs, model repos, papers, or datasets as they appear.
- If an experiment changes the plan, update both this file and the context doc.

## Program Summary

### Near-term goals

- `P0` productionize collaborator-built MLP models once handed off
- `P0` push the symbolic-regression track toward a paper-ready result set
- `P1` stabilize the universal embedding + graph ML codebase so research iteration becomes cheaper

### Mid-term goals

- `P1` expand IDC symbolic-regression studies after dataset growth
- `P1` define a versioned, testable embedding API
- `P2` build the GDS-native embedding engine and row-to-GDS dataset pipeline

### Long-term goals

- `P2` graph-ML transfer/generalization paper studies
- `P3` browser-based GUI and richer platform UX
- `P3` encoder-decoder and inverse-design extensions

## Cross-Track Milestones

| Milestone ID | Priority | Status | Description | Depends on |
|---|---|---:|---|---|
| `M0` | `P0` | `ready` | Lock stable taskboard and roadmap structure | none |
| `M1` | `P0` | `blocked` | Receive collaborator MLP handoff package | collaborator delivery |
| `M2` | `P0` | `ready` | Quarter-wave symbolic workflow fully reproducible and paper-ready | none |
| `M3` | `P1` | `in_progress` | Universal pipeline stabilized enough for repeatable experiments | test/API cleanup |
| `M4` | `P2` | `todo` | GDS-native embedding subsystem working end-to-end | `M3` |
| `M5` | `P2` | `todo` | Graph-ML generalization benchmark suite defined and runnable | `M3`, `M4` partly |

## Track A: MLP Productionization

### Track goal

Turn collaborator-developed MLP models into productionized SQuADDS assets:

- package inference API
- Hugging Face model repo
- Hugging Face Space
- updated tutorials/docs

### Tasks

| ID | Priority | Status | Task | Depends on | Deliverable |
|---|---|---:|---|---|---|
| `A1` | `P0` | `blocked` | Receive finalized model artifacts, preprocessing contract, metrics, and intended scope from collaborators | collaborator handoff | local model package inputs |
| `A2` | `P0` | `todo` | Define stable model I/O schema and validation rules for the MLP inference wrapper | `A1` | schema doc + validation code |
| `A3` | `P0` | `todo` | Add package-level inference wrapper for the transmoncross MLP workflow | `A1`, `A2` | importable inference API |
| `A4` | `P0` | `todo` | Prepare Hugging Face model repo structure and metadata | `A1` | model repo template / card |
| `A5` | `P0` | `todo` | Prepare Hugging Face Space deployment flow and demo UX | `A3`, `A4` | Space app |
| `A6` | `P0` | `todo` | Update tutorials and docs for end users | `A3`, `A5` | updated tutorial/docs |
| `A7` | `P1` | `todo` | Run production QA: consistency, latency, smoke tests, edge-case validation | `A3`, `A5`, `A6` | QA checklist/results |

### Notes

- This track is likely the cleanest short-term win.
- The main blocker is external collaborator handoff.

## Track B: Symbolic Regression / Physics Discovery Paper

### Track goal

Write and publish a paper showing that the symbolic-regression framework:

- supports physics discovery
- yields interpretable equations
- can outperform or complement black-box MLPs on extrapolation-style tasks

### Main result pillars

- quarter-wave qubit-cavity symbolic equations
- independent first-principles derivation and comparison
- IDC capacitance equations
- larger IDC dataset rerun after colleague expands the dataset
- second IDC geometry-family study for reparameterization / transfer
- AI Feynman / dimensional-analysis / parameter-reduction augmentation

### Tasks

| ID | Priority | Status | Task | Depends on | Deliverable |
|---|---|---:|---|---|---|
| `B1` | `P0` | `ready` | Reproduce the quarter-wave symbolic-regression workflow cleanly from code + data | none | reproducible script/notebook |
| `B2` | `P0` | `ready` | Compare learned quarter-wave equations against independent first-principles derivation | `B1` | theory-comparison figures/tables |
| `B3` | `P0` | `ready` | Define extrapolation benchmark splits and baseline comparisons versus MLPs | `B1` | benchmark protocol |
| `B4` | `P0` | `ready` | Prepare symbolic-regression paper figure pipeline for quarter-wave results | `B1`, `B2`, `B3` | reusable figure scripts |
| `B5` | `P1` | `blocked` | Wait for colleague to expand the IDC dataset | external collaborator | larger IDC dataset |
| `B6` | `P1` | `todo` | Retrain IDC symbolic models on the expanded dataset and reassess equation quality | `B5` | updated IDC results |
| `B7` | `P1` | `todo` | Build second IDC shape-family dataset and test cross-shape equation transfer / reparameterization | dataset design effort | transfer-study dataset/results |
| `B8` | `P1` | `todo` | Integrate AI Feynman-inspired search / post-processing ideas into the workflow | `B1` | method extension |
| `B9` | `P1` | `todo` | Add automated dimensional analysis and parameter-set reduction layer | `B1` | analysis utilities + results |
| `B10` | `P1` | `todo` | Run equation-stability study across resamples / random splits | `B1` | robustness results |
| `B11` | `P1` | `todo` | Run sample-efficiency / low-data study, especially for IDC | `B1`, `B6` if available | data-efficiency results |
| `B12` | `P1` | `todo` | Draft manuscript outline and claim table | `B2`, `B3` partly | paper skeleton |
| `B13` | `P2` | `todo` | Full manuscript drafting, section-by-section | `B12` | draft manuscript |

### Notes

- `B5` is a real dependency and should not be forgotten.
- Quarter-wave should likely anchor the paper even if IDC remains a smaller secondary result.
- Cross-shape IDC transfer could become one of the most interesting paper sections if it works.

## Track C: Universal Embeddings + Graph ML

### Track goal

Build a robust, versioned, eventually GDS-native embedding and graph-ML ecosystem that supports:

- meaningful universal component embeddings
- embedding arithmetic and latent-space analysis
- downstream graph-ML for topology-aware prediction and transfer learning
- future encoder-decoder / inverse-design directions

### Current reality

- useful proof of concept exists
- tests and packaging lag implementation
- current embedding source is not yet the final desired GDS-native path

### Workstream C1: Foundation hardening

| ID | Priority | Status | Task | Depends on | Deliverable |
|---|---|---:|---|---|---|
| `C1.1` | `P1` | `ready` | Decide and document that the universal path is a first-class internal research subsystem, not just notebook code | none | clarified scope |
| `C1.2` | `P1` | `done` | Update stale universal tests to match the current heterogeneous API | none | passing/aligned tests |
| `C1.3` | `P1` | `done` | Add dependency handling / extras for `torch` and `torch_geometric` | none | packaging update |
| `C1.4` | `P1` | `review` | Define stable internal interfaces for embeddings, graph building, readout, and dataset rows | none | interface spec |
| `C1.5` | `P1` | `review` | Move core Tutorial 12 orchestration into reusable package functions | `C1.4` | utility module(s) |
| `C1.6` | `P1` | `review` | Add smoke-test dataset build and end-to-end training sanity check | `C1.2`, `C1.3`, `C1.5` | reproducible smoke pipeline |

### C1 Recent Updates

- `2026-04-15 foundation pass` completed:
  - added `universal` extra in `pyproject.toml` and folded it into `all`
  - exposed lazy universal imports in `squadds/ml/__init__.py` and `squadds/ml/universal/__init__.py`
  - rewrote universal tests to match the current `HeteroData` / `UniversalGNN` / `UniversalTrainer` APIs
  - made universal tests skip cleanly when `torch`/`torch_geometric` are not installed
- `2026-04-15 workflow extraction pass` completed:
  - added `squadds/ml/universal/workflows.py`
  - defined reusable interfaces:
    - `UniversalRowSchema`
    - `TargetScales`
    - `build_layout_from_row()`
    - `attach_targets_from_row()`
    - `build_graph_from_row()`
    - `build_graph_dataset()`
    - `read_node_predictions()`
    - `read_edge_predictions()`
    - `read_prediction_summary()`
    - `make_standard_qubit_cavity_netlist()`
  - exported the workflow layer through both `squadds.ml.universal` and `squadds.ml`
  - folded workflow API coverage into the tracked `tests/ml/universal/test_graph.py`
- `2026-04-15 smoke-pipeline pass` completed:
  - added `UniversalModelDims`
  - added `infer_model_dims()`
  - added `build_model_from_graph()`
  - added a real row -> graph -> model -> trainer smoke test in `tests/ml/universal/test_trainer.py`
  - validated the standard quarter-wave workflow on a tiny multi-row dataset rather than only on synthetic tensors
- Verification:
  - `uv run --extra dev pytest tests/ml/universal/... -q` -> `4 skipped`
  - `uv run --extra dev --extra universal pytest tests/ml/universal/... -q` -> `41 passed`
- Follow-up verification after workflow extraction:
  - `uv run --extra dev --extra universal pytest tests/ml/universal/test_graph.py tests/ml/universal/test_model.py tests/ml/universal/test_trainer.py tests/ml/universal/test_features.py -q` -> `48 passed`
  - workflow lazy-import smoke checks from `squadds.ml` passed
- Follow-up verification after smoke pipeline:
  - `uv run --extra dev --extra universal pytest tests/ml/universal/test_trainer.py tests/ml/universal/test_graph.py tests/ml/universal/test_model.py tests/ml/universal/test_features.py -q` -> `49 passed`
- Important model note:
  - not every parameter in the final hetero layer receives gradients on tiny synthetic graphs because the last `component -> virtual` branch does not directly feed current readouts; tests should validate active readout paths, not require universal non-`None` gradients for every parameter

### Workstream C2: Embedding-science hardening

| ID | Priority | Status | Task | Depends on | Deliverable |
|---|---|---:|---|---|---|
| `C2.1` | `P1` | `review` | Formalize shared-frame embedding arithmetic as real code, not notebook-only logic | `C1.5` ideally | reusable arithmetic utilities |
| `C2.2` | `P1` | `review` | Define embedding invariance / equivariance desiderata | `C1.4` | design note / test matrix |
| `C2.3` | `P1` | `review` | Build robustness tests for rotation, mirroring, padding, layer remapping, and exporter noise | `C2.2` | robustness benchmark suite |
| `C2.4` | `P1` | `review` | Improve parameter encoding beyond `param_sum` | `C1.4` | upgraded embedding protocol candidate |
| `C2.5` | `P1` | `review` | Decide geometry-only vs context-aware vs device-level embedding modes | `C2.2` | embedding mode taxonomy |
| `C2.6` | `P1` | `review` | Introduce explicit embedding versioning API | `C1.4` | versioned embedding interface |

### C2 Recent Updates

- `2026-04-15 shared-frame arithmetic pass` completed:
  - added `squadds/ml/universal/features/arithmetic.py`
  - added:
    - `compute_shared_frame_shape_embedding()`
    - `compute_shared_frame_embedding()`
    - `embedding_cosine_similarity()`
  - promoted shared-bounds rasterization into public rasterizer utilities:
    - `compute_shared_bounds()`
    - `rasterize_in_bounds()`
  - fixed a correctness issue where `MultiPolygon` rasterization dropped all but the largest polygon, which would have broken compositional arithmetic and any future multipart embeddings
  - exported the arithmetic helpers through `squadds.ml.universal` and `squadds.ml`
- Verification:
  - `uv run --extra dev --extra universal pytest tests/ml/universal/test_features.py tests/ml/universal/test_graph.py tests/ml/universal/test_model.py tests/ml/universal/test_trainer.py -q` -> `54 passed`
  - shared-frame helper import smoke checks from `squadds.ml` passed
- `2026-04-15 embedding protocol sweep` completed:
  - added `squadds/ml/universal/features/protocol.py`
  - added version axis:
    - `v1_legacy`
    - `v2_hashed_params`
  - added mode axis:
    - `geometry_only`
    - `component_context`
    - `device_context`
  - added:
    - `EmbeddingVersion`
    - `EmbeddingMode`
    - `EmbeddingConfig`
    - `DEFAULT_EMBEDDING_CONFIG`
    - `encode_params()`
    - `compute_component_embedding()`
    - `embedding_dim()`
    - `param_feature_dim()`
  - threaded `embedding_config` support into `UniversalGraphBuilder`
  - added protocol note:
    - `UNIVERSAL_EMBEDDING_PROTOCOL_2026-04-15.md`
  - expanded robustness coverage for:
    - translation
    - rotation
    - mirroring
    - padding sensitivity
    - exporter-noise-style polygon variants
    - multipart geometry handling
- Follow-up verification after protocol sweep:
  - `uv run --extra dev --extra universal pytest tests/ml/universal/test_features.py tests/ml/universal/test_graph.py tests/ml/universal/test_model.py tests/ml/universal/test_trainer.py -q` -> `63 passed`
  - top-level import smoke checks for `EmbeddingConfig`, `EmbeddingMode`, `EmbeddingVersion`, `compute_component_embedding()`, and related helpers from `squadds.ml` passed

### Workstream C3: GDS-native embedding engine

| ID | Priority | Status | Task | Depends on | Deliverable |
|---|---|---:|---|---|---|
| `C3.1` | `P2` | `todo` | Define canonical internal layer ontology independent of lab-specific layer numbers | `C2.6` recommended | canonical layer spec |
| `C3.2` | `P2` | `todo` | Prototype row -> GDS generation engine from SQuADDS design options using Qiskit Metal / Quantum Metal | `C1.4` | generator prototype |
| `C3.3` | `P2` | `todo` | Emit per-row `qubit.gds`, `claw.gds`, `cpw.gds`, `feedline.gds`, and `device.gds` | `C3.2` | five-file output flow |
| `C3.4` | `P2` | `todo` | Add user-defined ground-plane padding in the GDS generation engine | `C3.2` | padding controls |
| `C3.5` | `P2` | `todo` | Encode layer-stack information into GDS layers / datatypes | `C3.1`, `C3.2` | stack-aware GDS output |
| `C3.6` | `P2` | `todo` | Create custom QComponents where needed for automated GDS generation | `C3.2` | custom components |
| `C3.7` | `P2` | `todo` | Implement layer-convention normalization adapters for cross-group consistency | `C3.1`, `C3.5` | normalization layer |
| `C3.8` | `P2` | `todo` | Prototype GDS-native embedding computation on generated files | `C3.3`, `C3.5` | GDS-native embedding prototype |

### Workstream C4: Embedding-space science / paper studies

| ID | Priority | Status | Task | Depends on | Deliverable |
|---|---|---:|---|---|---|
| `C4.1` | `P1` | `review` | Build polished projection/visualization utilities for cosine similarity, nearest neighbors, and difference vectors | `C1.5` helpful | reusable visualization module |
| `C4.2` | `P1` | `todo` | Define embedding clustering benchmark across component families | `C1.6` | benchmark + figures |
| `C4.3` | `P1` | `todo` | Define embedding arithmetic benchmark beyond single anecdotal examples | `C2.1` | benchmark + scorecard |
| `C4.4` | `P2` | `todo` | Define cross-convention normalization benchmark for GDS-native embeddings | `C3.7`, `C3.8` | robustness study |
| `C4.5` | `P2` | `todo` | Draft universal-embedding paper outline | `C4.2`, `C4.3` | paper skeleton |

### C4 Recent Updates

- `2026-04-15 visualization utility pass` completed:
  - added `squadds/ml/universal/visualization.py`
  - added:
    - `compute_embedding_projection()`
    - `compute_embedding_projections()`
    - `compute_cosine_similarity_matrix()`
    - `compute_label_centroids()`
    - `find_nearest_neighbors()`
    - `rank_difference_vector()`
    - `plot_embedding_projection()`
    - `plot_projection_grid()`
    - `plot_similarity_bars()`
  - exported the visualization helpers through `squadds.ml.universal` and `squadds.ml`
  - added tests for projection, cosine similarity, nearest neighbors, difference-vector ranking, and basic plotting return values in `tests/ml/universal/test_features.py`
- Verification:
  - `uv run --extra dev --extra universal pytest tests/ml/universal/test_features.py tests/ml/universal/test_graph.py tests/ml/universal/test_model.py tests/ml/universal/test_trainer.py -q` -> `70 passed`

### Workstream C5: Graph-ML science / paper studies

| ID | Priority | Status | Task | Depends on | Deliverable |
|---|---|---:|---|---|---|
| `C5.1` | `P2` | `todo` | Define topology holdout benchmark: train on one family, test on larger/changed family | `C1.6` | benchmark protocol |
| `C5.2` | `P2` | `todo` | Define transfer-learning benchmark: pretrain on one family, finetune on another | `C1.6` | protocol + baseline |
| `C5.3` | `P2` | `todo` | Build synthetic compositional dataset for fast graph-generalization debugging | `C1.4` | synthetic dataset |
| `C5.4` | `P2` | `todo` | Run ablation study over hub node, edge features, and embedding variants | `C1.6`, `C2.*` partly | ablation results |
| `C5.5` | `P2` | `todo` | Study node/edge readout faithfulness and label semantics | `C1.6` | faithfulness analysis |
| `C5.6` | `P3` | `todo` | Draft graph-ML paper outline once generalization results are credible | `C5.1`, `C5.2`, `C5.4` | paper skeleton |

### Workstream C6: Future directions

| ID | Priority | Status | Task | Depends on | Deliverable |
|---|---|---:|---|---|---|
| `C6.1` | `P3` | `deferred` | Explore encoder-decoder models on top of the embedding space | stronger embeddings first | concept note / prototype |
| `C6.2` | `P3` | `deferred` | Explore inverse-design / generative workflows using embeddings and graph representations | stronger forward models first | experiment plan |
| `C6.3` | `P3` | `deferred` | Evaluate richer downstream models beyond current hetero GATv2 baseline | `C1.6` | model-comparison study |

### Workstream C7: Platform and UX

| ID | Priority | Status | Task | Depends on | Deliverable |
|---|---|---:|---|---|---|
| `C7.1` | `P3` | `todo` | Design clean Python API for graph construction, embedding computation, and dataset authoring | `C1.4` | API draft/spec |
| `C7.2` | `P3` | `todo` | Define browser-based GUI scope and object model | `C7.1` | GUI scope doc |
| `C7.3` | `P3` | `todo` | Prototype lightweight browser GUI with undo/redo, copy/paste, node/edge assignment, and GDS attachment | `C7.2` | GUI prototype |

## Immediate Queue

These are the tasks I would put at the top of the queue right now.

| Order | ID | Why now |
|---|---|---|
| `1` | `A1` | external blocker for the most production-ready track |
| `2` | `B1` | strongest near-term paper foundation |
| `3` | `B2` | theory comparison sharpens the paper story |
| `4` | `B3` | extrapolation benchmark is central to the paper claim |
| `5` | `C4.1` | the protocol layer is in place, so polished latent-space utilities are the next obvious user-facing win |
| `6` | `C4.3` | arithmetic is now real package code, so it should become a benchmark instead of a single anecdote |
| `7` | `C5.3` | a synthetic compositional dataset would accelerate graph-generalization debugging |
| `8` | `C5.1` | topology holdout benchmarks are the next serious graph-ML science step |
| `9` | `C3.1` | the next foundational unknown is the canonical layer ontology for GDS-native work |
| `10` | `B5` | keep visible as a real dependency, even if blocked |

## External Dependencies and Waiting Items

| ID | Status | Dependency | Notes |
|---|---|---|---|
| `A1` | `blocked` | collaborator MLP handoff | model artifacts + preprocessing + metrics |
| `B5` | `blocked` | colleague IDC dataset expansion | needed before IDC retraining |

## Suggested Owner Split

This is only a suggested split, not a fixed assignment.

| Area | Suggested owner |
|---|---|
| MLP handoff / deployment integration | Shanto + collaborators + future AI support |
| Symbolic paper science | Shanto + theory collaborators + future AI support |
| Universal stabilization | Codex / future AI support + Shanto |
| GDS engine / QComponent generation | Shanto + future AI support |
| GUI / API product layer | future engineering pass after APIs stabilize |

## Session Template For Future Agents

When resuming work, start by answering:

1. Which track is active right now: `A`, `B`, or `C`?
2. Which task IDs are the actual focus this session?
3. What changed since this taskboard was last updated?
4. Should the context doc also be updated after the work is done?
