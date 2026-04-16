# ml-embedding Branch Context

Date: 2026-04-15
Repo: `/Users/shanto/LFL/SQuADDS_Refactor`
Branch: `ml-embedding`
HEAD: `505193f` (`feat: multi-projection t-SNE, embedding arithmetic star plot, Case 2 ground truth`)
Primary comparison baseline: `upstream/master` at `65c95e0`

## Purpose

This file is an internal handoff for future sessions and future AI agents.
It captures:

- what this repo is
- what this branch adds on top of upstream SQuADDS
- which parts of the ML work are package code vs notebook-driven research
- what is verified, what is stale, and what is still only a proposal

Read this before changing the ML code on this branch.

## Active Execution Tactic

Status: `Updated from active Codex work on 2026-04-15`

Because the original `ml-embedding` worktree should stay undisturbed while foundation work is underway, the current implementation pass is happening in:

- working clone: `/Users/shanto/Documents/New project/SQuADDS_Refactor_work`
- working branch: `codex/universal-foundation-pass1`

This branch is being used for first-pass universal-stack hardening:

- refresh stale universal tests against current code
- add explicit optional dependency handling for the universal torch stack
- expose a cleaner lazy import surface for universal package symbols
- record verified behavior so future sessions do not need to rediscover these basics

## Status Legend

- `Implemented / verified by code read`: I inspected the current source and this matches HEAD.
- `Implemented / notebook-driven`: this exists in code or notebook cells, but I did not rerun the full notebook end-to-end in this pass.
- `Inherited note / proposal`: useful idea from the previous AI note, but not necessarily implemented in the current code.
- `Known stale / broken`: current code and tests or notes no longer agree.

## Branch Snapshot

- `ml-embedding` is `45` commits ahead of `upstream/master`.
- It is `47` commits ahead of `origin/master`.
- Use `upstream/master` as the real baseline.
- Reason: `origin/master` is the fork and lags canonical upstream by a simulation API commit plus different contributor-bot noise.
- Diff vs `upstream/master` is large: `67` changed files, about `25,571` insertions and `27` deletions.
- The branch timeline is concentrated in two bursts:
  - `2026-01-29`: explainable ML and inverse-design workflow
  - `2026-03-09` and `2026-04-09/10`: graph and universal GNN work

### Milestone Timeline

- `ad87d7f` `2026-01-29` - add explainable ML module (EBM + PySR)
- `a590bf3` `2026-01-29` - add unit and integration tests for ML module
- `cb92350` `2026-03-09` - add graph neural network forward model sub-package
- `038dd06` `2026-03-09` - rework graph module for Keras 3
- `90aa387` `2026-03-09` - mixed precision and Tutorial 11 rewrite
- `eac1d6c` `2026-04-09` - universal pipeline Milestone 1: Shapely geometry generators
- `7c64549` `2026-04-09` - universal pipeline Milestone 2: feature extractors
- `9c7335f` `2026-04-09` - universal graph builder
- `b176c68` `2026-04-09` - GATv2 model and masked multi-task loss
- `65abc74` `2026-04-09` - universal trainer loop
- `9700403` `2026-04-09` - redesign universal GNN with static embeddings and composable layouts
- `9cfcd47` `2026-04-09` - switch to `HeteroData` architecture with typed predictions
- `505193f` `2026-04-10` - Tutorial 12 work: multi-projection t-SNE, embedding arithmetic star plot, Case 2 ground truth

## Repo Baseline

SQuADDS itself is a Python package for superconducting quantum device design and simulation. The non-ML core is still important context:

- [`squadds/core/db.py`](squadds/core/db.py) manages dataset discovery and loading from Hugging Face.
- [`squadds/simulations`](squadds/simulations) contains Ansys/Qiskit Metal simulation workflows.
- [`README.md`](README.md) frames the project as a database + workflow system, not originally as an ML-first repo.

The ML work in this branch sits on top of that baseline rather than replacing it.

## What This Branch Actually Adds

This branch adds three distinct ML layers:

1. Explainable ML / symbolic regression for inverse design
2. A Keras/Spektral graph-forward model
3. A newer "universal" geometry-first heterogeneous GNN pipeline

These three layers coexist. They are not yet fully unified.

## Program Vision and Product Direction

Status: `Updated from user guidance on 2026-04-15`

This section captures the intended future of the repo and the ML program, not just the current branch state.

### Three-track strategy

There are really three ML tracks now:

1. basic MLP models
2. symbolic regression / interpretable physics-discovery models
3. universal embedding + graph ML

They have different maturity levels, deliverables, and timelines.

### Maturity summary

- `MLP`:
  - model development is being handled primarily by collaborators
  - once ready, this track should be productionized
- `symbolic regression`:
  - closest to a paperable scientific result
  - also close to being a robust research workflow
- `universal embedding + graph ML`:
  - still research / dev
  - current code is a useful proof of concept, not the final envisioned system

### High-level product philosophy

The end state should be:

- production-ready ML workflows where appropriate
- publishable scientific outputs where appropriate
- a clean separation between:
  - research prototypes
  - stable package APIs
  - deployed models / demos

The symbolic and universal tracks should not be rushed into "production" before the scientific and technical foundations are solid.

## Track A: Basic MLP Models

Status: `User-directed near-term productization goal`

### Ownership and role

- this model family is being developed primarily by collaborators
- this branch/repo should be ready to receive and productionize those models once they settle

### Short-term goal

This is now an explicit short-term program goal:

- productionize the collaborator-developed MLP models
- current example: Hugging Face models for `TransmonCross`

### Expected productionization outputs

- packaged trained model artifacts
- Hugging Face model repos
- Hugging Face Spaces deployment
- updated tutorials and user-facing docs
- straightforward inference API for users

### Recommended packaging/deployment checklist

- stable preprocessing contract
- frozen feature schema and model versioning
- model card with training data scope and caveats
- inference wrapper inside the package
- tutorial notebook that reproduces the intended usage path
- Space that demonstrates prediction and explains allowed input ranges

### Relationship to this branch

The MLP track is not the main focus of `ml-embedding`, but this doc should remember that MLP productization is a real near-term deliverable for the broader SQuADDS ML roadmap.

## Track B: Symbolic Regression and Physics Discovery

Status: `High-priority paper track`

### Scientific objective

The symbolic regression work is intended to become a paper.

The paper should motivate:

- why symbolic / interpretable models matter for superconducting device design
- how they support physics discovery
- how they can assist extrapolation better than standard MLPs, which are often best at interpolation

### Core paper claims currently envisioned

1. symbolic regression can recover physically meaningful equations from SQuADDS-style design/simulation data
2. the learned equations can be compared against independently derived first-principles theory
3. this framework is promising for extrapolation, scientific interpretability, and parameter reduction

### Primary results to include

#### Quarter-wave qubit-cavity system

Main result set:

- equations mapping Hamiltonian parameters onto design-space parameters for the quarter-wave qubit-cavity system
- comparison to independently derived theoretical equations from first principles

This is likely the strongest current symbolic result and should probably anchor the paper.

#### Interdigitated capacitor (IDC) equations

A second result set is desired for IDC capacitance equations:

- use the symbolic framework on the IDC capacitance dataset
- determine whether the recovered equations are physically meaningful
- compare them to existing understanding / analytical expectations

Current reality:

- the IDC dataset is presently too small for ideal ML performance
- one colleague plans to expand that dataset
- after dataset expansion, retraining and reassessment should happen

This needs to stay in the roadmap because it is easy to forget and could materially improve the paper.

#### Cross-shape / reparameterization study

Another planned study:

- build a second IDC dataset with a different geometry family / shape
- test whether equations derived on one shape family can be:
  - interpreted
  - reparameterized
  - transferred
  - and still hold, approximately or structurally, on the new shape family

This could become a major paper contribution if it works because it would move the story from "fitted equations" to "transferable scientific structure".

### Methods the paper should likely incorporate

In addition to plain EBM + PySR, the user explicitly wants to incorporate ideas from:

- AI Feynman
- automated dimensional analysis
- parameter-set reduction
- variable / invariant discovery

This should be treated as a real roadmap item, not a vague future idea.

### Suggested symbolic-regression paper experiments

These are experiments I think would materially strengthen the paper.

#### 1. Extrapolation benchmark versus MLP baselines

Design a benchmark where train and test are separated by geometry regime, not just random rows:

- hold out large or small parameter regimes
- hold out coupled parameter corners
- compare:
  - symbolic regression
  - baseline MLP
  - maybe a tree ensemble baseline

Metrics:

- in-range performance
- out-of-range performance
- physical plausibility of predictions

This directly supports the extrapolation claim.

#### 2. Equation stability under resampling

Run repeated train/test splits or bootstrap resampling to test:

- whether the same terms recur
- whether the discovered symbolic structure is stable
- whether coefficients are robust

This helps distinguish real scientific signal from search noise.

#### 3. Theory-constrained regression study

Use known dimensionless groups, invariants, or first-principles functional forms as:

- input transformations
- search-space constraints
- post-hoc simplification priors

Compare unconstrained and constrained symbolic discovery.

#### 4. Parameter reduction study

Show that the framework identifies a reduced parameter set or reduced coordinate system:

- via EBM feature importance
- via symbolic expression sparsity
- via dimensional-analysis groups

This can be a major contribution in its own right.

#### 5. Noise / sample-efficiency study

Especially important for the small IDC dataset:

- subsample training data
- optionally add synthetic noise
- compare how symbolic and neural methods degrade

This would strengthen the case for symbolic methods when data is limited.

#### 6. Cross-family transfer for IDC shapes

For the second IDC geometry family:

- fit on family A
- test on family B
- reparameterize if needed
- measure whether equation structure transfers

This may become one of the most interesting results in the paper.

### Symbolic paper deliverables

- robust datasets for quarter-wave and IDC cases
- reproducible training/evaluation code
- theory comparison notebook or script
- symbolic equation tables
- extrapolation figures
- parameter-reduction / dimensional-analysis results
- manuscript draft and figure pipeline

## Track C: Universal Embedding + Graph ML

Status: `Research / dev, high upside, not yet production-ready`

### Honest current assessment

The current universal embedding + graph ML work is useful because it gives a working pipeline to test ideas.

But it is **not yet the final envisioned design**.

The current code should be treated as:

- a proof of concept
- an experimentation scaffold
- a place to learn what the real abstraction and API should be

### Long-term scientific and product vision

This track has two intertwined goals:

1. create a universal embedding space for superconducting device building blocks
2. use that space to support transfer learning, graph ML, and eventually encoder-decoder / generative workflows

### Current temporary compromise

Right now the embedding pipeline is based on the current rendered / geometry-derived component representation rather than a true GDS-native embedding source.

That is acceptable for now because it lets the team test:

- pipeline structure
- graph construction
- latent-space visualizations
- embedding arithmetic ideas
- transfer-learning hypotheses

### Short-to-medium term product direction for this track

Once the system is technically robust enough, the goal is to move away from the current rendering-first approach and toward a GDS-native embedding pipeline, likely as its own subpackage.

### Foundation pass started on 2026-04-15

Status: `Implemented in active working branch`

The first foundation-hardening pass has already started and established a better baseline for future work.

Implemented in `codex/universal-foundation-pass1`:

- added `universal` optional dependencies in `pyproject.toml` for `torch` and `torch-geometric`
- included `universal` in the package `all` extra
- added lazy universal symbol exports through:
  - `squadds/ml/__init__.py`
  - `squadds/ml/universal/__init__.py`
- rewrote stale universal tests to match the current heterogeneous `HeteroData` architecture, `UniversalGNN` constructor, and `UniversalTrainer`
- made the universal test files skip cleanly when optional torch dependencies are not installed

Verified behavior from that branch:

- `uv run --extra dev pytest tests/ml/universal/... -q` -> `4 skipped`
- `uv run --extra dev --extra universal pytest tests/ml/universal/... -q` -> `41 passed`

Important subtlety discovered during this pass:

- on small synthetic graphs, not every parameter in the final hetero layer receives gradients
- specifically, the last `component -> virtual` path can be inactive with the current readout design because predictions are read from component nodes and physical edges, not directly from the final virtual state
- future tests should therefore verify gradient flow through active readout paths rather than assume every trainable parameter always gets a gradient on toy data

### Workflow extraction pass started on 2026-04-15

Status: `Implemented in active working branch`

The next pass pulled the most duplicated Tutorial 12 orchestration into reusable package code.

Implemented in `codex/universal-foundation-pass1`:

- added [`squadds/ml/universal/workflows.py`](squadds/ml/universal/workflows.py)
- defined a reusable quarter-wave row/schema layer:
  - `UniversalRowSchema`
  - `TargetScales`
  - `STANDARD_SQUADDS_ROW_SCHEMA`
  - `STANDARD_TARGET_SCALES`
- moved row-to-layout and row-to-graph logic into:
  - `extract_layout_params()`
  - `build_layout_from_row()`
  - `attach_targets_from_row()`
  - `build_graph_from_row()`
  - `build_graph_dataset()`
- moved inference-readout logic into:
  - `read_node_predictions()`
  - `read_edge_predictions()`
  - `read_prediction_summary()`
- added `make_standard_qubit_cavity_netlist()` so the standard training topology is no longer notebook-only
- exported these helpers through both:
  - `squadds.ml.universal`
  - `squadds.ml`

Why this matters:

- future sessions no longer need to reconstruct the standard row schema from notebook cells
- training-graph generation for Tutorial 12 can now be reused from package code
- structured prediction readout is now testable and reusable for demos, papers, and future GUIs

Verified behavior from that branch:

- `uv run --extra dev --extra universal pytest tests/ml/universal/test_graph.py tests/ml/universal/test_model.py tests/ml/universal/test_trainer.py tests/ml/universal/test_features.py -q` -> `48 passed`
- import smoke checks for the new workflow helpers from `squadds.ml` passed

Important boundary:

- this workflow layer currently codifies the **standard quarter-wave row contract** used in Tutorial 12
- it is not yet the full future API for arbitrary GDS-native rows or arbitrary graph datasets
- that broader generalization remains future work

### Smoke-pipeline pass started on 2026-04-15

Status: `Implemented in active working branch`

The next pass added an actual end-to-end sanity path over the real universal workflow.

Implemented in `codex/universal-foundation-pass1`:

- added `UniversalModelDims` in [`squadds/ml/universal/workflows.py`](squadds/ml/universal/workflows.py)
- added `infer_model_dims()` so a `UniversalGNN` can be instantiated from a sample graph without manually reading tensor sizes
- added `build_model_from_graph()` so the standard smoke path now looks like:
  - rows
  - `build_graph_dataset()`
  - `build_model_from_graph(sample_graph)`
  - `UniversalTrainer`
- added a real smoke test in [`tests/ml/universal/test_trainer.py`](tests/ml/universal/test_trainer.py) that:
  - builds a tiny multi-row quarter-wave dataset
  - constructs real heterogeneous graphs via the workflow layer
  - instantiates the model from the graph dimensions
  - runs a short training loop
  - verifies checkpoint creation and prediction tensor shapes

Why this matters:

- the trainer is no longer only covered by synthetic random graphs
- future changes to layout-building, graph-building, or target attachment now have a better chance of failing in tests instead of much later in a notebook
- Tutorial 12 no longer needs to own the "how do I infer model input dimensions?" logic

Verified behavior from that branch:

- `uv run --extra dev --extra universal pytest tests/ml/universal/test_trainer.py tests/ml/universal/test_graph.py tests/ml/universal/test_model.py tests/ml/universal/test_features.py -q` -> `49 passed`

### Embedding protocol sweep started on 2026-04-15

Status: `Implemented in active working branch`

The next pass completed the core C2 embedding-science hardening items in a first reviewable form.

Implemented in `codex/universal-foundation-pass1`:

- added [`squadds/ml/universal/features/protocol.py`](squadds/ml/universal/features/protocol.py)
- introduced a version axis:
  - `v1_legacy`
  - `v2_hashed_params`
- introduced a mode axis:
  - `geometry_only`
  - `component_context`
  - `device_context`
- added explicit protocol objects and helpers:
  - `EmbeddingVersion`
  - `EmbeddingMode`
  - `EmbeddingConfig`
  - `DEFAULT_EMBEDDING_CONFIG`
  - `encode_params()`
  - `compute_component_embedding()`
  - `embedding_dim()`
  - `param_feature_dim()`
- threaded `embedding_config` into [`squadds/ml/universal/graph/builder.py`](squadds/ml/universal/graph/builder.py)
- added the dedicated protocol note:
  - [`UNIVERSAL_EMBEDDING_PROTOCOL_2026-04-15.md`](UNIVERSAL_EMBEDDING_PROTOCOL_2026-04-15.md)

Also completed the shared-frame arithmetic extraction:

- added [`squadds/ml/universal/features/arithmetic.py`](squadds/ml/universal/features/arithmetic.py)
- added:
  - `compute_shared_frame_shape_embedding()`
  - `compute_shared_frame_embedding()`
  - `embedding_cosine_similarity()`
- promoted shared-bounds rasterization into public utilities in [`rasterizer.py`](squadds/ml/universal/features/rasterizer.py):
  - `compute_shared_bounds()`
  - `rasterize_in_bounds()`

Important correctness fix discovered and applied during this sweep:

- both rasterization and geometric moments previously collapsed `MultiPolygon` geometries to their largest piece
- that would have broken compositional arithmetic and multipart/exporter-noise robustness
- the code now uses the full geometry instead

What this sweep now covers from the taskboard:

- `C2.1` shared-frame arithmetic -> implemented
- `C2.2` invariance / equivariance desiderata -> implemented as protocol note + tests
- `C2.3` robustness tests -> implemented for geometry-side transformations and exporter-noise-style polygon variants
- `C2.4` improved parameter encoding -> implemented as `v2_hashed_params`
- `C2.5` embedding mode taxonomy -> implemented in code + protocol note
- `C2.6` explicit embedding versioning API -> implemented in code + cache-aware builder metadata

Important nuance:

- the layer-remapping part of robustness is now captured as a protocol requirement, but a full executable test path still depends on the later GDS/layer-ontology work
- this is documented explicitly in the protocol note, rather than being left implicit

Verified behavior from that branch:

- `uv run --extra dev --extra universal pytest tests/ml/universal/test_features.py tests/ml/universal/test_graph.py tests/ml/universal/test_model.py tests/ml/universal/test_trainer.py -q` -> `63 passed`
- top-level imports for the new embedding protocol helpers from `squadds.ml` passed

### Visualization utility pass started on 2026-04-15

Status: `Implemented in active working branch`

The next pass extracted the core embedding-space analysis helpers from Tutorial 12 into reusable package code.

Implemented in `codex/universal-foundation-pass1`:

- added [`squadds/ml/universal/visualization.py`](squadds/ml/universal/visualization.py)
- added projection helpers:
  - `compute_embedding_projection()`
  - `compute_embedding_projections()`
- added similarity / neighborhood helpers:
  - `compute_cosine_similarity_matrix()`
  - `compute_label_centroids()`
  - `find_nearest_neighbors()`
  - `rank_difference_vector()`
- added lightweight plotting helpers:
  - `plot_embedding_projection()`
  - `plot_projection_grid()`
  - `plot_similarity_bars()`
- exported the visualization helpers through:
  - `squadds.ml.universal`
  - `squadds.ml`

Why this matters:

- Tutorial 12 no longer needs to own the projection and similarity-analysis logic
- future demos and papers can reuse one package-level interface for cosine similarity, neighbor search, and arithmetic ranking
- this also creates a better bridge from the research notebook to a future GUI/API layer

Verified behavior from that branch:

- `uv run --extra dev --extra universal pytest tests/ml/universal/test_features.py tests/ml/universal/test_graph.py tests/ml/universal/test_model.py tests/ml/universal/test_trainer.py -q` -> `70 passed`

### Benchmark + Tutorial 12 pass started on 2026-04-15

Status: `Implemented in active working branch`

The next pass turned the new embedding-space tooling into actual study primitives and refreshed Tutorial 12 so it demonstrates maintained package APIs instead of notebook-only analysis code.

Implemented in `codex/universal-foundation-pass1`:

- added [`squadds/ml/universal/benchmarks.py`](squadds/ml/universal/benchmarks.py)
- added component-family clustering helpers:
  - `build_component_embedding_collection()`
  - `benchmark_component_family_clustering()`
- added arithmetic-benchmark helpers:
  - `evaluate_standard_arithmetic_case()`
  - `benchmark_standard_embedding_arithmetic()`
- exported the benchmark layer through:
  - `squadds.ml.universal`
  - `squadds.ml`
- refreshed [`tutorials/Tutorial-12_Universal_GNN.ipynb`](tutorials/Tutorial-12_Universal_GNN.ipynb) so it now shows:
  - versioned embedding configuration via `EmbeddingConfig`
  - package-level projection / cosine / nearest-neighbor tooling
  - C4.2 clustering scorecards across component families
  - C4.3 arithmetic scorecards across many rows, plus single-case inspection
  - workflow helpers for graph construction, model bootstrapping, and structured prediction readout

Why this matters:

- C4.2 and C4.3 are no longer notebook ideas; they are reusable benchmark APIs
- Tutorial 12 is back to being a real productized research demo rather than a stash of custom cells
- the universal embedding paper can now point to concrete benchmark definitions instead of one-off visual anecdotes

Verified behavior from that branch:

- `uv run --extra dev --extra universal pytest tests/ml/universal/test_features.py tests/ml/universal/test_graph.py tests/ml/universal/test_model.py tests/ml/universal/test_trainer.py -q` -> `74 passed`
- `uv run --extra dev --extra universal python - <<'PY' ... PY` smoke-tested:
  - `build_component_embedding_collection()`
  - `benchmark_component_family_clustering()`
  - `evaluate_standard_arithmetic_case()`
  - `benchmark_standard_embedding_arithmetic()`

### GDS-native embedding target

The desired end state is:

- embeddings generated from GDS files, not the current temporary representation
- one GDS artifact family per dataset row:
  - `qubit.gds`
  - `claw.gds`
  - `cpw.gds`
  - `feedline.gds`
  - `device.gds`

### Data-generation plan tied to SQuADDS

The longer-term goal is end-to-end training connected directly to the SQuADDS dataset:

- start from `design_options` and sweep parameters
- render component and device GDS files for each row
- compute embeddings from those GDS files
- use them for downstream training

Likely implementation path:

- build or extend a rendering engine around Qiskit Metal / Quantum Metal
- generate the five GDS files per row
- create custom QComponents where necessary

### GDS rendering requirements

The user explicitly wants the GDS rendering pipeline to support:

- user-defined ground-plane padding around the design
- layer-stack information encoded directly into the GDS
- not just metal layers, but also other materials such as:
  - substrate
  - air gap
  - other relevant stack entries

The intent is for layer stack to be reflected in layer numbers / datatypes inside the GDS artifacts.

### New design problems this creates

The user already identified two major problems:

1. how to get equivalent embeddings if two groups use different GDS layer-number conventions
2. whether two designs with identical device geometry but different ground-plane padding should have the same embedding or merely nearby embeddings

These are both foundational problems and should be treated as first-class design questions.

### Additional anticipated GDS/embedding problems

I expect at least the following issues to matter:

#### 1. Cell naming and hierarchy inconsistency

Different generators may produce:

- flat GDS
- hierarchical GDS
- different cell naming schemes
- repeated cell references versus baked geometry

The embedding pipeline should decide whether to embed:

- flattened geometry
- hierarchy-aware geometry
- or both

#### 2. Coordinate-frame inconsistency

The same component can appear with:

- different origin choices
- different rotations
- mirrored orientations
- different surrounding crop windows

This can dominate embedding similarity if not normalized carefully.

#### 3. Process-stack mismatch

Two geometrically identical layouts may correspond to different fabrication stacks.

Question:

- should the embedding encode geometry only?
- geometry plus process?
- versioned modes for both?

This should become an API-level choice.

#### 4. Ground-plane semantics

Ground-plane padding, ground cuts, keep-outs, and chip outline may affect the rendered GDS strongly even when the "device identity" is unchanged.

This likely requires multiple embedding views:

- component-local embedding
- context-aware embedding
- full-device embedding

#### 5. Polygonization / discretization noise

The same abstract geometry may produce slightly different polygon tessellations depending on exporter settings.

Embedding should be robust to:

- point ordering
- polygon fragmentation
- arc approximation differences
- tiny sliver polygons

#### 6. Layer mapping standards across groups

If different labs use different layer conventions, the pipeline will need:

- a canonical internal layer ontology
- import adapters / mappers
- validation checks

#### 7. Versioning of embedding algorithms

This is extremely important and the user explicitly called it out.

Embedding algorithms must be versioned.

The API and code should be designed from the beginning to support:

- `embedding_version`
- `layer_mapping_version`
- `normalization_policy_version`
- `context_policy_version`

This will matter for reproducibility and backward compatibility.

### Proposed embedding architecture direction

I recommend thinking of the future embedding subsystem as a family of versioned modes, not a single forever-definition.

For example:

- `component_v1`
  - geometry-only, local normalization
- `component_v2`
  - geometry + canonicalized layer stack
- `component_context_v1`
  - geometry + local ground context
- `device_v1`
  - full chip embedding

That makes experimentation possible without freezing the project too early.

### Visualization goals

The embedding work needs strong visualization support.

The user specifically wants:

- cosine similarity visualizations
- difference-vector visualizations
- intuitive plots of where embedding differences lie
- nice and interpretable latent-space visualizations

These should become first-class utilities, not ad hoc notebook snippets.

### Desired API and GUI

The future system should support both scripting and an intuitive GUI.

#### Python API goals

- simple
- robust
- obvious defaults
- easy to script for real workflows

Users should be able to:

- define components
- assign GDS files
- define connectivity graphs
- assign node and edge targets
- set edge types
- compute embeddings
- build graph datasets
- train/evaluate models

#### GUI goals

The GUI should be:

- lightweight
- browser-based
- cross-platform
- forgiving and reversible

The user wants it to support:

- drawing the connectivity graph / layout network
- assigning node types
- assigning targets to nodes and edges
- defining edge types
- attaching GDS files to nodes
- intuitive editing
- undo / redo
- copy / paste

The GUI should mirror programmatic capability, but the Python API remains the primary serious workflow interface.

### Paper direction for universal embeddings

There is a second paper opportunity here.

Likely paper narrative:

1. motivate a universal embedding space for superconducting building blocks
2. show that similar components cluster in embedding space
3. show that embedding arithmetic is meaningful
4. show that embeddings support downstream transfer learning and graph ML

### Core embedding-paper experiments

#### 1. Clustering and structure-preservation study

Show in 2D projections that:

- same component families cluster
- geometry variations move smoothly within clusters
- different component types separate appropriately

Use multiple projections, not just one:

- PCA
- t-SNE
- UMAP if added later

#### 2. Embedding arithmetic study

Explicit examples:

- `e(qubit+claw) - e(claw)` should land near `e(qubit)`
- analogous decompositions for other component pairs
- possibly interpolation / analogy experiments

This needs to be formalized as a benchmark, not just a pretty anecdote.

#### 3. Robustness study

Test embedding behavior under:

- layer remapping
- ground-padding changes
- rotation / mirroring
- exporter noise
- hierarchy differences

This is crucial if GDS-native embeddings are to be credible.

#### 4. Cross-lab normalization study

Simulate or collect equivalent layouts from multiple layer conventions and show that the canonicalization pipeline recovers stable embeddings.

### Graph ML paper direction

The graph ML paper story is harder, but promising.

The desired claim is not just "GNN predicts well", but something stronger:

- train on one geometry/topology family
- extend truthfully / faithfully to related but larger or altered topologies
- demonstrate transfer learning and compositional generalization

This is difficult, but it is the right ambitious target.

### Core graph-ML experiments I recommend

#### 1. Topology holdout benchmark

Train on one topology family and test on another:

- train on canonical 4-component chain
- test on extended chain / branched / multi-resonator / multi-qubit variants

This should be the main benchmark for the graph work.

#### 2. Component holdout / insertion benchmark

Train without certain component arrangements, then test when those arrangements appear.

This probes whether node/edge semantics are learned compositionally.

#### 3. Transfer-learning benchmark

Pretrain on one geometry family, finetune on a second smaller dataset, compare against training from scratch.

This directly supports the transfer-learning claim.

#### 4. Label-faithfulness / readout study

Because all nodes/edges may predict all targets, evaluate whether:

- the correct node/edge types carry the strongest signal
- inference readout maps surface truthful quantities
- attention or attribution can justify readout behavior

#### 5. Ablation study

Ablate:

- hub node
- edge features
- layer-stack encoding
- context-aware versus local embeddings
- shared-bounds embedding arithmetic features if introduced

This will tell you what actually matters.

#### 6. Synthetic compositional dataset

Create controlled synthetic graph families where the physics rules are known or approximated.

This can help debug whether graph generalization is real before spending expensive simulation effort on full FEM data.

### Encoder-decoder future

The user explicitly wants to keep the future open for:

- encoder-decoder models
- better models in general
- inverse design / generative design

That should remain in the long-range roadmap, but not block current forward-model and embedding-foundation work.

## 1. Explainable ML and Inverse Design

Status: `Implemented / verified by code read`

### Key files

- [`squadds/ml/ebm.py`](squadds/ml/ebm.py)
- [`squadds/ml/symbolic.py`](squadds/ml/symbolic.py)
- [`squadds/ml/pipeline.py`](squadds/ml/pipeline.py)
- [`squadds/ml/utils.py`](squadds/ml/utils.py)
- [`tutorials/Coupler_EBM_Analysis.py`](tutorials/Coupler_EBM_Analysis.py)
- [`tutorials/Coupler_Inverse_Design_Analysis.ipynb`](tutorials/Coupler_Inverse_Design_Analysis.ipynb)
- [`curate_dataset.py`](curate_dataset.py)
- [`coupler_capacitance_data.csv`](coupler_capacitance_data.csv)

### What it does

This is the earliest ML layer added on this branch. It focuses on coupler inverse design and explainability:

- curate coupler capacitance data from the Hugging Face dataset
- fit an `ExplainableBoostingRegressor` to identify important features and interactions
- use those selected features as input to PySR symbolic regression
- produce interpretable equations and plots

The main wrapper is [`SQuADDSAnalysisPipeline`](squadds/ml/pipeline.py), which does:

1. split the dataframe into train/test
2. fit `EBMAnalyzer`
3. extract top features and interaction pairs
4. build interaction-augmented features
5. fit `SymbolicRegressor`
6. return metrics, selected features, and best discovered equations per target

### Package integration

- `pyproject.toml` gained an `ml` extra:
  - `interpret`
  - `pysr`
  - `scikit-learn`
- [`squadds/ml/__init__.py`](squadds/ml/__init__.py) was added as the ML package entry point.

### What seems solid

- The pipeline code is straightforward and reasonably self-contained.
- The tests around EBM/symbolic pieces exist and at least one of them passed in my environment.
- This part feels closest to "real package code" rather than pure notebook experimentation.

### Caveats

- It is mostly focused on coupler capacitance analysis, not yet a general inverse-design framework across all SQuADDS components.
- It depends on `interpret` and `pysr`, which are heavier dependencies than the core package.

## 2. Graph Forward Model (Keras + Spektral)

Status: `Implemented / verified by code read`

### Key files

- [`squadds/ml/graph/__init__.py`](squadds/ml/graph/__init__.py)
- [`squadds/ml/graph/featurizer.py`](squadds/ml/graph/featurizer.py)
- [`squadds/ml/graph/encoders.py`](squadds/ml/graph/encoders.py)
- [`squadds/ml/graph/gnn_model.py`](squadds/ml/graph/gnn_model.py)
- [`squadds/ml/graph/trainer.py`](squadds/ml/graph/trainer.py)
- [`squadds/ml/graph/component_data/CavityClaw.json`](squadds/ml/graph/component_data/CavityClaw.json)
- [`tutorials/Tutorial-11_Graph_Forward_Model.ipynb`](tutorials/Tutorial-11_Graph_Forward_Model.ipynb)

### Role in the branch

This is the middle phase of the branch. It is more ambitious than the explainable ML path and is closer to a circuit-level forward model:

- circuits are turned into graphs
- components become graph nodes
- edges represent physical connectivity
- a GNN predicts Hamiltonian parameters from graph structure + component features

### Data and feature pipeline

The core featurization flow is in [`featurizer.py`](squadds/ml/graph/featurizer.py):

- `build_vocab()` scans enriched component JSONs and builds a parameter-key vocabulary
- `ComponentFeaturizer` merges JSON defaults with per-row overrides
- it extracts:
  - layer stack
  - design parameters as `(key_id, value)` pairs
  - area and perimeter
  - 5-element port vector
- `CircuitGraphBuilder` assembles those into `spektral.data.Graph`

The flat node feature layout expected by the Keras model is:

```text
[layer_stack (n_ls*2) | design_params (k_max*2) | area (1) | perimeter (1) | ports (5)]
```

### Model architecture

The graph stack is in [`gnn_model.py`](squadds/ml/graph/gnn_model.py):

- `LayerStackEncoder`: Conv1D over the layer stack
- `GeometricEncoder`: DeepSets-like or sum-based aggregation over design params
- `PortEncoder`: dense encoder for the port counts
- `NodeEncoder`: fuses the three encoders into `E_static`
- `GraphForwardModel`: message passing + graph pooling + readout

Important details:

- This code was explicitly reworked for Keras 3.
- Custom graph layers are implemented directly in Keras:
  - `GCNConvK3`
  - `GraphAttentionConvK3`
  - `NormalizeAdjacencyK3`
  - `GlobalAttentionPoolK3`
- Message passing mode can be `gcn` or `gat`.
- `GraphTrainer` wraps training, evaluation, saving/loading, and embedding extraction.

### Packaging and API status

- This module is lazily exposed through [`squadds/ml/__init__.py`](squadds/ml/__init__.py).
- `pyproject.toml` gained a `graph` extra with `tensorflow` and `spektral`.
- This is the only ML subpackage currently wired into the public-ish ML entry point.

### Current caveats

- In my environment, TensorFlow import is currently broken because of a protobuf mismatch:
  - `ImportError: cannot import name 'runtime_version' from 'google.protobuf'`
- As a result, `tests/ml/test_gnn_model.py` skipped rather than ran cleanly.
- Tutorial 11 still contains saved traceback output from an older mixed tensor/non-tensor batching bug, even though current `GraphTrainer` has `_coerce_model_inputs()` to normalize batches into tensors.

Net: this graph path looks real and thought through, but it is environment-sensitive.

## 3. Universal Geometry-First GNN Pipeline

Status: `Implemented / notebook-driven`

This is the current frontier of the branch and the most important context for future work.

### Important high-level truth

The universal path is **not** yet fully integrated into the package in the same way as the Keras graph module:

- it lives under [`squadds/ml/universal`](squadds/ml/universal)
- [`squadds/ml/__init__.py`](squadds/ml/__init__.py) does **not** export it
- `pyproject.toml` does **not** declare `torch` or `torch_geometric` extras for it
- most of its active use seems to happen through [`tutorials/Tutorial-12_Universal_GNN.ipynb`](tutorials/Tutorial-12_Universal_GNN.ipynb)

So this is best understood as "research code on a branch that is converging toward a real subsystem", not yet a stable public API.

### 3.1 Geometry layer

Status: `Implemented / verified by code read`

#### Key files

- [`squadds/ml/universal/geometry/qubit.py`](squadds/ml/universal/geometry/qubit.py)
- [`squadds/ml/universal/geometry/claw.py`](squadds/ml/universal/geometry/claw.py)
- [`squadds/ml/universal/geometry/feedline.py`](squadds/ml/universal/geometry/feedline.py)
- [`squadds/ml/universal/geometry/resonator.py`](squadds/ml/universal/geometry/resonator.py)
- [`squadds/ml/universal/geometry/layout.py`](squadds/ml/universal/geometry/layout.py)
- [`squadds/ml/universal/geometry/composite.py`](squadds/ml/universal/geometry/composite.py)
- [`squadds/ml/universal/geometry/viz.py`](squadds/ml/universal/geometry/viz.py)

#### What exists

There are two layout builders:

- `build_layout(...)`
  - canonical 4-component SQuADDS chain
  - qubit, claw, resonator, feedline
  - placement defaults are chosen to mirror SQuADDS DB geometry conventions
- `build_composite_layout(...)`
  - generalized builder for arbitrary combinations of:
    - `TransmonCross`
    - `Claw`
    - `RouteMeander`
    - `CoupledLineTee`
  - driven by `PlacedComponent`

This matters because Tutorial 12 started with the standard 4-component chain, then evolved into "Case 2" and other extended topologies.

#### Why this geometry layer matters

The universal pipeline is geometry-first, not parameter-table-first:

```text
design parameters -> Shapely polygons -> deterministic embeddings -> hetero graph -> predictions
```

That is the key conceptual shift away from Tutorial 8 style tabular models.

### 3.2 Static component embedding

Status: `Implemented / verified by code read`

#### Key files

- [`squadds/ml/universal/features/node_encoder.py`](squadds/ml/universal/features/node_encoder.py)
- [`squadds/ml/universal/features/moments.py`](squadds/ml/universal/features/moments.py)
- [`squadds/ml/universal/features/rasterizer.py`](squadds/ml/universal/features/rasterizer.py)

#### Current embedding definition at HEAD

Each component gets a deterministic embedding:

```text
embedding = [param_sum (1)] || [moments (8)] || [shape tensor (R^2)]
```

with default resolution:

- `R = 16`
- embedding dimension = `1 + 8 + 16^2 = 265`

#### What each part means

1. `param_sum`
   - simple sum of numeric design parameter values
   - permutation-invariant
   - intentionally simple, but lossy

2. geometric moments
   - current code computes these 8 values:
     - `area`
     - `perimeter`
     - `bbox_area`
     - `bbox_perimeter`
     - `fill_factor`
     - `compactness`
     - `aspect_ratio`
     - `circularity`

3. shape tensor
   - rasterized binary mask of the polygon
   - flattened to length `R^2`

#### Important behavior

The current rasterizer normalizes to each polygon's own bounding box.
That means:

- shape tensor is mostly shape-sensitive and size-invariant
- size information is expected to come from the moment features
- this is good for generic component recognition
- this is **not** ideal for linear embedding arithmetic unless a shared coordinate frame is enforced

### 3.3 Embedding arithmetic and shared bounds

Status: `Implemented / notebook-driven`, plus `Inherited note / proposal`

The prior AI note you shared contains a genuinely important idea:

```text
embed(qubit + claw) - embed(claw) ~= embed(qubit)
```

That note attributes the effect to using a shared bounding box / shared coordinate frame when rasterizing multiple related shapes.

My take:

- this idea is worth preserving
- it is consistent with how rasterized additive geometry would behave
- it is likely demoed in Tutorial 12 now
- but it is **not** the default behavior of `compute_static_embedding()` at HEAD

So future work should treat "shared-frame embedding arithmetic" as a promising notebook/demo technique that should probably be turned into explicit reusable code.

### 3.4 Edge features and hub features

Status: `Implemented / verified by code read`

#### Key files

- [`squadds/ml/universal/features/edge_extractor.py`](squadds/ml/universal/features/edge_extractor.py)
- [`squadds/ml/universal/graph/virtual_hub.py`](squadds/ml/universal/graph/virtual_hub.py)

#### Physical edge features

Current physical edge feature vector is:

```text
[coupling one-hot (3)] || [dx, dy (2)] || [overlap_area, overlap_perimeter, overlap_bbox_area (3)] || [overlap raster (R^2)]
```

Dimension:

- `3 + 2 + 3 + R^2 = 8 + R^2`
- default at `R=16` -> `264`

Notes:

- For galvanic coupling, overlap is direct intersection.
- For non-galvanic coupling, both polygons are buffered before intersecting to approximate near-field interaction geometry.

#### Virtual hub embedding

Current hub embedding is richer than the prior note suggested.
At HEAD it contains:

```text
[layout-union raster (R^2)]
|| [layout moments (8)]
|| [param_sum, chip_area, metal_fill (3)]
|| [global_info padded to 5 slots]
```

Dimension:

- `R^2 + 8 + 3 + 5`
- default at `R=16` -> `272`

#### Spatial hub-edge features

Current spatial edge vector is:

```text
[relative centroid dx, dy (2)] || [area fraction, perimeter fraction (2)] || [masked raster in full-layout bounds (R^2)]
```

Dimension:

- `4 + R^2`
- default at `R=16` -> `260`

### 3.5 Graph schema

Status: `Implemented / verified by code read`

#### Key files

- [`squadds/ml/universal/graph/netlist.py`](squadds/ml/universal/graph/netlist.py)
- [`squadds/ml/universal/graph/builder.py`](squadds/ml/universal/graph/builder.py)

The universal graph builder now produces `torch_geometric.data.HeteroData`.

#### Node types

- `"component"`
  - one node per physical component
  - `.x` is the static component embedding
  - `.y` is node-target storage
  - metadata includes `component_type`, `component_name`, and `inference_readout`
- `"virtual"`
  - single global hub node
  - `.x` is the hub embedding

#### Edge types

- `("component", "physical", "component")`
  - physical component-to-component interactions
  - stored in both directions
- `("component", "spatial_in", "virtual")`
  - component -> hub
- `("virtual", "spatial_out", "component")`
  - hub -> component

Important correction relative to older notes:

- the reverse spatial edge type is `("virtual", "spatial_out", "component")`, not `("component", "spatial_out", "virtual")`

#### Labels

The builder initializes node and edge label tensors, but the real target assignment is tutorial/dataset driven.

Current constants in [`gat_model.py`](squadds/ml/universal/model/gat_model.py):

- node targets:
  - `qubit_freq_GHz`
  - `anharmonicity_MHz`
  - `cavity_freq_GHz`
- edge targets:
  - `g_MHz`
  - `kappa_kHz`

The current Tutorial 12 approach appears to assign all node targets to all nodes and all edge targets to all physical edges.

This is a deliberate design choice:

- let message passing learn which geometry/context controls which target
- use metadata maps only at inference/readout time

### 3.6 Universal GNN model and trainer

Status: `Implemented / verified by code read`

#### Key files

- [`squadds/ml/universal/model/gat_model.py`](squadds/ml/universal/model/gat_model.py)
- [`squadds/ml/universal/model/prediction_heads.py`](squadds/ml/universal/model/prediction_heads.py)
- [`squadds/ml/universal/model/loss.py`](squadds/ml/universal/model/loss.py)
- [`squadds/ml/universal/trainer.py`](squadds/ml/universal/trainer.py)

#### Current constructor shape

Important: the current `UniversalGNN` API expects:

```python
UniversalGNN(
    comp_dim=...,
    virt_dim=...,
    phys_edge_dim=...,
    spat_edge_dim=...,
    hidden_dim=128,
    edge_hidden=32,
    num_layers=3,
    num_heads=4,
)
```

This is the new heterogeneous API. Older tests still assume a homogeneous constructor like `node_dim=` / `edge_dim=`.

#### Internal architecture

The model does:

1. project each node/edge type into shared hidden spaces
2. run `HeteroConv` layers
3. predict node targets from component node states
4. predict edge targets from concatenated source node, destination node, and edge state

The relation-specific layers are:

- physical edges: `GATv2Conv`
- component -> virtual: `SAGEConv`
- virtual -> component: `GATv2Conv`

This is an interesting asymmetric design: the hub receives information through GraphSAGE-style aggregation and sends information back through attention.

#### Loss handling

There are two slightly different stories in the codebase:

- [`model/loss.py`](squadds/ml/universal/model/loss.py) defines `MaskedMultiTaskLoss`
- [`trainer.py`](squadds/ml/universal/trainer.py) does **not** currently use that class
- instead, `UniversalTrainer._step()` manually computes MSE with NaN masks

So the current trainer supports sparse labels in principle, even though the main Tutorial 12 story is "all nodes predict all node targets, all edges predict all edge targets".

#### Inference readout metadata

`gat_model.py` also defines:

- `NODE_INFERENCE_READOUT`
- `EDGE_INFERENCE_READOUT`

These are metadata for deciding which predictions to surface per component/edge type at inference time.
They are not used to mask the training loss.

### 3.7 Tutorial 12: what it currently seems to contain

Status: `Implemented / notebook-driven`

Canonical notebook:

- [`tutorials/Tutorial-12_Universal_GNN.ipynb`](tutorials/Tutorial-12_Universal_GNN.ipynb)

From code inspection, this notebook now includes:

- universal pipeline introduction
- imports from geometry, features, graph builder, and hetero model
- graph inspection sections that explicitly reference `HeteroData`
- training dataset build-up
- node and edge target readout
- multi-projection latent-space analysis
- t-SNE section
- embedding arithmetic content
- extended "Case 2" topology
- notes about ground truth for the extended topology

Important local-worktree note:

- there are multiple untracked scratch copies of Tutorial 12 in this repo at the moment
- the canonical committed notebook is still `tutorials/Tutorial-12_Universal_GNN.ipynb`
- do not treat untracked copies like `Tutorial-12_Universal_GNN_copy_v1.ipynb` or `Tutorial-12_Universal_GNN copy.ipynb` as authoritative branch history

## Useful Content Preserved From the Previous AI Note

Status: `Inherited note / proposal`

The note you shared is worth preserving, but some of it is older than current HEAD. The parts that still feel genuinely useful are below.

### Ideas that are still strategically important

1. Shared-frame embedding arithmetic
   - likely key to making embedding composition/decomposition meaningful
   - should become reusable code, not just notebook logic

2. GDS-native ingestion path
   - long-term goal: go from actual GDS geometry to embeddings directly
   - avoid tying the model entirely to Python parametric generators

3. Multichannel geometry encoding by GDS layer/datatype
   - current embedding is single-channel
   - future version should probably treat different layers as separate channels

4. Ground-plane invariance
   - a real issue for geometry-derived features
   - future versions should isolate signal metal from context metal or at least normalize better

5. Encoder-decoder / generative direction
   - useful long-term goal: targets -> embeddings -> parameters or GDS
   - probably start with parameter regression before full geometry generation

6. Better parameter aggregation than `param_sum`
   - the previous note is right that current parameter encoding is too lossy
   - DeepSets-like learned param aggregation is a strong next step

7. Production scaling goals
   - larger `R`
   - more topologies
   - stronger evaluation metrics
   - better hardware assumptions

### Things from the note that should be treated as ideas, not current truth

- claimed exact cosine `1.000` embedding subtraction result
- references to an external snippet file outside this repo
- older edge feature descriptions that no longer match current `edge_extractor.py`
- older hub embedding description that omits current moments and global statistics
- older schema descriptions that use outdated edge directions

## Corrections to the Previous AI Note

These are the biggest places where current HEAD differs from that note.

### 1. Moment definitions changed

Current code computes:

- `area`
- `perimeter`
- `bbox_area`
- `bbox_perimeter`
- `fill_factor`
- `compactness`
- `aspect_ratio`
- `circularity`

It does **not** currently compute centroid coordinates, convexity, or second moments.

### 2. Hub embedding is richer than the note says

Current hub embedding includes:

- layout raster
- layout moments
- `param_sum`
- `chip_area`
- `metal_fill`
- zero-padded global-info slots

It is not just "layout raster + 5 globals".

### 3. Spatial reverse edge type is different

Current reverse spatial edge is:

```text
("virtual", "spatial_out", "component")
```

not:

```text
("component", "spatial_out", "virtual")
```

### 4. Universal trainer still allows NaN masking

The prior note emphasized "no masking".
The real situation is:

- Tutorial 12 seems to use dense targets for all nodes/edges
- but the current `UniversalTrainer` still masks out `NaN` values if present

So sparse-label support was not actually removed from the trainer implementation.

### 5. Old universal API is stale

Older tests and notes refer to:

- `UniversalGNN(node_dim=..., edge_dim=...)`
- `VirtualHubInjector`
- `NodeFeatureEncoder`
- `NoInteractionError`

Current HEAD no longer matches that old homogeneous design.

## Known Issues and Technical Debt

Status: `Known stale / broken`

### Universal tests are stale

This is the biggest immediate issue if future work is going to build on the universal path.

Observed on 2026-04-15:

- `tests/ml/universal/test_features.py`
  - import error: `NoInteractionError` no longer exists in current `edge_extractor.py`
- `tests/ml/universal/test_graph.py`
  - import error: `VirtualHubInjector` no longer exists in current `virtual_hub.py`
- `tests/ml/universal/test_model.py`
  - uses old constructor args `node_dim` / `edge_dim`
- `tests/ml/universal/test_trainer.py`
  - same old constructor args

This means the universal code moved forward faster than the test suite.

### Universal packaging is incomplete

- `pyproject.toml` has no declared extra for `torch` / `torch_geometric`
- `squadds/ml/__init__.py` does not expose universal symbols
- `squadds/ml/universal/__init__.py` is only a docstring

So the universal subsystem is not yet packaged like the Keras graph subsystem.

### Committed generated artifacts

There are committed bytecode files under:

- [`squadds/ml/universal/features/__pycache__`](squadds/ml/universal/features/__pycache__)

These should probably not be in source control.

### `wish_list.md` contains pasted AI text

[`wish_list.md`](wish_list.md) includes a chunk of AI-generated guidance text that is unrelated to the normal wishlist format.
That file should be cleaned up at some point.

### TensorFlow environment friction

The Keras graph path may be correct in code but is brittle in the current environment because TensorFlow import is broken by protobuf version issues.

### Worktree was dirty when this document was written

At the time of writing:

- `tutorials/Tutorial-12_Universal_GNN.ipynb` was modified but uncommitted
- there were many untracked helper scripts, notebook copies, cached outputs, and checkpoints

Those local artifacts are useful for context, but they are **not** the same as committed branch history.
Future sessions should always check `git status` before assuming a local file is part of the branch narrative.

## What I Verified On 2026-04-15

### Branch comparison and structure

I fetched remotes and compared:

- `HEAD`
- `origin/master`
- `upstream/master`

Conclusion:

- canonical comparison should be against `upstream/master`
- current branch is cleanly on top of `upstream/master`

### Tests I actually ran

- `uv run pytest tests/ml/test_ebm.py -q`
  - result: `3 passed`
- `uv run pytest tests/ml/test_gnn_model.py -q`
  - result: skipped because TensorFlow import failed in this environment
- `uv run pytest tests/ml/universal/test_model.py -q`
  - result: failed because tests still use old `UniversalGNN(node_dim=...)` API
- `uv run pytest tests/ml/universal/test_graph.py -q`
  - result: import error for removed `VirtualHubInjector`
- `uv run pytest tests/ml/universal/test_trainer.py -q`
  - result: failed because tests still use old `node_dim` / `edge_dim` constructor
- `uv run pytest tests/ml/universal/test_features.py -q`
  - result: import error for removed `NoInteractionError`

### Code-level conclusions from that verification

- Explainable ML path is in the best shape right now.
- Keras graph path is real but environment-sensitive.
- Universal path is the most interesting part, but it is currently notebook-first and its tests lag behind the implementation.

## Prioritized Execution Plan

Status: `Action plan updated from user guidance on 2026-04-15`

This section is the practical roadmap.
It is organized around what should happen first, what can happen in parallel, and what depends on what.

### Overall priority order across the three model families

#### Priority 0: short-term MLP productionization

This is the most concrete near-term deliverable:

- receive collaborator-finalized MLP artifacts
- productionize them
- publish via Hugging Face models and Spaces
- update tutorials and docs

This is the least scientifically ambiguous track and should move quickly once collaborators finish the models.

#### Priority 1: symbolic-regression paper push

This is likely the best immediate paper opportunity.
It already has:

- a working methodological backbone
- interpretable results
- a strong physics-discovery angle

The main missing pieces are:

- improved IDC dataset size
- extrapolation experiments
- stronger comparison and theory-validation studies

#### Priority 2: universal pipeline stabilization

Before attempting major claims or papers on the universal track:

- align code, tests, and packaging
- formalize dataflow and APIs
- move notebook orchestration into reusable package functions

Without this, every new experiment will be expensive and brittle.

#### Priority 3: GDS-native embedding engine

Once the universal pipeline is stable enough to support iteration:

- build the GDS-native rendering / ingestion path
- generate the five GDS files per row
- support ground-plane padding and stack-aware layer rendering

This is a major engineering milestone and probably the true beginning of the "real" embedding system.

#### Priority 4: graph-ML generalization and transfer studies

Only after the embedding/data pipeline is stable enough should the graph-ML paper claims be pushed hard:

- truthful extension to larger / changed topologies
- transfer learning studies
- compositional generalization benchmarks

#### Priority 5: GUI and broader platform layer

The browser-based editor / graph builder is valuable, but it should not come before:

- stable file formats
- stable embedding APIs
- stable graph dataset contracts

Otherwise the GUI will freeze poor abstractions too early.

## Asynchronous Workstreams

The three tracks can and should progress asynchronously.

### Track A async plan: MLP productionization

This track can run mostly independently of the other two.

#### A1. Collaborator handoff

- receive finalized model artifacts
- receive preprocessing details
- receive train/val/test metrics and intended usage bounds

#### A2. Production wrapper

- package inference utilities inside SQuADDS
- create stable input schema and validation
- add model versioning and metadata

#### A3. Deployment

- publish Hugging Face model repo
- publish Hugging Face Space
- write/update tutorial for transmoncross workflow

#### A4. Production QA

- latency / consistency checks
- tutorial smoke tests
- doc polish

### Track B async plan: symbolic-regression paper

This can advance in parallel with MLP productionization and with early universal stabilization.

#### B1. Quarter-wave result hardening

- cleanly package the quarter-wave symbolic workflow
- reproduce the main equations
- compare them against independent first-principles derivation
- prepare final figure/table pipeline

#### B2. IDC dataset expansion

- colleague expands the IDC capacitance dataset
- retrain symbolic models after expansion
- reassess performance and equation quality

This dependency should be explicitly tracked because it affects whether IDC becomes a major or minor paper section.

#### B3. Extrapolation study

- define interpolation and extrapolation splits
- compare symbolic regression to MLP baselines
- report both accuracy and physical plausibility

#### B4. Scientific structure discovery layer

- integrate AI Feynman ideas where appropriate
- add automated dimensional analysis
- add parameter-set reduction
- identify dimensionless groups or low-dimensional effective coordinates

#### B5. Cross-shape IDC transfer study

- build second IDC shape-family dataset
- test equation transfer / reinterpretation / reparameterization
- determine whether the symbolic structure survives family shift

#### B6. Manuscript production

- problem statement and motivation
- methods section
- theory comparison
- extrapolation comparison
- IDC transfer section if results support it

### Track C async plan: universal embedding + graph ML

This is the biggest track and should be split into sub-workstreams.

#### C1. Foundation hardening

- decide that this is becoming a first-class internal subsystem
- update stale tests to match current heterogeneous API
- add dependency handling for `torch` / `torch_geometric`
- define stable internal interfaces
- move notebook orchestration into reusable code

#### C2. Embedding-science work

- formalize shared-frame embedding arithmetic
- define robustness tests for rotation, padding, layer remapping, and exporter noise
- improve parameter encoding beyond `param_sum`
- decide how to treat context versus geometry-only embeddings

#### C3. GDS pipeline engineering

- create row -> GDS generation engine
- emit:
  - `qubit.gds`
  - `claw.gds`
  - `cpw.gds`
  - `feedline.gds`
  - `device.gds`
- add user-defined ground-plane padding
- encode layer stack in canonicalized layers / datatypes
- create custom QComponents as needed

#### C4. Embedding-space paper studies

- cluster-structure visualizations
- embedding arithmetic benchmark
- robustness benchmark
- cross-convention normalization benchmark

#### C5. Graph-ML paper studies

- topology holdout benchmark
- transfer-learning benchmark
- compositional generalization benchmark
- ablations over hub/features/edge semantics

#### C6. Future-model direction

- encoder-decoder experiments
- inverse design ideas
- richer downstream models

#### C7. Platform / UX layer

- clean Python API
- visualization helpers
- lightweight browser GUI for graph construction and target assignment

## Detailed Near-Term Plan

This is the concrete recommendation for the next phase of work.

### Next 1-2 short-term deliverables

#### Deliverable 1: MLP productionization

- finish collaborator handoff
- package transmoncross HF models
- deploy Spaces demo
- update tutorials

#### Deliverable 2: symbolic paper readiness pass

- harden quarter-wave symbolic results
- lock in theory-comparison narrative
- prepare extrapolation benchmark setup
- track IDC dataset expansion dependency explicitly

### Immediate universal-track engineering goals

Before major new model ideas, do these first:

- update stale universal tests
- formalize current dataset-building flow
- factor Tutorial 12 orchestration into library functions
- decide naming/versioning conventions for embeddings
- write down the expected contracts for:
  - component embedding
  - device embedding
  - graph construction
  - readout maps

## Suggested Symbolic Paper Structure

One plausible paper structure:

1. Motivation
   - inverse design is valuable but black-box models are hard to trust
   - symbolic regression supports scientific interpretation and extrapolation
2. Framework
   - EBM for feature discovery
   - symbolic regression for equation discovery
   - optional dimensional-analysis / AI-Feynman-inspired reduction layer
3. Quarter-wave qubit-cavity case study
   - learned equations
   - first-principles derivation
   - agreement / discrepancies
4. Extrapolation benchmark
   - symbolic vs MLP
5. IDC case study
   - current dataset result
   - retrained expanded-dataset result
6. Cross-shape transfer / reparameterization study
7. Discussion
   - physics discovery
   - parameter reduction
   - limits and future directions

## Suggested Universal Embedding / Graph-ML Paper Structure

One plausible split is actually two papers:

### Option 1: one combined paper

- universal embedding space
- embedding arithmetic
- graph-ML downstream transfer learning

This is ambitious, but possibly too much for one clean story.

### Option 2: two-paper strategy

#### Paper A: universal embeddings

- motivation for a universal component embedding space
- clustering results
- arithmetic / composition results
- robustness and canonicalization
- GDS-native formulation

#### Paper B: graph ML on top of the universal embeddings

- hetero graph schema
- faithful transfer across topologies
- compositional generalization
- transfer learning experiments

I suspect the two-paper strategy may ultimately be cleaner unless the graph-ML results become very strong.

## Concrete Universal-Track Research Questions

These are the questions future work should explicitly answer.

### Embedding questions

1. Should embeddings be geometry-only, context-aware, or both?
2. What variations should preserve embedding identity exactly?
3. What variations should only preserve neighborhood / cluster closeness?
4. How should layer-stack information be canonicalized across groups?
5. What is the right versioning boundary for embedding algorithms?

### Graph questions

1. Can a model trained on one topology generalize truthfully to another?
2. What node/edge semantics are required for that to work?
3. Does the hub materially help transfer, or only training stability?
4. What labels should live on nodes versus edges?
5. When does graph structure help beyond strong per-component embeddings?

### Platform questions

1. What is the stable object model for a component, device, and graph?
2. What should the user-facing API look like?
3. What should be configurable versus version-locked?
4. How can GUI interactions map cleanly onto the scriptable API?

## Guidance For Future Sessions

If resuming work later, use this order:

1. Ask which track is the focus:
   - MLP productionization
   - symbolic paper
   - universal embedding / graph ML
2. If symbolic:
   - prioritize quarter-wave theory comparison and extrapolation experiments
3. If universal:
   - prioritize stabilization and dataset contracts before new architecture changes
4. Treat GDS-native embedding as the real long-term target, not the current endpoint
5. Keep embedding/versioning/API design in mind from the beginning

## Quick File Map For Future Sessions

### Repo-level orientation

- [`README.md`](README.md) - project framing
- [`pyproject.toml`](pyproject.toml) - dependencies and extras
- [`BRANCH_CONTEXT_ml-embedding_2026-04-15.md`](BRANCH_CONTEXT_ml-embedding_2026-04-15.md) - this file
- [`ML_TASKBOARD_2026-04-15.md`](ML_TASKBOARD_2026-04-15.md) - execution board across all three ML tracks
- [`UNIVERSAL_EMBEDDING_PROTOCOL_2026-04-15.md`](UNIVERSAL_EMBEDDING_PROTOCOL_2026-04-15.md) - embedding-mode/version protocol and invariance note

### Core SQuADDS baseline

- [`squadds/core/db.py`](squadds/core/db.py)
- [`squadds/simulations/ansys_simulator.py`](squadds/simulations/ansys_simulator.py)
- [`squadds/simulations/objects.py`](squadds/simulations/objects.py)

### Explainable ML

- [`squadds/ml/ebm.py`](squadds/ml/ebm.py)
- [`squadds/ml/symbolic.py`](squadds/ml/symbolic.py)
- [`squadds/ml/pipeline.py`](squadds/ml/pipeline.py)
- [`squadds/ml/utils.py`](squadds/ml/utils.py)
- [`tutorials/Coupler_Inverse_Design_Analysis.ipynb`](tutorials/Coupler_Inverse_Design_Analysis.ipynb)

### Keras graph forward model

- [`squadds/ml/graph/featurizer.py`](squadds/ml/graph/featurizer.py)
- [`squadds/ml/graph/encoders.py`](squadds/ml/graph/encoders.py)
- [`squadds/ml/graph/gnn_model.py`](squadds/ml/graph/gnn_model.py)
- [`squadds/ml/graph/trainer.py`](squadds/ml/graph/trainer.py)
- [`tutorials/Tutorial-11_Graph_Forward_Model.ipynb`](tutorials/Tutorial-11_Graph_Forward_Model.ipynb)

### Universal pipeline

- [`squadds/ml/universal/geometry/layout.py`](squadds/ml/universal/geometry/layout.py)
- [`squadds/ml/universal/geometry/composite.py`](squadds/ml/universal/geometry/composite.py)
- [`squadds/ml/universal/workflows.py`](squadds/ml/universal/workflows.py)
- [`squadds/ml/universal/features/protocol.py`](squadds/ml/universal/features/protocol.py)
- [`squadds/ml/universal/features/arithmetic.py`](squadds/ml/universal/features/arithmetic.py)
- [`squadds/ml/universal/features/node_encoder.py`](squadds/ml/universal/features/node_encoder.py)
- [`squadds/ml/universal/features/edge_extractor.py`](squadds/ml/universal/features/edge_extractor.py)
- [`squadds/ml/universal/graph/netlist.py`](squadds/ml/universal/graph/netlist.py)
- [`squadds/ml/universal/graph/builder.py`](squadds/ml/universal/graph/builder.py)
- [`squadds/ml/universal/graph/virtual_hub.py`](squadds/ml/universal/graph/virtual_hub.py)
- [`squadds/ml/universal/model/gat_model.py`](squadds/ml/universal/model/gat_model.py)
- [`squadds/ml/universal/trainer.py`](squadds/ml/universal/trainer.py)
- [`tutorials/Tutorial-12_Universal_GNN.ipynb`](tutorials/Tutorial-12_Universal_GNN.ipynb)

### Tests worth reading before editing

- [`tests/ml/test_ebm.py`](tests/ml/test_ebm.py)
- [`tests/ml/test_gnn_model.py`](tests/ml/test_gnn_model.py)
- [`tests/ml/test_graph_trainer.py`](tests/ml/test_graph_trainer.py)
- [`tests/ml/universal/test_features.py`](tests/ml/universal/test_features.py)
- [`tests/ml/universal/test_graph.py`](tests/ml/universal/test_graph.py)
- [`tests/ml/universal/test_model.py`](tests/ml/universal/test_model.py)
- [`tests/ml/universal/test_trainer.py`](tests/ml/universal/test_trainer.py)

The universal tests are now more trustworthy than they were at the start of this review pass, but still treat them as a guardrail around the current implementation rather than as a final statement of architecture.

## Suggested Reading Order For The Next Agent

1. Read this file.
2. Open [`tutorials/Tutorial-12_Universal_GNN.ipynb`](tutorials/Tutorial-12_Universal_GNN.ipynb) to see the current research direction.
3. Read the universal pipeline code in this order:
   - [`geometry/layout.py`](squadds/ml/universal/geometry/layout.py)
   - [`workflows.py`](squadds/ml/universal/workflows.py)
   - [`features/protocol.py`](squadds/ml/universal/features/protocol.py)
   - [`features/arithmetic.py`](squadds/ml/universal/features/arithmetic.py)
   - [`features/node_encoder.py`](squadds/ml/universal/features/node_encoder.py)
   - [`features/edge_extractor.py`](squadds/ml/universal/features/edge_extractor.py)
   - [`graph/virtual_hub.py`](squadds/ml/universal/graph/virtual_hub.py)
   - [`graph/builder.py`](squadds/ml/universal/graph/builder.py)
   - [`model/gat_model.py`](squadds/ml/universal/model/gat_model.py)
   - [`trainer.py`](squadds/ml/universal/trainer.py)
4. After that, read the older Keras graph path only if needed for comparison or reuse.
5. Treat the prior AI note as a source of useful future ideas, but not as the ground truth for current implementation.

## Bottom Line

This branch is best understood as the ML R&D branch for SQuADDS:

- one track is near-term productization:
  - collaborator-built MLP models
- one track is near-term science/paper:
  - symbolic regression for physics discovery and extrapolation
- one track is long-horizon research + platform building:
  - universal embeddings, GDS-native representations, and graph ML

The symbolic track is currently the best paper candidate.
The MLP track is the cleanest short-term productionization target.
The universal track has the highest upside, but also the most technical debt and open design uncertainty.

If continuing this work:

- ship the MLP deployment track when collaborators hand it off
- push the symbolic track toward a strong paper with extrapolation and theory-validation studies
- treat the universal track as a structured research program:
  - stabilize it
  - make the embedding system GDS-native and versioned
  - then pursue graph-ML transfer/generalization claims on top of that stronger foundation
