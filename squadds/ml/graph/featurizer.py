"""
Data pipeline utilities for converting SQuADDS layouts into graph objects.

Provides:
- ``ComponentFeaturizer``: Extracts raw heterogeneous features from a single component.
- ``CircuitGraphBuilder``: Assembles per-component features into a ``spektral.data.Graph``.
- ``SQuADDSGraphDataset``: A ``spektral.data.Dataset`` for batched training.
- ``build_vocab``: Builds a parameter-key vocabulary from enriched component JSONs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Lazy TF / Spektral imports — called only when functionality is used
# ---------------------------------------------------------------------------

_spektral = None


def _ensure_spektral():
    global _spektral
    if _spektral is None:
        try:
            import spektral

            _spektral = spektral
        except ImportError as exc:
            raise ImportError(
                "Spektral is required for the graph ML module. Install with: pip install SQuADDS[graph]"
            ) from exc
    return _spektral


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAD_TOKEN = "<PAD>"

# Default silicon/aluminium layer-stack properties
# Each row: [thickness_nm, dielectric_constant, penetration_depth_nm]
DEFAULT_LAYER_STACK = np.array(
    [
        [500_000.0, 11.45, 0.0],  # Substrate (Si, 500 µm)
        [100.0, 1.0, 50.0],  # Metal (Al, 100 nm)
        [1e6, 1.0, 0.0],  # Air gap
        [0.0, 0.0, 0.0],  # Pad (unused)
        [0.0, 0.0, 0.0],  # Pad (unused)
    ],
    dtype=np.float32,
)

N_LAYER_STACK_ROWS = 5
N_LAYER_STACK_COLS = 3
N_PORT_TYPES = 4  # [claw, gap, cpw, flux_line]

# Regex to strip unit suffixes like "20um", "5.3mm"
_UNIT_RE = re.compile(r"^([+-]?\d+\.?\d*)\s*(um|mm|nm|m|cm)?$", re.IGNORECASE)

_UNIT_TO_UM = {
    "nm": 1e-3,
    "um": 1.0,
    "mm": 1e3,
    "cm": 1e4,
    "m": 1e6,
    None: 1.0,  # dimensionless
}


def _parse_value(val: Any) -> float | None:
    """Parse a design parameter value to a float in µm (if it has units)."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        m = _UNIT_RE.match(val.strip())
        if m:
            num = float(m.group(1))
            unit = m.group(2)
            return num * _UNIT_TO_UM.get(unit.lower() if unit else None, 1.0)
    return None


# ---------------------------------------------------------------------------
# Vocabulary builder
# ---------------------------------------------------------------------------


def build_vocab(
    json_dir: str | Path | None = None,
    save_path: str | Path | None = None,
) -> dict[str, int]:
    """Build a parameter-key → integer vocabulary from enriched component JSONs.

    Args:
        json_dir: Directory containing enriched ``*.json`` component files.
            Defaults to ``qiskit_metal/dataset_output/`` inside the active
            virtual-environment site-packages.
        save_path: Optional path to persist the vocabulary as JSON.

    Returns:
        Mapping from parameter key string to integer id (0 is reserved for
        the ``<PAD>`` token).
    """
    if json_dir is None:
        try:
            import qiskit_metal

            json_dir = Path(qiskit_metal.__file__).parent / "dataset_output"
        except ImportError as exc:
            raise FileNotFoundError(
                "Cannot locate enriched component JSONs. Provide json_dir explicitly or install qiskit-metal."
            ) from exc
    json_dir = Path(json_dir)
    if not json_dir.is_dir():
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")

    keys: set[str] = set()
    for fp in sorted(json_dir.glob("*.json")):
        if fp.name == "all_components.json":
            continue
        try:
            with open(fp) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        for param in data.get("design_parameters", []):
            name = param.get("parameter_name", "")
            if name:
                keys.add(name)

    # Deterministic ordering: PAD at 0, then alphabetical
    vocab: dict[str, int] = {PAD_TOKEN: 0}
    for i, k in enumerate(sorted(keys), start=1):
        vocab[k] = i

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(vocab, f, indent=2)

    return vocab


# ---------------------------------------------------------------------------
# ComponentFeaturizer
# ---------------------------------------------------------------------------


class ComponentFeaturizer:
    """Extract raw features from a single qiskit-metal component.

    Args:
        vocab: Parameter-key vocabulary (from ``build_vocab``).
        json_dir: Path to enriched component JSONs (optional, auto-detected).
    """

    def __init__(
        self,
        vocab: dict[str, int],
        json_dir: str | Path | None = None,
    ):
        self.vocab = vocab
        self._json_cache: dict[str, dict] = {}

        if json_dir is None:
            try:
                import qiskit_metal

                json_dir = Path(qiskit_metal.__file__).parent / "dataset_output"
            except ImportError:
                json_dir = None
        self.json_dir = Path(json_dir) if json_dir else None

    # -- helpers --------------------------------------------------------

    def _load_json(self, component_type: str) -> dict | None:
        if component_type in self._json_cache:
            return self._json_cache[component_type]
        if self.json_dir is None:
            return None
        fp = self.json_dir / f"{component_type}.json"
        if not fp.exists():
            return None
        with open(fp) as f:
            data = json.load(f)
        self._json_cache[component_type] = data
        return data

    # -- public API -----------------------------------------------------

    def featurize(
        self,
        component_type: str,
        design_options: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert a component into a raw feature dictionary.

        Args:
            component_type: E.g. ``"TransmonCross"``.
            design_options: The design-parameter dict (values may include
                unit strings like ``"20um"``).

        Returns:
            Dict with keys:
            - ``layer_stack`` : ``np.ndarray`` of shape ``(5, 3)``
            - ``design_params`` : ``list[tuple[int, float]]`` — ``(key_id, value)``
            - ``area`` : ``float``
            - ``perimeter`` : ``float``
            - ``ports`` : ``np.ndarray`` of shape ``(4,)``
        """
        meta = self._load_json(component_type)

        # --- Layer stack (default for now) ---
        layer_stack = DEFAULT_LAYER_STACK.copy()

        # --- Design parameters → (key_id, parsed_value) ---
        design_params: list[tuple[int, float]] = []
        flat_opts = _flatten_dict(design_options)
        for key, val in flat_opts.items():
            parsed = _parse_value(val)
            if parsed is not None:
                key_id = self.vocab.get(key, self.vocab.get(PAD_TOKEN, 0))
                design_params.append((key_id, parsed))

        # --- Area & perimeter (from enriched JSON geometric equations) ---
        area = 0.0
        perimeter = 0.0
        if meta and "geometric_equations" in meta:
            geom = meta["geometric_equations"]
            area = _eval_geom_expr(geom.get("area", "0"), flat_opts)
            perimeter = _eval_geom_expr(geom.get("perimeter", "0"), flat_opts)

        # --- Ports vector ---
        ports = np.zeros(N_PORT_TYPES, dtype=np.float32)
        if meta and "pins" in meta:
            ports[0] = len(meta["pins"])  # approximate: count all pins as claw
        # Refine from connection_pads if available
        conn_pads = design_options.get("connection_pads", {})
        if isinstance(conn_pads, dict):
            for _pad_name, pad_opts in conn_pads.items():
                if isinstance(pad_opts, dict):
                    ctype = pad_opts.get("connector_type", "0")
                    try:
                        ctype = int(ctype)
                    except (ValueError, TypeError):
                        ctype = 0
                    if ctype == 0:
                        ports[0] += 1  # claw
                    elif ctype == 1:
                        ports[1] += 1  # gap

        return {
            "layer_stack": layer_stack,
            "design_params": design_params,
            "area": float(area),
            "perimeter": float(perimeter),
            "ports": ports,
        }


# ---------------------------------------------------------------------------
# CircuitGraphBuilder
# ---------------------------------------------------------------------------


class CircuitGraphBuilder:
    """Assemble per-component features into a Spektral ``Graph``.

    Args:
        vocab: Parameter-key vocabulary.
        k_max: Maximum number of design parameters per component (for padding).
        json_dir: Path to enriched component JSONs.
    """

    def __init__(
        self,
        vocab: dict[str, int],
        k_max: int = 20,
        json_dir: str | Path | None = None,
    ):
        self.vocab = vocab
        self.k_max = k_max
        self.featurizer = ComponentFeaturizer(vocab=vocab, json_dir=json_dir)

    def build(
        self,
        components: list[tuple[str, dict[str, Any]]],
        edges: list[tuple[int, int]],
        targets: np.ndarray | list[float] | None = None,
    ):
        """Build a ``spektral.data.Graph``.

        Args:
            components: List of ``(component_type, design_options)`` per node.
            edges: List of ``(src_idx, dst_idx)`` index pairs (0-based).
            targets: Optional Hamiltonian target vector.

        Returns:
            A ``spektral.data.Graph`` instance.
        """
        spektral = _ensure_spektral()
        n = len(components)

        # Collect raw features
        raw_features = [self.featurizer.featurize(ctype, opts) for ctype, opts in components]

        # --- Encode each node as a flat feature vector ---
        # Layout per row:
        #   [layer_stack_flat(15) | design_params(k_max*2) | area(1) | perimeter(1) | ports(4)]
        ls_len = N_LAYER_STACK_ROWS * N_LAYER_STACK_COLS  # 15
        dp_len = self.k_max * 2
        feat_dim = ls_len + dp_len + 1 + 1 + N_PORT_TYPES  # 15+40+1+1+4 = 61

        x = np.zeros((n, feat_dim), dtype=np.float32)
        for i, rf in enumerate(raw_features):
            offset = 0
            # layer stack
            x[i, offset : offset + ls_len] = rf["layer_stack"].flatten()
            offset += ls_len
            # design params (key_id, value) pairs, padded
            for j, (kid, val) in enumerate(rf["design_params"][: self.k_max]):
                x[i, offset + j * 2] = float(kid)
                x[i, offset + j * 2 + 1] = val
            offset += dp_len
            # area, perimeter
            x[i, offset] = rf["area"]
            x[i, offset + 1] = rf["perimeter"]
            offset += 2
            # ports
            x[i, offset : offset + N_PORT_TYPES] = rf["ports"]

        # --- Adjacency (symmetric, unweighted) ---
        if n > 0:
            row_idx = []
            col_idx = []
            for s, t in edges:
                if 0 <= s < n and 0 <= t < n:
                    row_idx.extend([s, t])
                    col_idx.extend([t, s])
            data = np.ones(len(row_idx), dtype=np.float32)
            a = sp.csr_matrix((data, (row_idx, col_idx)), shape=(n, n))
        else:
            a = sp.csr_matrix((n, n), dtype=np.float32)

        # --- Targets ---
        y = np.array(targets, dtype=np.float32) if targets is not None else None

        return spektral.data.Graph(x=x, a=a, y=y)


# ---------------------------------------------------------------------------
# SQuADDSGraphDataset
# ---------------------------------------------------------------------------


class SQuADDSGraphDataset:
    """Lightweight wrapper around a list of ``spektral.data.Graph`` objects.

    Provides train / validation / test splits and a Spektral-compatible
    ``read()`` interface.

    Args:
        graphs: Pre-built list of ``spektral.data.Graph`` objects.
        val_split: Fraction for validation set.
        test_split: Fraction for test set.
        seed: Random seed for splitting.
    """

    def __init__(
        self,
        graphs: list,
        val_split: float = 0.1,
        test_split: float = 0.1,
        seed: int = 42,
    ):
        _ensure_spektral()
        self.graphs = list(graphs)
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(self.graphs))

        n_test = max(1, int(len(idx) * test_split))
        n_val = max(1, int(len(idx) * val_split))

        self.idx_test = idx[:n_test]
        self.idx_val = idx[n_test : n_test + n_val]
        self.idx_train = idx[n_test + n_val :]

    # -- read() for compatibility with Spektral loaders --

    def read(self) -> list:
        return self.graphs

    @property
    def train_graphs(self) -> list:
        return [self.graphs[i] for i in self.idx_train]

    @property
    def val_graphs(self) -> list:
        return [self.graphs[i] for i in self.idx_val]

    @property
    def test_graphs(self) -> list:
        return [self.graphs[i] for i in self.idx_test]

    def __len__(self):
        return len(self.graphs)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten a nested dict, joining keys with *sep*."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _eval_geom_expr(expr: str, params: dict[str, Any]) -> float:
    """Safely evaluate a simple geometric expression with param substitution.

    Only supports basic arithmetic on known parameter names.  Falls back to 0
    on any failure.
    """
    if not expr or expr.strip() == "0":
        return 0.0

    # Build a safe namespace of parsed parameter values
    ns: dict[str, float] = {}
    for key, val in params.items():
        parsed = _parse_value(val)
        if parsed is not None:
            # Use the leaf name (after the last dot) as the variable name
            leaf = key.rsplit(".", 1)[-1]
            ns[leaf] = parsed

    try:
        # Restrict builtins for safety
        result = eval(expr, {"__builtins__": {}}, ns)  # noqa: S307
        return float(result)
    except Exception:
        return 0.0
