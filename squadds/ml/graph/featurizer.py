"""
Data pipeline utilities for converting SQuADDS layouts into graph objects.

Provides:
- ``ComponentFeaturizer``: Extracts raw heterogeneous features from a single component.
- ``CircuitGraphBuilder``: Assembles per-component features into a ``spektral.data.Graph``.
- ``SQuADDSGraphDataset``: A lightweight wrapper for train/val/test splits.
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
# Lazy Spektral import
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

# Default silicon/aluminium layer-stack: [(thickness_um, permittivity), ...]
# Ordered bottom-to-top.
DEFAULT_LAYER_STACK = [
    (350.0, 11.45),  # Si substrate
    (0.25, 0.0),  # Al metal (250 nm → 0.25 µm)
]

N_PORT_TYPES = 5  # [connector, mwave, o2g, RLC, LumpedPort]

# Regex to strip unit suffixes like "20um", "5.3mm"
_UNIT_RE = re.compile(r"^([+-]?\d+\.?\d*)\s*(um|mm|nm|m|cm)?$", re.IGNORECASE)

_UNIT_TO_UM = {
    "nm": 1e-3,
    "um": 1.0,
    "mm": 1e3,
    "cm": 1e4,
    "m": 1e6,
    None: 1.0,
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
    extra_jsons: list[str | Path] | None = None,
    save_path: str | Path | None = None,
) -> dict[str, int]:
    """Build a parameter-key → integer vocabulary from enriched component JSONs.

    Args:
        json_dir: Directory with enriched ``*.json`` files (defaults to
            ``qiskit_metal/dataset_output/``).
        extra_jsons: Additional JSON file paths to include (e.g. CavityClaw.json).
        save_path: Optional path to persist the vocabulary as JSON.

    Returns:
        ``{param_key: int_id}``.  Index 0 is reserved for ``<PAD>``.
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

    keys: set[str] = set()

    # Scan standard directory
    if json_dir.is_dir():
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

    # Scan extra JSONs
    for fp in extra_jsons or []:
        fp = Path(fp)
        if not fp.exists():
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

    The featurizer loads ALL design params from the component's enriched JSON
    (defaults) and overrides the swept values from the dataset row.  This
    ensures every component always produces the full parameter set.

    Args:
        vocab: Parameter-key vocabulary (from ``build_vocab``).
        json_dir: Path to enriched component JSONs.
        extra_json_dir: Additional directory for custom component JSONs
            (e.g. ``CavityClaw.json``).
    """

    def __init__(
        self,
        vocab: dict[str, int],
        json_dir: str | Path | None = None,
        extra_json_dir: str | Path | None = None,
    ):
        self.vocab = vocab
        self._json_cache: dict[str, dict] = {}

        self._json_dirs: list[Path] = []
        if json_dir is not None:
            self._json_dirs.append(Path(json_dir))
        else:
            try:
                import qiskit_metal

                self._json_dirs.append(Path(qiskit_metal.__file__).parent / "dataset_output")
            except ImportError:
                pass
        if extra_json_dir is not None:
            self._json_dirs.append(Path(extra_json_dir))

    def _load_json(self, component_type: str) -> dict | None:
        if component_type in self._json_cache:
            return self._json_cache[component_type]
        for d in self._json_dirs:
            fp = d / f"{component_type}.json"
            if fp.exists():
                with open(fp) as f:
                    data = json.load(f)
                self._json_cache[component_type] = data
                return data
        return None

    def featurize(
        self,
        component_type: str,
        design_overrides: dict[str, Any],
        layer_stack: list[tuple[float, float]] | None = None,
        ports_vector: list[float | int] | None = None,
    ) -> dict[str, Any]:
        """Convert a component into a raw feature dictionary.

        Args:
            component_type: E.g. ``"TransmonCross"`` or ``"CavityClaw"``.
            design_overrides: Dict of swept/overridden param values.  Merged
                on top of the JSON defaults.
            layer_stack: Ordered bottom-to-top list of ``(thickness_um, permittivity)``.
                Defaults to ``DEFAULT_LAYER_STACK``.
            ports_vector: 5-element list ``[connector, mwave, o2g, RLC, LumpedPort]``.
                Defaults to a zero vector.

        Returns:
            ``{ layer_stack, design_params, area, perimeter, ports }``
        """
        meta = self._load_json(component_type)

        # --- Layer stack ---
        ls = layer_stack if layer_stack is not None else DEFAULT_LAYER_STACK

        # --- ALL design params: defaults from JSON + overrides ---
        all_params: dict[str, Any] = {}
        if meta:
            for param in meta.get("design_parameters", []):
                pname = param.get("parameter_name", "")
                if pname:
                    all_params[pname] = param.get("default_value", "0")

        # Override with swept/user values
        flat_overrides = _flatten_dict(design_overrides)
        all_params.update(flat_overrides)

        # Convert to (key_id, value) pairs
        design_params: list[tuple[int, float]] = []
        for key, val in all_params.items():
            parsed = _parse_value(val)
            if parsed is not None:
                key_id = self.vocab.get(key, self.vocab.get(PAD_TOKEN, 0))
                design_params.append((key_id, parsed))

        # --- Area & perimeter ---
        area = 0.0
        perimeter = 0.0
        if meta and "geometric_equations" in meta:
            geom = meta["geometric_equations"]
            area = _eval_geom_expr(geom.get("area", "0"), all_params)
            perimeter = _eval_geom_expr(geom.get("perimeter", "0"), all_params)

        # --- Ports vector ---
        if ports_vector is not None:
            ports = np.array(ports_vector, dtype=np.float32)
            if len(ports) < N_PORT_TYPES:
                ports = np.pad(ports, (0, N_PORT_TYPES - len(ports)))
        else:
            ports = np.zeros(N_PORT_TYPES, dtype=np.float32)

        return {
            "layer_stack": ls,
            "design_params": design_params,
            "area": float(area),
            "perimeter": float(perimeter),
            "ports": ports[:N_PORT_TYPES],
        }


# ---------------------------------------------------------------------------
# CircuitGraphBuilder
# ---------------------------------------------------------------------------


class CircuitGraphBuilder:
    """Assemble per-component features into a Spektral ``Graph``.

    Args:
        vocab: Parameter-key vocabulary.
        k_max: Maximum design parameters per component (for padding).
        n_ls: Number of layer-stack rows (padded).
        json_dir: Path to enriched component JSONs.
        extra_json_dir: Additional directory for custom component JSONs.
    """

    def __init__(
        self,
        vocab: dict[str, int],
        k_max: int = 20,
        n_ls: int = 5,
        json_dir: str | Path | None = None,
        extra_json_dir: str | Path | None = None,
    ):
        self.vocab = vocab
        self.k_max = k_max
        self.n_ls = n_ls
        self.featurizer = ComponentFeaturizer(
            vocab=vocab,
            json_dir=json_dir,
            extra_json_dir=extra_json_dir,
        )

    def build(
        self,
        components: list[dict[str, Any]],
        edges: list[tuple[int, int]],
        targets: np.ndarray | list[float] | None = None,
    ):
        """Build a ``spektral.data.Graph``.

        Args:
            components: List of dicts, each containing::

                {
                    "type": "TransmonCross",
                    "design_overrides": {"cross_length": "310um", ...},
                    "layer_stack": [(350.0, 11.45), (0.25, 0.0)],  # optional
                    "ports_vector": [1, 0, 2, 0, 1],              # optional
                }

            edges: ``(src, dst)`` node-index pairs.
            targets: Optional Hamiltonian target vector.

        Returns:
            A ``spektral.data.Graph``.
        """
        spektral = _ensure_spektral()
        n = len(components)

        raw_features = []
        for comp in components:
            rf = self.featurizer.featurize(
                component_type=comp["type"],
                design_overrides=comp.get("design_overrides", {}),
                layer_stack=comp.get("layer_stack"),
                ports_vector=comp.get("ports_vector"),
            )
            raw_features.append(rf)

        # --- Flat feature vector per node ---
        # Layout: [layer_stack(n_ls*2) | design_params(k_max*2) | area(1) | perim(1) | ports(5)]
        ls_len = self.n_ls * 2
        dp_len = self.k_max * 2
        feat_dim = ls_len + dp_len + 2 + N_PORT_TYPES

        x = np.zeros((n, feat_dim), dtype=np.float32)
        for i, rf in enumerate(raw_features):
            offset = 0

            # Layer stack → pad/truncate to n_ls rows of 2 cols
            ls_arr = np.zeros((self.n_ls, 2), dtype=np.float32)
            for j, (thick, perm) in enumerate(rf["layer_stack"][: self.n_ls]):
                ls_arr[j] = [thick, perm]
            x[i, offset : offset + ls_len] = ls_arr.flatten()
            offset += ls_len

            # Design params (key_id, value) pairs, padded
            for j, (kid, val) in enumerate(rf["design_params"][: self.k_max]):
                x[i, offset + j * 2] = float(kid)
                x[i, offset + j * 2 + 1] = val
            offset += dp_len

            # Area, perimeter
            x[i, offset] = rf["area"]
            x[i, offset + 1] = rf["perimeter"]
            offset += 2

            # Ports
            x[i, offset : offset + N_PORT_TYPES] = rf["ports"]

        # --- Adjacency (symmetric, unweighted) ---
        if n > 0 and edges:
            row_idx, col_idx = [], []
            for s, t in edges:
                if 0 <= s < n and 0 <= t < n:
                    row_idx.extend([s, t])
                    col_idx.extend([t, s])
            data = np.ones(len(row_idx), dtype=np.float32)
            a = sp.csr_matrix((data, (row_idx, col_idx)), shape=(n, n))
        else:
            a = sp.csr_matrix((n, n), dtype=np.float32)

        y = np.array(targets, dtype=np.float32) if targets is not None else None

        return spektral.data.Graph(x=x, a=a, y=y)


# ---------------------------------------------------------------------------
# SQuADDSGraphDataset
# ---------------------------------------------------------------------------


class SQuADDSGraphDataset:
    """Lightweight wrapper around a list of ``spektral.data.Graph`` objects.

    Provides train / validation / test splits.

    Args:
        graphs: Pre-built ``Graph`` list.
        val_split: Fraction for validation.
        test_split: Fraction for test.
        seed: Random seed.
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
    """Safely evaluate a geometric expression with param substitution."""
    if not expr or expr.strip() == "0":
        return 0.0

    ns: dict[str, float] = {}
    for key, val in params.items():
        parsed = _parse_value(val)
        if parsed is not None:
            leaf = key.rsplit(".", 1)[-1]
            ns[leaf] = parsed

    try:
        result = eval(expr, {"__builtins__": {}}, ns)  # noqa: S307
        return float(result)
    except Exception:
        return 0.0
