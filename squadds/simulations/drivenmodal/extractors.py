"""Load driven-modal comparison values from exported solver artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from squadds.simulations.drivenmodal.capacitance import (
    capacitance_matrix_from_y,
    maxwell_capacitance_dataframe,
)
from squadds.simulations.drivenmodal.hfss_data import parameter_dataframe_to_tensor
from squadds.simulations.drivenmodal.workflows import (
    HAMILTONIAN_KEYS,
    NCAP_CAPACITANCE_KEYS,
    QUBIT_CLAW_CAPACITANCE_KEYS,
)


def _nearest_frequency_index(freqs_hz: np.ndarray, target_hz: float) -> int:
    freqs = np.asarray(freqs_hz, dtype=float)
    if freqs.size == 0:
        raise ValueError("freqs_hz must be non-empty.")
    return int(np.argmin(np.abs(freqs - target_hz)))


def _pair_fF_from_maxwell(maxwell: pd.DataFrame, row: str, col: str) -> float:
    return abs(float(maxwell.loc[row, col])) * 1e15


def _pair_capacitances_from_maxwell(maxwell: pd.DataFrame, *, system_kind: str) -> dict[str, float]:
    gnd = "ground"
    if system_kind == "qubit_claw":
        cross = "cross"
        claw = "claw"
        return {
            "cross_to_ground": _pair_fF_from_maxwell(maxwell, cross, gnd),
            "claw_to_ground": _pair_fF_from_maxwell(maxwell, claw, gnd),
            "cross_to_claw": _pair_fF_from_maxwell(maxwell, cross, claw),
            "cross_to_cross": _pair_fF_from_maxwell(maxwell, cross, cross),
            "claw_to_claw": _pair_fF_from_maxwell(maxwell, claw, claw),
            "ground_to_ground": _pair_fF_from_maxwell(maxwell, gnd, gnd),
        }
    if system_kind == "ncap":
        top = "top"
        bottom = "bottom"
        return {
            "top_to_top": _pair_fF_from_maxwell(maxwell, top, top),
            "top_to_bottom": _pair_fF_from_maxwell(maxwell, top, bottom),
            "top_to_ground": _pair_fF_from_maxwell(maxwell, top, gnd),
            "bottom_to_bottom": _pair_fF_from_maxwell(maxwell, bottom, bottom),
            "bottom_to_ground": _pair_fF_from_maxwell(maxwell, bottom, gnd),
            "ground_to_ground": _pair_fF_from_maxwell(maxwell, gnd, gnd),
        }
    raise ValueError("system_kind must be 'qubit_claw' or 'ncap'.")


def pair_capacitances_fF_from_y_frame(
    y_frame: pd.DataFrame,
    *,
    system_kind: str,
    extraction_freq_ghz: float,
) -> dict[str, float]:
    """Convert a Y-parameter sweep table into the same six pair labels used by Q3D rows."""
    if extraction_freq_ghz <= 0:
        raise ValueError("extraction_freq_ghz must be positive.")
    expected = QUBIT_CLAW_CAPACITANCE_KEYS if system_kind == "qubit_claw" else NCAP_CAPACITANCE_KEYS
    if system_kind not in ("qubit_claw", "ncap"):
        raise ValueError("system_kind must be 'qubit_claw' or 'ncap'.")
    freqs_hz, y_matrices = parameter_dataframe_to_tensor(
        y_frame,
        matrix_size=2,
        parameter_prefix="Y",
    )
    target_hz = extraction_freq_ghz * 1e9
    idx = _nearest_frequency_index(freqs_hz, target_hz)
    f_hz = float(freqs_hz[idx])
    y_matrix = y_matrices[idx]
    c_active = capacitance_matrix_from_y(f_hz, y_matrix)
    node_names = ["cross", "claw"] if system_kind == "qubit_claw" else ["top", "bottom"]
    maxwell = maxwell_capacitance_dataframe(c_active, node_names=node_names)
    pairs = _pair_capacitances_from_maxwell(maxwell, system_kind=system_kind)
    for key in expected:
        if key not in pairs:
            raise KeyError(f"Missing extracted key {key}.")
    return pairs


def pair_capacitances_fF_from_run_dir(
    run_dir: str | Path,
    *,
    system_kind: str,
    extraction_freq_ghz: float,
) -> dict[str, float]:
    """Load ``artifacts/y_parameters.pkl`` and return pair capacitances in fF."""
    y_path = Path(run_dir) / "artifacts" / "y_parameters.pkl"
    if not y_path.is_file():
        raise FileNotFoundError(
            f"Missing Y-parameter export at {y_path}. "
            "Run the Ansys driven-modal export so HFSS writes y_parameters.pkl under artifacts/."
        )
    y_frame = pd.read_pickle(y_path)
    return pair_capacitances_fF_from_y_frame(
        y_frame,
        system_kind=system_kind,
        extraction_freq_ghz=extraction_freq_ghz,
    )


def hamiltonian_from_summary_mapping(extracted: Mapping[str, Any] | None) -> dict[str, float]:
    """Normalize a post-processing ``extracted`` record to ``HAMILTONIAN_KEYS``."""
    if extracted is None:
        raise ValueError("extracted must not be None.")

    def as_float(key: str) -> float:
        raw = extracted.get(key)
        if raw is None:
            return float("nan")
        return float(raw)

    return {key: as_float(key) for key in HAMILTONIAN_KEYS}


def hamiltonian_from_summary_json(summary_path: str | Path) -> dict[str, float]:
    """Read ``artifacts/summary.json`` written by coupled driven-modal post-processing."""
    path = Path(summary_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing summary JSON at {path}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    extracted = payload.get("extracted")
    return hamiltonian_from_summary_mapping(extracted)


def coupled_hamiltonian_from_prepared_runs(prepared_runs: Mapping[str, Any]) -> dict[str, float]:
    """Load extracted Hamiltonian metrics from the first available band ``summary.json``."""
    band_preferences = ("resonator_band", "qubit_band", "bridge_band")
    tried: list[str] = []
    for band_name in band_preferences:
        if band_name not in prepared_runs:
            continue
        manifest = prepared_runs[band_name].get("manifest")
        if manifest is None:
            raise KeyError(f"prepared_runs[{band_name!r}] is missing 'manifest'.")
        run_dir = Path(manifest["run_dir"])
        summary_path = run_dir / "artifacts" / "summary.json"
        tried.append(str(summary_path))
        if summary_path.is_file():
            return hamiltonian_from_summary_json(summary_path)
    detail = "\n  - ".join(tried) if tried else "(no segmented band entries were present)."
    raise FileNotFoundError(
        "No coupled summary.json found in expected locations. Searched:\n  - "
        + detail
        + "\nRun coupled post-processing so it writes summary.json next to y_parameters.pkl "
        "(see tutorials/Tutorial-11_DrivenModal_Coupled_System_Postprocessing.py)."
    )
