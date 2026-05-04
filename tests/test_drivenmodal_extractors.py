import json

import numpy as np
import pandas as pd

from squadds.simulations.drivenmodal.extractors import (
    coupled_hamiltonian_from_prepared_runs,
    hamiltonian_from_summary_mapping,
    pair_capacitances_fF_from_y_frame,
)


def test_pair_capacitances_fF_from_y_frame_qubit_claw_maps_maxwell_pairs():
    frame = pd.DataFrame(
        {
            "Y11": [1.0j * 1e-3, 2.0j * 1e-3],
            "Y12": [3.0j * 1e-4, 4.0j * 1e-4],
            "Y21": [3.0j * 1e-4, 4.0j * 1e-4],
            "Y22": [2.0j * 1e-3, 3.0j * 1e-3],
        },
        index=[5.0, 6.0],
    )
    caps = pair_capacitances_fF_from_y_frame(
        frame,
        system_kind="qubit_claw",
        extraction_freq_ghz=5.0,
    )
    assert set(caps.keys()) == {
        "cross_to_ground",
        "claw_to_ground",
        "cross_to_claw",
        "cross_to_cross",
        "claw_to_claw",
        "ground_to_ground",
    }
    assert all(np.isfinite(v) for v in caps.values())
    assert all(v >= 0.0 for v in caps.values())


def test_hamiltonian_from_summary_mapping_fills_hamiltonian_keys_with_nan_for_missing():
    row = hamiltonian_from_summary_mapping({"qubit_frequency_ghz": 4.0})
    assert row["qubit_frequency_ghz"] == 4.0
    assert np.isnan(row["chi_mhz"])


def test_coupled_hamiltonian_from_prepared_runs_prefers_resonator_band_summary(tmp_path):
    resonator = tmp_path / "res"
    bridge = tmp_path / "bridge"
    resonator.mkdir(parents=True)
    bridge.mkdir(parents=True)
    summary = {"extracted": {"qubit_frequency_ghz": 3.0, "chi_mhz": 0.1}}
    (resonator / "artifacts").mkdir(parents=True, exist_ok=True)
    (resonator / "artifacts" / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    prepared = {
        "bridge_band": {"manifest": {"run_dir": str(bridge)}},
        "resonator_band": {"manifest": {"run_dir": str(resonator)}},
    }
    out = coupled_hamiltonian_from_prepared_runs(prepared)
    assert out["qubit_frequency_ghz"] == 3.0
    assert out["chi_mhz"] == 0.1
