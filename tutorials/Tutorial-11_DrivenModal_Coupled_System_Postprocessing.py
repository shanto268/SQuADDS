# %% [markdown]
# # Tutorial 11: HFSS Driven-Modal Post-Processing for Qubit-Coupled Resonator Systems
#
# This tutorial reruns a SQuADDS qubit-coupled resonator geometry in
# **HFSS driven-modal** and post-processes the raw multiport response into the
# quantities we care about for the SQuADDS workflow:
#
# - cavity resonance in the qubit ground and excited states;
# - dispersive shift `chi`;
# - loaded linewidth `kappa`; and
# - back-calculated coupling rate `g`.
#
# The geometry comes from the existing SQuADDS database and can be switched
# between **quarter-wave** and **half-wave** by changing `RESONATOR_TYPE`.
#
# The driven-modal logic uses:
#
# - feedline input/output lumped ports on the coupled resonator component; and
# - a JJ lumped port on the transmon junction.
#
# We then terminate the JJ port with state-dependent inductances obtained from a
# transmon model built from the qubit capacitance matrix and bare junction
# inductance, and compare the extracted values against the existing SQuADDS
# pre-simulated reference row.

# %%
from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scqubits as scq
import skrf as rf
from qiskit_metal import Dict
from qiskit_metal.qlibrary.terminations.open_to_ground import OpenToGround
from qiskit_metal.qlibrary.tlines.straight_path import RouteStraight
from qiskit_metal.renderers.renderer_ansys.hfss_renderer import QHFSSRenderer
from scipy.signal import peak_widths

from squadds import SQuADDS_DB
from squadds.components.coupled_systems import QubitCavity
from squadds.core.json_utils import deserialize_json_like
from squadds.simulations.drivenmodal.artifacts import load_run_manifest, mark_stage_complete
from squadds.simulations.drivenmodal.coupled_postprocess import (
    calculate_chi_hz,
    calculate_g_from_chi,
    calculate_kappa_hz,
    calculate_loaded_q,
    terminate_port_y,
    y_to_s,
)
from squadds.simulations.drivenmodal.design import (
    apply_buffered_chip_bounds,
    apply_cryo_silicon_material_properties,
    connect_renderer_to_new_ansys_design,
    create_multiplanar_design,
    ensure_drivenmodal_setup,
    ensure_perfect_e_boundary,
    format_exception_for_console,
    render_drivenmodal_design,
    run_drivenmodal_sweep,
    safe_ansys_design_name,
    save_renderer_project,
    simplify_collinear_path_points,
    snapshot_boundary_assignments,
)
from squadds.simulations.drivenmodal.hfss_data import (
    parameter_dataframe_to_tensor,
    write_touchstone_from_dataframe,
)
from squadds.simulations.drivenmodal.hfss_runner import run_drivenmodal_request
from squadds.simulations.drivenmodal.models import (
    CoupledSystemDrivenModalRequest,
    DrivenModalArtifactPolicy,
    DrivenModalLayerStackSpec,
    DrivenModalSetupSpec,
    DrivenModalSweepSpec,
)
from squadds.simulations.drivenmodal.ports import build_coupled_system_port_specs, split_rendered_ports
from squadds.simulations.drivenmodal.qubit_admittance import jj_parallel_impedance
from squadds.simulations.utils import mesh_objects, string_to_float
from squadds.simulations.utils_physics import find_g_a_fq

try:
    from IPython.display import HTML
    from IPython.display import display as display
except ImportError:  # pragma: no cover - plain Python fallback for non-notebook execution

    def display(obj):
        print(obj)

    def HTML(value):
        return value


# %% [markdown]
# ## Runtime knobs
#
# Set `RESONATOR_TYPE = "quarter"` or `"half"` and rerun the notebook/script.
# `RUN_TAG` is the easiest way to generate a clean new AEDT design without
# changing the rest of the file.

# %%
RESONATOR_TYPE = "quarter"  # change to "half" and rerun for the half-wave flow
REFERENCE_INDEX = 0
RUN_TAG = "v6"
FORCE_RERUN = False
MAX_SOLVE_ATTEMPTS = 3
ROOT_COMPONENT_NAME = "qubit_cavity"
FEEDLINE_COMPONENT_NAME = "feedline"
RENDER_LAUNCHERS = False
FEEDLINE_OTG_COMPONENT_NAMES = ("feedline_top_otg", "feedline_bottom_otg")

DISCOVERY_SWEEP_PADDING_GHZ = 2.0
DISCOVERY_SWEEP_COUNT = 4001
DISCOVERY_MAX_DELTA_S = 0.01
DISCOVERY_MIN_CONVERGED = 3
MAX_DISCOVERY_ATTEMPTS = 4
DISCOVERY_CENTER_SHIFT_GHZ = DISCOVERY_SWEEP_PADDING_GHZ

FINAL_SWEEP_PADDING_GHZ = 0.5

LAYER_STACK = DrivenModalLayerStackSpec(
    preset="squadds_hfss_v1",
    chip_name="main",
    metal_thickness_um=0.2,
    substrate_thickness_um=750.0,
)
SETUP_TEMPLATE = DrivenModalSetupSpec(
    name="DrivenModalSetup",
    freq_ghz=6.0,
    max_delta_s=0.005,
    max_passes=20,
    min_passes=2,
    min_converged=7,
    pct_refinement=30,
    basis_order=-1,
)
SWEEP_TEMPLATE = DrivenModalSweepSpec(
    name="DrivenModalSweep",
    start_ghz=5.0,
    stop_ghz=6.0,
    count=22000,
    sweep_type="Interpolating",
    save_fields=False,
    interpolation_tol=0.005,
    interpolation_max_solutions=22000,
)
ARTIFACTS = DrivenModalArtifactPolicy(
    export_touchstone=True,
    export_y_parameters=True,
    export_capacitance_tables=True,
    checkpoint_after_stage=True,
    resume_existing=True,
)

RUNTIME_ROOT = Path("tutorials/runtime/drivenmodal_coupled_system")
CHECKPOINT_ROOT = RUNTIME_ROOT / "checkpoints"
HFSS_PROJECT_ROOT = RUNTIME_ROOT / "hfss_projects"
LOCAL_ANALYSIS_ROOT = RUNTIME_ROOT / "local_analysis"


# %% [markdown]
# ## Helpers


# %%
def ensure_runtime_dirs() -> None:
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    HFSS_PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    LOCAL_ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)


def stage_is_complete(manifest_path: Path, stage_name: str) -> bool:
    manifest = load_run_manifest(manifest_path)
    return manifest["stages"][stage_name]["status"] == "complete"


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def write_plotly_html(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn")


def _design_component_name(design, raw_component: Any) -> str:
    if isinstance(raw_component, str):
        return raw_component
    try:
        component = design._components[int(raw_component)]
    except Exception:
        return str(raw_component)
    return component.name


def save_qiskit_layout_plot(
    design,
    *,
    output_path: Path,
    title: str,
    port_mapping: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))

    poly_table = design.qgeometry.tables.get("poly")
    if poly_table is not None and len(poly_table) > 0:
        for _, row in poly_table.iterrows():
            geom = row["geometry"]
            x, y = geom.exterior.xy
            ax.fill(
                x,
                y,
                alpha=0.35 if not row["subtract"] else 0.0,
                color="tab:blue" if not row["subtract"] else "white",
                edgecolor="black",
                linewidth=0.6,
            )

    path_table = design.qgeometry.tables.get("path")
    if path_table is not None and len(path_table) > 0:
        for _, row in path_table.iterrows():
            geom = row["geometry"]
            x, y = geom.xy
            ax.plot(
                x,
                y,
                color="magenta" if not row["subtract"] else "gray",
                linewidth=max(float(row["width"]) * 100, 1.0),
            )

    for port_kind, raw_spec in port_mapping.items():
        component = raw_spec.get("component")
        pin = raw_spec.get("pin")
        if component not in design.components:
            continue
        pin_dict = getattr(design.components[component], "pins", {}).get(pin)
        if pin_dict is None:
            continue
        middle = np.asarray(pin_dict["middle"], dtype=float)
        ax.scatter(
            [middle[0]],
            [middle[1]],
            s=36,
            label=port_kind,
        )
        ax.annotate(port_kind, (middle[0], middle[1]), textcoords="offset points", xytext=(4, 4), fontsize=8)

    ax.set_aspect("equal")
    ax.set_xlabel("mm")
    ax.set_ylabel("mm")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def build_design_geometry_summary(design) -> dict[str, Any]:
    components_summary: list[dict[str, Any]] = []
    for name, component in design.components.items():
        try:
            bounds = component.qgeometry_bounds()
        except Exception:
            bounds = None
        components_summary.append(
            {
                "name": name,
                "class_name": type(component).__name__,
                "pins": sorted(getattr(component, "pins", {}).keys()),
                "bounds": bounds,
            }
        )

    qgeometry_summary: dict[str, list[dict[str, Any]]] = {}
    for table_name, table in design.qgeometry.tables.items():
        entries: list[dict[str, Any]] = []
        for _, row in table.iterrows():
            entry = {
                "component": _design_component_name(design, row["component"]),
                "name": row["name"],
                "subtract": bool(row.get("subtract", False)),
            }
            if "width" in row:
                entry["width"] = float(row["width"])
            if "fillet" in row and not pd.isna(row["fillet"]):
                entry["fillet"] = float(row["fillet"])
            entries.append(entry)
        qgeometry_summary[table_name] = entries

    return {
        "components": components_summary,
        "qgeometry": qgeometry_summary,
    }


def build_design_options_payload(row: pd.Series) -> dict[str, Any]:
    raw_design_options = row.get("design_options")
    if raw_design_options is not None and not pd.isna(raw_design_options):
        return deserialize_json_like(raw_design_options)

    if "design_options_qubit" in row and "design_options_cavity_claw" in row:
        qubit_options = deserialize_json_like(row["design_options_qubit"])
        cavity_split = deserialize_json_like(row["design_options_cavity_claw"])
        coupler_options = cavity_split.get("coupler_options") or cavity_split.get("cplr_opts") or {}
        cpw_options = cavity_split.get("cpw_options") or cavity_split.get("cpw_opts") or {}

        return {
            "qubit_options": qubit_options,
            "cavity_claw_options": {
                "coupler_type": row["coupler_type"],
                "coupler_options": coupler_options,
                "cpw_opts": {"left_options": cpw_options},
            },
        }

    raise KeyError("Could not resolve a QubitCavity-compatible design options payload from the reference row.")


def regularize_design_options_for_drivenmodal(design_options: dict[str, Any]) -> dict[str, Any]:
    normalized = deserialize_json_like(design_options)
    cavity_options = normalized["cavity_claw_options"]
    cavity_options["coupler_type"] = cavity_options["coupler_type"]
    cpw_key = "cpw_opts" if "cpw_opts" in cavity_options else "cpw_options"
    cpw_options = deserialize_json_like(cavity_options.setdefault(cpw_key, {}))
    cavity_options[cpw_key] = cpw_options
    left_options = deserialize_json_like(cpw_options.setdefault("left_options", {}))
    cpw_options["left_options"] = left_options
    if "lead" in left_options:
        left_options["lead"] = deserialize_json_like(left_options["lead"])
    else:
        left_options["lead"] = {}
    if "meander" in left_options:
        left_options["meander"] = deserialize_json_like(left_options["meander"])
    else:
        left_options["meander"] = {}
    return normalized


def build_coupled_seed_mesh_lengths(design) -> dict[str, dict[str, Any]]:
    candidate_objects: list[str] = []
    for table_name in ["path", "poly"]:
        table = design.qgeometry.tables.get(table_name)
        if table is None:
            continue
        for _, row in table.iterrows():
            if bool(row.get("subtract", False)):
                continue
            component_name = _design_component_name(design, row["component"])
            if component_name == ROOT_COMPONENT_NAME:
                continue
            candidate_objects.append(f"{row['name']}_{component_name}")
    return {"mesh_coupled_system": {"objects": sorted(set(candidate_objects)), "MaxLength": "7um"}}


def assign_render_layers(design, *, metal_layer: str | int) -> None:
    metal_layer_str = str(metal_layer)
    for name, component in design.components.items():
        if name == ROOT_COMPONENT_NAME:
            continue
        try:
            component.options.layer = metal_layer_str
        except Exception:
            continue


def build_render_selection(design, *, include_launchers: bool = RENDER_LAUNCHERS) -> list[str]:
    preferred_names = [
        FEEDLINE_COMPONENT_NAME,
        f"{ROOT_COMPONENT_NAME}_CLT_coupler",
        f"{ROOT_COMPONENT_NAME}_capn_coupler",
        f"{ROOT_COMPONENT_NAME}_NCap_coupler",
        f"{ROOT_COMPONENT_NAME}_left_cpw",
        f"{ROOT_COMPONENT_NAME}_right_cpw",
        f"{ROOT_COMPONENT_NAME}_xmon",
    ]
    preferred_names.extend(sorted(name for name in design.components if name.startswith(f"{ROOT_COMPONENT_NAME}_")))

    selection: list[str] = []
    seen: set[str] = set()
    for name in preferred_names:
        if name not in design.components or name in seen:
            continue
        component = design.components[name]
        try:
            bounds = component.qgeometry_bounds()
        except Exception:
            bounds = None
        if bounds is None:
            continue
        if tuple(bounds) == (0, 0, 0, 0):
            continue
        selection.append(name)
        seen.add(name)
    return selection


def save_render_debug_artifacts(
    *,
    design,
    artifacts_dir: Path,
    title: str,
    selection: list[str],
    port_mapping: dict[str, Any],
    port_list: list[tuple[str, str, float]],
    jj_to_port: list[tuple[str, str, float, bool]],
    open_pins: list[tuple[str, str]],
    chip_bounds: dict[str, float],
) -> None:
    dump_json(artifacts_dir / "qiskit_geometry_summary.json", build_design_geometry_summary(design))
    dump_json(
        artifacts_dir / "resolved_render_inputs.json",
        {
            "selection": selection,
            "port_mapping": port_mapping,
            "open_pins": open_pins,
            "port_list": port_list,
            "jj_to_port": jj_to_port,
            "chip_bounds": chip_bounds,
        },
    )
    save_qiskit_layout_plot(
        design,
        output_path=artifacts_dir / "qiskit_layout.png",
        title=title,
        port_mapping=port_mapping,
    )


def prepare_renderer_project(renderer: QHFSSRenderer, project_dir: Path, project_name: str) -> Path:
    """Create a fresh HFSS project and save it to an absolute AEDT path.

    This avoids the Windows/pyEPR project-path duplication bug triggered by
    passing ``project_path``/``project_name`` through ``QHFSSRenderer`` options.
    """
    project_dir = project_dir.resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    project_file = project_dir / f"{project_name}.aedt"
    renderer.start()
    renderer.new_ansys_project()
    renderer.connect_ansys()
    return save_renderer_project(renderer, project_file)


def reset_ansys_desktop_processes() -> None:
    """Best-effort hard reset of stale AEDT desktop sessions on Windows.

    Qiskit Metal / pyEPR can keep a just-finished project active inside the
    Desktop session even after ``disconnect_ansys()``. The next
    ``renderer.start()`` then reconnects into that stale project and eagerly
    tries to resolve a setup before callers can create the new one. For this
    tutorial workflow we prefer a clean desktop between attempts/runs.
    """
    if os.name != "nt":
        return
    try:
        subprocess.run(
            ["taskkill", "/IM", "ansysedt.exe", "/F"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return


def network_from_sweep(freqs_hz: np.ndarray, s_matrices: np.ndarray, *, z0_ohms: float = 50.0) -> rf.Network:
    frequency = rf.Frequency.from_f(freqs_hz, unit="hz")
    return rf.Network(frequency=frequency, s=s_matrices, z0=z0_ohms)


def write_network_touchstone(network: rf.Network, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    network.write_touchstone(filename=output_path.with_suffix("").as_posix())
    return output_path


def extract_notch_metrics(freqs_hz: np.ndarray, s21: np.ndarray) -> dict[str, float]:
    s21_mag_db = 20 * np.log10(np.clip(np.abs(s21), 1e-15, None))
    inverted = -s21_mag_db
    min_idx = int(np.argmax(inverted))
    widths, _heights, left_ips, right_ips = peak_widths(inverted, [min_idx], rel_height=0.5)
    if len(freqs_hz) < 2:
        raise ValueError("At least two frequency points are required to extract linewidth.")
    step_hz = float(np.mean(np.diff(freqs_hz)))
    fwhm_hz = float(widths[0] * step_hz)
    return {
        "f_res_hz": float(freqs_hz[min_idx]),
        "fwhm_hz": fwhm_hz,
        "left_index": float(left_ips[0]),
        "right_index": float(right_ips[0]),
        "min_s21_db": float(s21_mag_db[min_idx]),
        "resonance_at_sweep_edge": float(min_idx in {0, len(freqs_hz) - 1}),
    }


def build_local_analysis_method_result(
    ground_freq_hz: float,
    excited_freq_hz: float,
    reference: dict[str, float],
) -> dict[str, float]:
    chi_hz = calculate_chi_hz(ground_freq_hz, excited_freq_hz)
    g_rad_s = calculate_g_from_chi(
        f_r_hz=ground_freq_hz,
        f_q_hz=reference["f_q_hz"],
        chi_hz=chi_hz,
        alpha_hz=reference["alpha_hz"],
    )
    return {
        "f_ground_ghz": float(ground_freq_hz / 1e9),
        "f_excited_ghz": float(excited_freq_hz / 1e9),
        "chi_mhz": float(chi_hz / 1e6),
        "g_mhz": float(g_rad_s / (2 * np.pi * 1e6)),
    }


def build_shift_threshold_rows(sweep_df: pd.DataFrame, *, bin_hz: float) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for c_ff in sorted(pd.unique(sweep_df["C_fF"])):
        subset = sweep_df[sweep_df["C_fF"] == c_ff].copy()
        subset["abs_delta_hz"] = subset["delta_vs_open_mhz"].abs() * 1e6
        hit = subset[subset["abs_delta_hz"] >= bin_hz].sort_values("L_nH").head(1)
        if hit.empty:
            continue
        row = hit.iloc[0]
        rows.append(
            {
                "C_fF": float(c_ff),
                "first_visible_shift_L_nH": float(row["L_nH"]),
                "first_visible_shift_freq_ghz": float(row["residual_feature_ghz"]),
                "delta_hz": float(row["abs_delta_hz"]),
                "bin_hz": float(bin_hz),
            }
        )
    return rows


def generate_local_analysis_artifacts(summary: dict[str, Any]) -> dict[str, Any]:
    raw_touchstone_path = Path(summary["artifacts"]["raw_touchstone"])
    if not raw_touchstone_path.exists():
        raise FileNotFoundError(f"Raw touchstone not found at {raw_touchstone_path}")

    run_id = Path(summary["run_dir"]).name
    output_dir = LOCAL_ANALYSIS_ROOT / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_network = rf.Network(str(raw_touchstone_path))
    freqs_hz = raw_network.f
    freq_ghz = freqs_hz / 1e9
    y_raw = raw_network.y
    reference = summary["reference"]
    lj_ground = reference["lj_ground_h"]
    lj_excited = reference["lj_excited_h"]

    def term_network(terminated_port: int, zload, remaining_port_1: str, remaining_port_2: str) -> dict[str, Any]:
        y_term = terminate_port_y(y_raw, terminated_port=terminated_port, load_impedance_ohms=zload)
        s_term = y_to_s(y_term, z0_ohms=50.0)
        metrics = extract_notch_metrics(freqs_hz, s_term[:, 1, 0])
        s21 = s_term[:, 1, 0]
        mag_db = 20 * np.log10(np.maximum(np.abs(s21), 1e-16))
        phase_deg = np.unwrap(np.angle(s21)) * 180 / np.pi
        phase_grad = np.abs(np.gradient(phase_deg, freq_ghz))
        complex_grad = np.abs(np.gradient(s21, freq_ghz))
        mag_idx = int(np.argmin(mag_db))
        phase_idx = int(np.argmax(phase_grad))
        complex_idx = int(np.argmax(complex_grad))
        return {
            "metrics": metrics,
            "mag_db": mag_db,
            "phase_deg": phase_deg,
            "mag_min_freq_ghz": float(freq_ghz[mag_idx]),
            "phase_grad_peak_freq_ghz": float(freq_ghz[phase_idx]),
            "complex_grad_peak_freq_ghz": float(freq_ghz[complex_idx]),
            "notch_freq_ghz": float(metrics["f_res_hz"] / 1e9),
            "notch_min_db": float(metrics["min_s21_db"]),
            "edge_flag": bool(metrics["resonance_at_sweep_edge"]),
            "fwhm_hz": float(metrics["fwhm_hz"]),
            "phase_at_feature_deg": float(phase_deg[phase_idx]),
            "remaining_port_1": remaining_port_1,
            "remaining_port_2": remaining_port_2,
        }

    qubit_scenarios = {
        "jj_open": term_network(2, np.full_like(freqs_hz, np.inf, dtype=complex), "feedline_input", "feedline_output"),
        "jj_short": term_network(2, np.zeros_like(freqs_hz, dtype=complex), "feedline_input", "feedline_output"),
        "jj_lj_ground_pure_L": term_network(
            2,
            jj_parallel_impedance(freqs_hz, lj_h=lj_ground, cj_f=0.0, rj_ohms=math.inf),
            "feedline_input",
            "feedline_output",
        ),
        "jj_lj_excited_pure_L": term_network(
            2,
            jj_parallel_impedance(freqs_hz, lj_h=lj_excited, cj_f=0.0, rj_ohms=math.inf),
            "feedline_input",
            "feedline_output",
        ),
        "jj_lj_ground_physical": term_network(
            2,
            jj_parallel_impedance(freqs_hz, lj_h=lj_ground, cj_f=2e-15, rj_ohms=50_000.0),
            "feedline_input",
            "feedline_output",
        ),
        "jj_lj_excited_physical": term_network(
            2,
            jj_parallel_impedance(freqs_hz, lj_h=lj_excited, cj_f=2e-15, rj_ohms=50_000.0),
            "feedline_input",
            "feedline_output",
        ),
    }
    feedline_scenarios = {
        "feedline_output_open": term_network(
            1,
            np.full_like(freqs_hz, np.inf, dtype=complex),
            "feedline_input",
            "jj",
        ),
        "feedline_output_short": term_network(
            1,
            np.zeros_like(freqs_hz, dtype=complex),
            "feedline_input",
            "jj",
        ),
    }

    fig = go.Figure()
    for i in range(3):
        for j in range(3):
            fig.add_trace(
                go.Scatter(
                    x=freq_ghz,
                    y=20 * np.log10(np.maximum(np.abs(raw_network.s[:, i, j]), 1e-16)),
                    name=f"|S{i + 1}{j + 1}| dB",
                )
            )
    fig.update_layout(
        title="Raw 3-port overview (cryo 11.45)", xaxis_title="Frequency (GHz)", yaxis_title="Magnitude (dB)"
    )
    write_plotly_html(fig, output_dir / "raw_3port_overview.html")

    for name, scenarios in [
        ("qubit_port_terminations", qubit_scenarios),
        ("feedline_port_terminations", feedline_scenarios),
    ]:
        fig_full = go.Figure()
        fig_zoom = go.Figure()
        for scenario_name, data in scenarios.items():
            fig_full.add_trace(go.Scatter(x=freq_ghz, y=data["mag_db"], name=scenario_name))
            fig_zoom.add_trace(go.Scatter(x=freq_ghz, y=data["mag_db"], name=scenario_name))
        fig_full.update_layout(
            title=f"{name.replace('_', ' ').title()} full",
            xaxis_title="Frequency (GHz)",
            yaxis_title="|Transmission| (dB)",
        )
        fig_zoom.update_layout(
            title=f"{name.replace('_', ' ').title()} zoom",
            xaxis_title="Frequency (GHz)",
            yaxis_title="|Transmission| (dB)",
            xaxis_range=[8.72, 8.83],
        )
        write_plotly_html(fig_full, output_dir / f"{name}_full.html")
        write_plotly_html(fig_zoom, output_dir / f"{name}_zoom.html")

    fig = go.Figure()
    for scenario_name in [
        "jj_open",
        "jj_short",
        "jj_lj_ground_pure_L",
        "jj_lj_excited_pure_L",
        "jj_lj_ground_physical",
        "jj_lj_excited_physical",
    ]:
        fig.add_trace(go.Scatter(x=freq_ghz, y=qubit_scenarios[scenario_name]["mag_db"], name=scenario_name))
    fig.update_layout(
        title="JJ reference terminations around resonance",
        xaxis_title="Frequency (GHz)",
        yaxis_title="|S21| (dB)",
        xaxis_range=[8.74, 8.81],
    )
    write_plotly_html(fig, output_dir / "jj_reference_terminations.html")

    rows: list[dict[str, Any]] = []
    for scenario_name, data in qubit_scenarios.items():
        rows.append(
            {
                "group": "qubit_port",
                "scenario": scenario_name,
                "terminated_port": 2,
                "load_kind": scenario_name,
                "remaining_port_1": data["remaining_port_1"],
                "remaining_port_2": data["remaining_port_2"],
                "mag_min_freq_ghz": data["mag_min_freq_ghz"],
                "residual_min_freq_ghz": data["notch_freq_ghz"],
                "phase_grad_peak_freq_ghz": data["phase_grad_peak_freq_ghz"],
                "complex_grad_peak_freq_ghz": data["complex_grad_peak_freq_ghz"],
                "notch_freq_ghz": data["notch_freq_ghz"],
                "notch_min_db": data["notch_min_db"],
                "edge_flag": data["edge_flag"],
                "fwhm_hz": data["fwhm_hz"],
                "phase_at_feature_deg": data["phase_at_feature_deg"],
            }
        )
    for scenario_name, data in feedline_scenarios.items():
        rows.append(
            {
                "group": "feedline_port",
                "scenario": scenario_name,
                "terminated_port": 1,
                "load_kind": scenario_name,
                "remaining_port_1": data["remaining_port_1"],
                "remaining_port_2": data["remaining_port_2"],
                "mag_min_freq_ghz": data["mag_min_freq_ghz"],
                "residual_min_freq_ghz": data["notch_freq_ghz"],
                "phase_grad_peak_freq_ghz": data["phase_grad_peak_freq_ghz"],
                "complex_grad_peak_freq_ghz": data["complex_grad_peak_freq_ghz"],
                "notch_freq_ghz": data["notch_freq_ghz"],
                "notch_min_db": data["notch_min_db"],
                "edge_flag": data["edge_flag"],
                "fwhm_hz": data["fwhm_hz"],
                "phase_at_feature_deg": data["phase_at_feature_deg"],
            }
        )
    scenario_df = pd.DataFrame(rows)
    scenario_csv = output_dir / "scenario_summary.csv"
    scenario_df.to_csv(scenario_csv, index=False)

    baseline = []
    for scenario_name in [
        "jj_open",
        "jj_short",
        "jj_lj_ground_pure_L",
        "jj_lj_excited_pure_L",
        "jj_lj_ground_physical",
        "jj_lj_excited_physical",
    ]:
        data = qubit_scenarios[scenario_name]
        baseline.append(
            {
                "scenario": scenario_name,
                "residual_feature_hz": data["notch_freq_ghz"] * 1e9,
                "mag_min_hz": data["mag_min_freq_ghz"] * 1e9,
                "phase_grad_peak_hz": data["phase_grad_peak_freq_ghz"] * 1e9,
                "complex_grad_peak_hz": data["complex_grad_peak_freq_ghz"] * 1e9,
                "min_db": data["notch_min_db"],
            }
        )
    dump_json(output_dir / "jj_reference_scenarios.json", {"baseline": baseline})

    exploratory = {
        "notch_method_pure_L": build_local_analysis_method_result(
            qubit_scenarios["jj_lj_ground_pure_L"]["notch_freq_ghz"] * 1e9,
            qubit_scenarios["jj_lj_excited_pure_L"]["notch_freq_ghz"] * 1e9,
            reference,
        ),
        "phase_gradient_method_pure_L": build_local_analysis_method_result(
            qubit_scenarios["jj_lj_ground_pure_L"]["phase_grad_peak_freq_ghz"] * 1e9,
            qubit_scenarios["jj_lj_excited_pure_L"]["phase_grad_peak_freq_ghz"] * 1e9,
            reference,
        ),
        "notch_method_physical": build_local_analysis_method_result(
            qubit_scenarios["jj_lj_ground_physical"]["notch_freq_ghz"] * 1e9,
            qubit_scenarios["jj_lj_excited_physical"]["notch_freq_ghz"] * 1e9,
            reference,
        ),
    }
    exploratory_json = output_dir / "exploratory_postprocessing.json"
    dump_json(exploratory_json, exploratory)

    open_freq = qubit_scenarios["jj_open"]["notch_freq_ghz"]
    ljg_physical_freq = qubit_scenarios["jj_lj_ground_physical"]["notch_freq_ghz"]
    l_nh_values = np.concatenate([np.logspace(np.log10(0.1), np.log10(8.0), 80), np.linspace(8.0, 12.5, 91)]).astype(
        float
    )
    c_ff_values = np.array([0.0, 0.5, 1.0, 2.0, 4.0])
    sweep_rows = []
    for c_ff in c_ff_values:
        for l_nh in l_nh_values:
            zload = jj_parallel_impedance(freqs_hz, lj_h=l_nh * 1e-9, cj_f=c_ff * 1e-15, rj_ohms=50_000.0)
            data = term_network(2, zload, "feedline_input", "feedline_output")
            sweep_rows.append(
                {
                    "L_nH": float(l_nh),
                    "C_fF": float(c_ff),
                    "residual_feature_ghz": data["notch_freq_ghz"],
                    "delta_vs_open_mhz": (data["notch_freq_ghz"] - open_freq) * 1e3,
                    "delta_vs_ljg_khz": (data["notch_freq_ghz"] - ljg_physical_freq) * 1e6,
                    "min_db": data["notch_min_db"],
                }
            )
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_csv = output_dir / "jj_parallel_lc_sweep.csv"
    sweep_df.to_csv(sweep_csv, index=False)

    bin_hz = float(np.min(np.diff(freqs_hz)))
    threshold_rows = build_shift_threshold_rows(sweep_df, bin_hz=bin_hz)
    pd.DataFrame(threshold_rows).to_csv(output_dir / "jj_shift_thresholds.csv", index=False)

    fig = px.line(
        sweep_df,
        x="L_nH",
        y="residual_feature_ghz",
        color="C_fF",
        title="Cryo run: resonance vs JJ inductance and capacitance",
        labels={
            "L_nH": "JJ inductance (nH)",
            "residual_feature_ghz": "Feature frequency (GHz)",
            "C_fF": "JJ capacitance (fF)",
        },
    )
    fig.add_vline(x=lj_ground * 1e9, line_dash="dash", line_color="green")
    fig.add_vline(x=lj_excited * 1e9, line_dash="dash", line_color="orange")
    write_plotly_html(fig, output_dir / "jj_parallel_lc_shift.html")

    pivot = sweep_df.pivot(index="L_nH", columns="C_fF", values="residual_feature_ghz")
    fig = px.imshow(
        pivot,
        aspect="auto",
        origin="lower",
        labels={"x": "JJ capacitance (fF)", "y": "JJ inductance (nH)", "color": "Feature frequency (GHz)"},
        title="Cryo run: JJ parallel LC resonance heatmap",
    )
    write_plotly_html(fig, output_dir / "jj_parallel_lc_heatmap.html")

    physical_df = sweep_df[sweep_df["C_fF"].isin([0.5, 1.0, 2.0, 4.0])]
    fig = px.line(
        physical_df,
        x="L_nH",
        y="delta_vs_open_mhz",
        color="C_fF",
        title="Cryo run: resonance shift vs open JJ",
        labels={
            "L_nH": "JJ inductance (nH)",
            "delta_vs_open_mhz": "Shift from open (MHz)",
            "C_fF": "JJ capacitance (fF)",
        },
    )
    write_plotly_html(fig, output_dir / "jj_physical_window.html")
    physical_csv = output_dir / "jj_physical_window.csv"
    physical_df.to_csv(physical_csv, index=False)

    local_analysis = {
        "output_dir": str(output_dir),
        "artifacts": {
            "scenario_summary_csv": str(scenario_csv),
            "exploratory_postprocessing_json": str(exploratory_json),
            "raw_3port_overview_html": str(output_dir / "raw_3port_overview.html"),
            "jj_reference_terminations_html": str(output_dir / "jj_reference_terminations.html"),
            "qubit_port_terminations_full_html": str(output_dir / "qubit_port_terminations_full.html"),
            "qubit_port_terminations_zoom_html": str(output_dir / "qubit_port_terminations_zoom.html"),
            "feedline_port_terminations_full_html": str(output_dir / "feedline_port_terminations_full.html"),
            "feedline_port_terminations_zoom_html": str(output_dir / "feedline_port_terminations_zoom.html"),
            "jj_parallel_lc_sweep_csv": str(sweep_csv),
            "jj_shift_thresholds_csv": str(output_dir / "jj_shift_thresholds.csv"),
            "jj_parallel_lc_shift_html": str(output_dir / "jj_parallel_lc_shift.html"),
            "jj_parallel_lc_heatmap_html": str(output_dir / "jj_parallel_lc_heatmap.html"),
            "jj_physical_window_csv": str(physical_csv),
            "jj_physical_window_html": str(output_dir / "jj_physical_window.html"),
        },
        "reference_extracted": summary["extracted"],
        "exploratory_postprocessing": exploratory,
    }
    dump_json(output_dir / "local_analysis_summary.json", local_analysis)
    return local_analysis


def display_local_analysis_summary(local_analysis: dict[str, Any]) -> None:
    output_dir = Path(local_analysis["output_dir"]).resolve()
    scenario_df = pd.read_csv(local_analysis["artifacts"]["scenario_summary_csv"])
    exploratory_rows = pd.DataFrame(local_analysis["exploratory_postprocessing"]).T.reset_index()
    exploratory_rows = exploratory_rows.rename(columns={"index": "method"})

    display(exploratory_rows)
    display(scenario_df)

    iframe_specs = [
        ("JJ reference terminations", output_dir / "jj_reference_terminations.html"),
        ("JJ parallel LC shift", output_dir / "jj_parallel_lc_shift.html"),
        ("JJ parallel LC heatmap", output_dir / "jj_parallel_lc_heatmap.html"),
        ("Qubit-port termination zoom", output_dir / "qubit_port_terminations_zoom.html"),
    ]
    for title, html_path in iframe_specs:
        resolved_path = html_path.resolve()
        display(
            HTML(
                f"<h4>{title}</h4>"
                f"<p><a href='file://{resolved_path.as_posix()}' target='_blank'>{resolved_path}</a></p>"
                f"<iframe src='file://{resolved_path.as_posix()}' "
                "width='100%' height='520' style='border: 1px solid #ccc;'></iframe>"
            )
        )


def load_reference_row(resonator_type: str, index: int) -> pd.Series:
    db = SQuADDS_DB()
    db.select_system(["qubit", "cavity_claw"])
    db.select_qubit("TransmonCross")
    db.select_cavity_claw("RouteMeander")
    db.select_resonator_type(resonator_type)
    df = db.create_system_df()
    return df.iloc[index]


def normalize_qubit_options(qubit_options: dict[str, Any]) -> dict[str, Any]:
    return deserialize_json_like(qubit_options)


def get_bare_lj_h(qubit_options: dict[str, Any]) -> float:
    for key in ["aedt_hfss_inductance", "hfss_inductance", "aedt_q3d_inductance", "q3d_inductance"]:
        value = qubit_options.get(key)
        if value in (None, ""):
            continue
        if isinstance(value, str):
            numeric = float(string_to_float(value))
            if value.lower().endswith("nh"):
                return numeric * 1e-9
            return numeric
        return float(value)
    raise KeyError("No junction inductance key was found in the qubit design options.")


def compute_transmon_state_data(*, cross_to_ground_fF: float, cross_to_claw_fF: float, lj_h: float) -> dict[str, float]:
    scq.set_units("GHz")

    e_charge = 1.602e-19
    hbar = 1.054e-34
    c_sigma_f = (abs(cross_to_ground_fF) + abs(cross_to_claw_fF)) * 1e-15
    ej_ghz = ((hbar / 2 / e_charge) ** 2) / lj_h * 1.5092e24
    ec_ghz = e_charge**2 / (2 * c_sigma_f) * 1.5092e24

    transmon = scq.Transmon(EJ=ej_ghz, EC=ec_ghz, ng=0, ncut=35)
    evals, evecs = transmon.eigensys(evals_count=3)
    cos_phi = transmon.cos_phi_operator()
    cos_expectations = [float(np.real(np.vdot(evecs[:, index], cos_phi @ evecs[:, index]))) for index in [0, 1]]

    return {
        "ej_ghz": ej_ghz,
        "ec_ghz": ec_ghz,
        "f_q_hz": float(transmon.E01() * 1e9),
        "alpha_hz": float(transmon.anharmonicity() * 1e9),
        "lj_ground_h": float(lj_h / cos_expectations[0]),
        "lj_excited_h": float(lj_h / cos_expectations[1]),
    }


def build_reference_summary(row: pd.Series) -> dict[str, float]:
    qubit_options = normalize_qubit_options(row["design_options_qubit"])
    lj_h = get_bare_lj_h(qubit_options)

    res_factor = 2 if row["resonator_type"] == "half" else 4
    g_mhz, alpha_mhz, f_q_ghz = find_g_a_fq(
        abs(float(row["cross_to_claw"])) * 1e-15,
        abs(float(row["cross_to_ground"])) * 1e-15,
        float(row["cavity_frequency"]),
        lj_h,
        N=res_factor,
    )
    state_data = compute_transmon_state_data(
        cross_to_ground_fF=float(row["cross_to_ground"]),
        cross_to_claw_fF=float(row["cross_to_claw"]),
        lj_h=lj_h,
    )

    return {
        "cavity_frequency_ghz": float(row["cavity_frequency"]) / 1e9,
        "kappa_mhz": float(row["kappa"]) / 1e6,
        "g_mhz": float(g_mhz),
        "qubit_frequency_ghz": float(f_q_ghz),
        "anharmonicity_mhz": float(alpha_mhz),
        "lj_ground_h": state_data["lj_ground_h"],
        "lj_excited_h": state_data["lj_excited_h"],
        "f_q_hz": state_data["f_q_hz"],
        "alpha_hz": state_data["alpha_hz"],
    }


def build_setup_and_sweep(
    *,
    center_frequency_ghz: float,
    sweep_padding_ghz: float,
    sweep_count: int,
    max_delta_s: float,
    min_converged: int,
) -> tuple[DrivenModalSetupSpec, DrivenModalSweepSpec]:
    return (
        DrivenModalSetupSpec(
            name=SETUP_TEMPLATE.name,
            freq_ghz=center_frequency_ghz,
            max_delta_s=max_delta_s,
            max_passes=SETUP_TEMPLATE.max_passes,
            min_passes=SETUP_TEMPLATE.min_passes,
            min_converged=min_converged,
            pct_refinement=SETUP_TEMPLATE.pct_refinement,
            basis_order=SETUP_TEMPLATE.basis_order,
        ),
        DrivenModalSweepSpec(
            name=SWEEP_TEMPLATE.name,
            start_ghz=max(1.0, center_frequency_ghz - sweep_padding_ghz),
            stop_ghz=center_frequency_ghz + sweep_padding_ghz,
            count=sweep_count,
            sweep_type=SWEEP_TEMPLATE.sweep_type,
            save_fields=SWEEP_TEMPLATE.save_fields,
            interpolation_tol=SWEEP_TEMPLATE.interpolation_tol,
            interpolation_max_solutions=SWEEP_TEMPLATE.interpolation_max_solutions,
        ),
    )


def build_discovery_setup_and_sweep(row: pd.Series) -> tuple[DrivenModalSetupSpec, DrivenModalSweepSpec]:
    cavity_frequency_ghz = float(row["cavity_frequency"]) / 1e9
    return build_setup_and_sweep(
        center_frequency_ghz=cavity_frequency_ghz,
        sweep_padding_ghz=DISCOVERY_SWEEP_PADDING_GHZ,
        sweep_count=DISCOVERY_SWEEP_COUNT,
        max_delta_s=DISCOVERY_MAX_DELTA_S,
        min_converged=DISCOVERY_MIN_CONVERGED,
    )


def build_final_setup_and_sweep(center_frequency_ghz: float) -> tuple[DrivenModalSetupSpec, DrivenModalSweepSpec]:
    return build_setup_and_sweep(
        center_frequency_ghz=center_frequency_ghz,
        sweep_padding_ghz=FINAL_SWEEP_PADDING_GHZ,
        sweep_count=SWEEP_TEMPLATE.count,
        max_delta_s=SETUP_TEMPLATE.max_delta_s,
        min_converged=SETUP_TEMPLATE.min_converged,
    )


def build_shifted_discovery_setup_and_sweep(
    center_frequency_ghz: float,
) -> tuple[DrivenModalSetupSpec, DrivenModalSweepSpec]:
    return build_setup_and_sweep(
        center_frequency_ghz=center_frequency_ghz,
        sweep_padding_ghz=DISCOVERY_SWEEP_PADDING_GHZ,
        sweep_count=DISCOVERY_SWEEP_COUNT,
        max_delta_s=DISCOVERY_MAX_DELTA_S,
        min_converged=DISCOVERY_MIN_CONVERGED,
    )


def build_request(
    row: pd.Series,
    *,
    setup: DrivenModalSetupSpec,
    sweep: DrivenModalSweepSpec,
    run_suffix: str,
) -> CoupledSystemDrivenModalRequest:
    run_id = f"tutorial11-{RESONATOR_TYPE}-{REFERENCE_INDEX:03d}-{RUN_TAG}-{run_suffix}"
    design_options = regularize_design_options_for_drivenmodal(build_design_options_payload(row))
    return CoupledSystemDrivenModalRequest(
        resonator_type=f"{RESONATOR_TYPE}_wave",
        design_payload={
            "design_options": design_options,
            "coupler_type": row["coupler_type"],
            "port_mapping": {
                "feedline_input": {"component": FEEDLINE_COMPONENT_NAME, "pin": "start"},
                "feedline_output": {"component": FEEDLINE_COMPONENT_NAME, "pin": "end"},
                "jj": {
                    "component": f"{ROOT_COMPONENT_NAME}_xmon",
                    "pin": "rect_jj",
                    "metadata": {"hfss_target": "junction", "draw_inductor": False},
                },
            },
        },
        layer_stack=LAYER_STACK,
        setup=setup,
        sweep=sweep,
        artifacts=ARTIFACTS,
        metadata={"run_id": run_id},
    )


def build_qiskit_cavity_options(request: CoupledSystemDrivenModalRequest) -> Dict:
    """Reuse the dataset's coupled-system payload with the QubitCavity builder."""
    design_options = deserialize_json_like(request.design_payload["design_options"])
    cavity_options = Dict(deserialize_json_like(design_options["cavity_claw_options"]))
    cpw_key = "cpw_opts" if "cpw_opts" in cavity_options else "cpw_options"
    cpw_options = Dict(deserialize_json_like(cavity_options[cpw_key]))
    left_options = Dict(deserialize_json_like(cpw_options["left_options"]))
    cpw_options["left_options"] = left_options
    cavity_options[cpw_key] = cpw_options
    cavity_options["coupler_type"] = request.design_payload["coupler_type"]

    qubit_options = Dict(deserialize_json_like(design_options["qubit_options"]))
    connection_pads = Dict(deserialize_json_like(qubit_options.get("connection_pads", {})))
    readout_pad = Dict(deserialize_json_like(connection_pads.get("readout", {})))
    if readout_pad:
        if float(string_to_float(readout_pad.get("claw_cpw_length", "0um"))) == 0.0:
            readout_pad["claw_cpw_length"] = "40um"
        # Keep the connector stem electrically continuous with the resonator CPW
        # instead of trusting any stale fixed width embedded in the payload.
        readout_pad["claw_cpw_width"] = left_options["trace_width"]
        connection_pads["readout"] = readout_pad
        qubit_options["connection_pads"] = connection_pads
    qubit_options["pos_y"] = "0um"

    return Dict(
        chip=request.layer_stack.chip_name,
        qubit_options=qubit_options,
        cavity_claw_options=cavity_options,
    )


def get_feedline_trace_dimensions(request: CoupledSystemDrivenModalRequest) -> tuple[str, str]:
    design_options = deserialize_json_like(request.design_payload["design_options"])
    cavity_options = deserialize_json_like(design_options["cavity_claw_options"])
    cpw_key = "cpw_opts" if "cpw_opts" in cavity_options else "cpw_options"
    left_options = deserialize_json_like(deserialize_json_like(cavity_options[cpw_key])["left_options"])
    return left_options["trace_width"], left_options["trace_gap"]


def create_explicit_feedline_anchors(design, *, trace_width: str, trace_gap: str) -> None:
    OpenToGround(
        design,
        FEEDLINE_OTG_COMPONENT_NAMES[0],
        options=Dict(pos_x="0mm", pos_y="2.1mm", orientation="-90"),
    )
    OpenToGround(
        design,
        FEEDLINE_OTG_COMPONENT_NAMES[1],
        options=Dict(pos_x="0mm", pos_y="-1.9mm", orientation="90"),
    )
    RouteStraight(
        design,
        FEEDLINE_COMPONENT_NAME,
        options=Dict(
            pin_inputs=Dict(
                start_pin=Dict(component=FEEDLINE_OTG_COMPONENT_NAMES[0], pin="open"),
                end_pin=Dict(component=FEEDLINE_OTG_COMPONENT_NAMES[1], pin="open"),
            ),
            trace_width=trace_width,
            trace_gap=trace_gap,
        ),
    )


def build_coupled_design(request: CoupledSystemDrivenModalRequest, layer_stack_csv: Path):
    design, csv_path = create_multiplanar_design(
        layer_stack=request.layer_stack,
        layer_stack_path=layer_stack_csv,
        enable_renderers=True,
    )
    QubitCavity(
        design,
        ROOT_COMPONENT_NAME,
        options=build_qiskit_cavity_options(request),
    )
    trace_width, trace_gap = get_feedline_trace_dimensions(request)
    create_explicit_feedline_anchors(
        design,
        trace_width=trace_width,
        trace_gap=trace_gap,
    )
    assign_render_layers(design, metal_layer=request.layer_stack.metal_layer)
    design.rebuild()
    simplify_collinear_path_points(design)
    return design, csv_path


def compare_metrics(extracted: dict[str, float], reference: dict[str, float]) -> pd.DataFrame:
    rows = []
    for key in ["cavity_frequency_ghz", "kappa_mhz", "g_mhz", "qubit_frequency_ghz", "anharmonicity_mhz"]:
        ref_value = reference[key]
        ext_value = extracted[key]
        error_pct = np.nan
        if ref_value != 0:
            error_pct = 100.0 * (ext_value - ref_value) / ref_value
        rows.append(
            {
                "quantity": key,
                "drivenmodal": ext_value,
                "reference": ref_value,
                "percent_error": error_pct,
            }
        )
    rows.append(
        {"quantity": "chi_mhz", "drivenmodal": extracted["chi_mhz"], "reference": np.nan, "percent_error": np.nan}
    )
    return pd.DataFrame(rows)


def build_postprocessing_warning(ground_metrics: dict[str, float], excited_metrics: dict[str, float]) -> str | None:
    if ground_metrics["resonance_at_sweep_edge"] or excited_metrics["resonance_at_sweep_edge"]:
        return (
            "Loaded resonance remained on the sweep boundary. The HFSS solve completed, but the "
            "current driven-modal sweep window does not bracket a full notch for linewidth extraction."
        )
    if ground_metrics["fwhm_hz"] <= 0:
        return (
            "Loaded resonance linewidth could not be extracted from the sampled |S21| trace. "
            "The notch is too narrow or too shallow for the current FWHM-based estimator."
        )
    return None


def select_resonance_for_followup(summary: dict[str, Any]) -> float | None:
    extracted_frequency = summary["extracted"]["cavity_frequency_ghz"]
    if extracted_frequency is not None and not np.isnan(extracted_frequency):
        return float(extracted_frequency)

    ground_metrics = summary["ground_metrics"]
    if ground_metrics["resonance_at_sweep_edge"]:
        return None

    f_res_hz = ground_metrics["f_res_hz"]
    if f_res_hz is None or np.isnan(f_res_hz):
        return None
    return float(f_res_hz) / 1e9


def discovery_boundary_direction(
    summary: dict[str, Any],
    sweep: DrivenModalSweepSpec,
) -> str | None:
    ground_metrics = summary["ground_metrics"]
    if not ground_metrics["resonance_at_sweep_edge"]:
        return None

    f_res_hz = ground_metrics["f_res_hz"]
    if f_res_hz is None or np.isnan(f_res_hz):
        return None
    f_res_ghz = f_res_hz / 1e9
    if abs(f_res_ghz - sweep.start_ghz) <= 1e-6:
        return "lower"
    if abs(f_res_ghz - sweep.stop_ghz) <= 1e-6:
        return "upper"
    return None


def run_coupled_demo(request: CoupledSystemDrivenModalRequest, reference: dict[str, float]) -> dict[str, Any]:
    prepared = run_drivenmodal_request(request, checkpoint_dir=CHECKPOINT_ROOT)
    run_dir = Path(prepared["manifest"]["run_dir"])
    manifest_path = run_dir / "manifest.json"
    artifacts_dir = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    s_pickle = artifacts_dir / "s_parameters.pkl"
    y_pickle = artifacts_dir / "y_parameters.pkl"
    z_pickle = artifacts_dir / "z_parameters.pkl"
    s3p_path = artifacts_dir / "raw_coupled_system.s3p"
    ground_s2p_path = artifacts_dir / "loaded_ground.s2p"
    excited_s2p_path = artifacts_dir / "loaded_excited.s2p"
    summary_path = artifacts_dir / "summary.json"

    if stage_is_complete(manifest_path, "postprocessed") and not FORCE_RERUN and summary_path.exists():
        print(f"[{request.metadata['run_id']}] Reusing checkpointed postprocessed outputs from {run_dir}")
        summary = json.loads(summary_path.read_text())
        if "local_analysis" not in summary:
            summary["local_analysis"] = generate_local_analysis_artifacts(summary)
            dump_json(summary_path, summary)
        return summary

    if (
        stage_is_complete(manifest_path, "artifacts_exported")
        and not FORCE_RERUN
        and s_pickle.exists()
        and y_pickle.exists()
    ):
        print(f"[{request.metadata['run_id']}] Loading checkpointed solver artifacts from {run_dir}")
        s_df = pd.read_pickle(s_pickle)
        y_df = pd.read_pickle(y_pickle)
    else:
        base_project_dir = HFSS_PROJECT_ROOT / request.metadata["run_id"]
        base_project_dir.mkdir(parents=True, exist_ok=True)
        dump_json(artifacts_dir / "resolved_layer_stack.json", {"rows": prepared["layer_stack"]})
        port_specs = build_coupled_system_port_specs(request.design_payload)
        port_list, jj_to_port = split_rendered_ports(port_specs)
        open_pins: list[tuple[str, str]] = []
        last_error: BaseException | None = None

        for attempt in range(1, MAX_SOLVE_ATTEMPTS + 1):
            attempt_label = f"attempt-{attempt:02d}"
            attempt_id = f"{request.metadata['run_id']}-{attempt_label}"
            attempt_project_dir = base_project_dir / attempt_label
            attempt_project_dir.mkdir(parents=True, exist_ok=True)
            attempt_log_path = artifacts_dir / f"solver_{attempt_label}.json"
            ansys_design_name = safe_ansys_design_name(attempt_id)
            renderer = None
            try:
                design, layer_stack_csv = build_coupled_design(request, artifacts_dir / "layer_stack.csv")
                selection = build_render_selection(design, include_launchers=RENDER_LAUNCHERS)
                chip_bounds = apply_buffered_chip_bounds(
                    design,
                    selection=selection,
                    chip_name=request.layer_stack.chip_name,
                )
                save_render_debug_artifacts(
                    design=design,
                    artifacts_dir=artifacts_dir,
                    title=f"Tutorial 11 {RESONATOR_TYPE}-wave Qiskit Metal layout",
                    selection=selection,
                    port_mapping=request.design_payload["port_mapping"],
                    port_list=port_list,
                    jj_to_port=jj_to_port,
                    open_pins=open_pins,
                    chip_bounds=chip_bounds,
                )
                renderer = QHFSSRenderer(
                    design,
                    initiate=False,
                    options=Dict(),
                )
                project_file = prepare_renderer_project(renderer, attempt_project_dir, attempt_id)
                connect_renderer_to_new_ansys_design(
                    renderer,
                    ansys_design_name,
                    "drivenmodal",
                )
                renderer.clean_active_design()
                cryo_material = apply_cryo_silicon_material_properties(renderer)
                dump_json(artifacts_dir / "material_properties.json", cryo_material)

                render_drivenmodal_design(
                    renderer,
                    selection=selection,
                    open_pins=open_pins,
                    port_list=port_list or None,
                    jj_to_port=jj_to_port or None,
                    box_plus_buffer=False,
                )
                assigned_perf_e = ensure_perfect_e_boundary(
                    renderer,
                    getattr(renderer, "assign_perfE", []),
                    boundary_name="PerfE_Metal",
                )
                boundary_snapshot = snapshot_boundary_assignments(renderer)
                save_renderer_project(renderer, project_file)
                hfss_objects = sorted(getattr(renderer.pinfo.design.modeler, "object_names", []))
                dump_json(
                    artifacts_dir / f"hfss_render_{attempt_label}.json",
                    {
                        "attempt_label": attempt_label,
                        "ansys_design_name": ansys_design_name,
                        "project_file": str(project_file),
                        "hfss_objects": hfss_objects,
                        "assign_perfE": assigned_perf_e,
                        "boundaries": boundary_snapshot,
                    },
                )
                try:
                    renderer.save_screenshot(path=str(artifacts_dir / f"hfss_render_{attempt_label}.png"), show=False)
                except Exception as exc:
                    dump_json(
                        artifacts_dir / f"hfss_render_{attempt_label}_screenshot_error.json",
                        {"error": format_exception_for_console(exc)},
                    )
                mark_stage_complete(manifest_path, "rendered")

                if hasattr(renderer, "options"):
                    renderer.options.max_mesh_length_port = "7um"
                mesh_lengths = build_coupled_seed_mesh_lengths(design)
                modeler = renderer.pinfo.design.modeler
                mesh_error = None
                try:
                    mesh_objects(modeler, mesh_lengths)
                except Exception as exc:
                    mesh_error = format_exception_for_console(exc)
                dump_json(
                    artifacts_dir / "mesh_lengths.json",
                    {"targets": mesh_lengths, "error": mesh_error},
                )

                setup = ensure_drivenmodal_setup(renderer, **request.setup.to_renderer_kwargs())
                save_renderer_project(renderer, project_file)
                mark_stage_complete(manifest_path, "setup_created")

                run_drivenmodal_sweep(
                    renderer,
                    setup,
                    setup_name=request.setup.name,
                    **request.sweep.to_renderer_kwargs(),
                )
                save_renderer_project(renderer, project_file)
                mark_stage_complete(manifest_path, "sweep_completed")

                s_df, y_df, z_df = renderer.get_all_Pparms_matrices(matrix_size=3)
                s_df.to_pickle(s_pickle)
                y_df.to_pickle(y_pickle)
                z_df.to_pickle(z_pickle)
                write_touchstone_from_dataframe(s_df, matrix_size=3, output_path=s3p_path)
                dump_json(
                    artifacts_dir / "solver_artifacts.json",
                    {
                        "raw_touchstone": str(s3p_path),
                        "s_pickle": str(s_pickle),
                        "y_pickle": str(y_pickle),
                        "z_pickle": str(z_pickle),
                        "layer_stack_csv": str(layer_stack_csv),
                        "project_dir": str(attempt_project_dir),
                        "project_file": str(project_file),
                        "ansys_design_name": ansys_design_name,
                        "attempt_label": attempt_label,
                        "render_launchers": RENDER_LAUNCHERS,
                    },
                )
                dump_json(
                    attempt_log_path,
                    {
                        "status": "success",
                        "attempt_label": attempt_label,
                        "project_dir": str(attempt_project_dir),
                        "project_file": str(project_file),
                        "ansys_design_name": ansys_design_name,
                    },
                )
                mark_stage_complete(manifest_path, "artifacts_exported")
                break
            except Exception as exc:
                last_error = exc
                dump_json(
                    attempt_log_path,
                    {
                        "status": "failed",
                        "attempt_label": attempt_label,
                        "project_dir": str(attempt_project_dir),
                        "ansys_design_name": ansys_design_name,
                        "error": format_exception_for_console(exc),
                    },
                )
                print(
                    f"[{request.metadata['run_id']}] HFSS solve {attempt}/{MAX_SOLVE_ATTEMPTS} failed: "
                    f"{format_exception_for_console(exc)}"
                )
                if attempt == MAX_SOLVE_ATTEMPTS:
                    raise
                print(f"[{request.metadata['run_id']}] Retrying with a fresh internal HFSS design...")
            finally:
                if renderer is not None:
                    try:
                        renderer.disconnect_ansys()
                    except Exception as exc:  # pragma: no cover - best effort cleanup on the HFSS machine
                        print(
                            f"[{request.metadata['run_id']}] Warning while disconnecting Ansys: "
                            f"{format_exception_for_console(exc)}"
                        )
                reset_ansys_desktop_processes()
        else:  # pragma: no cover - defensive guard for analyzers that exit the loop unexpectedly
            if last_error is not None:
                raise last_error

    freqs_hz, y_matrices = parameter_dataframe_to_tensor(y_df, matrix_size=3, parameter_prefix="Y")
    ground_load = 1j * 2 * np.pi * freqs_hz * reference["lj_ground_h"]
    excited_load = 1j * 2 * np.pi * freqs_hz * reference["lj_excited_h"]

    y_ground = terminate_port_y(y_matrices, terminated_port=2, load_impedance_ohms=ground_load)
    y_excited = terminate_port_y(y_matrices, terminated_port=2, load_impedance_ohms=excited_load)
    s_ground = y_to_s(y_ground, z0_ohms=50.0)
    s_excited = y_to_s(y_excited, z0_ohms=50.0)

    ground_network = network_from_sweep(freqs_hz, s_ground)
    excited_network = network_from_sweep(freqs_hz, s_excited)
    write_network_touchstone(ground_network, ground_s2p_path)
    write_network_touchstone(excited_network, excited_s2p_path)

    ground_metrics = extract_notch_metrics(freqs_hz, s_ground[:, 1, 0])
    excited_metrics = extract_notch_metrics(freqs_hz, s_excited[:, 1, 0])
    postprocessing_warning = build_postprocessing_warning(ground_metrics, excited_metrics)
    extracted = {
        "cavity_frequency_ghz": np.nan,
        "kappa_mhz": np.nan,
        "g_mhz": np.nan,
        "qubit_frequency_ghz": reference["f_q_hz"] / 1e9,
        "anharmonicity_mhz": reference["alpha_hz"] / 1e6,
        "chi_mhz": np.nan,
    }
    if postprocessing_warning is None:
        chi_hz = calculate_chi_hz(ground_metrics["f_res_hz"], excited_metrics["f_res_hz"])
        loaded_q = calculate_loaded_q(f_res_hz=ground_metrics["f_res_hz"], fwhm_hz=ground_metrics["fwhm_hz"])
        kappa_hz = calculate_kappa_hz(f_res_hz=ground_metrics["f_res_hz"], loaded_q=loaded_q)
        g_rad_s = calculate_g_from_chi(
            f_r_hz=ground_metrics["f_res_hz"],
            f_q_hz=reference["f_q_hz"],
            chi_hz=chi_hz,
            alpha_hz=reference["alpha_hz"],
        )
        extracted.update(
            {
                "cavity_frequency_ghz": ground_metrics["f_res_hz"] / 1e9,
                "kappa_mhz": kappa_hz / 1e6,
                "g_mhz": g_rad_s / (2 * np.pi * 1e6),
                "chi_mhz": chi_hz / 1e6,
            }
        )
    comparison_df = compare_metrics(extracted, reference)
    comparison_df.to_csv(artifacts_dir / "comparison.csv", index=False)

    summary = {
        "run_dir": str(run_dir),
        "reference": reference,
        "extracted": extracted,
        "ground_metrics": ground_metrics,
        "excited_metrics": excited_metrics,
        "comparison_rows": comparison_df.to_dict(orient="records"),
        "postprocessing_warning": postprocessing_warning,
        "artifacts": {
            "raw_touchstone": str(s3p_path),
            "loaded_ground_touchstone": str(ground_s2p_path),
            "loaded_excited_touchstone": str(excited_s2p_path),
            "comparison_csv": str(artifacts_dir / "comparison.csv"),
        },
    }
    summary["local_analysis"] = generate_local_analysis_artifacts(summary)
    dump_json(summary_path, summary)
    mark_stage_complete(manifest_path, "postprocessed")
    return summary


# %% [markdown]
# ## Load the reference geometry from SQuADDS


# %%
def main() -> None:
    ensure_runtime_dirs()

    reference_row = load_reference_row(RESONATOR_TYPE, REFERENCE_INDEX)
    reference_summary = build_reference_summary(reference_row)
    discovery_setup, discovery_sweep = build_discovery_setup_and_sweep(reference_row)
    discovery_center_ghz = discovery_setup.freq_ghz

    print("Selected resonator type:", RESONATOR_TYPE)
    print("Coupler type:", reference_row["coupler_type"])
    print("Reference design options:")
    initial_request = build_request(
        reference_row,
        setup=discovery_setup,
        sweep=discovery_sweep,
        run_suffix="discovery-01",
    )
    print(json.dumps(deserialize_json_like(initial_request.design_payload["design_options"]), indent=2))

    discovery_result = None
    discovery_frequency_ghz = None
    for attempt in range(1, MAX_DISCOVERY_ATTEMPTS + 1):
        discovery_setup, discovery_sweep = build_shifted_discovery_setup_and_sweep(discovery_center_ghz)
        discovery_request = build_request(
            reference_row,
            setup=discovery_setup,
            sweep=discovery_sweep,
            run_suffix=f"discovery-{attempt:02d}",
        )
        print(
            f"\nRunning discovery sweep attempt {attempt}/{MAX_DISCOVERY_ATTEMPTS} "
            f"from {discovery_sweep.start_ghz:.6f} to {discovery_sweep.stop_ghz:.6f} GHz..."
        )
        discovery_result = run_coupled_demo(discovery_request, reference_summary)
        discovery_df = pd.DataFrame(discovery_result["comparison_rows"])
        display(discovery_df)

        discovery_frequency_ghz = select_resonance_for_followup(discovery_result)
        if discovery_frequency_ghz is not None:
            break

        direction = discovery_boundary_direction(discovery_result, discovery_sweep)
        if direction is None:
            break

        if direction == "lower":
            discovery_center_ghz = max(
                1.0 + DISCOVERY_SWEEP_PADDING_GHZ,
                discovery_center_ghz - DISCOVERY_CENTER_SHIFT_GHZ,
            )
        else:
            discovery_center_ghz += DISCOVERY_CENTER_SHIFT_GHZ
        print(
            "Discovery notch remained on the "
            f"{direction} sweep boundary; shifting the discovery center to {discovery_center_ghz:.6f} GHz."
        )

    if discovery_frequency_ghz is None:
        coupled_result = discovery_result
        print("Discovery sweeps did not bracket a cavity resonance well enough for the final high-resolution sweep.")
    else:
        print(f"Discovery sweep located cavity resonance near {discovery_frequency_ghz:.6f} GHz")
        final_setup, final_sweep = build_final_setup_and_sweep(discovery_frequency_ghz)
        final_request = build_request(
            reference_row,
            setup=final_setup,
            sweep=final_sweep,
            run_suffix="final",
        )
        print("Running final high-resolution sweep...")
        coupled_result = run_coupled_demo(final_request, reference_summary)
        comparison_df = pd.DataFrame(coupled_result["comparison_rows"])
        display(comparison_df)

    ground_touchstone = rf.Network(coupled_result["artifacts"]["loaded_ground_touchstone"])
    excited_touchstone = rf.Network(coupled_result["artifacts"]["loaded_excited_touchstone"])

    plt.figure(figsize=(10, 4))
    plt.plot(ground_touchstone.f / 1e9, 20 * np.log10(np.abs(ground_touchstone.s[:, 1, 0])), label="ground-loaded S21")
    plt.plot(
        excited_touchstone.f / 1e9,
        20 * np.log10(np.abs(excited_touchstone.s[:, 1, 0])),
        label="excited-loaded S21",
    )
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("|S21| (dB)")
    plt.title(f"Driven-modal loaded responses ({RESONATOR_TYPE}-wave)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

    run_dir = Path(coupled_result["run_dir"])
    layer_stack_df = pd.read_csv(run_dir / "artifacts" / "layer_stack.csv")
    display(layer_stack_df)
    display_local_analysis_summary(coupled_result["local_analysis"])

    print("Run directory:", run_dir)
    print("Artifacts:")
    for name, value in coupled_result["artifacts"].items():
        print(f"  - {name}: {value}")
    print("Local analysis:")
    for name, value in coupled_result["local_analysis"]["artifacts"].items():
        print(f"  - {name}: {value}")


# %% [markdown]
# ## Dataset and API outlook
#
# The long-term SQuADDS driven-modal dataset layout for coupled-system runs is
# expected to include:
#
# - a compact summary record in `SQuADDS_DB` with:
#   - geometry identifiers,
#   - explicit layer-stack values,
#   - setup and sweep metadata,
#   - extracted `f_r`, `chi`, `kappa`, `g`, and provenance; and
# - heavy sidecar artifacts containing:
#   - raw `.s3p` touchstones,
#   - ground/excited loaded `.s2p` touchstones,
#   - dense Y-parameter tables,
#   - checkpoint manifests, and
#   - postprocessing summaries.
#
# The API direction is:
#
# - `CoupledSystemDrivenModalRequest(...)` for the geometry + setup contract;
# - `AnsysSimulator.run_drivenmodal(request)` or the internal driven-modal runner
#   to initialize checkpoint state and artifact directories; and
# - reusable postprocessing helpers that take the raw multiport response,
#   terminate the JJ port with state-dependent inductances, and recover the
#   SQuADDS-facing quantities without re-running HFSS.


if __name__ == "__main__":
    main()
