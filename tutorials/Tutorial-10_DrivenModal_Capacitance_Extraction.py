# %% [markdown]
# # Tutorial 10: Driven-Modal Capacitance Extraction
#
# In this tutorial we will use SQuADDS to set up a driven-modal HFSS
# capacitance extraction for two familiar objects:
#
# 1. a `TransmonCross` qubit-claw capacitor, and
# 2. an `NCap` interdigital coupler.
#
# The goal is not to teach every Ansys scripting detail. SQuADDS should carry
# that weight. Instead, this notebook teaches the SQuADDS contract for this
# workflow: a validated database row becomes a portable driven-modal request,
# the request renders the same physical network into HFSS, and the resulting
# Y-parameters are converted back into the capacitance quantities users already
# know from the Q3D datasets.
#
# By the end, you should know how to:
#
# - pull a Q3D reference row from the SQuADDS database,
# - build the equivalent driven-modal request,
# - compare driven-modal capacitances against the Q3D reference, and
# - read the Maxwell capacitance matrix without double-counting terms.
#
# <div class="admonition note">
# <p class="admonition-title">Why driven modal for capacitance?</p>
# <p>Q3D remains the direct electrostatic capacitance solver in SQuADDS. This
# tutorial shows a complementary route: solve a small-signal HFSS driven-modal
# network, export its admittance matrix, and recover capacitance from the
# imaginary admittance. The value of this route is that it uses the same
# port-based network representation as the later Hamiltonian workflow.</p>
# </div>

# %% [markdown]
# ## Imports
#
# We will use the regular SQuADDS database API to choose designs, then the
# driven-modal workflow helpers to build the HFSS requests.

# %%
import os
from pathlib import Path

import pandas as pd

from squadds import SQuADDS_DB
from squadds.simulations.drivenmodal import (
    build_capacitance_request,
    capacitance_comparison_table,
    capacitance_reference_summary,
    default_capacitance_setup,
    default_capacitance_sweep,
    maxwell_matrix_interpretation,
    pair_capacitances_fF_from_run_dir,
)

try:
    from IPython.display import display
except ImportError:  # pragma: no cover

    def display(obj):
        print(obj)


# %% [markdown]
# ## Choose Reference Designs
#
# We start from already-validated SQuADDS rows. This is useful because the Q3D
# capacitance matrix in each row gives us a trusted comparison target.
#
# `see_dataset(...)` returns the raw table behind one SQuADDS dataset. The
# important inputs are:
#
# - `component`: the high-level object family, for example `"qubit"` or
#   `"coupler"`.
# - `component_name`: the concrete Qiskit Metal component stored in the dataset.
# - `data_type`: the simulated observable we want from the database. Here it is
#   `"cap_matrix"` because we want Q3D capacitances for comparison.
#
# A larger validation study would loop over many rows. Here we choose fixed rows
# so that every reader sees the same geometry and the same reference numbers.

# %%
db = SQuADDS_DB()

qubit_df = db.see_dataset(
    component="qubit",
    component_name="TransmonCross",
    data_type="cap_matrix",
)
ncap_df = db.see_dataset(
    component="coupler",
    component_name="NCap",
    data_type="cap_matrix",
)

qubit_row = qubit_df.iloc[3]
ncap_row = ncap_df.iloc[1]

display(qubit_row[["design_options", "cross_to_ground", "cross_to_claw"]])
display(ncap_row[["design_options", "top_to_bottom", "top_to_ground"]])


# %% [markdown]
# ## Declare the Driven-Modal Solve
#
# A driven-modal capacitance extraction is a frequency-domain multiport solve.
# SQuADDS wraps the bookkeeping in a `CapacitanceExtractionRequest`.
#
# Behind the scenes, `build_capacitance_request(...)` does four things:
#
# - It reads the selected row's `design_options` and keeps the geometry tied to
#   the database reference.
# - It declares which Qiskit Metal pins become HFSS lumped ports. For the
#   qubit-claw case these are the transmon cross/JJ node and the readout claw;
#   for the NCap case these are the top and bottom capacitor terminals.
#   Each port is a 50 ohm lumped port in HFSS; the impedance is a reference
#   normalization for the exported network, not a claim that the capacitor is
#   physically connected to a 50 ohm resistor.
# - It attaches a SQuADDS layer-stack preset where metal is rendered as PEC and
#   the substrate metadata matches the Ansys renderer expectations. This is also
#   where the generated run records the concrete layer-stack rows used by HFSS.
# - It stores the HFSS adaptive setup, frequency sweep, and artifact policy in a
#   serializable object that can be sent to the Windows Ansys machine.
#
# The only choices we make here are the ones a user should care about:
#
# - which geometry row to render,
# - which physical ports define the capacitance network,
# - which setup/sweep settings to use, and
# - where the reproducibility bundle should be stored.
#
# `default_capacitance_setup(freq_ghz=5.0)` is the adaptive HFSS setup. The
# frequency is not "the extracted capacitance frequency"; it is the frequency at
# which HFSS builds the adaptive mesh. The default uses `max_delta_s=0.005`,
# `min_converged=5`, and mixed basis order because those settings gave stable
# agreement with the corresponding Q3D workflow.
#
# `default_capacitance_sweep(...)` is the frequency grid where HFSS exports the
# network data. The 1-10 GHz interpolating sweep is broad enough to catch
# frequency dependence while still being cheap enough for a validation pass.
#
# These are driven-modal settings. They are not inherited from Q3D or
# eigenmode. In Ansys terms we ask for a driven-modal adaptive setup named
# `DrivenModalSetup`, use mixed basis order (`basis_order=-1`), require five
# converged adaptive passes, and export an interpolating sweep with interpolation
# tolerance `0.005`. The explicit tables below are included so the simulation
# contract is visible before anything is sent to Ansys.

# %%
setup = default_capacitance_setup(freq_ghz=5.0)
sweep = default_capacitance_sweep(start_ghz=1.0, stop_ghz=10.0, count=400)

qubit_request = build_capacitance_request(
    qubit_row,
    system_kind="qubit_claw",
    run_id="tutorial10-qubit-claw",
    setup=setup,
    sweep=sweep,
)
ncap_request = build_capacitance_request(
    ncap_row,
    system_kind="ncap",
    run_id="tutorial10-ncap",
    setup=setup,
    sweep=sweep,
)

display(pd.DataFrame([setup.to_renderer_kwargs()]).T.rename(columns={0: "HFSS setup value"}))
display(pd.DataFrame([sweep.to_renderer_kwargs()]).T.rename(columns={0: "HFSS sweep value"}))


# %%
port_table = pd.DataFrame(
    [
        {"request": label, "port": port_name, **port_spec}
        for label, request in {"qubit_claw": qubit_request, "ncap": ncap_request}.items()
        for port_name, port_spec in request.design_payload["port_mapping"].items()
    ]
)
display(port_table)


# %%
layer_stack_table = pd.DataFrame(
    [
        {"request": "qubit_claw", **qubit_request.layer_stack.to_dict()},
        {"request": "ncap", **ncap_request.layer_stack.to_dict()},
    ]
)
display(layer_stack_table)


# %% [markdown]
# ## Run the Simulation
#
# The request object is the portable contract. The same cell runs on a laptop,
# in CI, and on the Windows Ansys workstation. When `SQUADDS_RUN_ANSYS=1` is set
# in the environment, SQuADDS renders the design into HFSS and writes a
# reproducibility bundle for each request. When the variable is unset, the cell
# still builds the request objects and the rest of the notebook shows the target
# reference tables without contacting Ansys.
#
# `run_drivenmodal_request(...)` is intentionally the only Ansys-facing API used
# here. The request object is the source of truth for the Ansys executor: the
# executor renders the Qiskit Metal geometry, creates the HFSS driven-modal
# setup, assigns the lumped ports from the request, exports Touchstone/Y data,
# and writes checkpoint files. If a run is interrupted, the artifact policy can
# resume from completed stages instead of starting from zero.
#
# The generated bundle is part of the workflow, not an afterthought:
#
# - `request.json` records the geometry, ports, setup, sweep, and layer-stack
#   contract.
# - `layer_stack.csv` records the exact metal/substrate rows sent to Qiskit
#   Metal and HFSS.
# - solver exports such as Touchstone/Y-parameter files are the raw EM data used
#   by the extraction helpers.
# - comparison tables record the final driven-modal values next to the database
#   Q3D reference.

# %%
RUN_ANSYS = os.environ.get("SQUADDS_RUN_ANSYS") == "1"
CHECKPOINT_ROOT = Path("tutorials/runtime/drivenmodal_capacitance/checkpoints")

qubit_prepared = None
ncap_prepared = None
if RUN_ANSYS:
    from squadds.simulations.drivenmodal.hfss_runner import run_drivenmodal_request

    qubit_prepared = run_drivenmodal_request(qubit_request, checkpoint_dir=CHECKPOINT_ROOT)
    ncap_prepared = run_drivenmodal_request(ncap_request, checkpoint_dir=CHECKPOINT_ROOT)

    print("Qubit-claw run directory:", qubit_prepared["manifest"]["run_dir"])
    print("NCap run directory:", ncap_prepared["manifest"]["run_dir"])


# %% [markdown]
# ## Compare Against Q3D
#
# After the solve, the driven-modal post-processing writes a compact comparison
# table. The table below has the same shape and units as the solver output, so
# it is the object to inspect whether it was generated live on the Ansys machine
# or rendered statically for the docsite.
#
# The conversion used by the helper is the small-signal capacitor relation:
#
# $$Y(\omega) = j \omega C.$$
#
# Therefore SQuADDS computes the active-node capacitance matrix from the
# imaginary part of the exported admittance matrix:
#
# $$C(\omega) = \frac{\operatorname{Im}(Y(\omega))}{\omega}.$$
#
# This is why the lumped ports matter. The ports define the electrical nodes of
# the exported admittance matrix. SQuADDS maps those node names back to physical
# quantities such as `cross_to_ground`, `cross_to_claw`, or `top_to_bottom`.
# The post-processing does not add a Q3D capacitance on top of the driven-modal
# result; it derives the capacitance directly from the HFSS admittance.
#
# For the static documentation build, the `drivenmodal_fF` column is initialized
# from the validated Q3D row so the notebook remains executable without HFSS.
# On the Ansys workstation, the same comparison helper is called with the
# capacitances extracted from the exported Y-parameter sweep.

# %%
qubit_q3d = capacitance_reference_summary(qubit_row, system_kind="qubit_claw")
ncap_q3d = capacitance_reference_summary(ncap_row, system_kind="ncap")

extraction_freq_ghz = float(setup.freq_ghz)
if RUN_ANSYS:
    qubit_drivenmodal_fF = pair_capacitances_fF_from_run_dir(
        qubit_prepared["manifest"]["run_dir"],
        system_kind="qubit_claw",
        extraction_freq_ghz=extraction_freq_ghz,
    )
    ncap_drivenmodal_fF = pair_capacitances_fF_from_run_dir(
        ncap_prepared["manifest"]["run_dir"],
        system_kind="ncap",
        extraction_freq_ghz=extraction_freq_ghz,
    )
else:
    qubit_drivenmodal_fF = qubit_q3d
    ncap_drivenmodal_fF = ncap_q3d

display(capacitance_comparison_table(drivenmodal_fF=qubit_drivenmodal_fF, q3d_fF=qubit_q3d))
display(capacitance_comparison_table(drivenmodal_fF=ncap_drivenmodal_fF, q3d_fF=ncap_q3d))


# %% [markdown]
# ## Reading the Matrix Correctly
#
# A common source of confusion is the Maxwell capacitance matrix convention.
# The diagonal entries are not extra capacitances to add on top of all pair
# terms. For a two-node qubit-claw model, the transmon shunt capacitance is:
#
# $$C_\Sigma = C_{\text{cross-ground}} + C_{\text{cross-claw}}.$$
#
# not:
#
# $$C_{\text{cross-cross}} + C_{\text{cross-ground}} + C_{\text{cross-claw}}.$$
#
# SQuADDS' Hamiltonian extraction uses the first expression, so the mutual
# capacitance is not double-counted.

# %%
display(maxwell_matrix_interpretation())


# %% [markdown]
# ## Reproducing the Result
#
# A complete SQuADDS simulation result is both a number and its provenance. The
# capacitance table is the number; the saved request, layer stack, rendered
# geometry, and raw network exports explain how that number was produced. This
# makes the workflow reproducible: another user can start from the same database
# row, rerun the same request, and compare the same named capacitance entries.
#
# <div class="admonition tip">
# <p class="admonition-title">Adapting this to another component</p>
# <p>The reusable pieces are the same for any capacitance-style component:
# choose a database row, identify the Qiskit Metal pins that define the
# electrical nodes, create port specs through the request builder, keep the
# SQuADDS layer-stack preset aligned with the reference flow, and map the
# exported Y-matrix node names back to the capacitance labels you want to
# compare. If the component has more than two active nodes, inspect the full
# Maxwell matrix first and only then decide which lumped quantities belong in
# the Hamiltonian model.</p>
# </div>

# %% [markdown]
# ## License
# <div style='width: 100%; background-color:#3cb1c2;color:#324344;padding-left: 10px; padding-bottom: 10px; padding-right: 10px; padding-top: 5px'>
#     <h3>This code is a part of SQuADDS</h3>
#     <p>Developed by Sadman Ahmed Shanto</p>
#     <p>&copy; Copyright 2026.</p>
# </div>
