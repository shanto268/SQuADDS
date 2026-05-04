# %% [markdown]
# # Tutorial 13: Unified Driven-Modal Hamiltonian Extraction
#
# In Tutorial 10 we used driven-modal HFSS to reproduce capacitance-style
# quantities. Here we use the same idea for a complete qubit-cavity system.
# The point of this tutorial is not just to press "solve" in Ansys. The point is
# to show how SQuADDS turns one database geometry into one electromagnetic
# network model, and how different post-processing views of that same network
# recover the Hamiltonian parameters users care about.
#
# The workflow is:
#
# 1. ask SQuADDS for a qubit-cavity design near a target Hamiltonian,
# 2. render that one coupled geometry into HFSS,
# 3. sweep finely around the qubit band and the resonator band,
# 4. extract $f_q$, $\alpha$, $f_r$, $\kappa$, $\chi$, and $g$, and
# 5. compare the driven-modal result against the SQuADDS reference row.
#
# The docsite labels this as Tutorial 11, because the intermediate development
# notebooks are intentionally not part of the public tutorial sequence.
#
# The workflow is deliberately split into two phases:
#
# - **Simulation declaration**: choose the geometry, layer stack, ports, adaptive
#   setup, and frequency sweeps. This produces portable request objects.
# - **Physics extraction**: read the exported S/Y-parameters, terminate the
#   unused ports in physically meaningful ways, and convert the resulting
#   resonances into Hamiltonian parameters.
#
# <div class="admonition note">
# <p class="admonition-title">Why this tutorial exists</p>
# <p>Earlier SQuADDS tutorials teach the traditional split workflow:
# capacitances from Q3D and mode frequencies/couplings from eigenmode
# simulations. This tutorial shows the driven-modal alternative. We render one
# coupled qubit-cavity-feedline network, keep all three physical ports in the
# exported HFSS network, and use post-processing to ask several physics
# questions of the same EM data.</p>
# </div>

# %% [markdown]
# ## Imports
#
# The important point is that the notebook uses SQuADDS-level objects. We do
# not build Qiskit Metal components or Ansys COM calls by hand here.
#
# The imported helper functions are the public API surface for this tutorial:
#
# - `build_segmented_coupled_system_requests(...)` creates the HFSS request
#   objects for the qubit, bridge, and resonator frequency windows.
# - `coupled_reference_summary(...)` normalizes the SQuADDS row into the
#   reference Hamiltonian fields and JJ quantities needed for post-processing.
# - `default_hamiltonian_setup(...)` and `segmented_hamiltonian_sweeps(...)`
#   provide tested defaults, while still leaving the important knobs visible.
# - `hamiltonian_comparison_table(...)` formats the final driven-modal versus
#   SQuADDS comparison.
#
# The lower-level heavy lifting is still visible through these objects:
# `DrivenModalSetupSpec` maps to the HFSS adaptive setup, `DrivenModalSweepSpec`
# maps to the exported frequency sweep, and the request payload contains the
# Qiskit Metal component options, layer stack, and port mapping.

# %%
import contextlib
import io
import os
from pathlib import Path

import pandas as pd

from squadds import Analyzer, SQuADDS_DB
from squadds.simulations.drivenmodal import (
    build_segmented_coupled_system_requests,
    coupled_hamiltonian_from_prepared_runs,
    coupled_reference_summary,
    default_hamiltonian_setup,
    hamiltonian_comparison_table,
    segmented_hamiltonian_sweeps,
)

try:
    from IPython.display import display
except ImportError:  # pragma: no cover

    def display(obj):
        print(obj)


# %% [markdown]
# ## Query a Target Design
#
# This follows the same pattern as Tutorial 1 and Tutorial 5: select a system,
# choose target Hamiltonian parameters, and let the `Analyzer` return the best
# matching SQuADDS design.
#
# The database calls below define the design family:
#
# - `select_system(["qubit", "cavity_claw"])` asks for a coupled
#   qubit-resonator dataset.
# - `select_qubit("TransmonCross")` fixes the transmon layout component.
# - `select_cavity_claw("RouteMeander")` fixes the readout resonator family.
# - `select_resonator_type("quarter")` selects a quarter-wave resonator.
#
# `Analyzer.find_closest(...)` then searches that system dataframe for rows
# whose simulated Hamiltonian parameters are close to `target_params`. The
# returned row contains both the Qiskit Metal design options and the reference
# simulation results that we will compare against later.

# %%
db = SQuADDS_DB()
db.select_system(["qubit", "cavity_claw"])
db.select_qubit("TransmonCross")
db.select_cavity_claw("RouteMeander")
db.select_resonator_type("quarter")

with contextlib.redirect_stdout(io.StringIO()):
    system_df = db.create_system_df()
analyzer = Analyzer(db)

target_params = {
    "qubit_frequency_GHz": 4.0,
    "anharmonicity_MHz": -130,
    "cavity_frequency_GHz": 8.9,
    "kappa_kHz": 300,
    "g_MHz": 50,
    "resonator_type": "quarter",
}

with contextlib.redirect_stdout(io.StringIO()):
    results = analyzer.find_closest(target_params=target_params, num_top=3, metric="Euclidean")
reference_row = results.iloc[0]

display(system_df.head())
display(results.head(3))


# %% [markdown]
# ## Build the Driven-Modal Requests
#
# A coupled-system driven-modal solve is a 3-port network:
#
# 1. feedline input,
# 2. feedline output, and
# 3. the Josephson junction port.
#
# These are not three separate devices. They are three ports on one rendered
# qubit-claw-resonator-feedline geometry. SQuADDS uses the port mapping inside
# the request to connect the feedline ends and the JJ cut to HFSS lumped ports.
# The ports are created as 50 ohm renormalizable lumped ports. That reference
# impedance is how HFSS normalizes the exported S-parameters; because the full
# multiport network is exported, SQuADDS can later change external loads,
# short/open a port, or attach a Josephson-junction surrogate in post-processing
# without rerendering the geometry.
#
# `coupled_reference_summary(reference_row)` extracts the database reference
# values in a consistent unit convention:
#
# - qubit frequency in GHz,
# - anharmonicity in MHz,
# - cavity frequency in GHz,
# - linewidth $\kappa / 2\pi$ in MHz,
# - coupling $g / 2\pi$ in MHz, and
# - the bare and state-dependent JJ inductance values used for terminations.
#
# `default_hamiltonian_setup(...)` controls the adaptive mesh. The important
# knobs are `freq_ghz`, `max_delta_s`, `min_converged`, `max_passes`, and
# `basis_order`. The default centers the adaptive solve near the expected
# cavity frequency and uses the production driven-modal settings validated for
# this flow: `max_delta_s=0.005`, `min_converged=7`, `basis_order=-1`
# (mixed order), and up to 20 adaptive passes.
#
# `segmented_hamiltonian_sweeps(...)` controls the exported frequency samples.
# This is where the dense/coarse/dense structure is declared:
#
# - `qubit_count` samples the qubit band where the JJ-port admittance crossing
#   determines $f_q$ and $\alpha$.
# - `bridge_count` samples the middle band more coarsely. It keeps the exported
#   network continuous without spending most of the solve budget away from the
#   resonances.
# - `resonator_count` samples the readout band where the feedline response gives
#   $f_r$, $\kappa$, $\chi$, and $g$.
#
# The dataframe printed after the cell is worth reading carefully: it is the
# explicit frequency plan that will be sent to HFSS.
#
# <div class="admonition important">
# <p class="admonition-title">One EM model, segmented sweeps</p>
# <p>The physics object is one 3-port coupled system. SQuADDS stores the qubit,
# bridge, and resonator windows as separate request records so each window can
# be checkpointed, inspected, and rerun independently. On the Ansys side these
# records use the same geometry, layer stack, ports, and adaptive setup.</p>
# </div>

# %%
reference = coupled_reference_summary(reference_row)

setup = default_hamiltonian_setup(freq_ghz=reference["cavity_frequency_ghz"])
sweeps = segmented_hamiltonian_sweeps(
    reference,
    qubit_count=22_000,
    bridge_count=4_001,
    resonator_count=22_000,
)
requests = build_segmented_coupled_system_requests(
    reference_row,
    resonator_type="quarter",
    run_id="tutorial11-quarter-wave",
    reference=reference,
    setup=setup,
    sweeps=sweeps,
)

sweep_table = pd.DataFrame(
    [
        {
            "band": name,
            "start_GHz": request.sweep.start_ghz,
            "stop_GHz": request.sweep.stop_ghz,
            "points": request.sweep.count,
            "type": request.sweep.sweep_type,
        }
        for name, request in requests.items()
    ]
)
display(sweep_table)


# %%
display(pd.DataFrame([setup.to_renderer_kwargs()]).T.rename(columns={0: "HFSS setup value"}))


# %%
port_table = pd.DataFrame(
    [
        {"port": port_name, **port_spec}
        for port_name, port_spec in next(iter(requests.values())).design_payload["port_mapping"].items()
    ]
)
display(port_table)


# %% [markdown]
# ## State-dependent Josephson inductance
#
# `LJ_bare_nH` is the linear Josephson inductance from the transmon design options
# (`aedt_q3d_inductance` / `Lj` / `LJ`). The ground-state and excited-state
# columns use the same workflow as `coupled_reference_summary(...)`: helper
# `transmon_state_inductances(...)` in `squadds.simulations.drivenmodal.workflows`
# builds a zero-temperature `scqubits.Transmon` with
# $E_J$ from the bare $L_J$ and $E_C$ from the qubit shunt capacitance (via the
# database `cross_to_ground` and `cross_to_claw` pair capacitances in fF).
# It then evaluates the transmon `cos_phi_operator` in the ground and
# first-excited eigenstates. The effective inductances are
# $L_J^{(g)} = L_{J,\mathrm{bare}} / \langle \cos\phi \rangle_g$ and
# $L_J^{(e)} = L_{J,\mathrm{bare}} / \langle \cos\phi \rangle_e$, matching how the
# coupled driven-modal workflow assigns JJ terminations when it reduces the
# exported multiport network.

# %%
jj_table = pd.DataFrame(
    [
        {
            "EJ_GHz": reference["ej_ghz"],
            "EC_GHz": reference["ec_ghz"],
            "LJ_bare_nH": 1e9 * reference["lj_bare_h"],
            "LJ_ground_nH": 1e9 * reference["lj_ground_h"],
            "LJ_excited_nH": 1e9 * reference["lj_excited_h"],
        }
    ]
)
display(jj_table)


# %% [markdown]
# ## Run the Simulation
#
# A production driven-modal sweep can take hours, so this notebook uses the
# environment variable `SQUADDS_RUN_ANSYS` as the switch between documentation
# mode and solver mode. With `SQUADDS_RUN_ANSYS=1`, the cell renders and solves
# the three frequency windows on the Windows Ansys workstation. With the
# variable unset, the notebook remains executable on machines without Ansys and
# still shows the exact sweep plan and reference target.
#
# `run_drivenmodal_request(...)` is the notebook-facing entry point for the
# Ansys execution contract. For each request, the Ansys executor uses that
# contract to:
#
# - builds the Qiskit Metal `QubitCavity` layout from the selected SQuADDS row,
# - applies the SQuADDS HFSS layer-stack preset with PEC metal and cryogenic
#   silicon metadata,
# - renders the coupled system into a fresh HFSS driven-modal design,
# - places the two feedline ports and the JJ lumped port,
# - creates the adaptive setup and requested sweep, and
# - exports Touchstone/Y-parameter data plus a manifest.
#
# The Touchstone export is where `scikit-rf` enters the workflow. HFSS gives us
# frequency-dependent S/Y/Z matrices; `scikit-rf` preserves the frequency axis,
# port order, complex S-parameters, and reference impedance metadata in a
# standard multiport `Network`. SQuADDS uses that representation to write `.s3p`
# files, reduce the 3-port network into loaded 2-port or 1-port views, and keep
# the port normalization explicit rather than hidden in a plot.
#
# The run writes a self-contained provenance bundle under `CHECKPOINT_ROOT`.
# That bundle contains the request payload, layer stack, rendered-geometry
# diagnostics, exported network data, and post-processing tables. The bundle is
# what lets the physics extraction be rerun without launching HFSS again.

# %%
RUN_ANSYS = os.environ.get("SQUADDS_RUN_ANSYS") == "1"
CHECKPOINT_ROOT = Path("tutorials/runtime/drivenmodal_combined_hamiltonian/checkpoints")

prepared_runs = None
if RUN_ANSYS:
    from squadds.simulations.drivenmodal.hfss_runner import run_drivenmodal_request

    prepared_runs = {
        band: run_drivenmodal_request(request, checkpoint_dir=CHECKPOINT_ROOT) for band, request in requests.items()
    }
    for band, prepared in prepared_runs.items():
        print(f"{band}: {prepared['manifest']['run_dir']}")


# %% [markdown]
# ## Post-Process the Same Network
#
# The HFSS output is a frequency-dependent 3-port network. SQuADDS post-processes
# the same exported data in three views. The views are different mathematical
# reductions of one exported network, not three unrelated simulations.
#
# ### 1. Resonator frequency and kappa
#
# The readout mode is read from feedline transmission. SQuADDS terminates the JJ
# port with a chosen JJ load, converts the reduced admittance back to a 2-port
# S-parameter network, and searches the loaded $S_{21}$ response for the
# resonance feature. The feature position gives $f_r$ and the linewidth gives
# $\kappa$.
#
# ### 2. Qubit frequency and anharmonicity
#
# The qubit mode is read from the JJ port. Raw Y-parameters assume all other
# ports are shorted, so SQuADDS first terminates the feedline ports with the
# intended external loads using a Schur-complement network reduction. It then
# adds the linear JJ surrogate admittance
#
# $$Y_\mathrm{JJ}(\omega) = \frac{1}{R_J} + j\omega C_J + \frac{1}{j\omega L_J}.$$
#
# The linear qubit resonance is the positive-slope zero crossing of
# $\operatorname{Im}(Y_\mathrm{env} + Y_\mathrm{JJ})$. The local slope gives the
# effective shunt capacitance seen by the junction,
#
# $$C_\Sigma = \frac{1}{2}\frac{\partial \operatorname{Im}(Y)}{\partial \omega}.$$
#
# SQuADDS then passes $E_J$ and $E_C$ to `scqubits.Transmon`. We use scqubits
# for $f_q$ and $\alpha$ rather than relying on a hand-written transmon
# expansion. The state-dependent inductances in the table above also come from
# scqubits: SQuADDS evaluates the transmon `cos_phi_operator` in the ground and
# excited eigenstates and uses those expectation values to turn the bare
# Josephson inductance into effective ground/excited loads.
#
# ### 3. Coupling from chi
#
# SQuADDS repeats the feedline reduction twice: once with the ground-state
# effective inductance and once with the excited-state effective inductance.
# The resonance positions of those two $S_{21}$ traces give the dispersive shift
# $\chi$. With $f_q$, $\alpha$, $f_r$, and $\chi$ known, SQuADDS estimates $g$
# using the dispersive transmon relation, including the non-RWA correction used
# by the helper.
#
# The table below is the Hamiltonian-level object users should expect from this
# workflow. In documentation mode it is initialized from the selected SQuADDS
# reference row so the notebook can be rendered without HFSS. In solver mode the
# same table is generated from the exported driven-modal network data.

# %%
if RUN_ANSYS:
    drivenmodal_hamiltonian = coupled_hamiltonian_from_prepared_runs(prepared_runs)
else:
    drivenmodal_hamiltonian = {
        "qubit_frequency_ghz": reference["qubit_frequency_ghz"],
        "anharmonicity_mhz": reference["anharmonicity_mhz"],
        "cavity_frequency_ghz": reference["cavity_frequency_ghz"],
        "kappa_mhz": reference["kappa_mhz"],
        "g_mhz": reference["g_mhz"],
        "chi_mhz": float("nan"),
    }

display(hamiltonian_comparison_table(drivenmodal=drivenmodal_hamiltonian, squadds=reference))


# %% [markdown]
# ## What the Comparison Means
#
# The SQuADDS row is the pre-simulated reference. The driven-modal row is a
# full-wave re-simulation of the same geometry with a different electromagnetic
# observable.
#
# A good result should satisfy three sanity checks:
#
# 1. the rendered HFSS geometry includes the qubit, claw, resonator, and
#    feedline ports,
# 2. silicon is set to the cryogenic value used by the Q3D/eigenmode flows, and
# 3. the Hamiltonian table agrees within the accuracy expected from the sweep
#    resolution and the chosen JJ termination model.
#
# If the resonator frequency shifts below the bare SQuADDS cavity frequency,
# that is not automatically a bug: the driven-modal model includes capacitive
# loading from the qubit.
#
# The port-normalization point is worth repeating. HFSS exports the network
# normalized to the 50 ohm port references. During post-processing SQuADDS
# converts between S and Y representations, changes the mathematical load on a
# port, and then converts back to whichever view is most useful. That is why we
# can ask "what does the feedline see when the JJ is in its ground-state
# inductive load?" without rerunning HFSS.
#
# Useful customization points:
#
# - Increase `qubit_count` or `resonator_count` when the resonance is narrow and
#   the fitted frequency is resolution-limited.
# - Increase the frequency padding when the resonance is outside the expected
#   band.
# - Use a finite `cj_f` or `rj_ohms` in the local post-processing if you want a
#   fuller Dolan-junction surrogate. The EM simulation still supplies the large
#   geometric capacitance of the transmon pads.
# - Keep the layer-stack preset aligned with the Q3D/eigenmode flows before
#   comparing numbers; material drift is an easy way to create misleading
#   Hamiltonian disagreement.
#
# <div class="admonition warning">
# <p class="admonition-title">Geometry sanity checks are not optional</p>
# <p>Before trusting a driven-modal Hamiltonian table, inspect the rendered
# Qiskit Metal and HFSS screenshots. The coupled qubit, claw, resonator, and
# feedline must all be present; metal sheets must be assigned to PEC; the
# substrate should use the cryogenic silicon permittivity $11.45$; and the
# lumped ports should land on the feedline ends and the JJ cut, not on launchers
# or unrelated metal.</p>
# </div>

# %% [markdown]
# ## License
# <div style='width: 100%; background-color:#3cb1c2;color:#324344;padding-left: 10px; padding-bottom: 10px; padding-right: 10px; padding-top: 5px'>
#     <h3>This code is a part of SQuADDS</h3>
#     <p>Developed by Sadman Ahmed Shanto</p>
#     <p>&copy; Copyright 2026.</p>
# </div>
