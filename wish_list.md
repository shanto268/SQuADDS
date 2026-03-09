# To Do List for SQuADDS:

Refer to [contribution guidelines](CONTRIBUTING.md) for more information on how to contribute code.

## Bug Fixes:

- Addressing issues on GitHub
- Addressing TODOs in the code
- Fixing any bugs in the code
- Robustly handling caching and environment variables setup for all OS (some Windows users had issues)
- Check front-end UI code for thoroughly against API calls

## Simulations:

- Stress-test and report any bugs for Ansys simulations using the SQuADDS package
- Add guesses for bare resonator frequencies (50 ohm impedance matched) using API calls to https://www.microwaves101.com/calculators/864-coplanar-waveguide-calculator
- Add more resonator property calculator methods via https://github.com/ooovector/cpw_coupling/blob/master/Conformal%20mapping%20of%20a%20CPW%20coupler.ipynb integration - may be a GUI? (https://smm.misis.ru/CPW-resonator-coupling/)
- Generalize [SQDMetal](https://github.com/sqdlab/SQDMetal) Palace simulations to work with **any HPC** and make a PR
- Conduct hyperparameter optimization study of Palace simulations and test for reliability/repeatability
- Integrate SQDMetal (fixed version) as a dependency of SQuADDS
- Add an easy Palace simulation API on SQuADDS (with parallel processing)
- Write a comprehensive tutorial on how to use Palace for simulations
- Make Ansys simulation handle arbitrary geometries for eigenmode and cap matrix simulations
- Provide compute resources to run simulations covering sparse regions of the Hamiltonian space

## Core:

- **Create a simple API for users to contribute experimental data to SQuADDS_DB**
- Handle cases where the user does not wish to specify a `resonator_type`
- Improve system design for both the SQuADDS package and SQuADDS_DB
- Develop a system to "metalize" any `.gds`/`.dxf` file (i.e., generate the corresponding Qiskit Metal file from the CAD file)
- Add support to handle designs generated via other tools (explicitly not Qiskit Metal), such as:
  - [KQCircuits](https://github.com/iqm-finland/KQCircuits)
  - [DXFWriter](https://github.com/SchusterLab/maskLib)
  - [gdsfactory](https://github.com/gdsfactory/gplugins)
  - [pyHFSS](https://github.com/QW-QubitDesign/pyHFSS)
  - [PHIDL](https://github.com/amccaugh/phidl)
- Provide APIs (modules/methods) to easily add more data columns to existing simulation entries in the database (e.g., allowing users to rerun geometries and add participation ratios)
- Transition datasets to `SQLite` or another format to handle larger-than-memory datasets as we scale
- Refactor code to implement faster methods with lower memory usage for handling DataFrame operations
- Speeding up the process for front-end UI user flow

## Contribution:

- Use local LLMs/free secure LLMs to create/update the `measured_device` dataset using the GitHub repo
- Utilize HuggingFace Hub API for handling contributions to the database in a more streamlined and automated way (e.g., create clone, branch, PR, etc.)
- Implement and deploy an acceptance server for handling contributions to the database (calculates simulation and measured value discrepancies, automatically simulates representative data points for reliability, notifies maintainers for approval) **[Not needed in the immediate future]**
- Automate integration of CAD files along with their measured Hamiltonian parameters **[Not needed in the immediate future]**

## Machine Learning (ML):

- Develop an architecture/framework for deploying ML models on SQuADDS (via HuggingFace endpoints/spaces, in code, etc.)
- Provide APIs (modules/methods) for incorporating ML interpolation features into SQuADDS
- Utilize HuggingFace Tasks for ML applications
- Identify relevant design space variables for any system given a set of $\hat{H}$ parameters using encoders
- Determine analytical dependence of $\hat{H}$ parameters on design space variables using Kolmogorov-Arnold Networks (KANs)
  - [KAN Paper](https://arxiv.org/abs/2404.19756)
  - [Symbolic Regression Example](https://kindxiaoming.github.io/pykan/Examples/Example_4_symbolic_regression.html)
- Expand training datasets using cGANs/VAEs/PINNs (i.e., attempt to simulate the simulator)
- Build upon the work by [Elie Genois et al., 2021](https://arxiv.org/pdf/2106.13126)
- system to add measured datasets to the database straight from arxiv/target journal papers

## Workflows:

- Add any other workflow that assists developers in contributing
- Implement an automated build check with comprehensive unit tests
- Automated tests upon PR submission

## Feature Requests:

- DRY sims
- async ansys
- add chi to the complete df
- Enable users to add methods with computation and append to `merged_df` for search in the `Analyzer` module
- Allow users to pass a circuit from SQCircuits and SQuADDS to provide a first-guess physical layout in Qiskit Metal
  - Helpful resources:
    - [SQCircuit](https://github.com/stanfordLINQS/SQcircuit/)
    - [Circuit Recognizer](https://github.com/mahmut-aksakalli/circuit_recognizer)
    - [CircuitNet](https://github.com/aaanthonyyy/CircuitNet)
- Incorporate [SCILLA](https://github.com/aspuru-guzik-group/scilla?tab=readme-ov-file) and/or its applications
- Add interpolation of individual components (both API and UI)

## Boring but Necessary:

- say that coupling is capacitive and handle inductive logic
- order of `select_system` and specifying it as we grow beyond two
- **Check for breaking changes in the latest version of dependencies and update the package accordingly**
- add input type error handling to sim code
- automated kink detection + meander smoothing prior to qiskit-metal rendering
- letting users choose the `.env` file location OR telling them where to find it
- Check for claw dimensions on the Hamiltonian space plot
- Create unit tests for each feature/file
- Establish proper train/test splits and modify `SQuADDS_DB()` to always return all data
- Standardize the handling of units for simulated results and implement necessary backend changes
- Add more tutorials on how to use the package and its various applications
- Verify if the precision of design parameters is handled correctly and fix as needed
- Change all instances of `NCap` to `CapNInterdigital`
- Separate the documentation from the main package

## Fancy/For Fun:

- Setup a Model Context Protocol server for SQuADDS
- Implement LLM-based queries for SQuADDS using [pandas-ai](https://docs.pandas-ai.com/intro) (support for OpenAI and local LLaMA models)
- Add visual components to the SQuADDS UI

That makes total sense, especially since your original `Tutorial-8` was already built using Keras and TensorFlow (`Sequential`, `Dense`, `Adam`, etc.). Transitioning the new graph architecture into TensorFlow is completely doable and keeps your codebase consistent.

Instead of PyTorch Geometric, the agent will need to use **TensorFlow GNN (TF-GNN)** or **Spektral** (a library specifically built for graph deep learning on Keras).

Here is the revised, comprehensive prompt adjusted strictly for the TensorFlow/Keras ecosystem.

---
