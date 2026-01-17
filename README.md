<center>
  <img src="docs/_static/images/squadds_not_transparent.png" width="100%" alt="SQuADDS Logo" />
</center>

# Superconducting Qubit And Device Design and Simulation Database ![Version](https://img.shields.io/github/v/release/LFL-Lab/SQuADDS) ![Pepy Total Downloads](https://img.shields.io/pepy/dt/squadds) ![Build Status](https://img.shields.io/github/actions/workflow/status/LFL-Lab/SQuADDS/ci.yml?branch=master) ![License](https://img.shields.io/github/license/LFL-Lab/SQuADDS) [![arXiv](https://img.shields.io/badge/arXiv-2312.13483-<COLOR>.svg)](https://arxiv.org/abs/2312.13483) ![Alpha Version](https://img.shields.io/badge/Status-Alpha%20Version-yellow)

> :warning: **This project is an alpha release and currently under active development. Some features and documentation may be incomplete. Please update to the latest release.**

The SQuADDS (Superconducting Qubit And Device Design and Simulation) Database Project is an open-source resource aimed at advancing research in superconducting quantum device designs. It provides a robust workflow for generating and simulating superconducting quantum device designs, facilitating the accurate prediction of Hamiltonian parameters across a wide range of design geometries.

**Paper Link:** [SQuADDS: A Database for Superconducting Quantum Device Design and Simulation](https://quantum-journal.org/papers/q-2024-09-09-1465/)

**Docsite Link:** [https://lfl-lab.github.io/SQuADDS/](https://lfl-lab.github.io/SQuADDS/)

**Hugging Face Link:** [https://huggingface.co/datasets/SQuADDS/SQuADDS_DB](https://huggingface.co/datasets/SQuADDS/SQuADDS_DB)

**Contribution Portal Link:** [https://squadds-portal.vercel.app](https://squadds-portal.vercel.app)

**Chat with the Codebase:** [https://deepwiki.com/LFL-Lab/SQuADDS/1-overview](https://deepwiki.com/LFL-Lab/SQuADDS/1-overview)

## Table of Contents

- [Citation](#citation)
- [Installation](#installation)
  - [Install from Source](#install-from-source-recommended-for-development)
  - [Install using pip](#install-using-pip)
  - [Run using Docker](#run-using-docker)
- [Tutorials](#tutorials)
- [Contributing](#contributing)
- [License](#license)
- [FAQs](#faqs)
- [Contact](#contact)
- [Contributors](#contributors)
- [Developers](#developers)

---

## Citation

If you use SQuADDS in your research, please cite the following paper:

```bibtex
@article{Shanto2024squaddsvalidated,
  doi = {10.22331/q-2024-09-09-1465},
  url = {https://doi.org/10.22331/q-2024-09-09-1465},
  title = {{SQ}u{ADDS}: {A} validated design database and simulation workflow for superconducting qubit design},
  author = {Shanto, Sadman and Kuo, Andre and Miyamoto, Clark and Zhang, Haimeng and Maurya, Vivek and Vlachos, Evangelos and Hecht, Malida and Shum, Chung Wa and Levenson-Falk, Eli},
  journal = {{Quantum}},
  issn = {2521-327X},
  publisher = {{Verein zur F{\"{o}}rderung des Open Access Publizierens in den Quantenwissenschaften}},
  volume = {8},
  pages = {1465},
  month = sep,
  year = {2024}
}
```

---

## Installation

SQuADDS uses [uv](https://docs.astral.sh/uv/) for fast, reliable Python package management.

### Prerequisites

Install `uv` (if you don't have it already):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install from Source (Recommended for Development)

```bash
git clone https://github.com/LFL-Lab/SQuADDS.git
cd SQuADDS
uv sync
```

Verify the installation:

```bash
uv run python -c "import squadds; print(squadds.__file__)"
```

### Install using pip

```bash
pip install SQuADDS
```

### Optional Dependencies

Install GDS processing tools:

```bash
uv sync --extra gds
```

Install documentation tools:

```bash
uv sync --extra docs
```

Install development tools:

```bash
uv sync --extra dev
```

Install contribution tools (for contributing data to SQuADDS):

```bash
uv sync --extra contrib
```

Install all optional dependencies:

```bash
uv sync --all-extras
```

### Setting up Jupyter Notebook

To use SQuADDS in Jupyter notebooks (including VS Code/Cursor), register the kernel:

```bash
uv sync --extra dev  # Installs ipykernel
uv run python -m ipykernel install --user --name squadds --display-name "SQuADDS (uv)"
```

Then select **"SQuADDS (uv)"** as your kernel in Jupyter/VS Code/Cursor.

### Run using Docker:

<details>
<summary>Click to expand/hide Docker instructions</summary>
<br>

We provide a pre-built Docker image that contains all dependencies, including `Qiskit-Metal` and the latest `SQuADDS` release.

#### Pull the Latest Docker Image

You can pull the latest image of **SQuADDS** from GitHub Packages:

```bash
docker pull ghcr.io/lfl-lab/squadds_env:latest
```

If you'd like to pull a specific version (support begins from `v0.3.4` onwards), use the following command:

```bash
docker pull ghcr.io/lfl-lab/squadds_env:v0.3.4
```

You can find all available versions and tags for the **squadds_env** Docker image on [LFL-Lab Packages](https://github.com/LFL-Lab?tab=packages&repo_name=SQuADDS).

#### Run the Docker Container

After pulling the image, you can run the container using:

```bash
docker run -it ghcr.io/lfl-lab/squadds_env:latest /bin/bash
```

This will give you access to a bash shell inside the container.

#### Activate the Conda Environment

Inside the container, activate the `squadds-env` environment:

```bash
conda activate squadds-env
```

#### Run SQuADDS

Once the environment is active, you can run **SQuADDS** by executing your Python scripts or starting an interactive Python session.

</details>

---

## Tutorials

The following tutorials are available to help you get started with `SQuADDS`:

- [Tutorial 0: Using the SQuADDS WebUI](https://lfl-lab.github.io/SQuADDS/source/tutorials/Tutorial-0_Using_the_SQuADDS_WebUI.html)
- [Tutorial 1: Getting Started with SQuADDS](https://lfl-lab.github.io/SQuADDS/source/tutorials/Tutorial-1_Getting_Started_with_SQuADDS.html)
- [Tutorial 2: Simulating Interpolated Designs](https://lfl-lab.github.io/SQuADDS/source/tutorials/Tutorial-2_Simulate_interpolated_designs.html)
- [Tutorial 3: Contributing Experimentally-Validated Simulation Data to the SQuADDS Database](https://lfl-lab.github.io/SQuADDS/source/tutorials/Tutorial-3_Contributing_Validated_Simulation_Data_to_SQuADDS.html)
- [Tutorial 4: Contributing Measured Devices' Data to the SQuADDS Database](https://lfl-lab.github.io/SQuADDS/source/tutorials/Tutorial_4_Contributing_Measured_Data_to_SQuADDS.html)
- [Tutorial 5: Designing a "fab-ready" chip with SQuADDS](https://lfl-lab.github.io/SQuADDS/source/tutorials/Tutorial-5_Designing_a_fab_ready_chip_with_SQuADDS.html)
- [Tutorial 6: Adding Airbridges](https://lfl-lab.github.io/SQuADDS/source/tutorials/Tutorial-6_Adding_Airbridges.html)
- [Tutorial 7: Simulate designs with palace](https://lfl-lab.github.io/SQuADDS/source/tutorials/Tutorial-7_Simulate_designs_with_palace.html)
- [Tutorial 8: ML Interpolation in SQuADDS](https://lfl-lab.github.io/SQuADDS/source/tutorials/Tutorial-8_ML_interpolation_in_SQuADDS.html)
- [Tutorial 9: Learning the Inverse Map](https://lfl-lab.github.io/SQuADDS/source/tutorials/Tutorial-9_Learing_the_Inverse_Design_Map.html)
- [PhysicsLearning: Explainable ML for Inverse Design](tutorials/PhysicsLearning.ipynb) - Uses EBM + Symbolic Regression to discover physics equations

### Running the PhysicsLearning Notebook

The `PhysicsLearning.ipynb` notebook demonstrates physics-informed machine learning for superconducting qubit design. To run it:

1. **Install dependencies** (includes `interpret` for EBM and `pysr` for symbolic regression):
   ```bash
   uv sync --extra ml
   ```

2. **Set up the Jupyter kernel**:
   ```bash
   uv run python -m ipykernel install --user --name squadds --display-name "SQuADDS (uv)"
   ```

3. **Open the notebook** and select the **"SQuADDS (uv)"** kernel.

4. **Required data files** (included in `tutorials/data/`):
   - `training_data.parquet` - Dataset 1 (fixed EJ)
   - `training_data_v5_expanded.parquet` - Dataset 2 (variable EJ)

---

## Contributing

We welcome contributions from the community! Here is our [work wish list](wish_list.md).

You can use our [web portal](https://squadds-portal.vercel.app) to contribute your files - [https://squadds-portal.vercel.app](https://squadds-portal.vercel.app)

Please see our [Contributing Guidelines](CONTRIBUTING.md) for more information on how to get started and absolutely feel free to reach out to us if you have any questions.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## FAQs

Check out our [FAQs](https://lfl-lab.github.io/SQuADDS/source/getting_started.html#faq-s) for common questions and answers.

---

## Contact

For inquiries or support, please contact [Sadman Ahmed Shanto](mailto:shanto@usc.edu).

---

## Contributors


| Name                  | Institution                        | Contribution                         |
|:----------------------|:-----------------------------------|:-------------------------------------|
| Clark Miyamoto        | New York University                | Code contributor                     |
| Madison Howard        | California Institute of Technology | Bug Hunter                           |
| Evangelos Vlachos     | University of Southern California  | Code contributor                     |
| Kaveh Pezeshki        | Stanford University                | Documentation contributor            |
| Anne Whelan           | US Navy                            | Documentation contributor            |
| Jenny Huang           | Columbia University                | Documentation contributor            |
| Connie Miao           | Stanford University                | Data Contributor                     |
| Malida Hecht          | University of Southern California  | Data contributor                     |
| Daria Kowsari, PhD    | University of Southern California  | Data contributor                     |
| Vivek Maurya          | University of Southern California  | Data contributor                     |
| Haimeng Zhang, PhD    | IBM                                | Data contributor                     |
| Elizabeth Kunz        | University of Southern California  | Documentation  and  Code contributor |
| Adhish Chakravorty    | University of Southern California  | Documentation  and  Code contributor |
| Ethan Zheng           | University of Southern California  | Data contributor  and Bug Hunter     |
| Sara Sussman, PhD     | Fermilab                           | Bug Hunter                           |
| Priyangshu Chatterjee | IIT Kharagpur                      | Documentation contributor            |

## Developers
- [shanto268](https://github.com/shanto268) - 414 contributions
- [elizabethkunz](https://github.com/elizabethkunz) - 17 contributions
- [LFL-Lab](https://github.com/LFL-Lab) - 9 contributions
- [NxtGenLegend](https://github.com/NxtGenLegend) - 1 contributions
- [ethanzhen7](https://github.com/ethanzhen7) - 1 contributions
- [PCodeShark25](https://github.com/PCodeShark25) - 1 contributions
---
