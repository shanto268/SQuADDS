#!/usr/bin/env python3
# %% [markdown]
# # True Inverse Design: From Hamiltonian Parameters to Geometry
#
# ## The Fundamental Problem
#
# In superconducting qubit design, we face two distinct problems:
#
# ### Forward Problem (What Tutorial 9 Does)
# H = F(xi) - Given geometry xi, predict Hamiltonian H.
#
# ### Inverse Problem (What Engineers Actually Need)
# xi = F^{-1}(H) - Given target Hamiltonian H, find the geometry xi that achieves it.
#
# ## Why This Is Harder
#
# 1. **Ill-posedness**: Multiple geometries may produce the same Hamiltonian
# 2. **Coupling**: Some physics parameters are tied to the same geometry
#    - Example: Both f_q and alpha depend on cross_length
# 3. **Non-uniqueness**: The inverse map may not exist or be multi-valued
#
# ## Our Approach
#
# 1. **Inverse Lasso**: Identify which Hamiltonian parameters control which geometry
# 2. **Inverse KAN**: Learn symbolic formulas: xi_i = g_i(H)
# 3. **Cycle Consistency**: Validate that predicted geometry reproduces target physics

# %%
import json
import random
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sympy as sp
import torch
from sklearn.linear_model import LassoCV, MultiTaskLassoCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

print("=" * 70)
print("TRUE INVERSE DESIGN: Hamiltonian → Geometry")
print("=" * 70)

# %% [markdown]
# ## 1. Data Loading and Role Swapping
#
# **Critical Conceptual Shift**: We swap the traditional ML roles.
#
# - **Features (X)**: Hamiltonian Parameters (what we want to achieve)
# - **Targets (Y)**: Design Parameters (what we need to find)
#
# This is the opposite of Tutorial 9, which predicts physics from geometry.

# %%
# Load training data
DATA_PATH = Path(__file__).parent / "data" / "training_data.parquet"
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Training data not found at {DATA_PATH}")

training_df = pd.read_parquet(DATA_PATH)
print(f"\nDataset loaded: {training_df.shape[0]:,} samples, {training_df.shape[1]} columns")

# Define parameter sets
# INPUTS: What the engineer specifies (target physics)
HAMILTONIAN_PARAMS = [
    "qubit_frequency_GHz",
    "anharmonicity_MHz",
    "cavity_frequency_GHz",
    "kappa_kHz",
    "g_MHz",
]

# OUTPUTS: What we need to find (geometry)
DESIGN_PARAMS = [
    "cross_length",
    "claw_length",
    "coupling_length",
    "total_length",
    "ground_spacing",
]

# SWAP THE ROLES: This is the key difference from Tutorial 9
# X = Physics (inputs), Y = Geometry (outputs)
X_physics = training_df[HAMILTONIAN_PARAMS].values
Y_geometry = training_df[DESIGN_PARAMS].values

print(f"\n=== INVERSE PROBLEM SETUP ===")
print(f"Input (X): Hamiltonian parameters → {X_physics.shape}")
print(f"Output (Y): Design parameters → {Y_geometry.shape}")
print(f"\nGoal: Given target physics, find the geometry that achieves it.")

# %% [markdown]
# ## 2. Check for Coupling and Ill-Conditioning
#
# Before attempting inversion, we must check if the problem is well-posed.
#
# ### Correlation Analysis
#
# If Hamiltonian parameters are highly correlated, the inverse map is
# **over-determined** - there's redundant information that won't help
# identify unique geometry.
#
# ### Condition Number
#
# The condition number of the design matrix tells us about numerical stability:
# - kappa ~ 1: Well-conditioned, stable inversion
# - kappa > 100: Ill-conditioned, sensitive to noise
# - kappa -> infinity: Singular, inversion impossible

# %%
print("\n" + "=" * 70)
print("STEP 0: Invertibility & Conditioning Analysis")
print("=" * 70)

# Compute correlation matrix of Hamiltonian parameters
corr_matrix = training_df[HAMILTONIAN_PARAMS].corr()

print("\n--- Hamiltonian Parameter Correlations ---")
print("(High correlation = redundant information for inversion)")
print(corr_matrix.round(3).to_string())

# Find highly correlated pairs
high_corr_pairs = []
for i, h1 in enumerate(HAMILTONIAN_PARAMS):
    for j, h2 in enumerate(HAMILTONIAN_PARAMS):
        if i < j and abs(corr_matrix.iloc[i, j]) > 0.7:
            high_corr_pairs.append((h1, h2, corr_matrix.iloc[i, j]))

if high_corr_pairs:
    print("\n⚠️  WARNING: Highly correlated Hamiltonian parameters detected:")
    for h1, h2, r in high_corr_pairs:
        print(f"   • {h1} ↔ {h2}: r = {r:.3f}")
    print("   This may cause the inverse map to be over-determined.")
else:
    print("\n✓ No highly correlated Hamiltonian parameters found.")

# Compute condition number of the normalized design matrix
X_normalized = StandardScaler().fit_transform(X_physics)
cond_number = np.linalg.cond(X_normalized)

print(f"\n--- Condition Number of Input Matrix ---")
print(f"κ(X) = {cond_number:.2f}")

if cond_number < 10:
    print("✓ Well-conditioned: Inverse problem is numerically stable.")
elif cond_number < 100:
    print("⚠️  Moderately conditioned: Some sensitivity to noise expected.")
else:
    print("❌ Ill-conditioned: Inverse problem may be unstable!")

# Visualize correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
)
plt.title("Hamiltonian Parameter Correlations\n(Input redundancy check)", fontsize=12)
plt.tight_layout()
plt.savefig("inverse_design_correlation.png", dpi=150)
print("\nSaved: inverse_design_correlation.png")
plt.close()

# %% [markdown]
# ## 3. Inverse Lasso: Which Physics Controls Which Geometry?
#
# We use Multi-Task Lasso to predict **all geometry parameters** from
# **all Hamiltonian parameters**. This reveals:
#
# 1. **Stiff Parameters**: Geometry that can be precisely determined from physics
# 2. **Sloppy Parameters**: Geometry that has little influence on physics
#    (and therefore cannot be uniquely recovered)
#
# ### Interpretation
#
# - Large coefficient |beta_ij|: Physics parameter H_j strongly determines geometry xi_i
# - Zero coefficient: No direct relationship (may be indirect or absent)
#
# ### Key Insight
#
# If a geometry parameter has small coefficients across ALL physics parameters,
# it's a "sloppy" direction - any value will produce similar physics.
# This is the "null space" of the inverse problem.

# %%
print("\n" + "=" * 70)
print("STEP 1: Inverse Lasso - Physics → Geometry Sensitivity")
print("=" * 70)


class InverseLassoAnalyzer:
    """
    Analyzes which Hamiltonian parameters determine which Design parameters.
    
    This is the INVERSE of Tutorial 9's approach:
    - Tutorial 9: Which geometry affects which physics?
    - This class: Which physics determines which geometry?
    """

    def __init__(
        self,
        X_hamiltonian: np.ndarray,
        Y_design: np.ndarray,
        hamiltonian_labels: list[str],
        design_labels: list[str],
    ):
        self.X_raw = X_hamiltonian
        self.Y_raw = Y_design
        self.hamiltonian_labels = hamiltonian_labels
        self.design_labels = design_labels

        # Normalize both inputs and outputs for stable coefficient interpretation
        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()

        self.X = self.scaler_X.fit_transform(self.X_raw)
        self.Y = self.scaler_Y.fit_transform(self.Y_raw)

        self.lasso_coef_df: pd.DataFrame | None = None
        self.r2_scores: dict[str, float] = {}

    def fit(self, alpha_grid: np.ndarray | None = None) -> pd.DataFrame:
        """
        Fit Multi-Task Lasso to predict geometry from Hamiltonian.
        
        Returns coefficient matrix showing which physics params
        determine which geometry params.
        """
        if alpha_grid is None:
            alpha_grid = np.logspace(-4, 1, 30)

        print("\nFitting Inverse Lasso (Hamiltonian → Geometry)...")

        model = MultiTaskLassoCV(alphas=alpha_grid, cv=5, random_state=SEED, max_iter=5000)
        model.fit(self.X, self.Y)

        # Coefficient matrix: (n_hamiltonian, n_design)
        # coef_[i, j] = how much Hamiltonian param j affects Design param i
        coef_matrix = model.coef_  # Shape: (n_design, n_hamiltonian)

        self.lasso_coef_df = pd.DataFrame(
            coef_matrix,
            index=self.design_labels,  # Rows = geometry (outputs)
            columns=self.hamiltonian_labels,  # Cols = physics (inputs)
        )

        # Compute R² for each design parameter
        Y_pred = model.predict(self.X)
        for i, design_param in enumerate(self.design_labels):
            self.r2_scores[design_param] = r2_score(self.Y[:, i], Y_pred[:, i])

        print(f"Optimal alpha: {model.alpha_:.6f}")

        return self.lasso_coef_df

    def identify_stiff_and_sloppy(self, threshold: float = 0.1) -> dict[str, Any]:
        """
        Classify geometry parameters as 'stiff' (predictable) or 'sloppy' (unpredictable).
        
        Stiff: High R² and/or large coefficients → can be recovered from physics
        Sloppy: Low R² and small coefficients → cannot be uniquely determined
        """
        if self.lasso_coef_df is None:
            raise ValueError("Call fit() first")

        result = {"stiff": [], "sloppy": [], "mixed": []}

        print("\n--- Geometry Parameter Classification ---")
        print(f"{'Parameter':<20} {'R²':<8} {'Max |coef|':<12} {'Classification'}")
        print("-" * 55)

        for design_param in self.design_labels:
            r2 = self.r2_scores[design_param]
            max_coef = self.lasso_coef_df.loc[design_param].abs().max()

            if r2 > 0.7 and max_coef > threshold:
                classification = "STIFF ✓"
                result["stiff"].append(design_param)
            elif r2 < 0.3 or max_coef < threshold / 2:
                classification = "SLOPPY ⚠️"
                result["sloppy"].append(design_param)
            else:
                classification = "MIXED"
                result["mixed"].append(design_param)

            print(f"{design_param:<20} {r2:<8.3f} {max_coef:<12.4f} {classification}")

        return result

    def get_dominant_physics(self, top_k: int = 2) -> dict[str, list[dict[str, Any]]]:
        """
        For each geometry parameter, identify which physics params dominate.
        """
        if self.lasso_coef_df is None:
            raise ValueError("Call fit() first")

        dominance = {}
        for design_param in self.design_labels:
            coeffs = self.lasso_coef_df.loc[design_param]
            top_indices = coeffs.abs().nlargest(top_k).index

            dominance[design_param] = [
                {"physics_param": idx, "coefficient": float(coeffs[idx])}
                for idx in top_indices
            ]

        return dominance

    def plot_inverse_heatmap(self, save_path: str = "inverse_lasso_heatmap.png") -> None:
        """
        Visualize the inverse mapping: which physics determines which geometry.
        
        NOTE: This is transposed from Tutorial 9!
        - Rows: Design parameters (what we want to predict)
        - Cols: Hamiltonian parameters (what we know)
        """
        if self.lasso_coef_df is None:
            raise ValueError("Call fit() first")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Heatmap of coefficients
        sns.heatmap(
            self.lasso_coef_df,
            annot=True,
            fmt=".3f",
            center=0,
            cmap="RdBu_r",
            ax=axes[0],
            cbar_kws={"label": "Lasso Coefficient"},
        )
        axes[0].set_title("Inverse Lasso: How Physics Determines Geometry", fontsize=11)
        axes[0].set_xlabel("Hamiltonian Parameter (Input)")
        axes[0].set_ylabel("Design Parameter (Output)")

        # Bar chart of R² scores
        r2_df = pd.Series(self.r2_scores)
        colors = ["green" if r2 > 0.7 else "orange" if r2 > 0.3 else "red" for r2 in r2_df.values]
        r2_df.plot(kind="barh", ax=axes[1], color=colors)
        axes[1].set_xlabel("R² Score")
        axes[1].set_title("Inverse Predictability of Geometry Parameters", fontsize=11)
        axes[1].axvline(x=0.7, color="green", linestyle="--", alpha=0.5, label="Good (R²>0.7)")
        axes[1].axvline(x=0.3, color="red", linestyle="--", alpha=0.5, label="Poor (R²<0.3)")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"\nSaved: {save_path}")
        plt.close()


# Run the Inverse Lasso analysis
inverse_lasso = InverseLassoAnalyzer(
    X_physics,
    Y_geometry,
    HAMILTONIAN_PARAMS,
    DESIGN_PARAMS,
)

coef_df = inverse_lasso.fit()
print("\n--- Inverse Lasso Coefficients ---")
print("(Rows = Geometry to predict, Cols = Physics inputs)")
print(coef_df.round(4).to_string())

# Classify parameters
classification = inverse_lasso.identify_stiff_and_sloppy()

# Get dominant physics for each geometry
dominance = inverse_lasso.get_dominant_physics(top_k=2)
print("\n--- Dominant Physics for Each Geometry ---")
for geom, physics_list in dominance.items():
    physics_str = ", ".join([f"{p['physics_param']} ({p['coefficient']:+.3f})" for p in physics_list])
    print(f"  {geom}: {physics_str}")

# Plot
inverse_lasso.plot_inverse_heatmap()

# %% [markdown]
# ## 4. Inverse KAN: Symbolic Regression for Geometry
#
# Now we train Kolmogorov-Arnold Networks to learn **symbolic formulas**
# for the STIFF geometry parameters (those with high R^2 from Inverse Lasso).
#
# ### Why Only Stiff Parameters?
#
# - **Stiff**: Physics uniquely determines geometry -> KAN can learn the inverse map
# - **Sloppy**: Physics doesn't constrain geometry -> any formula would be arbitrary
#
# ### Architecture Choices
#
# - Inputs: Most relevant Hamiltonian parameters (from Inverse Lasso)
# - Output: Single geometry parameter
# - Regularization: High sparsity to encourage simple formulas
#
# ### Expected Results
#
# For well-designed superconducting qubits, we expect:
# - cross_length ~ function of qubit_frequency, anharmonicity
# - total_length ~ function of cavity_frequency
# - coupling_length ~ function of kappa, cavity_frequency

# %%
print("\n" + "=" * 70)
print("STEP 2: Inverse KAN - Symbolic Geometry Formulas")
print("=" * 70)


def create_inverse_kan_dataset(
    df: pd.DataFrame,
    physics_features: list[str],
    geometry_target: str,
    train_ratio: float = 0.8,
    n_samples: int | None = 5000,
) -> tuple[dict[str, torch.Tensor], StandardScaler, StandardScaler, np.ndarray, np.ndarray]:
    """
    Create a dataset for Inverse KAN: Physics → Geometry.
    
    Returns normalized dataset plus scalers for inverse transform.
    """
    X = df[physics_features].values
    y = df[geometry_target].values.reshape(-1, 1)

    # Subsample for speed if needed
    if n_samples is not None and len(X) > n_samples:
        idx = np.random.choice(len(X), n_samples, replace=False)
        X, y = X[idx], y[idx]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_ratio, random_state=SEED
    )

    # Normalize (critical for KAN stability)
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_norm = X_scaler.fit_transform(X_train)
    X_test_norm = X_scaler.transform(X_test)
    y_train_norm = y_scaler.fit_transform(y_train)
    y_test_norm = y_scaler.transform(y_test)

    dataset = {
        "train_input": torch.tensor(X_train_norm, dtype=torch.float32),
        "train_label": torch.tensor(y_train_norm, dtype=torch.float32),
        "test_input": torch.tensor(X_test_norm, dtype=torch.float32),
        "test_label": torch.tensor(y_test_norm, dtype=torch.float32),
    }

    return dataset, X_scaler, y_scaler, X_test, y_test


def train_inverse_kan(
    df: pd.DataFrame,
    geometry_target: str,
    physics_features: list[str],
    hidden_width: int = 3,
    training_steps: int = 100,
    retrain_steps: int = 50,
) -> dict[str, Any]:
    """
    Train an Inverse KAN to predict a geometry parameter from physics.
    
    This learns: geometry = f(physics)
    
    Returns dict with model, scalers, formula, and metrics.
    """
    print(f"\n{'─' * 60}")
    print(f"Training Inverse KAN: {geometry_target}")
    print(f"Inputs: {physics_features}")
    print(f"{'─' * 60}")

    # Import KAN
    try:
        from kan import KAN
    except ImportError:
        print("ERROR: pykan not installed. Run: pip install pykan")
        return {"error": "pykan not installed"}

    # Create dataset
    dataset, X_scaler, y_scaler, X_test_raw, y_test_raw = create_inverse_kan_dataset(
        df, physics_features, geometry_target
    )

    print(f"Train: {len(dataset['train_input'])}, Test: {len(dataset['test_input'])}")

    # Architecture: [n_physics, hidden, 1]
    n_inputs = len(physics_features)
    architecture = [n_inputs, hidden_width, 1]

    model = KAN(width=architecture, grid=5, k=3)
    print(f"Architecture: {architecture}")

    # Train with strong regularization for simple formulas
    # Higher lamb = sparser coefficients (simpler formulas)
    # Higher lamb_entropy = smoother functions
    print(f"\nTraining Phase 1 ({training_steps} steps, high regularization)...")
    model.fit(
        dataset,
        steps=training_steps,
        lamb=0.01,  # Strong L1 for sparsity
        lamb_entropy=5.0,  # High entropy penalty for simple functions
    )

    # Check for NaN
    with torch.no_grad():
        test_out = model(dataset["test_input"])

    if torch.isnan(test_out).any():
        print("⚠️  NaN detected, skipping pruning")
    else:
        # Prune and retrain
        print("\nPruning redundant connections...")
        model = model.prune()
        model(dataset["train_input"])

        print(f"Training Phase 2 ({retrain_steps} steps, fine-tuning)...")
        model.fit(
            dataset,
            steps=retrain_steps,
            lamb=0.001,
            lamb_entropy=2.0,
        )

    # Extract symbolic formula
    # Physics-informed library: focus on functions that appear in qubit physics
    physics_library = [
        "x",  # Linear
        "x^2",  # Quadratic (capacitance scaling)
        "1/x",  # Inverse (frequency ~ 1/length)
        "sqrt",  # Square root
        "log",  # Logarithmic
    ]

    formula_str = None
    try:
        model.auto_symbolic(lib=physics_library)
        formula = model.symbolic_formula()
        formula_str = str(formula)
        print(f"\n📐 Symbolic Formula (normalized space):")
        print(f"   {geometry_target} = {formula_str}")
    except Exception as e:
        print(f"\n⚠️  Could not extract symbolic formula: {e}")

    # Evaluate on test set
    with torch.no_grad():
        y_pred_norm = model(dataset["test_input"]).numpy()

    # Inverse transform to original units
    y_pred_raw = y_scaler.inverse_transform(y_pred_norm)
    y_true_raw = y_test_raw.reshape(-1, 1)

    mae = mean_absolute_error(y_true_raw, y_pred_raw)
    r2 = r2_score(y_true_raw, y_pred_raw)

    print(f"\n📊 Test Metrics (original units):")
    print(f"   MAE: {mae:.2f} µm")
    print(f"   R²:  {r2:.4f}")

    return {
        "model": model,
        "X_scaler": X_scaler,
        "y_scaler": y_scaler,
        "formula": formula_str,
        "mae": mae,
        "r2": r2,
        "physics_features": physics_features,
        "geometry_target": geometry_target,
        "X_test": X_test_raw,
        "y_test": y_test_raw,
        "y_pred": y_pred_raw,
    }


# Train Inverse KANs for STIFF parameters only
inverse_kan_results = {}

# Determine which geometry params to model based on Inverse Lasso
params_to_model = classification["stiff"] + classification["mixed"]

if not params_to_model:
    print("\n⚠️  No stiff parameters found! Using all parameters.")
    params_to_model = DESIGN_PARAMS

print(f"\nWill train Inverse KANs for: {params_to_model}")

for geom_param in params_to_model:
    # Get the most relevant physics parameters from Inverse Lasso
    physics_for_geom = [p["physics_param"] for p in dominance[geom_param]]

    result = train_inverse_kan(
        training_df,
        geom_param,
        physics_for_geom,
        hidden_width=3,
        training_steps=100,
        retrain_steps=50,
    )

    if "error" not in result:
        inverse_kan_results[geom_param] = result

# %% [markdown]
# ## 5. Symbolic Formula Extraction and Physical Interpretation
#
# We now extract the learned symbolic formulas and interpret them
# in terms of superconducting qubit physics.
#
# ### Expected Physical Relationships
#
# Based on transmon physics:
# - **Qubit Frequency**: f_q = sqrt(8 E_C E_J) - E_C, where E_C ~ 1/C_total
#   - Larger cross_length -> larger capacitance -> lower E_C -> lower frequency
#   - Expected: cross_length ~ 1/f_q or similar
#
# - **Cavity Frequency**: f_c ~ 1/L_total (quarter-wave resonator)
#   - Expected: total_length ~ 1/f_c
#
# - **Coupling Strength**: g ~ sqrt(C_coupling / (C_q * C_r))
#   - Expected: claw_length ~ g or g^2

# %%
print("\n" + "=" * 70)
print("STEP 3: Symbolic Formula Summary")
print("=" * 70)

print("\n" + "=" * 50)
print("INVERSE DESIGN FORMULAS")
print("(Geometry as a function of Physics)")
print("=" * 50)

latex_formulas = []

for geom_param, result in inverse_kan_results.items():
    physics_str = ", ".join(result["physics_features"])
    print(f"\n📐 {geom_param}")
    print(f"   Inputs: {physics_str}")
    print(f"   Formula: {result['formula']}")
    print(f"   R² = {result['r2']:.4f}, MAE = {result['mae']:.2f} µm")

    # Create LaTeX representation
    if result["formula"]:
        latex_formulas.append(
            f"\\text{{{geom_param}}} = {result['formula']}"
        )

if latex_formulas:
    print("\n" + "-" * 50)
    print("LaTeX Equations:")
    print("-" * 50)
    for eq in latex_formulas:
        print(f"$${eq}$$")

# %% [markdown]
# ## 6. Cycle Consistency Validation
#
# The ultimate test of an inverse model is **cycle consistency**:
#
# 1. Start with target Hamiltonian parameters H_target
# 2. Use Inverse KAN to predict geometry xi_hat
# 3. (In practice) Simulate xi_hat to get H_predicted
# 4. Compare H_predicted vs H_target
#
# Since we don't have a simulator here, we validate by:
# - Comparing predicted geometry to ground truth geometry (from dataset)
# - This tests if the inverse map recovers the correct geometry
#
# ### Interpretation
#
# - High R^2 on geometry: Inverse map correctly identifies the design
# - Low R^2: Geometry parameter is "sloppy" (many designs achieve same physics)

# %%
print("\n" + "=" * 70)
print("STEP 4: Cycle Consistency Validation")
print("=" * 70)

print("\n--- Inverse Prediction Accuracy (Geometry Recovery) ---")
print(f"{'Geometry Parameter':<20} {'R²':<10} {'MAE (µm)':<12} {'Status'}")
print("-" * 52)

validation_results = []

for geom_param, result in inverse_kan_results.items():
    r2 = result["r2"]
    mae = result["mae"]

    if r2 > 0.9:
        status = "EXCELLENT ✓✓"
    elif r2 > 0.7:
        status = "GOOD ✓"
    elif r2 > 0.5:
        status = "MODERATE"
    else:
        status = "POOR ⚠️"

    print(f"{geom_param:<20} {r2:<10.4f} {mae:<12.2f} {status}")

    validation_results.append({
        "parameter": geom_param,
        "r2": r2,
        "mae": mae,
        "status": status,
    })

# Create scatter plots of predicted vs actual geometry
n_results = len(inverse_kan_results)
if n_results > 0:
    fig, axes = plt.subplots(1, min(n_results, 3), figsize=(5 * min(n_results, 3), 4))
    if n_results == 1:
        axes = [axes]

    for ax, (geom_param, result) in zip(axes, list(inverse_kan_results.items())[:3]):
        y_true = result["y_test"].flatten()
        y_pred = result["y_pred"].flatten()

        ax.scatter(y_true, y_pred, alpha=0.3, s=10)

        # Perfect prediction line
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", label="Perfect")

        ax.set_xlabel(f"True {geom_param} (µm)")
        ax.set_ylabel(f"Predicted {geom_param} (µm)")
        ax.set_title(f"Inverse KAN: {geom_param}\nR² = {result['r2']:.3f}")
        ax.legend()

    plt.tight_layout()
    plt.savefig("inverse_kan_validation.png", dpi=150)
    print(f"\nSaved: inverse_kan_validation.png")
    plt.close()

# %% [markdown]
# ## 7. Summary and Practical Usage
#
# ### What We Learned
#
# 1. **Stiff Parameters**: Can be uniquely determined from physics
#    - These have high R^2 in Inverse Lasso and clean symbolic formulas
#
# 2. **Sloppy Parameters**: Cannot be uniquely determined
#    - Multiple geometries achieve the same physics
#    - Designer has freedom to choose based on other constraints (fabrication, etc.)
#
# 3. **Coupling Effects**: Some physics parameters share geometry dependencies
#    - Example: Both f_q and alpha depend on cross_length
#    - This makes the inverse problem over-constrained in some directions
#
# ### How to Use for Design
#
# 1. Specify target Hamiltonian: (f_q, alpha, f_c, kappa, g)
# 2. Use Inverse KAN formulas to compute STIFF geometry parameters
# 3. For SLOPPY parameters, use default values or other constraints
# 4. Validate with full electromagnetic simulation

# %%
print("\n" + "=" * 70)
print("SUMMARY: Inverse Design Capability")
print("=" * 70)

print("\n📋 PARAMETER CLASSIFICATION:")
print(f"   STIFF (predictable from physics): {classification['stiff']}")
print(f"   SLOPPY (free design choice): {classification['sloppy']}")
print(f"   MIXED: {classification['mixed']}")

print("\n📐 LEARNED INVERSE FORMULAS:")
for geom_param, result in inverse_kan_results.items():
    print(f"   {geom_param}: R² = {result['r2']:.3f}")

print("\n💡 DESIGN WORKFLOW:")
print("   1. Input target physics (f_q, α, f_c, κ, g)")
print("   2. Apply inverse formulas for STIFF parameters")
print("   3. Choose SLOPPY parameters based on fab constraints")
print("   4. Validate with simulation before fabrication")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
