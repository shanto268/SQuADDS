#!/usr/bin/env python3
"""
Inverse Design Map Learning with KANs

This script implements the inverse design mapping from Hamiltonian parameters
to geometric design parameters using:
1. Design-Relevance Encoder (LASSO) - for feature selection
2. KAN-based Symbolic Decoder - for learning interpretable physics

Based on Tutorial 9 of SQuADDS with improvements to the KAN architecture.
"""

import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
import torch
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Design-Relevance Encoder
# =============================================================================

class DesignRelevanceEncoder:
    """
    Identifies which geometric parameters most influence each Hamiltonian parameter.
    Uses LASSO regression for feature selection with coefficient shrinkage.
    """

    def __init__(
        self,
        X_design: np.ndarray,
        Y_hamiltonian: np.ndarray,
        design_labels: list[str],
        hamiltonian_labels: list[str],
    ):
        """
        Initialize the encoder with design inputs and Hamiltonian outputs.

        Args:
            X_design: Design parameter values (N samples x D features)
            Y_hamiltonian: Hamiltonian parameter values (N samples x H targets)
            design_labels: Names of design parameters
            hamiltonian_labels: Names of Hamiltonian parameters
        """
        self.X_raw = X_design
        self.Y_raw = Y_hamiltonian
        self.design_labels = design_labels
        self.hamiltonian_labels = hamiltonian_labels

        self.scaler_X = StandardScaler()
        self.scaler_Y = StandardScaler()

        self.X = self.scaler_X.fit_transform(self.X_raw)
        self.Y = self.scaler_Y.fit_transform(self.Y_raw)

        self.lasso_coef_df: pd.DataFrame | None = None

    def run_multitask_lasso(
        self,
        alpha_grid: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Trains a multi-task Lasso model for feature selection.

        Args:
            alpha_grid: Grid of alpha values for cross-validation

        Returns:
            DataFrame with LASSO coefficients for each design-Hamiltonian pair
        """
        if alpha_grid is None:
            alpha_grid = np.logspace(-4, 1, 20)

        model = MultiTaskLassoCV(alphas=alpha_grid, cv=5, random_state=42)
        model.fit(self.X, self.Y)
        coef_matrix = model.coef_.T  # (n_design, n_hamiltonian)

        self.lasso_coef_df = pd.DataFrame(
            coef_matrix,
            index=self.design_labels,
            columns=self.hamiltonian_labels,
        )
        return self.lasso_coef_df

    def get_dependency_summary(
        self,
        top_k: int = 3,
        threshold: float = 1e-3,
    ) -> dict[str, Any]:
        """
        Returns top design parameters for each Hamiltonian parameter.

        Args:
            top_k: Number of top features to return
            threshold: Minimum absolute coefficient threshold

        Returns:
            Dictionary with feature importance summary
        """
        if self.lasso_coef_df is None:
            raise ValueError("Run run_multitask_lasso() first")

        summary: dict[str, Any] = {"lasso": {}}

        for h in self.hamiltonian_labels:
            top = self.lasso_coef_df[h].abs().sort_values(ascending=False)
            filtered = [
                {
                    "parameter": top.index[i],
                    "coef": float(self.lasso_coef_df[h][top.index[i]]),
                }
                for i in range(len(top))
                if abs(top.values[i]) >= threshold
            ][:top_k]
            summary["lasso"][h] = filtered

        return summary

    def plot_heatmap(self, save_path: str | None = None) -> None:
        """Plot LASSO coefficient heatmap."""
        if self.lasso_coef_df is None:
            raise ValueError("Run run_multitask_lasso() first")

        try:
            import seaborn as sns
        except ImportError:
            print("seaborn not installed, skipping heatmap")
            return

        plt.figure(figsize=(10, 6))
        sns.heatmap(
            self.lasso_coef_df,
            annot=True,
            center=0,
            cmap="coolwarm",
            cbar_kws={"label": "Coefficient Value"},
        )
        plt.title("Multi-Task Lasso: Design Influence on Hamiltonian Parameters")
        plt.xlabel("Hamiltonian Parameter")
        plt.ylabel("Design Parameter")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved heatmap to {save_path}")
        plt.show()


# =============================================================================
# KAN Dataset Preparation
# =============================================================================

def create_kan_dataset(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    device: str = "cpu",
    normalize: bool = True,
) -> tuple[dict[str, torch.Tensor], StandardScaler | None, StandardScaler | None]:
    """
    Create a dataset dictionary for KAN training with optional normalization.

    Args:
        X: Input features (N x D)
        y: Target values (N,) or (N, 1)
        train_ratio: Fraction of data for training
        device: Device to place tensors on
        normalize: Whether to normalize the data

    Returns:
        Tuple of (dataset dict, X_scaler, y_scaler)
    """
    y = y.reshape(-1, 1) if y.ndim == 1 else y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_ratio, random_state=42
    )

    X_scaler: StandardScaler | None = None
    y_scaler: StandardScaler | None = None

    if normalize:
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        X_train = X_scaler.fit_transform(X_train)
        X_test = X_scaler.transform(X_test)
        y_train = y_scaler.fit_transform(y_train)
        y_test = y_scaler.transform(y_test)

    dataset = {
        "train_input": torch.tensor(X_train, dtype=torch.float32).to(device),
        "train_label": torch.tensor(y_train, dtype=torch.float32).to(device),
        "test_input": torch.tensor(X_test, dtype=torch.float32).to(device),
        "test_label": torch.tensor(y_test, dtype=torch.float32).to(device),
    }

    return dataset, X_scaler, y_scaler


# =============================================================================
# Main Script
# =============================================================================

def main() -> None:
    """Main function to run the inverse design learning pipeline."""
    print("=" * 60)
    print("Inverse Design Map Learning with KANs")
    print("=" * 60)

    # Set random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Load training data
    data_path = Path(__file__).parent / "data" / "training_data.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found at {data_path}")

    print(f"\nLoading data from: {data_path}")
    training_df = pd.read_parquet(data_path)
    print(f"Loaded {len(training_df)} samples with {len(training_df.columns)} columns")

    # Define parameter sets
    hamiltonian_parameters = [
        "qubit_frequency_GHz",
        "anharmonicity_MHz",
        "cavity_frequency_GHz",
        "kappa_kHz",
        "g_MHz",
    ]
    design_parameters = [
        "cross_length",
        "claw_length",
        "coupling_length",
        "total_length",
        "ground_spacing",
    ]

    print(f"\nHamiltonian parameters: {hamiltonian_parameters}")
    print(f"Design parameters: {design_parameters}")

    # Extract data
    Y_hamiltonian = training_df[hamiltonian_parameters].values
    X_design = training_df[design_parameters].values

    print(f"\nDesign space shape: {X_design.shape}")
    print(f"Hamiltonian space shape: {Y_hamiltonian.shape}")

    # ==========================================================================
    # Step 1: Design-Relevance Encoder (LASSO)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Step 1: Design-Relevance Encoder (LASSO)")
    print("=" * 60)

    encoder = DesignRelevanceEncoder(
        X_design,
        Y_hamiltonian,
        design_parameters,
        hamiltonian_parameters,
    )

    lasso_coefs = encoder.run_multitask_lasso()
    print("\nLASSO Coefficients:")
    print(lasso_coefs.to_string())

    dependency_summary = encoder.get_dependency_summary(top_k=2, threshold=1e-3)
    print("\nDependency Summary (top 2 per Hamiltonian parameter):")
    print(json.dumps(dependency_summary, indent=2))

    # ==========================================================================
    # Step 2: KAN Model Setup
    # ==========================================================================
    print("\n" + "=" * 60)
    print("Step 2: KAN Model Setup")
    print("=" * 60)

    # Select target and input features based on LASSO results
    target_feature = "cavity_frequency_GHz"
    input_features = [
        item["parameter"]
        for item in dependency_summary["lasso"][target_feature]
    ]
    print(f"\nTarget: {target_feature}")
    print(f"Selected input features: {input_features}")

    # Prepare data for KAN
    X = training_df[input_features].values
    y = training_df[target_feature].values

    # Subsample for faster training
    subsample_size = min(5000, len(X))
    indices = np.random.choice(len(X), subsample_size, replace=False)
    X_sub = X[indices]
    y_sub = y[indices]

    print(f"\nUsing {subsample_size} samples for KAN training")

    # Try to import KAN
    try:
        from kan import KAN

        print("\nKAN library loaded successfully!")

        # Create normalized dataset (critical for KAN stability)
        dataset, X_scaler, y_scaler = create_kan_dataset(
            X_sub, y_sub, train_ratio=0.8, normalize=True
        )
        print(f"Train samples: {len(dataset['train_input'])}")
        print(f"Test samples: {len(dataset['test_input'])}")
        print("Data normalized for training stability")

        # Create KAN model
        n_inputs = len(input_features)
        model = KAN(width=[n_inputs, 3, 1], grid=5, k=3)
        print(f"\nKAN architecture: [{n_inputs}, 3, 1]")

        # Train with regularization for symbolic simplicity
        # Using lower lamb_entropy to prevent NaN issues
        print("\nTraining KAN (100 steps)...")
        metrics = model.fit(
            dataset,
            steps=100,
            lamb=0.001,  # Reduced L1 sparsity penalty
            lamb_entropy=2.0,  # Reduced entropy penalty
        )

        # Check for NaN before pruning
        with torch.no_grad():
            test_output = model(dataset["test_input"])
            if torch.isnan(test_output).any():
                print("Warning: NaN detected in model output, skipping prune")
            else:
                # Prune to simplify the model
                print("\nPruning model...")
                model = model.prune()
                model(dataset["train_input"])

                print("\nRetraining after pruning (50 steps)...")
                model.fit(dataset, steps=50, lamb=0.0001, lamb_entropy=1.0)

        # Check again for NaN before symbolic extraction
        with torch.no_grad():
            test_output = model(dataset["test_input"])

        if torch.isnan(test_output).any():
            print("\nWarning: Model has NaN outputs, cannot extract symbolic formula")
            print("This can happen with aggressive regularization or pruning.")
        else:
            # Extract symbolic formula
            print("\nExtracting symbolic formula...")
            lib = ["x", "x^2", "x^3", "x^4", "exp", "log", "sqrt", "tanh", "sin", "abs"]
            try:
                model.auto_symbolic(lib=lib)
                formula = model.symbolic_formula()
                print(f"\nSymbolic formula (normalized space): {formula}")
            except Exception as e:
                print(f"\nCould not extract symbolic formula: {e}")

        # Evaluate
        with torch.no_grad():
            train_pred = model(dataset["train_input"])
            test_pred = model(dataset["test_input"])

        if not torch.isnan(train_pred).any():
            train_mse = torch.mean((train_pred - dataset["train_label"]) ** 2).item()
            test_mse = torch.mean((test_pred - dataset["test_label"]) ** 2).item()
            print(f"\nTrain MSE (normalized): {train_mse:.6f}")
            print(f"Test MSE (normalized): {test_mse:.6f}")
        else:
            print("\nCannot compute MSE due to NaN values")

    except ImportError:
        print("\nKAN library not installed!")
        print("To install: pip install pykan")
        print("\nSkipping KAN training, but LASSO feature selection completed successfully.")

    print("\n" + "=" * 60)
    print("Pipeline completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
