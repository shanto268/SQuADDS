from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from squadds.ml.ebm import EBMAnalyzer
from squadds.ml.symbolic import SymbolicRegressor
from squadds.ml.utils import prepare_features_with_interactions


class SQuADDSAnalysisPipeline:
    """
    End-to-end pipeline for Explainable Inverse Design Analysis.
    Integrates EBM feature selection and PySR equation discovery.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.results = {}

    def analyze(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_cols: list[str],
        test_size: float = 0.2,
        ebm_kwargs: dict[str, Any] | None = None,
        symbolic_kwargs: dict[str, Any] | None = None,
        feature_threshold: float = 0.05,
    ) -> dict[str, Any]:
        """
        Run the analysis for each target column.

        Args:
            df: Input dataframe
            feature_cols: List of design parameter names
            target_cols: List of target parameter names (e.g. ['qubit_frequency_GHz'])
            test_size: Fraction of data to use for testing
            ebm_kwargs: Dict of arguments for EBMAnalyzer
            symbolic_kwargs: Dict of arguments for SymbolicRegressor
            feature_threshold: Threshold for EBM feature selection

        Returns:
            Dictionary mapping target_col -> result_dict
        """
        ebm_kwargs = ebm_kwargs or {}
        symbolic_kwargs = symbolic_kwargs or {}

        # Split data once
        X = df[feature_cols]
        # We handle targets individually

        for target in target_cols:
            print(f"Analyzing target: {target}...")
            y = df[target]

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_state
            )

            # 1. Train EBM
            print("  Training EBM for feature selection...")
            ebm = EBMAnalyzer(random_state=self.random_state, **ebm_kwargs)
            ebm.fit(X_train, y_train)
            ebm_metrics = ebm.evaluate(X_test, y_test)

            # 2. Extract Top Features
            base_features, interaction_pairs = ebm.get_top_features(threshold=feature_threshold)
            print(f"  Selected features: {base_features}")
            print(f"  Selected interactions: {interaction_pairs}")

            # 3. Prepare features for Symbolic Regression
            # We must use ONLY the selected base features and construct the interactions
            X_train_sym = prepare_features_with_interactions(X_train, base_features, interaction_pairs)
            X_test_sym = prepare_features_with_interactions(X_test, base_features, interaction_pairs)

            # 4. Train Symbolic Regressor
            print("  Training Symbolic Regressor...")
            sym_reg = SymbolicRegressor(**symbolic_kwargs)
            sym_reg.fit(X_train_sym, y_train)
            sym_metrics = sym_reg.evaluate(X_test_sym, y_test)
            best_eqn = sym_reg.get_best_equation()

            print(f"  Best Equation: {best_eqn}")

            self.results[target] = {
                "ebm_model": ebm,
                "ebm_metrics": ebm_metrics,
                "selected_features": base_features,
                "interaction_pairs": interaction_pairs,
                "symbolic_model": sym_reg,
                "symbolic_metrics": sym_metrics,
                "best_equation": best_eqn,
            }

        return self.results
