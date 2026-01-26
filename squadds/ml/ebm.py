from typing import Any

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from squadds.ml.utils import characterize_shape_function, get_top_features_from_ebm


class EBMAnalyzer:
    """
    Wrapper around DescribeableBoostingRegressor (EBM) with SQuADDS-specific analysis capabilities.
    """

    def __init__(
        self,
        random_state: int = 42,
        interactions: int = 10,
        outer_bags: int = 8,
        inner_bags: int = 0,
        learning_rate: float = 0.01,
        max_bins: int = 256,
        max_leaves: int = 3,
        **kwargs,
    ):
        """
        Initialize the EBM Analyzer.

        Args:
            random_state: Seed for reproducibility
            interactions: Number of interaction terms to search for
            outer_bags: Number of outer bags (bootstraps)
            inner_bags: Number of inner bags
            learning_rate: Learning rate for boosting
            max_bins: Max number of bins per feature
            max_leaves: Max count of leaves
            **kwargs: Additional arguments for ExplainableBoostingRegressor
        """
        self.model = ExplainableBoostingRegressor(
            random_state=random_state,
            interactions=interactions,
            outer_bags=outer_bags,
            inner_bags=inner_bags,
            learning_rate=learning_rate,
            max_bins=max_bins,
            max_leaves=max_leaves,
            **kwargs,
        )
        self.is_fitted = False
        self.feature_names_in_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "EBMAnalyzer":
        """
        Train the EBM model.

        Args:
            X: Feature dataframe
            y: Target values
        """
        self.feature_names_in_ = list(X.columns)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict target values."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self.model.predict(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
        """
        Evaluate model performance.

        Returns:
            Dictionary with R2, RMSE, MAE metrics.
        """
        y_pred = self.predict(X_test)
        return {
            "r2": r2_score(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
        }

    def get_top_features(self, threshold: float = 0.05) -> tuple[list[str], list[tuple[str, str]]]:
        """
        Extract important features and interactions.

        Args:
            threshold: Importance threshold (relative) to keep features.

        Returns:
            Tuple of (base_features, interaction_pairs)
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")

        return get_top_features_from_ebm(self.model, design_params=self.feature_names_in_, threshold=threshold)

    def characterize_term(self, term_name: str | None = None, term_idx: int | None = None) -> dict[str, Any]:
        """
        Characterize the shape of a learned function for a specific term.

        Args:
            term_name: Name of the term (feature)
            term_idx: Index of the term in model.term_names_

        Returns:
            Dictionary with shape characterization.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")

        if term_name is None and term_idx is None:
            raise ValueError("Must provide either term_name or term_idx")

        if term_idx is None:
            try:
                term_idx = self.model.term_names_.index(term_name)
            except ValueError as err:
                raise ValueError(f"Term '{term_name}' not found in model terms.") from err

        return characterize_shape_function(self.model, term_idx)
