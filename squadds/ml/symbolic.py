import numpy as np
import pandas as pd

try:
    from pysr import PySRRegressor

    PYSR_AVAILABLE = True
except ImportError:
    PYSR_AVAILABLE = False

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class SymbolicRegressor:
    """
    Wrapper around PySRRegressor with SQuADDS-specific defaults for physics-informed equation discovery.
    """

    def __init__(
        self,
        binary_operators: list[str] | None = None,
        unary_operators: list[str] | None = None,
        niterations: int = 40,
        populations: int = 15,
        population_size: int = 33,
        maxsize: int = 20,
        parsimony: float = 0.0032,
        model_selection: str = "best",
        **kwargs,
    ):
        """
        Initialize the Symbolic Regressor.

        Args:
            binary_operators: List of binary operators. Defaults to ["+", "-", "*", "/"].
            unary_operators: List of unary operators. Defaults to ["square", "sqrt", "inv"].
            niterations: Number of iterations (generations).
            populations: Number of populations.
            population_size: Size of each population.
            maxsize: Max complexity of equation.
            parsimony: Penalty for complexity.
            model_selection: Strategy to select the best model ('best', 'accuracy', 'score').
            **kwargs: Additional args for PySRRegressor.
        """
        if not PYSR_AVAILABLE:
            raise ImportError(
                "PySR is not installed. Please install with 'pip install pysr' or 'pip install squadds[ml]'."
            )

        self.binary_operators = binary_operators or ["+", "-", "*", "/"]
        self.unary_operators = unary_operators or ["square", "sqrt", "inv"]

        self.model = PySRRegressor(
            binary_operators=self.binary_operators,
            unary_operators=self.unary_operators,
            niterations=niterations,
            populations=populations,
            population_size=population_size,
            maxsize=maxsize,
            parsimony=parsimony,
            model_selection=model_selection,
            **kwargs,
        )
        self.is_fitted = False
        self.feature_names_in_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SymbolicRegressor":
        """
        Train the symbolic regression model.

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

    def get_best_equation(self) -> str:
        """Return the string representation of the best equation found."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        return str(self.model.sympy())

    def get_equations(self) -> pd.DataFrame:
        """Return the dataframe of all equations found (Pareto frontier)."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        return self.model.equations_
