import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# Mock pysr availability
sys.modules["pysr"] = MagicMock()

from squadds.ml.symbolic import SymbolicRegressor  # noqa: E402


class TestSymbolicRegressor(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        self.y = pd.Series([10, 20, 30])

    @patch("squadds.ml.symbolic.PySRRegressor")
    @patch("squadds.ml.symbolic.PYSR_AVAILABLE", True)
    def test_init_and_fit(self, MockPySR):
        # Setup mock
        mock_model = MagicMock()
        MockPySR.return_value = mock_model

        regressor = SymbolicRegressor(niterations=5)
        regressor.fit(self.X, self.y)

        self.assertTrue(regressor.is_fitted)
        self.assertEqual(regressor.feature_names_in_, ["a", "b"])
        mock_model.fit.assert_called_once()

    @patch("squadds.ml.symbolic.PySRRegressor")
    @patch("squadds.ml.symbolic.PYSR_AVAILABLE", True)
    def test_predict_evaluate(self, MockPySR):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([10, 20, 30])
        MockPySR.return_value = mock_model

        regressor = SymbolicRegressor()
        regressor.fit(self.X, self.y)

        metrics = regressor.evaluate(self.X, self.y)
        self.assertEqual(metrics["r2"], 1.0)
        self.assertEqual(metrics["rmse"], 0.0)

    @patch("squadds.ml.symbolic.PySRRegressor")
    @patch("squadds.ml.symbolic.PYSR_AVAILABLE", True)
    def test_get_equation(self, MockPySR):
        mock_model = MagicMock()
        mock_model.sympy.return_value = "x0 + x1"
        MockPySR.return_value = mock_model

        regressor = SymbolicRegressor()
        regressor.fit(self.X, self.y)

        eq = regressor.get_best_equation()
        self.assertEqual(eq, "x0 + x1")


if __name__ == "__main__":
    unittest.main()
