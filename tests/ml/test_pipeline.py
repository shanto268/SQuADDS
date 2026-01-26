import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# Mock pysr availability if needed, but pipeline imports it from symbolic.py
# If symbolic.py handles ImportError correctly, we just need to mock SymbolicRegressor class
# so we don't actually run Julia.
from squadds.ml.pipeline import SQuADDSAnalysisPipeline


class TestAnalysisPipeline(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"f1": np.random.rand(20), "f2": np.random.rand(20), "target1": np.random.rand(20)})
        self.feature_cols = ["f1", "f2"]
        self.target_cols = ["target1"]

    @patch("squadds.ml.pipeline.EBMAnalyzer")
    @patch("squadds.ml.pipeline.SymbolicRegressor")
    def test_pipeline_flow(self, MockSymbolic, MockEBM):
        # Setup EBM Mock
        mock_ebm = MagicMock()
        mock_ebm.evaluate.return_value = {"r2": 0.8}
        mock_ebm.get_top_features.return_value = (["f1"], [("f1", "f2")])
        MockEBM.return_value = mock_ebm

        # Setup Symbolic Mock
        mock_sym = MagicMock()
        mock_sym.evaluate.return_value = {"r2": 0.9}
        mock_sym.get_best_equation.return_value = "f1 * f2"
        MockSymbolic.return_value = mock_sym

        # Run pipeline
        pipeline = SQuADDSAnalysisPipeline()
        results = pipeline.analyze(
            self.df,
            self.feature_cols,
            self.target_cols,
            ebm_kwargs={"interactions": 2},
            symbolic_kwargs={"niterations": 5},
        )

        # Verify
        self.assertIn("target1", results)
        res = results["target1"]

        self.assertEqual(res["best_equation"], "f1 * f2")
        self.assertEqual(res["selected_features"], ["f1"])
        self.assertEqual(res["interaction_pairs"], [("f1", "f2")])

        # Verify calls
        mock_ebm.fit.assert_called_once()
        mock_sym.fit.assert_called_once()

        # Verify that symbolic regressor was trained on interactions
        # We can check the dataframe passed to fit
        args, _ = mock_sym.fit.call_args
        X_sym_train = args[0]
        self.assertIn("f1_x_f2", X_sym_train.columns)


if __name__ == "__main__":
    unittest.main()
