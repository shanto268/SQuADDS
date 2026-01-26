import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from squadds.ml.utils import characterize_shape_function, get_top_features_from_ebm, prepare_features_with_interactions


class TestMLUtils(unittest.TestCase):
    def test_prepare_features_with_interactions(self):
        # Setup data
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        base_features = ["a", "b"]
        interaction_pairs = [("a", "b")]

        # Execute
        result = prepare_features_with_interactions(df, base_features, interaction_pairs)

        # Verify
        self.assertIn("a", result.columns)
        self.assertIn("b", result.columns)
        self.assertIn("a_x_b", result.columns)
        self.assertNotIn("c", result.columns)

        # Check values
        np.testing.assert_array_equal(result["a_x_b"], df["a"] * df["b"])

    def test_get_top_features_from_ebm(self):
        # Mock EBM result dict
        ebm_result = {"term_names": ["a", "b", "c", "a x b", "b x c"], "term_importances": [10.0, 5.0, 0.1, 8.0, 0.05]}
        design_params = ["a", "b", "c"]

        # Execute
        base, interactions = get_top_features_from_ebm(ebm_result, design_params, threshold=0.1)

        # Total importance = 10+5+0.1+8+0.05 = 23.15
        # Threshold 10% = 2.315
        # 'a': 10 > 2.315 -> Keep
        # 'b': 5 > 2.315 -> Keep
        # 'c': 0.1 < 2.315 -> Drop
        # 'a x b': 8 > 2.315 -> Keep interaction
        # 'b x c': 0.05 < 2.315 -> Drop

        self.assertIn("a", base)
        self.assertIn("b", base)
        self.assertNotIn("c", base)

        self.assertIn(("a", "b"), interactions)
        self.assertNotIn(("b", "c"), interactions)

    def test_characterize_shape_function_monotonic(self):
        # Mock EBM
        mock_ebm = MagicMock()
        mock_ebm.term_names_ = ["feature_a"]

        # Mock explain_global
        mock_explanation = MagicMock()
        # Create a monotonically increasing linear shape
        mock_data = {"scores": np.linspace(0, 10, 20)}
        mock_explanation.data.return_value = mock_data
        mock_ebm.explain_global.return_value = mock_explanation

        result = characterize_shape_function(mock_ebm, 0)

        self.assertEqual(result["type"], "linear increasing")
        self.assertAlmostEqual(result["effect_size"], 10.0)


if __name__ == "__main__":
    unittest.main()
