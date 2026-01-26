import unittest

import numpy as np
import pandas as pd

from squadds.ml.pipeline import SQuADDSAnalysisPipeline


class TestMLIntegration(unittest.TestCase):
    def setUp(self):
        # Create synthetic dataset with known relationship
        np.random.seed(42)
        n_samples = 50
        self.df = pd.DataFrame(
            {
                "x": np.linspace(1, 10, n_samples),
                "z": np.random.rand(n_samples),  # noise feature
            }
        )
        # Target y = x^2
        self.df["y"] = self.df["x"] ** 2

        self.feature_cols = ["x", "z"]
        self.target_cols = ["y"]

    def test_end_to_end(self):
        pipeline = SQuADDSAnalysisPipeline(random_state=42)

        # Use minimal parameters for speed
        results = pipeline.analyze(
            self.df,
            self.feature_cols,
            self.target_cols,
            test_size=0.2,
            ebm_kwargs={"interactions": 0, "outer_bags": 1, "max_bins": 32},  # Fast EBM
            symbolic_kwargs={
                "niterations": 20,
                "populations": 5,
                "population_size": 20,
                "parsimony": 0.01,
            },
            feature_threshold=0.01,
        )

        self.assertIn("y", results)
        res = results["y"]

        # Validate that 'x' was selected
        self.assertIn("x", res["selected_features"])

        # Check equation - might not be exactly x^2 due to low iterations, but valid string
        print(f"Discovered equation: {res['best_equation']}")
        self.assertTrue(isinstance(res["best_equation"], str))
        self.assertTrue(len(res["best_equation"]) > 0)

        # Check metrics
        self.assertTrue(res["ebm_metrics"]["r2"] > 0.0)  # Should fit reasonably well


if __name__ == "__main__":
    unittest.main()
