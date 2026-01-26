import unittest

import numpy as np
import pandas as pd

from squadds.ml.ebm import EBMAnalyzer


class TestEBMAnalyzer(unittest.TestCase):
    def setUp(self):
        # Create synthetic dataset
        np.random.seed(42)
        n_samples = 100
        self.X = pd.DataFrame(
            {
                "f1": np.random.rand(n_samples),
                "f2": np.random.rand(n_samples),
                "f3": np.random.rand(n_samples),  # Irrelevant feature
            }
        )
        # Target depends on f1 and f2
        self.y = 2 * self.X["f1"] + 3 * self.X["f2"] ** 2 + np.random.normal(0, 0.1, n_samples)

    def test_init(self):
        analyzer = EBMAnalyzer(interactions=2, random_state=123)
        self.assertFalse(analyzer.is_fitted)
        self.assertEqual(analyzer.model.interactions, 2)
        self.assertEqual(analyzer.model.random_state, 123)

    def test_fit_predict_metrics(self):
        analyzer = EBMAnalyzer(interactions=0, outer_bags=1, inner_bags=0)  # Fast training
        analyzer.fit(self.X, self.y)

        self.assertTrue(analyzer.is_fitted)

        preds = analyzer.predict(self.X)
        self.assertEqual(len(preds), len(self.X))

        stats = analyzer.evaluate(self.X, self.y)
        self.assertIn("r2", stats)
        self.assertIn("rmse", stats)
        # Should be decent fit
        self.assertGreater(stats["r2"], 0.5)

    def test_feature_extraction(self):
        analyzer = EBMAnalyzer(interactions=0, outer_bags=1)
        analyzer.fit(self.X, self.y)

        # f1 and f2 should be important
        base, interactions = analyzer.get_top_features(threshold=0.01)
        self.assertIn("f1", base)
        self.assertIn("f2", base)
        # Should be no interactions as we set interactions=0
        self.assertEqual(len(interactions), 0)


if __name__ == "__main__":
    unittest.main()
