"""Tests for feature engineering and honest train/test splits."""

import unittest

import numpy as np

from retentionlab.clean import clean_customers
from retentionlab.datagen import generate_customers
from retentionlab.features import build_features


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = generate_customers(n=1200, seed=11)
        cleaned, _ = clean_customers(raw)
        cls.cleaned = cleaned
        cls.fs = build_features(cleaned, seed=11)

    def test_split_sizes_and_stratification(self):
        n = len(self.cleaned)
        expected_test = int(round(0.2 * n))
        self.assertAlmostEqual(expected_test, len(self.fs.y_test), delta=3)
        train_rate = self.fs.y_train.mean()
        test_rate = self.fs.y_test.mean()
        self.assertAlmostEqual(train_rate, test_rate, delta=0.02)

    def test_standardization_uses_train_statistics(self):
        means = self.fs.X_train.mean(axis=0)
        stds = self.fs.X_train.std(axis=0)
        self.assertTrue(np.all(np.abs(means) < 1e-9))
        self.assertTrue(np.all(np.abs(stds - 1.0) < 1e-9))

    def test_feature_names_include_engineered_and_encoded_columns(self):
        names = self.fs.feature_names
        for expected in ("engagement_score", "spend_ratio", "age_quadratic",
                         "region_south", "signup_channel_referral", "plan_premium"):
            self.assertIn(expected, names)

    def test_engagement_score_tracks_usage(self):
        from retentionlab.features import engagement_pca

        scores = engagement_pca(self.cleaned)
        correlation = np.corrcoef(scores, self.cleaned["logins_last_month"])[0, 1]
        self.assertGreater(correlation, 0.5)

    def test_no_nan_in_feature_matrices(self):
        self.assertFalse(np.isnan(self.fs.X_train).any())
        self.assertFalse(np.isnan(self.fs.X_test).any())


if __name__ == "__main__":
    unittest.main()
