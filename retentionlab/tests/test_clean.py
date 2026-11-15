"""Tests for the cleaning pipeline against known injected defects."""

import unittest

import numpy as np

from retentionlab import config
from retentionlab.clean import clean_customers
from retentionlab.datagen import generate_customers


class CleanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = generate_customers(n=6000, seed=42)
        cls.cleaned, cls.report = clean_customers(cls.raw)

    def test_row_counts_are_exact(self):
        self.assertEqual(15, self.report.duplicates_removed)
        self.assertEqual(9, self.report.rows_dropped_invalid)
        self.assertEqual(len(self.raw) - 15 - 9, len(self.cleaned))

    def test_leakage_identifier_and_constant_columns_dropped(self):
        dropped = dict(self.report.columns_dropped)
        self.assertIn("risk_score", dropped)
        self.assertIn("leakage", dropped["risk_score"])
        self.assertIn("customer_id", dropped)
        self.assertIn("data_source", dropped)

    def test_no_missing_numeric_values_remain(self):
        for column in config.NUMERICS:
            self.assertEqual(0, self.cleaned[column].isna().sum(), column)

    def test_impossible_values_removed(self):
        self.assertTrue((self.cleaned["age"] > 0).all())
        self.assertTrue((self.cleaned["tenure_months"] >= 0).all())

    def test_labels_are_canonical(self):
        self.assertTrue(self.cleaned["region"].isin(config.REGIONS).all())
        self.assertTrue(self.cleaned["plan"].isin(config.PLANS).all())
        self.assertTrue(self.cleaned["signup_channel"].isin(config.CHANNELS).all())

    def test_imputation_and_winsorizing_happened(self):
        self.assertGreater(self.report.values_imputed.get("watch_hours", 0), 0)
        self.assertGreater(self.report.values_winsorized, 0)
        self.assertIn("watch_hours_flagged", self.cleaned.columns)
        # Capping must reduce the maximum, and every capped row is flagged.
        self.assertLessEqual(
            self.cleaned["watch_hours"].max(),
            self.raw["watch_hours"].max() + 1e-9,
        )
        self.assertEqual(
            self.report.values_winsorized,
            int((self.cleaned["watch_hours_flagged"] == 1).sum()),
        )

    def test_churn_rate_survives_cleaning(self):
        before = self.raw["churned"].mean()
        after = self.cleaned["churned"].mean()
        self.assertAlmostEqual(before, after, delta=0.01)


if __name__ == "__main__":
    unittest.main()
