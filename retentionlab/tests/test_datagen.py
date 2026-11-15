"""Tests for the synthetic data generator and its injected defects."""

import unittest

import pandas as pd

from retentionlab.datagen import generate_customers


class DataGenTests(unittest.TestCase):
    def test_seed_is_deterministic(self):
        first = generate_customers(n=500, seed=7)
        second = generate_customers(n=500, seed=7)
        pd.testing.assert_frame_equal(first, second)

    def test_shape_and_columns(self):
        df = generate_customers(n=500, seed=3)
        self.assertEqual((515, 14), df.shape)  # 500 rows + 15 duplicates
        for column in (
            "customer_id", "age", "region", "signup_channel", "plan",
            "tenure_months", "monthly_charge", "discount_rate",
            "logins_last_month", "support_tickets", "watch_hours",
            "churned", "risk_score", "data_source",
        ):
            self.assertIn(column, df.columns)

    def test_churn_rate_is_in_a_sensible_range(self):
        df = generate_customers(n=6000, seed=42)
        self.assertGreaterEqual(df["churned"].mean(), 0.18)
        self.assertLessEqual(df["churned"].mean(), 0.38)

    def test_defects_are_present(self):
        df = generate_customers(n=6000, seed=42)
        self.assertEqual(15, int(df.duplicated().sum()))
        self.assertGreater(df["watch_hours"].isna().sum(), 0)
        self.assertGreater(df["support_tickets"].isna().sum(), 0)
        self.assertGreater(df["monthly_charge"].isna().sum(), 0)
        # leakage column correlates strongly with the target by construction
        self.assertGreater(df["risk_score"].corr(df["churned"]), 0.5)
        # label noise: non-canonical region spellings exist
        self.assertTrue((~df["region"].isin(["north", "south", "east", "west"])).any())
        # impossible values
        self.assertIn(0.0, df["age"].to_numpy())
        self.assertIn(-1, df["tenure_months"].to_numpy())
        # data-entry outliers: watch_hours far beyond the typical scale
        self.assertGreater(df["watch_hours"].max(), 300)


if __name__ == "__main__":
    unittest.main()
