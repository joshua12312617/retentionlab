"""Tests for the from-scratch statistics and metrics."""

import unittest

import numpy as np
import pandas as pd

from retentionlab.stats import (
    auc,
    cohens_d,
    cramers_v,
    point_biserial,
    pr_curve,
    roc_auc,
    roc_curve,
    youden_threshold,
)


class StatsTests(unittest.TestCase):
    def test_cramers_v_perfect_association_is_one(self):
        a = pd.Series(["x", "x", "y", "y"])
        b = pd.Series([0, 0, 1, 1])
        self.assertAlmostEqual(1.0, cramers_v(a, b), places=6)

    def test_cramers_v_independent_is_near_zero(self):
        rng = np.random.default_rng(0)
        a = pd.Series(rng.integers(0, 3, 2000))
        b = pd.Series(rng.integers(0, 3, 2000))
        self.assertLess(cramers_v(a, b), 0.05)

    def test_point_biserial_direction(self):
        binary = pd.Series([0, 0, 1, 1])
        up = pd.Series([1.0, 2.0, 3.0, 4.0])
        down = pd.Series([4.0, 3.0, 2.0, 1.0])
        self.assertGreater(point_biserial(binary, up), 0.0)
        self.assertLess(point_biserial(binary, down), 0.0)

    def test_cohens_d_zero_for_identical_groups(self):
        self.assertAlmostEqual(0.0, cohens_d(np.ones(10), np.ones(10)), places=9)

    def test_auc_of_a_line(self):
        self.assertAlmostEqual(0.5, auc(np.array([0.0, 1.0]), np.array([0.0, 1.0])), places=9)

    def test_roc_auc_perfect_and_inverted(self):
        y = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(1.0, roc_auc(y, np.array([0.1, 0.2, 0.9, 0.8])), places=9)
        self.assertAlmostEqual(0.0, roc_auc(y, np.array([0.9, 0.8, 0.1, 0.2])), places=9)

    def test_roc_and_pr_curve_endpoints(self):
        y = np.array([0, 1, 1, 0])
        scores = np.array([0.2, 0.8, 0.7, 0.3])
        fpr, tpr, _ = roc_curve(y, scores)
        self.assertEqual(0.0, fpr[0])
        self.assertEqual(1.0, fpr[-1])
        self.assertEqual(1.0, tpr[-1])
        precision, recall, _ = pr_curve(y, scores)
        self.assertEqual(0.0, recall[0])
        self.assertGreaterEqual(recall[-1], 0.99)

    def test_youden_threshold_picks_the_split(self):
        # Scores 0.9 and 0.8 belong to churners; 0.2 and 0.1 do not.  The
        # threshold must sit at 0.8: any lower and the 0.2 non-churner is
        # included, costing a false positive for no extra recall.
        y = np.array([0, 1, 0, 1])
        scores = np.array([0.1, 0.9, 0.2, 0.8])
        threshold, j = youden_threshold(y, scores)
        self.assertAlmostEqual(0.8, threshold, places=6)
        self.assertAlmostEqual(1.0, j, places=9)


if __name__ == "__main__":
    unittest.main()
