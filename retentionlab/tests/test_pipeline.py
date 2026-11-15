"""End-to-end pipeline test: every stage runs and every artifact appears."""

import argparse
import tempfile
import unittest
from pathlib import Path

from retentionlab import config
from retentionlab.cli import cmd_pipeline


class PipelineTests(unittest.TestCase):
    def test_full_pipeline_produces_all_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = argparse.Namespace(
                command="pipeline",
                data_dir=root / "data",
                outputs=root / "outputs",
                seed=42,
                n=600,
            )
            self.assertEqual(0, cmd_pipeline(args))

            data_dir = root / "data"
            outputs = root / "outputs"
            for filename in (config.RAW_FILE, config.CLEAN_FILE, config.FEATURES_FILE):
                self.assertTrue((data_dir / filename).exists(), filename)
            for filename in (
                config.PROFILE_FILE,
                config.CLEANING_REPORT_FILE,
                config.EFFECT_SIZES_FILE,
                config.MODEL_COMPARISON_FILE,
                config.SUMMARY_FILE,
                config.REPORT_FILE,
            ):
                self.assertTrue((outputs / filename).exists(), filename)
            for index in range(1, 13):
                self.assertTrue(
                    list(outputs.glob(f"fig_{index:02d}_*.png")),
                    f"figure {index} missing",
                )

            summary = (outputs / config.SUMMARY_FILE).read_text(encoding="utf-8")
            self.assertIn("Test AUC", summary)

            report = (outputs / config.REPORT_FILE).read_text(encoding="utf-8")
            self.assertIn("RetentionLab", report)
            self.assertIn("<img", report)


if __name__ == "__main__":
    unittest.main()
