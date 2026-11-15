"""Central configuration for RetentionLab: paths, constants, and schemas."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_SEED = 42
DEFAULT_N_CUSTOMERS = 6000

RAW_FILE = "raw_customers.csv"
CLEAN_FILE = "customers_clean.csv"
FEATURES_FILE = "features.csv"
PROFILE_FILE = "01_profile.txt"
CLEANING_REPORT_FILE = "02_cleaning_report.txt"
EFFECT_SIZES_FILE = "05_effect_sizes.csv"
MODEL_COMPARISON_FILE = "04_model_comparison.csv"
SUMMARY_FILE = "summary.txt"
REPORT_FILE = "report.html"

TARGET = "churned"
ID_COLUMN = "customer_id"

CATEGORICALS = ["region", "signup_channel", "plan"]
NUMERICS = [
    "age",
    "tenure_months",
    "monthly_charge",
    "discount_rate",
    "logins_last_month",
    "support_tickets",
    "watch_hours",
]
ENGAGEMENT_INPUTS = ["logins_last_month", "watch_hours"]

# Canonical labels used by the generator and expected by the cleaner.
REGIONS = ["north", "south", "east", "west"]
CHANNELS = ["organic", "referral", "partner", "ads"]
PLANS = ["basic", "standard", "premium"]
PLAN_BASE_CHARGE = {"basic": 9.0, "standard": 15.0, "premium": 25.0}

# Business parameters for the profit-vs-threshold analysis.
COST_OF_RETENTION_OFFER = 15.0  # cost per customer targeted with an offer
VALUE_OF_RETAINED_CUSTOMER = 120.0  # expected value of saving one churner

# Cleaning thresholds.
LEAKAGE_CORRELATION = 0.6  # drop non-target columns with |corr| >= this
WATCH_HOURS_CAP_QUANTILE = 0.99  # winsorize above this quantile
