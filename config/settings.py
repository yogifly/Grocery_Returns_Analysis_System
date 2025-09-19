"""Configuration settings for the grocery returns analysis project"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Data file paths
ORDERS_FILE = DATA_DIR / "orders.csv"
RETURNS_FILE = DATA_DIR / "returns.csv"
PRODUCTS_FILE = DATA_DIR / "products.csv"

# Analysis parameters
RETURN_RATE_THRESHOLD = 0.25  # Alert threshold for high return rates
SHELF_LIFE_CONFIDENCE = 0.95  # Confidence level for survival analysis
MIN_SAMPLE_SIZE = 10  # Minimum sample size for analysis

# Dashboard configuration
DASHBOARD_TITLE = "Grocery Returns Analysis Dashboard"
DEFAULT_DATE_RANGE = 90  # Default date range in days

# Color scheme for visualizations
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#F18F01',
    'warning': '#C73E1D',
    'info': '#6C757D',
    'light': '#F8F9FA',
    'dark': '#212529'
}

# Product categories
TEMP_SENSITIVE_PRODUCTS = ['BERRIES_001', 'AVOCADO_001', 'SPINACH_001', 'LETTUCE_001']
HIGH_RISK_CITIES = ['Mumbai', 'Chennai']  # Coastal cities with higher humidity
