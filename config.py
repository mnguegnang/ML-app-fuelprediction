"""
Configuration file for Fuel Consumption Prediction Application
All hardcoded values, paths, and parameters are centralized here
"""

import os

# =============================================================================
# APPLICATION SETTINGS
# =============================================================================

# Security
SECRET_KEY = 'your_secret_key_here_change_in_production'  # CHANGE IN PRODUCTION!

# Server Configuration
SERVER_PORT = 6003
DEBUG_MODE = True
SERVER_HOST = '0.0.0.0'  # Use '127.0.0.1' for localhost only

# =============================================================================
# FILE PATHS AND DIRECTORIES
# =============================================================================

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, 'pkl_objects')
MODEL_FILENAME = 'Randomforest.pkl'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

# Upload configuration
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'ods'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

# Logging configuration
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
LOG_FILENAME_FORMAT = 'fuel_app_%Y%m%d.log'

# =============================================================================
# DATA COLUMN NAMES
# =============================================================================

# Required input fields
REQUIRED_FIELDS = [
    'Cluster',
    'SITE Name',
    'PREVIOUS FUEL QTE',
    'QTE FUEL FOUND',
    'TYPE OF GENERATOR',
    'RUNNING TIME',
    'CONSUMPTION RATE',
    'NUMBER OF DAYS',
    'FUEL ADDED'
]

# Column mapping for flexibility (maps standard names to possible alternatives)
COLUMN_ALIASES = {
    'Cluster': ['cluster', 'CLUSTER', 'Cluster'],
    'SITE Name': ['site name', 'SITE NAME', 'Site Name', 'site'],
    'PREVIOUS FUEL QTE': ['previous fuel qte', 'PREVIOUS FUEL QTE', 'Previous Fuel Qty'],
    'QTE FUEL FOUND': ['qte fuel found', 'QTE FUEL FOUND', 'Qty Fuel Found'],
    'TYPE OF GENERATOR': ['type of generator', 'TYPE OF GENERATOR', 'Generator Type'],
    'RUNNING TIME': ['running time', 'RUNNING TIME', 'Runtime'],
    'CONSUMPTION RATE': ['consumption rate', 'CONSUMPTION RATE', 'Consumption'],
    'NUMBER OF DAYS': ['number of days', 'NUMBER OF DAYS', 'Days'],
    'FUEL ADDED': ['fuel added', 'FUEL ADDED', 'Added Fuel']
}

# Generator capacity column
GENERATOR_CAPACITY_COLUMN = 'GENERATOR 1 CAPACITY (KVA)'

# Required feature after one-hot encoding
REQUIRED_MODEL_FEATURE = 'GENERATOR 1 CAPACITY (KVA)_6,5 x 2'

# Calculated column
FUEL_PER_PERIOD_COLUMN = 'Fuel_per_period'

# Historical consumption column (for NSE calculation)
HISTORICAL_CONSUMPTION_COLUMN = 'CONSUMPTION HIS'

# =============================================================================
# MODEL PREDICTION FEATURES
# =============================================================================

# Features used in model prediction (order matters)
MODEL_FEATURES = [
    'Fuel_per_period',
    'RUNNING TIME',
    'CONSUMPTION RATE',
    'NUMBER OF DAYS',
    'FUEL ADDED',
    'GENERATOR 1 CAPACITY (KVA)_6,5 x 2'
]

# =============================================================================
# ANALYSIS THRESHOLDS AND PARAMETERS
# =============================================================================

# Anomaly detection
ANOMALY_STD_MULTIPLIER = 2  # Number of standard deviations for anomaly threshold
ANOMALY_UPPER_MULTIPLIER = 2  # For high anomalies
ANOMALY_LOWER_MULTIPLIER = 2  # For low anomalies

# Visualization limits
TOP_SITES_LIMIT = 20  # Number of top sites to display
TOP_CLUSTERS_LIMIT = 10  # Number of top clusters to display/export
COMPARISON_TOP_SITES = 10  # Number of top sites per cluster in comparison view

# Chart configuration
DISTRIBUTION_BINS = 20  # Number of bins in distribution histogram
MOVING_AVERAGE_WINDOW = 7  # Default window for moving average (can be dynamic)

# Export settings
PNG_EXPORT_DPI = 150  # DPI for PNG chart exports

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Log levels
LOG_LEVEL_CONSOLE = 'DEBUG'
LOG_LEVEL_FILE = 'INFO'

# Log format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# =============================================================================
# CHART COLORS AND STYLING
# =============================================================================

# Color palettes for different chart types
SITE_COLORS = [
    '#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', 
    '#00f2fe', '#43e97b', '#38f9d7', '#fa709a', '#fee140',
    '#30cfd0', '#a8edea', '#ff9a9e', '#fecfef', '#ffecd2',
    '#fcb69f', '#ff6e7f', '#bfe9ff', '#c1dfc4', '#deaaff'
]

CLUSTER_COLORS = [
    '#7986cb', '#4fc3f7', '#4db6ac', '#81c784', '#dce775',
    '#ffd54f', '#ffb74d', '#ff8a65', '#e57373', '#f06292'
]

# Default style parameters
DEFAULT_CHART_STYLE = {
    'value_font_size': 10,
    'value_label_font_size': 10,
    'show_legend': True,
    'x_label_rotation': 60
}

# =============================================================================
# VALIDATION RULES
# =============================================================================

# Data validation
MIN_DATA_ROWS = 1  # Minimum rows required for prediction
MIN_VALID_OBSERVATIONS_FOR_NSE = 1  # Minimum valid observations for NSE calculation

# File validation
MIN_FILE_SIZE = 100  # Minimum file size in bytes
MAX_SHEET_NAME_LENGTH = 100

# =============================================================================
# CACHE SETTINGS
# =============================================================================

# Cache timeouts (in seconds) - currently not used but available for future
DATA_CACHE_TIMEOUT = 3600  # 1 hour
PREDICTION_CACHE_TIMEOUT = 1800  # 30 minutes

# =============================================================================
# ERROR MESSAGES
# =============================================================================

ERROR_MESSAGES = {
    'model_not_found': f"Model file not found at {{path}}",
    'no_valid_sheet': "No valid sheet found in Excel file",
    'dataframe_empty': "DataFrame is empty after loading",
    'missing_required_field': "Missing required field: {field}",
    'column_not_found': "Columns not found in data: {columns}",
    'no_valid_data': "No valid data available for prediction after filtering",
    'required_feature_missing': f"Model requires feature '{REQUIRED_MODEL_FEATURE}' which is not present",
    'file_too_large': f"File size exceeds maximum allowed size of {MAX_FILE_SIZE / (1024*1024):.0f} MB",
    'invalid_file_extension': f"Invalid file extension. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
    'no_file_uploaded': "No file was uploaded",
    'prediction_cache_empty': "No predictions available. Please generate predictions first.",
    'cairosvg_not_installed': "cairosvg not installed - PNG export will not be available. Install with: pip install cairosvg"
}

# =============================================================================
# SUCCESS MESSAGES
# =============================================================================

SUCCESS_MESSAGES = {
    'model_loaded': "Model loaded successfully",
    'file_uploaded': "File uploaded successfully",
    'predictions_generated': "Predictions generated successfully",
    'export_completed': "Export completed successfully",
    'cache_cleared': "Cache cleared successfully"
}

# =============================================================================
# FEATURE FLAGS (for enabling/disabling features)
# =============================================================================

FEATURE_FLAGS = {
    'enable_nse_calculation': True,
    'enable_anomaly_detection': True,
    'enable_csv_export': True,
    'enable_excel_export': True,
    'enable_summary_export': True,
    'enable_svg_export': True,
    'enable_png_export': True,  # Requires cairosvg
    'enable_cache': True,
    'enable_metrics_tracking': True,
    'enable_file_logging': True
}

# =============================================================================
# MODEL METADATA
# =============================================================================

MODEL_METADATA = {
    'name': 'Random Forest Fuel Predictor',
    'version': '2.1.0',
    'algorithm': 'Random Forest Regressor',
    'training_date': 'July 2018',
    'last_updated': 'January 2019',
    'author': 'Gabin Maxime Nguegnang',
    
    # Performance Metrics (key indicators only)
    'performance': {
        'r2_score': 0.988,  # R² Score (0-1, higher is better) - measures prediction accuracy
        'nse': 0.986  # Nash-Sutcliffe Efficiency (0-1, higher is better) - model quality indicator
    },
    
    # Model specifications
    'specifications': {
        'n_estimators': 890,  # Number of trees in forest
        'max_depth': 60,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 7
    },
    
    # Input requirements
    'input_features': MODEL_FEATURES,
    'required_columns': REQUIRED_FIELDS,
    
    # Model capabilities
    'capabilities': [
        'Fuel consumption prediction for power generation plants',
        'Site-level and cluster-level analysis',
        'Warning on possible anomaly using statistical thresholds (mean + 2σ)',
        'Time series forecasting support',
        'Batch predictions on uploaded data'
    ],
    
    # Known limitations
    'limitations': [
        'Optimized for diesel generators 6.5 KVA x 2',
        #'Accuracy varies based on training data range'#,
        #'NSE model quality assessment requires CONSUMPTION HIS column (optional feature)'
    ],
    
    # Version history
    'changelog': {
        '2.1.0': 'Added dynamic column detection for exports',
        '2.0.0': 'Implemented configuration management system'#,
        #'1.5.0': 'Enhanced warning on possible anomaly with configurable statistical thresholds',
        #'1.0.0': 'Initial production model release'
    }
}
