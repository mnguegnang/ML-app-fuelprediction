"""
Fuel Consumption Prediction Application

A Flask web application for predicting fuel consumption using machine learning.
The application provides:
- File upload and validation for Excel data
- ML-powered prediction generation
- Interactive visualizations (cluster, site, distribution, comparison, time-series)
- Multiple export formats (CSV, Excel, text summary, SVG, PNG)
- Performance monitoring and caching

Architecture:
- Main app: Flask setup, model loading, blueprint registration
- Routes: Organized into 3 blueprints (main, visualization, export)
- Utils: 7 utility modules (file, validation, data, model, metrics, chart, export)
- Config: Centralized configuration constants

Author: Gabin Maxime Nguegnang
Version: 2.0.0 (Refactored with blueprints)
"""

from flask import Flask
import os
import logging
from datetime import datetime

# Import configuration
import config

# Import utilities (caches, model loader)
from utils.data_utils import data_cache, prediction_cache
from utils.model_utils import load_model

# Import route blueprints
from routes.main_routes import main_bp, init_blueprint as init_main_bp
from routes.visualization_routes import viz_bp, init_blueprint as init_viz_bp
from routes.export_routes import export_bp, init_blueprint as init_export_bp

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
cur_dir = os.path.dirname(__file__)
logs_dir = os.path.join(cur_dir, 'logs')
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
    logger.info(f"Created logs directory: {logs_dir}")

# File handler for persistent logging
log_file = os.path.join(logs_dir, f'fuel_app_{datetime.now().strftime("%Y%m%d")}.log')
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

logger.info("="*80)
logger.info("FUEL CONSUMPTION PREDICTION APPLICATION STARTING")
logger.info(f"Version: 2.0.0 | Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("="*80)

# ============================================================================
# METRICS TRACKING
# ============================================================================

request_metrics = {
    'total_predictions': 0,
    'successful_predictions': 0,
    'failed_predictions': 0,
    'cache_hits': 0,
    'cache_misses': 0,
    'total_processing_time':0.0,
    'last_request_time': None,
    # Statistical metrics (from latest prediction batch)
    'mean_prediction': None,
    'median_prediction': None,
    'std_prediction': None,
    'min_prediction': None,
    'max_prediction': None,
    'unique_clusters': None,
    'unique_sites': None,
    'nse': None  # Nash-Sutcliffe Efficiency
}

# ============================================================================
# FLASK APP CONFIGURATION
# ============================================================================

app = Flask(__name__)
app.secret_key = 'your_secret_key_here_change_in_production'  # Required for sessions and flash messages

# File upload configuration
UPLOAD_FOLDER = os.path.join(cur_dir, 'uploads')
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'ods'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    logger.info(f"Created upload folder: {UPLOAD_FOLDER}")

# ============================================================================
# MODEL LOADING
# ============================================================================

logger.info(f"Loading ML model from: {config.MODEL_PATH}")
clf = load_model(config.MODEL_PATH)

if clf is not None:
    logger.info(f"✓ Model loaded successfully: {type(clf).__name__}")
    try:
        model_features = clf.n_features_in_
        logger.info(f"  - Model expects {model_features} features")
    except:
        pass
else:
    logger.error("✗ Model loading failed - application may not function correctly")

# ============================================================================
# BLUEPRINT REGISTRATION
# ============================================================================

logger.info("Registering route blueprints...")

# Initialize and register main routes blueprint
logger.info("  - Initializing main_routes blueprint...")
init_main_bp(
    model=clf,
    upload_folder=UPLOAD_FOLDER,
    flask_app_config=app.config,
    metrics=request_metrics,
    d_cache=data_cache,
    p_cache=prediction_cache,
    logfile=log_file
)
app.register_blueprint(main_bp)
logger.info("    ✓ main_routes registered: /, /upload, /clear_upload, /model-info, /cache_stats")

# Initialize and register visualization routes blueprint
logger.info("  - Initializing visualization_routes blueprint...")
from routes.main_routes import results  # Import results function for visualization routes
init_viz_bp(
    model=clf,
    upload_folder=UPLOAD_FOLDER,
    results_func=results
)
app.register_blueprint(viz_bp)
logger.info("    ✓ visualization_routes registered: /clustergraph, /sitegraph, /distributiongraph, /comparisonview, /timeseriesgraph")

# Initialize and register export routes blueprint
logger.info("  - Initializing export_routes blueprint...")
init_export_bp(
    upload_folder=UPLOAD_FOLDER,
    p_cache=prediction_cache
)
app.register_blueprint(export_bp)
logger.info("    ✓ export_routes registered: /export/predictions/csv, /export/predictions/excel, /export/summary, /export/chart/<type>, /export/chart/<type>/png")

logger.info("✓ All blueprints registered successfully")
logger.info(f"Total routes registered: {len(app.url_map._rules)}")

# ============================================================================
# BACKWARD COMPATIBILITY - URL ALIASES FOR TEMPLATES
# ============================================================================
# Templates reference routes without blueprint prefix (e.g., 'model_info' instead of 'main.model_info')
# Add endpoint aliases for backward compatibility

logger.info("Adding URL aliases for template backward compatibility...")

# Create endpoint mappings for templates
endpoint_aliases = {
    # Main routes
    'index': 'main.index',
    'upload_file': 'main.upload_file',
    'clear_upload': 'main.clear_upload',
    'model_info': 'main.model_info',
    'model_info_json': 'main.model_info_json',
    'cache_stats': 'main.cache_stats',
    # Visualization routes
    'clustergraph': 'visualization.clustergraph',
    'sitegraph': 'visualization.sitegraph',
    'distributiongraph': 'visualization.distributiongraph',
    'comparisonview': 'visualization.comparisonview',
    'timeseriesgraph': 'visualization.timeseriesgraph',
    # Export routes
    'export_predictions_csv': 'export.export_predictions_csv',
    'export_predictions_excel': 'export.export_predictions_excel',
    'export_summary': 'export.export_summary',
    'export_chart': 'export.export_chart',
    'export_chart_png_route': 'export.export_chart_png_route'
}

# Add URL rules with old endpoint names using the same view functions
for old_endpoint, blueprint_endpoint in endpoint_aliases.items():
    view_func = app.view_functions.get(blueprint_endpoint)
    if view_func:
        # Find the URL rule for the blueprint endpoint
        for rule in app.url_map.iter_rules(blueprint_endpoint):
            # Add the same view function with the old endpoint name
            # This creates an alias in the routing system
            app.add_url_rule(
                rule.rule,  # Same URL path
                endpoint=old_endpoint,  # Old endpoint name for url_for()
                view_func=view_func,  # Same view function
                methods=rule.methods - {'HEAD', 'OPTIONS'}  # Same methods (exclude auto-added)
            )
            logger.debug(f"  - Aliased '{old_endpoint}' -> '{rule.rule}'")
            break

logger.info(f"✓ Added {len(endpoint_aliases)} endpoint aliases for template compatibility")

# ============================================================================
# APPLICATION SUMMARY
# ============================================================================

logger.info("="*80)
logger.info("APPLICATION READY")
logger.info(f"Upload folder: {UPLOAD_FOLDER}")
logger.info(f"Max file size: {MAX_FILE_SIZE / (1024*1024):.0f} MB")
logger.info(f"Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}")
logger.info(f"Cache systems: data_cache (id={id(data_cache)}), prediction_cache (id={id(prediction_cache)})")
logger.info("="*80)

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    logger.info("Starting Flask development server on port 6003...")
    logger.info("Access the application at: http://localhost:6003")
    logger.info("Press CTRL+C to stop the server")
    logger.info("="*80)
    
    app.run(port=6003, debug=True)
