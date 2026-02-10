"""
Main Routes Blueprint

This module contains the core application routes:
- index: Main dashboard displaying uploaded data and available actions
- upload_file: File upload handling and validation
- clear_upload: Clear uploaded file from session and cache
- model_info: Display model metadata and information
- model_info_json: Return model metadata as JSON
- cache_stats: Display cache and request metrics for monitoring
- results: Core prediction logic (helper function called by visualization routes)

All routes use shared resources from the parent application including:
- clf: Machine learning model
- UPLOAD_FOLDER: File storage directory
- request_metrics: Performance tracking dictionary
- data_cache, prediction_cache: Data caching dictionaries
"""

from flask import Blueprint, render_template, request, flash, redirect, url_for, session
from werkzeug.utils import secure_filename
import os
import json
import logging
import time
from datetime import datetime
import pandas as pd
import numpy as np

# Import configuration
import config

# Import utilities
from utils.file_utils import (
    allowed_file,
    read_excel_robust,
    find_valid_sheet,
    validate_file_format
)
from utils.validation_utils import (
    validate_form_inputs,
    validate_columns_exist,
    validate_dataframe,
    validate_required_feature,
    resolve_column_name
)
from utils.data_utils import (
    clear_cache,
    get_cached_data,
    get_data_file,
    update_prediction_cache,
    get_prediction_cache
)
from utils.model_utils import (
    get_model_metadata,
    validate_model_loaded
)

logger = logging.getLogger(__name__)

# Create Blueprint
main_bp = Blueprint('main', __name__)

# These will be injected when blueprint is registered
clf = None
UPLOAD_FOLDER = None
app_config = None
request_metrics = None
data_cache = None
prediction_cache = None
log_file = None


def init_blueprint(model, upload_folder, flask_app_config, metrics, d_cache, p_cache, logfile):
    """Initialize blueprint with shared resources from main app"""
    global clf, UPLOAD_FOLDER, app_config, request_metrics, data_cache, prediction_cache, log_file
    clf = model
    UPLOAD_FOLDER = upload_folder
    app_config = flask_app_config
    request_metrics = metrics
    data_cache = d_cache
    prediction_cache = p_cache
    log_file = logfile


@main_bp.route('/')
def index():
    """Main dashboard page"""
    if clf is None:
        return render_template('error.html', error="Model not loaded. Please contact administrator."), 500
    
    # Check if there's uploaded data file available
    excel_file = get_data_file(session, UPLOAD_FOLDER)
    
    if excel_file is None:
        # No uploaded file, redirect to upload form
        logger.info("No uploaded file found, redirecting to upload form")
        flash('Please upload a data file to continue', 'info')
        return redirect(url_for('main.upload_file'))
    
    try:
        # Use cached data (much faster than reading file every time)
        Clusters, sheet_name, col = get_cached_data(excel_file, request_metrics)
        
        if Clusters is None or col is None:
            logger.error("Failed to load data from file")
            return render_template('error.html', 
                                 error="Failed to load data file. The file may be corrupted or in an unsupported format."), 400
        
        if Clusters.empty:
            logger.warning("Excel file is empty")
            return render_template('error.html', error="Data file is empty."), 400
        
        logger.info(f"Index loaded successfully with {len(col)} columns (from cache: {data_cache['filepath'] == excel_file})")
        
        # Pass info about uploaded file
        is_uploaded = True
        filename = session.get('uploaded_file', 'Unknown file')
        
        # Get model metadata
        model_info = get_model_metadata(config, clf)
        
        return render_template('index3.html', col=col, is_uploaded=is_uploaded, 
                             filename=filename, model_info=model_info)
        
    except Exception as e:
        logger.error(f"Error in index route: {str(e)}")
        return render_template('error.html', error=f"An error occurred: {str(e)}"), 500


@main_bp.route('/model-info')
def model_info():
    """Display detailed model information and metadata"""
    try:
        model_metadata = get_model_metadata(config, clf)
        return render_template('model_info.html', model=model_metadata)
    except Exception as e:
        logger.error(f"Error in model-info route: {str(e)}")
        return render_template('error.html', error=f"Could not load model information: {str(e)}"), 500


@main_bp.route('/model-info/json')
def model_info_json():
    """Return model metadata as JSON (useful for API integration)"""
    try:
        model_metadata = get_model_metadata(config, clf)
        return json.dumps(model_metadata, indent=2, default=str), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        logger.error(f"Error in model-info JSON route: {str(e)}")
        return json.dumps({'error': str(e)}), 500, {'Content-Type': 'application/json'}


@main_bp.route('/upload', methods=['GET', 'POST'])
def upload_file():
    """Handle file upload"""
    if request.method == 'POST':
        try:
            # Check if file is in request
            if 'file' not in request.files:
                flash('No file selected', 'error')
                return redirect(request.url)
            
            file = request.files['file']
            
            # Check if file is selected
            if file.filename == '':
                flash('No file selected', 'error')
                return redirect(request.url)
            
            # Validate file
            if not allowed_file(file.filename):
                flash('Invalid file type. Only Excel files (.xlsx, .xls, .ods) are allowed.', 'error')
                return redirect(request.url)
            
            # Secure filename and save
            filename = secure_filename(file.filename)
            filepath = os.path.join(app_config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.info(f"File uploaded successfully: {filename}")
            
            # Verify file integrity immediately after save
            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                flash('File upload failed or file is empty.', 'error')
                return redirect(request.url)
            
            file_size = os.path.getsize(filepath)
            print(f"DEBUG: File saved successfully: {filepath}, size: {file_size} bytes")
            
            # Validate file format
            is_valid, validation_error = validate_file_format(filepath)
            if not is_valid:
                os.remove(filepath)
                flash(validation_error, 'error')
                return redirect(request.url)
            
            # Validate the Excel file
            try:
                # Check if file or sheet name is valid
                sheet_name = find_valid_sheet(filepath)
                if sheet_name is None:
                    try:
                        # Try multiple engines to read available sheets
                        xl_file = None
                        for engine in ['openpyxl', 'odf', 'xlrd', None]:
                            try:
                                xl_file = pd.ExcelFile(filepath, engine=engine) if engine else pd.ExcelFile(filepath)
                                break
                            except:
                                continue
                        
                        if xl_file:
                            available_sheets = xl_file.sheet_names
                        else:
                            available_sheets = []
                        
                        filename_only = os.path.basename(filepath)
                        # Keep file for debugging
                        import shutil
                        debug_path = filepath + '.debug'
                        shutil.copy(filepath, debug_path)
                        print(f"DEBUG: File saved for inspection at: {debug_path}")
                        os.remove(filepath)
                        flash(f'Invalid file: Neither the file name ("{filename_only}") nor any sheet names ({available_sheets}) contain the required keywords ("generator only", "gen only", or "generator"). Please rename your file or at least one sheet.', 'error')
                    except:
                        os.remove(filepath)
                        flash('Invalid file: File name or sheet name must contain "generator only", "gen only", or "generator". Please rename your file or sheet.', 'error')
                    return redirect(request.url)
                
                df = read_excel_robust(filepath, sheet_name=sheet_name)
                if df.empty:
                    os.remove(filepath)
                    flash('Uploaded file is empty or has no data.', 'error')
                    return redirect(request.url)
                logger.info(f"File validated: {len(df)} rows, {len(df.columns)} columns, sheet: {sheet_name}")
            except Exception as e:
                os.remove(filepath)
                flash(f'Invalid Excel file format: {str(e)}', 'error')
                return redirect(request.url)
            
            # Clear cache before updating session with new file
            clear_cache()
            
            # Store filename in session
            session['uploaded_file'] = filename
            flash(f'File "{filename}" uploaded successfully!', 'success')
            
            return redirect(url_for('main.index'))
            
        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}")
            flash(f'Error uploading file: {str(e)}', 'error')
            return redirect(request.url)
    
    return render_template('upload.html')


@main_bp.route('/clear_upload')
def clear_upload():
    """Clear uploaded file from session and cache"""
    if 'uploaded_file' in session:
        filename = session.pop('uploaded_file')
        logger.info(f"Cleared uploaded file: {filename}")
        flash('File cleared. Please upload a new file to continue.', 'info')
    
    # Clear the data cache
    clear_cache()
    
    return redirect(url_for('main.upload_file'))


@main_bp.route('/cache_stats')
def cache_stats():
    """Display cache statistics (for debugging/monitoring)"""
    stats = {
        'cache_active': data_cache['filepath'] is not None,
        'cached_file': data_cache['filepath'],
        'sheet_name': data_cache['sheet_name'],
        'rows': len(data_cache['dataframe']) if data_cache['dataframe'] is not None else 0,
        'columns': len(data_cache['columns']) if data_cache['columns'] else 0,
        'file_size_mb': round(data_cache['file_size'] / (1024*1024), 2) if data_cache['file_size'] else 0,
        'cached_at': datetime.fromtimestamp(data_cache['timestamp']).strftime('%Y-%m-%d %H:%M:%S') if data_cache['timestamp'] else None,
        'age_seconds': round(time.time() - data_cache['timestamp'], 2) if data_cache['timestamp'] else None
    }
    
    # Format statistical metrics for display
    mean_val = f"{request_metrics['mean_prediction']:.2f}L" if request_metrics['mean_prediction'] is not None else 'N/A'
    median_val = f"{request_metrics['median_prediction']:.2f}L" if request_metrics['median_prediction'] is not None else 'N/A'
    std_val = f"{request_metrics['std_prediction']:.2f}L" if request_metrics['std_prediction'] is not None else 'N/A'
    min_val = f"{request_metrics['min_prediction']:.2f}L" if request_metrics['min_prediction'] is not None else 'N/A'
    max_val = f"{request_metrics['max_prediction']:.2f}L" if request_metrics['max_prediction'] is not None else 'N/A'
    range_val = f"{(request_metrics['max_prediction'] - request_metrics['min_prediction']):.2f}L" if (request_metrics['max_prediction'] is not None and request_metrics['min_prediction'] is not None) else 'N/A'
    clusters_val = request_metrics['unique_clusters'] if request_metrics['unique_clusters'] is not None else 'N/A'
    sites_val = request_metrics['unique_sites'] if request_metrics['unique_sites'] is not None else 'N/A'
    
    # Format NSE with quality interpretation
    nse_val = f"{request_metrics['nse']:.4f}" if request_metrics['nse'] is not None else 'N/A'
    if request_metrics['nse'] is not None:
        if request_metrics['nse'] >= 0.75:
            nse_quality = 'Very Good'
            nse_color = 'green'
            nse_bg = '#d4edda'
        elif request_metrics['nse'] >= 0.65:
            nse_quality = 'Good'
            nse_color = '#28a745'
            nse_bg = '#d4edda'
        elif request_metrics['nse'] >= 0.50:
            nse_quality = 'Satisfactory'
            nse_color = '#ffc107'
            nse_bg = '#fff3cd'
        elif request_metrics['nse'] >= 0:
            nse_quality = 'Unsatisfactory'
            nse_color = '#dc3545'
            nse_bg = '#f8d7da'
        else:
            nse_quality = 'Poor (worse than mean)'
            nse_color = '#dc3545'
            nse_bg = '#f8d7da'
    else:
        nse_quality = 'Not Available'
        nse_color = '#666'
        nse_bg = 'white'
    
    # Check for negative minimum (data quality issue)
    min_is_negative = request_metrics['min_prediction'] is not None and request_metrics['min_prediction'] < 0
    min_row_bg = '#fff3cd' if min_is_negative else 'white'
    min_color = 'red' if min_is_negative else 'inherit'
    min_text_color = 'red' if min_is_negative else '#666'
    min_interpretation = '⚠️ Negative value - possible data quality issue!' if min_is_negative else 'Lowest consumption site'
    
    return f"""
    <html>
    <head><title>System Statistics & Monitoring</title></head>
    <body style="font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5;">
        <h1>🔍 Fuel Consumption Prediction - System Monitoring</h1>
        
        <div style="background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h2>📊 Model Performance Metrics</h2>
            <table border="1" cellpadding="10" style="border-collapse: collapse; width: 100%;">
                <tr style="background: #667eea; color: white;"><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Predictions Generated</td><td><strong>{request_metrics['total_predictions']:,}</strong></td></tr>
                <tr><td>Successful Requests</td><td style="color: green;"><strong>{request_metrics['successful_predictions']}</strong></td></tr>
                <tr><td>Failed Requests</td><td style="color: red;"><strong>{request_metrics['failed_predictions']}</strong></td></tr>
                <tr><td>Success Rate</td><td><strong>{(request_metrics['successful_predictions'] / max(1, request_metrics['successful_predictions'] + request_metrics['failed_predictions']) * 100):.1f}%</strong></td></tr>
                <tr><td>Avg Processing Time</td><td><strong>{(request_metrics['total_processing_time'] / max(1, request_metrics['successful_predictions'])):.3f}s</strong></td></tr>
                <tr><td>Last Request</td><td>{request_metrics['last_request_time'] or 'N/A'}</td></tr>
            </table>
        </div>
    
    <div style="background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        <h2>📈 Quality Assurance - Prediction Statistics</h2>
        <table border="1" cellpadding="10" style="border-collapse: collapse; width: 100%;">
            <tr style="background: #764ba2; color: white;"><th>Statistical Metric</th><th>Value</th><th>Interpretation</th></tr>
            <tr>
                <td><strong>Mean Prediction</strong></td>
                <td><strong>{mean_val}</strong></td>
                <td style="font-size: 0.9em; color: #666;">Average fuel consumption across all sites</td>
            </tr>
            <tr>
                <td><strong>Median Prediction</strong></td>
                <td><strong>{median_val}</strong></td>
                <td style="font-size: 0.9em; color: #666;">Middle value - less affected by outliers</td>
            </tr>
            <tr>
                <td><strong>Standard Deviation</strong></td>
                <td><strong>{std_val}</strong></td>
                <td style="font-size: 0.9em; color: #666;">Prediction consistency - lower is more uniform</td>
            </tr>
            <tr style="background: {min_row_bg};">
                <td><strong>Minimum Prediction</strong></td>
                <td><strong style="color: {min_color};">{min_val}</strong></td>
                <td style="font-size: 0.9em; color: {min_text_color};">{min_interpretation}</td>
            </tr>
            <tr>
                <td><strong>Maximum Prediction</strong></td>
                <td><strong>{max_val}</strong></td>
                <td style="font-size: 0.9em; color: #666;">Highest consumption site - check for anomalies</td>
            </tr>
            <tr>
                <td><strong>Range</strong></td>
                <td><strong>{range_val}</strong></td>
                <td style="font-size: 0.9em; color: #666;">Spread of predictions - indicates data diversity</td>
            </tr>
            <tr style="background: #f8f9fa;">
                <td><strong>Unique Clusters Analyzed</strong></td>
                <td><strong>{clusters_val}</strong></td>
                <td style="font-size: 0.9em; color: #666;">Number of distinct clusters in latest prediction</td>
            </tr>
            <tr style="background: #f8f9fa;">
                <td><strong>Unique Sites Analyzed</strong></td>
                <td><strong>{sites_val}</strong></td>
                <td style="font-size: 0.9em; color: #666;">Number of distinct sites in latest prediction</td>
            </tr>
            <tr style="background: {nse_bg}; border-top: 3px solid #764ba2;">
                <td><strong>Nash-Sutcliffe Efficiency (NSE)</strong></td>
                <td><strong style="color: {nse_color}; font-size: 1.1em;">{nse_val}</strong></td>
                <td style="font-size: 0.9em; color: {nse_color};"><strong>{nse_quality}</strong> - Model accuracy vs observed data</td>
            </tr>
        </table>
        <div style="margin-top: 15px; padding: 10px; background: #e7f3ff; border-left: 4px solid #2196F3; border-radius: 4px;">
            <strong>💡 Tip:</strong> Monitor these statistics to validate prediction quality. Large standard deviation or negative values may indicate data quality issues that need investigation.
        </div>
        <div style="margin-top: 10px; padding: 10px; background: #f0e7ff; border-left: 4px solid #764ba2; border-radius: 4px;">
            <strong>📊 NSE Interpretation:</strong> 1.0 = Perfect | ≥0.75 = Very Good | ≥0.65 = Good | ≥0.50 = Satisfactory | &lt;0.50 = Unsatisfactory | &lt;0 = Worse than mean
        </div>
    </div>
    
    <div style="background: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        <h2>💾 Data Cache Information</h2>
        <table border="1" cellpadding="10" style="border-collapse: collapse; width: 100%;">
            <tr style="background: #43e97b; color: white;"><th>Property</th><th>Value</th></tr>
            <tr><td>Cache Active</td><td><strong>{'Yes' if stats['cache_active'] else 'No'}</strong></td></tr>
            <tr><td>Cached File</td><td>{stats['cached_file'] or 'None'}</td></tr>
            <tr><td>Sheet Name</td><td>{stats['sheet_name'] or 'N/A'}</td></tr>
            <tr><td>Cached Rows</td><td>{stats['rows']:,}</td></tr>
            <tr><td>Cached Columns</td><td>{stats['columns']}</td></tr>
            <tr><td>File Size</td><td>{stats['file_size_mb']} MB</td></tr>
            <tr><td>Cached At</td><td>{stats['cached_at'] or 'N/A'}</td></tr>
            <tr><td>Cache Age</td><td>{stats['age_seconds']} seconds</td></tr>
        </table>
    </div>
    
    <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
        <h2>📝 Recent Logs</h2>
        <p><em>Check the log file at: {log_file}</em></p>
        <p style="color: #666;">Logs include detailed information about prediction requests, model performance, data loading times, and errors.</p>
    </div>
    
    <br>
    <a href="/" style="padding: 10px 20px; background: #667eea; color: white; text-decoration: none; border-radius: 5px;">← Back to Home</a>
    <a href="/cache_stats" style="padding: 10px 20px; background: #764ba2; color: white; text-decoration: none; border-radius: 5px; margin-left: 10px;">↻ Refresh Stats</a>
</body>
</html>
"""


def results():
    """
    Core prediction logic helper function.
    
    This function performs the ML prediction workflow:
    1. Load and validate data from uploaded file
    2. Process form inputs and resolve column names
    3. Prepare features for model prediction
    4. Generate predictions using the ML model
    5. Calculate performance metrics (NSE)
    6. Cache results for visualization routes
    
    Returns:
        pd.DataFrame: Aggregated predictions by cluster
        
    Raises:
        ValueError: If validation fails or data is invalid
        FileNotFoundError: If no data file is uploaded
        KeyError: If required columns are missing
    """
    # Track request start time for performance monitoring
    request_start_time = time.time()
    session_id = session.get('_id', 'anonymous')
    
    # Log prediction request initiated
    logger.info(f"[PREDICTION REQUEST] Session: {session_id} | Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Validate model is loaded
        if clf is None:
            logger.error(f"[MODEL ERROR] Session: {session_id} | Model not loaded")
            request_metrics['failed_predictions'] += 1
            raise ValueError("Model not loaded")
        
        # Load data from uploaded file (using cache)
        excel_file = get_data_file(session, UPLOAD_FOLDER)
        if excel_file is None:
            logger.error(f"[DATA ERROR] Session: {session_id} | No data file available")
            request_metrics['failed_predictions'] += 1
            raise FileNotFoundError("No data file available. Please upload a file first.")
        
        # Use cached data for performance
        data_load_start = time.time()
        Clusters, sheet_name, columns = get_cached_data(excel_file, request_metrics)
        data_load_time = time.time() - data_load_start
        
        if Clusters is None:
            logger.error(f"[DATA ERROR] Session: {session_id} | Failed to load data from {excel_file}")
            request_metrics['failed_predictions'] += 1
            raise ValueError("Failed to load data. The file may be corrupted or in an unsupported format.")
        
        # Log data loading performance
        logger.info(f"[DATA LOAD] Session: {session_id} | Loaded {len(Clusters)} rows x {len(columns)} columns in {data_load_time:.3f}s")
        
        # Validate form inputs
        required_fields = ['Cluster', 'SITE Name', 'PREVIOUS FUEL QTE', 'QTE FUEL FOUND', 
                           'TYPE OF GENERATOR', 'RUNNING TIME', 'CONSUMPTION RATE', 
                           'NUMBER OF DAYS', 'FUEL ADDED']
        
        is_valid, missing_fields, error_msg = validate_form_inputs(request.form, required_fields)
        if not is_valid:
            logger.warning(f"[INPUT ERROR] Session: {session_id} | {error_msg}")
            request_metrics['failed_predictions'] += 1
            raise ValueError(error_msg)
        
        # Log input parameters
        input_params = {field: request.form[field] for field in required_fields}
        logger.info(f"[INPUT PARAMS] Session: {session_id} | Columns: {json.dumps(input_params)}")

        CLUSTER = resolve_column_name(str(request.form['Cluster']), Clusters)
        SITE_NAME = resolve_column_name(str(request.form['SITE Name']), Clusters)
        PRE_QTE_FUEL = resolve_column_name(str(request.form['PREVIOUS FUEL QTE']), Clusters)
        QTE_FUEL_FOUND = resolve_column_name(str(request.form['QTE FUEL FOUND']), Clusters)
        TYPE_OF_GENERATOR = resolve_column_name(str(request.form['TYPE OF GENERATOR']), Clusters)
        RUNNING_TIME = resolve_column_name(str(request.form['RUNNING TIME']), Clusters)
        CONSUMPTION_RATE = resolve_column_name(str(request.form['CONSUMPTION RATE']), Clusters)
        NUMBER_OF_DAYS = resolve_column_name(str(request.form['NUMBER OF DAYS']), Clusters)
        FUEL_ADDED = resolve_column_name(str(request.form['FUEL ADDED']), Clusters)
        
        # Check for missing columns
        missing_cols = []
        if not CLUSTER: missing_cols.append(request.form['Cluster'])
        if not SITE_NAME: missing_cols.append(request.form['SITE Name'])
        if not PRE_QTE_FUEL: missing_cols.append(request.form['PREVIOUS FUEL QTE'])
        if not QTE_FUEL_FOUND: missing_cols.append(request.form['QTE FUEL FOUND'])
        if not TYPE_OF_GENERATOR: missing_cols.append(request.form['TYPE OF GENERATOR'])
        if not RUNNING_TIME: missing_cols.append(request.form['RUNNING TIME'])
        if not CONSUMPTION_RATE: missing_cols.append(request.form['CONSUMPTION RATE'])
        if not NUMBER_OF_DAYS: missing_cols.append(request.form['NUMBER OF DAYS'])
        if not FUEL_ADDED: missing_cols.append(request.form['FUEL ADDED'])
        
        if missing_cols:
            logger.error(f"[COLUMN ERROR] Session: {session_id} | Missing columns: {missing_cols}")
            request_metrics['failed_predictions'] += 1
            raise ValueError(f"Columns not found in data: {', '.join(missing_cols)}. Available columns: {list(Clusters.columns)}")
        
        Clusters['Fuel_per_period'] = Clusters[PRE_QTE_FUEL] - Clusters[QTE_FUEL_FOUND]
        
        # Validate required column for model
        if 'GENERATOR 1 CAPACITY (KVA)' not in Clusters.columns:
            raise ValueError("Required column 'GENERATOR 1 CAPACITY (KVA)' not found in data")
                    
        # Select the categorical columns
        categorical_subset = Clusters[[CLUSTER, TYPE_OF_GENERATOR, 'GENERATOR 1 CAPACITY (KVA)']]
        
        # One hot encode
        categorical_subset = pd.get_dummies(categorical_subset)
        
        num = Clusters.drop([CLUSTER, TYPE_OF_GENERATOR, 'GENERATOR 1 CAPACITY (KVA)'], axis=1)
        
        # Join the two dataframes using concat
        features = pd.concat([num, categorical_subset], axis=1)
        
        # Check for required feature column
        required_feature = 'GENERATOR 1 CAPACITY (KVA)_6,5 x 2'
        if required_feature not in features.columns:
            logger.warning(f"Required feature '{required_feature}' not found after one-hot encoding. Available: {list(features.columns)}")
            raise ValueError(f"Model requires feature '{required_feature}' which is not present in the data. This may indicate a data format mismatch.")
        
        Data = features[['Fuel_per_period', RUNNING_TIME, CONSUMPTION_RATE, NUMBER_OF_DAYS, 
                         FUEL_ADDED, required_feature]].dropna()
        
        if Data.empty:
            logger.error(f"[DATA ERROR] Session: {session_id} | No valid data after filtering")
            request_metrics['failed_predictions'] += 1
            raise ValueError("No valid data available for prediction after filtering. Please check your data quality.")
        
        # Log pre-prediction data summary
        logger.info(f"[PRE-PREDICTION] Session: {session_id} | Data shape: {Data.shape} | Features: {list(Data.columns)}")
        
        # Perform prediction with timing
        prediction_start = time.time()
        resfinal = clf.predict(Data)
        prediction_time = time.time() - prediction_start
        l = Data.index.values
        clus = Clusters[CLUSTER][l].values
        site = Clusters[SITE_NAME][l].values
        pred = list(resfinal)
        
        RF_Predict_Data = pd.DataFrame({
            'Clusters': clus,
            'Sites': site,
            'Predictions': pred
        })
        
        result = RF_Predict_Data.groupby(['Clusters']).sum()
        
        # Calculate Nash-Sutcliffe Efficiency (NSE) if observed data is available
        nse_value = None
        try:
            # Try to get observed consumption values
            if 'CONSUMPTION HIS' in Clusters.columns:
                observed = Clusters['CONSUMPTION HIS'][l].values
                # Remove NaN values from both observed and predicted
                valid_mask = ~np.isnan(observed)
                if valid_mask.sum() > 0:  # If we have at least some valid observations
                    observed_clean = observed[valid_mask]
                    predicted_clean = np.array(pred)[valid_mask]
                    
                    # Calculate NSE: 1 - (sum of squared errors / sum of squared deviations from mean)
                    numerator = np.sum((observed_clean - predicted_clean) ** 2)
                    denominator = np.sum((observed_clean - np.mean(observed_clean)) ** 2)
                    
                    if denominator != 0:
                        nse_value = float(1 - (numerator / denominator))
                        logger.info(f"[NSE CALCULATED] Session: {session_id} | NSE: {nse_value:.4f} | Valid observations: {len(observed_clean)}")
                    else:
                        logger.warning(f"[NSE WARNING] Session: {session_id} | Cannot calculate NSE - zero variance in observed data")
                else:
                    logger.warning(f"[NSE WARNING] Session: {session_id} | No valid observations found (all NaN)")
            else:
                logger.warning(f"[NSE WARNING] Session: {session_id} | Column 'CONSUMPTION HIS' not found in data")
        except Exception as e:
            logger.error(f"[NSE ERROR] Session: {session_id} | Error calculating NSE: {str(e)}")
        
        # Calculate model performance metrics
        total_request_time = time.time() - request_start_time
        prediction_stats = {
            'total_predictions': len(pred),
            'unique_clusters': len(result),
            'unique_sites': len(set(site)),
            'mean_prediction': float(np.mean(pred)),
            'median_prediction': float(np.median(pred)),
            'std_prediction': float(np.std(pred)),
            'min_prediction': float(np.min(pred)),
            'max_prediction': float(np.max(pred)),
            'nse': nse_value
        }
        
        # Store predictions in server-side cache instead of session (avoids 4KB cookie limit)
        update_prediction_cache(clus, site, pred, result)
        # Store NSE value separately in cache
        prediction_cache['nse_value'] = nse_value
        
        # Update global metrics
        request_metrics['total_predictions'] += len(pred)
        request_metrics['successful_predictions'] += 1
        request_metrics['total_processing_time'] += total_request_time
        request_metrics['last_request_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Update statistical metrics (from latest batch)
        request_metrics['mean_prediction'] = prediction_stats['mean_prediction']
        request_metrics['median_prediction'] = prediction_stats['median_prediction']
        request_metrics['std_prediction'] = prediction_stats['std_prediction']
        request_metrics['min_prediction'] = prediction_stats['min_prediction']
        request_metrics['max_prediction'] = prediction_stats['max_prediction']
        request_metrics['unique_clusters'] = prediction_stats['unique_clusters']
        request_metrics['unique_sites'] = prediction_stats['unique_sites']
        request_metrics['nse'] = prediction_stats['nse']
        
        # Log comprehensive prediction summary
        logger.info(f"[PREDICTION SUCCESS] Session: {session_id} | "
                   f"Predictions: {len(pred)} | Clusters: {len(result)} | Sites: {len(set(site))} | "
                   f"Data Load: {data_load_time:.3f}s | Prediction: {prediction_time:.3f}s | Total: {total_request_time:.3f}s")
        logger.info(f"[MODEL METRICS] Session: {session_id} | "
                   f"Mean: {prediction_stats['mean_prediction']:.2f}L | "
                   f"Median: {prediction_stats['median_prediction']:.2f}L | "
                   f"Range: [{prediction_stats['min_prediction']:.2f}L - {prediction_stats['max_prediction']:.2f}L] | "
                   f"Std: {prediction_stats['std_prediction']:.2f}L")
        
        return result
        
    except KeyError as e:
        request_metrics['failed_predictions'] += 1
        total_request_time = time.time() - request_start_time
        logger.error(f"[COLUMN ERROR] Session: {session_id} | Error: {str(e)} | Time: {total_request_time:.3f}s")
        raise ValueError(f"Column error: {str(e)}")
    except Exception as e:
        request_metrics['failed_predictions'] += 1
        total_request_time = time.time() - request_start_time
        logger.error(f"[PREDICTION FAILED] Session: {session_id} | Error: {str(e)} | Time: {total_request_time:.3f}s", exc_info=True)
        raise
