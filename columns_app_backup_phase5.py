




from flask import Flask, render_template, request, flash, redirect, url_for, session, make_response
from wtforms import Form, TextAreaField, validators
from werkzeug.utils import secure_filename

import pickle
import pandas as pd
import os
import numpy as np
import logging
import re
import time
from io import BytesIO, StringIO

import pygal
from datetime import datetime
import json

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
    data_cache,
    prediction_cache,
    clear_cache,
    get_cached_data,
    get_data_file,
    update_prediction_cache,
    get_prediction_cache
)
from utils.model_utils import (
    load_model,
    get_model_metadata,
    validate_model_loaded
)

# Try to import cairosvg for PNG export
try:
	import cairosvg
	HAS_CAIROSVG = True
except ImportError:
	HAS_CAIROSVG = False
	logger = logging.getLogger(__name__)
	logger.warning("cairosvg not installed - PNG export will not be available")

# Configure logging with both file and console handlers
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
cur_dir_for_logs = os.path.dirname(__file__)
logs_dir = os.path.join(cur_dir_for_logs, 'logs')
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
    print(f"Created logs directory: {logs_dir}")

# File handler for persistent logging
log_file = os.path.join(logs_dir, f'fuel_app_{datetime.now().strftime("%Y%m%d")}.log')
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Metrics tracking
request_metrics = {
    'total_predictions': 0,
    'successful_predictions': 0,
    'failed_predictions': 0,
    'cache_hits': 0,
    'cache_misses': 0,
    'total_processing_time': 0.0,
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

# Preparing the Prediction
cur_dir = os.path.dirname(__file__)
app = Flask(__name__)
app.secret_key = 'your_secret_key_here_change_in_production'  # Required for flash messages

# File upload configuration
UPLOAD_FOLDER = os.path.join(cur_dir, 'uploads')
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'ods'}  # Added .ods for LibreOffice files
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    logger.info(f"Created upload folder: {UPLOAD_FOLDER}")

# Load model with error handling
clf = load_model(config.MODEL_PATH)

# Cache dictionaries are now imported from utils.data_utils
# - data_cache: stores loaded Excel files
# - prediction_cache: stores prediction results

# Cache management functions moved to utils/data_utils.py
# - clear_cache(): Clear all caches
# - get_cached_data(): Get cached DataFrame or load from file


# File utility functions moved to utils/file_utils.py
# Validation functions moved to utils/validation_utils.py
# Data/cache functions moved to utils/data_utils.py
# Model functions moved to utils/model_utils.py
@app.route('/')
def index():
	if clf is None:
		return render_template('error.html', error="Model not loaded. Please contact administrator."), 500
	
	# Check if there's uploaded data file available
	excel_file = get_data_file(session, UPLOAD_FOLDER)
	
	if excel_file is None:
		# No uploaded file, redirect to upload form
		logger.info("No uploaded file found, redirecting to upload form")
		flash('Please upload a data file to continue', 'info')
		return redirect(url_for('upload_file'))
	
	try:
		# Use cached data (much faster than reading file every time)
		Clusters, sheet_name, col = get_cached_data(excel_file, request_metrics)
		
		if Clusters is None or col is None:
			logger.error("Failed to load data from file")
			return render_template('error.html', error="Failed to load data file. The file may be corrupted or in an unsupported format."), 400
		
		if Clusters.empty:
			logger.warning("Excel file is empty")
			return render_template('error.html', error="Data file is empty."), 400
		
		logger.info(f"Index loaded successfully with {len(col)} columns (from cache: {data_cache['filepath'] == excel_file})")
		
		# Pass info about uploaded file
		is_uploaded = True
		filename = session.get('uploaded_file', 'Unknown file')
		
		# Get model metadata
		model_info = get_model_metadata(config, clf)
		
		return render_template('index3.html', col=col, is_uploaded=is_uploaded, filename=filename, model_info=model_info)
		
	except Exception as e:
		logger.error(f"Error in index route: {str(e)}")
		return render_template('error.html', error=f"An error occurred: {str(e)}"), 500


@app.route('/model-info')
def model_info():
	"""Display detailed model information and metadata"""
	try:
		model_metadata = get_model_metadata(config, clf)
		return render_template('model_info.html', model=model_metadata)
	except Exception as e:
		logger.error(f"Error in model-info route: {str(e)}")
		return render_template('error.html', error=f"Could not load model information: {str(e)}"), 500


@app.route('/model-info/json')
def model_info_json():
	"""Return model metadata as JSON (useful for API integration)"""
	try:
		model_metadata = get_model_metadata(config, clf)
		return json.dumps(model_metadata, indent=2, default=str), 200, {'Content-Type': 'application/json'}
	except Exception as e:
		logger.error(f"Error in model-info JSON route: {str(e)}")
		return json.dumps({'error': str(e)}), 500, {'Content-Type': 'application/json'}


@app.route('/upload', methods=['GET', 'POST'])
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
			filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
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
			
			return redirect(url_for('index'))
			
		except Exception as e:
			logger.error(f"Error uploading file: {str(e)}")
			flash(f'Error uploading file: {str(e)}', 'error')
			return redirect(request.url)
	
	return render_template('upload.html')


@app.route('/clear_upload')
def clear_upload():
	"""Clear uploaded file from session and cache"""
	if 'uploaded_file' in session:
		filename = session.pop('uploaded_file')
		logger.info(f"Cleared uploaded file: {filename}")
		flash('File cleared. Please upload a new file to continue.', 'info')
	
	# Clear the data cache
	clear_cache()
	
	return redirect(url_for('upload_file'))

#@app.route('/' , methods=['POST'])
#def run_action():
#	return str(request.form['SITE Name'])
def results():
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




@app.route('/clustergraph', methods=['POST', 'GET'])
def clustergraph():
	try:
		# Check if we need to generate predictions (POST with form data) or use cached (GET)
		if request.method == 'POST' and request.form:
			result = results()
		else:
			# Use cached result
			result = prediction_cache.get('result_df')
			if result is None:
				logger.warning("No cached predictions available")
				return render_template('error.html', error="No predictions available. Please submit the form first."), 400
		
		if result.empty:
			logger.warning("No predictions to display")
			return render_template('error.html', error="No predictions available. Please check your data."), 400
		
		# Get prediction data from server-side cache
		pred_data = get_prediction_cache() if get_prediction_cache().get('predictions') else None
		
		# Create cluster-level bar chart
		graph = pygal.Bar()
		graph.title = 'Prediction of Fuel Consumption Per Cluster'
		graph.x_title = 'Clusters'
		graph.y_title = 'Predicted Fuel Consumption (L)'
		
		for k in result.index:
			graph.add(k, result['Predictions'][k])
		
		graph_data = graph.render_data_uri()
		
		# Calculate statistics for anomaly detection
		predictions = result['Predictions'].values
		mean_pred = float(predictions.mean())
		std_pred = float(predictions.std())
		threshold = mean_pred + 2 * std_pred  # 2 standard deviations
		
		# Identify anomalies (clusters with unusually high consumption)
		anomalies = []
		for k in result.index:
			if result['Predictions'][k] > threshold:
				anomalies.append({
					'cluster': k,
					'value': float(result['Predictions'][k]),
					'excess': float(result['Predictions'][k] - mean_pred)
				})
		
		stats = {
			'total_predictions': len(predictions),
			'mean': round(mean_pred, 2),
			'std': round(std_pred, 2),
			'min': round(float(predictions.min()), 2),
			'max': round(float(predictions.max()), 2),
			'threshold': round(threshold, 2),
			'anomaly_count': len(anomalies)
		}
		
		logger.info(f"Graph generated successfully with {len(anomalies)} anomalies detected")
		return render_template("graphdisplay.html", 
		                      graph_data=graph_data, 
		                      stats=stats, 
		                      anomalies=anomalies,
		                      has_site_data=pred_data is not None,
		                      chart_type='cluster')
		
	except ValueError as e:
		logger.error(f"Validation error in clustergraph: {str(e)}")
		return render_template('error.html', error=str(e)), 400
	except Exception as e:
		logger.error(f"Error generating graph: {str(e)}")
		return render_template('error.html', error=f"Failed to generate graph: {str(e)}"), 500


@app.route('/sitegraph')
def sitegraph():
	"""Generate site-level predictions chart"""
	try:
		# Get prediction data from server-side cache
		cache = get_prediction_cache()
		if not cache.get('predictions'):
			return render_template('error.html', error="No prediction data available. Please run predictions first."), 400
		
		# Create DataFrame from cache data (exclude result_df and timestamp)
		df = pd.DataFrame({
			'clusters': cache['clusters'],
			'sites': cache['sites'],
			'predictions': cache['predictions']
		})
		
		# Group by site
		site_result = df.groupby('sites')['predictions'].sum().sort_values(ascending=False)
		
		# Limit to top 20 sites for readability
		top_sites = site_result.head(20)
		
		# Create horizontal bar chart for better label visibility
		graph = pygal.HorizontalBar()
		graph.title = 'Top 20 Sites by Predicted Fuel Consumption'
		graph.x_title = 'Predicted Fuel Consumption (L)'
		
		for site, value in top_sites.items():
			graph.add(str(site), float(value))
		
		graph_data = graph.render_data_uri()
		
		# Calculate site-level statistics
		all_sites_count = len(site_result)
		mean_site = float(site_result.mean())
		threshold_site = mean_site + 2 * float(site_result.std())
		
		# Identify high-consumption sites
		high_consumption_sites = []
		for site, value in site_result.items():
			if value > threshold_site:
				high_consumption_sites.append({
					'site': str(site),
					'value': float(value),
					'excess': float(value - mean_site)
				})
		
		stats = {
			'total_sites': all_sites_count,
			'showing_top': min(20, all_sites_count),
			'mean': round(mean_site, 2),
			'threshold': round(threshold_site, 2),
			'high_consumption_count': len(high_consumption_sites)
		}
		
		logger.info(f"Site-level graph generated: {all_sites_count} sites, {len(high_consumption_sites)} high-consumption sites")
		return render_template("graphdisplay.html", 
		                      graph_data=graph_data,
		                      stats=stats,
		                      anomalies=high_consumption_sites[:10],  # Top 10 high-consumption sites
		                      chart_type='site')
		
	except Exception as e:
		logger.error(f"Error generating site graph: {str(e)}")
		return render_template('error.html', error=f"Failed to generate site graph: {str(e)}"), 500


@app.route('/distributiongraph')
def distributiongraph():
	"""Generate distribution and anomaly visualization"""
	try:
		# Get prediction data from server-side cache
		cache = get_prediction_cache()
		pred_data = cache if cache.get('predictions') else None
		
		if pred_data is None:
			return render_template('error.html', error="No prediction data available. Please run predictions first."), 400
		
		predictions = pred_data['predictions']
		
		# Create histogram/distribution chart
		import numpy as np
		hist, bin_edges = np.histogram(predictions, bins=20)
		
		graph = pygal.Bar()
		graph.title = 'Distribution of Fuel Consumption Predictions'
		graph.x_title = 'Fuel Consumption Range (L)'
		graph.y_title = 'Number of Sites'
		graph.x_label_rotation = 60  # Rotate labels for better visibility
		
		# Create labels for bins with proper formatting for negative values
		labels = []
		for i in range(len(bin_edges)-1):
			labels.append(f'[{int(bin_edges[i])}, {int(bin_edges[i+1])}]')
		
		graph.x_labels = labels
		graph.add('Frequency', list(hist))
		
		graph_data = graph.render_data_uri()
		
		# Calculate statistics
		mean_pred = np.mean(predictions)
		std_pred = np.std(predictions)
		median_pred = np.median(predictions)
		
		# Identify anomalies (beyond 2 std deviations)
		threshold_high = mean_pred + 2 * std_pred
		threshold_low = mean_pred - 2 * std_pred
		
		anomaly_count = sum(1 for p in predictions if p > threshold_high or p < threshold_low)
		
		# Find the most common consumption range (tallest bar)
		max_freq_idx = np.argmax(hist)
		peak_range = labels[max_freq_idx]
		peak_count = int(hist[max_freq_idx])
		peak_percentage = round((peak_count / len(predictions)) * 100, 1)
		
		stats = {
			'mean': round(mean_pred, 2),
			'median': round(median_pred, 2),
			'std': round(std_pred, 2),
			'min': round(min(predictions), 2),
			'max': round(max(predictions), 2),
			'threshold_high': round(threshold_high, 2),
			'threshold_low': round(threshold_low, 2),
			'anomaly_count': anomaly_count,
			'total_predictions': len(predictions),
			'peak_range': peak_range,
			'peak_count': peak_count,
			'peak_percentage': peak_percentage
		}
		
		logger.info(f"Distribution graph generated: {anomaly_count} anomalies detected")
		return render_template("graphdisplay.html", 
		                      graph_data=graph_data,
		                      stats=stats,
		                      chart_type='distribution')
		
	except Exception as e:
		logger.error(f"Error generating distribution graph: {str(e)}")
		return render_template('error.html', error=f"Failed to generate distribution graph: {str(e)}"), 500


@app.route('/comparisonview')
def comparisonview():
	"""Show comparison of cluster vs site-level predictions"""
	try:
		# Get prediction data from server-side cache
		cache = get_prediction_cache()
		if not cache.get('predictions'):
			return render_template('error.html', error="No prediction data available. Please run predictions first."), 400
		
		# Create DataFrame (exclude result_df and timestamp)
		df = pd.DataFrame({
			'clusters': cache['clusters'],
			'sites': cache['sites'],
			'predictions': cache['predictions']
		})
		
		# Group by cluster and site
		cluster_totals = df.groupby('clusters')['predictions'].sum().sort_values(ascending=False)
		
		# Define consistent colors for sites
		site_colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', 
		               '#00f2fe', '#43e97b', '#38f9d7', '#fa709a', '#fee140']
		
		# Create custom style with defined colors
		from pygal.style import Style
		custom_style = Style(
			colors=site_colors + ['#cccccc'],  # Add gray for "Others"
			value_font_size=10,
			value_label_font_size=10,
		)
		
		# Create stacked bar chart to show top site contribution within each cluster
		graph = pygal.StackedBar(style=custom_style)
		graph.title = 'Fuel Consumption: Cluster Breakdown by Top Sites'
		graph.x_title = 'Clusters'
		graph.y_title = 'Predicted Fuel Consumption (L)'
		graph.x_label_rotation = 60  # Rotate cluster names for readability
		graph.show_legend = False  # Hide default legend (we'll show custom on left)
		graph.print_values = False  # Don't print values on bars
		graph.print_labels = False  # Don't print labels on bars
		
		# Collect top sites and their data for each cluster
		cluster_top_sites = {}  # {cluster: (site_name, value, pct)}
		all_top_sites = {}  # {site_name: total_consumption} for ranking
		
		for cluster in cluster_totals.index:
			cluster_sites = df[df['clusters'] == cluster].groupby('sites')['predictions'].sum()
			if not cluster_sites.empty:
				top_site = cluster_sites.idxmax()
				top_value = float(cluster_sites.max())
				cluster_total = float(cluster_totals[cluster])
				contribution_pct = round((top_value / cluster_total) * 100, 1)
				
				cluster_top_sites[cluster] = (top_site, top_value, contribution_pct)
				
				# Track total consumption per site
				if top_site in all_top_sites:
					all_top_sites[top_site] += top_value
				else:
					all_top_sites[top_site] = top_value
		
		# Sort sites by total consumption for legend
		ranked_sites = sorted(all_top_sites.items(), key=lambda x: x[1], reverse=True)
		
		# Create series for each unique top site (for different colors)
		site_series = {site: [] for site, _ in ranked_sites}
		other_sites_series = []
		
		for cluster in cluster_totals.index:
			if cluster in cluster_top_sites:
				top_site, top_value, contribution_pct = cluster_top_sites[cluster]
				cluster_total = float(cluster_totals[cluster])
				other_value = cluster_total - top_value
				
				# Add value for the top site in this cluster
				for site in site_series:
					if site == top_site:
						# Minimal tooltip: Only label and percentage (no cluster/site names)
						site_series[site].append({
							'value': top_value,
							'label': f'{contribution_pct}%'
						})
					else:
						site_series[site].append(0)
				
				# Add "other sites" value
				other_pct = round((other_value / cluster_total) * 100, 1)
				other_sites_series.append({
					'value': other_value,
					'label': f'{other_pct}%'
				})
			else:
				# No data for this cluster
				for site in site_series:
					site_series[site].append(0)
				other_sites_series.append(float(cluster_totals[cluster]))
		
		# Set cluster names as x-labels
		graph.x_labels = list(cluster_totals.index)
		
		# Add series for each top site (creates different colors - show ALL sites)
		for site, _ in ranked_sites:
			graph.add(site, site_series[site])
		
		# Add "Other Sites" series
		graph.add('Other Sites', other_sites_series)
		
		graph_data = graph.render_data_uri()
		
		# Prepare legend info with rankings, values and percentages
		total_all_consumption = sum(val for _, val in ranked_sites)
		site_legend = []
		site_legend_detailed = []
		for idx, (site, total_value) in enumerate(ranked_sites, 1):
			# Calculate percentage of total consumption across all sites
			site_pct = round((total_value / total_all_consumption) * 100, 1)
			# Simple string for backward compatibility
			site_legend.append(f"{idx}. {site}: {total_value:.2f}L ({site_pct}%)")
			# Detailed dict for template with value and percentage
			site_legend_detailed.append({
				'rank': idx,
				'site': site,
				'site_short': site[:20] if len(site) > 20 else site,
				'value': f"{total_value:.2f}",
				'percentage': f"{site_pct}%",
				'color': site_colors[(idx-1) % len(site_colors)]
			})
		
		stats = {
			'total_clusters': len(cluster_totals),
			'total_consumption': round(float(cluster_totals.sum()), 2),
			'site_legend': site_legend,
			'site_legend_detailed': site_legend_detailed
		}
		
		logger.info(f"Comparison view generated for {len(cluster_totals)} clusters with {len(ranked_sites)} top sites")
		return render_template("graphdisplay.html", 
		                      graph_data=graph_data,
		                      stats=stats,
		                      chart_type='comparison')
		
	except Exception as e:
		logger.error(f"Error generating comparison view: {str(e)}")
		return render_template('error.html', error=f"Failed to generate comparison view: {str(e)}"), 500


@app.route('/timeseriesgraph')
def timeseriesgraph():
	"""Generate time-series trend chart if date columns are available"""
	try:
		# Get the data file
		excel_file = get_data_file(session, UPLOAD_FOLDER)
		if excel_file is None:
			return render_template('error.html', error="No data file available."), 400
		
		# Get cached data
		Clusters, sheet_name, columns = get_cached_data(excel_file)
		
		if Clusters is None:
			return render_template('error.html', error="Failed to load data."), 400
		
		# Look for date columns (common patterns)
		date_columns = []
		for col in Clusters.columns:
			col_lower = str(col).lower()
			if any(keyword in col_lower for keyword in ['date', 'time', 'month', 'year', 'period', 'day']):
				date_columns.append(col)
		
		if not date_columns:
			return render_template('error.html', 
			                      error="No date/time columns found in the data. Time-series analysis requires date information. Available columns: " + ", ".join(columns[:10]) + "..."), 400
		
		# Use the first date column found
		date_col = date_columns[0]
		
		# Try to parse dates
		try:
			Clusters[date_col] = pd.to_datetime(Clusters[date_col], errors='coerce')
			Clusters = Clusters.dropna(subset=[date_col])
			
			if Clusters.empty:
				return render_template('error.html', error=f"Could not parse dates from column '{date_col}'."), 400
		except Exception as e:
			return render_template('error.html', error=f"Error parsing dates from '{date_col}': {str(e)}"), 400
		
		# Get prediction data from server-side cache (if available)
		cache = get_prediction_cache()
		pred_data = cache if cache.get('predictions') else None
		
		# Identify numeric columns for fuel analysis
		numeric_cols = Clusters.select_dtypes(include=['float64', 'int64']).columns
		
		fuel_col = None
		for col in numeric_cols:
			col_lower = str(col).lower()
			if any(keyword in col_lower for keyword in ['fuel', 'consumption', 'qty', 'qte', 'litre', 'liter']):
				fuel_col = col
				break
		
		if fuel_col is None and len(numeric_cols) > 0:
			fuel_col = numeric_cols[0]  # Use first numeric column as fallback
		
		if fuel_col is None:
			return render_template('error.html', error="No numeric columns found for time-series analysis."), 400
		
		# Group by date and aggregate
		Clusters = Clusters.sort_values(date_col)
		
		# Determine grouping frequency based on data span
		date_range = (Clusters[date_col].max() - Clusters[date_col].min()).days
		if date_range > 365:
			freq = 'M'  # Monthly
			title_suffix = '(Monthly)'
		elif date_range > 30:
			freq = 'W'  # Weekly
			title_suffix = '(Weekly)'
		else:
			freq = 'D'  # Daily
			title_suffix = '(Daily)'
		
		# Aggregate data
		Clusters['period'] = Clusters[date_col].dt.to_period(freq)
		time_series = Clusters.groupby('period')[fuel_col].sum()
		
		# Create line chart
		graph = pygal.Line()
		graph.title = f'Fuel Consumption Trend Over Time {title_suffix}'
		graph.x_title = 'Time Period'
		graph.y_title = 'Fuel Consumption (L)'
		graph.show_legend = False  # Hide default legend (we'll show custom on left)
		
		# Format x-labels with readable date formats
		x_labels = []
		for period in time_series.index:
			period_dt = period.to_timestamp()
			if freq == 'M':  # Monthly
				x_labels.append(period_dt.strftime('%b %Y'))  # e.g., "Jan 2024"
			elif freq == 'W':  # Weekly
				week_num = period_dt.isocalendar()[1]
				x_labels.append(f'W{week_num} {period_dt.year}')  # e.g., "W1 2024"
			else:  # Daily
				x_labels.append(period_dt.strftime('%d %b %Y'))  # e.g., "15 Jan 2024"
		
		graph.x_labels = x_labels
		graph.x_labels_major_every = max(1, len(x_labels) // 10)  # Show ~10 labels
		graph.show_minor_x_labels = False
		graph.x_label_rotation = 45  # Rotate labels for better readability
		
		graph.add('Actual Consumption', [float(v) for v in time_series.values])
		
		# Add trend line (moving average)
		if len(time_series) >= 3:
			window = min(3, len(time_series) // 3)
			moving_avg = time_series.rolling(window=window, center=True).mean()
			graph.add('Trend (Moving Avg)', [float(v) if pd.notna(v) else None for v in moving_avg.values])
		
		graph_data = graph.render_data_uri()
		
		# Prepare custom legend for left-side display
		series_legend = [
			{'name': 'Actual Consumption', 'color': '#F44336'},  # Red line
			{'name': 'Trend (Moving Avg)', 'color': '#2196F3'}   # Blue line
		]
		
		# Calculate trend statistics
		values = time_series.values
		trend_direction = "📈 Increasing" if values[-1] > values[0] else "📉 Decreasing"
		change_pct = ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
		
		stats = {
			'date_column': date_col,
			'fuel_column': fuel_col,
			'periods': len(time_series),
			'mean': round(float(time_series.mean()), 2),
			'trend': trend_direction,
			'change_percent': round(change_pct, 1),
			'first_value': round(float(values[0]), 2),
			'last_value': round(float(values[-1]), 2)
		}
		
		logger.info(f"Time-series graph generated: {len(time_series)} periods")
		return render_template("graphdisplay.html", 
		                      graph_data=graph_data,
		                      stats=stats,
		                      chart_type='timeseries',
		                      series_legend=series_legend)
		
	except Exception as e:
		logger.error(f"Error generating time-series graph: {str(e)}")
		import traceback
		traceback.print_exc()
		return render_template('error.html', error=f"Failed to generate time-series graph: {str(e)}"), 500



@app.route('/cache_stats')
def cache_stats():
	"""Display cache statistics (for debugging/monitoring)"""
	import time
	from datetime import datetime
	
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

	




# ============================================================================
# EXPORT AND DOWNLOAD ROUTES
# ============================================================================

@app.route('/export/predictions/csv')
def export_predictions_csv():
	"""Export predictions to CSV file"""
	try:
		cache = get_prediction_cache()
		if cache.get('result_df') is None:
			flash(config.ERROR_MESSAGES['prediction_cache_empty'], 'warning')
			return redirect(url_for('index'))
		
		# Create CSV in memory
		output = StringIO()
		result_df = cache['result_df']
		result_df.to_csv(output)
		output.seek(0)
		
		# Create response
		response = make_response(output.getvalue())
		response.headers['Content-Disposition'] = f'attachment; filename=predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
		response.headers['Content-Type'] = 'text/csv'
		
		logger.info("[EXPORT CSV] Predictions exported to CSV")
		return response
		
	except Exception as e:
		logger.error(f"[EXPORT ERROR] CSV export failed: {str(e)}")
		flash(f'Export failed: {str(e)}', 'danger')
		return redirect(url_for('index'))


@app.route('/export/predictions/excel')
def export_predictions_excel():
	"""Export predictions to Excel file"""
	try:
		cache = get_prediction_cache()
		if cache.get('result_df') is None:
			flash(config.ERROR_MESSAGES['prediction_cache_empty'], 'warning')
			return redirect(url_for('index'))
		
		# Create Excel file in memory
		output = BytesIO()
		result_df = cache['result_df']
		
		with pd.ExcelWriter(output, engine='openpyxl') as writer:
			result_df.to_excel(writer, sheet_name='Predictions', index=True)
		
		output.seek(0)
		
		# Create response
		response = make_response(output.getvalue())
		response.headers['Content-Disposition'] = f'attachment; filename=predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
		response.headers['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
		
		logger.info("[EXPORT EXCEL] Predictions exported to Excel")
		return response
		
	except Exception as e:
		logger.error(f"[EXPORT ERROR] Excel export failed: {str(e)}")
		flash(f'Export failed: {str(e)}', 'danger')
		return redirect(url_for('index'))


@app.route('/export/summary')
def export_summary_report():
	"""Export comprehensive summary report as text file"""
	try:
		cache = get_prediction_cache()
		if cache.get('predictions') is None:
			flash(config.ERROR_MESSAGES['prediction_cache_empty'], 'warning')
			return redirect(url_for('index'))
		
		predictions = cache.get('predictions', [])
		clusters = cache.get('clusters', [])
		sites = cache.get('sites', [])
		result_df = cache.get('result_df')
		
		# Build comprehensive report
		report_lines = []
		report_lines.append("=" * 80)
		report_lines.append("FUEL CONSUMPTION PREDICTION SUMMARY REPORT")
		report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
		report_lines.append("=" * 80)
		report_lines.append("")
		
		# Overview Statistics
		report_lines.append("OVERVIEW")
		report_lines.append("-" * 80)
		report_lines.append(f"Total Predictions: {len(predictions)}")
		report_lines.append(f"Total Clusters: {len(set(clusters))}")
		report_lines.append(f"Total Sites: {len(set(sites))}")
		report_lines.append(f"Total Fuel Consumption: {sum(predictions):.2f} L")
		report_lines.append(f"Average Consumption: {np.mean(predictions):.2f} L")
		report_lines.append(f"Min Consumption: {min(predictions):.2f} L")
		report_lines.append(f"Max Consumption: {max(predictions):.2f} L")
		report_lines.append(f"Std Deviation: {np.std(predictions):.2f} L")
		report_lines.append("")
		
		# Quartile Analysis
		report_lines.append("QUARTILE ANALYSIS")
		report_lines.append("-" * 80)
		q1, q2, q3 = np.percentile(predictions, [25, 50, 75])
		report_lines.append(f"Q1 (25th percentile): {q1:.2f} L")
		report_lines.append(f"Q2 (50th percentile/Median): {q2:.2f} L")
		report_lines.append(f"Q3 (75th percentile): {q3:.2f} L")
		report_lines.append(f"Interquartile Range (IQR): {q3 - q1:.2f} L")
		report_lines.append("")
		
		# Anomaly Detection
		mean_pred = np.mean(predictions)
		std_pred = np.std(predictions)
		threshold = mean_pred + config.ANOMALY_STD_MULTIPLIER * std_pred
		anomalies = [(clusters[i], sites[i], predictions[i]) for i in range(len(predictions)) if predictions[i] > threshold]
		
		report_lines.append("ANOMALY DETECTION")
		report_lines.append("-" * 80)
		report_lines.append(f"Threshold (mean + {config.ANOMALY_STD_MULTIPLIER}σ): {threshold:.2f} L")
		report_lines.append(f"Number of Anomalies: {len(anomalies)}")
		if anomalies:
			report_lines.append("\nAnomalous Sites (High Consumption):")
			for cluster, site, value in sorted(anomalies, key=lambda x: x[2], reverse=True):
				report_lines.append(f"  - Cluster: {cluster}, Site: {site}, Consumption: {value:.2f} L")
		report_lines.append("")
		
		# Top 10 Clusters
		if result_df is not None:
			report_lines.append("TOP 10 CLUSTERS BY PREDICTED CONSUMPTION")
			report_lines.append("-" * 80)
			top_clusters = result_df.nlargest(config.TOP_CLUSTERS_LIMIT, 'Predictions')
			for idx, (cluster, row) in enumerate(top_clusters.iterrows(), 1):
				report_lines.append(f"{idx}. {cluster}: {row['Predictions']:.2f} L")
			report_lines.append("")
		
		# Model Performance (if available)
		nse = prediction_cache.get('nse_value')
		if nse is not None:
			report_lines.append("MODEL PERFORMANCE")
			report_lines.append("-" * 80)
			report_lines.append(f"Nash-Sutcliffe Efficiency (NSE): {nse:.4f}")
			if nse > 0.75:
				report_lines.append("Model Quality: Excellent")
			elif nse > 0.65:
				report_lines.append("Model Quality: Good")
			elif nse > 0.50:
				report_lines.append("Model Quality: Acceptable")
			else:
				report_lines.append("Model Quality: Needs Improvement")
			report_lines.append("")
		
		report_lines.append("=" * 80)
		report_lines.append("END OF REPORT")
		report_lines.append("=" * 80)
		
		# Create text file
		report_text = "\n".join(report_lines)
		response = make_response(report_text)
		response.headers['Content-Disposition'] = f'attachment; filename=summary_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
		response.headers['Content-Type'] = 'text/plain'
		
		logger.info("[EXPORT SUMMARY] Summary report exported")
		return response
		
	except Exception as e:
		logger.error(f"[EXPORT ERROR] Summary report export failed: {str(e)}")
		flash(f'Export failed: {str(e)}', 'danger')
		return redirect(url_for('index'))


@app.route('/export/chart/<chart_type>')
def export_chart(chart_type):
	"""Export chart as SVG file"""
	try:
		cache = get_prediction_cache()
		if cache.get('predictions') is None:
			flash(config.ERROR_MESSAGES['prediction_cache_empty'], 'warning')
			return redirect(url_for('index'))
		
		result_df = prediction_cache.get('result_df')
		
		# Generate chart based on type
		if chart_type == 'cluster':
			if result_df is None:
				flash('No cluster data available', 'warning')
				return redirect(url_for('index'))
			
			graph = pygal.Bar()
			graph.title = 'Prediction of Fuel Consumption Per Cluster'
			graph.x_title = 'Clusters'
			graph.y_title = 'Predicted Fuel Consumption (L)'
			graph.x_label_rotation = 60
			
			for k in result_df.index:
				graph.add(k, result_df['Predictions'][k])
			
			filename_base = f'cluster_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
			
		elif chart_type == 'site':
			predictions = prediction_cache.get('predictions', [])
			sites = prediction_cache.get('sites', [])
			
			if len(predictions) > 0 and len(sites) > 0:
				df = pd.DataFrame({'sites': sites, 'predictions': predictions})
				site_result = df.groupby('sites')['predictions'].sum().sort_values(ascending=False)
				top_20 = site_result.head(config.TOP_SITES_LIMIT)
				
				graph = pygal.HorizontalBar()
				graph.title = 'Top 20 Sites by Predicted Fuel Consumption'
				graph.x_title = 'Predicted Fuel Consumption (L)'
				
				for site, value in top_20.items():
					graph.add(str(site), float(value))
				
				filename_base = f'site_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
			else:
				flash('No site data available', 'warning')
				return redirect(url_for('index'))
			
		elif chart_type == 'distribution':
			predictions = cache.get('predictions', [])
			if len(predictions) > 0:
				pred_array = np.array(predictions)
				hist_data, bin_edges = np.histogram(pred_array, bins=config.DISTRIBUTION_BINS)
				
				bin_labels = []
				for i in range(len(bin_edges)-1):
					bin_labels.append(f'[{int(bin_edges[i])}, {int(bin_edges[i+1])}]')
				
				graph = pygal.Bar()
				graph.title = 'Distribution of Fuel Consumption Predictions'
				graph.x_title = 'Fuel Consumption Range (L)'
				graph.y_title = 'Number of Sites'
				graph.x_label_rotation = 60
				graph.x_labels = bin_labels
				graph.add('Frequency', list(hist_data))
				
				filename_base = f'distribution_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
			else:
				flash('No prediction data available', 'warning')
				return redirect(url_for('index'))
			
		elif chart_type == 'comparison':
			predictions = prediction_cache.get('predictions', [])
			clusters = prediction_cache.get('clusters', [])
			sites = prediction_cache.get('sites', [])
			if len(predictions) > 0 and len(clusters) > 0 and len(sites) > 0:
				df = pd.DataFrame({'clusters': clusters, 'sites': sites, 'predictions': predictions})
				cluster_totals = df.groupby('clusters')['predictions'].sum().sort_values(ascending=False)
				
				# Use same 10 colors as comparison view
				site_colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', 
				               '#00f2fe', '#43e97b', '#38f9d7', '#fa709a', '#fee140']
				from pygal.style import Style
				custom_style = Style(colors=site_colors + ['#cccccc'], value_font_size=10, value_label_font_size=10)
				graph = pygal.StackedBar(style=custom_style)
				graph.title = 'Fuel Consumption: Cluster Breakdown by Top Sites'
				graph.x_title = 'Clusters'
				graph.y_title = 'Predicted Fuel Consumption (L)'
				graph.x_label_rotation = 60
				graph.show_legend = True
				
				# Match the exact logic from comparison view
				cluster_top_sites = {}
				all_top_sites = {}
				for cluster in cluster_totals.index:
					cluster_sites = df[df['clusters'] == cluster].groupby('sites')['predictions'].sum()
					if not cluster_sites.empty:
						top_site = cluster_sites.idxmax()
						top_value = float(cluster_sites.max())
						cluster_total = float(cluster_totals[cluster])
						contribution_pct = round((top_value / cluster_total) * 100, 1)
						cluster_top_sites[cluster] = (top_site, top_value, contribution_pct)
						if top_site in all_top_sites:
							all_top_sites[top_site] += top_value
						else:
							all_top_sites[top_site] = top_value
				
				# Sort sites by total consumption (ALL unique top sites, not limited)
				ranked_sites = sorted(all_top_sites.items(), key=lambda x: x[1], reverse=True)
				
				# Create series for each unique top site
				site_series = {site: [] for site, _ in ranked_sites}
				other_sites_series = []
				
				for cluster in cluster_totals.index:
					if cluster in cluster_top_sites:
						top_site, top_value, contribution_pct = cluster_top_sites[cluster]
						cluster_total = float(cluster_totals[cluster])
						other_value = cluster_total - top_value
						
						# Add value for the top site in this cluster
						for site in site_series:
							if site == top_site:
								site_series[site].append(top_value)
							else:
								site_series[site].append(0)
						
						# Add "other sites" value
						other_sites_series.append(other_value)
					else:
						# No data for this cluster
						for site in site_series:
							site_series[site].append(0)
						other_sites_series.append(float(cluster_totals[cluster]))
				
				# Set cluster names as x-labels
				graph.x_labels = list(cluster_totals.index)
				
				# Add series for each top site (creates different colors - show ALL sites)
				for site, _ in ranked_sites:
					graph.add(site, site_series[site])
				
				# Add "Other Sites" series
				graph.add('Other Sites', other_sites_series)
				
				filename_base = f'comparison_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
			else:
				flash('No comparison data available', 'warning')
				return redirect(url_for('index'))
			
		elif chart_type == 'timeseries':
			excel_file = get_data_file(session, UPLOAD_FOLDER)
			if excel_file is None:
				flash('No data file available for time series', 'warning')
				return redirect(url_for('index'))
			
			raw_data, _, _ = get_cached_data(excel_file)
			if raw_data is not None and len(raw_data) > 0:
				# Find date column dynamically
				date_columns = []
				for col in raw_data.columns:
					col_lower = str(col).lower()
					if any(keyword in col_lower for keyword in ['date', 'time', 'month', 'year', 'period', 'day']):
						date_columns.append(col)
				if not date_columns:
					flash('No date column found in data', 'warning')
					return redirect(url_for('index'))
				date_col = date_columns[0]
				
				# Find fuel column dynamically
				numeric_cols = raw_data.select_dtypes(include=['float64', 'int64']).columns
				fuel_col = None
				for col in numeric_cols:
					col_lower = str(col).lower()
					if any(keyword in col_lower for keyword in ['fuel', 'consumption', 'qty', 'qte', 'litre', 'liter']):
						fuel_col = col
						break
				if fuel_col is None and len(numeric_cols) > 0:
					fuel_col = numeric_cols[0]
				if fuel_col is None:
					flash('No numeric column found for time series', 'warning')
					return redirect(url_for('index'))
				
				raw_data[date_col] = pd.to_datetime(raw_data[date_col], errors='coerce')
				raw_data = raw_data.dropna(subset=[date_col])
				raw_data = raw_data.sort_values(date_col)
				
				# Determine grouping frequency
				date_range = (raw_data[date_col].max() - raw_data[date_col].min()).days
				if date_range > 365:
					freq = 'M'
					title_suffix = '(Monthly)'
				elif date_range > 30:
					freq = 'W'
					title_suffix = '(Weekly)'
				else:
					freq = 'D'
					title_suffix = '(Daily)'
			
			raw_data['period'] = raw_data[date_col].dt.to_period(freq)
			time_series = raw_data.groupby('period')[fuel_col].sum()
			
			graph = pygal.Line()
			graph.title = f'Fuel Consumption Trend Over Time {title_suffix}'
			graph.x_title = 'Time Period'
			graph.y_title = 'Fuel Consumption (L)'
			graph.x_label_rotation = 45
			graph.show_legend = True
			
			# Format x-labels
			x_labels = []
			for period in time_series.index:
				period_dt = period.to_timestamp()
				if freq == 'M':
					x_labels.append(period_dt.strftime('%b %Y'))
				elif freq == 'W':
					week_num = period_dt.isocalendar()[1]
					x_labels.append(f'W{week_num} {period_dt.year}')
				else:
					x_labels.append(period_dt.strftime('%d %b %Y'))
			
			graph.x_labels = x_labels
			graph.x_labels_major_every = max(1, len(x_labels) // 10)
			graph.show_minor_x_labels = False
			graph.add('Actual Consumption', [float(v) for v in time_series.values])
			
			if len(time_series) >= 3:
				window = min(3, len(time_series) // 3)
				moving_avg = time_series.rolling(window=window, center=True).mean()
				graph.add('Trend (Moving Avg)', [float(v) if pd.notna(v) else None for v in moving_avg.values])
			
				filename_base = f'timeseries_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
			else:
				flash('No time series data available', 'warning')
				return redirect(url_for('index'))
		else:
			flash('Invalid chart type', 'danger')
			return redirect(url_for('index'))
		
		# Render SVG
		svg_data = graph.render()
		response = make_response(svg_data)
		response.headers['Content-Disposition'] = f'attachment; filename={filename_base}.svg'
		response.headers['Content-Type'] = 'image/svg+xml'
		
		logger.info(f"[EXPORT CHART SVG] {chart_type} chart exported")
		return response
		
	except Exception as e:
		logger.error(f"[EXPORT ERROR] SVG chart export failed: {str(e)}")
		flash(f'Export failed: {str(e)}', 'danger')
		return redirect(url_for('index'))


@app.route('/export/chart/<chart_type>/png')
def export_chart_png(chart_type):
	"""Export chart as PNG file"""
	try:
		if not HAS_CAIROSVG:
			flash('PNG export not available. Install cairosvg: pip install cairosvg', 'warning')
			return redirect(url_for('index'))
		
		cache = get_prediction_cache()
		if cache.get('predictions') is None:
			flash(config.ERROR_MESSAGES['prediction_cache_empty'], 'warning')
			return redirect(url_for('index'))
		
		result_df = prediction_cache.get('result_df')
		
		# Generate chart based on type (same logic as SVG export)
		if chart_type == 'cluster':
			if result_df is None:
				flash('No cluster data available', 'warning')
				return redirect(url_for('index'))
			graph = pygal.Bar()
			graph.title = 'Prediction of Fuel Consumption Per Cluster'
			graph.x_title = 'Clusters'
			graph.y_title = 'Predicted Fuel Consumption (L)'
			graph.x_label_rotation = 60
			for k in result_df.index:
				graph.add(k, result_df['Predictions'][k])
			filename_base = f'cluster_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
			
		elif chart_type == 'site':
			predictions = prediction_cache.get('predictions', [])
			sites = prediction_cache.get('sites', [])
			if len(predictions) > 0 and len(sites) > 0:
				df = pd.DataFrame({'sites': sites, 'predictions': predictions})
				site_result = df.groupby('sites')['predictions'].sum().sort_values(ascending=False)
				top_20 = site_result.head(config.TOP_SITES_LIMIT)
				
				graph = pygal.HorizontalBar()
				graph.title = 'Top 20 Sites by Predicted Fuel Consumption'
				graph.x_title = 'Predicted Fuel Consumption (L)'
				
				for site, value in top_20.items():
					graph.add(str(site), float(value))
				
				filename_base = f'site_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
			else:
				flash('No site data available', 'warning')
				return redirect(url_for('index'))
				
		elif chart_type == 'distribution':
			predictions = cache.get('predictions', [])
			if len(predictions) > 0:
				pred_array = np.array(predictions)
				hist_data, bin_edges = np.histogram(pred_array, bins=config.DISTRIBUTION_BINS)
				
				bin_labels = []
				for i in range(len(bin_edges)-1):
					bin_labels.append(f'[{int(bin_edges[i])}, {int(bin_edges[i+1])}]')
				
				graph = pygal.Bar()
				graph.title = 'Distribution of Fuel Consumption Predictions'
				graph.x_title = 'Fuel Consumption Range (L)'
				graph.y_title = 'Number of Sites'
				graph.x_label_rotation = 60
				graph.x_labels = bin_labels
				graph.add('Frequency', list(hist_data))
				
				filename_base = f'distribution_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
			else:
				flash('No prediction data available', 'warning')
				return redirect(url_for('index'))
			
		elif chart_type == 'comparison':
			predictions = prediction_cache.get('predictions', [])
			clusters = prediction_cache.get('clusters', [])
			sites = prediction_cache.get('sites', [])
			if len(predictions) > 0 and len(clusters) > 0 and len(sites) > 0:
				df = pd.DataFrame({'clusters': clusters, 'sites': sites, 'predictions': predictions})
				cluster_totals = df.groupby('clusters')['predictions'].sum().sort_values(ascending=False)
				
				# Use same 10 colors as comparison view
				site_colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', 
				               '#00f2fe', '#43e97b', '#38f9d7', '#fa709a', '#fee140']
				from pygal.style import Style
				custom_style = Style(colors=site_colors + ['#cccccc'], value_font_size=10, value_label_font_size=10)
				graph = pygal.StackedBar(style=custom_style)
				graph.title = 'Fuel Consumption: Cluster Breakdown by Top Sites'
				graph.x_title = 'Clusters'
				graph.y_title = 'Predicted Fuel Consumption (L)'
				graph.x_label_rotation = 60
				graph.show_legend = True
				
				# Match the exact logic from comparison view
				cluster_top_sites = {}
				all_top_sites = {}
				for cluster in cluster_totals.index:
					cluster_sites = df[df['clusters'] == cluster].groupby('sites')['predictions'].sum()
					if not cluster_sites.empty:
						top_site = cluster_sites.idxmax()
						top_value = float(cluster_sites.max())
						cluster_total = float(cluster_totals[cluster])
						contribution_pct = round((top_value / cluster_total) * 100, 1)
						cluster_top_sites[cluster] = (top_site, top_value, contribution_pct)
						if top_site in all_top_sites:
							all_top_sites[top_site] += top_value
						else:
							all_top_sites[top_site] = top_value
				
				# Sort sites by total consumption (ALL unique top sites, not limited)
				ranked_sites = sorted(all_top_sites.items(), key=lambda x: x[1], reverse=True)
				
				# Create series for each unique top site
				site_series = {site: [] for site, _ in ranked_sites}
				other_sites_series = []
				
				for cluster in cluster_totals.index:
					if cluster in cluster_top_sites:
						top_site, top_value, contribution_pct = cluster_top_sites[cluster]
						cluster_total = float(cluster_totals[cluster])
						other_value = cluster_total - top_value
						
						# Add value for the top site in this cluster
						for site in site_series:
							if site == top_site:
								site_series[site].append(top_value)
							else:
								site_series[site].append(0)
						
						# Add "other sites" value
						other_sites_series.append(other_value)
					else:
						# No data for this cluster
						for site in site_series:
							site_series[site].append(0)
						other_sites_series.append(float(cluster_totals[cluster]))
				
				# Set cluster names as x-labels
				graph.x_labels = list(cluster_totals.index)
				
				# Add series for each top site (creates different colors - show ALL sites)
				for site, _ in ranked_sites:
					graph.add(site, site_series[site])
				
				# Add "Other Sites" series
				graph.add('Other Sites', other_sites_series)
				
				filename_base = f'comparison_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
			else:
				flash('No comparison data available', 'warning')
				return redirect(url_for('index'))
			
		elif chart_type == 'timeseries':
			excel_file = get_data_file(session, UPLOAD_FOLDER)
			if excel_file is None:
				flash('No data file available for time series', 'warning')
				return redirect(url_for('index'))
			
			raw_data, _, _ = get_cached_data(excel_file)
			if raw_data is not None and len(raw_data) > 0:
				# Find date column dynamically
				date_columns = []
				for col in raw_data.columns:
					col_lower = str(col).lower()
					if any(keyword in col_lower for keyword in ['date', 'time', 'month', 'year', 'period', 'day']):
						date_columns.append(col)
				if not date_columns:
					flash('No date column found in data', 'warning')
					return redirect(url_for('index'))
				date_col = date_columns[0]
				
				# Find fuel column dynamically
				numeric_cols = raw_data.select_dtypes(include=['float64', 'int64']).columns
				fuel_col = None
				for col in numeric_cols:
					col_lower = str(col).lower()
					if any(keyword in col_lower for keyword in ['fuel', 'consumption', 'qty', 'qte', 'litre', 'liter']):
						fuel_col = col
						break
				if fuel_col is None and len(numeric_cols) > 0:
					fuel_col = numeric_cols[0]
				if fuel_col is None:
					flash('No numeric column found for time series', 'warning')
					return redirect(url_for('index'))
				
				raw_data[date_col] = pd.to_datetime(raw_data[date_col], errors='coerce')
				raw_data = raw_data.dropna(subset=[date_col])
				raw_data = raw_data.sort_values(date_col)
				
				# Determine grouping frequency
				date_range = (raw_data[date_col].max() - raw_data[date_col].min()).days
				if date_range > 365:
					freq = 'M'
					title_suffix = '(Monthly)'
				elif date_range > 30:
					freq = 'W'
					title_suffix = '(Weekly)'
				else:
					freq = 'D'
					title_suffix = '(Daily)'
			
			raw_data['period'] = raw_data[date_col].dt.to_period(freq)
			time_series = raw_data.groupby('period')[fuel_col].sum()
			
			graph = pygal.Line()
			graph.title = f'Fuel Consumption Trend Over Time {title_suffix}'
			graph.x_title = 'Time Period'
			graph.y_title = 'Fuel Consumption (L)'
			graph.x_label_rotation = 45
			graph.show_legend = True
			
			# Format x-labels
			x_labels = []
			for period in time_series.index:
				period_dt = period.to_timestamp()
				if freq == 'M':
					x_labels.append(period_dt.strftime('%b %Y'))
				elif freq == 'W':
					week_num = period_dt.isocalendar()[1]
					x_labels.append(f'W{week_num} {period_dt.year}')
				else:
					x_labels.append(period_dt.strftime('%d %b %Y'))
			
			graph.x_labels = x_labels
			graph.x_labels_major_every = max(1, len(x_labels) // 10)
			graph.show_minor_x_labels = False
			graph.add('Actual Consumption', [float(v) for v in time_series.values])
			
			if len(time_series) >= 3:
				window = min(3, len(time_series) // 3)
				moving_avg = time_series.rolling(window=window, center=True).mean()
				graph.add('Trend (Moving Avg)', [float(v) if pd.notna(v) else None for v in moving_avg.values])
			
				filename_base = f'timeseries_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
			else:
				flash('No time series data available', 'warning')
				return redirect(url_for('index'))
		else:
			flash('Invalid chart type', 'danger')
			return redirect(url_for('index'))
		
		# Render to PNG using cairosvg
		svg_data = graph.render()
		png_data = cairosvg.svg2png(bytestring=svg_data, dpi=config.PNG_EXPORT_DPI)
		
		response = make_response(png_data)
		response.headers['Content-Disposition'] = f'attachment; filename={filename_base}.png'
		response.headers['Content-Type'] = 'image/png'
		
		logger.info(f"[EXPORT CHART PNG] {chart_type} chart exported")
		return response
		
	except Exception as e:
		logger.error(f"[EXPORT ERROR] PNG chart export failed: {str(e)}")
		flash(f'Export failed: {str(e)}', 'danger')
		return redirect(url_for('index'))


if __name__ == '__main__':
	app.run(port=6003,debug=True)




