"""
Data caching and management utilities
Handles data cache, prediction cache, and data loading operations
"""

import os
import time
import logging
import pandas as pd
from .file_utils import read_excel_robust, find_valid_sheet

logger = logging.getLogger(__name__)

# Data caching system - stores loaded Excel files to avoid repeated disk reads
data_cache = {
    'filepath': None,       # Path to the cached file
    'dataframe': None,      # Cached DataFrame
    'sheet_name': None,     # Sheet name used
    'columns': None,        # Column names list
    'timestamp': None,      # When cached (for debugging)
    'file_size': None       # File size (helps detect changes)
}

# Prediction cache - stores prediction results (avoids session cookie size limits)
prediction_cache = {
    'clusters': None,
    'sites': None,
    'predictions': None,
    'result_df': None,  # Grouped DataFrame by cluster
    'timestamp': None
}


def clear_cache():
    """Clear the data cache and prediction cache"""
    global data_cache, prediction_cache
    
    # Clear dictionaries in-place to preserve references
    data_cache.clear()
    data_cache.update({
        'filepath': None,
        'dataframe': None,
        'sheet_name': None,
        'columns': None,
        'timestamp': None,
        'file_size': None
    })
    
    prediction_cache.clear()
    prediction_cache.update({
        'clusters': None,
        'sites': None,
        'predictions': None,
        'result_df': None,
        'timestamp': None
    })
    
    logger.info("Cache cleared (data and predictions)")


def get_cached_data(filepath, request_metrics=None):
    """
    Get cached DataFrame if available and valid, otherwise load and cache it.
    
    Args:
        filepath: Path to Excel file
        request_metrics: Optional metrics dict to track cache hits/misses
    
    Returns:
        tuple: (DataFrame, sheet_name, columns) or (None, None, None) if error
    """
    # Check if file exists
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return None, None, None
    
    file_size = os.path.getsize(filepath)
    
    # Check if cache is valid
    if (data_cache['filepath'] == filepath and 
        data_cache['dataframe'] is not None and
        data_cache['file_size'] == file_size):
        if request_metrics is not None:
            request_metrics['cache_hits'] += 1
        logger.info(f"[CACHE HIT] Using cached data for {filepath}")
        return data_cache['dataframe'], data_cache['sheet_name'], data_cache['columns']
    
    # Cache miss or invalid - load the file
    if request_metrics is not None:
        request_metrics['cache_misses'] += 1
    logger.info(f"[CACHE MISS] Loading file: {filepath}")
    
    try:
        # Find valid sheet
        sheet_name = find_valid_sheet(filepath)
        if sheet_name is None:
            logger.error("No valid sheet found")
            return None, None, None
        
        # Load DataFrame
        df = read_excel_robust(filepath, sheet_name=sheet_name)
        
        if df is None or df.empty:
            logger.error("DataFrame is empty")
            return None, None, None
        
        columns = list(df.columns)
        
        # Update cache
        data_cache['filepath'] = filepath
        data_cache['dataframe'] = df
        data_cache['sheet_name'] = sheet_name
        data_cache['columns'] = columns
        data_cache['timestamp'] = time.time()
        data_cache['file_size'] = file_size
        
        logger.info(f"Cached data: {len(df)} rows, {len(columns)} columns, sheet: {sheet_name}")
        return df, sheet_name, columns
        
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")
        return None, None, None


def get_data_file(session, upload_folder):
    """
    Get the current data file path from session
    
    Args:
        session: Flask session object
        upload_folder: Path to upload folder
    
    Returns:
        str: Path to data file, or None if not found
    """
    if 'uploaded_file' in session:
        uploaded_path = os.path.join(upload_folder, session['uploaded_file'])
        if os.path.exists(uploaded_path):
            return uploaded_path
        else:
            logger.warning(f"Uploaded file not found: {uploaded_path}")
            session.pop('uploaded_file', None)
    
    return None


def update_prediction_cache(clusters, sites, predictions, result_df):
    """
    Update prediction cache with new results
    
    Args:
        clusters: List of cluster names
        sites: List of site names
        predictions: List of prediction values
        result_df: Grouped DataFrame by cluster
    """
    global prediction_cache
    prediction_cache.update({
        'clusters': list(clusters),
        'sites': list(sites),
        'predictions': predictions,
        'result_df': result_df,
        'timestamp': time.time()
    })
    logger.info(f"Prediction cache updated: {len(predictions)} predictions")


def get_prediction_cache():
    """
    Get current prediction cache
    
    Returns:
        dict: Prediction cache dictionary
    """
    return prediction_cache
