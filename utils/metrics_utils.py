"""
Metrics tracking and statistical analysis utilities
Handles request metrics, NSE calculation, anomaly detection, and prediction statistics
"""

import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

# Global request metrics tracker
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
    'nse': None
}


def reset_metrics():
    """Reset all metrics to initial values"""
    global request_metrics
    request_metrics.clear()
    request_metrics.update({
        'total_predictions': 0,
        'successful_predictions': 0,
        'failed_predictions': 0,
        'cache_hits': 0,
        'cache_misses': 0,
        'total_processing_time': 0.0,
        'last_request_time': None,
        'mean_prediction': None,
        'median_prediction': None,
        'std_prediction': None,
        'min_prediction': None,
        'max_prediction': None,
        'unique_clusters': None,
        'unique_sites': None,
        'nse': None
    })
    logger.info("Request metrics reset")


def increment_failed_predictions():
    """Increment failed predictions counter"""
    request_metrics['failed_predictions'] += 1


def calculate_nse(observed, predicted, session_id='anonymous'):
    """
    Calculate Nash-Sutcliffe Efficiency (NSE) coefficient.
    
    Args:
        observed: Array of observed values
        predicted: Array of predicted values
        session_id: Session identifier for logging
    
    Returns:
        float: NSE value or None if calculation fails
    """
    try:
        # Remove NaN values from both observed and predicted
        valid_mask = ~np.isnan(observed)
        if valid_mask.sum() == 0:
            logger.warning(f"[NSE WARNING] Session: {session_id} | No valid observations found (all NaN)")
            return None
        
        observed_clean = observed[valid_mask]
        predicted_clean = np.array(predicted)[valid_mask]
        
        # Calculate NSE: 1 - (sum of squared errors / sum of squared deviations from mean)
        numerator = np.sum((observed_clean - predicted_clean) ** 2)
        denominator = np.sum((observed_clean - np.mean(observed_clean)) ** 2)
        
        if denominator == 0:
            logger.warning(f"[NSE WARNING] Session: {session_id} | Cannot calculate NSE - zero variance in observed data")
            return None
        
        nse_value = float(1 - (numerator / denominator))
        logger.info(f"[NSE CALCULATED] Session: {session_id} | NSE: {nse_value:.4f} | Valid observations: {len(observed_clean)}")
        return nse_value
        
    except Exception as e:
        logger.error(f"[NSE ERROR] Session: {session_id} | Error calculating NSE: {str(e)}")
        return None


def calculate_prediction_statistics(predictions, clusters, sites, nse_value=None):
    """
    Calculate comprehensive statistics for predictions.
    
    Args:
        predictions: Array or list of prediction values
        clusters: Array or list of cluster identifiers
        sites: Array or list of site identifiers
        nse_value: Optional NSE value
    
    Returns:
        dict: Dictionary containing all statistical metrics
    """
    pred_array = np.array(predictions)
    
    stats = {
        'total_predictions': len(pred_array),
        'unique_clusters': len(set(clusters)) if clusters else 0,
        'unique_sites': len(set(sites)) if sites else 0,
        'mean_prediction': float(np.mean(pred_array)),
        'median_prediction': float(np.median(pred_array)),
        'std_prediction': float(np.std(pred_array)),
        'min_prediction': float(np.min(pred_array)),
        'max_prediction': float(np.max(pred_array)),
        'nse': nse_value
    }
    
    return stats


def detect_anomalies(predictions_series, multiplier=2):
    """
    Detect anomalies in predictions using standard deviation threshold.
    
    Args:
        predictions_series: pandas Series or array of predictions
        multiplier: Number of standard deviations for threshold (default: 2)
    
    Returns:
        tuple: (anomalies_list, stats_dict)
            anomalies_list: List of dicts with 'cluster', 'value', 'excess'
            stats_dict: Dict with 'mean', 'std', 'threshold', 'anomaly_count'
    """
    if isinstance(predictions_series, pd.Series):
        predictions = predictions_series.values
        index = predictions_series.index
    else:
        predictions = np.array(predictions_series)
        index = range(len(predictions))
    
    mean_pred = float(predictions.mean())
    std_pred = float(predictions.std())
    threshold = mean_pred + multiplier * std_pred
    
    # Identify anomalies
    anomalies = []
    for idx, value in zip(index, predictions):
        if value > threshold:
            anomalies.append({
                'cluster': str(idx),
                'value': float(value),
                'excess': float(value - mean_pred)
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
    
    return anomalies, stats


def update_request_metrics(prediction_stats, processing_time):
    """
    Update global request metrics with new prediction batch.
    
    Args:
        prediction_stats: Dict with prediction statistics from calculate_prediction_statistics
        processing_time: Time taken for prediction processing (seconds)
    """
    request_metrics['total_predictions'] += prediction_stats['total_predictions']
    request_metrics['successful_predictions'] += 1
    request_metrics['total_processing_time'] += processing_time
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


def get_metrics_summary():
    """
    Get a formatted summary of current metrics.
    
    Returns:
        dict: Current metrics dictionary
    """
    return dict(request_metrics)


def format_metrics_for_display():
    """
    Format metrics for display in UI.
    
    Returns:
        dict: Formatted metrics with user-friendly labels
    """
    avg_time = (request_metrics['total_processing_time'] / request_metrics['successful_predictions'] 
                if request_metrics['successful_predictions'] > 0 else 0)
    
    return {
        'total_predictions': request_metrics['total_predictions'],
        'successful_requests': request_metrics['successful_predictions'],
        'failed_requests': request_metrics['failed_predictions'],
        'cache_hits': request_metrics['cache_hits'],
        'cache_misses': request_metrics['cache_misses'],
        'avg_processing_time': f"{avg_time:.3f}s",
        'total_processing_time': f"{request_metrics['total_processing_time']:.2f}s",
        'last_request': request_metrics['last_request_time'] or 'N/A',
        'mean_prediction': f"{request_metrics['mean_prediction']:.2f}L" if request_metrics['mean_prediction'] else 'N/A',
        'median_prediction': f"{request_metrics['median_prediction']:.2f}L" if request_metrics['median_prediction'] else 'N/A',
        'std_prediction': f"{request_metrics['std_prediction']:.2f}L" if request_metrics['std_prediction'] else 'N/A',
        'min_prediction': f"{request_metrics['min_prediction']:.2f}L" if request_metrics['min_prediction'] else 'N/A',
        'max_prediction': f"{request_metrics['max_prediction']:.2f}L" if request_metrics['max_prediction'] else 'N/A',
        'unique_clusters': request_metrics['unique_clusters'] or 'N/A',
        'unique_sites': request_metrics['unique_sites'] or 'N/A',
        'nse': f"{request_metrics['nse']:.4f}" if request_metrics['nse'] is not None else 'N/A'
    }
