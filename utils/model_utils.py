"""
Model loading and metadata utilities
Handles ML model operations and metadata retrieval
"""

import os
import pickle
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def load_model(model_path):
    """
    Load the machine learning model from pickle file
    
    Args:
        model_path: Path to the model pickle file
    
    Returns:
        object: Loaded model, or None if error
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
        
        logger.info(f"Model loaded successfully from {model_path}")
        return clf
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None


def get_model_metadata(config, clf=None):
    """
    Get comprehensive model metadata including version, performance metrics, and specifications.
    
    Args:
        config: Configuration module with MODEL_METADATA
        clf: Optional model object (for status check)
    
    Returns:
        dict: Model metadata for display in UI
    """
    metadata = config.MODEL_METADATA.copy()
    
    # Add runtime information
    metadata['status'] = 'loaded' if clf is not None else 'not_loaded'
    metadata['model_file'] = config.MODEL_FILENAME
    
    # Add current configuration
    metadata['current_config'] = {
        'anomaly_threshold': config.ANOMALY_STD_MULTIPLIER,
        'top_sites_limit': config.TOP_SITES_LIMIT,
        'distribution_bins': config.DISTRIBUTION_BINS
    }
    
    # Calculate model file info if it exists
    try:
        model_file_path = config.MODEL_PATH
        if os.path.exists(model_file_path):
            model_size = os.path.getsize(model_file_path) / (1024 * 1024)  # MB
            model_modified = datetime.fromtimestamp(os.path.getmtime(model_file_path))
            metadata['file_info'] = {
                'size_mb': round(model_size, 2),
                'last_modified': model_modified.strftime('%Y-%m-%d %H:%M:%S')
            }
    except Exception as e:
        logger.warning(f"Could not get model file info: {e}")
    
    # Add feature count
    metadata['feature_count'] = len(config.MODEL_FEATURES)
    
    return metadata


def validate_model_loaded(clf):
    """
    Validate that model is loaded and ready
    
    Args:
        clf: Model object
    
    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    if clf is None:
        return False, "Model not loaded. Please contact administrator."
    
    return True, None
