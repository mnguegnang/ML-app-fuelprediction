"""
Data validation utilities
Handles form validation, column checking, and data quality validation
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def validate_form_inputs(form_data, required_fields):
    """
    Validate that all required form fields are present and non-empty
    
    Args:
        form_data: Flask request.form object
        required_fields: List of required field names
    
    Returns:
        tuple: (is_valid: bool, missing_fields: list, error_message: str or None)
    """
    missing_fields = []
    
    for field in required_fields:
        if field not in form_data or not form_data[field]:
            missing_fields.append(field)
    
    if missing_fields:
        error_msg = f"Missing required field(s): {', '.join(missing_fields)}"
        logger.warning(f"Form validation failed: {error_msg}")
        return False, missing_fields, error_msg
    
    logger.info(f"Form validation passed: All {len(required_fields)} required fields present")
    return True, [], None


def resolve_column_name(column_name, dataframe):
    """
    Resolve column name with case-insensitive matching
    
    Args:
        column_name: Column name to find
        dataframe: DataFrame to search in
    
    Returns:
        str: Actual column name in DataFrame, or None if not found
    """
    if column_name in dataframe.columns:
        return column_name
    
    lower_map = {c.lower(): c for c in dataframe.columns}
    return lower_map.get(column_name.lower())


def validate_columns_exist(column_mapping, dataframe):
    """
    Validate that all required columns exist in the DataFrame
    
    Args:
        column_mapping: Dict mapping form fields to column names
        dataframe: DataFrame to check
    
    Returns:
        tuple: (is_valid: bool, missing_columns: list, resolved_mapping: dict)
    """
    missing_columns = []
    resolved_mapping = {}
    
    for form_field, column_name in column_mapping.items():
        resolved = resolve_column_name(column_name, dataframe)
        if resolved:
            resolved_mapping[form_field] = resolved
        else:
            missing_columns.append(column_name)
    
    if missing_columns:
        logger.error(f"Missing columns: {missing_columns}")
        logger.debug(f"Available columns: {list(dataframe.columns)}")
        return False, missing_columns, {}
    
    logger.info(f"Column validation passed: All {len(column_mapping)} columns found")
    return True, [], resolved_mapping


def validate_dataframe(df, min_rows=1):
    """
    Validate DataFrame has minimum required data
    
    Args:
        df: DataFrame to validate
        min_rows: Minimum number of rows required
    
    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    if df is None:
        return False, "DataFrame is None"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if len(df) < min_rows:
        return False, f"DataFrame has {len(df)} rows, but minimum {min_rows} required"
    
    logger.info(f"DataFrame validation passed: {len(df)} rows, {len(df.columns)} columns")
    return True, None


def validate_required_feature(features_df, required_feature):
    """
    Validate that required model feature exists after one-hot encoding
    
    Args:
        features_df: DataFrame with encoded features
        required_feature: Name of required feature
    
    Returns:
        tuple: (is_valid: bool, error_message: str or None)
    """
    if required_feature not in features_df.columns:
        available_features = list(features_df.columns)
        error_msg = (
            f"Model requires feature '{required_feature}' which is not present in the data. "
            f"This may indicate a data format mismatch. "
            f"Available features: {available_features}"
        )
        logger.error(error_msg)
        return False, error_msg
    
    logger.info(f"Required feature '{required_feature}' found in data")
    return True, None
