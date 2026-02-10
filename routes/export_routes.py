"""
Export Routes Blueprint

This module contains routes for exporting data and charts in various formats:
- export_predictions_csv: Export predictions to CSV format
- export_predictions_excel: Export predictions to Excel format with formatting
- export_summary_report: Generate comprehensive text summary report
- export_chart: Export charts as SVG format (vector graphics)
- export_chart_png: Export charts as PNG format (raster images, requires cairosvg)

All export routes:
1. Check for cached predictions
2. Use export_utils functions to generate export files
3. Return files with appropriate headers for download
4. Handle errors gracefully with flash messages
"""

from flask import Blueprint, flash, redirect, url_for, make_response, session
import pandas as pd
import logging

# Import configuration
import config

# Import utilities
from utils.data_utils import (
    get_cached_data,
    get_data_file,
    get_prediction_cache
)
from utils.export_utils import (
    export_to_csv,
    export_to_excel,
    generate_summary_report,
    export_chart_svg,
    export_chart_png,
    HAS_CAIROSVG
)

logger = logging.getLogger(__name__)

# Create Blueprint
export_bp = Blueprint('export', __name__)

# These will be injected when blueprint is registered
UPLOAD_FOLDER = None
prediction_cache = None


def init_blueprint(upload_folder, p_cache):
    """Initialize blueprint with shared resources from main app"""
    global UPLOAD_FOLDER, prediction_cache
    UPLOAD_FOLDER = upload_folder
    prediction_cache = p_cache


@export_bp.route('/export/predictions/csv')
def export_predictions_csv():
    """Export predictions to CSV file"""
    try:
        cache = get_prediction_cache()
        if cache.get('result_df') is None:
            flash(config.ERROR_MESSAGES['prediction_cache_empty'], 'warning')
            return redirect(url_for('main.index'))
        
        result_df = cache['result_df']
        csv_data, timestamp = export_to_csv(result_df)
        
        # Create response
        response = make_response(csv_data)
        response.headers['Content-Disposition'] = f'attachment; filename=predictions_{timestamp}.csv'
        response.headers['Content-Type'] = 'text/csv'
        
        logger.info("[EXPORT CSV] Predictions exported to CSV")
        return response
        
    except Exception as e:
        logger.error(f"[EXPORT ERROR] CSV export failed: {str(e)}")
        flash(f'Export failed: {str(e)}', 'danger')
        return redirect(url_for('main.index'))


@export_bp.route('/export/predictions/excel')
def export_predictions_excel():
    """Export predictions to Excel file"""
    try:
        cache = get_prediction_cache()
        if cache.get('result_df') is None:
            flash(config.ERROR_MESSAGES['prediction_cache_empty'], 'warning')
            return redirect(url_for('main.index'))
        
        result_df = cache['result_df']
        excel_data, timestamp = export_to_excel(result_df)
        
        # Create response
        response = make_response(excel_data)
        response.headers['Content-Disposition'] = f'attachment; filename=predictions_{timestamp}.xlsx'
        response.headers['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        
        logger.info("[EXPORT EXCEL] Predictions exported to Excel")
        return response
        
    except Exception as e:
        logger.error(f"[EXPORT ERROR] Excel export failed: {str(e)}")
        flash(f'Export failed: {str(e)}', 'danger')
        return redirect(url_for('main.index'))


@export_bp.route('/export/summary')
def export_summary():
    """Export comprehensive summary report as text file"""
    try:
        cache = get_prediction_cache()
        if cache.get('predictions') is None:
            flash(config.ERROR_MESSAGES['prediction_cache_empty'], 'warning')
            return redirect(url_for('main.index'))
        
        predictions = cache.get('predictions', [])
        clusters = cache.get('clusters', [])
        sites = cache.get('sites', [])
        result_df = cache.get('result_df')
        nse_value = prediction_cache.get('nse_value')
        
        report_text, timestamp = generate_summary_report(
            predictions, clusters, sites, result_df, nse_value
        )
        
        # Create text file
        response = make_response(report_text)
        response.headers['Content-Disposition'] = f'attachment; filename=summary_report_{timestamp}.txt'
        response.headers['Content-Type'] = 'text/plain'
        
        logger.info("[EXPORT SUMMARY] Summary report exported")
        return response
        
    except Exception as e:
        logger.error(f"[EXPORT ERROR] Summary report export failed: {str(e)}")
        flash(f'Export failed: {str(e)}', 'danger')
        return redirect(url_for('main.index'))


@export_bp.route('/export/chart/<chart_type>')
def export_chart(chart_type):
    """Export chart as SVG file"""
    try:
        cache = get_prediction_cache()
        if cache.get('predictions') is None:
            flash(config.ERROR_MESSAGES['prediction_cache_empty'], 'warning')
            return redirect(url_for('main.index'))
        
        # Prepare data dictionary based on chart type
        data_dict = {}
        
        if chart_type == 'cluster':
            result_df = prediction_cache.get('result_df')
            if result_df is None:
                flash('No cluster data available', 'warning')
                return redirect(url_for('main.index'))
            data_dict = {'result_df': result_df}
            
        elif chart_type == 'site':
            data_dict = {
                'predictions': prediction_cache.get('predictions', []),
                'sites': prediction_cache.get('sites', [])
            }
            
        elif chart_type == 'distribution':
            data_dict = {
                'predictions': cache.get('predictions', [])
            }
            
        elif chart_type == 'comparison':
            data_dict = {
                'predictions': prediction_cache.get('predictions', []),
                'clusters': prediction_cache.get('clusters', []),
                'sites': prediction_cache.get('sites', [])
            }
            
        elif chart_type == 'timeseries':
            excel_file = get_data_file(session, UPLOAD_FOLDER)
            if excel_file is None:
                flash('No data file available for time series', 'warning')
                return redirect(url_for('main.index'))
            
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
                    return redirect(url_for('main.index'))
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
                    return redirect(url_for('main.index'))
                
                raw_data[date_col] = pd.to_datetime(raw_data[date_col], errors='coerce')
                raw_data = raw_data.dropna(subset=[date_col])
                raw_data = raw_data.sort_values(date_col)
                
                # Determine grouping frequency
                date_range = (raw_data[date_col].max() - raw_data[date_col].min()).days
                if date_range > 365:
                    freq = 'M'
                elif date_range > 30:
                    freq = 'W'
                else:
                    freq = 'D'
                
                raw_data['period'] = raw_data[date_col].dt.to_period(freq)
                time_series = raw_data.groupby('period')[fuel_col].sum()
                
                data_dict = {
                    'time_series': time_series,
                    'freq': freq,
                    'fuel_col': fuel_col
                }
            else:
                flash('No time series data available', 'warning')
                return redirect(url_for('main.index'))
        else:
            flash('Invalid chart type', 'danger')
            return redirect(url_for('main.index'))
        
        # Export chart as SVG
        svg_data, filename = export_chart_svg(chart_type, data_dict)
        response = make_response(svg_data)
        response.headers['Content-Disposition'] = f'attachment; filename={filename}'
        response.headers['Content-Type'] = 'image/svg+xml'
        
        logger.info(f"[EXPORT CHART SVG] {chart_type} chart exported")
        return response
        
    except Exception as e:
        logger.error(f"[EXPORT ERROR] SVG chart export failed: {str(e)}")
        flash(f'Export failed: {str(e)}', 'danger')
        return redirect(url_for('main.index'))


@export_bp.route('/export/chart/<chart_type>/png')
def export_chart_png_route(chart_type):
    """Export chart as PNG file"""
    try:
        if not HAS_CAIROSVG:
            flash('PNG export not available. Install cairosvg: pip install cairosvg', 'warning')
            return redirect(url_for('main.index'))
        
        cache = get_prediction_cache()
        if cache.get('predictions') is None:
            flash(config.ERROR_MESSAGES['prediction_cache_empty'], 'warning')
            return redirect(url_for('main.index'))
        
        # Prepare data dictionary based on chart type (same logic as SVG export)
        data_dict = {}
        
        if chart_type == 'cluster':
            result_df = prediction_cache.get('result_df')
            if result_df is None:
                flash('No cluster data available', 'warning')
                return redirect(url_for('main.index'))
            data_dict = {'result_df': result_df}
            
        elif chart_type == 'site':
            data_dict = {
                'predictions': prediction_cache.get('predictions', []),
                'sites': prediction_cache.get('sites', [])
            }
            
        elif chart_type == 'distribution':
            data_dict = {
                'predictions': cache.get('predictions', [])
            }
            
        elif chart_type == 'comparison':
            data_dict = {
                'predictions': prediction_cache.get('predictions', []),
                'clusters': prediction_cache.get('clusters', []),
                'sites': prediction_cache.get('sites', [])
            }
            
        elif chart_type == 'timeseries':
            excel_file = get_data_file(session, UPLOAD_FOLDER)
            if excel_file is None:
                flash('No data file available for time series', 'warning')
                return redirect(url_for('main.index'))
            
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
                    return redirect(url_for('main.index'))
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
                    return redirect(url_for('main.index'))
                
                raw_data[date_col] = pd.to_datetime(raw_data[date_col], errors='coerce')
                raw_data = raw_data.dropna(subset=[date_col])
                raw_data = raw_data.sort_values(date_col)
                
                # Determine grouping frequency
                date_range = (raw_data[date_col].max() - raw_data[date_col].min()).days
                if date_range > 365:
                    freq = 'M'
                elif date_range > 30:
                    freq = 'W'
                else:
                    freq = 'D'
                
                raw_data['period'] = raw_data[date_col].dt.to_period(freq)
                time_series = raw_data.groupby('period')[fuel_col].sum()
                
                data_dict = {
                    'time_series': time_series,
                    'freq': freq,
                    'fuel_col': fuel_col
                }
            else:
                flash('No time series data available', 'warning')
                return redirect(url_for('main.index'))
        else:
            flash('Invalid chart type', 'danger')
            return redirect(url_for('main.index'))
        
        # Export chart as PNG
        png_data, filename = export_chart_png(chart_type, data_dict)
        response = make_response(png_data)
        response.headers['Content-Disposition'] = f'attachment; filename={filename}'
        response.headers['Content-Type'] = 'image/png'
        
        logger.info(f"[EXPORT CHART PNG] {chart_type} chart exported")
        return response
        
    except Exception as e:
        logger.error(f"[EXPORT ERROR] PNG chart export failed: {str(e)}")
        flash(f'Export failed: {str(e)}', 'danger')
        return redirect(url_for('main.index'))
