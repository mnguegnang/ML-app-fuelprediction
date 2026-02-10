"""
Export Utilities Module

This module provides functions for exporting predictions and charts in various formats:
- CSV export for tabular data
- Excel export with formatted worksheets
- Summary reports in text format
- SVG chart export for vector graphics
- PNG chart export for raster images

Functions:
    export_to_csv: Export DataFrame to CSV format in memory
    export_to_excel: Export DataFrame to Excel format with formatting
    generate_summary_report: Create comprehensive text summary report
    export_chart_svg: Generate SVG chart file from chart type
    export_chart_png: Generate PNG chart file from chart type (requires cairosvg)
"""

import pandas as pd
import numpy as np
from io import BytesIO, StringIO
from datetime import datetime
import logging
import pygal
from pygal.style import Style

# Import configuration
import config

logger = logging.getLogger(__name__)

# Try to import cairosvg for PNG export
try:
    import cairosvg
    HAS_CAIROSVG = True
except ImportError:
    HAS_CAIROSVG = False
    logger.warning("cairosvg not installed - PNG export will not be available")


def export_to_csv(result_df):
    """
    Export DataFrame to CSV format in memory.
    
    Args:
        result_df (pd.DataFrame): DataFrame containing predictions to export
        
    Returns:
        tuple: (csv_data_string, timestamp_string)
            csv_data_string: CSV formatted string
            timestamp_string: Formatted timestamp for filename
            
    Raises:
        ValueError: If result_df is None or empty
    """
    if result_df is None or len(result_df) == 0:
        raise ValueError("Cannot export empty DataFrame")
    
    # Create CSV in memory
    output = StringIO()
    result_df.to_csv(output)
    output.seek(0)
    csv_data = output.getvalue()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"[EXPORT CSV] Generated CSV with {len(result_df)} rows")
    
    return csv_data, timestamp


def export_to_excel(result_df, sheet_name='Predictions'):
    """
    Export DataFrame to Excel format with formatting.
    
    Args:
        result_df (pd.DataFrame): DataFrame containing predictions to export
        sheet_name (str): Name of the Excel worksheet (default: 'Predictions')
        
    Returns:
        tuple: (excel_bytes, timestamp_string)
            excel_bytes: Binary Excel file data
            timestamp_string: Formatted timestamp for filename
            
    Raises:
        ValueError: If result_df is None or empty
    """
    if result_df is None or len(result_df) == 0:
        raise ValueError("Cannot export empty DataFrame")
    
    # Create Excel file in memory
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        result_df.to_excel(writer, sheet_name=sheet_name, index=True)
    
    output.seek(0)
    excel_data = output.getvalue()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"[EXPORT EXCEL] Generated Excel with {len(result_df)} rows in sheet '{sheet_name}'")
    
    return excel_data, timestamp


def generate_summary_report(predictions, clusters, sites, result_df=None, nse_value=None):
    """
    Generate a comprehensive summary report in text format.
    
    Args:
        predictions (list): List of prediction values
        clusters (list): List of cluster identifiers
        sites (list): List of site identifiers
        result_df (pd.DataFrame, optional): DataFrame with cluster-level aggregated predictions
        nse_value (float, optional): Nash-Sutcliffe Efficiency value for model performance
        
    Returns:
        tuple: (report_text, timestamp_string)
            report_text: Formatted text report
            timestamp_string: Formatted timestamp for filename
            
    Raises:
        ValueError: If predictions list is empty
    """
    if not predictions or len(predictions) == 0:
        raise ValueError("Cannot generate report with empty predictions")
    
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
    anomalies = [(clusters[i], sites[i], predictions[i]) 
                 for i in range(len(predictions)) if predictions[i] > threshold]
    
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
    if result_df is not None and len(result_df) > 0:
        report_lines.append("TOP 10 CLUSTERS BY PREDICTED CONSUMPTION")
        report_lines.append("-" * 80)
        top_clusters = result_df.nlargest(config.TOP_CLUSTERS_LIMIT, 'Predictions')
        for idx, (cluster, row) in enumerate(top_clusters.iterrows(), 1):
            report_lines.append(f"{idx}. {cluster}: {row['Predictions']:.2f} L")
        report_lines.append("")
    
    # Model Performance
    if nse_value is not None:
        report_lines.append("MODEL PERFORMANCE")
        report_lines.append("-" * 80)
        report_lines.append(f"Nash-Sutcliffe Efficiency (NSE): {nse_value:.4f}")
        if nse_value > 0.75:
            report_lines.append("Model Quality: Excellent")
        elif nse_value > 0.65:
            report_lines.append("Model Quality: Good")
        elif nse_value > 0.50:
            report_lines.append("Model Quality: Acceptable")
        else:
            report_lines.append("Model Quality: Needs Improvement")
        report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info(f"[EXPORT SUMMARY] Generated summary report with {len(predictions)} predictions")
    
    return report_text, timestamp


def create_chart_object(chart_type, data_dict):
    """
    Create a Pygal chart object based on chart type and data.
    
    Args:
        chart_type (str): Type of chart ('cluster', 'site', 'distribution', 'comparison', 'timeseries')
        data_dict (dict): Dictionary containing data needed for the chart:
            For 'cluster': {'result_df': DataFrame}
            For 'site': {'predictions': list, 'sites': list}
            For 'distribution': {'predictions': list}
            For 'comparison': {'predictions': list, 'clusters': list, 'sites': list}
            For 'timeseries': {'time_series': pd.Series, 'freq': str, 'fuel_col': str}
    
    Returns:
        tuple: (pygal_chart_object, filename_base)
        
    Raises:
        ValueError: If chart_type is invalid or required data is missing
    """
    
    if chart_type == 'cluster':
        result_df = data_dict.get('result_df')
        if result_df is None or len(result_df) == 0:
            raise ValueError("No cluster data available for chart")
        
        graph = pygal.Bar()
        graph.title = 'Prediction of Fuel Consumption Per Cluster'
        graph.x_title = 'Clusters'
        graph.y_title = 'Predicted Fuel Consumption (L)'
        graph.x_label_rotation = 60
        
        for k in result_df.index:
            graph.add(k, result_df['Predictions'][k])
        
        filename_base = f'cluster_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
    elif chart_type == 'site':
        predictions = data_dict.get('predictions', [])
        sites = data_dict.get('sites', [])
        
        if len(predictions) == 0 or len(sites) == 0:
            raise ValueError("No site data available for chart")
        
        df = pd.DataFrame({'sites': sites, 'predictions': predictions})
        site_result = df.groupby('sites')['predictions'].sum().sort_values(ascending=False)
        top_20 = site_result.head(config.TOP_SITES_LIMIT)
        
        graph = pygal.HorizontalBar()
        graph.title = 'Top 20 Sites by Predicted Fuel Consumption'
        graph.x_title = 'Predicted Fuel Consumption (L)'
        
        for site, value in top_20.items():
            graph.add(str(site), float(value))
        
        filename_base = f'site_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
    elif chart_type == 'distribution':
        predictions = data_dict.get('predictions', [])
        
        if len(predictions) == 0:
            raise ValueError("No prediction data available for chart")
        
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
        
    elif chart_type == 'comparison':
        predictions = data_dict.get('predictions', [])
        clusters = data_dict.get('clusters', [])
        sites = data_dict.get('sites', [])
        
        if len(predictions) == 0 or len(clusters) == 0 or len(sites) == 0:
            raise ValueError("No comparison data available for chart")
        
        df = pd.DataFrame({'clusters': clusters, 'sites': sites, 'predictions': predictions})
        cluster_totals = df.groupby('clusters')['predictions'].sum().sort_values(ascending=False)
        
        # Use same 10 colors as comparison view
        site_colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', 
                       '#00f2fe', '#43e97b', '#38f9d7', '#fa709a', '#fee140']
        custom_style = Style(colors=site_colors + ['#cccccc'], 
                           value_font_size=10, value_label_font_size=10)
        graph = pygal.StackedBar(style=custom_style)
        graph.title = 'Fuel Consumption: Cluster Breakdown by Top Sites'
        graph.x_title = 'Clusters'
        graph.y_title = 'Predicted Fuel Consumption (L)'
        graph.x_label_rotation = 60
        graph.show_legend = True
        
        # Build cluster-site breakdown
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
        
        # Sort sites by total consumption
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
        
        # Add series for each top site
        for site, _ in ranked_sites:
            graph.add(site, site_series[site])
        
        # Add "Other Sites" series
        graph.add('Other Sites', other_sites_series)
        
        filename_base = f'comparison_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
    elif chart_type == 'timeseries':
        time_series = data_dict.get('time_series')
        freq = data_dict.get('freq', 'D')
        fuel_col = data_dict.get('fuel_col', 'Fuel')
        
        if time_series is None or len(time_series) == 0:
            raise ValueError("No time series data available for chart")
        
        # Determine title suffix based on frequency
        title_suffix_map = {'M': '(Monthly)', 'W': '(Weekly)', 'D': '(Daily)'}
        title_suffix = title_suffix_map.get(freq, '')
        
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
        
        # Add moving average trend if enough data points
        if len(time_series) >= 3:
            window = min(3, len(time_series) // 3)
            moving_avg = time_series.rolling(window=window, center=True).mean()
            graph.add('Trend (Moving Avg)', 
                     [float(v) if pd.notna(v) else None for v in moving_avg.values])
        
        filename_base = f'timeseries_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        
    else:
        raise ValueError(f"Invalid chart type: {chart_type}")
    
    logger.info(f"[CHART OBJECT] Created {chart_type} chart")
    return graph, filename_base


def export_chart_svg(chart_type, data_dict):
    """
    Export chart as SVG format.
    
    Args:
        chart_type (str): Type of chart to export
        data_dict (dict): Data required for chart generation
        
    Returns:
        tuple: (svg_bytes, filename)
            svg_bytes: Binary SVG data
            filename: Complete filename with timestamp and .svg extension
            
    Raises:
        ValueError: If chart generation fails
    """
    graph, filename_base = create_chart_object(chart_type, data_dict)
    svg_data = graph.render()
    filename = f"{filename_base}.svg"
    
    logger.info(f"[EXPORT SVG] {chart_type} chart exported as SVG")
    return svg_data, filename


def export_chart_png(chart_type, data_dict, dpi=None):
    """
    Export chart as PNG format (requires cairosvg).
    
    Args:
        chart_type (str): Type of chart to export
        data_dict (dict): Data required for chart generation
        dpi (int, optional): DPI for PNG rendering (default from config)
        
    Returns:
        tuple: (png_bytes, filename)
            png_bytes: Binary PNG data
            filename: Complete filename with timestamp and .png extension
            
    Raises:
        ImportError: If cairosvg is not installed
        ValueError: If chart generation fails
    """
    if not HAS_CAIROSVG:
        raise ImportError("PNG export requires cairosvg. Install with: pip install cairosvg")
    
    if dpi is None:
        dpi = config.PNG_EXPORT_DPI
    
    graph, filename_base = create_chart_object(chart_type, data_dict)
    svg_data = graph.render()
    png_data = cairosvg.svg2png(bytestring=svg_data, dpi=dpi)
    filename = f"{filename_base}.png"
    
    logger.info(f"[EXPORT PNG] {chart_type} chart exported as PNG (DPI={dpi})")
    return png_data, filename
