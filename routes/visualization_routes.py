"""
Visualization Routes Blueprint

This module contains routes for chart generation and data visualization:
- clustergraph: Cluster-level fuel consumption bar chart with anomaly detection
- sitegraph: Top 20 sites by consumption (horizontal bar chart)
- distributiongraph: Histogram of prediction distribution with statistics
- comparisonview: Stacked bar chart showing cluster breakdown by top sites
- timeseriesgraph: Time-series trend analysis with moving average

All visualization routes:
1. Check for cached predictions or generate new ones from form data
2. Create Pygal charts with appropriate styling
3. Calculate relevant statistics and metrics
4. Render graphdisplay.html template with chart and stats
"""

from flask import Blueprint, render_template, request, session
import pandas as pd
import numpy as np
import pygal
from pygal.style import Style
import logging

# Import configuration
import config

# Import utilities
from utils.data_utils import (
    get_cached_data,
    get_data_file,
    get_prediction_cache
)

logger = logging.getLogger(__name__)

# Create Blueprint
viz_bp = Blueprint('visualization', __name__)

# These will be injected when blueprint is registered
clf = None
UPLOAD_FOLDER = None
results_function = None  # Will be set to main_routes.results


def init_blueprint(model, upload_folder, results_func):
    """Initialize blueprint with shared resources from main app"""
    global clf, UPLOAD_FOLDER, results_function
    clf = model
    UPLOAD_FOLDER = upload_folder
    results_function = results_func


@viz_bp.route('/clustergraph', methods=['POST', 'GET'])
def clustergraph():
    """Generate cluster-level fuel consumption bar chart with anomaly detection"""
    try:
        # Check if we need to generate predictions (POST with form data) or use cached (GET)
        if request.method == 'POST' and request.form:
            result = results_function()
        else:
            # Use cached result
            result = get_prediction_cache().get('result_df')
            if result is None:
                logger.warning("No cached predictions available")
                return render_template('error.html', 
                                     error="No predictions available. Please submit the form first."), 400
        
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


@viz_bp.route('/sitegraph')
def sitegraph():
    """Generate site-level predictions chart (top 20 sites)"""
    try:
        # Get prediction data from server-side cache
        cache = get_prediction_cache()
        if not cache.get('predictions'):
            return render_template('error.html', 
                                 error="No prediction data available. Please run predictions first."), 400
        
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


@viz_bp.route('/distributiongraph')
def distributiongraph():
    """Generate distribution histogram with anomaly visualization"""
    try:
        # Get prediction data from server-side cache
        cache = get_prediction_cache()
        pred_data = cache if cache.get('predictions') else None
        
        if pred_data is None:
            return render_template('error.html', 
                                 error="No prediction data available. Please run predictions first."), 400
        
        predictions = pred_data['predictions']
        
        # Create histogram/distribution chart
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


@viz_bp.route('/comparisonview')
def comparisonview():
    """Show comparison of cluster vs site-level predictions with stacked bar chart"""
    try:
        # Get prediction data from server-side cache
        cache = get_prediction_cache()
        if not cache.get('predictions'):
            return render_template('error.html', 
                                 error="No prediction data available. Please run predictions first."), 400
        
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


@viz_bp.route('/timeseriesgraph')
def timeseriesgraph():
    """Generate time-series trend chart with moving average if date columns are available"""
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
