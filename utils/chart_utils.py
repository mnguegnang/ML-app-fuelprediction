"""
Chart generation utilities
Handles all chart creation using Pygal for fuel consumption visualizations
"""

import logging
import numpy as np
import pandas as pd
import pygal
from pygal.style import Style

logger = logging.getLogger(__name__)


def create_cluster_chart(result_df):
    """
    Create cluster-level bar chart.
    
    Args:
        result_df: DataFrame with cluster predictions (grouped by cluster)
    
    Returns:
        str: Chart data URI
    """
    graph = pygal.Bar()
    graph.title = 'Prediction of Fuel Consumption Per Cluster'
    graph.x_title = 'Clusters'
    graph.y_title = 'Predicted Fuel Consumption (L)'
    
    for k in result_df.index:
        graph.add(k, result_df['Predictions'][k])
    
    return graph.render_data_uri()


def create_site_chart(df, top_n=20):
    """
    Create site-level horizontal bar chart showing top consuming sites.
    
    Args:
        df: DataFrame with 'sites' and 'predictions' columns
        top_n: Number of top sites to display (default: 20)
    
    Returns:
        tuple: (graph_data_uri, site_result, top_sites)
    """
    # Group by site
    site_result = df.groupby('sites')['predictions'].sum().sort_values(ascending=False)
    
    # Limit to top N sites for readability
    top_sites = site_result.head(top_n)
    
    # Create horizontal bar chart for better label visibility
    graph = pygal.HorizontalBar()
    graph.title = f'Top {top_n} Sites by Predicted Fuel Consumption'
    graph.x_title = 'Predicted Fuel Consumption (L)'
    
    for site, value in top_sites.items():
        graph.add(str(site), float(value))
    
    graph_data = graph.render_data_uri()
    
    return graph_data, site_result, top_sites


def create_distribution_chart(predictions, bins=20):
    """
    Create distribution histogram chart.
    
    Args:
        predictions: Array or list of prediction values
        bins: Number of histogram bins (default: 20)
    
    Returns:
        tuple: (graph_data_uri, hist, bin_edges)
    """
    # Create histogram
    hist, bin_edges = np.histogram(predictions, bins=bins)
    
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
    
    return graph_data, hist, bin_edges


def create_comparison_chart(df, show_all_sites=True):
    """
    Create stacked bar chart showing cluster breakdown by top sites.
    
    Args:
        df: DataFrame with 'clusters', 'sites', and 'predictions' columns
        show_all_sites: Whether to show all unique top sites (default: True)
    
    Returns:
        tuple: (graph_data_uri, cluster_top_sites, ranked_sites)
    """
    # Group by cluster
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
    
    # Create stacked bar chart
    graph = pygal.StackedBar(style=custom_style)
    graph.title = 'Fuel Consumption: Cluster Breakdown by Top Sites'
    graph.x_title = 'Clusters'
    graph.y_title = 'Predicted Fuel Consumption (L)'
    graph.x_label_rotation = 60  # Rotate cluster names for readability
    graph.show_legend = False  # Hide default legend
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
                    # Minimal tooltip: Only label and percentage
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
    
    return graph_data, cluster_top_sites, ranked_sites


def create_timeseries_chart(dataframe, date_col, fuel_col):
    """
    Create time-series line chart with trend.
    
    Args:
        dataframe: DataFrame with date and fuel consumption columns
        date_col: Name of date column
        fuel_col: Name of fuel consumption column
    
    Returns:
        tuple: (graph_data_uri, time_series, trend_stats)
    """
    # Make a copy to avoid modifying original
    df = dataframe.copy()
    
    # Ensure date column is datetime
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col)
    
    if df.empty:
        logger.error("No valid dates found in time series data")
        return None, None, None
    
    # Determine grouping frequency based on data span
    date_range = (df[date_col].max() - df[date_col].min()).days
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
    df['period'] = df[date_col].dt.to_period(freq)
    time_series = df.groupby('period')[fuel_col].sum()
    
    # Create line chart
    graph = pygal.Line()
    graph.title = f'Fuel Consumption Trend Over Time {title_suffix}'
    graph.x_title = 'Time Period'
    graph.y_title = 'Fuel Consumption (L)'
    graph.show_legend = False
    
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
    moving_avg_values = None
    if len(time_series) >= 3:
        window = min(3, len(time_series) // 3)
        moving_avg = time_series.rolling(window=window, center=True).mean()
        moving_avg_values = [float(v) if pd.notna(v) else None for v in moving_avg.values]
        graph.add('Trend (Moving Avg)', moving_avg_values)
    
    graph_data = graph.render_data_uri()
    
    # Calculate trend statistics
    values = time_series.values
    trend_direction = "📈 Increasing" if values[-1] > values[0] else "📉 Decreasing"
    change_pct = ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
    
    trend_stats = {
        'trend_direction': trend_direction,
        'change_percent': round(change_pct, 2),
        'start_value': float(values[0]),
        'end_value': float(values[-1]),
        'mean': round(float(values.mean()), 2),
        'max': round(float(values.max()), 2),
        'min': round(float(values.min()), 2),
        'periods': len(time_series),
        'frequency': freq,
        'title_suffix': title_suffix
    }
    
    return graph_data, time_series, trend_stats


def create_chart_for_export(chart_type, data, config=None):
    """
    Create a chart for export (SVG/PNG) without rendering to data URI.
    
    Args:
        chart_type: Type of chart ('cluster', 'site', 'distribution', 'comparison', 'timeseries')
        data: Data required for the chart (format depends on chart_type)
        config: Optional configuration dict with chart parameters
    
    Returns:
        pygal.Graph: Pygal graph object
    """
    config = config or {}
    
    if chart_type == 'cluster':
        result_df = data
        graph = pygal.Bar()
        graph.title = 'Prediction of Fuel Consumption Per Cluster'
        graph.x_title = 'Clusters'
        graph.y_title = 'Predicted Fuel Consumption (L)'
        graph.x_label_rotation = 60
        for k in result_df.index:
            graph.add(k, result_df['Predictions'][k])
        return graph
    
    elif chart_type == 'site':
        df = data
        top_n = config.get('top_n', 20)
        site_result = df.groupby('sites')['predictions'].sum().sort_values(ascending=False)
        top_sites = site_result.head(top_n)
        
        graph = pygal.HorizontalBar()
        graph.title = f'Top {top_n} Sites by Predicted Fuel Consumption'
        graph.x_title = 'Predicted Fuel Consumption (L)'
        
        for site, value in top_sites.items():
            graph.add(str(site), float(value))
        return graph
    
    elif chart_type == 'distribution':
        predictions = data
        bins = config.get('bins', 20)
        hist, bin_edges = np.histogram(predictions, bins=bins)
        
        graph = pygal.Bar()
        graph.title = 'Distribution of Fuel Consumption Predictions'
        graph.x_title = 'Fuel Consumption Range (L)'
        graph.y_title = 'Number of Sites'
        graph.x_label_rotation = 60
        
        labels = []
        for i in range(len(bin_edges)-1):
            labels.append(f'[{int(bin_edges[i])}, {int(bin_edges[i+1])}]')
        
        graph.x_labels = labels
        graph.add('Frequency', list(hist))
        return graph
    
    elif chart_type == 'comparison':
        df = data
        graph_data, _, _ = create_comparison_chart(df)
        # Note: This returns data URI, not graph object
        # For export, we need to reconstruct the graph
        # This is handled in the export routes
        return None
    
    elif chart_type == 'timeseries':
        dataframe, date_col, fuel_col = data
        graph_data, _, _ = create_timeseries_chart(dataframe, date_col, fuel_col)
        # Note: This returns data URI, not graph object
        # For export, we need to reconstruct the graph
        # This is handled in the export routes
        return None
    
    else:
        logger.error(f"Unknown chart type: {chart_type}")
        return None
