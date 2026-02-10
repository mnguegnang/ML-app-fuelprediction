#!/usr/bin/env python3
"""
Add all export routes and fixes to columns_app.py
"""

# Read the file
with open('columns_app.py', 'r') as f:
    lines = f.readlines()

# Fix 1: Add chart_type='cluster' to clustergraph route (around line 813)
for i, line in enumerate(lines):
    if 'has_site_data=pred_data is not None)' in line and i > 800 and i < 820:
        # Replace with version that includes chart_type
        lines[i] = line.replace(
            'has_site_data=pred_data is not None)',
            'has_site_data=pred_data is not None,\n\t\t                      chart_type=\'cluster\')'
        )
        break

# Find where to insert export routes (before if __name__)
insert_index = None
for i, line in enumerate(lines):
    if line.strip().startswith('if __name__'):
        insert_index = i
        break

if insert_index:
    # Create the export routes code
    export_routes = '''

# ============================================================================
# EXPORT AND DOWNLOAD ROUTES
# ============================================================================

@app.route('/export/predictions/csv')
def export_predictions_csv():
\t"""Export predictions to CSV file"""
\ttry:
\t\tif prediction_cache.get('result_df') is None:
\t\t\tflash(config.ERROR_MESSAGES['prediction_cache_empty'], 'warning')
\t\t\treturn redirect(url_for('index'))
\t\t
\t\t# Create CSV in memory
\t\toutput = StringIO()
\t\tresult_df = prediction_cache['result_df']
\t\tresult_df.to_csv(output)
\t\toutput.seek(0)
\t\t
\t\t# Create response
\t\tresponse = make_response(output.getvalue())
\t\tresponse.headers['Content-Disposition'] = f'attachment; filename=predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
\t\tresponse.headers['Content-Type'] = 'text/csv'
\t\t
\t\tlogger.info("[EXPORT CSV] Predictions exported to CSV")
\t\treturn response
\t\t
\texcept Exception as e:
\t\tlogger.error(f"[EXPORT ERROR] CSV export failed: {str(e)}")
\t\tflash(f'Export failed: {str(e)}', 'danger')
\t\treturn redirect(url_for('index'))


@app.route('/export/predictions/excel')
def export_predictions_excel():
\t"""Export predictions to Excel file"""
\ttry:
\t\tif prediction_cache.get('result_df') is None:
\t\t\tflash(config.ERROR_MESSAGES['prediction_cache_empty'], 'warning')
\t\t\treturn redirect(url_for('index'))
\t\t
\t\t# Create Excel file in memory
\t\toutput = BytesIO()
\t\tresult_df = prediction_cache['result_df']
\t\t
\t\twith pd.ExcelWriter(output, engine='openpyxl') as writer:
\t\t\tresult_df.to_excel(writer, sheet_name='Predictions', index=True)
\t\t
\t\toutput.seek(0)
\t\t
\t\t# Create response
\t\tresponse = make_response(output.getvalue())
\t\tresponse.headers['Content-Disposition'] = f'attachment; filename=predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
\t\tresponse.headers['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
\t\t
\t\tlogger.info("[EXPORT EXCEL] Predictions exported to Excel")
\t\treturn response
\t\t
\texcept Exception as e:
\t\tlogger.error(f"[EXPORT ERROR] Excel export failed: {str(e)}")
\t\tflash(f'Export failed: {str(e)}', 'danger')
\t\treturn redirect(url_for('index'))


@app.route('/export/summary')
def export_summary_report():
\t"""Export comprehensive summary report as text file"""
\ttry:
\t\tif prediction_cache.get('predictions') is None:
\t\t\tflash(config.ERROR_MESSAGES['prediction_cache_empty'], 'warning')
\t\t\treturn redirect(url_for('index'))
\t\t
\t\tpredictions = prediction_cache.get('predictions', [])
\t\tclusters = prediction_cache.get('clusters', [])
\t\tsites = prediction_cache.get('sites', [])
\t\tresult_df = prediction_cache.get('result_df')
\t\t
\t\t# Build comprehensive report
\t\treport_lines = []
\t\treport_lines.append("=" * 80)
\t\treport_lines.append("FUEL CONSUMPTION PREDICTION SUMMARY REPORT")
\t\treport_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
\t\treport_lines.append("=" * 80)
\t\treport_lines.append("")
\t\t
\t\t# Overview Statistics
\t\treport_lines.append("OVERVIEW")
\t\treport_lines.append("-" * 80)
\t\treport_lines.append(f"Total Predictions: {len(predictions)}")
\t\treport_lines.append(f"Total Clusters: {len(set(clusters))}")
\t\treport_lines.append(f"Total Sites: {len(set(sites))}")
\t\treport_lines.append(f"Total Fuel Consumption: {sum(predictions):.2f} L")
\t\treport_lines.append(f"Average Consumption: {np.mean(predictions):.2f} L")
\t\treport_lines.append(f"Min Consumption: {min(predictions):.2f} L")
\t\treport_lines.append(f"Max Consumption: {max(predictions):.2f} L")
\t\treport_lines.append(f"Std Deviation: {np.std(predictions):.2f} L")
\t\treport_lines.append("")
\t\t
\t\t# Quartile Analysis
\t\treport_lines.append("QUARTILE ANALYSIS")
\t\treport_lines.append("-" * 80)
\t\tq1, q2, q3 = np.percentile(predictions, [25, 50, 75])
\t\treport_lines.append(f"Q1 (25th percentile): {q1:.2f} L")
\t\treport_lines.append(f"Q2 (50th percentile/Median): {q2:.2f} L")
\t\treport_lines.append(f"Q3 (75th percentile): {q3:.2f} L")
\t\treport_lines.append(f"Interquartile Range (IQR): {q3 - q1:.2f} L")
\t\treport_lines.append("")
\t\t
\t\t# Anomaly Detection
\t\tmean_pred = np.mean(predictions)
\t\tstd_pred = np.std(predictions)
\t\tthreshold = mean_pred + config.ANOMALY_STD_MULTIPLIER * std_pred
\t\tanomalies = [(clusters[i], sites[i], predictions[i]) for i in range(len(predictions)) if predictions[i] > threshold]
\t\t
\t\treport_lines.append("ANOMALY DETECTION")
\t\treport_lines.append("-" * 80)
\t\treport_lines.append(f"Threshold (mean + {config.ANOMALY_STD_MULTIPLIER}σ): {threshold:.2f} L")
\t\treport_lines.append(f"Number of Anomalies: {len(anomalies)}")
\t\tif anomalies:
\t\t\treport_lines.append("\\nAnomalous Sites (High Consumption):")
\t\t\tfor cluster, site, value in sorted(anomalies, key=lambda x: x[2], reverse=True):
\t\t\t\treport_lines.append(f"  - Cluster: {cluster}, Site: {site}, Consumption: {value:.2f} L")
\t\treport_lines.append("")
\t\t
\t\t# Top 10 Clusters
\t\tif result_df is not None:
\t\t\treport_lines.append("TOP 10 CLUSTERS BY PREDICTED CONSUMPTION")
\t\t\treport_lines.append("-" * 80)
\t\t\ttop_clusters = result_df.nlargest(config.TOP_CLUSTERS_LIMIT, 'Predictions')
\t\t\tfor idx, (cluster, row) in enumerate(top_clusters.iterrows(), 1):
\t\t\t\treport_lines.append(f"{idx}. {cluster}: {row['Predictions']:.2f} L")
\t\t\treport_lines.append("")
\t\t
\t\t# Model Performance (if available)
\t\tnse = prediction_cache.get('nse_value')
\t\tif nse is not None:
\t\t\treport_lines.append("MODEL PERFORMANCE")
\t\t\treport_lines.append("-" * 80)
\t\t\treport_lines.append(f"Nash-Sutcliffe Efficiency (NSE): {nse:.4f}")
\t\t\tif nse > 0.75:
\t\t\t\treport_lines.append("Model Quality: Excellent")
\t\t\telif nse > 0.65:
\t\t\t\treport_lines.append("Model Quality: Good")
\t\t\telif nse > 0.50:
\t\t\t\treport_lines.append("Model Quality: Acceptable")
\t\t\telse:
\t\t\t\treport_lines.append("Model Quality: Needs Improvement")
\t\t\treport_lines.append("")
\t\t
\t\treport_lines.append("=" * 80)
\t\treport_lines.append("END OF REPORT")
\t\treport_lines.append("=" * 80)
\t\t
\t\t# Create text file
\t\treport_text = "\\n".join(report_lines)
\t\tresponse = make_response(report_text)
\t\tresponse.headers['Content-Disposition'] = f'attachment; filename=summary_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
\t\tresponse.headers['Content-Type'] = 'text/plain'
\t\t
\t\tlogger.info("[EXPORT SUMMARY] Summary report exported")
\t\treturn response
\t\t
\texcept Exception as e:
\t\tlogger.error(f"[EXPORT ERROR] Summary report export failed: {str(e)}")
\t\tflash(f'Export failed: {str(e)}', 'danger')
\t\treturn redirect(url_for('index'))


@app.route('/export/chart/<chart_type>')
def export_chart(chart_type):
\t"""Export chart as SVG file"""
\ttry:
\t\tif prediction_cache.get('predictions') is None:
\t\t\tflash(config.ERROR_MESSAGES['prediction_cache_empty'], 'warning')
\t\t\treturn redirect(url_for('index'))
\t\t
\t\tresult_df = prediction_cache.get('result_df')
\t\t
\t\t# Generate chart based on type
\t\tif chart_type == 'cluster':
\t\t\tif result_df is None:
\t\t\t\tflash('No cluster data available', 'warning')
\t\t\t\treturn redirect(url_for('index'))
\t\t\t
\t\t\tgraph = pygal.Bar()
\t\t\tgraph.title = 'Prediction of Fuel Consumption Per Cluster'
\t\t\tgraph.x_title = 'Clusters'
\t\t\tgraph.y_title = 'Predicted Fuel Consumption (L)'
\t\t\tgraph.x_label_rotation = 60
\t\t\t
\t\t\tfor k in result_df.index:
\t\t\t\tgraph.add(k, result_df['Predictions'][k])
\t\t\t
\t\t\tfilename_base = f'cluster_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
\t\t\t
\t\telif chart_type == 'site':
\t\t\tpredictions = prediction_cache.get('predictions', [])
\t\t\tsites = prediction_cache.get('sites', [])
\t\t\t
\t\t\tif len(predictions) > 0 and len(sites) > 0:
\t\t\t\tdf = pd.DataFrame({'sites': sites, 'predictions': predictions})
\t\t\t\tsite_result = df.groupby('sites')['predictions'].sum().sort_values(ascending=False)
\t\t\t\ttop_20 = site_result.head(config.TOP_SITES_LIMIT)
\t\t\t\t
\t\t\t\tgraph = pygal.HorizontalBar()
\t\t\t\tgraph.title = 'Top 20 Sites by Predicted Fuel Consumption'
\t\t\t\tgraph.x_title = 'Predicted Fuel Consumption (L)'
\t\t\t\t
\t\t\t\tfor site, value in top_20.items():
\t\t\t\t\tgraph.add(str(site), float(value))
\t\t\t\t
\t\t\t\tfilename_base = f'site_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
\t\t\telse:
\t\t\t\tflash('No site data available', 'warning')
\t\t\t\treturn redirect(url_for('index'))
\t\t\t
\t\telif chart_type == 'distribution':
\t\t\tpredictions = prediction_cache.get('predictions', [])
\t\t\tif len(predictions) > 0:
\t\t\t\tpred_array = np.array(predictions)
\t\t\t\thist_data, bin_edges = np.histogram(pred_array, bins=config.DISTRIBUTION_BINS)
\t\t\t\t
\t\t\t\tbin_labels = []
\t\t\t\tfor i in range(len(bin_edges)-1):
\t\t\t\t\tbin_labels.append(f'[{int(bin_edges[i])}, {int(bin_edges[i+1])}]')
\t\t\t\t
\t\t\t\tgraph = pygal.Bar()
\t\t\t\tgraph.title = 'Distribution of Fuel Consumption Predictions'
\t\t\t\tgraph.x_title = 'Fuel Consumption Range (L)'
\t\t\t\tgraph.y_title = 'Number of Sites'
\t\t\t\tgraph.x_label_rotation = 60
\t\t\t\tgraph.x_labels = bin_labels
\t\t\t\tgraph.add('Frequency', list(hist_data))
\t\t\t\t
\t\t\t\tfilename_base = f'distribution_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
\t\t\telse:
\t\t\t\tflash('No prediction data available', 'warning')
\t\t\t\treturn redirect(url_for('index'))
\t\t\t
\t\telif chart_type == 'comparison':
\t\t\tpredictions = prediction_cache.get('predictions', [])
\t\t\tclusters = prediction_cache.get('clusters', [])
\t\t\tsites = prediction_cache.get('sites', [])
\t\t\tif len(predictions) > 0 and len(clusters) > 0 and len(sites) > 0:
\t\t\t\tdf = pd.DataFrame({'clusters': clusters, 'sites': sites, 'predictions': predictions})
\t\t\t\tcluster_totals = df.groupby('clusters')['predictions'].sum().sort_values(ascending=False)
\t\t\t\t
\t\t\t\t# Use same 10 colors as comparison view
\t\t\t\tsite_colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', 
\t\t\t\t               '#00f2fe', '#43e97b', '#38f9d7', '#fa709a', '#fee140']
\t\t\t\tfrom pygal.style import Style
\t\t\t\tcustom_style = Style(colors=site_colors + ['#cccccc'], value_font_size=10, value_label_font_size=10)
\t\t\t\tgraph = pygal.StackedBar(style=custom_style)
\t\t\t\tgraph.title = 'Fuel Consumption: Cluster Breakdown by Top Sites'
\t\t\t\tgraph.x_title = 'Clusters'
\t\t\t\tgraph.y_title = 'Predicted Fuel Consumption (L)'
\t\t\t\tgraph.x_label_rotation = 60
\t\t\t\tgraph.show_legend = True
\t\t\t\t
\t\t\t\t# Match the exact logic from comparison view
\t\t\t\tcluster_top_sites = {}
\t\t\t\tall_top_sites = {}
\t\t\t\tfor cluster in cluster_totals.index:
\t\t\t\t\tcluster_sites = df[df['clusters'] == cluster].groupby('sites')['predictions'].sum()
\t\t\t\t\tif not cluster_sites.empty:
\t\t\t\t\t\ttop_site = cluster_sites.idxmax()
\t\t\t\t\t\ttop_value = float(cluster_sites.max())
\t\t\t\t\t\tcluster_total = float(cluster_totals[cluster])
\t\t\t\t\t\tcontribution_pct = round((top_value / cluster_total) * 100, 1)
\t\t\t\t\t\tcluster_top_sites[cluster] = (top_site, top_value, contribution_pct)
\t\t\t\t\t\tif top_site in all_top_sites:
\t\t\t\t\t\t\tall_top_sites[top_site] += top_value
\t\t\t\t\t\telse:
\t\t\t\t\t\t\tall_top_sites[top_site] = top_value
\t\t\t\t
\t\t\t\t# Sort sites by total consumption (ALL unique top sites, not limited)
\t\t\t\tranked_sites = sorted(all_top_sites.items(), key=lambda x: x[1], reverse=True)
\t\t\t\t
\t\t\t\t# Create series for each unique top site
\t\t\t\tsite_series = {site: [] for site, _ in ranked_sites}
\t\t\t\tother_sites_series = []
\t\t\t\t
\t\t\t\tfor cluster in cluster_totals.index:
\t\t\t\t\tif cluster in cluster_top_sites:
\t\t\t\t\t\ttop_site, top_value, contribution_pct = cluster_top_sites[cluster]
\t\t\t\t\t\tcluster_total = float(cluster_totals[cluster])
\t\t\t\t\t\tother_value = cluster_total - top_value
\t\t\t\t\t\t
\t\t\t\t\t\t# Add value for the top site in this cluster
\t\t\t\t\t\tfor site in site_series:
\t\t\t\t\t\t\tif site == top_site:
\t\t\t\t\t\t\t\tsite_series[site].append(top_value)
\t\t\t\t\t\t\telse:
\t\t\t\t\t\t\t\tsite_series[site].append(0)
\t\t\t\t\t\t
\t\t\t\t\t\t# Add "other sites" value
\t\t\t\t\t\tother_sites_series.append(other_value)
\t\t\t\t\telse:
\t\t\t\t\t\t# No data for this cluster
\t\t\t\t\t\tfor site in site_series:
\t\t\t\t\t\t\tsite_series[site].append(0)
\t\t\t\t\t\tother_sites_series.append(float(cluster_totals[cluster]))
\t\t\t\t
\t\t\t\t# Set cluster names as x-labels
\t\t\t\tgraph.x_labels = list(cluster_totals.index)
\t\t\t\t
\t\t\t\t# Add series for each top site (creates different colors - show ALL sites)
\t\t\t\tfor site, _ in ranked_sites:
\t\t\t\t\tgraph.add(site, site_series[site])
\t\t\t\t
\t\t\t\t# Add "Other Sites" series
\t\t\t\tgraph.add('Other Sites', other_sites_series)
\t\t\t\t
\t\t\t\tfilename_base = f'comparison_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
\t\t\telse:
\t\t\t\tflash('No comparison data available', 'warning')
\t\t\t\treturn redirect(url_for('index'))
\t\t\t
\t\telif chart_type == 'timeseries':
\t\t\texcel_file = get_data_file()
\t\t\tif excel_file is None:
\t\t\t\tflash('No data file available for time series', 'warning')
\t\t\t\treturn redirect(url_for('index'))
\t\t\t
\t\t\traw_data, _, _ = get_cached_data(excel_file)
\t\t\tif raw_data is not None and len(raw_data) > 0:
\t\t\t\traw_data['Date'] = pd.to_datetime(raw_data['Date'], errors='coerce')
\t\t\t\traw_data = raw_data.dropna(subset=['Date'])
\t\t\t\traw_data = raw_data.sort_values('Date')
\t\t\t\ttime_series = raw_data.groupby('Date')['Fuel_Consumed'].sum()
\t\t\t\tgraph = pygal.Line()
\t\t\t\tgraph.title = 'Fuel Consumption Trends Over Time'
\t\t\t\tgraph.x_title = 'Date'
\t\t\t\tgraph.y_title = 'Fuel Consumption (L)'
\t\t\t\tgraph.x_label_rotation = 45
\t\t\t\tgraph.show_legend = True
\t\t\t\tgraph.x_labels = [d.strftime('%Y-%m-%d') for d in time_series.index]
\t\t\t\tgraph.add('Actual Consumption', list(time_series.values))
\t\t\t\tif len(time_series) >= 7:
\t\t\t\t\twindow = 7
\t\t\t\t\tmoving_avg = time_series.rolling(window=window, center=True).mean()
\t\t\t\t\tgraph.add(f'{window}-Day Moving Average', list(moving_avg.values))
\t\t\t\tfilename_base = f'timeseries_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
\t\t\telse:
\t\t\t\tflash('No time series data available', 'warning')
\t\t\t\treturn redirect(url_for('index'))
\t\telse:
\t\t\tflash('Invalid chart type', 'danger')
\t\t\treturn redirect(url_for('index'))
\t\t
\t\t# Render SVG
\t\tsvg_data = graph.render()
\t\tresponse = make_response(svg_data)
\t\tresponse.headers['Content-Disposition'] = f'attachment; filename={filename_base}.svg'
\t\tresponse.headers['Content-Type'] = 'image/svg+xml'
\t\t
\t\tlogger.info(f"[EXPORT CHART SVG] {chart_type} chart exported")
\t\treturn response
\t\t
\texcept Exception as e:
\t\tlogger.error(f"[EXPORT ERROR] SVG chart export failed: {str(e)}")
\t\tflash(f'Export failed: {str(e)}', 'danger')
\t\treturn redirect(url_for('index'))


@app.route('/export/chart/<chart_type>/png')
def export_chart_png(chart_type):
\t"""Export chart as PNG file"""
\ttry:
\t\tif not HAS_CAIROSVG:
\t\t\tflash('PNG export not available. Install cairosvg: pip install cairosvg', 'warning')
\t\t\treturn redirect(url_for('index'))
\t\t
\t\tif prediction_cache.get('predictions') is None:
\t\t\tflash(config.ERROR_MESSAGES['prediction_cache_empty'], 'warning')
\t\t\treturn redirect(url_for('index'))
\t\t
\t\tresult_df = prediction_cache.get('result_df')
\t\t
\t\t# Generate chart based on type (same logic as SVG export)
\t\tif chart_type == 'cluster':
\t\t\tif result_df is None:
\t\t\t\tflash('No cluster data available', 'warning')
\t\t\t\treturn redirect(url_for('index'))
\t\t\tgraph = pygal.Bar()
\t\t\tgraph.title = 'Prediction of Fuel Consumption Per Cluster'
\t\t\tgraph.x_title = 'Clusters'
\t\t\tgraph.y_title = 'Predicted Fuel Consumption (L)'
\t\t\tgraph.x_label_rotation = 60
\t\t\tfor k in result_df.index:
\t\t\t\tgraph.add(k, result_df['Predictions'][k])
\t\t\tfilename_base = f'cluster_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
\t\t\t
\t\telif chart_type == 'site':
\t\t\tpredictions = prediction_cache.get('predictions', [])
\t\t\tsites = prediction_cache.get('sites', [])
\t\t\tif len(predictions) > 0 and len(sites) > 0:
\t\t\t\tdf = pd.DataFrame({'sites': sites, 'predictions': predictions})
\t\t\t\tsite_result = df.groupby('sites')['predictions'].sum().sort_values(ascending=False)
\t\t\t\ttop_20 = site_result.head(config.TOP_SITES_LIMIT)
\t\t\t\t
\t\t\t\tgraph = pygal.HorizontalBar()
\t\t\t\tgraph.title = 'Top 20 Sites by Predicted Fuel Consumption'
\t\t\t\tgraph.x_title = 'Predicted Fuel Consumption (L)'
\t\t\t\t
\t\t\t\tfor site, value in top_20.items():
\t\t\t\t\tgraph.add(str(site), float(value))
\t\t\t\t
\t\t\t\tfilename_base = f'site_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
\t\t\telse:
\t\t\t\tflash('No site data available', 'warning')
\t\t\t\treturn redirect(url_for('index'))
\t\t\t\t
\t\telif chart_type == 'distribution':
\t\t\tpredictions = prediction_cache.get('predictions', [])
\t\t\tif len(predictions) > 0:
\t\t\t\tpred_array = np.array(predictions)
\t\t\t\thist_data, bin_edges = np.histogram(pred_array, bins=config.DISTRIBUTION_BINS)
\t\t\t\t
\t\t\t\tbin_labels = []
\t\t\t\tfor i in range(len(bin_edges)-1):
\t\t\t\t\tbin_labels.append(f'[{int(bin_edges[i])}, {int(bin_edges[i+1])}]')
\t\t\t\t
\t\t\t\tgraph = pygal.Bar()
\t\t\t\tgraph.title = 'Distribution of Fuel Consumption Predictions'
\t\t\t\tgraph.x_title = 'Fuel Consumption Range (L)'
\t\t\t\tgraph.y_title = 'Number of Sites'
\t\t\t\tgraph.x_label_rotation = 60
\t\t\t\tgraph.x_labels = bin_labels
\t\t\t\tgraph.add('Frequency', list(hist_data))
\t\t\t\t
\t\t\t\tfilename_base = f'distribution_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
\t\t\telse:
\t\t\t\tflash('No prediction data available', 'warning')
\t\t\t\treturn redirect(url_for('index'))
\t\t\t
\t\telif chart_type == 'comparison':
\t\t\tpredictions = prediction_cache.get('predictions', [])
\t\t\tclusters = prediction_cache.get('clusters', [])
\t\t\tsites = prediction_cache.get('sites', [])
\t\t\tif len(predictions) > 0 and len(clusters) > 0 and len(sites) > 0:
\t\t\t\tdf = pd.DataFrame({'clusters': clusters, 'sites': sites, 'predictions': predictions})
\t\t\t\tcluster_totals = df.groupby('clusters')['predictions'].sum().sort_values(ascending=False)
\t\t\t\t
\t\t\t\t# Use same 10 colors as comparison view
\t\t\t\tsite_colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', 
\t\t\t\t               '#00f2fe', '#43e97b', '#38f9d7', '#fa709a', '#fee140']
\t\t\t\tfrom pygal.style import Style
\t\t\t\tcustom_style = Style(colors=site_colors + ['#cccccc'], value_font_size=10, value_label_font_size=10)
\t\t\t\tgraph = pygal.StackedBar(style=custom_style)
\t\t\t\tgraph.title = 'Fuel Consumption: Cluster Breakdown by Top Sites'
\t\t\t\tgraph.x_title = 'Clusters'
\t\t\t\tgraph.y_title = 'Predicted Fuel Consumption (L)'
\t\t\t\tgraph.x_label_rotation = 60
\t\t\t\tgraph.show_legend = True
\t\t\t\t
\t\t\t\t# Match the exact logic from comparison view
\t\t\t\tcluster_top_sites = {}
\t\t\t\tall_top_sites = {}
\t\t\t\tfor cluster in cluster_totals.index:
\t\t\t\t\tcluster_sites = df[df['clusters'] == cluster].groupby('sites')['predictions'].sum()
\t\t\t\t\tif not cluster_sites.empty:
\t\t\t\t\t\ttop_site = cluster_sites.idxmax()
\t\t\t\t\t\ttop_value = float(cluster_sites.max())
\t\t\t\t\t\tcluster_total = float(cluster_totals[cluster])
\t\t\t\t\t\tcontribution_pct = round((top_value / cluster_total) * 100, 1)
\t\t\t\t\t\tcluster_top_sites[cluster] = (top_site, top_value, contribution_pct)
\t\t\t\t\t\tif top_site in all_top_sites:
\t\t\t\t\t\t\tall_top_sites[top_site] += top_value
\t\t\t\t\t\telse:
\t\t\t\t\t\t\tall_top_sites[top_site] = top_value
\t\t\t\t
\t\t\t\t# Sort sites by total consumption (ALL unique top sites, not limited)
\t\t\t\tranked_sites = sorted(all_top_sites.items(), key=lambda x: x[1], reverse=True)
\t\t\t\t
\t\t\t\t# Create series for each unique top site
\t\t\t\tsite_series = {site: [] for site, _ in ranked_sites}
\t\t\t\tother_sites_series = []
\t\t\t\t
\t\t\t\tfor cluster in cluster_totals.index:
\t\t\t\t\tif cluster in cluster_top_sites:
\t\t\t\t\t\ttop_site, top_value, contribution_pct = cluster_top_sites[cluster]
\t\t\t\t\t\tcluster_total = float(cluster_totals[cluster])
\t\t\t\t\t\tother_value = cluster_total - top_value
\t\t\t\t\t\t
\t\t\t\t\t\t# Add value for the top site in this cluster
\t\t\t\t\t\tfor site in site_series:
\t\t\t\t\t\t\tif site == top_site:
\t\t\t\t\t\t\t\tsite_series[site].append(top_value)
\t\t\t\t\t\t\telse:
\t\t\t\t\t\t\t\tsite_series[site].append(0)
\t\t\t\t\t\t
\t\t\t\t\t\t# Add "other sites" value
\t\t\t\t\t\tother_sites_series.append(other_value)
\t\t\t\t\telse:
\t\t\t\t\t\t# No data for this cluster
\t\t\t\t\t\tfor site in site_series:
\t\t\t\t\t\t\tsite_series[site].append(0)
\t\t\t\t\t\tother_sites_series.append(float(cluster_totals[cluster]))
\t\t\t\t
\t\t\t\t# Set cluster names as x-labels
\t\t\t\tgraph.x_labels = list(cluster_totals.index)
\t\t\t\t
\t\t\t\t# Add series for each top site (creates different colors - show ALL sites)
\t\t\t\tfor site, _ in ranked_sites:
\t\t\t\t\tgraph.add(site, site_series[site])
\t\t\t\t
\t\t\t\t# Add "Other Sites" series
\t\t\t\tgraph.add('Other Sites', other_sites_series)
\t\t\t\t
\t\t\t\tfilename_base = f'comparison_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
\t\t\telse:
\t\t\t\tflash('No comparison data available', 'warning')
\t\t\t\treturn redirect(url_for('index'))
\t\t\t
\t\telif chart_type == 'timeseries':
\t\t\texcel_file = get_data_file()
\t\t\tif excel_file is None:
\t\t\t\tflash('No data file available for time series', 'warning')
\t\t\t\treturn redirect(url_for('index'))
\t\t\t
\t\t\traw_data, _, _ = get_cached_data(excel_file)
\t\t\tif raw_data is not None and len(raw_data) > 0:
\t\t\t\traw_data['Date'] = pd.to_datetime(raw_data['Date'], errors='coerce')
\t\t\t\traw_data = raw_data.dropna(subset=['Date'])
\t\t\t\traw_data = raw_data.sort_values('Date')
\t\t\t\ttime_series = raw_data.groupby('Date')['Fuel_Consumed'].sum()
\t\t\t\tgraph = pygal.Line()
\t\t\t\tgraph.title = 'Fuel Consumption Trends Over Time'
\t\t\t\tgraph.x_title = 'Date'
\t\t\t\tgraph.y_title = 'Fuel Consumption (L)'
\t\t\t\tgraph.x_label_rotation = 45
\t\t\t\tgraph.show_legend = True
\t\t\t\tgraph.x_labels = [d.strftime('%Y-%m-%d') for d in time_series.index]
\t\t\t\tgraph.add('Actual Consumption', list(time_series.values))
\t\t\t\tif len(time_series) >= 7:
\t\t\t\t\twindow = 7
\t\t\t\t\tmoving_avg = time_series.rolling(window=window, center=True).mean()
\t\t\t\t\tgraph.add(f'{window}-Day Moving Average', list(moving_avg.values))
\t\t\t\tfilename_base = f'timeseries_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
\t\t\telse:
\t\t\t\tflash('No time series data available', 'warning')
\t\t\t\treturn redirect(url_for('index'))
\t\telse:
\t\t\tflash('Invalid chart type', 'danger')
\t\t\treturn redirect(url_for('index'))
\t\t
\t\t# Render to PNG using cairosvg
\t\tsvg_data = graph.render()
\t\tpng_data = cairosvg.svg2png(bytestring=svg_data, dpi=config.PNG_EXPORT_DPI)
\t\t
\t\tresponse = make_response(png_data)
\t\tresponse.headers['Content-Disposition'] = f'attachment; filename={filename_base}.png'
\t\tresponse.headers['Content-Type'] = 'image/png'
\t\t
\t\tlogger.info(f"[EXPORT CHART PNG] {chart_type} chart exported")
\t\treturn response
\t\t
\texcept Exception as e:
\t\tlogger.error(f"[EXPORT ERROR] PNG chart export failed: {str(e)}")
\t\tflash(f'Export failed: {str(e)}', 'danger')
\t\treturn redirect(url_for('index'))


'''
    
    # Insert the export routes before if __name__
    lines.insert(insert_index, export_routes)
    
# Write back to file
with open('columns_app.py', 'w') as f:
    f.writelines(lines)

print("✅ Successfully added all export routes")
print("✅ Added chart_type='cluster' to clustergraph route")
print("✅ Fixed comparison export to match view (all unique sites, 10 colors)")
print("✅ Fixed timeseries export to properly load data file")
print(f"✅ Total lines: {len(lines)}")
