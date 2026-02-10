#!/usr/bin/env python3
"""Fix the export routes in columns_app.py"""
import re

# Read the file
with open('columns_app.py', 'r') as f:
    content = f.read()

# Fix 1: Fix the timeseries export (SVG route) - replace get_cached_data(None) with proper file loading
old_timeseries_svg = r"(elif chart_type == 'timeseries':)\s+raw_data = get_cached_data\(None\)\s+if raw_data is not None and len\(raw_data\) > 0:"

new_timeseries_svg = r"""\1
\t\texcel_file = get_data_file()
\t\tif excel_file is None:
\t\t\tflash('No data file available for time series', 'warning')
\t\t\treturn redirect(url_for('index'))
\t\t
\t\traw_data, _, _ = get_cached_data(excel_file)
\t\tif raw_data is not None and len(raw_data) > 0:"""

content = re.sub(old_timeseries_svg, new_timeseries_svg, content, count=1)

# Find the PNG export section and fix it too (it's duplicated)
# Search for the second occurrence in the PNG export route
matches = list(re.finditer(r"(elif chart_type == 'timeseries':)\s+raw_data = get_cached_data\(None\)\s+if raw_data is not None and len\(raw_data\) > 0:", content))
if len(matches) > 0:
    # Replace the match (should be in PNG route)
    start = matches[0].start()
    end = matches[0].end()
    replacement = matches[0].group(1) + """
\t\texcel_file = get_data_file()
\t\tif excel_file is None:
\t\t\tflash('No data file available for time series', 'warning')
\t\t\treturn redirect(url_for('index'))
\t\t
\t\traw_data, _, _ = get_cached_data(excel_file)
\t\tif raw_data is not None and len(raw_data) > 0:"""
    content = content[:start] + replacement + content[end:]

# Fix 2: Fix comparison export to match the view (use all sites, not top 10)
# Find the comparison export section and replace the chart generation logic
old_comparison = r"elif chart_type == 'comparison':\s+predictions = prediction_cache\.get\('predictions', \[\]\)\s+clusters = prediction_cache\.get\('clusters', \[\]\)\s+sites = prediction_cache\.get\('sites', \[\]\)\s+if len\(predictions\) > 0 and len\(clusters\) > 0 and len\(sites\) > 0:\s+df = pd\.DataFrame\(\{'clusters': clusters, 'sites': sites, 'predictions': predictions\}\)\s+cluster_totals = df\.groupby\('clusters'\)\['predictions'\]\.sum\(\)\.sort_values\(ascending=False\)\s+site_colors = config\.SITE_COLORS"

new_comparison_start = """elif chart_type == 'comparison':
\t\tpredictions = prediction_cache.get('predictions', [])
\t\tclusters = prediction_cache.get('clusters', [])
\t\tsites = prediction_cache.get('sites', [])
\t\tif len(predictions) > 0 and len(clusters) > 0 and len(sites) > 0:
\t\t\tdf = pd.DataFrame({'clusters': clusters, 'sites': sites, 'predictions': predictions})
\t\t\tcluster_totals = df.groupby('clusters')['predictions'].sum().sort_values(ascending=False)
\t\t\t
\t\t\t# Use same colors as comparison view (10 colors, not 20)
\t\t\tsite_colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', 
\t\t\t               '#00f2fe', '#43e97b', '#38f9d7', '#fa709a', '#fee140']"""

# This is complex, so let's do a simpler find-replace
content = content.replace(
    "\t\t\tsite_colors = config.SITE_COLORS",
    "\t\t\t# Use same colors as comparison view\n\t\t\tsite_colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', \n\t\t\t               '#00f2fe', '#43e97b', '#38f9d7', '#fa709a', '#fee140']"
)

# Now fix the comparison logic to match the view exactly
# Replace the simplified logic with the full logic from the view
old_comparison_logic = """\t\t\ttop_sites_overall = sorted(all_top_sites.items(), key=lambda x: x[1], reverse=True)[:config.COMPARISON_TOP_SITES]
\t\t\tdisplayed_sites = set([site for site, _ in top_sites_overall])
\t\t\tgraph.x_labels = list(cluster_totals.index)
\t\t\tfor site_name, _ in top_sites_overall:
\t\t\t\tsite_values = []
\t\t\t\tfor cluster in cluster_totals.index:
\t\t\t\t\tcluster_df = df[df['clusters'] == cluster]
\t\t\t\t\tsite_val = cluster_df[cluster_df['sites'] == site_name]['predictions'].sum()
\t\t\t\t\tsite_values.append(float(site_val) if site_val > 0 else 0)
\t\t\t\tgraph.add(str(site_name), site_values)
\t\t\tother_values = []
\t\t\tfor cluster in cluster_totals.index:
\t\t\t\tcluster_df = df[df['clusters'] == cluster]
\t\t\t\tother_sum = cluster_df[~cluster_df['sites'].isin(displayed_sites)]['predictions'].sum()
\t\t\t\tother_values.append(float(other_sum))
\t\t\tif sum(other_values) > 0:
\t\t\t\tgraph.add('Others', other_values)"""

new_comparison_logic = """\t\t\t# Sort sites by total consumption (ALL unique top sites, not limited)
\t\t\tranked_sites = sorted(all_top_sites.items(), key=lambda x: x[1], reverse=True)
\t\t\t
\t\t\t# Create series for each unique top site
\t\t\tsite_series = {site: [] for site, _ in ranked_sites}
\t\t\tother_sites_series = []
\t\t\t
\t\t\tfor cluster in cluster_totals.index:
\t\t\t\tif cluster in cluster_top_sites:
\t\t\t\t\ttop_site, top_value, contribution_pct = cluster_top_sites[cluster]
\t\t\t\t\tcluster_total = float(cluster_totals[cluster])
\t\t\t\t\tother_value = cluster_total - top_value
\t\t\t\t\t
\t\t\t\t\t# Add value for the top site in this cluster
\t\t\t\t\tfor site in site_series:
\t\t\t\t\t\tif site == top_site:
\t\t\t\t\t\t\tsite_series[site].append(top_value)
\t\t\t\t\t\telse:
\t\t\t\t\t\t\tsite_series[site].append(0)
\t\t\t\t\t
\t\t\t\t\t# Add "other sites" value
\t\t\t\t\tother_sites_series.append(other_value)
\t\t\t\telse:
\t\t\t\t\t# No data for this cluster
\t\t\t\t\tfor site in site_series:
\t\t\t\t\t\tsite_series[site].append(0)
\t\t\t\t\tother_sites_series.append(float(cluster_totals[cluster]))
\t\t\t
\t\t\t# Set cluster names as x-labels
\t\t\tgraph.x_labels = list(cluster_totals.index)
\t\t\t
\t\t\t# Add series for each top site (creates different colors - show ALL sites)
\t\t\tfor site, _ in ranked_sites:
\t\t\t\tgraph.add(site, site_series[site])
\t\t\t
\t\t\t# Add "Other Sites" series
\t\t\tgraph.add('Other Sites', other_sites_series)"""

content = content.replace(old_comparison_logic, new_comparison_logic)

# Write the file back
with open('columns_app.py', 'w') as f:
    f.write(content)

print("✓ Fixed comparison chart export logic to match view")
print("✓ Fixed timeseries export to properly load data file")
print("✓ All export routes should now work correctly")
