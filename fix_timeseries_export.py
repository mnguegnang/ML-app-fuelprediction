#!/usr/bin/env python3
"""
Fix timeseries export to dynamically detect date and fuel columns
"""

with open('columns_app.py', 'r') as f:
    content = f.read()

# Fix 1: Replace hardcoded column names in SVG export with dynamic detection
old_svg = """raw_data, _, _ = get_cached_data(excel_file)
\t\tif raw_data is not None and len(raw_data) > 0:
\t\t\traw_data['Date'] = pd.to_datetime(raw_data['Date'], errors='coerce')
\t\t\traw_data = raw_data.dropna(subset=['Date'])
\t\t\traw_data = raw_data.sort_values('Date')
\t\t\ttime_series = raw_data.groupby('Date')['Fuel_Consumed'].sum()
\t\t\tgraph = pygal.Line()
\t\t\tgraph.title = 'Fuel Consumption Trends Over Time'
\t\t\tgraph.x_title = 'Date'
\t\t\tgraph.y_title = 'Fuel Consumption (L)'
\t\t\tgraph.x_label_rotation = 45
\t\t\tgraph.show_legend = True
\t\t\tgraph.x_labels = [d.strftime('%Y-%m-%d') for d in time_series.index]
\t\t\tgraph.add('Actual Consumption', list(time_series.values))
\t\t\tif len(time_series) >= 7:
\t\t\t\twindow = 7
\t\t\t\tmoving_avg = time_series.rolling(window=window, center=True).mean()
\t\t\t\tgraph.add(f'{window}-Day Moving Average', list(moving_avg.values))
\t\t\tfilename_base = f'timeseries_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}'"""

new_svg = """raw_data, _, _ = get_cached_data(excel_file)
\t\tif raw_data is not None and len(raw_data) > 0:
\t\t\t# Find date column dynamically
\t\t\tdate_columns = []
\t\t\tfor col in raw_data.columns:
\t\t\t\tcol_lower = str(col).lower()
\t\t\t\tif any(keyword in col_lower for keyword in ['date', 'time', 'month', 'year', 'period', 'day']):
\t\t\t\t\tdate_columns.append(col)
\t\t\tif not date_columns:
\t\t\t\tflash('No date column found in data', 'warning')
\t\t\t\treturn redirect(url_for('index'))
\t\t\tdate_col = date_columns[0]
\t\t\t
\t\t\t# Find fuel column dynamically
\t\t\tnumeric_cols = raw_data.select_dtypes(include=['float64', 'int64']).columns
\t\t\tfuel_col = None
\t\t\tfor col in numeric_cols:
\t\t\t\tcol_lower = str(col).lower()
\t\t\t\tif any(keyword in col_lower for keyword in ['fuel', 'consumption', 'qty', 'qte', 'litre', 'liter']):
\t\t\t\t\tfuel_col = col
\t\t\t\t\tbreak
\t\t\tif fuel_col is None and len(numeric_cols) > 0:
\t\t\t\tfuel_col = numeric_cols[0]
\t\t\tif fuel_col is None:
\t\t\t\tflash('No numeric column found for time series', 'warning')
\t\t\t\treturn redirect(url_for('index'))
\t\t\t
\t\t\traw_data[date_col] = pd.to_datetime(raw_data[date_col], errors='coerce')
\t\t\traw_data = raw_data.dropna(subset=[date_col])
\t\t\traw_data = raw_data.sort_values(date_col)
\t\t\t
\t\t\t# Determine grouping frequency
\t\t\tdate_range = (raw_data[date_col].max() - raw_data[date_col].min()).days
\t\t\tif date_range > 365:
\t\t\t\tfreq = 'M'
\t\t\t\ttitle_suffix = '(Monthly)'
\t\t\telif date_range > 30:
\t\t\t\tfreq = 'W'
\t\t\t\ttitle_suffix = '(Weekly)'
\t\t\telse:
\t\t\t\tfreq = 'D'
\t\t\t\ttitle_suffix = '(Daily)'
\t\t\t
\t\t\traw_data['period'] = raw_data[date_col].dt.to_period(freq)
\t\t\ttime_series = raw_data.groupby('period')[fuel_col].sum()
\t\t\t
\t\t\tgraph = pygal.Line()
\t\t\tgraph.title = f'Fuel Consumption Trend Over Time {title_suffix}'
\t\t\tgraph.x_title = 'Time Period'
\t\t\tgraph.y_title = 'Fuel Consumption (L)'
\t\t\tgraph.x_label_rotation = 45
\t\t\tgraph.show_legend = True
\t\t\t
\t\t\t# Format x-labels
\t\t\tx_labels = []
\t\t\tfor period in time_series.index:
\t\t\t\tperiod_dt = period.to_timestamp()
\t\t\t\tif freq == 'M':
\t\t\t\t\tx_labels.append(period_dt.strftime('%b %Y'))
\t\t\t\telif freq == 'W':
\t\t\t\t\tweek_num = period_dt.isocalendar()[1]
\t\t\t\t\tx_labels.append(f'W{week_num} {period_dt.year}')
\t\t\t\telse:
\t\t\t\t\tx_labels.append(period_dt.strftime('%d %b %Y'))
\t\t\t
\t\t\tgraph.x_labels = x_labels
\t\t\tgraph.x_labels_major_every = max(1, len(x_labels) // 10)
\t\t\tgraph.show_minor_x_labels = False
\t\t\tgraph.add('Actual Consumption', [float(v) for v in time_series.values])
\t\t\t
\t\t\tif len(time_series) >= 3:
\t\t\t\twindow = min(3, len(time_series) // 3)
\t\t\t\tmoving_avg = time_series.rolling(window=window, center=True).mean()
\t\t\t\tgraph.add('Trend (Moving Avg)', [float(v) if pd.notna(v) else None for v in moving_avg.values])
\t\t\t
\t\t\tfilename_base = f'timeseries_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}'"""

# Replace first occurrence (SVG export)
if old_svg in content:
    content = content.replace(old_svg, new_svg, 1)
    print("✓ Fixed timeseries SVG export (dynamic column detection)")
else:
    print("✗ Could not find timeseries SVG export code")

# Replace second occurrence (PNG export) 
count = content.count(old_svg)
if count > 0:
    content = content.replace(old_svg, new_svg)
    print(f"✓ Fixed timeseries PNG export (dynamic column detection)")
else:
    print("✗ Could not find timeseries PNG export code")

with open('columns_app.py', 'w') as f:
    f.write(content)

print("✓ All timeseries export routes updated")
print("✓ Now using dynamic date and fuel column detection")
print("✓ Matches timeseriesgraph route logic")
