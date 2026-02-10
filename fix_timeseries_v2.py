#!/usr/bin/env python3
"""
Fix timeseries export - replace hardcoded column names with dynamic detection
"""

with open('columns_app.py', 'r') as f:
    lines = f.readlines()

# Find and replace the timeseries export logic (both SVG and PNG)
i = 0
replacements_made = 0

while i < len(lines):
    # Look for the elif chart_type == 'timeseries': line
    if "elif chart_type == 'timeseries':" in lines[i]:
        # Skip ahead to find the raw_data['Date'] line
        j = i + 1
        while j < len(lines) and j < i + 30:
            if "raw_data['Date'] = pd.to_datetime(raw_data['Date']" in lines[j]:
                # Found it! Now replace the next ~17 lines
                indent = '\t\t\t'  # The indentation level
                
                # Build the replacement lines
                replacement = [
                    f"{indent}# Find date column dynamically\n",
                    f"{indent}date_columns = []\n",
                    f"{indent}for col in raw_data.columns:\n",
                    f"{indent}\tcol_lower = str(col).lower()\n",
                    f"{indent}\tif any(keyword in col_lower for keyword in ['date', 'time', 'month', 'year', 'period', 'day']):\n",
                    f"{indent}\t\tdate_columns.append(col)\n",
                    f"{indent}if not date_columns:\n",
                    f"{indent}\tflash('No date column found in data', 'warning')\n",
                    f"{indent}\treturn redirect(url_for('index'))\n",
                    f"{indent}date_col = date_columns[0]\n",
                    f"{indent}\n",
                    f"{indent}# Find fuel column dynamically\n",
                    f"{indent}numeric_cols = raw_data.select_dtypes(include=['float64', 'int64']).columns\n",
                    f"{indent}fuel_col = None\n",
                    f"{indent}for col in numeric_cols:\n",
                    f"{indent}\tcol_lower = str(col).lower()\n",
                    f"{indent}\tif any(keyword in col_lower for keyword in ['fuel', 'consumption', 'qty', 'qte', 'litre', 'liter']):\n",
                    f"{indent}\t\tfuel_col = col\n",
                    f"{indent}\t\tbreak\n",
                    f"{indent}if fuel_col is None and len(numeric_cols) > 0:\n",
                    f"{indent}\tfuel_col = numeric_cols[0]\n",
                    f"{indent}if fuel_col is None:\n",
                    f"{indent}\tflash('No numeric column found for time series', 'warning')\n",
                    f"{indent}\treturn redirect(url_for('index'))\n",
                    f"{indent}\n",
                    f"{indent}raw_data[date_col] = pd.to_datetime(raw_data[date_col], errors='coerce')\n",
                    f"{indent}raw_data = raw_data.dropna(subset=[date_col])\n",
                    f"{indent}raw_data = raw_data.sort_values(date_col)\n",
                    f"{indent}\n",
                    f"{indent}# Determine grouping frequency\n",
                    f"{indent}date_range = (raw_data[date_col].max() - raw_data[date_col].min()).days\n",
                    f"{indent}if date_range > 365:\n",
                    f"{indent}\tfreq = 'M'\n",
                    f"{indent}\ttitle_suffix = '(Monthly)'\n",
                    f"{indent}elif date_range > 30:\n",
                    f"{indent}\tfreq = 'W'\n",
                    f"{indent}\ttitle_suffix = '(Weekly)'\n",
                    f"{indent}else:\n",
                    f"{indent}\tfreq = 'D'\n",
                    f"{indent}\ttitle_suffix = '(Daily)'\n",
                    f"{indent}\n",
                    f"{indent}raw_data['period'] = raw_data[date_col].dt.to_period(freq)\n",
                    f"{indent}time_series = raw_data.groupby('period')[fuel_col].sum()\n",
                    f"{indent}\n",
                    f"{indent}graph = pygal.Line()\n",
                    f"{indent}graph.title = f'Fuel Consumption Trend Over Time {{title_suffix}}'\n",
                    f"{indent}graph.x_title = 'Time Period'\n",
                    f"{indent}graph.y_title = 'Fuel Consumption (L)'\n",
                    f"{indent}graph.x_label_rotation = 45\n",
                    f"{indent}graph.show_legend = True\n",
                    f"{indent}\n",
                    f"{indent}# Format x-labels\n",
                    f"{indent}x_labels = []\n",
                    f"{indent}for period in time_series.index:\n",
                    f"{indent}\tperiod_dt = period.to_timestamp()\n",
                    f"{indent}\tif freq == 'M':\n",
                    f"{indent}\t\tx_labels.append(period_dt.strftime('%b %Y'))\n",
                    f"{indent}\telif freq == 'W':\n",
                    f"{indent}\t\tweek_num = period_dt.isocalendar()[1]\n",
                    f"{indent}\t\tx_labels.append(f'W{{week_num}} {{period_dt.year}}')\n",
                    f"{indent}\telse:\n",
                    f"{indent}\t\tx_labels.append(period_dt.strftime('%d %b %Y'))\n",
                    f"{indent}\n",
                    f"{indent}graph.x_labels = x_labels\n",
                    f"{indent}graph.x_labels_major_every = max(1, len(x_labels) // 10)\n",
                    f"{indent}graph.show_minor_x_labels = False\n",
                    f"{indent}graph.add('Actual Consumption', [float(v) for v in time_series.values])\n",
                    f"{indent}\n",
                    f"{indent}if len(time_series) >= 3:\n",
                    f"{indent}\twindow = min(3, len(time_series) // 3)\n",
                    f"{indent}\tmoving_avg = time_series.rolling(window=window, center=True).mean()\n",
                    f"{indent}\tgraph.add('Trend (Moving Avg)', [float(v) if pd.notna(v) else None for v in moving_avg.values])\n",
                    f"{indent}\n",
                ]
                
                # Find where to stop deleting (at the filename_base line)
                k = j
                while k < len(lines) and k < j + 20:
                    if 'filename_base = ' in lines[k] and 'timeseries_chart_' in lines[k]:
                        # Delete from j to k-1, insert replacement
                        del lines[j:k]
                        # Insert all replacement lines
                        for idx, new_line in enumerate(replacement):
                            lines.insert(j + idx, new_line)
                        replacements_made += 1
                        print(f"✓ Fixed timeseries export #{replacements_made} at line {i+1}")
                        break
                    k += 1
                break
            j += 1
    i += 1

with open('columns_app.py', 'w') as f:
    f.writelines(lines)

print(f"\n✓ Made {replacements_made} replacements")
print("✓ Timeseries exports now use dynamic column detection")
print("✓ Matches the logic from timeseriesgraph route")
