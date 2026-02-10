#!/usr/bin/env python3
"""
Fix indentation in timeseries export sections by properly indenting nested structures
"""

with open('columns_app.py', 'r') as f:
    lines = f.readlines()

def fix_section(lines, start_line):
    """Fix indentation in a timeseries export section starting at the given line"""
    # Expected indentation map based on the structure
    # Line content pattern -> number of tabs
    indent_map = {
        'for col in raw_data.columns:': 4,
        'col_lower = str(col).lower()': 5,
        'if any(keyword in col_lower for keyword in [\'date\'': 5,
        'date_columns.append(col)': 6,
        'if not date_columns:': 4,
        'flash(\'No date column found': 5,
        'return redirect(url_for(\'index\'))': 5,
        'date_col = date_columns[0]': 4,
        'for col in numeric_cols:': 4,
        # second col_lower will match same pattern
        'if any(keyword in col_lower for keyword in [\'fuel\'': 5,
        'fuel_col = col': 6,
        'break': 6,
        'if fuel_col is None and len(numeric_cols) > 0:': 4,
        'fuel_col = numeric_cols[0]': 5,
        'if fuel_col is None:': 4,
        'flash(\'No numeric column found': 5,
        'if date_range > 365:': 4,
        'freq = \'M\'': 5,
        'title_suffix = \'(Monthly)\'': 5,
        'elif date_range > 30:': 4,
        'freq = \'W\'': 5,
        'title_suffix = \'(Weekly)\'': 5,
        'else:': 4,
        'freq = \'D\'': 5,
        'title_suffix = \'(Daily)\'': 5,
        'for period in time_series.index:': 5,
        'period_dt = period.to_timestamp()': 6,
        'if freq == \'M\':': 6,
        'x_labels.append(period_dt.strftime': 7,
        'elif freq == \'W\':': 6,
        'week_num = period_dt.isocalendar()[1]': 7,
        'x_labels.append(f\'W{week_num}': 7,
        'x_labels.append(period_dt.strftime(\'%d': 7,
    }
    
    i = start_line
    lines_fixed = 0
    
    # Process lines in this section
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Stop at end of section
        if 'filename_base = ' in line:
            break
        
        # Find matching pattern and fix indentation
        for pattern, expected_tabs in indent_map.items():
            if pattern in stripped:
                # Count current tabs
                current_tabs = 0
                for ch in line:
                    if ch == '\t':
                        current_tabs += 1
                    else:
                        break
                
                # Fix if needed
                if current_tabs != expected_tabs:
                    lines[i] = '\t' * expected_tabs + stripped + '\n'
                    lines_fixed += 1
                break
        
        i += 1
    
    return lines_fixed

# Find both timeseries sections and fix them
sections_fixed = 0
i = 0

while i < len(lines):
    if 'elif chart_type == \'timeseries\':' in lines[i]:
        # Found a timeseries section
        lines_fixed = fix_section(lines, i)
        sections_fixed += 1
        print(f"✓ Fixed timeseries export section #{sections_fixed} starting at line {i+1}: {lines_fixed} lines corrected")
        
        if sections_fixed >= 2:
            break
    i += 1

# Write fixed file
with open('columns_app.py', 'w') as f:
    f.writelines(lines)

print(f"\n✓ Total sections fixed: {sections_fixed}")
