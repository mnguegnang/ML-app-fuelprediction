# Fuel Consumption Prediction Application

A Flask-based web application for predicting fuel consumption in power generation plant using machine learning.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This application provides an end-to-end solution for predicting fuel consumption at generator sites using a trained machine learning model (GradientBoostingRegressor). Users can upload Excel data files, generate predictions, visualize results through interactive charts, and export data in multiple formats.

**Key Metrics:**
- Model Performance: NSE = 0.9816 (excellent accuracy)
- Supported Data: 6000+ sites across multiple clusters
- Export Formats: CSV, Excel, SVG, PNG, Text summaries

## ✨ Features

### Core Functionality
- **📤 File Upload**: Secure Excel file upload with validation (.xlsx, .xls, .ods)
- **🤖 ML Predictions**: Generate fuel consumption predictions using GradientBoostingRegressor
- **📊 Interactive Visualizations**:
  - Cluster-level bar chart with anomaly detection
  - Top 20 sites horizontal bar chart
  - Distribution histogram with statistics
  - Cluster breakdown comparison (stacked bar)
  - Time-series trend analysis with moving averages
- **💾 Export Options**:
  - CSV and Excel spreadsheets
  - Comprehensive text summaries
  - SVG vector graphics
  - PNG raster images (requires cairosvg)
- **📈 Performance Monitoring**: Real-time metrics tracking and cache statistics
- **🔍 Anomaly Detection**: Automatic identification of high-consumption sites

### Technical Features
- In-memory caching for fast performance
- Session management for multi-user support
- Comprehensive error handling and logging
- Nash-Sutcliffe Efficiency (NSE) calculation for model validation
- Dynamic column name resolution for flexible data formats

## 🏗️ Architecture

The application follows a modular blueprint architecture for maintainability and scalability.

```:
fuel_prediction_app/
├── columns_app.py              # Main application (191 lines)
├── config.py                   # Configuration constants
├── pkl_objects/
│   └── filename.joblib         # Trained ML model
├── uploads/                    # User uploaded files
├── logs/                       # Application logs
├── templates/                  # HTML templates
├── utils/                      # Utility modules (7 modules)
│   ├── file_utils.py          # File operations & validation
│   ├── validation_utils.py    # Input & data validation
│   ├── data_utils.py          # Caching & data management
│   ├── model_utils.py         # Model loading & metadata
│   ├── metrics_utils.py       # Performance metrics & NSE
│   ├── chart_utils.py         # Chart generation (Pygal)
│   └── export_utils.py        # Export functionality
└── routes/                     # Route blueprints (3 modules)
    ├── main_routes.py         # Main pages & prediction logic
    ├── visualization_routes.py # Chart generation routes
    └── export_routes.py       # Export/download routes
```

### Module Overview

**Application Core** (`columns_app.py` - 191 lines):
- Flask app initialization and configuration
- Blueprint registration
- Model loading
- Metrics tracking setup

**Route Blueprints** (3 modules, ~1100 lines total):
1. **main_routes.py** (~600 lines): Homepage, upload, model info, cache stats, core prediction logic
2. **visualization_routes.py** (~450 lines): 5 chart types (cluster, site, distribution, comparison, time-series)
3. **export_routes.py** (~300 lines): 5 export endpoints (CSV, Excel, text, SVG, PNG)

**Utility Modules** (7 modules, ~1700 lines total):
1. **file_utils.py** (254 lines): File validation, Excel reading, sheet detection
2. **validation_utils.py** (132 lines): Form validation, column resolution, data checks
3. **data_utils.py** (175 lines): Caching system, data retrieval, cache management
4. **model_utils.py** (95 lines): Model loading, metadata extraction
5. **metrics_utils.py** (241 lines): NSE calculation, statistics, anomaly detection
6. **chart_utils.py** (378 lines): Pygal chart generation for all chart types
7. **export_utils.py** (450 lines): Multi-format export generation

**Total Codebase**: ~3200 lines (down from 2178 lines in single file, +47% more functionality, better organization)

## 🚀 Installation

### Prerequisites

- Python 3.7+
- pip package manager
- Virtual environment (recommended)

### Quick Installation

1. **Clone/Download the repository**
```bash
cd /path/to/your/workspace
```

2. **Create virtual environment** (recommended)
```bash
python -m venv fuel_app_env
source fuel_app_env/bin/activate  # Linux/Mac
# OR
fuel_app_env\Scripts\activate  # Windows
```

3. **Install all dependencies**
```bash
pip install -r requirements.txt
```

**That's it!** All required packages will be installed automatically.

### Manual Installation (Alternative)

If you prefer to install packages individually:

```bash
# Core dependencies
pip install Flask>=2.3.0 pandas>=2.0.0 numpy>=1.24.0
pip install scikit-learn>=1.3.0 pygal>=3.0.0 openpyxl>=3.1.0

# Optional: For PNG export (requires system libraries)
pip install cairosvg>=2.7.0
```

### Required Dependencies

The application requires the following packages (automatically installed with `requirements.txt`):

- **Flask** (>=2.3.0): Web framework
- **pandas** (>=2.0.0): Data manipulation
- **numpy** (>=1.24.0): Numerical operations
- **scikit-learn** (>=1.3.0): Machine learning model support
- **pygal** (>=3.0.0): Chart generation
- **openpyxl** (>=3.1.0): Excel file support
- **xlrd** (>=2.0.0): Legacy Excel format support
- **cairosvg** (>=2.7.0, optional): PNG export from SVG charts

### Verify Installation

```bash
python columns_app.py
# Should start server on http://localhost:6003
```

If successful, you'll see:
```
INFO - ✓ Model loaded successfully
INFO - ✓ All blueprints registered successfully
INFO - Starting Flask development server on port 6003...
```


## 💻 Usage

### Starting the Application

```bash
python columns_app.py
```

Access at: `http://localhost:6003`

### Workflow

1. **Upload Data File**
   - Click "Upload File" on homepage
   - Select Excel file (.xlsx, .xls, .ods)
   - File must contain sheet named "generator only", "gen only", or "generator"
   - Required columns: Cluster, Site Name, Previous Fuel Qty, Fuel Found, etc.

2. **Generate Predictions**
   - Select columns from dropdown menus on homepage
   - Click "Predict" button
   - View cluster-level predictions

3. **Visualize Results**
   - **Cluster Graph**: Bar chart of consumption by cluster
   - **Site Graph**: Top 20 sites ranked by consumption
   - **Distribution**: Histogram showing consumption patterns
   - **Comparison**: Stacked bar showing cluster breakdown by top sites
   - **Time Series**: Trend analysis over time (if date columns available)

4. **Export Data**
   - Click export buttons on any chart page:
     - "Download CSV": Spreadsheet format
     - "Download Excel": Formatted workbook
     - "Download Summary": Text report with stats
     - "Download SVG": Vector graphic
     - "Download PNG": Raster image (if cairosvg installed)

5. **Monitor Performance**
   - Visit `/cache_stats` for system metrics
   - View NSE score, prediction statistics
   - Check cache status and processing times

### Data Format Requirements

**Required Columns** (case-insensitive matching supported):
- Cluster identifier
- Site name
- Previous fuel quantity
- Fuel quantity found
- Generator type
- Running time
- Consumption rate
- Number of days
- Fuel added
- Generator capacity (specific value: "6,5 x 2")

**Optional Columns**:
- Date/Time column (for time-series analysis)
- Consumption history (for NSE calculation)

**File Constraints**:
- Maximum size: 100 MB
- Supported formats: .xlsx, .xls, .ods
- Must have valid sheet name containing "generator"

## 📚 API Documentation

### Main Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Homepage with prediction form |
| `/upload` | GET, POST | File upload page |
| `/clear_upload` | GET | Clear uploaded file and cache |
| `/model-info` | GET | Model metadata and information |
| `/model-info/json` | GET | Model metadata as JSON |
| `/cache_stats` | GET | System monitoring dashboard |

### Visualization Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/clustergraph` | GET, POST | Cluster-level bar chart |
| `/sitegraph` | GET | Top 20 sites horizontal bar |
| `/distributiongraph` | GET | Distribution histogram |
| `/comparisonview` | GET | Cluster breakdown stacked bar |
| `/timeseriesgraph` | GET | Time-series trend analysis |

### Export Routes

| Route | Method | Description |
|-------|--------|-------------|
| `/export/predictions/csv` | GET | Download predictions as CSV |
| `/export/predictions/excel` | GET | Download predictions as Excel |
| `/export/summary` | GET | Download text summary report |
| `/export/chart/<type>` | GET | Download chart as SVG |
| `/export/chart/<type>/png` | GET | Download chart as PNG |

**Chart Types**: `cluster`, `site`, `distribution`, `comparison`, `timeseries`

### Response Formats

**Success Response**:
- Status: 200
- Content: Rendered HTML template or file download

**Error Response**:
- Status: 400 (validation error), 500 (server error)
- Content: Error template with detailed message

## ⚙️ Configuration

### Main Configuration (`config.py`)

```python
# Model Configuration
MODEL_PATH = 'pkl_objects/filename.joblib'
MODEL_METADATA = {
    'name': 'GradientBoostingRegressor',
    'version': '0.20.4',
    'nse_on_real_data': 0.9816,
    ...
}

# Application Constants
TOP_SITES_LIMIT = 20
TOP_CLUSTERS_LIMIT = 10
DISTRIBUTION_BINS = 20
ANOMALY_STD_MULTIPLIER = 2
PNG_EXPORT_DPI = 300

# Error Messages
ERROR_MESSAGES = {
    'model_not_loaded': 'Model could not be loaded...',
    'prediction_cache_empty': 'No predictions available...',
    ...
}
```

### Environment Variables

```bash
# Flask Configuration
export FLASK_ENV=development  # or production
export FLASK_DEBUG=1          # Enable debug mode
export SECRET_KEY='your-secret-key'  # Override app.secret_key
```

### Upload Configuration

- **Location**: `uploads/` directory (auto-created)
- **Max Size**: 100 MB
- **Allowed Extensions**: .xlsx, .xls, .ods
- **Validation**: File format, sheet names, required columns

## � For Collaborators

### Quick Setup for Team Members

Share this repository with your team. Collaborators can get started quickly:

```bash
# 1. Clone the repository
git clone <repository-url>
cd Prediction_visualization

# 2. Create virtual environment
python -m venv fuel_app_env
source fuel_app_env/bin/activate  # Linux/Mac
# or: fuel_app_env\Scripts\activate  # Windows

# 3. Install all dependencies (one command!)
pip install -r requirements.txt

# 4. Start the application
python columns_app.py
```

**That's it!** The `requirements.txt` file ensures everyone has the exact same package versions.

### Sharing Your Environment

If you've added new packages, update `requirements.txt`:

```bash
# Generate from your current environment
pip freeze > requirements.txt

# Or manually add specific packages
echo "new-package>=1.0.0" >> requirements.txt
```

### For Different Python Versions

The app supports Python 3.7+. If you need specific version compatibility:

```bash
# Create environment with specific Python version
python3.8 -m venv fuel_app_env
python3.9 -m venv fuel_app_env

# Then install requirements as usual
pip install -r requirements.txt
```

### Dependencies Troubleshooting

**Issue**: Package installation fails
```bash
# Update pip first
pip install --upgrade pip

# Then retry
pip install -r requirements.txt
```

**Issue**: cairosvg installation fails (PNG export)
```bash
# cairosvg is optional - comment it out in requirements.txt
# Or install system dependencies first:

# Ubuntu/Debian:
sudo apt-get install libcairo2-dev libpango1.0-dev

# macOS:
brew install cairo pango

# Windows: Download pre-built wheels from:
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#cairosvg
```

## �🔧 Development

### Code Organization Best Practices

1. **Route Handlers**: Keep routes thin, delegate to utility functions
2. **Error Handling**: Use try-except blocks, log errors, flash user messages
3. **Caching**: Leverage data_cache and prediction_cache for performance
4. **Logging**: Use logger.info(), logger.warning(), logger.error() consistently
5. **Validation**: Validate all inputs using validation_utils functions

### Adding New Features

#### Adding a New Chart Type

1. Create chart generation function in `utils/chart_utils.py`:
```python
def create_new_chart(data, options):
    graph = pygal.ChartType()
    # Configure and populate chart
    return graph.render_data_uri()
```

2. Add route in `routes/visualization_routes.py`:
```python
@viz_bp.route('/newchart')
def new_chart():
    # Get data from cache
    # Call create_new_chart()
    # Return template with chart
```

3. Add export logic in `routes/export_routes.py`:
```python
elif chart_type == 'new':
    data_dict = {'data': cache.get('data')}
    # Export using export_utils
```

#### Adding New Utility Functions

1. Determine appropriate module (file, validation, data, model, metrics, chart, export)
2. Add function with comprehensive docstring
3. Import in `utils/__init__.py` if needed
4. Update this README with new functionality

### Testing

```bash
# Test file upload
curl -F "file=@test_data.xlsx" http://localhost:6003/upload

# Test prediction endpoint
curl -X POST http://localhost:6003/clustergraph \
  -d "Cluster=column1&SITE Name=column2..."

# Test export
curl http://localhost:6003/export/predictions/csv -o predictions.csv
```

### Logging

Logs are written to `logs/fuel_app_YYYYMMDD.log` with format:
```
2024-02-09 14:30:15 - INFO - [PREDICTION REQUEST] Session: abc123 | Started
2024-02-09 14:30:16 - INFO - [PREDICTION SUCCESS] Predictions: 100 | NSE: 0.9850
```

**Log Levels**:
- DEBUG: Detailed diagnostic information
- INFO: General informational messages
- WARNING: Warning messages (non-critical issues)
- ERROR: Error messages (critical issues)

## 🐛 Troubleshooting

### Common Issues

**Issue**: Model not loaded error
```
Solution: Verify pkl_objects/filename.joblib exists and is valid
Check: config.MODEL_PATH is correct
```

**Issue**: Column not found error
```
Solution: The app supports flexible column name matching
Ensure your Excel columns match required fields (case-insensitive)
Check available columns in error message
```

**Issue**: No predictions available
```
Solution: Submit prediction form first before viewing charts
Clear cache with /clear_upload if stale data
```

**Issue**: PNG export not available
```
Solution: Install cairosvg: pip install cairosvg
System dependencies may be required: libcairo2-dev libpango1.0-dev
```

**Issue**: Cache showing stale data
```
Solution: Visit /clear_upload to reset cache
Or upload new file (automatically clears cache)
```

**Issue**: Time-series chart not available
```
Solution: Ensure your Excel file has date/time columns
Supported keywords: date, time, month, year, period, day
```

### Debug Mode

Enable detailed error messages:
```python
# In columns_app.py
if __name__ == '__main__':
    app.run(port=6003, debug=True)  # debug=True shows tracebacks
```

### Performance Optimization

1. **Caching**: Data is cached after first load - subsequent requests are fast
2. **File Size**: Keep uploaded files under 50 MB for best performance
3. **Memory**: Large predictions (10,000+ sites) may require more RAM
4. **Logging**: Set `logging.basicConfig(level=logging.INFO)` in production

## 📊 Model Information

**Algorithm**: GradientBoostingRegressor (scikit-learn)

**Performance**:
- Nash-Sutcliffe Efficiency (NSE): 0.9816 (excellent)
- R² Score: 0.887
- Training data: Generator fuel consumption records

**Features** (6 input features):
1. Fuel per period (calculated)
2. Running time
3. Consumption rate
4. Number of days
5. Fuel added
6. Generator 1 capacity (6,5 x 2)

**Output**: Predicted fuel consumption in liters

## 📝 Version History

### Version 2.0.0 (Current) - Refactored Architecture
- ✅ Modular blueprint architecture (3 blueprints)
- ✅ Utility modules extracted (7 modules)
- ✅ 90% reduction in main file size (1896 → 191 lines)
- ✅ Improved error handling and logging
- ✅ Comprehensive documentation

### Version 1.0.0 - Initial Release
- Single monolithic file (2178 lines)
- All functionality in columns_app.py
- Basic caching and validation

## 👥 Contributors

**AIMS-Cameroon Fuel Prediction Team**
- Application Development
- ML Model Training
- Refactoring & Optimization

## 📄 License

Internal use - AIMS-Cameroon

## 🤝 Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review error messages in browser
3. Visit `/cache_stats` for system status
4. Consult this README for troubleshooting

---

**Last Updated**: February 2026  
**Application Status**: Production Ready ✅
