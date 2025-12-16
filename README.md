# Taxi Trip Time Prediction System

A comprehensive machine learning web application that predicts taxi trip durations using GPS trajectory data and Random Forest regression. This tool helps taxi companies and transportation services estimate trip times accurately, optimize fleet management, and improve customer experience.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project solves the critical problem of predicting taxi trip durations using machine learning. Instead of relying on simple calculations or manual estimates, this application:

- Learns from historical data - Analyzes thousands of past taxi trips
- Identifies patterns - Understands how different factors affect trip times
- Makes accurate predictions - Provides reliable time estimates for new trips
- Interactive web interface - Easy-to-use dashboard for real-time predictions

![Prediction Interface](images/Prediction%20Interface.png)

### Key Benefits

- Accurate Predictions - R² score typically > 0.85 (85%+ accuracy)
- Real-time Analysis - Instant predictions with interactive interface
- Batch Processing - Predict multiple trips simultaneously
- Performance Tracking - Monitor prediction history and accuracy
- Model Insights - Understand which factors most influence trip times

---

## Features

### Web Application Features

**Prediction Page**
- Single trip predictions with detailed results
- Batch predictions (CSV format)
- Model selection from multiple saved models
- Comparison with actual trip times (optional)

**Model Information**
- Model architecture and hyperparameters
- Performance metrics (RMSE, R² scores)
- Feature importance visualization
- Model retraining capabilities

**Data Overview**
- Dataset statistics and distributions
- Call type breakdown charts
- Trip time histograms
- Sample data exploration

![Data Overview](images/Data%20Overview.png)
![Sample Data](images/Sample%20Data.png)

**Prediction History**
- Complete prediction tracking
- Filterable history table
- Error analysis and visualizations
- CSV export functionality

**Model Comparison**
- Side-by-side model performance
- Comparative metrics and charts
- Best model identification

### Machine Learning Features

- Random Forest Regressor with 200 decision trees
- Feature Engineering - Polyline length extraction from GPS data
- Data Preprocessing - Automatic cleaning and normalization
- Model Persistence - Save and load trained models
- Performance Evaluation - Comprehensive metrics and validation

---

## How It Works

### Data Processing Pipeline

The application follows this workflow:
1. CSV Data File - Loads taxi trip data from CSV format
2. Data Loading & Cleaning - Filters invalid records and missing data
3. Feature Extraction - Extracts polyline length from GPS coordinates and encodes call types
4. Model Training - Trains Random Forest with 200 trees using 70/30 train-test split
5. Feature Scaling - Standardizes features using StandardScaler
6. Model Persistence - Saves trained model for future use
7. Real-time Predictions - Makes predictions on new data

### Prediction Workflow

1. Input: User provides polyline length and call type
2. Preprocessing: Features are standardized using saved scaler
3. Prediction: Model processes input through 200 decision trees
4. Output: Trip duration in seconds (converted to readable format)
5. Storage: Prediction saved to history for tracking

### Machine Learning Model

**Algorithm**: Random Forest Regressor

**Features Used**:
- Polyline Length: Number of GPS coordinate points (each point ≈ 15 seconds)
- CALL_TYPE_A: Binary indicator for central dispatch trips
- CALL_TYPE_B: Binary indicator for taxi stand trips
- CALL_TYPE_C: Binary indicator for random street hails

**Hyperparameters**:
- n_estimators: 200 (number of decision trees)
- min_samples_split: 2
- min_samples_leaf: 1
- random_state: 42 (for reproducibility)

---

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/Dhanush010/Taxi-Trip-Time-Prediction-System.git
cd Taxi-Trip-Time-Prediction-System
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- streamlit >= 1.28.0
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- joblib >= 1.2.0
- matplotlib >= 3.6.0

### Step 3: Prepare Data

Ensure you have the training data file `train(1).csv` in the project directory. The CSV should contain:
- POLYLINE: GPS coordinate arrays (string format)
- CALL_TYPE: Trip booking type (A, B, or C)
- MISSING_DATA: Boolean flag for data completeness

---

## Usage

### Running the Application

**Method 1: Streamlit Command**
```bash
streamlit run app.py
```

**Method 2: Windows Batch File**
```bash
run_app.bat
```

The application will automatically open in your default web browser at `http://localhost:8501`

### Making Predictions

**Single Prediction**
1. Navigate to the Prediction page
2. Enter Polyline Length (number of GPS points - 1)
3. Select Call Type (A, B, or C)
4. Optionally enter actual trip time for comparison
5. Click "Predict Trip Time"

Example:
- Polyline Length: 100
- Call Type: B (Taxi Stand)
- Prediction: ~25 minutes (1500 seconds)

**Batch Prediction**
1. Expand the "Batch Prediction" section
2. Enter trips in CSV format:
   ```
   50,A
   100,B
   200,C
   ```
3. Click "Predict Batch"
4. Download results as CSV if needed

### First Run

On the first run, the application will:
- Load and process the training data (takes 10-30 seconds)
- Train the machine learning model (takes 30-60 seconds)
- Save the trained model automatically
- Display performance metrics

Subsequent runs will be much faster as the model is loaded from disk.

---

## Model Details

![Model Information](images/Model%20Information.png)

### Performance Metrics

The model is evaluated using:

- RMSE (Root Mean Squared Error): Average prediction error in seconds
  - Lower is better
  - Typically ranges based on dataset characteristics

- R² Score (Coefficient of Determination): Proportion of variance explained
  - Range: 0 to 1 (higher is better)
  - Typical Performance: > 0.85 (85%+ accuracy)

### Model Architecture

The model uses Random Forest Regressor with:
- 200 Decision Trees
- Feature Scaling (StandardScaler)
- Train-Test Split (70-30)
- Performance Metrics (RMSE, R²)

### Feature Importance

The model learns which features are most important:

1. Polyline Length - Primary predictor (most important)
2. Call Type - Secondary factors (may vary slightly by type)

Feature importance can be viewed in the Model Info page.

---

## Project Structure

```
Taxi-Trip-Time-Prediction-System/
│
├── app.py                          # Main Streamlit application
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── run_app.bat                     # Windows launcher script
├── Taxi Trip Time Prediction.ipynb # Jupyter notebook (optional)
│
├── images/                         # Screenshots and visualizations
│   ├── Prediction Interface.png
│   ├── Model Information.png
│   ├── Data Overview.png
│   └── Sample Data.png
│
├── train(1).csv                    # Training dataset (not in repo - too large)
├── winnig_model_random_forest.pkl  # Trained model (generated)
├── scaler.pkl                      # Feature scaler (generated)
├── prediction_history.csv          # Prediction logs (generated)
│
└── saved_models/                   # Custom saved models (generated)
    └── [model_name]_model.pkl
    └── [model_name]_scaler.pkl
```

Note: Large files (`.pkl`, `.csv`, etc.) are excluded from Git via `.gitignore`

---

## Technologies Used

### Core Technologies

- Python - Programming language (3.7+)
- Streamlit - Web application framework (1.28+)
- scikit-learn - Machine learning library (1.2+)
- Pandas - Data manipulation (1.5+)
- NumPy - Numerical computing (1.23+)
- Matplotlib - Data visualization (3.6+)
- Joblib - Model serialization (1.2+)

### Machine Learning

- Random Forest Regressor - Ensemble learning algorithm
- StandardScaler - Feature normalization
- Train-Test Split - Data validation strategy

---

## Example Use Cases

### For Taxi Companies
- Estimate arrival times for customers
- Optimize fleet allocation and routing
- Analyze trip patterns by call type
- Plan driver schedules based on trip durations

### For Ride-Sharing Services
- Predict trip times for better pricing
- Improve customer experience with accurate ETAs
- Analyze demand patterns across different areas
- Optimize matching algorithms

### For Researchers
- Study transportation patterns
- Analyze GPS trajectory data
- Experiment with different ML models
- Research trip duration factors

---

## Contributing

Contributions are welcome! If you'd like to contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Ideas for Contribution

- Add more features (time of day, weather, traffic data)
- Implement different ML algorithms (XGBoost, Neural Networks)
- Improve UI/UX design
- Add API endpoints
- Write unit tests
- Enhance documentation

---

## Notes

- The model automatically trains on first run (30-60 seconds)
- Training uses a subset of 50,000 records for performance
- Model is saved automatically for future use
- All predictions are logged to history
- Large data files should be stored separately or use Git LFS

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Author

**Dhanush**  
- GitHub: [@Dhanush010](https://github.com/Dhanush010)

**Vedant Pradhan**  
- GitHub: [@Vedant0703](https://github.com/Vedant0703)
  
- Project Link: [https://github.com/Dhanush010/Taxi-Trip-Time-Prediction-System](https://github.com/Dhanush010/Taxi-Trip-Time-Prediction-System)

---

## Acknowledgments

- Taxi trip dataset providers
- Streamlit community for the amazing framework
- scikit-learn developers for robust ML tools
- Open source community for inspiration and support

---

Made with Python, Streamlit, and Machine Learning
