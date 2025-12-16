# Taxi Trip Time Prediction - Streamlit Web App

A machine learning web application that predicts taxi trip durations based on GPS trajectory data using a Random Forest regression model.

## What This Project Does

Predicts taxi trip times by analyzing:
- **Polyline Length**: Number of GPS coordinate points (each point = ~15 seconds)
- **Call Type**: How the taxi was requested
  - Type A: Dispatched from central
  - Type B: Requested at taxi stand
  - Type C: Hailed on random street

The model uses Random Forest Regressor with 200 decision trees to learn patterns from historical trip data.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Ensure `train(1).csv` is in the project directory
2. Run the application:
   ```bash
   streamlit run app.py
   ```
   Or on Windows: `run_app.bat`
3. Enter polyline length and select call type to get predictions

## Features

- **Prediction Page**: Single and batch predictions
- **Model Info**: View performance metrics (RMSE, R²) and feature importance
- **Data Overview**: Dataset statistics and visualizations
- **Prediction History**: Track all predictions with export capability
- **Model Comparison**: Compare multiple saved models

## Model Details

- **Algorithm**: Random Forest Regressor
- **Features**: Polyline Length + Call Type (A/B/C)
- **Training**: Automatically trains on first run using 50,000 records
- **Performance**: Typically achieves R² > 0.85

## Example

Polyline Length: 100, Call Type: B → Predicts ~25 minutes (1500 seconds)

## Notes

- Model saves automatically as `winnig_model_random_forest.pkl`
- Predictions are saved to history automatically
- Training takes 30-60 seconds on first run
