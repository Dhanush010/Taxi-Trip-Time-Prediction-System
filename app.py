import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import datetime
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Taxi Trip Time Prediction",
    page_icon="🚕",
    layout="wide"
)

@st.cache_data
def load_data():
    """Load and preprocess the taxi trajectory data"""
    try:
        # Try reading with different methods to handle malformed CSV
        try:
            # First try with standard settings
            df = pd.read_csv('train(1).csv', on_bad_lines='skip', engine='python', quoting=1)
        except TypeError:
            # Fallback for older pandas versions
            try:
                df = pd.read_csv('train(1).csv', error_bad_lines=False, warn_bad_lines=False, engine='python', quoting=1)
            except:
                # Last resort: read with minimal error handling
                df = pd.read_csv('train(1).csv', engine='python', quoting=1, sep=',', quotechar='"', escapechar='\\')
        
        if df is None or df.empty:
            st.error("Error: Could not load data or file is empty")
            return None
        
        # Drop rows with missing GPS data
        if 'MISSING_DATA' in df.columns:
            df = df[df['MISSING_DATA'] == False]
        
        # Drop rows with empty polyline
        if 'POLYLINE' in df.columns:
            df = df[df['POLYLINE'] != '[]']
            df = df[df['POLYLINE'].notna()]
            df = df[df['POLYLINE'] != '']
        
        if df.empty:
            st.error("Error: No valid data after filtering")
            return None
        
        # Calculate polyline length and trip time
        def safe_eval_polyline(x):
            try:
                if isinstance(x, str) and x.strip() and x != '[]':
                    parsed = eval(x)
                    if isinstance(parsed, list):
                        return len(parsed) - 1
                return 0
            except:
                return 0
        
        df['Polyline Length'] = df['POLYLINE'].apply(safe_eval_polyline)
        df = df[df['Polyline Length'] > 0]  # Remove rows with invalid polyline
        df['Trip Time(sec)'] = df['Polyline Length'] * 15
        
        # One-hot encode CALL_TYPE
        if 'CALL_TYPE' in df.columns:
            df = pd.get_dummies(df, columns=['CALL_TYPE'], prefix='CALL_TYPE')
        
        # Ensure all CALL_TYPE columns exist
        for col in ['CALL_TYPE_A', 'CALL_TYPE_B', 'CALL_TYPE_C']:
            if col not in df.columns:
                df[col] = 0
        
        # Drop duplicates
        df = df.drop_duplicates()
        
        if df.empty:
            st.error("Error: No valid data after processing")
            return None
        
        # Use subset of data for faster processing
        if len(df) > 50000:
            df = df.iloc[:50000]
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

@st.cache_resource
def train_model(df):
    """Train the Random Forest model"""
    try:
        # Prepare features and target
        X = df[['Polyline Length', 'CALL_TYPE_A', 'CALL_TYPE_B', 'CALL_TYPE_C']]
        y = df['Trip Time(sec)']
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42
        )
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=200,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Calculate metrics
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_rmse = sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        return model, scaler, {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None

def load_saved_model():
    """Load saved model if it exists"""
    model_path = 'winnig_model_random_forest.pkl'
    scaler_path = 'scaler.pkl'
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            return model, scaler
        except Exception as e:
            st.warning(f"Could not load saved model: {str(e)}")
            return None, None
    return None, None

def save_model(model, scaler):
    """Save model and scaler"""
    try:
        joblib.dump(model, 'winnig_model_random_forest.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        return True
    except Exception as e:
        st.error(f"Error saving model: {str(e)}")
        return False

def format_time(seconds):
    """Convert seconds to readable format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def save_prediction_history(prediction_data):
    """Save prediction to history file"""
    history_file = 'prediction_history.csv'
    try:
        if os.path.exists(history_file):
            history_df = pd.read_csv(history_file)
        else:
            history_df = pd.DataFrame(columns=['Timestamp', 'Polyline Length', 'Call Type', 
                                               'Predicted Time (sec)', 'Actual Time (sec)', 
                                               'Model Name'])
        
        new_row = pd.DataFrame([prediction_data])
        history_df = pd.concat([history_df, new_row], ignore_index=True)
        history_df.to_csv(history_file, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving prediction history: {str(e)}")
        return False

def load_prediction_history():
    """Load prediction history"""
    history_file = 'prediction_history.csv'
    if os.path.exists(history_file):
        try:
            return pd.read_csv(history_file)
        except:
            return pd.DataFrame()
    return pd.DataFrame()

def clear_prediction_history():
    """Clear prediction history"""
    history_file = 'prediction_history.csv'
    if os.path.exists(history_file):
        os.remove(history_file)
        return True
    return False

def get_available_models():
    """Get list of available saved models"""
    models_dir = 'saved_models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        return []
    
    models = []
    for file in os.listdir(models_dir):
        if file.endswith('_model.pkl'):
            model_name = file.replace('_model.pkl', '')
            if os.path.exists(os.path.join(models_dir, f"{model_name}_scaler.pkl")):
                models.append(model_name)
    return models

def save_model_with_name(model, scaler, model_name):
    """Save model with a specific name"""
    models_dir = 'saved_models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    try:
        model_path = os.path.join(models_dir, f"{model_name}_model.pkl")
        scaler_path = os.path.join(models_dir, f"{model_name}_scaler.pkl")
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        return True
    except Exception as e:
        st.error(f"Error saving model: {str(e)}")
        return False

def load_model_by_name(model_name):
    """Load model by name"""
    models_dir = 'saved_models'
    model_path = os.path.join(models_dir, f"{model_name}_model.pkl")
    scaler_path = os.path.join(models_dir, f"{model_name}_scaler.pkl")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            return model, scaler
        except Exception as e:
            st.warning(f"Could not load model {model_name}: {str(e)}")
            return None, None
    return None, None

def main():
    st.title("🚕 Taxi Trip Time Prediction")
    st.markdown("Predict the total travel time of a taxi trip based on trip characteristics")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page", ["Prediction", "Model Info", "Data Overview", 
                                               "Prediction History", "Model Comparison"])
    
    if page == "Prediction":
        prediction_page()
    elif page == "Model Info":
        model_info_page()
    elif page == "Data Overview":
        data_overview_page()
    elif page == "Prediction History":
        prediction_history_page()
    elif page == "Model Comparison":
        model_comparison_page()

def prediction_page():
    st.header("Make a Prediction")
    
    # Model selection in sidebar
    st.sidebar.subheader("Model Selection")
    available_models = get_available_models()
    default_model = "Default Model"
    
    if available_models:
        selected_model_name = st.sidebar.selectbox(
            "Select Model",
            options=[default_model] + available_models,
            help="Choose which trained model to use for predictions"
        )
    else:
        selected_model_name = default_model
    
    # Load or train model
    model = None
    scaler = None
    metrics = None
    current_model_name = selected_model_name
    
    if selected_model_name == default_model:
        # Try to load default saved model first
        model, scaler = load_saved_model()
        
        if model is None or scaler is None:
            st.info("No saved model found. Training a new model...")
            with st.spinner("Loading data and training model. This may take a moment..."):
                df = load_data()
                if df is not None:
                    model, scaler, metrics = train_model(df)
                    if model is not None:
                        # Save the model
                        if save_model(model, scaler):
                            st.success("Model trained and saved successfully!")
                        else:
                            st.warning("Model trained but could not be saved.")
        else:
            st.success("Loaded saved model successfully!")
    else:
        # Load selected model
        model, scaler = load_model_by_name(selected_model_name)
        if model is not None:
            st.success(f"Loaded model: {selected_model_name}")
        else:
            st.error(f"Could not load model: {selected_model_name}")
            return
    
    if model is None or scaler is None:
        st.error("Unable to load or train model. Please check your data file.")
        return
    
    # Input form
    st.subheader("Enter Trip Details")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        polyline_length = st.number_input(
            "Polyline Length (Number of GPS points - 1)",
            min_value=1,
            max_value=4000,
            value=50,
            help="Number of GPS coordinate pairs in the trip polyline minus 1"
        )
    
    with col2:
        call_type = st.selectbox(
            "Call Type",
            options=["A", "B", "C"],
            help="A: Dispatched from central, B: Demanded at taxi stand, C: Random street"
        )
    
    with col3:
        actual_time = st.number_input(
            "Actual Time (seconds) - Optional",
            min_value=0.0,
            value=0.0,
            help="Enter actual trip time if available (for comparison and history)"
        )
    
    # Prepare input features
    call_type_a = 1 if call_type == "A" else 0
    call_type_b = 1 if call_type == "B" else 0
    call_type_c = 1 if call_type == "C" else 0
    
    input_features = np.array([[polyline_length, call_type_a, call_type_b, call_type_c]])
    
    # Make prediction
    if st.button("Predict Trip Time", type="primary"):
        try:
            # Scale features
            input_scaled = scaler.transform(input_features)
            
            # Predict
            prediction = model.predict(input_scaled)[0]
            
            # Save to history
            prediction_data = {
                'Timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Polyline Length': polyline_length,
                'Call Type': call_type,
                'Predicted Time (sec)': round(prediction, 2),
                'Actual Time (sec)': actual_time if actual_time > 0 else None,
                'Model Name': current_model_name
            }
            save_prediction_history(prediction_data)
            
            # Display results
            st.success("Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Trip Time", format_time(prediction))
            
            with col2:
                st.metric("Predicted Time (seconds)", f"{prediction:.2f}")
            
            with col3:
                # Calculate estimated time based on polyline length
                estimated_time = polyline_length * 15
                st.metric("Estimated Time (15s per point)", format_time(estimated_time))
            
            # Show actual time comparison if provided
            if actual_time > 0:
                error = abs(prediction - actual_time)
                error_pct = (error / actual_time) * 100 if actual_time > 0 else 0
                st.info(f"""
                **Comparison with Actual:**
                - Actual Time: {format_time(actual_time)}
                - Prediction Error: {format_time(error)} ({error_pct:.2f}%)
                """)
            
            # Additional info
            st.info(f"""
            **Trip Details:**
            - Polyline Length: {polyline_length} points
            - Call Type: {call_type}
            - Predicted Duration: {format_time(prediction)}
            - Model Used: {current_model_name}
            """)
            
            # Show difference between model prediction and simple calculation
            simple_estimate = polyline_length * 15
            difference = prediction - simple_estimate
            if abs(difference) > 1:
                st.info(f"**Model Adjustment:** The ML model predicts {format_time(abs(difference))} {'more' if difference > 0 else 'less'} than the simple 15s-per-point estimate.")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    # Batch prediction section
    with st.expander("Batch Prediction (Predict Multiple Trips)"):
        st.write("Enter multiple trips to predict at once. Use one trip per line in CSV format: PolylineLength,CallType")
        batch_input = st.text_area(
            "Batch Input (CSV format)",
            placeholder="50,A\n75,B\n100,C",
            help="Format: PolylineLength,CallType (one per line)"
        )
        
        if st.button("Predict Batch", type="secondary"):
            if batch_input.strip():
                try:
                    lines = [line.strip() for line in batch_input.strip().split('\n') if line.strip()]
                    predictions_list = []
                    
                    for line in lines:
                        parts = line.split(',')
                        if len(parts) == 2:
                            try:
                                pl_length = int(parts[0].strip())
                                ct = parts[1].strip().upper()
                                
                                if ct in ['A', 'B', 'C']:
                                    ct_a = 1 if ct == 'A' else 0
                                    ct_b = 1 if ct == 'B' else 0
                                    ct_c = 1 if ct == 'C' else 0
                                    
                                    feat = np.array([[pl_length, ct_a, ct_b, ct_c]])
                                    feat_scaled = scaler.transform(feat)
                                    pred = model.predict(feat_scaled)[0]
                                    
                                    predictions_list.append({
                                        'Polyline Length': pl_length,
                                        'Call Type': ct,
                                        'Predicted Time (sec)': round(pred, 2),
                                        'Predicted Time': format_time(pred)
                                    })
                            except ValueError:
                                st.warning(f"Skipping invalid line: {line}")
                    
                    if predictions_list:
                        batch_df = pd.DataFrame(predictions_list)
                        st.success(f"Successfully predicted {len(predictions_list)} trips!")
                        st.dataframe(batch_df, use_container_width=True)
                        
                        # Download button
                        csv = batch_df.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv,
                            file_name="taxi_predictions.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error("No valid predictions generated. Please check your input format.")
                        
                except Exception as e:
                    st.error(f"Error processing batch predictions: {str(e)}")
            else:
                st.warning("Please enter some data for batch prediction.")

def model_info_page():
    st.header("Model Information")
    
    # Load or train model to get metrics
    model, scaler = load_saved_model()
    
    if model is None or scaler is None:
        st.info("Training model to get performance metrics...")
        with st.spinner("Loading data and training model..."):
            df = load_data()
            if df is not None:
                model, scaler, metrics = train_model(df)
            else:
                st.error("Could not load data.")
                return
    
    if model is None:
        st.error("Could not train model.")
        return
    
    # Get metrics if not already available
    if 'metrics' not in locals() or metrics is None:
        df = load_data()
        if df is not None:
            X = df[['Polyline Length', 'CALL_TYPE_A', 'CALL_TYPE_B', 'CALL_TYPE_C']]
            y = df['Trip Time(sec)']
            X_scaled = scaler.transform(X)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42
            )
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            metrics = {
                'train_rmse': sqrt(mean_squared_error(y_train, y_train_pred)),
                'test_rmse': sqrt(mean_squared_error(y_test, y_test_pred)),
                'train_r2': r2_score(y_train, y_train_pred),
                'test_r2': r2_score(y_test, y_test_pred)
            }
    
    st.subheader("Model Architecture")
    st.write("**Algorithm:** Random Forest Regressor")
    st.write("**Parameters:**")
    st.write(f"- Number of Estimators: {model.n_estimators}")
    st.write(f"- Min Samples Split: {model.min_samples_split}")
    st.write(f"- Min Samples Leaf: {model.min_samples_leaf}")
    
    st.subheader("Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Training Set:**")
        st.metric("RMSE", f"{metrics['train_rmse']:.2f} seconds")
        st.metric("R² Score", f"{metrics['train_r2']:.4f}")
    
    with col2:
        st.write("**Test Set:**")
        st.metric("RMSE", f"{metrics['test_rmse']:.2f} seconds")
        st.metric("R² Score", f"{metrics['test_r2']:.4f}")
    
    st.subheader("Features Used")
    st.write("The model uses the following features:")
    st.write("1. **Polyline Length**: Number of GPS coordinate pairs minus 1")
    st.write("2. **CALL_TYPE_A**: Binary indicator for trips dispatched from central")
    st.write("3. **CALL_TYPE_B**: Binary indicator for trips demanded at taxi stand")
    st.write("4. **CALL_TYPE_C**: Binary indicator for trips demanded on random street")
    
    # Feature importance
    st.subheader("Feature Importance")
    try:
        feature_names = ['Polyline Length', 'CALL_TYPE_A', 'CALL_TYPE_B', 'CALL_TYPE_C']
        importances = model.feature_importances_
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        st.bar_chart(importance_df.set_index('Feature'))
        st.dataframe(importance_df, use_container_width=True)
    except Exception as e:
        st.info("Feature importance not available for this model type.")
    
    st.subheader("How It Works")
    st.write("""
    The model predicts trip time based on:
    - The number of GPS points in the trip trajectory (each point represents 15 seconds)
    - The method used to request the taxi (central dispatch, taxi stand, or random street)
    
    The Random Forest algorithm combines multiple decision trees to make accurate predictions.
    """)
    
    # Model retraining section
    st.subheader("Model Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Retrain Model", type="primary", use_container_width=True):
            with st.spinner("Retraining model with current data..."):
                # Clear cache to reload fresh data
                load_data.clear()
                train_model.clear()
                df = load_data()
                if df is not None:
                    new_model, new_scaler, new_metrics = train_model(df)
                    if new_model is not None:
                        if save_model(new_model, new_scaler):
                            st.success("Model retrained and saved successfully!")
                            st.rerun()
                        else:
                            st.error("Model retrained but could not be saved.")
                    else:
                        st.error("Failed to retrain model.")
                else:
                    st.error("Could not load data for retraining.")
    
    with col2:
        model_name_input = st.text_input("Save Model As (optional)", placeholder="e.g., model_v2")
        if st.button("💾 Save Model with Name", use_container_width=True):
            if model_name_input.strip():
                if save_model_with_name(model, scaler, model_name_input.strip()):
                    st.success(f"Model saved as: {model_name_input.strip()}")
                    st.rerun()
                else:
                    st.error("Failed to save model.")
            else:
                st.warning("Please enter a model name.")
    
    # List available models
    available_models = get_available_models()
    if available_models:
        st.subheader("Available Saved Models")
        st.write(f"Found {len(available_models)} saved model(s):")
        for model_name in available_models:
            st.write(f"- {model_name}")

def data_overview_page():
    st.header("Data Overview")
    
    with st.spinner("Loading data..."):
        df = load_data()
    
    if df is None:
        st.error("Could not load data. Please ensure 'train(1).csv' is in the project directory.")
        return
    
    st.subheader("Dataset Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trips", f"{len(df):,}")
    
    with col2:
        avg_time = df['Trip Time(sec)'].mean()
        st.metric("Average Trip Time", format_time(avg_time))
    
    with col3:
        min_time = df['Trip Time(sec)'].min()
        st.metric("Shortest Trip", format_time(min_time))
    
    with col4:
        max_time = df['Trip Time(sec)'].max()
        st.metric("Longest Trip", format_time(max_time))
    
    st.subheader("Call Type Distribution")
    call_type_counts = {}
    if 'CALL_TYPE_A' in df.columns:
        call_type_counts['A (Central Dispatch)'] = df['CALL_TYPE_A'].sum()
    if 'CALL_TYPE_B' in df.columns:
        call_type_counts['B (Taxi Stand)'] = df['CALL_TYPE_B'].sum()
    if 'CALL_TYPE_C' in df.columns:
        call_type_counts['C (Random Street)'] = df['CALL_TYPE_C'].sum()
    
    if call_type_counts:
        st.bar_chart(call_type_counts)
    
    st.subheader("Trip Time Distribution")
    st.write("Distribution of trip times in the dataset:")
    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['Trip Time(sec)'], bins=50, edgecolor='black')
    ax.set_xlabel('Trip Time (seconds)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Trip Times')
    st.pyplot(fig)
    
    st.subheader("Sample Data")
    display_cols = ['Polyline Length', 'CALL_TYPE_A', 'CALL_TYPE_B', 'CALL_TYPE_C', 'Trip Time(sec)']
    available_cols = [col for col in display_cols if col in df.columns]
    st.dataframe(df[available_cols].head(10), use_container_width=True)
    
    # Statistics table
    st.subheader("Detailed Statistics")
    if 'Trip Time(sec)' in df.columns:
        stats_df = pd.DataFrame({
            'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', '25th Percentile', '75th Percentile'],
            'Trip Time (seconds)': [
                df['Trip Time(sec)'].mean(),
                df['Trip Time(sec)'].median(),
                df['Trip Time(sec)'].std(),
                df['Trip Time(sec)'].min(),
                df['Trip Time(sec)'].max(),
                df['Trip Time(sec)'].quantile(0.25),
                df['Trip Time(sec)'].quantile(0.75)
            ]
        })
        stats_df['Trip Time (readable)'] = stats_df['Trip Time (seconds)'].apply(format_time)
        st.dataframe(stats_df, use_container_width=True)

def prediction_history_page():
    st.header("Prediction History")
    st.write("View and manage your prediction history")
    
    history_df = load_prediction_history()
    
    if history_df.empty:
        st.info("No prediction history found. Make some predictions to see them here!")
        return
    
    # Statistics
    st.subheader("History Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Predictions", len(history_df))
    
    with col2:
        if 'Actual Time (sec)' in history_df.columns:
            with_actual = history_df[history_df['Actual Time (sec)'].notna()]
            st.metric("With Actual Times", len(with_actual))
        else:
            st.metric("With Actual Times", 0)
    
    with col3:
        if 'Actual Time (sec)' in history_df.columns:
            with_actual = history_df[history_df['Actual Time (sec)'].notna()]
            if len(with_actual) > 0:
                avg_error = abs(with_actual['Predicted Time (sec)'] - with_actual['Actual Time (sec)']).mean()
                st.metric("Average Error", format_time(avg_error))
            else:
                st.metric("Average Error", "N/A")
        else:
            st.metric("Average Error", "N/A")
    
    with col4:
        if 'Model Name' in history_df.columns:
            unique_models = history_df['Model Name'].nunique()
            st.metric("Models Used", unique_models)
        else:
            st.metric("Models Used", 1)
    
    # Filter options
    st.subheader("Filter History")
    col1, col2 = st.columns(2)
    
    with col1:
        if 'Model Name' in history_df.columns:
            selected_models = st.multiselect(
                "Filter by Model",
                options=history_df['Model Name'].unique().tolist(),
                default=history_df['Model Name'].unique().tolist()
            )
            if selected_models:
                filtered_df = history_df[history_df['Model Name'].isin(selected_models)]
            else:
                filtered_df = history_df
        else:
            filtered_df = history_df
    
    with col2:
        if 'Call Type' in history_df.columns:
            selected_call_types = st.multiselect(
                "Filter by Call Type",
                options=history_df['Call Type'].unique().tolist(),
                default=history_df['Call Type'].unique().tolist()
            )
            if selected_call_types:
                filtered_df = filtered_df[filtered_df['Call Type'].isin(selected_call_types)]
    
    # Display history
    st.subheader("Prediction History Table")
    st.dataframe(filtered_df, use_container_width=True)
    
    # Visualizations
    if 'Actual Time (sec)' in filtered_df.columns:
        with_actual = filtered_df[filtered_df['Actual Time (sec)'].notna()]
        if len(with_actual) > 0:
            st.subheader("Predictions vs Actual")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(with_actual['Actual Time (sec)'], with_actual['Predicted Time (sec)'], alpha=0.6)
            
            # Perfect prediction line
            max_val = max(with_actual['Actual Time (sec)'].max(), with_actual['Predicted Time (sec)'].max())
            min_val = min(with_actual['Actual Time (sec)'].min(), with_actual['Predicted Time (sec)'].min())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
            
            ax.set_xlabel('Actual Time (seconds)')
            ax.set_ylabel('Predicted Time (seconds)')
            ax.set_title('Predictions vs Actual Trip Times')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Error distribution
            errors = with_actual['Predicted Time (sec)'] - with_actual['Actual Time (sec)']
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.hist(errors, bins=30, edgecolor='black', alpha=0.7)
            ax2.axvline(x=0, color='r', linestyle='--', label='Zero Error')
            ax2.set_xlabel('Prediction Error (seconds)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distribution of Prediction Errors')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)
    
    # Download history
    st.subheader("Export History")
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download History as CSV",
        data=csv,
        file_name=f"prediction_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
    # Clear history
    if st.button("🗑️ Clear All History", type="secondary"):
        if clear_prediction_history():
            st.success("History cleared successfully!")
            st.rerun()
        else:
            st.error("Failed to clear history.")

def model_comparison_page():
    st.header("Model Comparison & Visualization")
    st.write("Compare model predictions with actual values and visualize performance")
    
    # Load data for comparison
    df = load_data()
    if df is None:
        st.error("Could not load data. Please ensure 'train(1).csv' is in the project directory.")
        return
    
    # Get available models
    available_models = get_available_models()
    models_to_compare = ["Default Model"] + available_models
    
    if len(models_to_compare) < 2:
        st.warning("You need at least 2 models to compare. Train and save multiple models first.")
        return
    
    st.subheader("Select Models to Compare")
    selected_models = st.multiselect(
        "Choose models",
        options=models_to_compare,
        default=models_to_compare[:min(3, len(models_to_compare))]
    )
    
    if len(selected_models) < 2:
        st.info("Please select at least 2 models to compare.")
        return
    
    # Load models and make predictions on test data
    st.subheader("Model Performance Comparison")
    
    with st.spinner("Loading models and generating predictions..."):
        # Prepare test data
        X = df[['Polyline Length', 'CALL_TYPE_A', 'CALL_TYPE_B', 'CALL_TYPE_C']]
        y = df['Trip Time(sec)']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model_results = {}
        
        for model_name in selected_models:
            if model_name == "Default Model":
                model, model_scaler = load_saved_model()
                if model is None:
                    # Train default model
                    model = RandomForestRegressor(n_estimators=200, min_samples_split=2, 
                                                  min_samples_leaf=1, random_state=42, n_jobs=-1)
                    model.fit(X_train_scaled, y_train)
            else:
                model, model_scaler = load_model_by_name(model_name)
                if model is None:
                    continue
            
            # Make predictions
            if model_name == "Default Model":
                y_pred = model.predict(X_test_scaled)
            else:
                # Transform with model's scaler
                X_test_model_scaled = model_scaler.transform(X_test)
                y_pred = model.predict(X_test_model_scaled)
            
            # Calculate metrics
            rmse = sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mae = np.mean(np.abs(y_test - y_pred))
            
            model_results[model_name] = {
                'predictions': y_pred,
                'rmse': rmse,
                'r2': r2,
                'mae': mae
            }
    
    # Display comparison table
    comparison_df = pd.DataFrame({
        'Model': list(model_results.keys()),
        'RMSE (seconds)': [model_results[m]['rmse'] for m in model_results.keys()],
        'R² Score': [model_results[m]['r2'] for m in model_results.keys()],
        'MAE (seconds)': [model_results[m]['mae'] for m in model_results.keys()]
    }).sort_values('RMSE (seconds)')
    
    st.dataframe(comparison_df, use_container_width=True)
    
    # Visualizations
    st.subheader("Visualizations")
    
    # RMSE comparison
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.barh(comparison_df['Model'], comparison_df['RMSE (seconds)'])
    ax1.set_xlabel('RMSE (seconds)')
    ax1.set_title('Model RMSE Comparison (Lower is Better)')
    ax1.grid(True, alpha=0.3, axis='x')
    st.pyplot(fig1)
    
    # R² comparison
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.barh(comparison_df['Model'], comparison_df['R² Score'])
    ax2.set_xlabel('R² Score')
    ax2.set_title('Model R² Score Comparison (Higher is Better)')
    ax2.set_xlim([0, 1])
    ax2.grid(True, alpha=0.3, axis='x')
    st.pyplot(fig2)
    
    # Predictions vs Actual for best model
    best_model = comparison_df.iloc[0]['Model']
    st.subheader(f"Predictions vs Actual - Best Model: {best_model}")
    
    y_pred_best = model_results[best_model]['predictions']
    
    # Sample for visualization (if too many points)
    sample_size = min(1000, len(y_test))
    indices = np.random.choice(len(y_test), sample_size, replace=False)
    y_test_sample = y_test.iloc[indices]
    y_pred_sample = y_pred_best[indices]
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.scatter(y_test_sample, y_pred_sample, alpha=0.5)
    
    max_val = max(y_test_sample.max(), y_pred_sample.max())
    min_val = min(y_test_sample.min(), y_pred_sample.min())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction', linewidth=2)
    
    ax3.set_xlabel('Actual Time (seconds)')
    ax3.set_ylabel('Predicted Time (seconds)')
    ax3.set_title(f'Predictions vs Actual - {best_model}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)
    
    # Error distribution for best model
    errors_best = y_pred_best - y_test
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.hist(errors_best, bins=50, edgecolor='black', alpha=0.7)
    ax4.axvline(x=0, color='r', linestyle='--', label='Zero Error', linewidth=2)
    ax4.set_xlabel('Prediction Error (seconds)')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'Error Distribution - {best_model}')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    st.pyplot(fig4)

if __name__ == "__main__":
    main()

