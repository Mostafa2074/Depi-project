import streamlit as st
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
from mlflow.tracking import MlflowClient
from statsmodels.tsa.arima.model import ARIMA
import lightgbm as lgb
from model_wrappers import ARIMAModelWrapper, LightGBMModelWrapper
from prediction_logger import PredictionLogger
import os

def setup_mlflow_training():
    """Setup MLflow training interface with session context"""
    st.title("🔬 MLflow Model Training & Tracking")
    
    # Session context header
    st.info("""
    **Session-Based Training**: 
    - Models are saved only for this browser session
    - Perfect for testing and demonstrations
    - Automatically cleared when you leave
    - After training, models are automatically registered for use in Forecast Engine
    """)
    
    # Initialize MLflow client
    client = MlflowClient()
    
    # Experiment setup
    experiment_name = "best_models"
    mlflow.set_experiment(experiment_name)
    
    st.sidebar.header("Training Configuration")
    
    # Model selection - Only LightGBM and ARIMA
    model_choice = st.sidebar.selectbox(
        "Select Model to Train",
        ["LightGBM", "ARIMA"]
    )
    
    # Data loading
    from data_loader import load_data
    train, min_date, max_date, sort_state, prophet_df = load_data()
    
    if prophet_df.empty:
        st.error("No data available for training")
        return
    
    st.sidebar.header("Training Data Info")
    st.sidebar.write(f"Data range: {prophet_df['ds'].min().date()} to {prophet_df['ds'].max().date()}")
    st.sidebar.write(f"Total records: {len(prophet_df)}")
    
    # Show model descriptions
    with st.expander("📋 Model Descriptions"):
        st.write("""
        **LightGBM**: 
        - Gradient boosting framework
        - Fast training and high accuracy
        - Handles large datasets efficiently
        - Recommended for most use cases
        
        **ARIMA**:
        - Classical time series model
        - Good for data with clear trends/seasonality
        - Interpretable parameters
        """)
    
    # Session management
    add_session_cleanup()
    
    if st.sidebar.button("🚀 Train Selected Model", type="primary"):
        with st.spinner(f"Training {model_choice} model..."):
            try:
                if model_choice == "ARIMA":
                    train_arima(prophet_df, client)
                elif model_choice == "LightGBM":
                    train_lightgbm(prophet_df, client)
                
                st.success(f"✅ {model_choice} training completed!")
                st.success("🎯 Model automatically registered for use in Forecast Engine!")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Training failed: {e}")

# ... (keep the rest of training.py functions the same as before)

def train_prophet(prophet_df, client):
    """Prophet training disabled due to dependency issues"""
    st.error("🚫 Prophet training is temporarily unavailable")
    st.info("""
    **Why Prophet is disabled:**
    - Prophet dependencies are incompatible with Streamlit Cloud's Python 3.13
    - Use LightGBM or ARIMA models instead
    - LightGBM provides excellent performance for time series forecasting
    """)
    return

def train_arima(prophet_df, client):
    """Train and log ARIMA model"""
    with mlflow.start_run(run_name="arima_model"):
        try:
            # Prepare data
            ts_data = prophet_df.set_index('ds')['y']
            
            # Train ARIMA model
            model = ARIMA(ts_data, order=(5,1,0))
            fitted_model = model.fit()
            
            # Log parameters
            mlflow.log_params({
                "model_type": "arima",
                "order": "(5,1,0)",
                "method": "css"
            })
            
            # Calculate validation metrics
            if len(ts_data) > 30:
                train_data = ts_data[:-30]
                test_data = ts_data[-30:]
                
                # Retrain on subset for validation
                val_model = ARIMA(train_data, order=(5,1,0))
                val_fitted = val_model.fit()
                forecast = val_fitted.forecast(steps=30)
                
                mae = np.mean(np.abs(test_data.values - forecast))
                rmse = np.sqrt(np.mean((test_data.values - forecast)**2))
                
                mlflow.log_metrics({
                    "validation_mae": mae,
                    "validation_rmse": rmse
                })
            
            # Log model
            wrapped_model = ARIMAModelWrapper(fitted_model)
            mlflow.pyfunc.log_model(
                "arima_model",
                python_model=wrapped_model,
                registered_model_name="BestForecastModels"
            )
            
            # Set tags
            mlflow.set_tags({
                "model_type": "arima",
                "framework": "statsmodels",
                "task": "time_series_forecasting"
            })
            
            st.success("✅ ARIMA model trained and registered successfully!")
            
        except Exception as e:
            st.error(f"❌ ARIMA training failed: {e}")
            raise

def train_lightgbm(prophet_df, client):
    """Train and log LightGBM model"""
    with mlflow.start_run(run_name="lightgbm_model"):
        try:
            # Prepare features
            df = prophet_df.copy()
            df['ds'] = pd.to_datetime(df['ds'])
            df['year'] = df['ds'].dt.year
            df['month'] = df['ds'].dt.month
            df['day'] = df['ds'].dt.day
            df['dayofweek'] = df['ds'].dt.dayofweek
            df['quarter'] = df['ds'].dt.quarter
            df['dayofyear'] = df['ds'].dt.dayofyear
            df['weekofyear'] = df['ds'].dt.isocalendar().week
            
            # Split features and target
            X = df.drop(['ds', 'y'], axis=1)
            y = df['y']
            
            # Train/test split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train model
            model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Log parameters
            mlflow.log_params({
                "model_type": "lightgbm",
                "n_estimators": 100,
                "learning_rate": 0.1,
                "random_state": 42
            })
            
            # Calculate metrics
            y_pred = model.predict(X_test)
            mae = np.mean(np.abs(y_test - y_pred))
            rmse = np.sqrt(np.mean((y_test - y_pred)**2))
            
            mlflow.log_metrics({
                "validation_mae": mae,
                "validation_rmse": rmse
            })
            
            # Log model
            wrapped_model = LightGBMModelWrapper(model)
            mlflow.pyfunc.log_model(
                "lightgbm_model",
                python_model=wrapped_model,
                registered_model_name="BestForecastModels"
            )
            
            # Set tags
            mlflow.set_tags({
                "model_type": "lightgbm",
                "framework": "lightgbm",
                "task": "time_series_forecasting"
            })
            
            st.success("✅ LightGBM model trained and registered successfully!")
            
        except Exception as e:
            st.error(f"❌ LightGBM training failed: {e}")
            raise

def add_session_cleanup():
    """Add option to manually clear session models"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 Session Management")
    
    if st.sidebar.button("🧹 Clear Session Models", type="secondary"):
        try:
            reset_mlflow_completely()
            st.sidebar.success("Session models cleared! Refresh to see changes.")
        except Exception as e:
            st.sidebar.error(f"Clear failed: {e}")

def reset_mlflow_completely():
    """Completely reset MLflow - use with caution!"""
    try:
        import shutil
        
        # Delete MLflow artifacts
        if os.path.exists('mlruns'):
            shutil.rmtree('mlruns')
        
        if os.path.exists('mlflow.db'):
            os.remove('mlflow.db')
        
        # Recreate necessary directories
        os.makedirs('mlruns', exist_ok=True)
        
        return True
        
    except Exception as e:
        st.error(f"Error resetting MLflow: {e}")
        return False

