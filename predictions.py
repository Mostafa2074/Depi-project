import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import mlflow.pyfunc
import streamlit as st

def batch_predict_mlflow(model, model_type, forecast_end_date, prophet_df=None):
    """Generate batch predictions using MLflow model"""
    try:
        if model_type == "prophet":
            return _prophet_batch_predict(model, forecast_end_date, prophet_df)
        elif model_type == "arima":
            return _arima_batch_predict(model, forecast_end_date, prophet_df)
        elif model_type == "lightgbm":
            return _lightgbm_batch_predict(model, forecast_end_date, prophet_df)
        else:
            st.error(f"Unsupported model type: {model_type}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Batch prediction error: {e}")
        return pd.DataFrame()

def _prophet_batch_predict(model, forecast_end_date, prophet_df):
    """Prophet batch prediction"""
    if prophet_df is None or prophet_df.empty:
        st.error("Prophet model requires historical data")
        return pd.DataFrame()
    
    # Calculate periods needed
    last_date = prophet_df['ds'].max()
    end_date = pd.to_datetime(forecast_end_date)
    periods = (end_date - last_date).days
    
    if periods <= 0:
        st.error("Forecast end date must be after last historical date")
        return pd.DataFrame()
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    # Filter only future predictions
    future_forecast = forecast[forecast['ds'] > last_date][['ds', 'yhat']]
    future_forecast = future_forecast.rename(columns={'ds': 'date', 'yhat': 'prediction'})
    
    return future_forecast

def _arima_batch_predict(model, forecast_end_date, prophet_df):
    """ARIMA batch prediction"""
    if prophet_df is None or prophet_df.empty:
        st.error("ARIMA model requires historical data")
        return pd.DataFrame()
    
    last_date = prophet_df['ds'].max()
    end_date = pd.to_datetime(forecast_end_date)
    periods = (end_date - last_date).days
    
    if periods <= 0:
        st.error("Forecast end date must be after last historical date")
        return pd.DataFrame()
    
    # Generate predictions
    forecast = model.forecast(steps=periods)
    
    # Create date range
    dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
    
    return pd.DataFrame({
        'date': dates,
        'prediction': forecast
    })

def _lightgbm_batch_predict(model, forecast_end_date, prophet_df):
    """LightGBM batch prediction"""
    if prophet_df is None or prophet_df.empty:
        st.error("LightGBM model requires historical data")
        return pd.DataFrame()
    
    last_date = prophet_df['ds'].max()
    end_date = pd.to_datetime(forecast_end_date)
    periods = (end_date - last_date).days
    
    if periods <= 0:
        st.error("Forecast end date must be after last historical date")
        return pd.DataFrame()
    
    # Create future dates
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
    
    # Prepare features
    future_df = pd.DataFrame({'ds': future_dates})
    future_df['year'] = future_df['ds'].dt.year
    future_df['month'] = future_df['ds'].dt.month
    future_df['day'] = future_df['ds'].dt.day
    future_df['dayofweek'] = future_df['ds'].dt.dayofweek
    future_df['quarter'] = future_df['ds'].dt.quarter
    future_df['dayofyear'] = future_df['ds'].dt.dayofyear
    future_df['weekofyear'] = future_df['ds'].dt.isocalendar().week
    
    # Make predictions
    predictions = model.predict(future_df.drop('ds', axis=1))
    
    return pd.DataFrame({
        'date': future_dates,
        'prediction': predictions
    })

def real_time_predict_mlflow(model, model_type, target_date, prophet_df=None):
    """Generate real-time prediction for a specific date"""
    try:
        if model_type == "prophet":
            future_date_df = pd.DataFrame({'ds': [target_date]})
            prediction = model.predict(future_date_df)['yhat'].iloc[0]
        elif model_type == "arima":
            # For ARIMA, we need to specify this is 1 step ahead from last data point
            prediction = model.forecast(steps=1)[0]
        elif model_type == "lightgbm":
            # Prepare features for the target date
            features = pd.DataFrame({
                'year': [target_date.year],
                'month': [target_date.month],
                'day': [target_date.day],
                'dayofweek': [target_date.weekday()],
                'quarter': [(target_date.month-1)//3 + 1],
                'dayofyear': [target_date.timetuple().tm_yday],
                'weekofyear': [target_date.isocalendar()[1]]
            })
            prediction = model.predict(features)[0]
        else:
            return {'error': f'Unsupported model type: {model_type}'}
        
        return {
            'date': target_date.strftime('%Y-%m-%d'),
            'prediction': float(prediction),
            'model_type': model_type,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {'error': str(e)}

def standardize_forecast_data(forecast_data, model_type):
    """Standardize forecast data to common format"""
    try:
        if forecast_data.empty:
            return pd.DataFrame()
        
        result = forecast_data.copy()
        
        # Ensure we have the right column names
        if 'ds' in result.columns and 'date' not in result.columns:
            result = result.rename(columns={'ds': 'date'})
        
        if 'yhat' in result.columns and 'prediction' not in result.columns:
            result = result.rename(columns={'yhat': 'prediction'})
        
        # Ensure date is datetime
        if 'date' in result.columns:
            result['date'] = pd.to_datetime(result['date'])
        
        # Select only necessary columns
        required_cols = ['date', 'prediction']
        available_cols = [col for col in required_cols if col in result.columns]
        
        return result[available_cols]
        
    except Exception as e:
        st.error(f"Error standardizing forecast data: {e}")
        return pd.DataFrame()