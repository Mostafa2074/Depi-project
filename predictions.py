# [file name]: predictions.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import Dict, Optional
from prediction_logger import PredictionLogger

# Initialize the logger
prediction_logger = PredictionLogger()

def batch_predict_mlflow(model, model_type, end_date: date, prophet_df=None) -> pd.DataFrame:
    """Generate batch predictions until a specific end date using MLflow model."""
    try:
        st.info(f"🎯 Starting batch prediction with {model_type} model...")
        
        # Calculate periods based on end date
        if prophet_df is not None and not prophet_df.empty:
            last_data_date = prophet_df['ds'].max()
            st.info(f"📅 Last historical data date: {last_data_date.date()}")
        else:
            last_data_date = pd.Timestamp(datetime.now().date())
            st.warning("⚠️ No historical data provided, using current date as reference")
        
        end_date_dt = pd.Timestamp(end_date)
        periods = (end_date_dt - last_data_date).days
        
        if periods <= 0:
            st.error(f"❌ End date {end_date} must be after the last data date {last_data_date.date()}")
            return pd.DataFrame()
        
        st.success(f"🔮 Generating {periods} days of predictions from {last_data_date.date()} to {end_date}")
        
        if model_type == "prophet":
            # For Prophet models, create future dataframe with ds column only
            future_dates = pd.date_range(start=last_data_date + timedelta(days=1), periods=periods, freq='D')
            future = pd.DataFrame({'ds': future_dates})
            
            st.info("📊 Using Prophet model for forecasting...")
            
            try:
                forecast = model.predict(future)
                
                # Create the result dataframe
                result_df = pd.DataFrame({
                    'date': forecast['ds'],
                    'prediction': forecast['yhat']
                })
                
                st.info(f"✅ Prophet forecast completed: {len(result_df)} predictions generated")
                
                # Log the predictions - ONLY TO MLFLOW
                log_count = prediction_logger.log_batch_prediction(
                    forecast_data=result_df,
                    actual_data=prophet_df,
                    model_type=model_type,
                    forecast_end_date=end_date
                )
                
                st.info(f"📊 Logged {log_count} predictions to MLflow")
                return result_df
                
            except Exception as e:
                st.error(f"❌ Prophet prediction failed: {e}")
                return pd.DataFrame()
            
        elif model_type == "arima":
            # For ARIMA models - create proper input format
            future_dates = pd.date_range(start=last_data_date + timedelta(days=1), periods=periods, freq='D')
            
            # Create input with date information that matches the wrapper's expected schema
            future_df = pd.DataFrame({
                'ds': future_dates,
                'year': future_dates.year.astype('int32'),
                'month': future_dates.month.astype('int32'),
                'day': future_dates.day.astype('int32')
            })
            
            st.info("📊 Using ARIMA model for forecasting...")
            
            try:
                # Get predictions
                predictions_df = model.predict(future_df)
                
                # Create forecast DataFrame
                result_df = pd.DataFrame({
                    'date': future_dates,
                    'prediction': predictions_df['prediction'].values if 'prediction' in predictions_df.columns else predictions_df.iloc[:, 0].values
                })
                
                st.info(f"✅ ARIMA forecast completed: {len(result_df)} predictions generated")
                
                # Log the predictions - ONLY TO MLFLOW
                log_count = prediction_logger.log_batch_prediction(
                    forecast_data=result_df,
                    actual_data=prophet_df,
                    model_type=model_type,
                    forecast_end_date=end_date
                )
                
                st.info(f"📊 Logged {log_count} predictions to MLflow")
                return result_df
                
            except Exception as e:
                st.error(f"❌ ARIMA prediction failed: {e}")
                return pd.DataFrame()
            
        elif model_type == "lightgbm":
            # For LightGBM models - create feature dataframe with ALL required features
            future_dates = pd.date_range(start=last_data_date + timedelta(days=1), periods=periods, freq='D')
            
            # Create input with ALL feature columns that the model expects
            future_df = pd.DataFrame({
                'ds': future_dates,
                'year': future_dates.year.astype('float64'),
                'month': future_dates.month.astype('float64'),
                'day': future_dates.day.astype('float64'),
                'dayofweek': future_dates.dayofweek.astype('float64'),
                'quarter': future_dates.quarter.astype('float64'),
                'dayofyear': future_dates.dayofyear.astype('float64'),
                'weekofyear': future_dates.isocalendar().week.astype('float64')
            })
            
            st.info("📊 Using LightGBM model for forecasting...")
            
            try:
                # Get predictions
                predictions_df = model.predict(future_df)
                
                result_df = pd.DataFrame({
                    'date': future_dates,
                    'prediction': predictions_df['prediction'].values if 'prediction' in predictions_df.columns else predictions_df.iloc[:, 0].values
                })
                
                st.info(f"✅ LightGBM forecast completed: {len(result_df)} predictions generated")
                
                # Log the predictions - ONLY TO MLFLOW
                log_count = prediction_logger.log_batch_prediction(
                    forecast_data=result_df,
                    actual_data=prophet_df,
                    model_type=model_type,
                    forecast_end_date=end_date
                )
                
                st.info(f"📊 Logged {log_count} predictions to MLflow")
                return result_df
                
            except Exception as e:
                st.error(f"❌ LightGBM prediction failed: {e}")
                return pd.DataFrame()
            
        else:
            st.error(f"❌ Unsupported model type: {model_type}")
            return pd.DataFrame()
            
    except Exception as e:
        st.error(f"❌ Error in batch prediction: {e}")
        import traceback
        st.error(f"🔍 Detailed error: {traceback.format_exc()}")
        return pd.DataFrame()
    
def real_time_predict_mlflow(model, model_type, target_date: datetime, context_data: Optional[Dict] = None, prophet_df=None) -> Dict:
    """Generate prediction for a single specific date using MLflow model."""
    try:
        target_date_dt = pd.to_datetime(target_date)
        st.info(f"🎯 Generating real-time prediction for {target_date_dt.date()} using {model_type} model")
        
        if model_type == "prophet":
            # For Prophet - simple date input
            future_df = pd.DataFrame({'ds': [target_date_dt]})
            
            try:
                forecast = model.predict(future_df)
                prediction_value = float(forecast['yhat'].iloc[0])
                
                result = {
                    'date': target_date,
                    'prediction': prediction_value,
                    'model_type': model_type,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success'
                }
                
                st.success(f"✅ Real-time prediction: ${prediction_value:,.0f}")
                return result
                
            except Exception as e:
                st.error(f"❌ Prophet real-time prediction failed: {e}")
                return {
                    'date': target_date,
                    'prediction': None,
                    'model_type': model_type,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                    'status': 'error'
                }
            
        elif model_type == "arima":
            # For ARIMA, we need to generate a sequence of predictions and take the last one
            if prophet_df is not None and not prophet_df.empty:
                last_data_date = prophet_df['ds'].max()
            else:
                last_data_date = pd.Timestamp(datetime.now().date())
            
            days_ahead = (target_date_dt - last_data_date).days
            
            if days_ahead <= 0:
                error_msg = f"Target date {target_date_dt.date()} must be after the last training data date {last_data_date.date()}"
                st.error(f"❌ {error_msg}")
                return {
                    'date': target_date,
                    'prediction': 0,
                    'model_type': model_type,
                    'error': error_msg,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'error'
                }
            
            # Create input for the required number of periods
            future_dates = pd.date_range(start=last_data_date + timedelta(days=1), periods=days_ahead, freq='D')
            
            future_df = pd.DataFrame({
                'ds': future_dates,
                'year': future_dates.year.astype('int32'),
                'month': future_dates.month.astype('int32'),
                'day': future_dates.day.astype('int32')
            })
            
            try:
                # Get predictions for all days up to the target date
                predictions_df = model.predict(future_df)
                
                # Take the last prediction (the one for our target date)
                if 'prediction' in predictions_df.columns:
                    prediction_value = float(predictions_df['prediction'].iloc[-1])
                else:
                    prediction_value = float(predictions_df.iloc[-1, 0])
                
                result = {
                    'date': target_date,
                    'prediction': prediction_value,
                    'model_type': model_type,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success'
                }
                
                st.success(f"✅ Real-time prediction: ${prediction_value:,.0f}")
                return result
                
            except Exception as e:
                st.error(f"❌ ARIMA real-time prediction failed: {e}")
                return {
                    'date': target_date,
                    'prediction': None,
                    'model_type': model_type,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                    'status': 'error'
                }
            
        elif model_type == "lightgbm":
            # LightGBM single prediction - create ALL required features
            lightgbm_input = pd.DataFrame({
                'ds': [target_date_dt],
                'year': np.array([target_date_dt.year], dtype='float64'),
                'month': np.array([target_date_dt.month], dtype='float64'),
                'day': np.array([target_date_dt.day], dtype='float64'),
                'dayofweek': np.array([target_date_dt.dayofweek], dtype='float64'),
                'quarter': np.array([target_date_dt.quarter], dtype='float64'),
                'dayofyear': np.array([target_date_dt.dayofyear], dtype='float64'),
                'weekofyear': np.array([target_date_dt.isocalendar().week], dtype='float64')
            })
            
            try:
                prediction_df = model.predict(lightgbm_input)
                prediction_value = float(prediction_df['prediction'].iloc[0])
                
                result = {
                    'date': target_date,
                    'prediction': prediction_value,
                    'model_type': model_type,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success'
                }
                
                st.success(f"✅ Real-time prediction: ${prediction_value:,.0f}")
                return result
                
            except Exception as e:
                st.error(f"❌ LightGBM real-time prediction failed: {e}")
                return {
                    'date': target_date,
                    'prediction': None,
                    'model_type': model_type,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                    'status': 'error'
                }
            
        else:
            error_msg = f"Unsupported model type: {model_type}"
            st.error(f"❌ {error_msg}")
            return {
                'date': target_date,
                'prediction': None,
                'model_type': model_type,
                'error': error_msg,
                'timestamp': datetime.now().isoformat(),
                'status': 'error'
            }
            
    except Exception as e:
        st.error(f"❌ Error in real-time prediction: {e}")
        return {
            'date': target_date,
            'prediction': None,
            'model_type': model_type,
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'status': 'error'
        }

def standardize_forecast_data(forecast_data, model_type):
    """Standardize forecast data to have consistent column names."""
    if forecast_data is None or forecast_data.empty:
        st.warning("⚠️ No forecast data to standardize")
        return pd.DataFrame()
    
    try:
        # Create a copy to avoid modifying original
        result = forecast_data.copy()
        
        st.info(f"🔄 Standardizing {model_type} forecast data...")
        
        # Handle different column naming conventions
        if 'ds' in result.columns and 'yhat' in result.columns:
            # Prophet format
            result = result.rename(columns={'ds': 'date', 'yhat': 'prediction'})
            st.info("✅ Converted Prophet format (ds→date, yhat→prediction)")
        elif 'prediction' in result.columns and 'date' not in result.columns:
            # Add date column if missing
            if 'ds' in result.columns:
                result = result.rename(columns={'ds': 'date'})
                st.info("✅ Renamed ds column to date")
            elif len(result) > 0:
                # Create date range if no date column exists
                result['date'] = pd.date_range(start=datetime.now(), periods=len(result), freq='D')
                st.info("✅ Added date column with default range")
        
        # Ensure we have the required columns
        if 'date' not in result.columns:
            st.error(f"❌ Missing 'date' column in {model_type} forecast data")
            st.info(f"Available columns: {list(result.columns)}")
            return pd.DataFrame()
            
        if 'prediction' not in result.columns:
            # Look for alternative prediction column names
            possible_pred_cols = ['yhat', 'forecast', 'y_pred', 'sales_pred']
            for col in possible_pred_cols:
                if col in result.columns:
                    result = result.rename(columns={col: 'prediction'})
                    st.info(f"✅ Renamed {col} column to prediction")
                    break
            
            if 'prediction' not in result.columns and len(result.columns) > 1:
                # Use the first numeric column as prediction
                numeric_cols = result.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    result = result.rename(columns={numeric_cols[0]: 'prediction'})
                    st.info(f"✅ Using first numeric column '{numeric_cols[0]}' as prediction")
                else:
                    st.error(f"❌ No prediction column found in {model_type} forecast data")
                    st.info(f"Available columns: {list(result.columns)}")
                    return pd.DataFrame()
        
        # Ensure date is datetime type
        result['date'] = pd.to_datetime(result['date'])
        
        # Ensure prediction is numeric
        result['prediction'] = pd.to_numeric(result['prediction'], errors='coerce')
        
        # Remove any NaN values
        result = result.dropna(subset=['prediction'])
        
        st.success(f"✅ Successfully standardized {len(result)} forecast records")
        
        return result[['date', 'prediction']]
        
    except Exception as e:
        st.error(f"❌ Error standardizing forecast data: {e}")
        import traceback
        st.error(f"🔍 Detailed error: {traceback.format_exc()}")
        return pd.DataFrame()

def validate_forecast_data(forecast_data, model_type):
    """Validate forecast data for quality and completeness."""
    if forecast_data is None or forecast_data.empty:
        return False, "No forecast data provided"
    
    try:
        # Check required columns
        if 'date' not in forecast_data.columns:
            return False, "Missing 'date' column"
        
        if 'prediction' not in forecast_data.columns:
            return False, "Missing 'prediction' column"
        
        # Check data types
        if not pd.api.types.is_datetime64_any_dtype(forecast_data['date']):
            return False, "Date column is not datetime type"
        
        if not pd.api.types.is_numeric_dtype(forecast_data['prediction']):
            return False, "Prediction column is not numeric"
        
        # Check for NaN values
        if forecast_data['prediction'].isna().any():
            return False, "Found NaN values in predictions"
        
        # Check date range
        date_range = forecast_data['date'].max() - forecast_data['date'].min()
        if date_range.days < 0:
            return False, "Invalid date range (end date before start date)"
        
        # Check prediction values
        if forecast_data['prediction'].min() < 0:
            st.warning("⚠️ Warning: Negative values found in predictions")
        
        return True, f"✅ Forecast data validated: {len(forecast_data)} records, model: {model_type}"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"
