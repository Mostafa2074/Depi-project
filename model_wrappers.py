# [file name]: model_wrappers.py
import mlflow.pyfunc
import pandas as pd
import numpy as np

class ARIMAModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        
    def predict(self, context, model_input):
        """
        ARIMA prediction that handles different input formats.
        """
        try:
            # Handle different input formats
            if 'ds' in model_input.columns:
                periods = len(model_input)
            else:
                periods = len(model_input)
            
            # Generate predictions
            predictions = self.model.predict(start=0, end=periods-1)
            
            # Ensure we return a DataFrame with consistent structure
            if isinstance(predictions, pd.Series):
                return pd.DataFrame({'prediction': predictions.values})
            else:
                return pd.DataFrame({'prediction': predictions})
                
        except Exception as e:
            # Fallback: return predictions as DataFrame
            import traceback
            print(f"ARIMA prediction error: {e}")
            return pd.DataFrame({'prediction': [0] * len(model_input)})

class ProphetModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        
    def predict(self, context, model_input):
        """
        Generates forecast using the Prophet model.
        Expects input with 'ds' column containing dates.
        """
        try:
            # Ensure we have the required 'ds' column
            if 'ds' not in model_input.columns:
                raise ValueError("Prophet model requires 'ds' column with dates")
            
            # Call the actual Prophet model predict method
            full_forecast = self.model.predict(model_input)
            
            # Return both 'ds' and 'yhat' columns
            return full_forecast[['ds', 'yhat']]
        except Exception as e:
            print(f"Prophet prediction error: {e}")
            # Return empty dataframe with expected columns
            return pd.DataFrame({'ds': model_input['ds'], 'yhat': [0] * len(model_input)})

class LightGBMModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model, feature_columns=None):
        self.model = model
        self.feature_columns = feature_columns or ['year', 'month', 'day', 'dayofweek', 'quarter', 'dayofyear', 'weekofyear']
        
    def predict(self, context, model_input):
        """
        LightGBM prediction with proper feature handling.
        """
        try:
            # Extract features from input
            if 'ds' in model_input.columns:
                # Convert date to features
                dates = pd.to_datetime(model_input['ds'])
                features_df = pd.DataFrame({
                    'year': dates.dt.year.astype('float64'),
                    'month': dates.dt.month.astype('float64'),
                    'day': dates.dt.day.astype('float64'),
                    'dayofweek': dates.dt.dayofweek.astype('float64'),
                    'quarter': dates.dt.quarter.astype('float64'),
                    'dayofyear': dates.dt.dayofyear.astype('float64'),
                    'weekofyear': dates.dt.isocalendar().week.astype('float64')
                })
            else:
                # Use input as-is, but ensure we have the right columns and types
                features_df = model_input[self.feature_columns].copy()
                # Convert all columns to float64 to avoid dtype issues
                for col in features_df.columns:
                    features_df[col] = features_df[col].astype('float64')
            
            # Make predictions
            predictions = self.model.predict(features_df)
            return pd.DataFrame({'prediction': predictions})
            
        except Exception as e:
            print(f"LightGBM prediction error: {e}")
            return pd.DataFrame({'prediction': [0] * len(model_input)})
