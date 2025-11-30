# [file name]: model_wrappers.py
import pandas as pd
import numpy as np
from typing import Dict, Any
import mlflow.pyfunc

class LightGBMModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        # Convert to DataFrame if it's not already
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
        
        # Ensure we have the required features
        required_features = ['year', 'month', 'day', 'dayofweek', 'quarter', 'dayofyear', 'weekofyear']
        
        # Create missing features if needed
        if 'ds' in model_input.columns:
            dates = pd.to_datetime(model_input['ds'])
            model_input['year'] = dates.dt.year
            model_input['month'] = dates.dt.month
            model_input['day'] = dates.dt.day
            model_input['dayofweek'] = dates.dt.dayofweek
            model_input['quarter'] = dates.dt.quarter
            model_input['dayofyear'] = dates.dt.dayofyear
            model_input['weekofyear'] = dates.dt.isocalendar().week
        
        # Select only numeric columns for prediction
        numeric_input = model_input.select_dtypes(include=[np.number])
        
        try:
            # Try to get predictions
            predictions = self.model.predict(numeric_input)
            
            # Return as DataFrame
            return pd.DataFrame({'prediction': predictions})
        except Exception as e:
            print(f"Prediction error: {e}")
            # Return zeros as fallback
            return pd.DataFrame({'prediction': np.zeros(len(model_input))})

class ProphetModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
        
        # Ensure we have 'ds' column
        if 'ds' not in model_input.columns:
            if 'date' in model_input.columns:
                model_input = model_input.rename(columns={'date': 'ds'})
            else:
                raise ValueError("Input must contain 'ds' or 'date' column")
        
        # Convert to datetime
        model_input['ds'] = pd.to_datetime(model_input['ds'])
        
        # Get predictions
        forecast = self.model.predict(model_input)
        
        return pd.DataFrame({'prediction': forecast['yhat'].values})

class ARIMAModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
        
        # Create features for ARIMA
        if 'ds' in model_input.columns:
            dates = pd.to_datetime(model_input['ds'])
            model_input['year'] = dates.dt.year
            model_input['month'] = dates.dt.month
            model_input['day'] = dates.dt.day
        
        try:
            # Get predictions
            predictions = self.model.predict(model_input)
            
            if hasattr(predictions, 'values'):
                return pd.DataFrame({'prediction': predictions.values})
            else:
                return pd.DataFrame({'prediction': predictions})
        except Exception as e:
            print(f"ARIMA prediction error: {e}")
            return pd.DataFrame({'prediction': np.zeros(len(model_input))})
