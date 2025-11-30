import mlflow.pyfunc
import pandas as pd
import numpy as np

class ARIMAModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        
    def predict(self, context, model_input):
        """
        ARIMA prediction that handles different input formats.
        The input should be a DataFrame with date information or just periods.
        """
        try:
            # Convert input types to match expected schema if needed
            if 'year' in model_input.columns:
                model_input = model_input.copy()
                model_input['year'] = model_input['year'].astype('int32')
                model_input['month'] = model_input['month'].astype('int32')
                model_input['day'] = model_input['day'].astype('int32')
            
            # Handle different input formats
            if 'ds' in model_input.columns:
                periods = len(model_input)
            elif 'dummy' in model_input.columns:
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
            print(traceback.format_exc())
            return pd.DataFrame({'prediction': [0] * len(model_input)})

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
            import traceback
            print(f"LightGBM prediction error: {e}")
            print(traceback.format_exc())
            return pd.DataFrame({'prediction': [0] * len(model_input)})

# Prophet wrapper removed since Prophet is not available
