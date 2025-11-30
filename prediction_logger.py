# [file name]: prediction_logger.py
# [file content begin]
import pandas as pd
import os
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import mlflow
import tempfile

class PredictionLogger:
    def __init__(self):
        self.experiment_name = "prediction_logs"
        self._setup_mlflow_experiment()
    
    def _setup_mlflow_experiment(self):
        """Setup MLflow experiment for prediction logs"""
        try:
            # Create or get the prediction logs experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(self.experiment_name)
            
            mlflow.set_experiment(self.experiment_name)
            print(f"✅ MLflow experiment '{self.experiment_name}' is ready")
            
        except Exception as e:
            print(f"❌ Failed to setup MLflow experiment: {e}")
    
    def log_batch_prediction(self, forecast_data, actual_data=None, model_type="unknown", forecast_end_date=None):
        """Log batch predictions directly to MLflow only"""
        try:
            # Prepare new log entries
            new_logs = []
            
            for _, row in forecast_data.iterrows():
                date = row['date']
                predicted = row['prediction']
                
                # Find actual value if available
                actual = None
                if actual_data is not None and not actual_data.empty:
                    actual_match = actual_data[actual_data['ds'] == date]
                    if not actual_match.empty:
                        actual = actual_match['y'].iloc[0]
                
                # Calculate metrics if actual value is available
                mae = None
                rmse = None
                alert = ""
                
                if actual is not None and not pd.isna(actual):
                    mae = abs(predicted - actual)
                    rmse = (predicted - actual) ** 2
                    # Simple alert logic - you can customize this
                    if mae > actual * 0.25:  # If error is more than 25% of actual value
                        alert = "HIGH_ERROR"
                    else:
                        alert = "OK"
                
                new_log = {
                    'date': date,
                    'predicted': predicted,
                    'actual': actual,
                    'mae': mae,
                    'rmse': rmse,
                    'alert': alert,
                    'model_type': model_type,
                    'prediction_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'forecast_end_date': forecast_end_date
                }
                new_logs.append(new_log)
            
            # Create DataFrame from new logs
            new_logs_df = pd.DataFrame(new_logs)
            
            # Log to MLflow only - NO CSV FILE
            success = self._log_to_mlflow(new_logs_df, model_type, forecast_end_date)
            
            if success:
                print(f"✅ Successfully logged {len(new_logs_df)} predictions for {model_type} to MLflow")
                return len(new_logs_df)
            else:
                print(f"❌ MLflow logging failed for {model_type}")
                return 0
            
        except Exception as e:
            print(f"Error logging predictions: {e}")
            return 0
    
    def _log_to_mlflow(self, predictions_df, model_type, forecast_end_date):
        """Log predictions to MLflow with robust error handling"""
        try:
            # Set the experiment
            mlflow.set_experiment(self.experiment_name)
            
            # Create a unique run name
            run_name = f"prediction_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            print(f"🔄 Starting MLflow logging for {model_type}...")
            
            # Start MLflow run with explicit success tracking
            with mlflow.start_run(run_name=run_name) as run:
                try:
                    # Log basic parameters
                    mlflow.log_params({
                        "model_type": model_type,
                        "forecast_end_date": str(forecast_end_date),
                        "total_predictions": len(predictions_df),
                        "prediction_timestamp": datetime.now().isoformat()
                    })
                    print("✅ Parameters logged")
                    
                    # Calculate and log metrics
                    metrics = self._calculate_metrics(predictions_df)
                    if metrics:
                        mlflow.log_metrics(metrics)
                        print(f"✅ Metrics logged: {metrics}")
                    
                    # Log the predictions as artifact with temp file
                    temp_csv_path = None
                    try:
                        # Create a temporary CSV file for logging
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
                            predictions_df.to_csv(temp_file.name, index=False)
                            temp_csv_path = temp_file.name
                        
                        mlflow.log_artifact(temp_csv_path, "predictions")
                        print("✅ Predictions artifact logged to MLflow")
                        
                    except Exception as e:
                        print(f"⚠️  CSV artifact logging failed: {e}")
                        # Try alternative approach - log as table
                        try:
                            mlflow.log_table(predictions_df.head(20), "prediction_samples.json")
                            print("✅ Table artifact logged as fallback")
                        except Exception as e2:
                            print(f"⚠️  Table logging also failed: {e2}")
                    
                    finally:
                        # Clean up temp file
                        if temp_csv_path and os.path.exists(temp_csv_path):
                            os.unlink(temp_csv_path)
                    
                    # Set tags for easy searching
                    mlflow.set_tags({
                        "prediction_type": "batch",
                        "model_family": model_type,
                        "environment": "production",
                        "log_type": "prediction",
                        "status": "success"
                    })
                    print("✅ Tags set")
                    
                    # Explicitly mark the run as successful
                    mlflow.set_tag("mlflow.runStatus", "FINISHED")
                    
                    print(f"✅ Successfully completed MLflow logging for {model_type} with run ID: {run.info.run_id}")
                    return True
                    
                except Exception as inner_e:
                    print(f"❌ Error during MLflow run for {model_type}: {inner_e}")
                    # Mark the run as failed
                    mlflow.set_tag("mlflow.runStatus", "FAILED")
                    mlflow.set_tag("error_message", str(inner_e))
                    return False
                
        except Exception as e:
            print(f"❌ MLflow logging failed for {model_type}: {e}")
            return False
    
    def _calculate_metrics(self, predictions_df):
        """Calculate metrics from predictions with robust error handling"""
        metrics = {}
        
        try:
            # Basic counts that should always work
            metrics["total_predictions"] = len(predictions_df)
            metrics["high_error_count"] = (predictions_df['alert'] == 'HIGH_ERROR').sum() if 'alert' in predictions_df.columns else 0
            
            # Filter rows where we have both actual and predicted values
            if 'actual' in predictions_df.columns and 'predicted' in predictions_df.columns:
                valid_data = predictions_df.dropna(subset=['actual', 'predicted'])
                metrics["predictions_with_actuals"] = len(valid_data)
                
                if len(valid_data) > 0:
                    metrics["coverage_ratio"] = len(valid_data) / len(predictions_df)
                    
                    if len(valid_data) >= 2:  # Need at least 2 points for these metrics
                        try:
                            actuals = valid_data['actual']
                            predictions = valid_data['predicted']
                            
                            metrics["mae"] = float(mean_absolute_error(actuals, predictions))
                            metrics["rmse"] = float(np.sqrt(mean_squared_error(actuals, predictions)))
                            metrics["mean_actual"] = float(actuals.mean())
                            metrics["mean_predicted"] = float(predictions.mean())
                        except Exception as e:
                            print(f"⚠️  Advanced metric calculation failed: {e}")
                            # Still log basic metrics even if advanced ones fail
                    
            return metrics
            
        except Exception as e:
            print(f"⚠️  Metric calculation failed: {e}")
            # Return at least basic metrics
            return {"total_predictions": len(predictions_df)}
    
    def get_recent_logs_from_mlflow(self, days=7):
        """Get recent prediction logs from MLflow"""
        try:
            client = mlflow.tracking.MlflowClient()
            
            # Search for recent prediction runs
            runs = client.search_runs(
                experiment_ids=[mlflow.get_experiment_by_name(self.experiment_name).experiment_id],
                filter_string=f"attributes.start_time >= {int((datetime.now() - pd.Timedelta(days=days)).timestamp() * 1000)}",
                order_by=["attributes.start_time DESC"],
                max_results=50
            )
            
            all_predictions = []
            for run in runs:
                predictions = self._extract_predictions_from_run(run)
                if predictions is not None and not predictions.empty:
                    all_predictions.append(predictions)
            
            if all_predictions:
                return pd.concat(all_predictions, ignore_index=True)
            return pd.DataFrame()
            
        except Exception as e:
            print(f"Error reading logs from MLflow: {e}")
            return pd.DataFrame()
    
    def _extract_predictions_from_run(self, run):
        """Extract predictions from a single MLflow run"""
        try:
            client = mlflow.tracking.MlflowClient()
            run_id = run.info.run_id
            
            # List artifacts in the run
            artifacts = client.list_artifacts(run_id)
            
            for artifact in artifacts:
                if artifact.path.endswith('.csv') or 'predictions' in artifact.path:
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        local_path = client.download_artifacts(run_id, artifact.path, tmp_dir)
                        
                        if os.path.isdir(local_path):
                            csv_files = [f for f in os.listdir(local_path) if f.endswith('.csv')]
                            if csv_files:
                                local_path = os.path.join(local_path, csv_files[0])
                        
                        if os.path.exists(local_path):
                            predictions_df = pd.read_csv(local_path)
                            return predictions_df
            
            return None
            
        except Exception as e:
            print(f"Error extracting predictions from run {run.info.run_id}: {e}")
            return None
    
    def get_log_stats_from_mlflow(self):
        """Get statistics from MLflow prediction logs"""
        try:
            recent_logs = self.get_recent_logs_from_mlflow(days=30)
            
            if not recent_logs.empty:
                stats = {
                    'total_predictions': len(recent_logs),
                    'predictions_with_actuals': recent_logs['actual'].notna().sum() if 'actual' in recent_logs.columns else 0,
                    'high_error_count': (recent_logs['alert'] == 'HIGH_ERROR').sum() if 'alert' in recent_logs.columns else 0,
                    'unique_model_types': recent_logs['model_type'].nunique() if 'model_type' in recent_logs.columns else 0
                }
                return stats
            return {}
        except Exception as e:
            print(f"Error getting log stats from MLflow: {e}")
            return {}
# [file content end]