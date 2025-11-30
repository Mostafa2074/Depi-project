# [file name]: monitoring.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import mlflow
from mlflow.tracking import MlflowClient
import os
import tempfile
import numpy as np
import time
import sys
from project_paths import get_model_paths

# Robust email alert imports with multiple fallbacks
EMAIL_ALERTS_AVAILABLE = False
EmailAlert = None
EmailContent = None

try:
    # Try direct import first
    from email_alert import EmailAlert
    from email_content import EmailContent
    EMAIL_ALERTS_AVAILABLE = True
    print("✅ Email alert modules loaded successfully")
except ImportError as e:
    print(f"❌ Direct import failed: {e}")
    try:
        # Try relative import
        from .email_alert import EmailAlert
        from .email_content import EmailContent
        EMAIL_ALERTS_AVAILABLE = True
        print("✅ Email alert modules loaded via relative import")
    except ImportError as e2:
        print(f"❌ Relative import also failed: {e2}")
        st.warning("Email alert system not available. Proceeding without email notifications.")

def run_monitoring_app():
    st.title("📊 Prediction Monitoring Dashboard")
    
    # Email configuration section
    if EMAIL_ALERTS_AVAILABLE:
        with st.expander("📧 Email Alert Configuration", expanded=False):
            st.info("Configure email alerts for high prediction errors")
            
            email_enabled = st.checkbox("Enable Email Alerts", value=False)  # Default to False for cloud
            recipient_email = st.text_input(
                "Recipient Email", 
                value=st.secrets.get("EMAIL", {}).get("RECIPIENT", "") if st.secrets.get("EMAIL") else "",
                help="Email address to receive high error alerts"
            )
            
            if st.button("Test Email Connection") and recipient_email:
                test_email_connection(recipient_email)
    else:
        email_enabled = False
        recipient_email = ""
        st.warning("Email alert system not available. Please ensure email_alert.py and email_content.py are in the same directory.")
    
    # Load actual dataset first
    try:
        paths = get_model_paths()
        actual_dataset_path = paths['actual_dataset']
        
        # Use Streamlit's file uploader as fallback for cloud deployment
        if not os.path.exists(actual_dataset_path):
            st.info("Upload actual dataset CSV file for monitoring:")
            uploaded_file = st.file_uploader("Upload actual_dataset.csv", type="csv")
            if uploaded_file is not None:
                actual_df = pd.read_csv(uploaded_file)
                actual_df['date'] = pd.to_datetime(actual_df['date'])
                actual_df = actual_df.rename(columns={'sales': 'actual_sales'})
                st.success("✅ Loaded actual dataset from uploaded file")
            else:
                st.error("Please upload actual_dataset.csv file")
                return
        else:
            actual_df = pd.read_csv(actual_dataset_path)
            actual_df['date'] = pd.to_datetime(actual_df['date'])
            actual_df = actual_df.rename(columns={'sales': 'actual_sales'})
            st.success(f"✅ Loaded actual dataset from {actual_dataset_path}")
        
        # Show actual data date range
        if not actual_df.empty:
            st.info(f"📅 Actual data range: {actual_df['date'].min().date()} to {actual_df['date'].max().date()}")
    except Exception as e:
        st.error(f"❌ Could not load actual dataset: {e}")
        st.info("Please ensure 'actual_dataset.csv' exists in your directory or upload it")
        return
    
    # Try to get prediction data directly from MLflow
    prediction_data = get_latest_prediction_from_mlflow()
    
    if prediction_data is None:
        show_prediction_instructions(actual_df)
        return
    
    # Show prediction data source and range
    st.success(f"✅ Loaded predictions from MLflow (latest run)")
    st.info(f"📅 Prediction data range: {prediction_data['date'].min().date()} to {prediction_data['date'].max().date()}")
    
    # Check if we have overlapping dates
    overlapping_dates = pd.merge(
        prediction_data[['date']],
        actual_df[['date']],
        on='date',
        how='inner'
    )
    
    if overlapping_dates.empty:
        st.warning("⚠️ No overlapping dates between predictions and actual data!")
        
        # Offer to create historical predictions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Generate Historical Predictions", type="primary"):
                create_historical_predictions(actual_df)
                st.rerun()
        
        with col2:
            if st.button("📊 Show Demo Comparison"):
                show_demo_comparison(actual_df)
        
        return
    
    # Merge predictions with actual data
    comparison_df = merge_predictions_with_actuals(prediction_data, actual_df)
    
    if comparison_df.empty:
        st.warning("No overlapping data found between predictions and actual dataset.")
        return
    
    # Display main comparison with email alerts
    display_comparison_analysis(comparison_df, prediction_data, email_enabled, recipient_email)

def test_email_connection(recipient_email):
    """Test email connection and send a test message"""
    try:
        # Check if email credentials are available in secrets
        if not st.secrets.get("EMAIL"):
            st.error("Email configuration not found in secrets. Please configure EMAIL section in .streamlit/secrets.toml")
            return
            
        email_alert = EmailAlert.get_instance()
        
        # Create test email content
        email_content = EmailContent()
        email_content.recipient = recipient_email
        email_content.subject = "Test Email - Monitoring System"
        
        test_body = {
            'title': 'Test Email Connection',
            'message': 'This is a test email from the Monitoring Dashboard to verify the email alert system is working correctly.',
            'table_rows': {
                'System': 'Prediction Monitoring Dashboard',
                'Status': 'Operational',
                'Test Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            },
            'notes': 'If you receive this email, the alert system is configured correctly.'
        }
        email_content.message_body = test_body
        email_content.prepare_html()
        
        # Send test email
        email_alert.send_email(email_content)
        st.success(f"✅ Test email sent successfully to {recipient_email}!")
        
    except Exception as e:
        st.error(f"❌ Failed to send test email: {e}")
        st.info("Please check your .streamlit/secrets.toml file with SENDER_EMAIL and SENDER_PASSWORD")

# In monitoring.py, update the send_high_error_alert function:

def send_high_error_alert(recipient_email, error_data, comparison_stats):
    """Send email alert for high prediction error"""
    try:
        if not EMAIL_ALERTS_AVAILABLE:
            print("Email alerts not available - EMAIL_ALERTS_AVAILABLE is False")
            return False
            
        if EmailAlert is None:
            print("EmailAlert is None - cannot send email")
            return False
            
        email_alert = EmailAlert.get_instance()
        email_content = EmailContent()
        email_content.recipient = recipient_email
        email_content.subject = f"🚨 High Prediction Error Alert - {error_data['date'].strftime('%Y-%m-%d')}"
        
        # Calculate error percentage
        error_percentage = (error_data['absolute_error'] / error_data['actual_sales']) * 100
        
        # Prepare email body
        alert_body = {
            'title': 'High Prediction Error Detected',
            'message': f'A significant prediction error has been detected in the monitoring system. The error rate of {error_percentage:.1f}% exceeds the 25% threshold.',
            'table_rows': {
                'Date': error_data['date'].strftime('%Y-%m-%d'),
                'Actual Sales': f"${error_data['actual_sales']:,.0f}",
                'Predicted Sales': f"${error_data['predicted']:,.0f}",
                'Absolute Error': f"${error_data['absolute_error']:,.0f}",
                'Error Percentage': f"{error_percentage:.1f}%",
                'Model Type': error_data.get('model_type', 'Unknown'),
                'Current MAE': f"${comparison_stats['mae']:,.0f}",
                'Current Accuracy': f"{comparison_stats['accuracy']:.1f}%"
            },
            'notes': 'Please review the model performance and consider retraining or adjusting parameters. This alert was triggered when the absolute error exceeded 25% of the actual sales value.'
        }
        email_content.message_body = alert_body
        email_content.prepare_html()
        
        # Send alert email with better error handling
        try:
            success = email_alert.send_email(email_content)
            if success:
                print("✅ Email sent successfully!")
            return success
        except Exception as email_error:
            if "Daily user sending limit exceeded" in str(email_error):
                print("⚠️ Gmail daily limit reached - email not sent")
                return False
            elif "Authentication failed" in str(email_error):
                print("❌ Email authentication failed")
                return False
            else:
                print(f"❌ Email sending failed: {email_error}")
                return False
        
    except Exception as e:
        print(f"Failed to send email alert: {e}")
        return False
        
def get_latest_prediction_from_mlflow():
    """Fetch the latest prediction file directly from MLflow"""
    try:
        client = MlflowClient()
        
        # Search for prediction runs specifically
        runs = client.search_runs(
            experiment_ids=["0"],  # Search all experiments
            filter_string="tags.mlflow.runStatus = 'FINISHED' AND tags.log_type = 'prediction'",
            order_by=["start_time DESC"],
            max_results=10
        )
        
        if not runs:
            # Fallback: search any finished run with prediction artifacts
            runs = client.search_runs(
                experiment_ids=["0"],
                filter_string="tags.mlflow.runStatus = 'FINISHED'",
                order_by=["start_time DESC"],
                max_results=10
            )
        
        for run in runs:
            result = extract_predictions_from_mlflow_run(run)
            if result is not None and not result.empty:
                st.info(f"📁 Using predictions from MLflow run: {run.info.run_name}")
                return result
        
        # If no predictions found in runs, check the prediction_logs experiment
        return search_prediction_logs_experiment()
                
    except Exception as e:
        st.error(f"Error fetching predictions from MLflow: {e}")
        return None

def search_prediction_logs_experiment():
    """Search the dedicated prediction_logs experiment"""
    try:
        client = MlflowClient()
        
        # Get the prediction_logs experiment
        experiment = client.get_experiment_by_name("prediction_logs")
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string="tags.mlflow.runStatus = 'FINISHED'",
                order_by=["start_time DESC"],
                max_results=5
            )
            
            for run in runs:
                result = extract_predictions_from_mlflow_run(run)
                if result is not None and not result.empty:
                    st.info(f"📁 Using predictions from prediction_logs run: {run.info.run_name}")
                    return result
        
        return None
        
    except Exception as e:
        st.error(f"Error searching prediction_logs experiment: {e}")
        return None

def extract_predictions_from_mlflow_run(run):
    """Extract prediction data from an MLflow run artifact"""
    try:
        client = MlflowClient()
        run_id = run.info.run_id
        
        # List all artifacts in the run
        artifacts = client.list_artifacts(run_id)
        
        for artifact in artifacts:
            if artifact.path.endswith('.csv') or 'predictions' in artifact.path:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    try:
                        # Download the artifact
                        local_path = client.download_artifacts(run_id, artifact.path, tmp_dir)
                        
                        # Handle directory artifacts
                        if os.path.isdir(local_path):
                            csv_files = [f for f in os.listdir(local_path) if f.endswith('.csv')]
                            if csv_files:
                                local_path = os.path.join(local_path, csv_files[0])
                        
                        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                            pred_df = pd.read_csv(local_path)
                            
                            # Standardize the dataframe
                            standardized_df = standardize_prediction_dataframe(pred_df)
                            
                            if not standardized_df.empty:
                                return standardized_df
                                
                    except Exception as e:
                        st.warning(f"Could not read artifact {artifact.path}: {e}")
                        continue
        
        return None
        
    except Exception as e:
        st.warning(f"Error extracting from run {run.info.run_name}: {e}")
        return None

def standardize_prediction_dataframe(df):
    """Standardize prediction dataframe to common format"""
    try:
        # Create a copy to avoid modifying original
        result_df = df.copy()
        
        # Handle date column
        if 'date' in result_df.columns:
            result_df['date'] = pd.to_datetime(result_df['date'])
        elif 'ds' in result_df.columns:
            result_df = result_df.rename(columns={'ds': 'date'})
            result_df['date'] = pd.to_datetime(result_df['date'])
        else:
            # Try to find any datetime column
            date_cols = [col for col in result_df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_cols:
                result_df = result_df.rename(columns={date_cols[0]: 'date'})
                result_df['date'] = pd.to_datetime(result_df['date'])
            else:
                st.warning("No date column found in prediction data")
                return pd.DataFrame()
        
        # Handle prediction column
        if 'predicted' in result_df.columns:
            # Already standardized
            pass
        elif 'prediction' in result_df.columns:
            result_df = result_df.rename(columns={'prediction': 'predicted'})
        elif 'yhat' in result_df.columns:
            result_df = result_df.rename(columns={'yhat': 'predicted'})
        elif 'forecast' in result_df.columns:
            result_df = result_df.rename(columns={'forecast': 'predicted'})
        else:
            # Try to find any prediction-like column
            pred_cols = [col for col in result_df.columns if any(x in col.lower() for x in ['pred', 'yhat', 'forecast', 'estimate'])]
            if pred_cols:
                result_df = result_df.rename(columns={pred_cols[0]: 'predicted'})
            else:
                # Use first numeric column as prediction
                numeric_cols = result_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    result_df = result_df.rename(columns={numeric_cols[0]: 'predicted'})
                else:
                    st.warning("No prediction column found in data")
                    return pd.DataFrame()
        
        # Ensure model_type exists
        if 'model_type' not in result_df.columns:
            result_df['model_type'] = 'mlflow_model'
        
        # Select only necessary columns
        required_cols = ['date', 'predicted', 'model_type']
        available_cols = [col for col in required_cols if col in result_df.columns]
        
        return result_df[available_cols]
        
    except Exception as e:
        st.error(f"Error standardizing prediction data: {e}")
        return pd.DataFrame()

def create_historical_predictions(actual_df):
    """Create predictions for historical dates that exist in the actual dataset"""
    try:
        # Use the last 30 days of actual data for predictions
        historical_dates = actual_df['date'].tail(30).copy()
        
        # Create realistic predictions with some noise
        predictions_data = []
        for date in historical_dates:
            actual_row = actual_df[actual_df['date'] == date]
            if not actual_row.empty:
                actual_sales = actual_row['actual_sales'].iloc[0]
                # Add realistic noise (5-15% variation)
                noise = np.random.normal(0, 0.1)  # 10% standard deviation
                predicted_sales = actual_sales * (1 + noise)
                
                predictions_data.append({
                    'date': date,
                    'predicted': max(predicted_sales, 0),  # Ensure non-negative
                    'model_type': 'historical_demo',
                    'prediction_timestamp': datetime.now()
                })
        
        if predictions_data:
            pred_df = pd.DataFrame(predictions_data)
            
            # Log to MLflow instead of saving to CSV
            log_predictions_to_mlflow(pred_df, "historical_demo")
            
            st.success(f"✅ Generated {len(pred_df)} historical predictions and logged to MLflow!")
            st.info("These predictions use actual historical data with realistic noise for demonstration.")
        else:
            st.error("Could not generate historical predictions")
            
    except Exception as e:
        st.error(f"Error generating historical predictions: {e}")

def log_predictions_to_mlflow(predictions_df, model_type):
    """Log predictions directly to MLflow"""
    try:
        mlflow.set_experiment("prediction_logs")
        
        with mlflow.start_run(run_name=f"monitoring_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            # Log parameters
            mlflow.log_params({
                "model_type": model_type,
                "total_predictions": len(predictions_df),
                "purpose": "monitoring_demo",
                "timestamp": datetime.now().isoformat()
            })
            
            # Log metrics
            if 'actual_sales' in predictions_df.columns and 'predicted' in predictions_df.columns:
                valid_data = predictions_df.dropna(subset=['actual_sales', 'predicted'])
                if len(valid_data) > 0:
                    mae = (valid_data['actual_sales'] - valid_data['predicted']).abs().mean()
                    rmse = ((valid_data['actual_sales'] - valid_data['predicted']) ** 2).mean() ** 0.5
                    mlflow.log_metrics({"demo_mae": mae, "demo_rmse": rmse})
            
            # Log the predictions as artifact
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
                predictions_df.to_csv(temp_file.name, index=False)
                mlflow.log_artifact(temp_file.name, "predictions")
            
            mlflow.set_tags({
                "prediction_type": "batch",
                "environment": "demo",
                "log_type": "prediction"
            })
            
    except Exception as e:
        st.warning(f"Could not log to MLflow: {e}")

def show_demo_comparison(actual_df):
    """Show a demonstration comparison using the last 30 days of actual data"""
    st.subheader("🎯 Demonstration: Historical Prediction Analysis")
    
    # Use last 30 days for demonstration
    demo_dates = actual_df['date'].tail(30).copy()
    demo_actual = actual_df[actual_df['date'].isin(demo_dates)].copy()
    
    # Create simulated predictions with realistic patterns
    demo_predictions = []
    for date in demo_dates:
        actual_row = demo_actual[demo_actual['date'] == date]
        if not actual_row.empty:
            actual_sales = actual_row['actual_sales'].iloc[0]
            # Simulate different prediction scenarios
            if len(demo_predictions) < 10:
                # Good predictions (small error)
                error = np.random.normal(0, 0.05)  # 5% error
            elif len(demo_predictions) < 20:
                # Medium predictions
                error = np.random.normal(0, 0.1)   # 10% error
            else:
                # Some bad predictions
                error = np.random.normal(0, 0.15)  # 15% error
            
            predicted_sales = actual_sales * (1 + error)
            
            demo_predictions.append({
                'date': date,
                'predicted': max(predicted_sales, 0),
                'model_type': 'demo_model'
            })
    
    demo_pred_df = pd.DataFrame(demo_predictions)
    comparison_df = pd.merge(demo_actual, demo_pred_df, on='date')
    
    # Calculate errors
    comparison_df['absolute_error'] = abs(comparison_df['predicted'] - comparison_df['actual_sales'])
    comparison_df['percentage_error'] = (comparison_df['absolute_error'] / comparison_df['actual_sales']) * 100
    comparison_df['error'] = comparison_df['predicted'] - comparison_df['actual_sales']
    
    display_comparison_analysis(comparison_df, demo_pred_df, False, "")
    
    st.info("💡 This is a demonstration using simulated predictions. To see real model performance, generate predictions using the Forecast Engine.")

def show_prediction_instructions(actual_df):
    """Show instructions for generating predictions"""
    st.warning("No prediction data found in MLflow!")
    
    st.subheader("🚀 How to Generate Predictions for Monitoring")
    
    st.write(f"""
    Your actual data goes from **{actual_df['date'].min().date()}** to **{actual_df['date'].max().date()}**.
    
    To see meaningful comparisons in the Monitoring dashboard, you need predictions for dates that exist in your actual dataset.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Option 1: Generate Historical Predictions**")
        st.write("""
        - Click the button below to create demo predictions
        - Uses your actual historical data
        - Adds realistic noise for demonstration
        - Perfect for testing the monitoring features
        """)
        
        if st.button("🔄 Generate Historical Demo Predictions", type="primary"):
            create_historical_predictions(actual_df)
            st.rerun()
    
    with col2:
        st.write("**Option 2: Use Forecast Engine**")
        st.write("""
        - Go to **Forecast Engine** tab
        - Choose dates within your actual data range
        - For example: 2017-07-01 to 2017-08-15
        - This will create comparable predictions in MLflow
        """)
    
    st.write("**Recommended Date Range for Testing:**")
    # Show the last 30 days of actual data as recommended range
    last_date = actual_df['date'].max().date()
    start_date = (actual_df['date'].max() - timedelta(days=30)).date()
    st.code(f"Start: {start_date}\nEnd: {last_date}")

def merge_predictions_with_actuals(pred_df, actual_df):
    """Merge prediction data with actual sales data"""
    try:
        # Ensure we have the required columns
        if 'predicted' not in pred_df.columns:
            st.error("Prediction data missing 'predicted' column")
            return pd.DataFrame()
        
        # Ensure model_type exists in prediction data
        if 'model_type' not in pred_df.columns:
            pred_df['model_type'] = 'unknown'
        
        # Merge on date
        merged_df = pd.merge(
            pred_df[['date', 'predicted', 'model_type']],
            actual_df[['date', 'actual_sales']],
            on='date',
            how='inner'
        )
        
        if merged_df.empty:
            return pd.DataFrame()
        
        # Calculate errors
        merged_df['absolute_error'] = abs(merged_df['predicted'] - merged_df['actual_sales'])
        merged_df['percentage_error'] = (merged_df['absolute_error'] / merged_df['actual_sales']) * 100
        merged_df['error'] = merged_df['predicted'] - merged_df['actual_sales']
        
        st.success(f"✅ Successfully merged {len(merged_df)} data points for comparison")
        return merged_df
        
    except Exception as e:
        st.error(f"Error merging data: {e}")
        return pd.DataFrame()

def display_comparison_analysis(comparison_df, pred_df, email_enabled=False, recipient_email=""):
    """Display the main comparison charts with smooth point-by-point animation"""
    
    st.subheader("📊 Prediction vs Actual Comparison - Live Monitoring")
    
    # Create placeholders for dynamic updates
    metrics_placeholder = st.empty()
    chart_placeholder1 = st.empty()
    chart_placeholder2 = st.empty()
    chart_placeholder3 = st.empty()
    table_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Initialize empty data for animation
    animated_df = pd.DataFrame()
    high_error_detected = False
    email_sent = False
    
    # Get the date range for progress tracking
    total_rows = len(comparison_df)
    
    # Progress bar
    progress_bar = st.progress(0)
    
    # Process data point by point for smooth animation
    for idx, (index, row) in enumerate(comparison_df.iterrows()):
        # Add current point to animated dataframe
        current_row_df = pd.DataFrame([row])
        animated_df = pd.concat([animated_df, current_row_df], ignore_index=True)
        
        # Calculate current metrics for email
        current_mae = animated_df['absolute_error'].mean()
        current_mape = animated_df['percentage_error'].mean()
        current_accuracy = 100 - current_mape
        comparison_stats = {
            'mae': current_mae,
            'accuracy': current_accuracy
        }
        
        # Check for high error
        if row['absolute_error'] > row['actual_sales'] * 0.25 and not high_error_detected:
            high_error_detected = True
            first_high_error = row.copy()
            
            # Send email alert if enabled
            if email_enabled and not email_sent and recipient_email:
                with status_placeholder.container():
                    st.warning("🚨 Sending email alert...")
                
                email_sent = send_high_error_alert(recipient_email, first_high_error, comparison_stats)
                
                if email_sent:
                    with status_placeholder.container():
                        st.success(f"✅ Email alert sent to {recipient_email}")
                else:
                    with status_placeholder.container():
                        st.error("❌ Failed to send email alert")
        
        # Update progress
        progress = (idx + 1) / total_rows
        progress_bar.progress(progress)
        
        # Update metrics
        with metrics_placeholder.container():
            col1, col2, col3, col4 = st.columns(4)
            
            mae = current_mae
            rmse = (animated_df['absolute_error'] ** 2).mean() ** 0.5
            mape = current_mape
            accuracy = current_accuracy
            
            with col1:
                st.metric("Mean Absolute Error (MAE)", f"${mae:,.0f}")
            with col2:
                st.metric("Root Mean Square Error (RMSE)", f"${rmse:,.0f}")
            with col3:
                st.metric("Mean Absolute Percentage Error (MAPE)", f"{mape:.1f}%")
            with col4:
                st.metric("Overall Accuracy", f"{accuracy:.1f}%")
        
        # Update comparison chart - Smooth point-by-point animation
        with chart_placeholder1.container():
            fig_comparison = go.Figure()
            
            # Add actual sales line
            if len(animated_df) > 1:
                fig_comparison.add_trace(go.Scatter(
                    x=animated_df['date'][:-1],
                    y=animated_df['actual_sales'][:-1],
                    mode='lines+markers',
                    name='Actual Sales',
                    line=dict(color='#2ecc71', width=2),
                    marker=dict(size=4, color='#2ecc71', opacity=0.7)
                ))
            
            # Add predicted sales line
            if len(animated_df) > 1:
                fig_comparison.add_trace(go.Scatter(
                    x=animated_df['date'][:-1],
                    y=animated_df['predicted'][:-1],
                    mode='lines+markers',
                    name='Predicted Sales',
                    line=dict(color='#e74c3c', width=2, dash='dash'),
                    marker=dict(size=4, color='#e74c3c', opacity=0.7)
                ))
            
            # Add the current point with emphasis
            if len(animated_df) > 0:
                current_point = animated_df.iloc[-1]
                
                # Draw line from previous point to current point
                if len(animated_df) > 1:
                    prev_point = animated_df.iloc[-2]
                    # Actual line segment
                    fig_comparison.add_trace(go.Scatter(
                        x=[prev_point['date'], current_point['date']],
                        y=[prev_point['actual_sales'], current_point['actual_sales']],
                        mode='lines',
                        name='_Actual Segment',
                        line=dict(color='#2ecc71', width=2.5),
                        showlegend=False
                    ))
                    # Predicted line segment
                    fig_comparison.add_trace(go.Scatter(
                        x=[prev_point['date'], current_point['date']],
                        y=[prev_point['predicted'], current_point['predicted']],
                        mode='lines',
                        name='_Predicted Segment',
                        line=dict(color='#e74c3c', width=2.5, dash='dash'),
                        showlegend=False
                    ))
                
                # Highlight current points
                fig_comparison.add_trace(go.Scatter(
                    x=[current_point['date']],
                    y=[current_point['actual_sales']],
                    mode='markers',
                    name='Current Actual',
                    marker=dict(size=10, color='#2ecc71', symbol='circle', 
                               line=dict(width=2, color='white'))
                ))
                fig_comparison.add_trace(go.Scatter(
                    x=[current_point['date']],
                    y=[current_point['predicted']],
                    mode='markers',
                    name='Current Predicted',
                    marker=dict(size=10, color='#e74c3c', symbol='circle',
                               line=dict(width=2, color='white'))
                ))
            
            # Add vertical line at high error point if detected
            if high_error_detected:
                fig_comparison.add_shape(
                    type="line",
                    x0=first_high_error['date'],
                    y0=0,
                    x1=first_high_error['date'],
                    y1=1,
                    xref="x",
                    yref="paper",
                    line=dict(color="red", width=2, dash="dot")
                )
                fig_comparison.add_annotation(
                    x=first_high_error['date'],
                    y=1,
                    xref="x",
                    yref="paper",
                    text="🚨 High Error!",
                    showarrow=False,
                    yshift=15,
                    bgcolor="red",
                    font=dict(color="white", size=12)
                )
            
            fig_comparison.update_layout(
                title=f"📈 Live Sales Comparison - Point {idx+1}/{total_rows}",
                xaxis_title="Date",
                yaxis_title="Sales Amount",
                hovermode='x unified',
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Update additional charts
        with chart_placeholder2.container():
            col1, col2 = st.columns(2)
            
            with col1:
                # Error distribution
                fig_errors = px.histogram(
                    animated_df,
                    x='absolute_error',
                    title=f"📊 Error Distribution",
                    nbins=min(8, max(3, len(animated_df)//2)),
                    color_discrete_sequence=['#f39c12'],
                    opacity=0.8
                )
                fig_errors.update_layout(
                    xaxis_title="Absolute Error", 
                    yaxis_title="Frequency", 
                    height=300,
                    showlegend=False
                )
                st.plotly_chart(fig_errors, use_container_width=True)
            
            with col2:
                # Scatter plot
                fig_scatter = px.scatter(
                    animated_df,
                    x='actual_sales',
                    y='predicted',
                    title=f"🎯 Predicted vs Actual",
                    color_discrete_sequence=['#3498db']
                )
                # Add perfect prediction line
                if len(animated_df) > 0:
                    max_val = max(animated_df['actual_sales'].max(), animated_df['predicted'].max())
                    fig_scatter.add_trace(go.Scatter(
                        x=[0, max_val],
                        y=[0, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='red', dash='dash', width=2)
                    ))
                # Highlight latest point
                if len(animated_df) > 0:
                    latest_point = animated_df.iloc[-1]
                    fig_scatter.add_trace(go.Scatter(
                        x=[latest_point['actual_sales']],
                        y=[latest_point['predicted']],
                        mode='markers',
                        name='Current Point',
                        marker=dict(size=10, color='#e74c3c', symbol='star', 
                                   line=dict(width=2, color='white'))
                    ))
                fig_scatter.update_layout(
                    showlegend=True, 
                    height=300,
                    xaxis_title="Actual Sales",
                    yaxis_title="Predicted Sales"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Update error trend chart
        with chart_placeholder3.container():
            fig_error_trend = go.Figure()
            
            # Add existing error line
            if len(animated_df) > 1:
                fig_error_trend.add_trace(go.Scatter(
                    x=animated_df['date'][:-1],
                    y=animated_df['absolute_error'][:-1],
                    mode='lines+markers',
                    name='Absolute Error',
                    line=dict(color='#e74c3c', width=2),
                    marker=dict(size=4, color='#e74c3c', opacity=0.7)
                ))
            
            # Add current error point
            if len(animated_df) > 0:
                current_point = animated_df.iloc[-1]
                
                # Draw line segment from previous to current error
                if len(animated_df) > 1:
                    prev_point = animated_df.iloc[-2]
                    fig_error_trend.add_trace(go.Scatter(
                        x=[prev_point['date'], current_point['date']],
                        y=[prev_point['absolute_error'], current_point['absolute_error']],
                        mode='lines',
                        name='_Error Segment',
                        line=dict(color='#e74c3c', width=2.5),
                        showlegend=False
                    ))
                
                # Highlight current error point
                fig_error_trend.add_trace(go.Scatter(
                    x=[current_point['date']],
                    y=[current_point['absolute_error']],
                    mode='markers',
                    name='Current Error',
                    marker=dict(size=10, color='#e74c3c', symbol='circle',
                               line=dict(width=2, color='white'))
                ))
            
            # Add vertical line at high error point if detected
            if high_error_detected:
                fig_error_trend.add_shape(
                    type="line",
                    x0=first_high_error['date'],
                    y0=0,
                    x1=first_high_error['date'],
                    y1=1,
                    xref="x",
                    yref="paper",
                    line=dict(color="red", width=2, dash="dot")
                )
                fig_error_trend.add_annotation(
                    x=first_high_error['date'],
                    y=1,
                    xref="x",
                    yref="paper",
                    text="🚨 High Error!",
                    showarrow=False,
                    yshift=15,
                    bgcolor="red",
                    font=dict(color="white", size=12)
                )
            
            fig_error_trend.update_layout(
                title=f"📉 Live Error Trend - Point {idx+1}/{total_rows}",
                xaxis_title="Date",
                yaxis_title="Absolute Error",
                height=300,
                showlegend=True
            )
            st.plotly_chart(fig_error_trend, use_container_width=True)
        
        # Update table
        with table_placeholder.container():
            st.subheader("📋 Live Data Points")
            
            # Show only the current point and a few recent ones
            display_count = min(8, len(animated_df))
            recent_df = animated_df.tail(display_count).copy()
            
            # Create display dataframe
            display_df = recent_df.copy()
            display_df = display_df.rename(columns={
                'actual_sales': 'actual',
                'absolute_error': 'mae',
                'percentage_error': 'mape_percent'
            })
            
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
            display_df['prediction_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            display_df['rmse'] = (display_df['mae'] ** 2)
            
            # Add alert column
            for display_idx, (_, display_row) in enumerate(display_df.iterrows()):
                if display_row['mae'] > display_row['actual'] * 0.25:
                    display_df.at[display_row.name, 'alert'] = '🚨 HIGH_ERROR'
                else:
                    display_df.at[display_row.name, 'alert'] = '✅ OK'
            
            # Format for display
            log_columns = ['date', 'predicted', 'actual', 'mae', 'rmse', 'alert', 'model_type', 'prediction_timestamp']
            available_columns = [col for col in log_columns if col in display_df.columns]
            display_df = display_df[available_columns]
            
            formatted_df = display_df.copy()
            if 'predicted' in formatted_df.columns:
                formatted_df['predicted'] = formatted_df['predicted'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
            if 'actual' in formatted_df.columns:
                formatted_df['actual'] = formatted_df['actual'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
            if 'mae' in formatted_df.columns:
                formatted_df['mae'] = formatted_df['mae'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
            if 'rmse' in formatted_df.columns:
                formatted_df['rmse'] = formatted_df['rmse'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A")
            
            st.dataframe(formatted_df, use_container_width=True, height=280)
        
        # Update status
        with status_placeholder.container():
            current_date = row['date'].strftime('%Y-%m-%d')
            
            if high_error_detected:
                st.error(f"🚨 HIGH ERROR DETECTED at {current_date}!")
                error_percentage = (first_high_error['absolute_error'] / first_high_error['actual_sales']) * 100
                st.warning(f"MAE ${first_high_error['absolute_error']:,.0f} is {error_percentage:.1f}% of actual value")
                
                if email_enabled:
                    if email_sent:
                        st.success(f"✅ Email alert sent to {recipient_email}")
                    else:
                        st.error("❌ Failed to send email alert")
                
                # Quick countdown before stopping
                for i in range(2, 0, -1):
                    status_placeholder.warning(f"🛑 Stopping in {i}...")
                    time.sleep(0.5)
                break
            else:
                st.success(f"🔄 Processing point {idx+1}/{total_rows}")
                st.info(f"📅 Date: {current_date}")
        
        # Faster pause for quicker animation (0.8 seconds between points)
        time.sleep(0.8)
    
    # Final state after animation completes
    progress_bar.empty()
    
    if high_error_detected:
        st.error("🎯 MONITORING COMPLETE - High Error Detected!")
        st.warning("Processing stopped due to high prediction error.")
        
        # Show email status in final state
        if email_enabled:
            if email_sent:
                st.success(f"📧 Email alert was sent to {recipient_email}")
            else:
                st.error("📧 Failed to send email alert")
    else:
        st.success("🎉 MONITORING COMPLETE - All Points Processed Successfully!")
        st.balloons()
        
        # Show download options
        st.subheader("📥 Download Complete Dataset")
        col1, col2 = st.columns(2)
        
        download_df = animated_df.copy()
        download_df = download_df.rename(columns={
            'actual_sales': 'actual',
            'absolute_error': 'mae',
            'percentage_error': 'mape_percent'
        })
        download_df['date'] = download_df['date'].dt.strftime('%Y-%m-%d')
        download_df['prediction_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        download_df['rmse'] = (download_df['mae'] ** 2)
        download_df['alert'] = 'OK'
        
        with col1:
            csv_formatted = download_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Formatted Data",
                data=csv_formatted,
                file_name=f"monitoring_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
        
        with col2:
            csv_raw = download_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Raw Data",
                data=csv_raw,
                file_name=f"raw_monitoring_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

def test_email_system():
    """Test the email system independently"""
    try:
        # Test EmailAlert initialization
        email_alert = EmailAlert.get_instance()
        st.success("✅ EmailAlert initialized successfully")
        
        # Test EmailContent
        email_content = EmailContent()
        email_content.recipient = "test@example.com"  # Use a test email
        email_content.subject = "Test Email"
        email_content.message_body = {
            'title': 'Test',
            'message': 'This is a test message',
            'table_rows': {'Test': 'Value'},
            'notes': 'Test notes'
        }
        email_content.prepare_html()
        st.success("✅ EmailContent created successfully")
        
        return True
    except Exception as e:
        st.error(f"❌ Email system test failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return False


