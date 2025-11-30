# Utility functions can be added here as needed
def check_model_performance(prophet_df, forecast_data, mae_percent_threshold=0.25):
    """Calculates MAE on historical data fit."""
    performance_df = prophet_df.merge(
        forecast_data[['ds', 'yhat']], 
        on='ds', 
        how='inner'
    )
    
    if not performance_df.empty:
        performance_df['abs_error'] = abs(performance_df['y'] - performance_df['yhat'])
        mae = performance_df['abs_error'].mean()
        
        mean_y = performance_df['y'].mean()
        threshold_value = mean_y * mae_percent_threshold
        
        if mae > threshold_value:
            alert_status = "ALERT: High MAE (Poor Fit)"
        else:
            alert_status = "Performance OK"
            
        return mae, threshold_value, alert_status
    
    return None, None, "Error: Historical data missing or merge failed."