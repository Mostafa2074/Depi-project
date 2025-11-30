import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
from predictions import batch_predict_mlflow, real_time_predict_mlflow, standardize_forecast_data

def run_forecast_app(model, prophet_df, model_type="unknown"):
    st.title("📈 Time Series Forecasting (MLflow Production Model)")
    
    if model is None:
        st.error("No production model found in MLflow Registry.")
        st.info("Please train and promote a model to Production stage first.")
        return
    
    # Display model info
    st.sidebar.header("Model Information")
    st.sidebar.info(f"**Model Type:** {model_type}")
    st.sidebar.info(f"**Source:** MLflow Model Registry (Production)")
    
    # Model info section
    with st.expander("ℹ️ Model Details"):
        st.write(f"**Loaded Model Type:** {model_type}")
        st.write("**Source:** MLflow Model Registry - Production Stage")
        if not prophet_df.empty:
            data_last_date = prophet_df['ds'].max().date()
            st.write(f"**Current Data Up To:** {data_last_date}")

    # -------------------------------------------------------------------------
    # MODE SELECTION
    # -------------------------------------------------------------------------
    prediction_mode = st.radio(
        "Prediction Mode",
        ["📦 Batch Predictions", "⚡ Real-Time Predictions"],
        horizontal=True
    )
    st.markdown("---")

    # ========================================================================
    # MODE 1: BATCH PREDICTIONS (UPDATED WITH DATE INPUT)
    # ========================================================================
    if prediction_mode == "📦 Batch Predictions":
        st.sidebar.header("Batch Forecast Settings")
        
        # Get prediction end date with default value 2017/9/15
        default_end_date = date(2017, 9, 15)
        
        # Calculate default periods for info display
        if not prophet_df.empty:
            last_data_date = prophet_df['ds'].max().date()
            default_periods = (default_end_date - last_data_date).days
            st.sidebar.info(f"Last data date: {last_data_date}")
            st.sidebar.info(f"Default forecast days: {default_periods}")
        
        forecast_end_date = st.sidebar.date_input(
            "Forecast End Date:",
            value=default_end_date,
            min_value=date.today() if prophet_df.empty else prophet_df['ds'].max().date() + timedelta(days=1),
            max_value=date(2030, 1, 1)
        )

        if st.sidebar.button("🚀 Run Batch Forecast"):
            with st.spinner(f'Generating forecast until {forecast_end_date}...'):
                try:
                    # Generate forecast using MLflow model
                    forecast = batch_predict_mlflow(model, model_type, forecast_end_date, prophet_df=prophet_df)
                    
                    if not forecast.empty:
                        # Standardize the forecast data
                        standardized_forecast = standardize_forecast_data(forecast, model_type)
                        
                        if not standardized_forecast.empty:
                            # Calculate actual periods for display
                            if not prophet_df.empty:
                                last_data_date = prophet_df['ds'].max().date()
                                actual_periods = (forecast_end_date - last_data_date).days
                            else:
                                actual_periods = len(standardized_forecast)
                            
                            # Store in session state
                            st.session_state.forecast_data = standardized_forecast
                            st.session_state.model_type = model_type
                            st.session_state.forecast_end_date = forecast_end_date
                            st.session_state.forecast_periods = actual_periods

                            st.success(f"✅ Forecast generated for {actual_periods} days until {forecast_end_date}")
                                
                        else:
                            st.error("Failed to standardize forecast data")
                    else:
                        st.error("Failed to generate forecast")
                        
                except Exception as e:
                    st.error(f"Error generating forecast: {e}")

        # DISPLAY RESULTS
        if st.session_state.get('forecast_data') is not None:
            display_mlflow_forecast_results(
                st.session_state.forecast_data,
                prophet_df,
                st.session_state.model_type,
                st.session_state.forecast_end_date,
                st.session_state.forecast_periods
            )

    # ========================================================================
    # MODE 2: REAL-TIME PREDICTIONS
    # ========================================================================
    elif prediction_mode == "⚡ Real-Time Predictions":
        col1, col2 = st.columns([2, 1])
        with col1:
            target_date = st.date_input("Target Date", value=datetime.now().date() + timedelta(days=1))
        with col2:
            st.markdown("###")
            if st.button("Predict"):
                # Pass prophet_df to real_time_predict_mlflow for ARIMA context
                res = real_time_predict_mlflow(model, model_type, pd.to_datetime(target_date), prophet_df=prophet_df)
                if 'real_time_predictions' not in st.session_state:
                    st.session_state.real_time_predictions = []
                st.session_state.real_time_predictions.insert(0, res)
        
        if st.session_state.get('real_time_predictions'):
            latest = st.session_state.real_time_predictions[0]
            
            if 'error' not in latest:
                # Metric Cards
                m1, m2 = st.columns(2)
                m1.metric("Predicted Sales", f"{latest['prediction']:,.0f}")
                m2.metric("Model Type", latest['model_type'])
                
                # Visualization
                if not prophet_df.empty:
                    fig_rt = go.Figure()
                    # History
                    fig_rt.add_trace(go.Scatter(
                        x=prophet_df['ds'], y=prophet_df['y'], 
                        mode='markers', name='Historical', 
                        marker=dict(color='blue', size=4)
                    ))
                    # Prediction
                    fig_rt.add_trace(go.Scatter(
                        x=[pd.to_datetime(latest['date'])], y=[latest['prediction']], 
                        mode='markers', name='Prediction', 
                        marker=dict(color='red', size=14, symbol='star')
                    ))
                    
                    fig_rt.update_layout(
                        title=f"Real-time Prediction using {model_type}",
                        xaxis_title="Date",
                        yaxis_title="Sales"
                    )
                    st.plotly_chart(fig_rt, use_container_width=True)
                
                with st.expander("JSON Response"):
                    st.json(latest)
            else:
                st.error(f"Prediction error: {latest.get('error', 'Unknown error')}")


def display_mlflow_forecast_results(forecast_data, prophet_df, model_type, end_date=None, periods=None):
    """Display forecast results for MLflow models."""
    
    # Ensure we have standardized data
    standardized_data = standardize_forecast_data(forecast_data, model_type)
    
    if standardized_data.empty:
        st.error("No valid forecast data to display.")
        return
    
    # 1. Header with forecast info
    if end_date and periods:
        st.subheader(f"Future Forecast Data (Until {end_date}, {periods} days)")
    else:
        st.subheader("Future Forecast Data")
    
    # Clean formatting for display
    display_df = standardized_data.copy()
    display_df = display_df.set_index('date')
    display_df.columns = ['Forecast']
    st.dataframe(display_df.style.format("{:,.0f}"), use_container_width=True)

    # 2. Chart
    st.subheader("Forecast Visualization")
    fig = go.Figure()

    # Actual Data (if available)
    if not prophet_df.empty:
        fig.add_trace(go.Scatter(
            x=prophet_df['ds'], 
            y=prophet_df['y'],
            mode='lines+markers',
            name='Historical Sales',
            line=dict(color='#1abc9c', width=2),  # Same color as forecast
            marker=dict(color='#1abc9c', size=4),
            opacity=0.4  # Slight fade to distinguish history
        ))


    # Forecast Line
    fig.add_trace(go.Scatter(
        x=standardized_data['date'], y=standardized_data['prediction'],
        mode='lines+markers', name=f'{model_type} Forecast',
        line=dict(color='#1abc9c', width=2),
        marker=dict(size=4)
    ))

    # Add vertical line separating history and forecast
    if not prophet_df.empty:
        last_historical_date_dt = pd.to_datetime(prophet_df['ds'].max())

        fig.add_shape(
            type="line",
            x0=last_historical_date_dt,
            x1=last_historical_date_dt,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="red", width=2, dash="dash")
        )

        fig.add_annotation(
            x=last_historical_date_dt,
            y=1,
            xref="x",
            yref="paper",
            text="Forecast Start",
            showarrow=False,
            yshift=10
        )

    fig.update_layout(
        title=f"Sales Forecast using {model_type}",
        xaxis_title="Date",
        yaxis_title="Sales"
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # 3. Download option
    csv = standardized_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Forecast as CSV",
        data=csv,
        file_name=f"forecast_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )
