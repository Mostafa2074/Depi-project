# [file name]: forecast_ui.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
from predictions import batch_predict_mlflow, real_time_predict_mlflow, standardize_forecast_data

def run_forecast_app(model, prophet_df, model_type="unknown"):
    st.title("📈 Time Series Forecasting (MLflow Production Model)")
    
    if model is None:
        st.error("❌ No production model found in MLflow Registry.")
        st.info("""
        **To use the Forecast Engine:**
        
        1. Go to **🔬 MLflow Tracking** tab
        2. Load your training dataset (`model_dataset.csv`)
        3. Train at least one model (Prophet, ARIMA, or LightGBM)
        4. Promote the best model to **Production** stage
        5. Return here to make forecasts
        """)
        
        with st.expander("📊 Current Data Overview"):
            if not prophet_df.empty:
                st.write("**Available Historical Data:**")
                st.write(f"- Date range: {prophet_df['ds'].min().date()} to {prophet_df['ds'].max().date()}")
                st.write(f"- Total records: {len(prophet_df):,}")
                st.write(f"- Sales range: ${prophet_df['y'].min():,.0f} to ${prophet_df['y'].max():,.0f}")
            else:
                st.warning("No historical data available for forecasting")
        
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
    # MODE 1: BATCH PREDICTIONS
    # ========================================================================
    if prediction_mode == "📦 Batch Predictions":
        st.subheader("📦 Batch Forecast Settings")
        
        # Get prediction end date with default value 2017/9/15
        default_end_date = date(2017, 9, 15)
        
        # Calculate default periods for info display
        if not prophet_df.empty:
            last_data_date = prophet_df['ds'].max().date()
            default_periods = (default_end_date - last_data_date).days
            st.info(f"📅 Last data date: {last_data_date}")
            st.info(f"📊 Default forecast days: {default_periods}")
        
        forecast_end_date = st.date_input(
            "Forecast End Date:",
            value=default_end_date,
            min_value=date.today() if prophet_df.empty else prophet_df['ds'].max().date() + timedelta(days=1),
            max_value=date(2030, 1, 1)
        )

        if st.button("🚀 Run Batch Forecast", type="primary"):
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
                            
                            # Display results immediately
                            display_mlflow_forecast_results(
                                standardized_forecast,
                                prophet_df,
                                model_type,
                                forecast_end_date,
                                actual_periods
                            )
                                
                        else:
                            st.error("Failed to standardize forecast data")
                    else:
                        st.error("Failed to generate forecast")
                        
                except Exception as e:
                    st.error(f"Error generating forecast: {e}")
                    import traceback
                    st.error(f"Detailed error: {traceback.format_exc()}")

        # Also display results if they exist in session state
        elif st.session_state.get('forecast_data') is not None:
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
        st.subheader("⚡ Real-Time Prediction")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            target_date = st.date_input("Target Date", value=datetime.now().date() + timedelta(days=1))
        with col2:
            st.markdown("###")
            if st.button("🔮 Predict", type="primary"):
                with st.spinner('Generating prediction...'):
                    # Pass prophet_df to real_time_predict_mlflow for context
                    res = real_time_predict_mlflow(model, model_type, pd.to_datetime(target_date), prophet_df=prophet_df)
                    if 'real_time_predictions' not in st.session_state:
                        st.session_state.real_time_predictions = []
                    st.session_state.real_time_predictions.insert(0, res)
        
        if st.session_state.get('real_time_predictions'):
            latest = st.session_state.real_time_predictions[0]
            
            if 'error' not in latest:
                # Metric Cards
                st.success("✅ Prediction generated successfully!")
                m1, m2 = st.columns(2)
                m1.metric("Predicted Sales", f"${latest['prediction']:,.0f}")
                m2.metric("Model Type", latest['model_type'])
                
                # Visualization with historical context
                display_real_time_prediction_with_history(latest, prophet_df, model_type)
                
                with st.expander("📋 Prediction Details"):
                    st.json(latest)
            else:
                st.error(f"❌ Prediction error: {latest.get('error', 'Unknown error')}")

def display_real_time_prediction_with_history(prediction_result, prophet_df, model_type):
    """Display real-time prediction with historical context"""
    if prophet_df.empty:
        st.warning("No historical data available for context")
        return
    
    # Create figure
    fig = go.Figure()
    
    # Show last 90 days of historical data for context
    recent_history = prophet_df.tail(90).copy()
    
    # Add historical sales line
    fig.add_trace(go.Scatter(
        x=recent_history['ds'], 
        y=recent_history['y'], 
        mode='lines+markers', 
        name='Historical Sales', 
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=4, color='#1f77b4'),
        opacity=0.8,
        hovertemplate='<b>Date:</b> %{x}<br><b>Sales:</b> $%{y:,.0f}<extra></extra>'
    ))
    
    # Add prediction point
    pred_date = pd.to_datetime(prediction_result['date'])
    pred_value = prediction_result['prediction']
    
    fig.add_trace(go.Scatter(
        x=[pred_date], 
        y=[pred_value], 
        mode='markers', 
        name=f'Prediction ({pred_date.strftime("%Y-%m-%d")})', 
        marker=dict(
            color='#ff7f0e', 
            size=16, 
            symbol='star',
            line=dict(width=2, color='white')
        ),
        hovertemplate=f'<b>Date:</b> {pred_date.strftime("%Y-%m-%d")}<br><b>Predicted Sales:</b> ${pred_value:,.0f}<extra></extra>'
    ))
    
    # Add vertical line at the last historical date
    last_historical_date = recent_history['ds'].max()
    fig.add_shape(
        type="line",
        x0=last_historical_date,
        x1=last_historical_date,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="red", width=2, dash="dot")
    )
    
    fig.add_annotation(
        x=last_historical_date,
        y=0.95,
        xref="x",
        yref="paper",
        text="Last Historical Data",
        showarrow=True,
        arrowhead=2,
        bgcolor="red",
        font=dict(color="white", size=10)
    )
    
    # Update layout
    fig.update_layout(
        title=f"Real-time Prediction using {model_type} Model",
        xaxis_title="Date",
        yaxis_title="Sales Amount ($)",
        height=500,
        showlegend=True,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_mlflow_forecast_results(forecast_data, prophet_df, model_type, end_date=None, periods=None):
    """Display forecast results for MLflow models with proper historical context."""
    
    st.subheader("📊 Forecast Results")
    
    # Ensure we have standardized data
    standardized_data = standardize_forecast_data(forecast_data, model_type)
    
    if standardized_data.empty:
        st.error("No valid forecast data to display.")
        return
    
    # 1. Forecast Summary
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Forecast Periods", f"{len(standardized_data)} days")
    with col2:
        avg_prediction = standardized_data['prediction'].mean()
        st.metric("Average Prediction", f"${avg_prediction:,.0f}")
    with col3:
        total_prediction = standardized_data['prediction'].sum()
        st.metric("Total Predicted Sales", f"${total_prediction:,.0f}")
    
    # 2. Data Table
    with st.expander("📋 Forecast Data Table", expanded=True):
        display_df = standardized_data.copy()
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        display_df['prediction'] = display_df['prediction'].round(2)
        display_df = display_df.rename(columns={'date': 'Date', 'prediction': 'Predicted Sales'})
        st.dataframe(display_df, use_container_width=True, height=300)

    # 3. Interactive Chart with Historical Context
    st.subheader("📈 Forecast Visualization")
    
    fig = create_forecast_chart_with_history(standardized_data, prophet_df, model_type)
    st.plotly_chart(fig, use_container_width=True)

    # 4. Download option
    st.subheader("📥 Download Forecast")
    col1, col2 = st.columns(2)
    
    with col1:
        # Formatted CSV
        download_df = standardized_data.copy()
        download_df['date'] = download_df['date'].dt.strftime('%Y-%m-%d')
        csv_formatted = download_df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv_formatted,
            file_name=f"forecast_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            type="primary"
        )
    
    with col2:
        # Raw data
        csv_raw = standardized_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Raw Data",
            data=csv_raw,
            file_name=f"raw_forecast_{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

def create_forecast_chart_with_history(forecast_data, prophet_df, model_type):
    """Create a comprehensive chart showing historical data and forecast together."""
    
    fig = go.Figure()

    # Historical Data (if available)
    if not prophet_df.empty:
        # Show last 180 days of historical data for better context
        recent_history = prophet_df.tail(180)
        
        fig.add_trace(go.Scatter(
            x=recent_history['ds'], 
            y=recent_history['y'],
            mode='lines+markers',
            name='Historical Sales',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=4, color='#1f77b4'),
            opacity=0.8,
            hovertemplate='<b>Date:</b> %{x}<br><b>Sales:</b> $%{y:,.0f}<extra></extra>'
        ))

    # Forecast Data
    if not forecast_data.empty:
        fig.add_trace(go.Scatter(
            x=forecast_data['date'], 
            y=forecast_data['prediction'],
            mode='lines+markers',
            name=f'{model_type} Forecast',
            line=dict(color='#ff7f0e', width=3, dash='dash'),
            marker=dict(size=5, color='#ff7f0e'),
            hovertemplate='<b>Date:</b> %{x}<br><b>Predicted Sales:</b> $%{y:,.0f}<extra></extra>'
        ))

    # Add vertical line separating history and forecast
    if not prophet_df.empty and not forecast_data.empty:
        last_historical_date = prophet_df['ds'].max()
        first_forecast_date = forecast_data['date'].min()

        # Add separation line
        fig.add_shape(
            type="line",
            x0=last_historical_date,
            x1=last_historical_date,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="red", width=3, dash="dot")
        )

        # Add annotation for forecast start
        fig.add_annotation(
            x=last_historical_date,
            y=0.95,
            xref="x",
            yref="paper",
            text="Forecast Start",
            showarrow=True,
            arrowhead=2,
            bgcolor="red",
            font=dict(color="white", size=12),
            yshift=10
        )

        # Add confidence interval if available (for Prophet models)
        if model_type.lower() == 'prophet' and 'yhat_lower' in forecast_data.columns and 'yhat_upper' in forecast_data.columns:
            fig.add_trace(go.Scatter(
                x=forecast_data['date'].tolist() + forecast_data['date'].tolist()[::-1],
                y=forecast_data['yhat_upper'].tolist() + forecast_data['yhat_lower'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(255, 127, 14, 0.2)',
                line=dict(color='rgba(255, 127, 14, 0)'),
                name='Confidence Interval',
                showlegend=True,
                hovertemplate='<b>Confidence Interval</b><extra></extra>'
            ))

    fig.update_layout(
        title=f"Sales Forecast using {model_type} Model",
        xaxis_title="Date",
        yaxis_title="Sales Amount ($)",
        height=600,
        showlegend=True,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig
