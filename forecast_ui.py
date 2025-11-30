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
    
    # Create a combined dataset for the chart
    fig = go.Figure()
    
    # Get the last 90 days of historical data
    recent_history = prophet_df.tail(90).copy()
    
    # Add historical sales as a continuous line
    fig.add_trace(go.Scatter(
        x=recent_history['ds'], 
        y=recent_history['y'], 
        mode='lines+markers',
        name='Historical Sales', 
        line=dict(color='#2E86AB', width=3),
        marker=dict(size=5, color='#2E86AB'),
        hovertemplate='<b>Historical</b><br>Date: %{x}<br>Sales: $%{y:,.0f}<extra></extra>'
    ))
    
    # Add the prediction point
    pred_date = pd.to_datetime(prediction_result['date'])
    pred_value = prediction_result['prediction']
    
    # Draw a line from the last historical point to the prediction
    last_historical_point = recent_history.iloc[-1]
    fig.add_trace(go.Scatter(
        x=[last_historical_point['ds'], pred_date],
        y=[last_historical_point['y'], pred_value],
        mode='lines',
        name='Prediction Trend',
        line=dict(color='#A23B72', width=2, dash='dot'),
        showlegend=False,
        hovertemplate=False
    ))
    
    # Add the prediction point with emphasis
    fig.add_trace(go.Scatter(
        x=[pred_date], 
        y=[pred_value], 
        mode='markers+text',
        name=f'Prediction ({pred_date.strftime("%Y-%m-%d")})', 
        marker=dict(
            color='#F18F01', 
            size=20, 
            symbol='star',
            line=dict(width=3, color='white')
        ),
        text=[f"${pred_value:,.0f}"],
        textposition="top center",
        textfont=dict(size=14, color='#F18F01'),
        hovertemplate=f'<b>Prediction</b><br>Date: {pred_date.strftime("%Y-%m-%d")}<br>Sales: ${pred_value:,.0f}<extra></extra>'
    ))
    
    # Update layout
    fig.update_layout(
        title=f"📊 Real-time Sales Prediction using {model_type}",
        xaxis_title="Date",
        yaxis_title="Sales Amount ($)",
        height=500,
        showlegend=True,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
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
    st.subheader("📈 Historical Sales vs Forecast")
    
    # Create the combined chart
    fig = create_combined_historical_forecast_chart(prophet_df, standardized_data, model_type)
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

def create_combined_historical_forecast_chart(historical_df, forecast_df, model_type):
    """Create a comprehensive chart that properly combines historical and forecast data."""
    
    fig = go.Figure()

    # HISTORICAL DATA - Show last 180 days for context
    if not historical_df.empty:
        recent_history = historical_df.tail(180)
        
        # Add historical sales line
        fig.add_trace(go.Scatter(
            x=recent_history['ds'], 
            y=recent_history['y'],
            mode='lines+markers',
            name='📊 Historical Sales',
            line=dict(color='#2E86AB', width=4),
            marker=dict(size=5, color='#2E86AB'),
            hovertemplate='<b>Historical</b><br>Date: %{x}<br>Sales: $%{y:,.0f}<extra></extra>'
        ))

    # FORECAST DATA
    if not forecast_df.empty:
        # Connect the last historical point to the first forecast point
        if not historical_df.empty:
            last_historical_point = historical_df.iloc[-1]
            first_forecast_point = forecast_df.iloc[0]
            
            # Add connecting line
            fig.add_trace(go.Scatter(
                x=[last_historical_point['ds'], first_forecast_point['date']],
                y=[last_historical_point['y'], first_forecast_point['prediction']],
                mode='lines',
                name='Transition',
                line=dict(color='#A23B72', width=3, dash='dash'),
                showlegend=False,
                hovertemplate=False
            ))

        # Add forecast line
        fig.add_trace(go.Scatter(
            x=forecast_df['date'], 
            y=forecast_df['prediction'],
            mode='lines+markers',
            name=f'🎯 {model_type} Forecast',
            line=dict(color='#F18F01', width=4),
            marker=dict(size=6, color='#F18F01', symbol='diamond'),
            hovertemplate='<b>Forecast</b><br>Date: %{x}<br>Predicted Sales: $%{y:,.0f}<extra></extra>'
        ))

    # Add clear separation between historical and forecast
    if not historical_df.empty and not forecast_df.empty:
        last_historical_date = historical_df['ds'].max()
        
        # Add vertical separation line
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

        # Add separation annotation
        fig.add_annotation(
            x=last_historical_date,
            y=0.02,
            xref="x",
            yref="paper",
            text="FORECAST START",
            showarrow=False,
            bgcolor="red",
            font=dict(color="white", size=12, weight="bold"),
            yshift=-10
        )

    # Update layout for better visualization
    fig.update_layout(
        title={
            'text': f"📈 Sales Timeline: Historical Data vs {model_type} Forecast",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        xaxis_title="Date",
        yaxis_title="Sales Amount ($)",
        height=600,
        showlegend=True,
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='black',
            borderwidth=1
        ),
        plot_bgcolor='rgba(240,240,240,0.8)'
    )

    # Add grid for better readability
    fig.update_xaxis(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxis(showgrid=True, gridwidth=1, gridcolor='lightgray')
    
    return fig
