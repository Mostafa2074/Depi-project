def display_mlflow_forecast_results(forecast_data, prophet_df, model_type, end_date=None, periods=None):
    """Display forecast results for MLflow models."""
    
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

    # 3. Interactive Chart - UPDATED TO MATCH SCREENSHOT
    st.subheader("📈 Forecast Visualization")
    
    fig = go.Figure()

    # Historical Data (if available) - using blue color from screenshot
    if not prophet_df.empty:
        fig.add_trace(go.Scatter(
            x=prophet_df['ds'], 
            y=prophet_df['y'],
            mode='lines',
            name='Historical Sales',
            line=dict(color='#1f77b4', width=2),
            opacity=0.8
        ))

    # Forecast Data - using orange color from screenshot
    if not standardized_data.empty:
        fig.add_trace(go.Scatter(
            x=standardized_data['date'], 
            y=standardized_data['prediction'],
            mode='lines',
            name=f'{model_type} Forecast',
            line=dict(color='#ff7f0e', width=2),
            opacity=0.8
        ))

    # Update layout to match screenshot style
    fig.update_layout(
        title=f"Sales Forecast using {model_type.lower()}",
        xaxis_title="Date",
        yaxis_title="Sales",
        height=500,
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='white',
        # Y-axis formatting to show "M" for millions
        yaxis=dict(
            tickformat='.1f',
            tickprefix='$',
            ticksuffix='M',
            # Adjust range to match typical sales data (0-1.5M)
            range=[0, max(prophet_df['y'].max() if not prophet_df.empty else 0, 
                         standardized_data['prediction'].max() if not standardized_data.empty else 0) * 1.1]
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgray'
        ),
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="black"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Add grid for better readability
    fig.update_yaxes(showgrid=True, gridcolor='lightgray', gridwidth=1)
    fig.update_xaxes(showgrid=True, gridcolor='lightgray', gridwidth=1)
    
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
            label="📥 Download Forecast as CSV",
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
