import streamlit as st
import pandas as pd
import plotly.express as px

def run_dashboard(train, min_date, max_date, sort_state):
    st.title("🛒️ Store Sales Forecasting Project")
    
    if train.empty:
        st.error("No data available. Please ensure 'Data.zip' is in the directory.")
        return

    # Metrics Section
    st.subheader("Data Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{train.shape[0]:,}")
    col2.metric("States", train['state'].nunique() if 'state' in train.columns else 0)
    col3.metric("Cities", train['city'].nunique() if 'city' in train.columns else 0)
    col4.metric("Stores", train['store_nbr'].nunique() if 'store_nbr' in train.columns else 0)
    st.markdown("---")

    # Controls
    st.subheader("Dashboard Controls")
    col_chosen = st.multiselect(
        "Choose State",
        options=sort_state.index.tolist(),
        default=sort_state.index.tolist()[:1] if not sort_state.empty else [],
        placeholder="Select states to filter"
    )

    date_range = st.date_input(
        "Select Date Range",
        value=[min_date, max_date],
        min_value=pd.Timestamp('2000-01-01').date(), 
        max_value=pd.Timestamp('2100-01-01').date()
    )

    # Handle Date Input logic
    if isinstance(date_range, tuple):
        if len(date_range) == 2:
            start_date, end_date = date_range
        elif len(date_range) == 1:
            start_date, end_date = date_range[0], date_range[0]
        else:
            start_date, end_date = min_date, max_date
    else:
         start_date, end_date = min_date, max_date

    # Filter Logic
    if col_chosen and 'sales' in train.columns:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        mask = (
            (train['state'].isin(col_chosen)) &
            (train.index >= start_date) &
            (train.index <= end_date)
        )
        filtered_df = train[mask]

        if not filtered_df.empty:
            agg_method = st.radio("Aggregation", ['Sum', 'Mean'], horizontal=True)
            
            if agg_method == 'Mean':
                city_sales = filtered_df.groupby('city')['sales'].mean().reset_index()
            else:
                city_sales = filtered_df.groupby('city')['sales'].sum().reset_index()

            fig = px.bar(
                city_sales, x='city', y='sales',
                title=f'City Sales ({agg_method})',
                color='sales',
                color_continuous_scale='Teal'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data found for these filters.")
    else:
        st.info("Select a state to view chart.")