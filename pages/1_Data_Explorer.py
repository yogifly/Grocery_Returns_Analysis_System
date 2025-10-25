"""Data exploration page"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from data_processor import DataProcessor

st.set_page_config(
    page_title="Data Explorer",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Data Explorer")

@st.cache_data
def load_data():
    processor = DataProcessor()
    return processor.process_all()

# Load data
try:
    data = load_data()
    df = data['master_df']
    
    st.success(f"Data loaded successfully! {len(df):,} records")
    
    # Data overview
    st.header("📊 Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Orders", f"{len(df):,}")
    
    with col2:
        st.metric("Unique Products", f"{df['product_id'].nunique()}")
    
    with col3:
        st.metric("Locations", f"{df['delivered_location'].nunique()}")
    
    with col4:
        st.metric("Date Range", f"{(df['delivery_date'].max() - df['delivery_date'].min()).days} days")
    
    # Raw data viewer
    st.header("📋 Raw Data")
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        selected_columns = st.multiselect(
            "Select Columns to Display",
            df.columns.tolist(),
            default=['order_id', 'product_name', 'delivered_location', 'delivery_date', 'is_returned']
        )
    
    with col2:
        max_rows = st.slider("Max Rows to Display", 10, 1000, 100)
    
    if selected_columns:
        st.dataframe(df[selected_columns].head(max_rows), use_container_width=True)
    
    # Statistical summary
    st.header("📈 Statistical Summary")
    
    numeric_columns = df.select_dtypes(include=['number']).columns
    if len(numeric_columns) > 0:
        st.dataframe(df[numeric_columns].describe(), use_container_width=True)
    
    # Distribution plots
    st.header("📊 Data Distributions")
    
    selected_column = st.selectbox(
        "Select Column for Distribution Plot",
        numeric_columns
    )
    
    if selected_column:
        fig = px.histogram(
            df,
            x=selected_column,
            title=f"Distribution of {selected_column}",
            nbins=30
        )
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Please run the data generation script first: `python scripts/generate_data.py`")
