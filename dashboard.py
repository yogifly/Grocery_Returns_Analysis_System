"""Main Streamlit dashboard for grocery returns analysis"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processor import DataProcessor
from survival_analysis import ShelfLifeAnalyzer, analyze_product_shelf_life
from predictive_models import ReturnPredictionModel, RiskScorer
from nlp_classifier import ReturnReasonClassifier
from config.settings import *

# Page configuration
st.set_page_config(
    page_title="Grocery Returns Analysis",
    page_icon="🥬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
    }
    .alert-high {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #f5c6cb;
    }
    .alert-medium {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #ffeaa7;
    }
    .alert-low {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    """Load and process data with caching"""
    processor = DataProcessor()
    return processor.process_all()

@st.cache_data
def get_survival_analysis(df, product_id):
    """Get survival analysis with caching"""
    return analyze_product_shelf_life(df, product_id)

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">🥬 Grocery Returns Analysis </h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        try:
            data = load_and_process_data()
            df = data['master_df']
            aggregates = data['aggregates']
            anomalies = data['anomalies']
            reason_dist = data['reason_distribution']
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.info("Please run the data generation script first: `python scripts/generate_data.py`")
            return
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Date range filter
    min_date = df['delivery_date'].min().date()
    max_date = df['delivery_date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Product filter
    products = ['All'] + sorted(df['product_id'].unique().tolist())
    selected_products = st.sidebar.multiselect(
        "Select Products",
        products,
        default=['All']
    )
    
    # Location filter
    locations = ['All'] + sorted(df['delivered_location'].unique().tolist())
    selected_locations = st.sidebar.multiselect(
        "Select Locations",
        locations,
        default=['All']
    )
    
    # Carrier filter
    carriers = ['All'] + sorted(df['carrier_id'].unique().tolist())
    selected_carriers = st.sidebar.multiselect(
        "Select Carriers",
        carriers,
        default=['All']
    )
    
    # Apply filters
    filtered_df = df.copy()
    
    # Date filter
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['delivery_date'].dt.date >= start_date) &
            (filtered_df['delivery_date'].dt.date <= end_date)
        ]
    
    # Product filter
    if 'All' not in selected_products and selected_products:
        filtered_df = filtered_df[filtered_df['product_id'].isin(selected_products)]
    
    # Location filter
    if 'All' not in selected_locations and selected_locations:
        filtered_df = filtered_df[filtered_df['delivered_location'].isin(selected_locations)]
    
    # Carrier filter
    if 'All' not in selected_carriers and selected_carriers:
        filtered_df = filtered_df[filtered_df['carrier_id'].isin(selected_carriers)]
    
    # Main dashboard tabs
    tab1, tab2, tab5 = st.tabs([
        "📊 Overview", 
        "🔍 Deep Dive",   
        "⚠️ Alerts & Recommendations"
    ])
    
    with tab1:
        show_overview(filtered_df, aggregates)
    
    with tab2:
        show_deep_dive(filtered_df)

    
    with tab5:
        show_alerts_recommendations(filtered_df, anomalies)

def show_overview(df, aggregates):
    """Show overview dashboard"""
    
    st.header("📊 Executive Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_orders = len(df)
    total_returns = df['is_returned'].sum()
    return_rate = df['is_returned'].mean() * 100
    stale_returns = (df['return_reason_category'] == 'stale').sum()
    stale_rate = (df['return_reason_category'] == 'stale').mean() * 100
    
    with col1:
        st.metric("Total Orders", f"{total_orders:,}")
    
    with col2:
        st.metric("Total Returns", f"{total_returns:,}")
    
    with col3:
        st.metric("Return Rate", f"{return_rate:.1f}%")
    
    with col4:
        st.metric("Stale Return Rate", f"{stale_rate:.1f}%")
    
    # Charts row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Return Rate by Product")
        product_returns = df.groupby('product_name').agg({
            'order_id': 'count',
            'is_returned': 'mean'
        }).reset_index()
        product_returns['return_rate'] = product_returns['is_returned'] * 100
        product_returns = product_returns.sort_values('return_rate', ascending=True)
        
        fig = px.bar(
            product_returns,
            x='return_rate',
            y='product_name',
            orientation='h',
            title="Return Rate by Product (%)",
            color='return_rate',
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Return Reasons Distribution")
        reason_counts = df[df['is_returned']]['return_reason_category'].value_counts()
        
        fig = px.pie(
            values=reason_counts.values,
            names=reason_counts.index,
            title="Distribution of Return Reasons"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Charts row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Return Rate by Location")
        location_returns = df.groupby('delivered_location').agg({
            'order_id': 'count',
            'is_returned': 'mean'
        }).reset_index()
        location_returns['return_rate'] = location_returns['is_returned'] * 100
        location_returns = location_returns.sort_values('return_rate', ascending=False)
        
        fig = px.bar(
            location_returns,
            x='delivered_location',
            y='return_rate',
            title="Return Rate by Location (%)",
            color='return_rate',
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Monthly Return Trends")
        monthly_data = df.groupby(df['delivery_date'].dt.to_period('M')).agg({
            'order_id': 'count',
            'is_returned': 'mean'
        }).reset_index()
        monthly_data['return_rate'] = monthly_data['is_returned'] * 100
        monthly_data['month'] = monthly_data['delivery_date'].astype(str)
        
        fig = px.line(
            monthly_data,
            x='month',
            y='return_rate',
            title="Monthly Return Rate Trend (%)",
            markers=True
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def show_deep_dive(df):
    """Show detailed analysis"""
    
    st.header("🔍 Deep Dive Analysis")
    
    # Product selection for deep dive
    selected_product = st.selectbox(
        "Select Product for Deep Dive",
        df['product_id'].unique()
    )
    
    product_df = df[df['product_id'] == selected_product]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Product Return Rate",
            f"{product_df['is_returned'].mean() * 100:.1f}%"
        )
    
    with col2:
        st.metric(
            "Avg Time to Return",
            f"{product_df['time_to_return_days'].mean():.1f} days"
        )
    
    with col3:
        st.metric(
            "Total Orders",
            f"{len(product_df):,}"
        )
    
    # Detailed charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Return Rate by Location")
        location_analysis = product_df.groupby('delivered_location').agg({
            'order_id': 'count',
            'is_returned': 'mean'
        }).reset_index()
        location_analysis['return_rate'] = location_analysis['is_returned'] * 100
        
        fig = px.bar(
            location_analysis,
            x='delivered_location',
            y='return_rate',
            title=f"Return Rate by Location - {selected_product}",
            color='return_rate',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Return Rate by Carrier")
        carrier_analysis = product_df.groupby('carrier_id').agg({
            'order_id': 'count',
            'is_returned': 'mean'
        }).reset_index()
        carrier_analysis['return_rate'] = carrier_analysis['is_returned'] * 100
        
        fig = px.bar(
            carrier_analysis,
            x='carrier_id',
            y='return_rate',
            title=f"Return Rate by Carrier - {selected_product}",
            color='return_rate',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal analysis
    st.subheader("Seasonal Analysis")
    seasonal_analysis = product_df.groupby('season').agg({
        'order_id': 'count',
        'is_returned': 'mean'
    }).reset_index()
    seasonal_analysis['return_rate'] = seasonal_analysis['is_returned'] * 100
    
    fig = px.bar(
        seasonal_analysis,
        x='season',
        y='return_rate',
        title=f"Seasonal Return Rate - {selected_product}",
        color='return_rate',
        color_continuous_scale='Reds'
    )
    st.plotly_chart(fig, use_container_width=True)


def show_alerts_recommendations(df, anomalies):
    """Show alerts and recommendations"""
    
    st.header("⚠️ Alerts & Recommendations")
    
    # Anomaly detection
    st.subheader("Return Rate Anomalies")
    
    if not anomalies.empty:
        # High anomalies
        high_anomalies = anomalies[anomalies['z_score'] > 2].head(10)
        
        if not high_anomalies.empty:
            st.markdown("### 🔴 High Priority Alerts")
            
            for _, row in high_anomalies.iterrows():
                alert_class = "alert-high" if row['z_score'] > 3 else "alert-medium"
                
                st.markdown(f"""
                <div class="{alert_class}">
                    <strong>{row['product_id']} in {row['delivered_location']}</strong><br>
                    Return Rate: {row['return_rate']:.1%} (Z-score: {row['z_score']:.2f})<br>
                    Sample Size: {row['total_orders']} orders, {row['total_returns']} returns
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
    
    # Performance summary
    st.subheader("Carrier Performance Summary")
    
    carrier_perf = df.groupby('carrier_id').agg({
        'order_id': 'count',
        'is_returned': 'mean',
        'time_in_transit': 'mean'
    }).reset_index()
    
    carrier_perf.columns = ['Carrier', 'Total Orders', 'Return Rate', 'Avg Transit Time']
    carrier_perf['Return Rate'] = carrier_perf['Return Rate'] * 100
    
    st.dataframe(
        carrier_perf.style.format({
            'Return Rate': '{:.1f}%',
            'Avg Transit Time': '{:.1f} days'
        }).background_gradient(subset=['Return Rate'], cmap='Reds'),
        use_container_width=True
    )

if __name__ == "__main__":
    main()
