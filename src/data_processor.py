"""Data processing pipeline for grocery returns analysis"""

import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
from config.settings import *

class DataProcessor:
    """Main data processing class for grocery returns analysis"""
    
    def __init__(self):
        self.orders_df = None
        self.returns_df = None
        self.products_df = None
        self.processed_df = None
        
    def load_data(self) -> None:
        """Load data from CSV files"""
        try:
            self.orders_df = pd.read_csv(ORDERS_FILE)
            self.returns_df = pd.read_csv(RETURNS_FILE)
            self.products_df = pd.read_csv(PRODUCTS_FILE)
            
            # Convert date columns
            self.orders_df['order_date'] = pd.to_datetime(self.orders_df['order_date'])
            self.orders_df['delivery_date'] = pd.to_datetime(self.orders_df['delivery_date'])
            self.returns_df['return_date'] = pd.to_datetime(self.returns_df['return_date'])
            
            print(f"✅ Data loaded successfully!")
            print(f"Orders: {len(self.orders_df)}, Returns: {len(self.returns_df)}, Products: {len(self.products_df)}")
            
        except FileNotFoundError as e:
            print(f"❌ Error loading data: {e}")
            print("Please run the data generation script first.")
            
    def clean_return_reasons(self) -> None:
        """Clean and categorize return reasons using rule-based mapping"""
        
        def map_reason(text):
            if not isinstance(text, str):
                return 'unknown'
            
            text_lower = text.lower()
            
            # Rule-based mapping
            if re.search(r'stale|rotten|mould|mold|spoilt|spoiled', text_lower):
                return 'stale'
            elif re.search(r'damage|broken|bruised|crushed', text_lower):
                return 'damaged'
            elif re.search(r'quality|fresh|old', text_lower):
                return 'quality_issues'
            elif re.search(r'wrong|incorrect|different', text_lower):
                return 'wrong_item'
            elif re.search(r'packaging|package|leaked', text_lower):
                return 'packaging_issues'
            elif re.search(r'late|delay', text_lower):
                return 'late_delivery'
            else:
                return 'other'
        
        # Apply mapping if return_reason_category doesn't exist
        if 'return_reason_category' not in self.returns_df.columns:
            self.returns_df['return_reason_category'] = self.returns_df['return_reason'].apply(map_reason)
            
    def create_master_dataset(self) -> pd.DataFrame:
        """Create master dataset by joining orders, returns, and products"""
        
        # Start with orders
        master_df = self.orders_df.copy()
        
        # Add product information
        master_df = master_df.merge(
            self.products_df[['product_id', 'category', 'expected_shelf_life_days']], 
            on='product_id', 
            how='left'
        )
        
        # Add return information (left join to keep all orders)
        master_df = master_df.merge(
            self.returns_df, 
            on='order_id', 
            how='left'
        )
        
        # Create derived features
        master_df['is_returned'] = master_df['return_id'].notna()
        master_df['time_to_return_days'] = (
            master_df['return_date'] - master_df['delivery_date']
        ).dt.days
        
        # Time-based features
        master_df['delivery_month'] = master_df['delivery_date'].dt.month
        master_df['delivery_quarter'] = master_df['delivery_date'].dt.quarter
        master_df['delivery_weekday'] = master_df['delivery_date'].dt.day_name()
        master_df['season'] = master_df['delivery_month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
        })
        
        # Transit time
        master_df['time_in_transit'] = (
            master_df['delivery_date'] - master_df['order_date']
        ).dt.days
        
        # Temperature sensitivity flag
        master_df['temp_sensitive'] = master_df['product_id'].isin(TEMP_SENSITIVE_PRODUCTS)
        
        # High risk location flag
        master_df['high_risk_location'] = master_df['delivered_location'].isin(HIGH_RISK_CITIES)
        
        self.processed_df = master_df
        return master_df
    
    def calculate_aggregates(self) -> Dict[str, pd.DataFrame]:
        """Calculate various aggregate metrics"""
        
        if self.processed_df is None:
            raise ValueError("Please create master dataset first")
        
        aggregates = {}
        
        # Overall metrics
        aggregates['overall'] = pd.DataFrame([{
            'total_orders': len(self.processed_df),
            'total_returns': self.processed_df['is_returned'].sum(),
            'return_rate': self.processed_df['is_returned'].mean(),
            'avg_time_to_return': self.processed_df['time_to_return_days'].mean(),
            'stale_returns': (self.processed_df['return_reason_category'] == 'stale').sum(),
            'stale_return_rate': (self.processed_df['return_reason_category'] == 'stale').mean()
        }])
        
        # Product-level aggregates
        product_agg = self.processed_df.groupby('product_id').agg({
            'order_id': 'count',
            'is_returned': ['sum', 'mean'],
            'time_to_return_days': 'mean',
            'return_reason_category': lambda x: (x == 'stale').sum()
        }).round(3)
        
        product_agg.columns = ['total_orders', 'total_returns', 'return_rate', 
                              'avg_time_to_return', 'stale_returns']
        product_agg = product_agg.reset_index()
        aggregates['by_product'] = product_agg
        
        # Location-level aggregates
        location_agg = self.processed_df.groupby('delivered_location').agg({
            'order_id': 'count',
            'is_returned': ['sum', 'mean'],
            'time_to_return_days': 'mean',
            'return_reason_category': lambda x: (x == 'stale').sum()
        }).round(3)
        
        location_agg.columns = ['total_orders', 'total_returns', 'return_rate', 
                               'avg_time_to_return', 'stale_returns']
        location_agg = location_agg.reset_index()
        aggregates['by_location'] = location_agg
        
        # Carrier performance
        carrier_agg = self.processed_df.groupby('carrier_id').agg({
            'order_id': 'count',
            'is_returned': ['sum', 'mean'],
            'time_to_return_days': 'mean',
            'time_in_transit': 'mean'
        }).round(3)
        
        carrier_agg.columns = ['total_orders', 'total_returns', 'return_rate', 
                              'avg_time_to_return', 'avg_transit_time']
        carrier_agg = carrier_agg.reset_index()
        aggregates['by_carrier'] = carrier_agg
        
        # Monthly trends
        monthly_agg = self.processed_df.groupby(['delivery_month', 'product_id']).agg({
            'order_id': 'count',
            'is_returned': ['sum', 'mean']
        }).round(3)
        
        monthly_agg.columns = ['total_orders', 'total_returns', 'return_rate']
        monthly_agg = monthly_agg.reset_index()
        aggregates['monthly_trends'] = monthly_agg
        
        return aggregates
    
    def detect_anomalies(self, threshold_multiplier: float = 2.0) -> pd.DataFrame:
        """Detect anomalies in return rates using statistical methods"""
        
        if self.processed_df is None:
            raise ValueError("Please create master dataset first")
        
        # Calculate return rates by product-location combination
        anomaly_df = self.processed_df.groupby(['product_id', 'delivered_location']).agg({
            'order_id': 'count',
            'is_returned': ['sum', 'mean']
        }).round(3)
        
        anomaly_df.columns = ['total_orders', 'total_returns', 'return_rate']
        anomaly_df = anomaly_df.reset_index()
        
        # Filter for sufficient sample size
        anomaly_df = anomaly_df[anomaly_df['total_orders'] >= MIN_SAMPLE_SIZE]
        
        # Calculate z-scores for return rates
        mean_return_rate = anomaly_df['return_rate'].mean()
        std_return_rate = anomaly_df['return_rate'].std()
        
        anomaly_df['z_score'] = (anomaly_df['return_rate'] - mean_return_rate) / std_return_rate
        anomaly_df['is_anomaly'] = np.abs(anomaly_df['z_score']) > threshold_multiplier
        
        # Sort by z-score (highest first)
        anomaly_df = anomaly_df.sort_values('z_score', ascending=False)
        
        return anomaly_df
    
    def get_return_reason_distribution(self) -> pd.DataFrame:
        """Get distribution of return reasons"""
        
        if self.returns_df is None:
            raise ValueError("Please load data first")
        
        reason_dist = self.returns_df['return_reason_category'].value_counts().reset_index()
        reason_dist.columns = ['return_reason', 'count']
        reason_dist['percentage'] = (reason_dist['count'] / reason_dist['count'].sum() * 100).round(1)
        
        return reason_dist
    
    def process_all(self) -> Dict:
        """Run complete data processing pipeline"""
        
        print("🔄 Starting data processing pipeline...")
        
        # Load data
        self.load_data()
        
        # Clean return reasons
        self.clean_return_reasons()
        
        # Create master dataset
        master_df = self.create_master_dataset()
        
        # Calculate aggregates
        aggregates = self.calculate_aggregates()
        
        # Detect anomalies
        anomalies = self.detect_anomalies()
        
        # Get return reason distribution
        reason_dist = self.get_return_reason_distribution()
        
        print("✅ Data processing completed!")
        
        return {
            'master_df': master_df,
            'aggregates': aggregates,
            'anomalies': anomalies,
            'reason_distribution': reason_dist
        }

# Utility functions
def filter_data_by_date(df: pd.DataFrame, start_date: str, end_date: str, date_column: str = 'delivery_date') -> pd.DataFrame:
    """Filter dataframe by date range"""
    mask = (df[date_column] >= start_date) & (df[date_column] <= end_date)
    return df.loc[mask]

def filter_data_by_product(df: pd.DataFrame, products: List[str]) -> pd.DataFrame:
    """Filter dataframe by product list"""
    return df[df['product_id'].isin(products)]

def filter_data_by_location(df: pd.DataFrame, locations: List[str]) -> pd.DataFrame:
    """Filter dataframe by location list"""
    return df[df['delivered_location'].isin(locations)]
