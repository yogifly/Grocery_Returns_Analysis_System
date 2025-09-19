"""Report generation utilities"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import base64
from io import BytesIO
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from data_processor import DataProcessor
from survival_analysis import ShelfLifeAnalyzer
from predictive_models import RiskScorer

class ReportGenerator:
    """Generate comprehensive reports for grocery returns analysis"""
    
    def __init__(self):
        self.processor = DataProcessor()
        self.analyzer = ShelfLifeAnalyzer()
        self.risk_scorer = RiskScorer()
        
    def generate_executive_summary(self, df: pd.DataFrame) -> dict:
        """Generate executive summary metrics"""
        
        summary = {
            'total_orders': len(df),
            'total_returns': df['is_returned'].sum(),
            'return_rate': df['is_returned'].mean(),
            'stale_return_rate': (df['return_reason_category'] == 'stale').mean(),
            'avg_time_to_return': df['time_to_return_days'].mean(),
            'top_problem_products': df[df['is_returned']]['product_name'].value_counts().head(5).to_dict(),
            'worst_locations': df.groupby('delivered_location')['is_returned'].mean().nlargest(5).to_dict(),
            'carrier_performance': df.groupby('carrier_id')['is_returned'].mean().to_dict()
        }
        
        return summary
    
    def generate_product_report(self, df: pd.DataFrame, product_id: str) -> dict:
        """Generate detailed report for a specific product"""
        
        product_df = df[df['product_id'] == product_id]
        
        if len(product_df) == 0:
            return {"error": f"No data found for product {product_id}"}
        
        # Basic metrics
        report = {
            'product_id': product_id,
            'product_name': product_df['product_name'].iloc[0],
            'total_orders': len(product_df),
            'total_returns': product_df['is_returned'].sum(),
            'return_rate': product_df['is_returned'].mean(),
            'avg_time_to_return': product_df['time_to_return_days'].mean()
        }
        
        # Location analysis
        location_analysis = product_df.groupby('delivered_location').agg({
            'order_id': 'count',
            'is_returned': ['sum', 'mean']
        }).round(3)
        location_analysis.columns = ['orders', 'returns', 'return_rate']
        report['location_analysis'] = location_analysis.to_dict('index')
        
        # Seasonal analysis
        seasonal_analysis = product_df.groupby('season').agg({
            'order_id': 'count',
            'is_returned': ['sum', 'mean']
        }).round(3)
        seasonal_analysis.columns = ['orders', 'returns', 'return_rate']
        report['seasonal_analysis'] = seasonal_analysis.to_dict('index')
        
        # Return reasons
        return_reasons = product_df[product_df['is_returned']]['return_reason_category'].value_counts()
        report['return_reasons'] = return_reasons.to_dict()
        
        # Survival analysis
        try:
            shelf_life_estimates = self.analyzer.estimate_shelf_life(product_df)
            if not shelf_life_estimates.empty:
                report['shelf_life_estimates'] = shelf_life_estimates.to_dict('records')
        except Exception as e:
            report['shelf_life_error'] = str(e)
        
        return report
    
    def generate_location_report(self, df: pd.DataFrame, location: str) -> dict:
        """Generate detailed report for a specific location"""
        
        location_df = df[df['delivered_location'] == location]
        
        if len(location_df) == 0:
            return {"error": f"No data found for location {location}"}
        
        report = {
            'location': location,
            'total_orders': len(location_df),
            'total_returns': location_df['is_returned'].sum(),
            'return_rate': location_df['is_returned'].mean(),
            'avg_time_to_return': location_df['time_to_return_days'].mean()
        }
        
        # Product performance in this location
        product_analysis = location_df.groupby('product_name').agg({
            'order_id': 'count',
            'is_returned': ['sum', 'mean']
        }).round(3)
        product_analysis.columns = ['orders', 'returns', 'return_rate']
        product_analysis = product_analysis.sort_values('return_rate', ascending=False)
        report['product_analysis'] = product_analysis.to_dict('index')
        
        # Carrier performance in this location
        carrier_analysis = location_df.groupby('carrier_id').agg({
            'order_id': 'count',
            'is_returned': ['sum', 'mean'],
            'time_in_transit': 'mean'
        }).round(3)
        carrier_analysis.columns = ['orders', 'returns', 'return_rate', 'avg_transit_time']
        report['carrier_analysis'] = carrier_analysis.to_dict('index')
        
        return report
    
    def generate_risk_report(self, df: pd.DataFrame) -> dict:
        """Generate risk assessment report"""
        
        try:
            risk_df = self.risk_scorer.calculate_risk_scores(df)
            recommendations = self.risk_scorer.generate_recommendations(risk_df)
            
            report = {
                'risk_distribution': risk_df['risk_category'].value_counts().to_dict(),
                'high_risk_items': risk_df[risk_df['risk_category'] == 'High'].nlargest(10, 'risk_score')[
                    ['product_name', 'delivered_location', 'risk_score']
                ].to_dict('records'),
                'recommendations': recommendations[:10],  # Top 10 recommendations
                'avg_risk_by_product': risk_df.groupby('product_name')['risk_score'].mean().nlargest(10).to_dict(),
                'avg_risk_by_location': risk_df.groupby('delivered_location')['risk_score'].mean().nlargest(10).to_dict()
            }
            
            return report
            
        except Exception as e:
            return {"error": f"Error generating risk report: {str(e)}"}
    
    def export_to_excel(self, df: pd.DataFrame, filename: str = None) -> BytesIO:
        """Export analysis results to Excel file"""
        
        if filename is None:
            filename = f"grocery_returns_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Executive summary
            summary = self.generate_executive_summary(df)
            summary_df = pd.DataFrame([summary])
            summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)
            
            # Raw data
            df.to_excel(writer, sheet_name='Raw Data', index=False)
            
            # Product analysis
            product_summary = df.groupby('product_name').agg({
                'order_id': 'count',
                'is_returned': ['sum', 'mean'],
                'time_to_return_days': 'mean'
            }).round(3)
            product_summary.columns = ['Total Orders', 'Total Returns', 'Return Rate', 'Avg Time to Return']
            product_summary.to_excel(writer, sheet_name='Product Analysis')
            
            # Location analysis
            location_summary = df.groupby('delivered_location').agg({
                'order_id': 'count',
                'is_returned': ['sum', 'mean'],
                'time_to_return_days': 'mean'
            }).round(3)
            location_summary.columns = ['Total Orders', 'Total Returns', 'Return Rate', 'Avg Time to Return']
            location_summary.to_excel(writer, sheet_name='Location Analysis')
            
            # Return reasons
            return_reasons = df[df['is_returned']]['return_reason_category'].value_counts().reset_index()
            return_reasons.columns = ['Return Reason', 'Count']
            return_reasons.to_excel(writer, sheet_name='Return Reasons', index=False)
            
            # Risk assessment
            try:
                risk_report = self.generate_risk_report(df)
                if 'error' not in risk_report:
                    risk_summary_df = pd.DataFrame([risk_report['risk_distribution']])
                    risk_summary_df.to_excel(writer, sheet_name='Risk Summary', index=False)
                    
                    high_risk_df = pd.DataFrame(risk_report['high_risk_items'])
                    high_risk_df.to_excel(writer, sheet_name='High Risk Items', index=False)
            except Exception as e:
                print(f"Error adding risk assessment to Excel: {e}")
        
        output.seek(0)
        return output
    
    def create_dashboard_pdf(self, df: pd.DataFrame) -> BytesIO:
        """Create PDF dashboard report"""
        
        # This would require additional libraries like reportlab or weasyprint
        # For now, return a placeholder
        
        output = BytesIO()
        
        # Placeholder PDF content
        pdf_content = f"""
        Grocery Returns Analysis Report
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Executive Summary:
        - Total Orders: {len(df):,}
        - Total Returns: {df['is_returned'].sum():,}
        - Return Rate: {df['is_returned'].mean():.1%}
        - Stale Return Rate: {(df['return_reason_category'] == 'stale').mean():.1%}
        
        This is a placeholder for PDF generation.
        Full PDF functionality would require additional libraries.
        """
        
        output.write(pdf_content.encode('utf-8'))
        output.seek(0)
        
        return output

def create_download_link(data: BytesIO, filename: str, link_text: str) -> str:
    """Create download link for data"""
    
    b64 = base64.b64encode(data.read()).decode()
    data.seek(0)  # Reset buffer
    
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{link_text}</a>'
