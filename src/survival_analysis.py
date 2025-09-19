"""Survival analysis for shelf-life estimation"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional
from config.settings import SHELF_LIFE_CONFIDENCE

class ShelfLifeAnalyzer:
    """Survival analysis for estimating product shelf-life"""
    
    def __init__(self):
        self.kmf = KaplanMeierFitter()
        self.cph = CoxPHFitter()
        
    def prepare_survival_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for survival analysis"""
        
        # Filter for returned items with stale reason
        stale_returns = df[df['return_reason_category'] == 'stale'].copy()
        
        # For survival analysis, we need:
        # - Duration: time from delivery to return (for returned items)
        # - Event: whether the item was returned as stale (1) or not (0)
        
        survival_df = df.copy()
        
        # Calculate duration (days from delivery to return or censoring)
        survival_df['duration'] = np.where(
            survival_df['is_returned'] & (survival_df['return_reason_category'] == 'stale'),
            survival_df['time_to_return_days'],
            survival_df['expected_shelf_life_days']  # Use expected shelf life for censored observations
        )
        
        # Event indicator (1 if returned as stale, 0 if censored)
        survival_df['event'] = (
            survival_df['is_returned'] & 
            (survival_df['return_reason_category'] == 'stale')
        ).astype(int)
        
        # Remove invalid durations
        survival_df = survival_df[survival_df['duration'] > 0]
        
        return survival_df
    
    def fit_kaplan_meier(self, df: pd.DataFrame, group_by: Optional[str] = None) -> Dict:
        """Fit Kaplan-Meier survival curves"""
        
        survival_df = self.prepare_survival_data(df)
        
        results = {}
        
        if group_by is None:
            # Overall survival curve
            self.kmf.fit(
                durations=survival_df['duration'],
                event_observed=survival_df['event'],
                label='Overall'
            )
            
            results['overall'] = {
                'kmf': self.kmf,
                'median_survival': self.kmf.median_survival_time_,
                'confidence_interval': self.kmf.confidence_interval_survival_function_,
                'survival_function': self.kmf.survival_function_
            }
            
        else:
            # Group-wise survival curves
            groups = survival_df[group_by].unique()
            
            for group in groups:
                group_data = survival_df[survival_df[group_by] == group]
                
                if len(group_data) < 5:  # Skip groups with too few observations
                    continue
                
                kmf_group = KaplanMeierFitter()
                kmf_group.fit(
                    durations=group_data['duration'],
                    event_observed=group_data['event'],
                    label=str(group)
                )
                
                results[str(group)] = {
                    'kmf': kmf_group,
                    'median_survival': kmf_group.median_survival_time_,
                    'confidence_interval': kmf_group.confidence_interval_survival_function_,
                    'survival_function': kmf_group.survival_function_
                }
        
        return results
    
    def compare_survival_curves(self, df: pd.DataFrame, group_by: str) -> Dict:
        """Compare survival curves between groups using log-rank test"""
        
        survival_df = self.prepare_survival_data(df)
        groups = survival_df[group_by].unique()
        
        if len(groups) < 2:
            return {"error": "Need at least 2 groups for comparison"}
        
        # Perform pairwise log-rank tests
        comparisons = {}
        
        for i, group1 in enumerate(groups):
            for group2 in groups[i+1:]:
                group1_data = survival_df[survival_df[group_by] == group1]
                group2_data = survival_df[survival_df[group_by] == group2]
                
                if len(group1_data) < 5 or len(group2_data) < 5:
                    continue
                
                # Log-rank test
                results = logrank_test(
                    group1_data['duration'], group2_data['duration'],
                    group1_data['event'], group2_data['event']
                )
                
                comparisons[f"{group1}_vs_{group2}"] = {
                    'test_statistic': results.test_statistic,
                    'p_value': results.p_value,
                    'is_significant': results.p_value < 0.05
                }
        
        return comparisons
    
    def fit_cox_model(self, df: pd.DataFrame, covariates: List[str]) -> Dict:
        """Fit Cox Proportional Hazards model"""
        
        survival_df = self.prepare_survival_data(df)
        
        # Prepare covariates
        model_df = survival_df[['duration', 'event'] + covariates].copy()
        
        # Handle categorical variables
        categorical_cols = []
        for col in covariates:
            if survival_df[col].dtype == 'object':
                categorical_cols.append(col)
                # One-hot encode
                dummies = pd.get_dummies(survival_df[col], prefix=col)
                model_df = pd.concat([model_df, dummies], axis=1)
                model_df.drop(col, axis=1, inplace=True)
        
        # Remove rows with missing values
        model_df = model_df.dropna()
        
        if len(model_df) < 20:
            return {"error": "Insufficient data for Cox model"}
        
        try:
            # Fit Cox model
            self.cph.fit(model_df, duration_col='duration', event_col='event')
            
            return {
                'summary': self.cph.summary,
                'hazard_ratios': np.exp(self.cph.params_),
                'concordance': self.cph.concordance_index_,
                'log_likelihood': self.cph.log_likelihood_,
                'aic': self.cph.AIC_
            }
            
        except Exception as e:
            return {"error": f"Cox model fitting failed: {str(e)}"}
    
    def estimate_shelf_life(self, df: pd.DataFrame, confidence_level: float = SHELF_LIFE_CONFIDENCE) -> pd.DataFrame:
        """Estimate shelf-life for each product-location combination"""
        
        survival_df = self.prepare_survival_data(df)
        
        shelf_life_estimates = []
        
        # Group by product and location
        for (product_id, location), group in survival_df.groupby(['product_id', 'delivered_location']):
            
            if len(group) < MIN_SAMPLE_SIZE:
                continue
            
            # Fit Kaplan-Meier for this group
            kmf = KaplanMeierFitter()
            kmf.fit(
                durations=group['duration'],
                event_observed=group['event']
            )
            
            # Get survival estimates
            median_survival = kmf.median_survival_time_
            
            # Get confidence intervals
            ci = kmf.confidence_interval_survival_function_
            
            # Estimate shelf-life at different confidence levels
            survival_func = kmf.survival_function_
            
            # Find time when survival probability drops below (1 - confidence_level)
            target_survival = 1 - confidence_level
            
            try:
                shelf_life_estimate = survival_func[survival_func.iloc[:, 0] <= target_survival].index[0]
            except IndexError:
                shelf_life_estimate = median_survival
            
            shelf_life_estimates.append({
                'product_id': product_id,
                'location': location,
                'sample_size': len(group),
                'events': group['event'].sum(),
                'median_shelf_life': median_survival,
                'estimated_shelf_life': shelf_life_estimate,
                'confidence_level': confidence_level
            })
        
        return pd.DataFrame(shelf_life_estimates)
    
    def create_survival_plot(self, df: pd.DataFrame, group_by: Optional[str] = None, 
                           title: str = "Survival Curves") -> go.Figure:
        """Create interactive survival plot using Plotly"""
        
        survival_results = self.fit_kaplan_meier(df, group_by)
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, (group_name, results) in enumerate(survival_results.items()):
            if 'error' in results:
                continue
                
            survival_func = results['survival_function']
            ci = results['confidence_interval']
            
            # Main survival curve
            fig.add_trace(go.Scatter(
                x=survival_func.index,
                y=survival_func.iloc[:, 0],
                mode='lines',
                name=group_name,
                line=dict(color=colors[i % len(colors)], width=2),
                hovertemplate=f'<b>{group_name}</b><br>' +
                             'Days: %{x}<br>' +
                             'Survival Probability: %{y:.3f}<extra></extra>'
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=list(ci.index) + list(ci.index[::-1]),
                y=list(ci.iloc[:, 1]) + list(ci.iloc[:, 0][::-1]),
                fill='toself',
                fillcolor=colors[i % len(colors)],
                opacity=0.2,
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                name=f'{group_name} CI',
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Days',
            yaxis_title='Survival Probability',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig

# Utility functions
def analyze_product_shelf_life(df: pd.DataFrame, product_id: str) -> Dict:
    """Comprehensive shelf-life analysis for a specific product"""
    
    product_data = df[df['product_id'] == product_id]
    
    if len(product_data) == 0:
        return {"error": f"No data found for product {product_id}"}
    
    analyzer = ShelfLifeAnalyzer()
    
    # Overall analysis
    overall_km = analyzer.fit_kaplan_meier(product_data)
    
    # By location
    location_km = analyzer.fit_kaplan_meier(product_data, group_by='delivered_location')
    
    # By season
    season_km = analyzer.fit_kaplan_meier(product_data, group_by='season')
    
    # Shelf-life estimates
    shelf_life_estimates = analyzer.estimate_shelf_life(product_data)
    
    return {
        'overall_survival': overall_km,
        'by_location': location_km,
        'by_season': season_km,
        'shelf_life_estimates': shelf_life_estimates
    }
