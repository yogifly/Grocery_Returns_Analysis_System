"""Predictive models for return probability and risk scoring"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
import shap
from typing import Dict, List, Tuple, Optional
from config.settings import MODELS_DIR

class ReturnPredictionModel:
    """Predictive model for estimating return probability"""
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.is_trained = False
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def prepare_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """Prepare features for modeling"""
        
        feature_df = df.copy()
        
        # Select relevant features
        feature_columns = [
            'product_id', 'delivered_location', 'carrier_id', 'package_type',
            'category', 'expected_shelf_life_days', 'quantity', 'time_in_transit',
            'delivery_month', 'delivery_quarter', 'season', 'temp_sensitive',
            'high_risk_location'
        ]
        
        # Filter to available columns
        available_columns = [col for col in feature_columns if col in feature_df.columns]
        feature_df = feature_df[available_columns + ['is_returned']].copy()
        
        # Handle categorical variables
        categorical_columns = ['product_id', 'delivered_location', 'carrier_id', 
                             'package_type', 'category', 'season']
        
        for col in categorical_columns:
            if col in feature_df.columns:
                if is_training:
                    # Fit label encoder
                    le = LabelEncoder()
                    feature_df[col] = le.fit_transform(feature_df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    # Transform using existing encoder
                    if col in self.label_encoders:
                        # Handle unseen categories
                        le = self.label_encoders[col]
                        feature_df[col] = feature_df[col].astype(str)
                        
                        # Map unseen categories to a default value
                        unseen_mask = ~feature_df[col].isin(le.classes_)
                        if unseen_mask.any():
                            # Add unseen categories to encoder
                            new_classes = feature_df.loc[unseen_mask, col].unique()
                            le.classes_ = np.append(le.classes_, new_classes)
                        
                        feature_df[col] = le.transform(feature_df[col])
        
        # Handle missing values
        feature_df = feature_df.fillna(0)
        
        if is_training:
            self.feature_names = [col for col in feature_df.columns if col != 'is_returned']
        
        return feature_df
    
    def train(self, df: pd.DataFrame, target_column: str = 'is_returned') -> Dict:
        """Train the prediction model"""
        
        # Prepare features
        feature_df = self.prepare_features(df, is_training=True)
        
        # Separate features and target
        X = feature_df[self.feature_names]
        y = feature_df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features for logistic regression
        if self.model_type == 'logistic_regression':
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
            
        else:
            # Train model (tree-based models don't need scaling)
            self.model.fit(X_train, y_train)
            
            # Predictions
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation
        if self.model_type == 'logistic_regression':
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        else:
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='roc_auc')
        
        self.is_trained = True
        
        # Save model
        model_path = MODELS_DIR / f"return_prediction_{self.model_type}.joblib"
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }, model_path)
        
        results = {
            'auc_score': auc_score,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'feature_importance': self.get_feature_importance()
        }
        
        print(f"✅ Model trained successfully!")
        print(f"AUC Score: {auc_score:.3f}")
        print(f"CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return results
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        
        if not self.is_trained:
            self.load_model()
        
        # Prepare features
        feature_df = self.prepare_features(df, is_training=False)
        X = feature_df[self.feature_names]
        
        # Scale if needed
        if self.model_type == 'logistic_regression':
            X = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict_proba(X)[:, 1]
        
        return predictions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores"""
        
        if not self.is_trained:
            return pd.DataFrame()
        
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            importance_scores = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # Linear models
            importance_scores = np.abs(self.model.coef_[0])
        else:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def load_model(self) -> None:
        """Load trained model"""
        
        model_path = MODELS_DIR / f"return_prediction_{self.model_type}.joblib"
        
        try:
            saved_data = joblib.load(model_path)
            self.model = saved_data['model']
            self.scaler = saved_data['scaler']
            self.label_encoders = saved_data['label_encoders']
            self.feature_names = saved_data['feature_names']
            self.is_trained = True
            print("✅ Model loaded successfully")
        except FileNotFoundError:
            print("❌ No trained model found. Please train the model first.")
    
    def explain_predictions(self, df: pd.DataFrame, sample_size: int = 100) -> Dict:
        """Generate SHAP explanations for predictions"""
        
        if not self.is_trained:
            self.load_model()
        
        # Prepare features
        feature_df = self.prepare_features(df, is_training=False)
        X = feature_df[self.feature_names].sample(min(sample_size, len(feature_df)))
        
        # Scale if needed
        if self.model_type == 'logistic_regression':
            X_scaled = self.scaler.transform(X)
            explainer = shap.LinearExplainer(self.model, X_scaled)
            shap_values = explainer.shap_values(X_scaled)
        else:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification
        
        return {
            'shap_values': shap_values,
            'feature_names': self.feature_names,
            'base_value': explainer.expected_value if hasattr(explainer, 'expected_value') else 0
        }

class RiskScorer:
    """Risk scoring system for inventory management"""
    
    def __init__(self):
        self.prediction_model = None
        
    def calculate_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate risk scores for products/locations"""
        
        # Initialize prediction model
        if self.prediction_model is None:
            self.prediction_model = ReturnPredictionModel()
            try:
                self.prediction_model.load_model()
            except:
                print("Training new prediction model...")
                self.prediction_model.train(df)
        
        # Get return probabilities
        return_probabilities = self.prediction_model.predict(df)
        
        # Calculate risk scores
        risk_df = df.copy()
        risk_df['return_probability'] = return_probabilities
        
        # Risk score components
        risk_df['shelf_life_risk'] = 1 / (risk_df['expected_shelf_life_days'] + 1)
        risk_df['temp_sensitivity_risk'] = risk_df['temp_sensitive'].astype(int) * 0.2
        risk_df['location_risk'] = risk_df['high_risk_location'].astype(int) * 0.15
        
        # Combined risk score (0-1 scale)
        risk_df['risk_score'] = (
            0.5 * risk_df['return_probability'] +
            0.2 * risk_df['shelf_life_risk'] +
            0.2 * risk_df['temp_sensitivity_risk'] +
            0.1 * risk_df['location_risk']
        )
        
        # Risk categories
        risk_df['risk_category'] = pd.cut(
            risk_df['risk_score'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        return risk_df
    
    def generate_recommendations(self, risk_df: pd.DataFrame) -> List[Dict]:
        """Generate actionable recommendations based on risk scores"""
        
        recommendations = []
        
        # High-risk products
        high_risk = risk_df[risk_df['risk_category'] == 'High']
        
        for _, row in high_risk.iterrows():
            rec = {
                'product_id': row['product_id'],
                'location': row['delivered_location'],
                'risk_score': row['risk_score'],
                'recommendations': []
            }
            
            if row['temp_sensitive']:
                rec['recommendations'].append("Use insulated packaging")
            
            if row['high_risk_location']:
                rec['recommendations'].append("Reduce delivery time")
                rec['recommendations'].append("Consider alternative carriers")
            
            if row['expected_shelf_life_days'] < 5:
                rec['recommendations'].append("Implement faster inventory turnover")
                rec['recommendations'].append("Consider local sourcing")
            
            recommendations.append(rec)
        
        return recommendations

# Utility functions
def train_all_models(df: pd.DataFrame) -> Dict:
    """Train all prediction models and return results"""
    
    results = {}
    
    model_types = ['random_forest', 'gradient_boosting', 'logistic_regression']
    
    for model_type in model_types:
        print(f"\n🔄 Training {model_type} model...")
        
        model = ReturnPredictionModel(model_type)
        results[model_type] = model.train(df)
    
    return results

def compare_model_performance(results: Dict) -> pd.DataFrame:
    """Compare performance of different models"""
    
    comparison_data = []
    
    for model_type, result in results.items():
        comparison_data.append({
            'model_type': model_type,
            'auc_score': result['auc_score'],
            'cv_mean': result['cv_mean'],
            'cv_std': result['cv_std']
        })
    
    return pd.DataFrame(comparison_data).sort_values('auc_score', ascending=False)
