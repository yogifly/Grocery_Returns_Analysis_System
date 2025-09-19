"""NLP classifier for return reason categorization"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
from typing import List, Tuple
from config.settings import MODELS_DIR

class ReturnReasonClassifier:
    """NLP classifier for categorizing return reasons"""
    
    def __init__(self):
        self.pipeline = None
        self.categories = ['stale', 'damaged', 'quality_issues', 'wrong_item', 'packaging_issues', 'late_delivery', 'other']
        
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for classification"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def create_training_data(self) -> Tuple[List[str], List[str]]:
        """Create synthetic training data for the classifier"""
        
        training_examples = {
            'stale': [
                'product was stale', 'items were rotten', 'found mold on the product', 
                'spoiled items received', 'fruits were overripe', 'vegetables were wilted',
                'bad smell from product', 'product expired', 'not fresh at all',
                'moldy berries', 'rotten apples', 'spoiled vegetables'
            ],
            'damaged': [
                'items were damaged', 'bruised fruits', 'packaging was damaged',
                'products were crushed', 'broken items', 'dented packaging',
                'squashed tomatoes', 'damaged during transport', 'items were smashed',
                'bruised apples', 'cracked packaging', 'torn bags'
            ],
            'quality_issues': [
                'poor quality', 'not fresh', 'quality below expectations',
                'items looked old', 'substandard quality', 'inferior products',
                'quality not as expected', 'disappointing quality', 'low grade items',
                'quality issues', 'not up to standard', 'poor condition'
            ],
            'wrong_item': [
                'wrong product delivered', 'incorrect item', 'different product than ordered',
                'received wrong variety', 'not what I ordered', 'incorrect product',
                'wrong brand delivered', 'different size than ordered', 'wrong quantity',
                'ordered apples got oranges', 'wrong product type', 'incorrect order'
            ],
            'packaging_issues': [
                'poor packaging', 'inadequate packaging', 'packaging leaked',
                'bad packaging quality', 'packaging was torn', 'insufficient protection',
                'packaging problems', 'leaky containers', 'damaged packaging',
                'poor packing', 'packaging not secure', 'loose packaging'
            ],
            'late_delivery': [
                'delivery was late', 'delayed delivery', 'received after expected date',
                'late arrival', 'delivery delay', 'not delivered on time',
                'arrived late', 'delayed shipment', 'late delivery service',
                'delivery took too long', 'behind schedule', 'delayed order'
            ],
            'other': [
                'general complaint', 'not satisfied', 'other issues',
                'miscellaneous problem', 'various issues', 'other reasons',
                'different problem', 'unspecified issue', 'general dissatisfaction'
            ]
        }
        
        texts = []
        labels = []
        
        for category, examples in training_examples.items():
            for example in examples:
                texts.append(example)
                labels.append(category)
        
        return texts, labels
    
    def train_classifier(self) -> None:
        """Train the return reason classifier"""
        
        # Get training data
        texts, labels = self.create_training_data()
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                stop_words='english',
                lowercase=True
            )),
            ('classifier', LogisticRegression(
                random_state=42,
                max_iter=1000
            ))
        ])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Train model
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Classifier trained with accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save model
        model_path = MODELS_DIR / "return_reason_classifier.joblib"
        joblib.dump(self.pipeline, model_path)
        print(f"Model saved to: {model_path}")
    
    def load_classifier(self) -> None:
        """Load trained classifier"""
        model_path = MODELS_DIR / "return_reason_classifier.joblib"
        
        try:
            self.pipeline = joblib.load(model_path)
            print("✅ Classifier loaded successfully")
        except FileNotFoundError:
            print("❌ No trained model found. Training new classifier...")
            self.train_classifier()
    
    def predict(self, texts: List[str]) -> List[str]:
        """Predict return reason categories for given texts"""
        
        if self.pipeline is None:
            self.load_classifier()
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Make predictions
        predictions = self.pipeline.predict(processed_texts)
        
        return predictions.tolist()
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Get prediction probabilities"""
        
        if self.pipeline is None:
            self.load_classifier()
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Get probabilities
        probabilities = self.pipeline.predict_proba(processed_texts)
        
        return probabilities
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """Get most important features for each category"""
        
        if self.pipeline is None:
            self.load_classifier()
        
        # Get feature names and coefficients
        feature_names = self.pipeline.named_steps['tfidf'].get_feature_names_out()
        coefficients = self.pipeline.named_steps['classifier'].coef_
        
        importance_data = []
        
        for i, category in enumerate(self.pipeline.classes_):
            # Get top positive coefficients (most indicative of this category)
            top_indices = np.argsort(coefficients[i])[-top_n:][::-1]
            
            for idx in top_indices:
                importance_data.append({
                    'category': category,
                    'feature': feature_names[idx],
                    'coefficient': coefficients[i][idx]
                })
        
        return pd.DataFrame(importance_data)

# Utility function
def classify_return_reasons(return_texts: List[str]) -> pd.DataFrame:
    """Classify a list of return reason texts"""
    
    classifier = ReturnReasonClassifier()
    predictions = classifier.predict(return_texts)
    probabilities = classifier.predict_proba(return_texts)
    
    results_df = pd.DataFrame({
        'return_text': return_texts,
        'predicted_category': predictions,
        'confidence': np.max(probabilities, axis=1)
    })
    
    return results_df
