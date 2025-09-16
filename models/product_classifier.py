"""
Product Categorization Model
Advanced ML classifier for automatically categorizing products based on their names and descriptions.
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class ProductCategorizer:
    """
    Advanced product categorization system using machine learning.
    Supports multiple algorithms and comprehensive text preprocessing.
    """
    
    def __init__(self, algorithm: str = 'logistic_regression', max_features: int = 5000):
        """
        Initialize the product categorizer.
        
        Args:
            algorithm: ML algorithm to use ('logistic_regression', 'random_forest')
            max_features: Maximum number of TF-IDF features to extract
        """
        self.algorithm = algorithm
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True,
            strip_accents='ascii'
        )
        self.label_encoder = LabelEncoder()
        self.model = None
        self.is_trained = False
        self.categories = []
        self.feature_names = []
        
        # Initialize the ML model based on algorithm choice
        if algorithm == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
        elif algorithm == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def preprocess_text(self, df: pd.DataFrame) -> pd.Series:
        """
        Combine and preprocess product name and description for better classification.
        
        Args:
            df: DataFrame with ProductName and Description columns
            
        Returns:
            Series with combined and cleaned text
        """
        # Combine product name and description with weighted importance
        combined_text = df['ProductName'].fillna('') + ' ' + df['Description'].fillna('')
        
        # Clean and normalize text
        combined_text = combined_text.str.lower()
        combined_text = combined_text.str.replace(r'[^\w\s]', ' ', regex=True)
        combined_text = combined_text.str.replace(r'\s+', ' ', regex=True)
        combined_text = combined_text.str.strip()
        
        return combined_text
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the product categorization model.
        
        Args:
            df: Training data with ProductName, Description, and Category columns
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary with training metrics and results
        """
        if len(df) < 10:
            raise ValueError("Need at least 10 samples for training")
        
        # Preprocess text data
        X_text = self.preprocess_text(df)
        y = df['Category'].values
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.categories = self.label_encoder.classes_.tolist()
        
        # Vectorize text
        X_vectorized = self.vectorizer.fit_transform(X_text)
        self.feature_names = self.vectorizer.get_feature_names_out().tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(
            y_test, y_pred, target_names=self.categories, output_dict=True
        )
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
        
        results = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_rep,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'test_predictions': y_pred,
            'test_probabilities': y_pred_proba,
            'test_labels': y_test,
            'categories': self.categories
        }
        
        return results
    
    def predict(self, product_names: List[str], descriptions: List[str] = None) -> List[Dict[str, Any]]:
        """
        Predict categories for new products.
        
        Args:
            product_names: List of product names
            descriptions: List of product descriptions (optional)
            
        Returns:
            List of prediction dictionaries with category, confidence, and probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if descriptions is None:
            descriptions = [''] * len(product_names)
        
        # Create DataFrame for preprocessing
        df = pd.DataFrame({
            'ProductName': product_names,
            'Description': descriptions
        })
        
        # Preprocess and vectorize
        X_text = self.preprocess_text(df)
        X_vectorized = self.vectorizer.transform(X_text)
        
        # Make predictions
        predictions = self.model.predict(X_vectorized)
        probabilities = self.model.predict_proba(X_vectorized)
        
        results = []
        for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
            category = self.label_encoder.inverse_transform([pred])[0]
            confidence = np.max(proba)
            
            # Create probability dictionary for all categories
            prob_dict = {
                cat: float(prob) for cat, prob in zip(self.categories, proba)
            }
            
            results.append({
                'product_name': product_names[i],
                'predicted_category': category,
                'confidence': float(confidence),
                'all_probabilities': prob_dict
            })
        
        return results
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get the most important features for each category.
        
        Args:
            top_n: Number of top features to return per category
            
        Returns:
            Dictionary mapping categories to their top features
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before extracting feature importance")
        
        feature_importance = {}
        
        if self.algorithm == 'logistic_regression':
            # For logistic regression, use coefficients
            coef = self.model.coef_
            for i, category in enumerate(self.categories):
                # Get feature coefficients for this category
                category_coef = coef[i] if len(coef.shape) > 1 else coef
                
                # Get top positive and negative features
                feature_scores = list(zip(self.feature_names, category_coef))
                feature_scores.sort(key=lambda x: abs(x[1]), reverse=True)
                
                feature_importance[category] = feature_scores[:top_n]
        
        elif self.algorithm == 'random_forest':
            # For random forest, use feature importances
            importances = self.model.feature_importances_
            feature_scores = list(zip(self.feature_names, importances))
            feature_scores.sort(key=lambda x: x[1], reverse=True)
            
            # For random forest, we get global importance, not per-category
            feature_importance['global'] = feature_scores[:top_n]
        
        return feature_importance
    
    def plot_confusion_matrix(self, results: Dict[str, Any], figsize: Tuple[int, int] = (10, 8)):
        """
        Plot confusion matrix from training results.
        
        Args:
            results: Results dictionary from train() method
            figsize: Figure size tuple
        """
        plt.figure(figsize=figsize)
        cm = results['confusion_matrix']
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.categories,
            yticklabels=self.categories
        )
        
        plt.title(f'Confusion Matrix - Product Categorization\nAccuracy: {results["accuracy"]:.3f}')
        plt.xlabel('Predicted Category')
        plt.ylabel('Actual Category')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        return plt.gcf()
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'categories': self.categories,
            'algorithm': self.algorithm,
            'max_features': self.max_features,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.label_encoder = model_data['label_encoder']
        self.categories = model_data['categories']
        self.algorithm = model_data['algorithm']
        self.max_features = model_data['max_features']
        self.is_trained = model_data['is_trained']
    
    def optimize_hyperparameters(self, df: pd.DataFrame, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Optimize model hyperparameters using grid search.
        
        Args:
            df: Training data
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with best parameters and scores
        """
        X_text = self.preprocess_text(df)
        y_encoded = self.label_encoder.fit_transform(df['Category'].values)
        X_vectorized = self.vectorizer.fit_transform(X_text)
        
        if self.algorithm == 'logistic_regression':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        elif self.algorithm == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
        
        grid_search = GridSearchCV(
            self.model, param_grid, cv=cv_folds, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_vectorized, y_encoded)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_trained = True
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }

if __name__ == "__main__":
    # Example usage
    from data.products_database import ProductDatabase
    
    # Load data
    db = ProductDatabase()
    df = db.get_dataframe()
    
    # Initialize and train model
    classifier = ProductCategorizer(algorithm='logistic_regression')
    results = classifier.train(df)
    
    print(f"Model Accuracy: {results['accuracy']:.3f}")
    print(f"Cross-validation Score: {results['cv_mean']:.3f} (+/- {results['cv_std']:.3f})")
    
    # Test prediction
    test_products = [
        "Apple MacBook Pro 14-inch M2",
        "Nike Air Max Running Shoes",
        "KitchenAid Stand Mixer"
    ]
    
    predictions = classifier.predict(test_products)
    for pred in predictions:
        print(f"Product: {pred['product_name']}")
        print(f"Category: {pred['predicted_category']} (Confidence: {pred['confidence']:.3f})")
        print()
