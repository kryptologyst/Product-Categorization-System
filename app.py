"""
Flask Web Application for Product Categorization System
Modern web interface for the ML-powered product categorizer.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import sys
import json
import pandas as pd
from datetime import datetime
import logging

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.product_classifier import ProductCategorizer
from data.products_database import ProductDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Global variables for model and database
classifier = None
database = None
model_trained = False

def initialize_system():
    """Initialize the ML model and database."""
    global classifier, database, model_trained
    
    try:
        # Initialize database
        database = ProductDatabase()
        logger.info(f"Database initialized with {len(database.df)} products")
        
        # Initialize and train classifier
        classifier = ProductCategorizer(algorithm='logistic_regression', max_features=3000)
        
        # Train the model
        df = database.get_dataframe()
        results = classifier.train(df, test_size=0.2)
        model_trained = True
        
        logger.info(f"Model trained successfully. Accuracy: {results['accuracy']:.3f}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {str(e)}")
        return False

@app.route('/')
def index():
    """Main page with product categorization interface."""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_category():
    """API endpoint for product category prediction."""
    try:
        data = request.get_json()
        
        if not data or 'product_name' not in data:
            return jsonify({'error': 'Product name is required'}), 400
        
        product_name = data['product_name'].strip()
        description = data.get('description', '').strip()
        
        if not product_name:
            return jsonify({'error': 'Product name cannot be empty'}), 400
        
        if not model_trained:
            return jsonify({'error': 'Model is not trained yet'}), 500
        
        # Make prediction
        predictions = classifier.predict([product_name], [description] if description else None)
        
        if predictions:
            result = predictions[0]
            return jsonify({
                'success': True,
                'prediction': {
                    'category': result['predicted_category'],
                    'confidence': round(result['confidence'], 3),
                    'all_probabilities': {k: round(v, 3) for k, v in result['all_probabilities'].items()}
                }
            })
        else:
            return jsonify({'error': 'Failed to make prediction'}), 500
            
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """API endpoint for batch product category prediction."""
    try:
        data = request.get_json()
        
        if not data or 'products' not in data:
            return jsonify({'error': 'Products list is required'}), 400
        
        products = data['products']
        
        if not isinstance(products, list) or len(products) == 0:
            return jsonify({'error': 'Products must be a non-empty list'}), 400
        
        if len(products) > 100:
            return jsonify({'error': 'Maximum 100 products allowed per batch'}), 400
        
        if not model_trained:
            return jsonify({'error': 'Model is not trained yet'}), 500
        
        # Extract product names and descriptions
        product_names = []
        descriptions = []
        
        for product in products:
            if isinstance(product, str):
                product_names.append(product)
                descriptions.append('')
            elif isinstance(product, dict) and 'name' in product:
                product_names.append(product['name'])
                descriptions.append(product.get('description', ''))
            else:
                return jsonify({'error': 'Invalid product format'}), 400
        
        # Make predictions
        predictions = classifier.predict(product_names, descriptions)
        
        # Format results
        results = []
        for pred in predictions:
            results.append({
                'product_name': pred['product_name'],
                'category': pred['predicted_category'],
                'confidence': round(pred['confidence'], 3),
                'all_probabilities': {k: round(v, 3) for k, v in pred['all_probabilities'].items()}
            })
        
        return jsonify({
            'success': True,
            'predictions': results
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/categories')
def get_categories():
    """Get all available product categories."""
    try:
        if not model_trained:
            return jsonify({'error': 'Model is not trained yet'}), 500
        
        categories = classifier.categories
        return jsonify({
            'success': True,
            'categories': categories
        })
        
    except Exception as e:
        logger.error(f"Categories error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/sample_products')
def get_sample_products():
    """Get sample products from the database."""
    try:
        if database is None:
            return jsonify({'error': 'Database not initialized'}), 500
        
        sample_size = request.args.get('size', 10, type=int)
        sample_size = min(max(sample_size, 1), 50)  # Limit between 1 and 50
        
        sample_df = database.get_sample_data(sample_size)
        
        products = []
        for _, row in sample_df.iterrows():
            products.append({
                'name': row['ProductName'],
                'description': row['Description'],
                'category': row['Category'],
                'brand': row['Brand'],
                'price': row['Price']
            })
        
        return jsonify({
            'success': True,
            'products': products
        })
        
    except Exception as e:
        logger.error(f"Sample products error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/model_info')
def get_model_info():
    """Get information about the trained model."""
    try:
        if not model_trained:
            return jsonify({'error': 'Model is not trained yet'}), 500
        
        info = {
            'algorithm': classifier.algorithm,
            'max_features': classifier.max_features,
            'categories': classifier.categories,
            'num_categories': len(classifier.categories),
            'is_trained': classifier.is_trained,
            'training_data_size': len(database.df) if database else 0
        }
        
        return jsonify({
            'success': True,
            'model_info': info
        })
        
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/search_products')
def search_products():
    """Search products in the database."""
    try:
        query = request.args.get('q', '').strip()
        
        if not query:
            return jsonify({'error': 'Search query is required'}), 400
        
        if database is None:
            return jsonify({'error': 'Database not initialized'}), 500
        
        results_df = database.search_products(query)
        
        products = []
        for _, row in results_df.iterrows():
            products.append({
                'name': row['ProductName'],
                'description': row['Description'],
                'category': row['Category'],
                'brand': row['Brand'],
                'price': row['Price']
            })
        
        return jsonify({
            'success': True,
            'products': products,
            'count': len(products)
        })
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Initializing Product Categorization System...")
    
    if initialize_system():
        print("✅ System initialized successfully!")
        print("🚀 Starting web server...")
        print("📱 Open http://localhost:5000 in your browser")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to initialize system. Check logs for details.")
        sys.exit(1)
