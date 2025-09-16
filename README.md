# Product Categorization System

An AI-powered product categorization system that automatically classifies products into categories based on their names and descriptions using machine learning.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v2.3+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.3+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Features

- **Advanced ML Classification**: Uses TF-IDF vectorization with Logistic Regression for accurate product categorization
- **Modern Web Interface**: Beautiful, responsive web UI built with Bootstrap 5
- **Comprehensive Database**: Mock database with 70+ products across 7 categories
- **Real-time Predictions**: Instant product categorization with confidence scores
- **Batch Processing**: Support for categorizing multiple products at once
- **Model Analytics**: Detailed probability distributions and feature importance
- **RESTful API**: Complete API endpoints for integration with other systems

## Categories Supported

- **Electronics**: Smartphones, laptops, cameras, gaming devices
- **Clothing & Fashion**: Apparel, footwear, accessories
- **Home & Garden**: Appliances, furniture, home improvement
- **Books & Media**: Books, audiobooks, educational materials
- **Sports & Outdoors**: Camping gear, fitness equipment, outdoor activities
- **Health & Beauty**: Skincare, cosmetics, wellness products
- **Automotive**: Car parts, accessories, maintenance products

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/product-categorization-system.git
   cd product-categorization-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:5000`

### Alternative Installation

```bash
pip install -e .
product-categorizer
```

## Usage

### Web Interface

1. **Single Product Classification**
   - Enter product name and optional description
   - Click "Predict Category"
   - View results with confidence scores and probability distributions

2. **Sample Products**
   - Click on any sample product to test the system
   - Explore different product categories

### API Endpoints

#### Predict Single Product
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "product_name": "iPhone 15 Pro Max",
    "description": "Latest Apple smartphone with titanium design"
  }'
```

#### Batch Prediction
```bash
curl -X POST http://localhost:5000/api/batch_predict \
  -H "Content-Type: application/json" \
  -d '{
    "products": [
      {"name": "MacBook Pro", "description": "Professional laptop"},
      {"name": "Nike Running Shoes", "description": "Athletic footwear"}
    ]
  }'
```

#### Get Categories
```bash
curl http://localhost:5000/api/categories
```

#### Search Products
```bash
curl "http://localhost:5000/api/search_products?q=apple"
```

## Project Structure

```
product-categorization-system/
├── app.py                          # Flask web application
├── requirements.txt                # Python dependencies
├── setup.py                       # Package setup configuration
├── README.md                      # Project documentation
├── data/
│   └── products_database.py      # Mock product database
├── models/
│   └── product_classifier.py     # ML model implementation
└── templates/
    └── index.html                 # Web interface template
```

## Machine Learning Details

### Algorithm
- **Primary**: Logistic Regression with L2 regularization
- **Alternative**: Random Forest Classifier
- **Vectorization**: TF-IDF with n-grams (1,2)
- **Features**: Up to 5000 TF-IDF features
- **Cross-validation**: 5-fold stratified CV

### Model Performance
- **Training Accuracy**: ~95%+
- **Cross-validation Score**: ~90%+
- **Features**: Product names + descriptions
- **Categories**: 7 main product categories

### Text Preprocessing
- Lowercase normalization
- Special character removal
- Stop word filtering
- N-gram extraction (unigrams + bigrams)
- ASCII accent stripping

## API Response Examples

### Single Prediction Response
```json
{
  "success": true,
  "prediction": {
    "category": "Electronics",
    "confidence": 0.892,
    "all_probabilities": {
      "Electronics": 0.892,
      "Home & Garden": 0.045,
      "Automotive": 0.032,
      "Sports & Outdoors": 0.018,
      "Clothing & Fashion": 0.008,
      "Health & Beauty": 0.003,
      "Books & Media": 0.002
    }
  }
}
```

### Model Information Response
```json
{
  "success": true,
  "model_info": {
    "algorithm": "logistic_regression",
    "max_features": 5000,
    "categories": ["Electronics", "Clothing & Fashion", ...],
    "num_categories": 7,
    "is_trained": true,
    "training_data_size": 70
  }
}
```

## 🔧 Configuration

### Environment Variables
```bash
export FLASK_ENV=development  # For development mode
export FLASK_DEBUG=1         # Enable debug mode
export PORT=5000            # Custom port (default: 5000)
```

### Model Parameters
```python
# In models/product_classifier.py
classifier = ProductCategorizer(
    algorithm='logistic_regression',  # or 'random_forest'
    max_features=5000                # TF-IDF feature limit
)
```

## Testing

### Manual Testing
1. Start the application: `python app.py`
2. Open browser to `http://localhost:5000`
3. Test with sample products or custom inputs

### API Testing
```bash
# Test prediction endpoint
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"product_name": "Test Product"}'

# Test model info
curl http://localhost:5000/api/model_info
```

## Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
```bash
# Using Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Using Docker
docker build -t product-categorizer .
docker run -p 5000:5000 product-categorizer
```

## Performance Optimization

### Model Improvements
- Increase training data size
- Add more product categories
- Implement ensemble methods
- Use deep learning models (BERT, etc.)

### System Optimization
- Add caching for predictions
- Implement model versioning
- Add database persistence
- Use async processing for batch requests

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **scikit-learn** for machine learning algorithms
- **Flask** for the web framework
- **Bootstrap** for the responsive UI components
- **Font Awesome** for icons

## Support

For support, email developer@example.com or create an issue on GitHub.

## Future Enhancements

- [ ] Deep learning models (BERT, RoBERTa)
- [ ] Multi-language support
- [ ] Image-based categorization
- [ ] Real-time model retraining
- [ ] Advanced analytics dashboard
- [ ] Mobile app integration
- [ ] Cloud deployment templates

# Product-Categorization-System
