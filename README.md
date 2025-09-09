# 🏠 Advanced House Price Prediction System

A comprehensive machine learning solution for house price prediction featuring a world-class Streamlit application with advanced data science techniques and professional UI/UX.

## 🎯 Project Overview

This project implements a state-of-the-art house price prediction system using advanced regression techniques, featuring:
- **90.4% Cross-Validated Accuracy** 
- **Professional Streamlit Web Application**
- **Advanced SHAP Explanations**
- **Real-time Predictions with AI Explanations**
- **Comprehensive Feature Engineering**

## 🚀 Live Demo

Run the application locally:
```bash
cd house-price-prediction-streamlit
streamlit run streamlit_app.py --server.port 8503
```

Visit: **http://localhost:8503**

## 📁 Project Structure

```
house_price_prediction_advanced/
├── 📱 house-price-prediction-streamlit/   # Main Streamlit Application
│   ├── streamlit_app.py                   # Main application entry point
│   ├── config/                            # Configuration files
│   ├── utils/                             # Utility modules
│   ├── models/                            # Trained ML models
│   └── data/                              # Processed datasets
├── 📊 scripts/                           # Analysis & utility scripts
│   ├── analysis/                         # Data analysis scripts
│   ├── testing/                          # Test scripts
│   └── utilities/                        # Helper utilities
├── 📖 docs/                             # Documentation & guides
│   └── guides/                          # Implementation guides
├── 🗂️ dataset/                          # Raw datasets
├── 📈 eda/                              # Exploratory data analysis
├── ⚙️ preprocessing/                     # Data preprocessing
├── 🎨 visualizations/                   # Generated visualizations
├── 📋 documentation/                    # Project documentation
└── 🚀 deployment/                       # Deployment configurations
```

## ✨ Key Features

### 🔮 Prediction Interface
- **Quick Prediction**: Fast estimates with default values
- **Advanced Mode**: Precise control over all 223+ features
- **Batch Prediction**: Process multiple properties via CSV upload
- **Real-time SHAP Explanations**: AI-powered prediction explanations

### 📊 Model Analytics
- **Cross-Validation Performance**: 90.4% accuracy with robust validation
- **Feature Importance Analysis**: 6 different importance methods
- **Enhanced Charts**: User-friendly names and professional design
- **Performance Gauges**: Interactive metrics dashboard

### 🧠 Model Interpretation
- **SHAP Analysis**: Individual and global feature explanations
- **Feature Categories**: Impact analysis by feature groups
- **Partial Dependence**: Feature effect visualization

### 📈 Market Intelligence  
- **Business Insights**: Strategic recommendations
- **Market Segments**: Property distribution analysis
- **Investment Opportunities**: High ROI identification

## 🛠️ Technical Implementation

### Machine Learning Pipeline
- **CatBoost Regressor**: Champion model with 223 engineered features
- **Box-Cox Transformation**: λ = -0.07693 for price normalization
- **5-Fold Cross-Validation**: Robust performance evaluation
- **Advanced Feature Engineering**: 175% feature expansion (81 → 223)

### Data Processing
- **Comprehensive Preprocessing**: Missing value imputation, outlier handling
- **Feature Transformation**: Categorical encoding, numerical scaling
- **Domain Expertise**: Real estate context-aware feature creation

### User Experience
- **Professional UI**: Clean, intuitive interface design
- **Responsive Design**: Works across different screen sizes
- **Interactive Visualizations**: Plotly-powered charts and graphs
- **Real-time Feedback**: Instant predictions and explanations

## 📚 Documentation

- **[Complete Transformation Guide](docs/guides/COMPLETE_TRANSFORMATION_GUIDE.md)**: Detailed preprocessing steps
- **[Field Mappings Documentation](docs/guides/COMPLETE_FIELD_MAPPINGS_DOCUMENTATION.md)**: Comprehensive feature mappings
- **[Advanced Mode Implementation](docs/guides/ADVANCED_MODE_IMPLEMENTATION_COMPLETE.md)**: Advanced interface guide
- **[Enhanced SHAP Implementation](docs/guides/ENHANCED_SHAP_IMPLEMENTATION_COMPLETE.md)**: SHAP integration details

## 🏆 Model Performance

| Metric | Value |
|--------|-------|
| **Cross-Validated Accuracy** | 90.4% |
| **R² Score** | 0.904 |
| **RMSE** | 0.0485 |
| **Features** | 223 engineered features |
| **Validation** | 5-fold cross-validation |

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Required packages: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `catboost`, `shap`, `plotly`

### Installation
1. Clone the repository
2. Navigate to the Streamlit app directory:
   ```bash
   cd house-price-prediction-streamlit
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   streamlit run streamlit_app.py --server.port 8503
   ```

## 👨‍💻 Author

**Victor Collins Oppon, FCCA, MBA, BSc.**  
Data Scientist and AI Consultant  
Videbimus AI  
www.videbimusai.com

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

*Built with ❤️ using advanced data science techniques and modern web technologies.*