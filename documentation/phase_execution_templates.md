# Phase Execution Templates & Guidelines

## 📋 Standardized Phase Implementation Framework

This document provides **standardized templates and guidelines** for implementing future phases (7-9) while maintaining the **world-class quality standards** established in the preprocessing pipeline.

---

## 🎯 **Phase Template Structure**

### **Standard Phase Directory Layout**
```
phaseX_description/
├── main_pipeline.py              # Primary execution script
├── config/                       # Configuration management
│   ├── default_config.json
│   └── hyperparameters.json
├── modules/                      # Modular components
│   ├── data_loader.py
│   ├── processor.py
│   └── validator.py
├── results/                      # Output artifacts
│   ├── processed_data/
│   ├── metrics/
│   └── visualizations/
├── tests/                        # Quality assurance
│   ├── unit_tests.py
│   └── integration_tests.py
└── documentation/               # Phase-specific docs
    ├── methodology.md
    └── results_analysis.md
```

---

## 🔧 **Phase 7 Template: Advanced Modeling**

### **7.1 Model Selection Pipeline Template**
**File**: `modeling/phase7_advanced_modeling/model_selection_pipeline.py`

```python
"""
PHASE 7: ADVANCED MACHINE LEARNING MODELING PIPELINE
======================================================================
State-of-the-art regression modeling using optimally prepared dataset.

Key Features:
- Multiple algorithm comparison (Linear, Tree-based, Neural, Ensemble)
- Automated hyperparameter optimization  
- Comprehensive cross-validation framework
- Performance benchmarking and selection
- Business metric optimization

Author: Advanced ML Pipeline
Date: 2025
======================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
from pathlib import Path

class AdvancedModelingPipeline:
    """
    Comprehensive modeling pipeline with automated model selection
    and optimization for house price prediction.
    """
    
    def __init__(self, config_path: str = "config/modeling_config.json"):
        """Initialize modeling pipeline with configuration."""
        self.config = self._load_config(config_path)
        self.models = self._initialize_models()
        self.results = {}
        self.best_model = None
        self.performance_metrics = {}
        
    def _initialize_models(self) -> Dict[str, Any]:
        """Initialize model portfolio with default hyperparameters."""
        return {
            # Linear Models - optimal for normalized features
            'ridge': Ridge(random_state=42),
            'lasso': Lasso(random_state=42),
            'elastic_net': ElasticNet(random_state=42),
            
            # Tree-Based Models - handle interactions naturally  
            'random_forest': RandomForestRegressor(random_state=42),
            'xgboost': xgb.XGBRegressor(random_state=42),
            'lightgbm': lgb.LGBMRegressor(random_state=42),
            
            # Advanced Models
            'neural_network': self._create_neural_network(),
            
            # Ensemble Methods (implemented after base models)
            'ensemble': None  # Will be created from best performers
        }
    
    def load_prepared_data(self) -> 'AdvancedModelingPipeline':
        """Load optimally prepared data from Phase 6."""
        print("Loading prepared data from Phase 6...")
        
        # Load final prepared datasets
        self.X_train = pd.read_csv('../preprocessing/phase6_scaling/final_train_prepared.csv')
        self.X_test = pd.read_csv('../preprocessing/phase6_scaling/final_test_prepared.csv')
        
        # Separate features and target
        self.y_train = self.X_train['SalePrice_transformed']
        self.X_train = self.X_train.drop('SalePrice_transformed', axis=1)
        
        print(f"Training data: {self.X_train.shape[0]} samples, {self.X_train.shape[1]} features")
        print(f"Test data: {self.X_test.shape[0]} samples, {self.X_test.shape[1]} features")
        print(f"Target statistics - Mean: {self.y_train.mean():.4f}, Std: {self.y_train.std():.4f}")
        
        return self
    
    def evaluate_models(self) -> 'AdvancedModelingPipeline':
        """Comprehensive model evaluation with cross-validation."""
        print("\n" + "="*60)
        print("EVALUATING MODEL PERFORMANCE")
        print("="*60)
        
        for model_name, model in self.models.items():
            if model is None:  # Skip ensemble for now
                continue
                
            print(f"\nEvaluating {model_name}...")
            
            try:
                # Cross-validation evaluation
                cv_scores = cross_val_score(
                    model, self.X_train, self.y_train,
                    cv=5, scoring='neg_mean_squared_error', n_jobs=-1
                )
                
                # Calculate metrics
                cv_rmse = np.sqrt(-cv_scores.mean())
                cv_std = cv_scores.std()
                
                # Fit model for additional metrics
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_train)
                
                # Store results
                self.results[model_name] = {
                    'cv_rmse_mean': cv_rmse,
                    'cv_rmse_std': cv_std,
                    'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_pred)),
                    'train_r2': r2_score(self.y_train, y_pred),
                    'train_mae': mean_absolute_error(self.y_train, y_pred),
                    'model': model
                }
                
                print(f"  CV RMSE: {cv_rmse:.4f} (+/- {cv_std:.4f})")
                print(f"  Train R²: {r2_score(self.y_train, y_pred):.4f}")
                print(f"  Train MAE: {mean_absolute_error(self.y_train, y_pred):.4f}")
                
            except Exception as e:
                print(f"  ERROR: {str(e)}")
                self.results[model_name] = {'error': str(e)}
        
        return self
    
    def select_best_model(self) -> 'AdvancedModelingPipeline':
        """Select best performing model based on cross-validation."""
        print("\n" + "="*60)  
        print("SELECTING BEST MODEL")
        print("="*60)
        
        # Rank models by CV RMSE (lower is better)
        valid_results = {k: v for k, v in self.results.items() if 'error' not in v}
        
        if not valid_results:
            raise ValueError("No models completed successfully!")
        
        best_model_name = min(valid_results, key=lambda x: valid_results[x]['cv_rmse_mean'])
        self.best_model = valid_results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        print(f"Best Model: {best_model_name}")
        print(f"CV RMSE: {valid_results[best_model_name]['cv_rmse_mean']:.4f}")
        print(f"Train R²: {valid_results[best_model_name]['train_r2']:.4f}")
        
        return self
    
    def optimize_hyperparameters(self) -> 'AdvancedModelingPipeline':
        """Optimize hyperparameters for the best model."""
        print("\n" + "="*60)
        print("OPTIMIZING HYPERPARAMETERS")
        print("="*60)
        
        # Define hyperparameter grids
        param_grids = self._get_hyperparameter_grids()
        
        if self.best_model_name in param_grids:
            print(f"Optimizing {self.best_model_name} hyperparameters...")
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                estimator=self.models[self.best_model_name],
                param_grid=param_grids[self.best_model_name],
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            # Update best model
            self.best_model = grid_search.best_estimator_
            
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {np.sqrt(-grid_search.best_score_):.4f}")
        
        return self
    
    def generate_predictions(self) -> 'AdvancedModelingPipeline':
        """Generate predictions on test set."""
        print("\n" + "="*60)
        print("GENERATING PREDICTIONS")
        print("="*60)
        
        # Train final model on full training set
        self.best_model.fit(self.X_train, self.y_train)
        
        # Generate predictions
        self.test_predictions = self.best_model.predict(self.X_test)
        
        print(f"Generated {len(self.test_predictions)} predictions")
        print(f"Prediction range: [{self.test_predictions.min():.4f}, {self.test_predictions.max():.4f}]")
        
        return self
    
    def save_results(self) -> 'AdvancedModelingPipeline':
        """Save model, predictions, and results."""
        print("\nSaving modeling results...")
        
        # Create results directory
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # Save best model
        joblib.dump(self.best_model, 'results/best_model.pkl')
        
        # Save predictions
        predictions_df = pd.DataFrame({
            'Id': range(1461, 1461 + len(self.test_predictions)),
            'SalePrice_transformed': self.test_predictions
        })
        predictions_df.to_csv('results/test_predictions.csv', index=False)
        
        # Save performance metrics
        with open('results/model_performance.json', 'w') as f:
            json.dump(self.results, f, indent=4, default=str)
        
        print("SUCCESS: All results saved to results/ directory")
        
        return self
    
    def run_complete_pipeline(self) -> 'AdvancedModelingPipeline':
        """Execute complete advanced modeling pipeline."""
        print("PHASE 7: ADVANCED MACHINE LEARNING MODELING")
        print("="*70)
        
        return (self
                .load_prepared_data()
                .evaluate_models()
                .select_best_model()
                .optimize_hyperparameters()  
                .generate_predictions()
                .save_results())

if __name__ == "__main__":
    pipeline = AdvancedModelingPipeline()
    pipeline.run_complete_pipeline()
    
    print("\n" + "="*70)
    print("SUCCESS PHASE 7 COMPLETED!")
    print("="*70)
    print(f"RESULT Best Model: {pipeline.best_model_name}")
    print(f"RESULT CV RMSE: {pipeline.results[pipeline.best_model_name]['cv_rmse_mean']:.4f}")
    print(f"RESULT Train R²: {pipeline.results[pipeline.best_model_name]['train_r2']:.4f}")
    print("TARGET: Ready for Phase 8 (Model Interpretation)")
    print("="*70)
```

### **7.2 Configuration Template**
**File**: `modeling/phase7_advanced_modeling/config/modeling_config.json`

```json
{
    "data_paths": {
        "train_data": "../preprocessing/phase6_scaling/final_train_prepared.csv",
        "test_data": "../preprocessing/phase6_scaling/final_test_prepared.csv"
    },
    "model_parameters": {
        "ridge": {
            "alpha": [0.1, 1.0, 10.0, 100.0]
        },
        "random_forest": {
            "n_estimators": [100, 500, 1000],
            "max_depth": [10, 20, 30, null],
            "min_samples_split": [2, 5, 10]
        },
        "xgboost": {
            "n_estimators": [100, 500, 1000],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 6, 9]
        }
    },
    "validation": {
        "cv_folds": 5,
        "test_size": 0.2,
        "random_state": 42
    },
    "performance_targets": {
        "min_r2": 0.90,
        "max_rmse": 0.12,
        "max_mae": 0.09
    }
}
```

---

## 🧠 **Phase 8 Template: Model Interpretation**

### **8.1 SHAP Explanations Template**
**File**: `modeling/phase8_model_interpretation/shap_explanations.py`

```python
"""
PHASE 8: MODEL INTERPRETATION & EXPLAINABILITY PIPELINE
======================================================================
Comprehensive model interpretation using SHAP, LIME, and business analysis.

Key Features:
- Global and local feature importance with SHAP
- Individual prediction explanations
- Business insight generation
- Interactive visualization dashboards
- Regulatory compliance reporting

Author: Advanced ML Pipeline  
Date: 2025
======================================================================
"""

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

class ModelInterpretationPipeline:
    """
    Advanced model interpretation and explainability system.
    """
    
    def __init__(self):
        """Initialize interpretation pipeline."""
        self.model = None
        self.X_train = None
        self.X_test = None
        self.shap_explainer = None
        self.shap_values = None
        self.feature_importance = {}
        
    def load_model_and_data(self) -> 'ModelInterpretationPipeline':
        """Load trained model and datasets."""
        print("Loading trained model and data...")
        
        # Load best model from Phase 7
        self.model = joblib.load('../phase7_advanced_modeling/results/best_model.pkl')
        
        # Load datasets
        train_data = pd.read_csv('../phase7_advanced_modeling/results/training_data.csv')
        self.X_train = train_data.drop('SalePrice_transformed', axis=1)
        self.y_train = train_data['SalePrice_transformed']
        
        self.X_test = pd.read_csv('../phase7_advanced_modeling/results/test_data.csv')
        
        print(f"Model type: {type(self.model).__name__}")
        print(f"Training data: {self.X_train.shape}")
        print(f"Test data: {self.X_test.shape}")
        
        return self
    
    def generate_shap_explanations(self) -> 'ModelInterpretationPipeline':
        """Generate SHAP explanations for model interpretability."""
        print("\n" + "="*60)
        print("GENERATING SHAP EXPLANATIONS")
        print("="*60)
        
        # Create SHAP explainer based on model type
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            self.shap_explainer = shap.TreeExplainer(self.model)
            print("Using TreeExplainer for tree-based model")
        else:
            # Linear models
            self.shap_explainer = shap.LinearExplainer(self.model, self.X_train)
            print("Using LinearExplainer for linear model")
        
        # Calculate SHAP values for training set (sample for efficiency)
        sample_size = min(1000, len(self.X_train))
        X_sample = self.X_train.sample(n=sample_size, random_state=42)
        
        print(f"Calculating SHAP values for {sample_size} samples...")
        self.shap_values = self.shap_explainer.shap_values(X_sample)
        
        return self
    
    def analyze_global_importance(self) -> 'ModelInterpretationPipeline':
        """Analyze global feature importance."""
        print("\n" + "="*60)
        print("ANALYZING GLOBAL FEATURE IMPORTANCE")
        print("="*60)
        
        # Calculate mean absolute SHAP values
        mean_shap_values = np.mean(np.abs(self.shap_values), axis=0)
        feature_names = self.X_train.columns
        
        # Create importance ranking
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_shap_values
        }).sort_values('importance', ascending=False)
        
        self.feature_importance['global'] = importance_df
        
        # Display top 15 features
        print("Top 15 Most Important Features:")
        for i, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<30} {row['importance']:.4f}")
        
        return self
    
    def generate_business_insights(self) -> 'ModelInterpretationPipeline':
        """Generate actionable business insights."""
        print("\n" + "="*60)
        print("GENERATING BUSINESS INSIGHTS")
        print("="*60)
        
        top_features = self.feature_importance['global'].head(10)
        
        # Categorize features by type
        feature_categories = {
            'Size/Area': [],
            'Quality/Condition': [],
            'Location': [], 
            'Age/Time': [],
            'Amenities': []
        }
        
        for _, row in top_features.iterrows():
            feature = row['feature']
            if any(term in feature.lower() for term in ['sf', 'area', 'size']):
                feature_categories['Size/Area'].append(feature)
            elif any(term in feature.lower() for term in ['qual', 'cond', 'quality']):
                feature_categories['Quality/Condition'].append(feature)
            elif any(term in feature.lower() for term in ['neighborhood', 'location']):
                feature_categories['Location'].append(feature)
            elif any(term in feature.lower() for term in ['age', 'year', 'built']):
                feature_categories['Age/Time'].append(feature)
            else:
                feature_categories['Amenities'].append(feature)
        
        # Generate insights
        insights = []
        for category, features in feature_categories.items():
            if features:
                insights.append(f"{category}: {len(features)} key factors - {', '.join(features[:3])}")
        
        print("Key Business Insights:")
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
        
        self.business_insights = insights
        
        return self
    
    def create_interpretation_dashboard(self) -> 'ModelInterpretationPipeline':
        """Create comprehensive interpretation dashboard."""
        print("\nGenerating interpretation dashboard...")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Global feature importance
        top_features = self.feature_importance['global'].head(15)
        axes[0,0].barh(range(len(top_features)), top_features['importance'])
        axes[0,0].set_yticks(range(len(top_features)))
        axes[0,0].set_yticklabels([f[:20] + '...' if len(f) > 20 else f for f in top_features['feature']])
        axes[0,0].set_title('Top 15 Feature Importance (SHAP)', fontweight='bold')
        axes[0,0].set_xlabel('Mean |SHAP Value|')
        
        # 2. SHAP summary plot (if available)
        if hasattr(shap, 'summary_plot'):
            plt.sca(axes[0,1])
            shap.summary_plot(self.shap_values, self.X_train.sample(100), show=False)
            axes[0,1].set_title('SHAP Summary Plot', fontweight='bold')
        
        # 3. Feature category importance
        category_importance = {}
        for category, features in [('Size/Area', ['TotalSF', 'GrLivArea']), 
                                 ('Quality', ['OverallQual', 'ExterQual']),
                                 ('Location', ['Neighborhood'])]:
            category_features = [f for f in features if f in self.feature_importance['global']['feature'].values]
            if category_features:
                category_importance[category] = self.feature_importance['global'][
                    self.feature_importance['global']['feature'].isin(category_features)
                ]['importance'].sum()
        
        if category_importance:
            axes[1,0].pie(category_importance.values(), labels=category_importance.keys(), 
                         autopct='%1.1f%%', startangle=90)
            axes[1,0].set_title('Feature Category Importance', fontweight='bold')
        
        # 4. Business insights text
        axes[1,1].axis('off')
        insights_text = "KEY BUSINESS INSIGHTS:\n\n" + "\n\n".join([f"{i}. {insight}" for i, insight in enumerate(self.business_insights, 1)])
        axes[1,1].text(0.1, 0.9, insights_text, transform=axes[1,1].transAxes, 
                      fontsize=11, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('interpretation_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self
    
    def run_complete_pipeline(self) -> 'ModelInterpretationPipeline':
        """Execute complete interpretation pipeline."""
        print("PHASE 8: MODEL INTERPRETATION & EXPLAINABILITY")
        print("="*70)
        
        return (self
                .load_model_and_data()
                .generate_shap_explanations()
                .analyze_global_importance()
                .generate_business_insights()
                .create_interpretation_dashboard())

if __name__ == "__main__":
    pipeline = ModelInterpretationPipeline()
    pipeline.run_complete_pipeline()
    
    print("\n" + "="*70)
    print("SUCCESS PHASE 8 COMPLETED!")
    print("="*70)
    print("RESULT: Model interpretation and business insights generated")
    print("TARGET: Ready for Phase 9 (Production Deployment)")
    print("="*70)
```

---

## 🚀 **Phase 9 Template: Production Deployment**

### **9.1 Production API Template**
**File**: `deployment/phase9_production/api/prediction_service.py`

```python
"""
PHASE 9: PRODUCTION DEPLOYMENT - PREDICTION SERVICE
======================================================================
Production-ready API service for house price prediction with enterprise
features including monitoring, logging, and performance optimization.

Key Features:
- FastAPI REST endpoints with <100ms response time
- Input validation and preprocessing pipeline integration
- Confidence intervals and prediction explanations
- Comprehensive logging and monitoring
- Auto-scaling and load balancing ready

Author: Advanced ML Pipeline
Date: 2025
======================================================================
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np
import joblib
import logging
import time
from typing import List, Dict, Optional, Any
from datetime import datetime
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Advanced House Price Prediction API",
    description="State-of-the-art house price prediction with ML interpretability",
    version="1.0.0"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model and preprocessor
model = None
feature_names = None
scaler = None

class HouseFeatures(BaseModel):
    """Input schema for house price prediction."""
    
    # Core features (example - adapt based on your final feature set)
    TotalSF: float = Field(..., gt=0, description="Total square footage")
    OverallQual: int = Field(..., ge=1, le=10, description="Overall quality rating")
    GrLivArea: float = Field(..., gt=0, description="Ground living area")
    YearBuilt: int = Field(..., ge=1800, le=2030, description="Year built")
    Neighborhood: str = Field(..., description="Neighborhood name")
    
    # Add all your processed features here...
    # This would include all 224 features from your final dataset
    
    @validator('Neighborhood')
    def validate_neighborhood(cls, v):
        # Add validation for known neighborhoods
        valid_neighborhoods = ['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel']  # Add all
        if v not in valid_neighborhoods:
            raise ValueError(f'Invalid neighborhood: {v}')
        return v

class PricePrediction(BaseModel):
    """Output schema for price prediction."""
    
    predicted_price: float = Field(..., description="Predicted house price")
    confidence_interval_lower: float = Field(..., description="Lower bound (95% CI)")
    confidence_interval_upper: float = Field(..., description="Upper bound (95% CI)")
    prediction_confidence: float = Field(..., description="Model confidence score")
    top_price_factors: List[Dict[str, Any]] = Field(..., description="Key factors driving price")
    prediction_timestamp: datetime = Field(default_factory=datetime.now)

@app.on_event("startup")
async def load_model():
    """Load trained model and preprocessing components on startup."""
    global model, feature_names, scaler
    
    try:
        # Load trained model
        model = joblib.load('../modeling/phase7_advanced_modeling/results/best_model.pkl')
        
        # Load feature names and preprocessing components
        with open('../modeling/phase7_advanced_modeling/results/feature_names.json', 'r') as f:
            feature_names = json.load(f)
        
        scaler = joblib.load('../preprocessing/phase6_scaling/scaler.pkl')
        
        logger.info("Model and preprocessing components loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

@app.get("/")
async def root():
    """API health check endpoint."""
    return {
        "message": "Advanced House Price Prediction API",
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now()
    }

@app.get("/health")
async def health_check():
    """Detailed health check with model status."""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "features_count": len(feature_names) if feature_names else 0,
        "timestamp": datetime.now()
    }

@app.post("/predict/price", response_model=PricePrediction)
async def predict_house_price(
    features: HouseFeatures,
    background_tasks: BackgroundTasks
) -> PricePrediction:
    """
    Predict house price with confidence intervals and explanations.
    
    Args:
        features: House characteristics for prediction
        
    Returns:
        Comprehensive prediction with confidence and explanations
    """
    start_time = time.time()
    
    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([features.dict()])
        
        # Apply preprocessing pipeline (feature engineering, encoding, scaling)
        processed_data = preprocess_for_prediction(input_data)
        
        # Generate prediction
        prediction = model.predict(processed_data)[0]
        
        # Calculate confidence intervals (using prediction intervals)
        prediction_std = calculate_prediction_uncertainty(processed_data)
        ci_lower = prediction - 1.96 * prediction_std
        ci_upper = prediction + 1.96 * prediction_std
        
        # Generate feature importance for this prediction
        top_factors = get_prediction_explanations(processed_data)
        
        # Log prediction
        prediction_time = time.time() - start_time
        background_tasks.add_task(log_prediction, features.dict(), prediction, prediction_time)
        
        # Transform prediction back to original price scale
        original_price = inverse_transform_price(prediction)
        ci_lower_price = inverse_transform_price(ci_lower)
        ci_upper_price = inverse_transform_price(ci_upper)
        
        return PricePrediction(
            predicted_price=original_price,
            confidence_interval_lower=ci_lower_price,
            confidence_interval_upper=ci_upper_price,
            prediction_confidence=calculate_confidence_score(prediction_std),
            top_price_factors=top_factors
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=List[PricePrediction])
async def predict_batch(
    houses: List[HouseFeatures],
    background_tasks: BackgroundTasks
) -> List[PricePrediction]:
    """Batch prediction endpoint for multiple houses."""
    
    if len(houses) > 100:
        raise HTTPException(status_code=400, detail="Batch size limited to 100 houses")
    
    predictions = []
    for house in houses:
        prediction = await predict_house_price(house, background_tasks)
        predictions.append(prediction)
    
    return predictions

def preprocess_for_prediction(input_data: pd.DataFrame) -> pd.DataFrame:
    """Apply full preprocessing pipeline to input data."""
    # This would apply the same transformations as your preprocessing pipeline
    # Including feature engineering, encoding, and scaling
    
    # Example structure:
    # 1. Apply feature engineering from Phase 3
    # 2. Apply encoding from Phase 5  
    # 3. Apply scaling from Phase 6
    # 4. Ensure feature order matches training data
    
    processed = input_data.copy()
    
    # Apply your preprocessing steps here...
    # processed = apply_feature_engineering(processed)
    # processed = apply_encoding(processed)
    # processed = apply_scaling(processed)
    
    return processed

def calculate_prediction_uncertainty(data: pd.DataFrame) -> float:
    """Calculate prediction uncertainty/standard error."""
    # Implement uncertainty quantification
    # This could use model-specific methods or bootstrapping
    return 0.1  # Placeholder

def get_prediction_explanations(data: pd.DataFrame) -> List[Dict[str, Any]]:
    """Generate explanations for individual predictions."""
    # Use SHAP or similar for local explanations
    return [
        {"feature": "TotalSF", "importance": 0.25, "value": 2000},
        {"feature": "OverallQual", "importance": 0.20, "value": 8},
        {"feature": "Neighborhood", "importance": 0.15, "value": "NoRidge"}
    ]

def calculate_confidence_score(uncertainty: float) -> float:
    """Calculate normalized confidence score."""
    # Convert uncertainty to confidence score (0-1)
    return max(0, min(1, 1 - uncertainty * 5))

def inverse_transform_price(transformed_price: float) -> float:
    """Transform prediction back to original price scale."""
    # Apply inverse of BoxCox transformation from Phase 1
    # This would use the saved lambda parameter
    return np.expm1(transformed_price)  # Simplified

async def log_prediction(features: dict, prediction: float, response_time: float):
    """Log prediction for monitoring and analytics."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "features": features,
        "prediction": prediction,
        "response_time_ms": response_time * 1000,
        "model_version": "1.0.0"
    }
    
    # Log to file, database, or monitoring system
    logger.info(f"Prediction logged: {log_entry}")

if __name__ == "__main__":
    uvicorn.run(
        "prediction_service:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to False in production
        workers=4  # Adjust based on system resources
    )
```

---

## 📊 **Quality Assurance Templates**

### **Testing Template**
**File**: `tests/test_phase_template.py`

```python
"""
Comprehensive testing template for all phases.
Ensures consistent quality across all pipeline components.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

class TestPhaseTemplate(unittest.TestCase):
    """Template for phase testing with standard test cases."""
    
    def setUp(self):
        """Set up test fixtures and sample data."""
        self.sample_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.1, 0.2, 0.3, 0.4, 0.5],
            'target': [100, 200, 300, 400, 500]
        })
        
    def test_data_loading(self):
        """Test data loading functionality."""
        # Test that data loads correctly
        self.assertIsInstance(self.sample_data, pd.DataFrame)
        self.assertEqual(len(self.sample_data), 5)
        self.assertEqual(list(self.sample_data.columns), ['feature1', 'feature2', 'target'])
        
    def test_data_validation(self):
        """Test data validation and quality checks."""
        # Test for missing values
        self.assertEqual(self.sample_data.isnull().sum().sum(), 0)
        
        # Test data types
        self.assertTrue(pd.api.types.is_numeric_dtype(self.sample_data['feature1']))
        self.assertTrue(pd.api.types.is_numeric_dtype(self.sample_data['feature2']))
        
    def test_transformation_logic(self):
        """Test core transformation functionality."""
        # Test that transformations preserve data integrity
        transformed = self.sample_data.copy()
        
        # Example transformation
        transformed['feature1_scaled'] = (transformed['feature1'] - transformed['feature1'].mean()) / transformed['feature1'].std()
        
        # Validate transformation
        self.assertAlmostEqual(transformed['feature1_scaled'].mean(), 0, places=10)
        self.assertAlmostEqual(transformed['feature1_scaled'].std(), 1, places=10)
        
    def test_error_handling(self):
        """Test error handling and edge cases."""
        # Test empty dataframe
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            # Should raise error for empty data
            self.validate_non_empty_data(empty_df)
            
    def test_output_format(self):
        """Test output format and structure."""
        # Test that outputs match expected format
        result = self.sample_data.copy()
        
        # Validate output structure
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result.columns), 0)
        
    def test_performance_benchmarks(self):
        """Test performance requirements."""
        import time
        
        start_time = time.time()
        
        # Simulate processing
        result = self.sample_data.copy()
        for i in range(1000):
            result['temp'] = result['feature1'] * 2
        
        processing_time = time.time() - start_time
        
        # Should complete within reasonable time
        self.assertLess(processing_time, 1.0)  # Less than 1 second
        
    def validate_non_empty_data(self, data):
        """Helper method to validate data is not empty."""
        if len(data) == 0:
            raise ValueError("Data cannot be empty")
        return True

class TestIntegrationTemplate(unittest.TestCase):
    """Template for integration testing between phases."""
    
    def test_phase_data_compatibility(self):
        """Test that output from one phase is compatible with next phase."""
        # This would test that Phase N output is valid Phase N+1 input
        pass
        
    def test_end_to_end_pipeline(self):
        """Test complete pipeline execution."""
        # This would test the entire pipeline from start to finish
        pass

if __name__ == '__main__':
    unittest.main()
```

---

## 📋 **Execution Checklist Template**

### **Phase Completion Checklist**
```markdown
# Phase X Completion Checklist

## ✅ Implementation
- [ ] Core pipeline class implemented with standard structure
- [ ] Configuration management with JSON files
- [ ] Modular components with clear interfaces
- [ ] Error handling and logging throughout
- [ ] Input/output validation implemented

## ✅ Quality Assurance  
- [ ] Unit tests written and passing (>90% coverage)
- [ ] Integration tests with previous phase
- [ ] Performance benchmarks met
- [ ] Code review completed
- [ ] Documentation updated

## ✅ Results & Validation
- [ ] Success metrics achieved (quantified)
- [ ] Results visualization generated
- [ ] Statistical validation completed
- [ ] Business validation performed
- [ ] Comparison with baseline/benchmarks

## ✅ Documentation
- [ ] Technical methodology documented  
- [ ] Implementation guide written
- [ ] Results analysis completed
- [ ] Configuration parameters explained
- [ ] Usage examples provided

## ✅ Artifacts Generated
- [ ] Processed data files saved
- [ ] Model/transformation objects serialized
- [ ] Configuration files saved
- [ ] Visualization dashboards created
- [ ] Performance metrics logged

## ✅ Integration & Handoff
- [ ] Output format validated for next phase
- [ ] Compatibility tested with downstream processes
- [ ] Handoff documentation completed
- [ ] Next phase requirements defined
- [ ] Success criteria for next phase established
```

---

This comprehensive template system ensures **consistent, world-class implementation** across all future phases while maintaining the **exceptional standards** established in the preprocessing pipeline! 🎯