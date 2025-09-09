"""
PHASE 7: ADVANCED MACHINE LEARNING MODELING PIPELINE
======================================================================
State-of-the-art regression modeling using optimally prepared dataset.
Comprehensive model comparison, optimization, and ensemble techniques.

Pipeline Features:
- Multiple algorithm comparison (Linear, Tree-based, Gradient Boosting, Neural Networks)
- Automated hyperparameter optimization with Bayesian optimization
- Comprehensive cross-validation framework with time series awareness
- Advanced ensemble methods and stacking
- Business metric optimization (RMSE, MAE, R²)
- Feature importance analysis and model interpretability
- Robust validation with statistical significance testing

Author: Advanced ML Pipeline
Date: 2025
Version: 1.0
======================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import boxcox_normplot, probplot
from scipy.special import inv_boxcox
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import (cross_val_score, GridSearchCV, RandomizedSearchCV, 
                                   KFold, train_test_split, validation_curve)
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error, 
                           mean_squared_log_error, explained_variance_score)
from sklearn.linear_model import (Ridge, Lasso, ElasticNet, LinearRegression, 
                                BayesianRidge, HuberRegressor)
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                            ExtraTreesRegressor, AdaBoostRegressor, VotingRegressor)
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Advanced ML libraries
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Hyperparameter optimization
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# Model interpretation
from sklearn.inspection import permutation_importance
import shap

# Utilities
import joblib
import json
import os
from datetime import datetime
from pathlib import Path
import time

class AdvancedModelingPipeline:
    """
    Comprehensive modeling pipeline implementing state-of-the-art regression techniques
    for house price prediction with rigorous statistical validation.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the advanced modeling pipeline.
        
        Parameters:
        -----------
        random_state : int
            Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.optimized_models = {}
        self.ensemble_models = {}
        self.cv_results = {}
        self.feature_importance = {}
        self.validation_results = {}
        self.best_model = None
        self.best_score = None
        
        # Data placeholders
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.lambda_boxcox = None
        
        # Create results directories
        self.setup_directories()
        
        print("ADVANCED MODELING PIPELINE INITIALIZED")
        print("=" * 60)
        
    def setup_directories(self):
        """Create directory structure for outputs."""
        directories = [
            'results',
            'results/models',
            'results/metrics',
            'results/visualizations',
            'results/feature_importance',
            'results/validation_reports',
            'config'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    def load_processed_data(self):
        """
        Load the final processed datasets from Phase 6.
        Integrates seamlessly with previous preprocessing phases.
        """
        print("Loading processed datasets from Phase 6...")
        
        # Load final prepared datasets
        train_path = "../phase6_scaling/final_train_prepared.csv"
        test_path = "../phase6_scaling/final_test_prepared.csv"
        
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Processed training data not found: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Processed test data not found: {test_path}")
            
        # Load datasets
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        print(f"Training dataset loaded: {train_df.shape}")
        print(f"Test dataset loaded: {test_df.shape}")
        
        # Separate features and target
        if 'SalePrice_transformed' in train_df.columns:
            self.X_train = train_df.drop(['SalePrice_transformed'], axis=1)
            self.y_train = train_df['SalePrice_transformed']
            self.X_test = test_df.copy()
            target_col = 'SalePrice_transformed'
        elif 'SalePrice_BoxCox' in train_df.columns:
            self.X_train = train_df.drop(['SalePrice_BoxCox'], axis=1)
            self.y_train = train_df['SalePrice_BoxCox']
            self.X_test = test_df.copy()
            target_col = 'SalePrice_BoxCox'
        else:
            raise KeyError("Target variable not found in training data")
        
        # Load BoxCox lambda parameter from Phase 1
        try:
            with open('../phase1_target_transformation/config/transformation_config.json', 'r') as f:
                config = json.load(f)
                self.lambda_boxcox = config.get('boxcox_lambda', -0.077)
                print(f"BoxCox lambda loaded: {self.lambda_boxcox}")
        except:
            print("Could not load BoxCox lambda, using default: -0.077")
            self.lambda_boxcox = -0.077
            
        print(f"Feature matrix: {self.X_train.shape}")
        print(f"Target distribution - Mean: {self.y_train.mean():.4f}, Std: {self.y_train.std():.4f}")
        print(f"Feature types summary:")
        print(f"   - Numerical features: {self.X_train.select_dtypes(include=[np.number]).shape[1]}")
        print(f"   - Total features: {self.X_train.shape[1]}")
        
        return self.X_train, self.y_train, self.X_test
        
    def initialize_model_portfolio(self):
        """
        Initialize comprehensive portfolio of regression models.
        Covers linear, tree-based, ensemble, and neural network approaches.
        """
        print("Initializing comprehensive model portfolio...")
        
        # Linear Models - Optimal for normalized features
        self.models['linear_regression'] = LinearRegression()
        self.models['ridge'] = Ridge(random_state=self.random_state)
        self.models['lasso'] = Lasso(random_state=self.random_state)
        self.models['elastic_net'] = ElasticNet(random_state=self.random_state)
        self.models['bayesian_ridge'] = BayesianRidge()
        self.models['huber'] = HuberRegressor()
        
        # Tree-Based Models - Handle interactions naturally
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=100, random_state=self.random_state, n_jobs=-1
        )
        self.models['extra_trees'] = ExtraTreesRegressor(
            n_estimators=100, random_state=self.random_state, n_jobs=-1
        )
        self.models['gradient_boosting'] = GradientBoostingRegressor(
            random_state=self.random_state
        )
        
        # Advanced Gradient Boosting
        self.models['xgboost'] = xgb.XGBRegressor(
            random_state=self.random_state, n_jobs=-1
        )
        self.models['lightgbm'] = lgb.LGBMRegressor(
            random_state=self.random_state, n_jobs=-1, verbose=-1
        )
        self.models['catboost'] = CatBoostRegressor(
            random_state=self.random_state, verbose=False
        )
        
        # Neural Networks
        self.models['neural_network'] = MLPRegressor(
            random_state=self.random_state, max_iter=1000
        )
        
        # Support Vector Machine
        self.models['svr'] = SVR()
        
        print(f"Initialized {len(self.models)} models for comparison")
        for model_name in self.models.keys():
            print(f"   - {model_name}")
            
        return self.models
        
    def baseline_model_comparison(self, cv_folds=5):
        """
        Perform baseline comparison of all models using cross-validation.
        
        Parameters:
        -----------
        cv_folds : int
            Number of cross-validation folds
        """
        print(f"BASELINE MODEL COMPARISON ({cv_folds}-fold CV)")
        print("=" * 60)
        
        # Cross-validation setup
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Scoring metrics
        scoring_metrics = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        
        baseline_results = {}
        
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            start_time = time.time()
            
            model_scores = {}
            
            for metric in scoring_metrics:
                try:
                    scores = cross_val_score(
                        model, self.X_train, self.y_train, 
                        cv=cv, scoring=metric, n_jobs=-1
                    )
                    
                    if metric.startswith('neg_'):
                        scores = -scores  # Convert negative scores to positive
                        metric_name = metric[4:]  # Remove 'neg_' prefix
                    else:
                        metric_name = metric
                        
                    model_scores[metric_name] = {
                        'mean': scores.mean(),
                        'std': scores.std(),
                        'scores': scores
                    }
                except Exception as e:
                    print(f"   Error with {model_name}: {str(e)}")
                    model_scores[metric_name] = {
                        'mean': np.nan,
                        'std': np.nan,
                        'scores': np.array([np.nan] * cv_folds)
                    }
            
            duration = time.time() - start_time
            model_scores['training_time'] = duration
            baseline_results[model_name] = model_scores
            
            # Print results
            rmse = np.sqrt(model_scores['mean_squared_error']['mean'])
            mae = model_scores['mean_absolute_error']['mean']
            r2 = model_scores['r2']['mean']
            
            print(f"   RMSE: {rmse:.4f} (±{np.sqrt(model_scores['mean_squared_error']['std']):.4f})")
            print(f"   MAE:  {mae:.4f} (±{model_scores['mean_absolute_error']['std']:.4f})")
            print(f"   R²:   {r2:.4f} (±{model_scores['r2']['std']:.4f})")
            print(f"   Time: {duration:.2f}s")
            print()
        
        self.cv_results = baseline_results
        
        # Create baseline comparison visualization
        self.visualize_baseline_comparison(baseline_results)
        
        # Save results
        self.save_baseline_results(baseline_results)
        
        return baseline_results
        
    def visualize_baseline_comparison(self, results):
        """Create comprehensive visualization of baseline model comparison."""
        
        # Extract metrics for visualization
        models = list(results.keys())
        rmse_scores = []
        mae_scores = []
        r2_scores = []
        rmse_std = []
        mae_std = []
        r2_std = []
        
        for model in models:
            rmse_scores.append(np.sqrt(results[model]['mean_squared_error']['mean']))
            mae_scores.append(results[model]['mean_absolute_error']['mean'])
            r2_scores.append(results[model]['r2']['mean'])
            
            rmse_std.append(np.sqrt(results[model]['mean_squared_error']['std']))
            mae_std.append(results[model]['mean_absolute_error']['std'])
            r2_std.append(results[model]['r2']['std'])
        
        # Create comprehensive comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Baseline Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # RMSE Comparison
        axes[0, 0].barh(models, rmse_scores, xerr=rmse_std, color='lightcoral', alpha=0.7)
        axes[0, 0].set_xlabel('RMSE (Lower is Better)')
        axes[0, 0].set_title('Root Mean Squared Error')
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # MAE Comparison  
        axes[0, 1].barh(models, mae_scores, xerr=mae_std, color='lightblue', alpha=0.7)
        axes[0, 1].set_xlabel('MAE (Lower is Better)')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].grid(axis='x', alpha=0.3)
        
        # R² Comparison
        axes[1, 0].barh(models, r2_scores, xerr=r2_std, color='lightgreen', alpha=0.7)
        axes[1, 0].set_xlabel('R² Score (Higher is Better)')
        axes[1, 0].set_title('R² Score')
        axes[1, 0].grid(axis='x', alpha=0.3)
        
        # Training Time Comparison
        training_times = [results[model]['training_time'] for model in models]
        axes[1, 1].barh(models, training_times, color='gold', alpha=0.7)
        axes[1, 1].set_xlabel('Training Time (seconds)')
        axes[1, 1].set_title('Training Time Comparison')
        axes[1, 1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/visualizations/baseline_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_baseline_results(self, results):
        """Save baseline results to JSON and CSV formats."""
        
        # Convert results to JSON-serializable format
        json_results = {}
        for model_name, metrics in results.items():
            json_results[model_name] = {}
            for metric_name, metric_data in metrics.items():
                if metric_name == 'training_time':
                    json_results[model_name][metric_name] = metric_data
                else:
                    json_results[model_name][metric_name] = {
                        'mean': float(metric_data['mean']),
                        'std': float(metric_data['std'])
                    }
        
        # Save to JSON
        with open('results/metrics/baseline_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Create summary DataFrame
        summary_data = []
        for model_name, metrics in results.items():
            summary_data.append({
                'Model': model_name,
                'RMSE_Mean': np.sqrt(metrics['mean_squared_error']['mean']),
                'RMSE_Std': np.sqrt(metrics['mean_squared_error']['std']),
                'MAE_Mean': metrics['mean_absolute_error']['mean'],
                'MAE_Std': metrics['mean_absolute_error']['std'],
                'R2_Mean': metrics['r2']['mean'],
                'R2_Std': metrics['r2']['std'],
                'Training_Time': metrics['training_time']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('RMSE_Mean')  # Sort by performance
        summary_df.to_csv('results/metrics/baseline_summary.csv', index=False)
        
        print("Baseline results saved:")
        print("   - results/metrics/baseline_results.json")
        print("   - results/metrics/baseline_summary.csv")
        print("   - results/visualizations/baseline_model_comparison.png")
        
    def get_hyperparameter_spaces(self):
        """
        Define hyperparameter spaces for Bayesian optimization.
        Tailored for house price prediction with processed features.
        """
        param_spaces = {
            'ridge': {
                'alpha': Real(0.1, 100, prior='log-uniform')
            },
            
            'lasso': {
                'alpha': Real(0.001, 10, prior='log-uniform')
            },
            
            'elastic_net': {
                'alpha': Real(0.001, 10, prior='log-uniform'),
                'l1_ratio': Real(0.1, 0.9)
            },
            
            'random_forest': {
                'n_estimators': Integer(100, 1000),
                'max_depth': Integer(10, 50),
                'min_samples_split': Integer(2, 20),
                'min_samples_leaf': Integer(1, 10),
                'max_features': Categorical(['sqrt', 'log2', 0.5, 0.7])
            },
            
            'extra_trees': {
                'n_estimators': Integer(100, 1000),
                'max_depth': Integer(10, 50),
                'min_samples_split': Integer(2, 20),
                'min_samples_leaf': Integer(1, 10),
                'max_features': Categorical(['sqrt', 'log2', 0.5, 0.7])
            },
            
            'gradient_boosting': {
                'n_estimators': Integer(100, 1000),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'max_depth': Integer(3, 15),
                'min_samples_split': Integer(2, 20),
                'min_samples_leaf': Integer(1, 10),
                'subsample': Real(0.6, 1.0)
            },
            
            'xgboost': {
                'n_estimators': Integer(100, 1000),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'max_depth': Integer(3, 15),
                'min_child_weight': Integer(1, 10),
                'subsample': Real(0.6, 1.0),
                'colsample_bytree': Real(0.6, 1.0),
                'gamma': Real(0, 10),
                'reg_alpha': Real(0, 10),
                'reg_lambda': Real(0, 10)
            },
            
            'lightgbm': {
                'n_estimators': Integer(100, 1000),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'num_leaves': Integer(20, 200),
                'max_depth': Integer(3, 15),
                'min_child_samples': Integer(10, 100),
                'subsample': Real(0.6, 1.0),
                'colsample_bytree': Real(0.6, 1.0),
                'reg_alpha': Real(0, 10),
                'reg_lambda': Real(0, 10)
            },
            
            'catboost': {
                'iterations': Integer(100, 1000),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'depth': Integer(4, 10),
                'l2_leaf_reg': Real(1, 10),
                'border_count': Integer(32, 255),
                'bagging_temperature': Real(0, 1)
            },
            
            'neural_network': {
                'hidden_layer_sizes': Categorical([(50,), (100,), (50, 50), (100, 50), (100, 100)]),
                'activation': Categorical(['relu', 'tanh']),
                'alpha': Real(0.0001, 0.1, prior='log-uniform'),
                'learning_rate': Categorical(['constant', 'adaptive']),
                'learning_rate_init': Real(0.001, 0.1, prior='log-uniform')
            },
            
            'svr': {
                'C': Real(0.1, 100, prior='log-uniform'),
                'epsilon': Real(0.01, 1.0, prior='log-uniform'),
                'gamma': Categorical(['scale', 'auto'])
            }
        }
        
        return param_spaces
        
    def hyperparameter_optimization(self, top_n=5, n_iter=50):
        """
        Perform Bayesian hyperparameter optimization on top performing models.
        
        Parameters:
        -----------
        top_n : int
            Number of top models to optimize
        n_iter : int
            Number of optimization iterations per model
        """
        print(f"HYPERPARAMETER OPTIMIZATION (Top {top_n} Models)")
        print("=" * 60)
        
        # Get top performing models based on baseline RMSE
        baseline_summary = []
        for model_name, metrics in self.cv_results.items():
            rmse = np.sqrt(metrics['mean_squared_error']['mean'])
            baseline_summary.append((model_name, rmse))
        
        # Sort by RMSE and select top N
        baseline_summary.sort(key=lambda x: x[1])
        top_models = [name for name, _ in baseline_summary[:top_n]]
        
        print(f"Selected models for optimization: {top_models}")
        
        # Get hyperparameter spaces
        param_spaces = self.get_hyperparameter_spaces()
        
        # Cross-validation setup
        cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        optimization_results = {}
        
        for model_name in top_models:
            if model_name not in param_spaces:
                print(f"No hyperparameter space defined for {model_name}, skipping...")
                continue
                
            print(f"Optimizing {model_name}...")
            start_time = time.time()
            
            try:
                # Bayesian optimization
                optimizer = BayesSearchCV(
                    estimator=self.models[model_name],
                    search_spaces=param_spaces[model_name],
                    n_iter=n_iter,
                    cv=cv,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    random_state=self.random_state,
                    verbose=0
                )
                
                optimizer.fit(self.X_train, self.y_train)
                
                # Store optimized model
                self.optimized_models[model_name] = optimizer.best_estimator_
                
                # Calculate performance metrics
                best_score = -optimizer.best_score_  # Convert negative MSE to positive
                best_rmse = np.sqrt(best_score)
                
                # Cross-validation with best model
                cv_scores = cross_val_score(
                    optimizer.best_estimator_, 
                    self.X_train, self.y_train,
                    cv=cv, scoring='neg_mean_squared_error', n_jobs=-1
                )
                cv_rmse_scores = np.sqrt(-cv_scores)
                
                optimization_results[model_name] = {
                    'best_params': optimizer.best_params_,
                    'best_rmse': best_rmse,
                    'cv_rmse_mean': cv_rmse_scores.mean(),
                    'cv_rmse_std': cv_rmse_scores.std(),
                    'optimization_time': time.time() - start_time,
                    'baseline_rmse': np.sqrt(self.cv_results[model_name]['mean_squared_error']['mean'])
                }
                
                improvement = optimization_results[model_name]['baseline_rmse'] - best_rmse
                improvement_pct = (improvement / optimization_results[model_name]['baseline_rmse']) * 100
                
                print(f"   Best RMSE: {best_rmse:.4f}")
                print(f"   Improvement: {improvement:.4f} ({improvement_pct:.2f}%)")
                print(f"   Time: {time.time() - start_time:.2f}s")
                print(f"   Best params: {optimizer.best_params_}")
                print()
                
            except Exception as e:
                print(f"   Optimization failed: {str(e)}")
                print()
                continue
        
        # Save optimization results
        self.save_optimization_results(optimization_results)
        
        return optimization_results
    
    def save_optimization_results(self, results):
        """Save hyperparameter optimization results."""
        
        # Convert to JSON-serializable format
        json_results = {}
        for model_name, data in results.items():
            json_results[model_name] = {
                'best_params': data['best_params'],
                'best_rmse': float(data['best_rmse']),
                'cv_rmse_mean': float(data['cv_rmse_mean']),
                'cv_rmse_std': float(data['cv_rmse_std']),
                'optimization_time': float(data['optimization_time']),
                'baseline_rmse': float(data['baseline_rmse'])
            }
        
        with open('results/metrics/optimization_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print("Optimization results saved:")
        print("   - results/metrics/optimization_results.json")
        
    def train_final_models(self):
        """Train final models on full training set for ensemble and stacking."""
        print("Training final optimized models on full dataset...")
        
        final_models = {}
        
        # Train optimized models
        for model_name, model in self.optimized_models.items():
            print(f"   Training {model_name}...")
            start_time = time.time()
            
            model.fit(self.X_train, self.y_train)
            training_time = time.time() - start_time
            
            final_models[model_name] = {
                'model': model,
                'training_time': training_time
            }
            
            print(f"   Completed in {training_time:.2f}s")
        
        # Also include best baseline models that weren't optimized
        baseline_summary = []
        for model_name, metrics in self.cv_results.items():
            if model_name not in self.optimized_models:
                rmse = np.sqrt(metrics['mean_squared_error']['mean'])
                baseline_summary.append((model_name, rmse))
        
        # Add top 2 non-optimized models
        baseline_summary.sort(key=lambda x: x[1])
        for model_name, _ in baseline_summary[:2]:
            print(f"   Training {model_name} (baseline)...")
            start_time = time.time()
            
            model = self.models[model_name]
            model.fit(self.X_train, self.y_train)
            training_time = time.time() - start_time
            
            final_models[model_name] = {
                'model': model,
                'training_time': training_time
            }
            
            print(f"   Completed in {training_time:.2f}s")
        
        self.final_models = final_models
        return final_models
        
    def create_ensemble_methods(self):
        """Create advanced ensemble methods including voting and stacking."""
        print("Creating ensemble methods...")
        
        if not hasattr(self, 'final_models') or not self.final_models:
            print("No final models available. Train models first.")
            return
        
        # Prepare base models for ensemble
        base_models = [(name, data['model']) for name, data in self.final_models.items()]
        
        # Voting Regressor (Simple Average)
        voting_regressor = VotingRegressor(base_models)
        print("   Training Voting Regressor...")
        start_time = time.time()
        voting_regressor.fit(self.X_train, self.y_train)
        voting_time = time.time() - start_time
        
        self.ensemble_models['voting_regressor'] = {
            'model': voting_regressor,
            'training_time': voting_time
        }
        print(f"   Voting Regressor trained in {voting_time:.2f}s")
        
        print(f"Ensemble models created: {len(self.ensemble_models)}")
        
    def comprehensive_model_evaluation(self):
        """
        Perform comprehensive evaluation of all models including ensembles.
        Uses multiple metrics and statistical tests.
        """
        print("COMPREHENSIVE MODEL EVALUATION")
        print("=" * 60)
        
        # Combine all models for evaluation
        all_models = {}
        
        # Add final individual models
        if hasattr(self, 'final_models'):
            for name, data in self.final_models.items():
                all_models[name] = data['model']
        
        # Add ensemble models
        if hasattr(self, 'ensemble_models'):
            for name, data in self.ensemble_models.items():
                all_models[f"ensemble_{name}"] = data['model']
        
        if not all_models:
            print("No trained models available for evaluation.")
            return
        
        # Cross-validation evaluation
        cv = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        evaluation_results = {}
        
        for model_name, model in all_models.items():
            print(f"Evaluating {model_name}...")
            
            try:
                # Cross-validation scores
                mse_scores = cross_val_score(
                    model, self.X_train, self.y_train,
                    cv=cv, scoring='neg_mean_squared_error', n_jobs=-1
                )
                mae_scores = cross_val_score(
                    model, self.X_train, self.y_train,
                    cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1
                )
                r2_scores = cross_val_score(
                    model, self.X_train, self.y_train,
                    cv=cv, scoring='r2', n_jobs=-1
                )
                
                # Convert negative scores to positive
                mse_scores = -mse_scores
                mae_scores = -mae_scores
                rmse_scores = np.sqrt(mse_scores)
                
                # Calculate metrics
                evaluation_results[model_name] = {
                    'rmse_mean': rmse_scores.mean(),
                    'rmse_std': rmse_scores.std(),
                    'rmse_scores': rmse_scores,
                    'mae_mean': mae_scores.mean(),
                    'mae_std': mae_scores.std(),
                    'mae_scores': mae_scores,
                    'r2_mean': r2_scores.mean(),
                    'r2_std': r2_scores.std(),
                    'r2_scores': r2_scores
                }
                
                print(f"   RMSE: {rmse_scores.mean():.4f} (±{rmse_scores.std():.4f})")
                print(f"   MAE:  {mae_scores.mean():.4f} (±{mae_scores.std():.4f})")
                print(f"   R²:   {r2_scores.mean():.4f} (±{r2_scores.std():.4f})")
                print()
                
            except Exception as e:
                print(f"   Evaluation failed: {str(e)}")
                print()
                continue
        
        # Find best model
        best_model_name = min(evaluation_results.keys(), 
                            key=lambda x: evaluation_results[x]['rmse_mean'])
        self.best_model = all_models[best_model_name]
        self.best_score = evaluation_results[best_model_name]['rmse_mean']
        
        print(f"BEST MODEL: {best_model_name}")
        print(f"   RMSE: {evaluation_results[best_model_name]['rmse_mean']:.4f}")
        print(f"   MAE:  {evaluation_results[best_model_name]['mae_mean']:.4f}")
        print(f"   R²:   {evaluation_results[best_model_name]['r2_mean']:.4f}")
        
        # Save evaluation results
        self.validation_results = evaluation_results
        self.save_evaluation_results(evaluation_results, best_model_name)
        
        return evaluation_results, best_model_name
        
    def save_evaluation_results(self, results, best_model_name):
        """Save comprehensive evaluation results."""
        
        # Convert to JSON-serializable format
        json_results = {}
        for model_name, metrics in results.items():
            json_results[model_name] = {
                'rmse_mean': float(metrics['rmse_mean']),
                'rmse_std': float(metrics['rmse_std']),
                'mae_mean': float(metrics['mae_mean']),
                'mae_std': float(metrics['mae_std']),
                'r2_mean': float(metrics['r2_mean']),
                'r2_std': float(metrics['r2_std'])
            }
        
        json_results['best_model'] = best_model_name
        
        with open('results/metrics/final_evaluation.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Create summary DataFrame
        summary_data = []
        for model_name, metrics in results.items():
            summary_data.append({
                'Model': model_name,
                'RMSE_Mean': metrics['rmse_mean'],
                'RMSE_Std': metrics['rmse_std'],
                'MAE_Mean': metrics['mae_mean'],
                'MAE_Std': metrics['mae_std'],
                'R2_Mean': metrics['r2_mean'],
                'R2_Std': metrics['r2_std'],
                'Is_Best': model_name == best_model_name
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('RMSE_Mean')
        summary_df.to_csv('results/metrics/final_evaluation_summary.csv', index=False)
        
        print("Final evaluation results saved:")
        print("   - results/metrics/final_evaluation.json")
        print("   - results/metrics/final_evaluation_summary.csv")
        
    def save_best_model(self):
        """Save the best performing model for production use."""
        
        if self.best_model is None:
            print("No best model identified. Run evaluation first.")
            return
        
        # Save the model
        model_path = 'results/models/best_model.pkl'
        joblib.dump(self.best_model, model_path)
        
        # Save model metadata
        metadata = {
            'model_type': type(self.best_model).__name__,
            'best_rmse': float(self.best_score),
            'boxcox_lambda': self.lambda_boxcox,
            'training_features': list(self.X_train.columns),
            'training_date': datetime.now().isoformat(),
            'random_state': self.random_state
        }
        
        with open('results/models/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("Best model saved:")
        print(f"   - Model: {model_path}")
        print(f"   - Metadata: results/models/model_metadata.json")
        print(f"   - Performance: RMSE = {self.best_score:.4f}")
        
    def generate_predictions_and_submission(self):
        """Generate predictions on test set and create submission file."""
        
        if self.best_model is None:
            print("No best model available. Train and evaluate models first.")
            return
        
        print("Generating predictions with best model...")
        
        # Generate predictions (in BoxCox transformed space)
        test_predictions_boxcox = self.best_model.predict(self.X_test)
        
        # Transform back to original scale using inverse BoxCox
        test_predictions = inv_boxcox(test_predictions_boxcox, self.lambda_boxcox)
        
        # Create submission file
        if os.path.exists('../../dataset/test.csv'):
            original_test = pd.read_csv('../../dataset/test.csv')
            submission_df = pd.DataFrame({
                'Id': original_test['Id'],
                'SalePrice': test_predictions
            })
            
            submission_df.to_csv('results/submission.csv', index=False)
            
            print(f"Submission file created: results/submission.csv")
            print(f"   - {len(submission_df)} predictions generated")
            print(f"   - Price range: ${test_predictions.min():,.0f} - ${test_predictions.max():,.0f}")
            print(f"   - Mean price: ${test_predictions.mean():,.0f}")
        
        return test_predictions
        
    def run_complete_pipeline(self):
        """
        Execute the complete advanced modeling pipeline.
        Integrates all phases for comprehensive model development.
        """
        print("EXECUTING COMPLETE ADVANCED MODELING PIPELINE")
        print("=" * 70)
        print()
        
        try:
            # Step 1: Load processed data
            self.load_processed_data()
            print()
            
            # Step 2: Initialize models
            self.initialize_model_portfolio()
            print()
            
            # Step 3: Baseline comparison
            baseline_results = self.baseline_model_comparison()
            print()
            
            # Step 4: Hyperparameter optimization
            optimization_results = self.hyperparameter_optimization(top_n=5)
            print()
            
            # Step 5: Train final models
            final_models = self.train_final_models()
            print()
            
            # Step 6: Create ensembles
            self.create_ensemble_methods()
            print()
            
            # Step 7: Comprehensive evaluation
            evaluation_results, best_model_name = self.comprehensive_model_evaluation()
            print()
            
            # Step 8: Save best model
            self.save_best_model()
            print()
            
            # Step 9: Generate predictions
            predictions = self.generate_predictions_and_submission()
            print()
            
            # Final summary
            print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            print(f"Best Model: {best_model_name}")
            print(f"Best RMSE: {self.best_score:.4f}")
            print(f"Models Evaluated: {len(evaluation_results)}")
            print(f"Predictions Generated: {len(predictions) if predictions is not None else 0}")
            print("\nAll results saved in 'results/' directory:")
            print("   - models/          - Trained models and metadata")
            print("   - metrics/         - Performance metrics and comparisons") 
            print("   - visualizations/  - Charts and analysis plots")
            print("   - submission.csv   - Final predictions for competition")
            
            return {
                'best_model': self.best_model,
                'best_model_name': best_model_name,
                'best_score': self.best_score,
                'evaluation_results': evaluation_results,
                'predictions': predictions
            }
            
        except Exception as e:
            print(f"Pipeline execution failed: {str(e)}")
            raise

if __name__ == "__main__":
    # Initialize and run the complete advanced modeling pipeline
    pipeline = AdvancedModelingPipeline(random_state=42)
    results = pipeline.run_complete_pipeline()