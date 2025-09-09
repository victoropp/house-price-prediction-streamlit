"""
PHASE 6: SCALING AND FINAL DATASET PREPARATION PIPELINE
======================================================================
State-of-the-art scaling techniques and final dataset preparation using 
advanced methods for optimal machine learning performance.

Key Features:
- Missing value imputation for encoded features
- Robust scaling method selection based on distribution analysis
- Feature stability validation and outlier detection
- Cross-validation stability testing
- Final dataset quality assurance
- Comprehensive visualization and reporting

Author: Advanced ML Pipeline
Date: 2025
======================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, 
    QuantileTransformer, PowerTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from scipy import stats
from scipy.stats import boxcox
import warnings
import json
import os
from pathlib import Path

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

class ScalingFinalPreparationPipeline:
    """
    State-of-the-art scaling and final preparation pipeline with intelligent
    missing value handling, method selection and comprehensive validation.
    """
    
    def __init__(self):
        """Initialize the scaling pipeline with comprehensive tracking."""
        self.train_data = None
        self.test_data = None
        self.target = None
        self.target_original = None
        self.lambda_param = None
        self.numerical_features = None
        self.encoded_features = None
        self.binary_features = None
        
        self.scaling_methods = {
            'StandardScaler': StandardScaler(),
            'RobustScaler': RobustScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'QuantileTransformer': QuantileTransformer(output_distribution='normal', random_state=42),
            'PowerTransformer': PowerTransformer(method='yeo-johnson', standardize=True)
        }
        
        self.selected_scaler = None
        self.scaler_performance = {}
        self.imputer_numerical = None
        self.imputer_encoded = None
        self.final_stats = {}
        self.validation_results = {}
        
    def load_encoded_data(self):
        """Load the properly encoded data from Phase 5."""
        print("Loading cleaned encoded data from Phase 5...")
        
        # Load training and test data
        self.train_data = pd.read_csv('../phase5_encoding/encoded_train_phase5.csv')
        self.test_data = pd.read_csv('../phase5_encoding/encoded_test_phase5.csv')
        
        # Extract and transform target variable (apply optimal transformation from Phase 1)
        self.target_original = self.train_data['SalePrice'].copy()
        # Apply BoxCox transformation (optimal from Phase 1 analysis)
        self.target, self.lambda_param = boxcox(self.target_original + 1)
        
        self.train_data = self.train_data.drop('SalePrice', axis=1)
        
        # Categorize features by type for appropriate processing
        self._categorize_features()
        
        print(f"Training data: {self.train_data.shape[0]} samples, {self.train_data.shape[1]} features")
        print(f"Test data: {self.test_data.shape[0]} samples, {self.test_data.shape[1]} features")
        print(f"Numerical features: {len(self.numerical_features)}")
        print(f"Encoded features: {len(self.encoded_features)}")  
        print(f"Binary features: {len(self.binary_features)}")
        print(f"Target variable: BoxCox transformed (lambda: {self.lambda_param:.3f})")
        print(f"Original target skewness: {stats.skew(self.target_original):.3f}")
        print(f"Transformed target skewness: {stats.skew(self.target):.3f}")
        
        return self
    
    def _categorize_features(self):
        """Categorize features into numerical, encoded, and binary for proper processing."""
        self.numerical_features = []
        self.encoded_features = []
        self.binary_features = []
        
        for col in self.train_data.columns:
            if col == 'Id':
                continue
            elif col.endswith('_encoded') or col.endswith('_transformed'):
                # These are encoded/transformed features that may need imputation
                self.encoded_features.append(col)
            elif (col.startswith('MSZoning_') or col.startswith('LotShape_') or 
                  col.startswith('LandContour_') or col.startswith('LotConfig_') or
                  col.startswith('LandSlope_') or col.startswith('BldgType_') or
                  col.startswith('HouseStyle_') or col.startswith('RoofStyle_') or
                  col.startswith('RoofMatl_') or col.startswith('MasVnrType_') or
                  col.startswith('Foundation_') or col.startswith('BsmtExposure_') or
                  col.startswith('Heating_') or col.startswith('Electrical_') or
                  col.startswith('Functional_') or col.startswith('GarageType_') or
                  col.startswith('PavedDrive_') or col.startswith('Fence_') or
                  col.startswith('MiscFeature_') or col.startswith('SaleType_') or
                  col.startswith('HouseAgeGroup_') or col.startswith('NeighborhoodTier_')):
                # These are binary one-hot encoded features
                self.binary_features.append(col)
            else:
                # These are continuous numerical features that need scaling
                if (len(self.train_data[col].unique()) > 10 and 
                    self.train_data[col].dtype in ['int64', 'float64']):
                    self.numerical_features.append(col)
                else:
                    # Low cardinality numerical features (treat as encoded)
                    self.encoded_features.append(col)
    
    def handle_missing_values(self):
        """Handle missing values in encoded and numerical features."""
        print("\n" + "="*60)
        print("HANDLING MISSING VALUES")
        print("="*60)
        
        # Check missing values
        train_missing = self.train_data.isnull().sum()
        test_missing = self.test_data.isnull().sum()
        
        missing_features = train_missing[train_missing > 0].sort_values(ascending=False)
        print(f"Features with missing values in training: {len(missing_features)}")
        
        if len(missing_features) > 0:
            print("Top missing value features:")
            for feature, count in missing_features.head(10).items():
                pct = (count / len(self.train_data)) * 100
                print(f"  {feature}: {count} ({pct:.1f}%)")
        
        # Impute encoded features (ordinal encoded can have meaningful missing values)
        encoded_missing = [col for col in self.encoded_features if train_missing[col] > 0]
        if encoded_missing:
            print(f"\nImputing {len(encoded_missing)} encoded features with missing values...")
            
            # Use median for ordinal encoded features, mode for others
            for col in encoded_missing:
                if col.endswith('_encoded') and not col.endswith('WasMissing_encoded'):
                    # Use median for ordinal encoded quality/condition features
                    median_value = self.train_data[col].median()
                    self.train_data[col] = self.train_data[col].fillna(median_value)
                    self.test_data[col] = self.test_data[col].fillna(median_value)
                    print(f"  {col}: filled with median ({median_value})")
                else:
                    # Use mode for other encoded features
                    mode_value = self.train_data[col].mode().iloc[0] if len(self.train_data[col].mode()) > 0 else 0
                    self.train_data[col] = self.train_data[col].fillna(mode_value)
                    self.test_data[col] = self.test_data[col].fillna(mode_value)
                    print(f"  {col}: filled with mode ({mode_value})")
        
        # Impute numerical features if any missing
        numerical_missing = [col for col in self.numerical_features if train_missing[col] > 0]
        if numerical_missing:
            print(f"\nImputing {len(numerical_missing)} numerical features...")
            self.imputer_numerical = KNNImputer(n_neighbors=5)
            
            # Fit on training data and transform both
            train_numerical = self.train_data[self.numerical_features]
            test_numerical = self.test_data[self.numerical_features]
            
            train_imputed = self.imputer_numerical.fit_transform(train_numerical)
            test_imputed = self.imputer_numerical.transform(test_numerical)
            
            # Update dataframes
            self.train_data[self.numerical_features] = train_imputed
            self.test_data[self.numerical_features] = test_imputed
            
            print(f"  Applied KNN imputation to {len(self.numerical_features)} numerical features")
        
        # Final missing value check
        final_missing_train = self.train_data.isnull().sum().sum()
        final_missing_test = self.test_data.isnull().sum().sum()
        
        print(f"\nFinal missing values - Training: {final_missing_train}, Test: {final_missing_test}")
        
        if final_missing_train == 0 and final_missing_test == 0:
            print("SUCCESS: All missing values handled!")
        else:
            print("WARNING: Some missing values remain")
            
        return self
    
    def analyze_feature_distributions(self):
        """Analyze distributions of numerical features to guide scaling method selection."""
        print("\n" + "="*60)
        print("ANALYZING FEATURE DISTRIBUTIONS FOR SCALING")
        print("="*60)
        
        self.distribution_analysis = {}
        
        for feature in self.numerical_features:
            data = self.train_data[feature]
            
            # Calculate distribution statistics
            skewness = stats.skew(data)
            kurtosis = stats.kurtosis(data)
            outlier_pct = self._calculate_outlier_percentage(data)
            
            # Normality test
            _, p_value = stats.normaltest(data)
            is_normal = p_value > 0.05
            
            self.distribution_analysis[feature] = {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'outlier_percentage': outlier_pct,
                'is_normal': is_normal,
                'p_value': p_value,
                'min': data.min(),
                'max': data.max(),
                'std': data.std(),
                'range': data.max() - data.min()
            }
        
        # Summarize distribution characteristics
        normal_features = [f for f, stats_dict in self.distribution_analysis.items() if stats_dict['is_normal']]
        skewed_features = [f for f, stats_dict in self.distribution_analysis.items() if abs(stats_dict['skewness']) > 1]
        high_outlier_features = [f for f, stats_dict in self.distribution_analysis.items() if stats_dict['outlier_percentage'] > 5]
        
        print(f"Normal distribution features: {len(normal_features)}")
        print(f"Highly skewed features (|skew| > 1): {len(skewed_features)}")
        print(f"High outlier features (>5% outliers): {len(high_outlier_features)}")
        
        return self
    
    def _calculate_outlier_percentage(self, data):
        """Calculate percentage of outliers using IQR method."""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        return (len(outliers) / len(data)) * 100
    
    def evaluate_scaling_methods(self):
        """Evaluate different scaling methods using cross-validation."""
        print("\n" + "="*60)
        print("EVALUATING SCALING METHODS")
        print("="*60)
        
        # Prepare feature sets
        X_numerical = self.train_data[self.numerical_features].copy()
        X_encoded = self.train_data[self.encoded_features].copy()
        X_binary = self.train_data[self.binary_features].copy()
        
        evaluation_results = {}
        
        for method_name, scaler in self.scaling_methods.items():
            print(f"Evaluating {method_name}...")
            
            try:
                # Scale only numerical features
                X_numerical_scaled = pd.DataFrame(
                    scaler.fit_transform(X_numerical),
                    columns=X_numerical.columns,
                    index=X_numerical.index
                )
                
                # Combine all feature types
                X_combined = pd.concat([X_numerical_scaled, X_encoded, X_binary], axis=1)
                
                # Cross-validation with Ridge regression
                ridge = Ridge(alpha=1.0, random_state=42)
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                cv_scores = cross_val_score(
                    ridge, X_combined, self.target,
                    cv=cv, scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                
                # Calculate metrics
                mean_cv_score = -cv_scores.mean()
                std_cv_score = cv_scores.std()
                
                # Feature stability (variance consistency)
                feature_variance = X_numerical_scaled.var().mean()
                scale_consistency = self._calculate_scale_consistency(X_numerical_scaled)
                
                evaluation_results[method_name] = {
                    'cv_rmse_mean': np.sqrt(mean_cv_score),
                    'cv_rmse_std': std_cv_score,
                    'feature_variance': feature_variance,
                    'scale_consistency': scale_consistency,
                    'scaler': scaler
                }
                
                print(f"  CV RMSE: {np.sqrt(mean_cv_score):.4f} (+/- {std_cv_score:.4f})")
                print(f"  Feature variance: {feature_variance:.4f}")
                print(f"  Scale consistency: {scale_consistency:.4f}")
                
            except Exception as e:
                print(f"  ERROR: {str(e)}")
                evaluation_results[method_name] = {
                    'cv_rmse_mean': np.inf,
                    'cv_rmse_std': np.inf,
                    'feature_variance': np.inf,
                    'scale_consistency': 0,
                    'scaler': scaler,
                    'error': str(e)
                }
        
        # Select best scaling method
        valid_methods = {k: v for k, v in evaluation_results.items() if v['cv_rmse_mean'] != np.inf}
        
        if valid_methods:
            # Rank by CV performance with stability consideration
            method_scores = {}
            for method, results in valid_methods.items():
                # Lower RMSE is better, higher consistency is better
                score = results['cv_rmse_mean'] - (results['scale_consistency'] * 0.1)
                method_scores[method] = score
            
            best_method = min(method_scores, key=method_scores.get)
            self.selected_scaler = evaluation_results[best_method]['scaler']
            self.best_method_name = best_method
            
            print(f"\nBEST SCALING METHOD: {best_method}")
            print(f"CV RMSE: {evaluation_results[best_method]['cv_rmse_mean']:.4f}")
            print(f"Scale Consistency: {evaluation_results[best_method]['scale_consistency']:.4f}")
        else:
            print("\nWARNING: All scaling methods failed, using RobustScaler as fallback")
            self.selected_scaler = RobustScaler()
            best_method = 'RobustScaler'
            self.best_method_name = best_method
        
        self.scaler_performance = evaluation_results
        
        return self
    
    def _calculate_scale_consistency(self, scaled_data):
        """Calculate how consistently features are scaled."""
        feature_ranges = scaled_data.max() - scaled_data.min()
        range_consistency = 1 / (1 + feature_ranges.std())
        
        # Check centering for standard/robust scaling
        mean_consistency = 1 / (1 + np.abs(scaled_data.mean()).mean())
        
        return (range_consistency + mean_consistency) / 2
    
    def apply_final_scaling(self):
        """Apply the selected scaling method to numerical features only."""
        print("\n" + "="*60)
        print("APPLYING FINAL SCALING")
        print("="*60)
        
        # Separate feature types
        X_train_numerical = self.train_data[self.numerical_features].copy()
        X_train_encoded = self.train_data[self.encoded_features].copy()
        X_train_binary = self.train_data[self.binary_features].copy()
        
        X_test_numerical = self.test_data[self.numerical_features].copy()
        X_test_encoded = self.test_data[self.encoded_features].copy()
        X_test_binary = self.test_data[self.binary_features].copy()
        
        # Scale only numerical features
        print(f"Scaling {len(self.numerical_features)} numerical features with {self.best_method_name}...")
        X_train_numerical_scaled = pd.DataFrame(
            self.selected_scaler.fit_transform(X_train_numerical),
            columns=X_train_numerical.columns,
            index=X_train_numerical.index
        )
        
        X_test_numerical_scaled = pd.DataFrame(
            self.selected_scaler.transform(X_test_numerical),
            columns=X_test_numerical.columns,
            index=X_test_numerical.index
        )
        
        # Combine all feature types (scaled numerical + encoded + binary)
        self.final_train_data = pd.concat([
            X_train_numerical_scaled, 
            X_train_encoded, 
            X_train_binary
        ], axis=1)
        
        self.final_test_data = pd.concat([
            X_test_numerical_scaled, 
            X_test_encoded, 
            X_test_binary
        ], axis=1)
        
        # Add target back to training data
        self.final_train_data['SalePrice_transformed'] = self.target
        
        print(f"Final training data shape: {self.final_train_data.shape}")
        print(f"Final test data shape: {self.final_test_data.shape}")
        print(f"  - Scaled numerical features: {len(self.numerical_features)}")
        print(f"  - Encoded features: {len(self.encoded_features)}")
        print(f"  - Binary features: {len(self.binary_features)}")
        
        return self
    
    def validate_final_dataset(self):
        """Comprehensive validation of the final prepared dataset."""
        print("\n" + "="*60)
        print("VALIDATING FINAL DATASET")
        print("="*60)
        
        # Data quality checks
        train_missing = self.final_train_data.isnull().sum().sum()
        test_missing = self.final_test_data.isnull().sum().sum()
        
        # Feature consistency
        train_features = set(self.final_train_data.columns) - {'SalePrice_transformed'}
        test_features = set(self.final_test_data.columns)
        feature_mismatch = train_features.symmetric_difference(test_features)
        
        # Statistical properties
        scaled_numerical_stats = {}
        for feature in self.numerical_features:
            if feature in self.final_train_data.columns:
                data = self.final_train_data[feature]
                scaled_numerical_stats[feature] = {
                    'mean': data.mean(),
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max()
                }
        
        # Target validation
        target_stats = {
            'mean': float(self.target.mean()),
            'std': float(self.target.std()),
            'min': float(self.target.min()),
            'max': float(self.target.max()),
            'skewness': float(stats.skew(self.target)),
            'kurtosis': float(stats.kurtosis(self.target))
        }
        
        # Validation results
        self.validation_results = {
            'data_quality': {
                'train_missing_values': int(train_missing),
                'test_missing_values': int(test_missing),
                'train_shape': self.final_train_data.shape,
                'test_shape': self.final_test_data.shape
            },
            'feature_consistency': {
                'feature_mismatch_count': len(feature_mismatch),
                'mismatched_features': list(feature_mismatch)
            },
            'scaled_features_stats': scaled_numerical_stats,
            'target_stats': target_stats
        }
        
        # Report validation results
        print(f"Data Quality:")
        print(f"  Training missing values: {train_missing}")
        print(f"  Test missing values: {test_missing}")
        print(f"  Training shape: {self.final_train_data.shape}")
        print(f"  Test shape: {self.final_test_data.shape}")
        
        print(f"\nFeature Consistency:")
        print(f"  Feature mismatch count: {len(feature_mismatch)}")
        if feature_mismatch:
            print(f"  Mismatched features: {list(feature_mismatch)[:5]}...")
        
        print(f"\nTarget Variable:")
        print(f"  Mean: {target_stats['mean']:.3f}")
        print(f"  Std: {target_stats['std']:.3f}")
        print(f"  Skewness: {target_stats['skewness']:.3f}")
        
        if scaled_numerical_stats:
            mean_of_means = np.mean([s['mean'] for s in scaled_numerical_stats.values()])
            mean_of_stds = np.mean([s['std'] for s in scaled_numerical_stats.values()])
            print(f"\nScaled Numerical Features:")
            print(f"  Average feature mean: {mean_of_means:.3f}")
            print(f"  Average feature std: {mean_of_stds:.3f}")
        
        # Validation status
        validation_passed = (
            train_missing == 0 and
            test_missing == 0 and
            len(feature_mismatch) == 0
        )
        
        self.validation_results['validation_passed'] = validation_passed
        
        if validation_passed:
            print("\nSUCCESS: All validation checks passed!")
        else:
            print("\nWARNING: Some validation checks failed!")
        
        return self
    
    def generate_final_visualization(self):
        """Generate comprehensive state-of-the-art visualization."""
        print("\nGenerating final dataset visualization...")
        
        fig = plt.figure(figsize=(24, 18))
        
        # Color palette
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83']
        
        # 1. Scaling method performance comparison
        ax1 = plt.subplot(3, 5, 1)
        valid_methods = [k for k, v in self.scaler_performance.items() if v['cv_rmse_mean'] != np.inf]
        if valid_methods:
            rmse_scores = [self.scaler_performance[m]['cv_rmse_mean'] for m in valid_methods]
            bar_colors = [colors[0] if m == self.best_method_name else colors[3] for m in valid_methods]
            
            bars = plt.bar(range(len(valid_methods)), rmse_scores, color=bar_colors, alpha=0.8)
            plt.xticks(range(len(valid_methods)), valid_methods, rotation=45, ha='right')
            plt.title('Scaling Method Performance\n(Lower RMSE Better)', fontweight='bold', fontsize=12)
            plt.ylabel('CV RMSE')
            plt.grid(axis='y', alpha=0.3)
            
            # Highlight best method
            best_idx = valid_methods.index(self.best_method_name)
            bars[best_idx].set_edgecolor('black')
            bars[best_idx].set_linewidth(2)
        
        # 2. Target distribution (original vs transformed)
        ax2 = plt.subplot(3, 5, 2)
        plt.hist(self.target_original, bins=50, alpha=0.7, color=colors[1], 
                label=f'Original (skew: {stats.skew(self.target_original):.2f})')
        plt.hist(self.target, bins=50, alpha=0.7, color=colors[0], 
                label=f'Transformed (skew: {stats.skew(self.target):.2f})')
        plt.title('Target Distribution\nOriginal vs Transformed', fontweight='bold', fontsize=12)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        
        # 3. Scaled features distribution sample
        ax3 = plt.subplot(3, 5, 3)
        sample_features = self.numerical_features[:6]
        for i, feature in enumerate(sample_features):
            if feature in self.final_train_data.columns:
                data = self.final_train_data[feature]
                plt.hist(data, bins=30, alpha=0.6, color=colors[i % len(colors)], 
                        label=feature[:12] + '...' if len(feature) > 12 else feature, density=True)
        plt.title('Scaled Numerical Features\nDistribution Sample', fontweight='bold', fontsize=12)
        plt.xlabel('Scaled Values')
        plt.ylabel('Density')
        plt.legend(fontsize=8, loc='upper right')
        plt.grid(alpha=0.3)
        
        # 4. Feature correlation with target (top 15)
        ax4 = plt.subplot(3, 5, 4)
        correlations = {}
        target_series = pd.Series(self.target, index=self.final_train_data.index)
        for col in self.final_train_data.columns:
            if col != 'SalePrice_transformed' and pd.api.types.is_numeric_dtype(self.final_train_data[col]):
                corr = self.final_train_data[col].corr(target_series)
                if not np.isnan(corr):
                    correlations[col] = abs(corr)
        
        top_correlations = dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:15])
        
        if top_correlations:
            y_pos = np.arange(len(top_correlations))
            plt.barh(y_pos, list(top_correlations.values()), color=colors[2], alpha=0.8)
            plt.yticks(y_pos, [k[:20] + '...' if len(k) > 20 else k for k in top_correlations.keys()])
            plt.title('Top 15 Feature Correlations\nwith Target', fontweight='bold', fontsize=12)
            plt.xlabel('Absolute Correlation')
            plt.grid(axis='x', alpha=0.3)
        
        # 5. Feature type distribution
        ax5 = plt.subplot(3, 5, 5)
        feature_counts = {
            'Scaled Numerical': len(self.numerical_features),
            'Encoded': len(self.encoded_features),
            'Binary (One-Hot)': len(self.binary_features)
        }
        
        wedges, texts, autotexts = plt.pie(feature_counts.values(), 
                                          labels=feature_counts.keys(), 
                                          autopct='%1.1f%%',
                                          colors=colors[:3],
                                          startangle=90)
        plt.title('Feature Type Distribution', fontweight='bold', fontsize=12)
        
        # 6. Data quality metrics
        ax6 = plt.subplot(3, 5, 6)
        metrics = [
            ('Samples', self.final_train_data.shape[0]),
            ('Features', self.final_train_data.shape[1] - 1),
            ('Numerical', len(self.numerical_features)),
            ('Missing', self.validation_results['data_quality']['train_missing_values'])
        ]
        
        metric_names, metric_values = zip(*metrics)
        bar_colors_6 = [colors[0], colors[1], colors[2], colors[3]]
        bars = plt.bar(metric_names, metric_values, color=bar_colors_6, alpha=0.8)
        plt.title('Dataset Quality Metrics', fontweight='bold', fontsize=12)
        plt.ylabel('Count')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(metric_values)*0.01,
                    f'{value:,}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        plt.grid(axis='y', alpha=0.3)
        
        # 7. Outlier analysis for scaled features
        ax7 = plt.subplot(3, 5, 7)
        if len(self.numerical_features) > 0:
            outlier_counts = []
            feature_sample = self.numerical_features[:8]
            
            for feature in feature_sample:
                if feature in self.final_train_data.columns:
                    data = self.final_train_data[feature]
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
                    outlier_counts.append(len(outliers))
            
            if outlier_counts:
                plt.bar(range(len(feature_sample)), outlier_counts, color=colors[4], alpha=0.8)
                plt.xticks(range(len(feature_sample)), 
                          [f[:8] + '..' if len(f) > 10 else f for f in feature_sample], 
                          rotation=45, ha='right')
                plt.title('Outlier Counts\n(Scaled Features Sample)', fontweight='bold', fontsize=12)
                plt.ylabel('Number of Outliers')
                plt.grid(axis='y', alpha=0.3)
        
        # 8. Feature variance after scaling
        ax8 = plt.subplot(3, 5, 8)
        if len(self.numerical_features) > 0:
            variances = []
            feature_names = []
            for feature in self.numerical_features[:12]:
                if feature in self.final_train_data.columns:
                    var = self.final_train_data[feature].var()
                    variances.append(var)
                    feature_names.append(feature[:10])
            
            if variances:
                plt.bar(range(len(variances)), variances, color=colors[1], alpha=0.8)
                plt.xticks(range(len(variances)), feature_names, rotation=45, ha='right')
                plt.title('Feature Variances\n(After Scaling)', fontweight='bold', fontsize=12)
                plt.ylabel('Variance')
                plt.grid(axis='y', alpha=0.3)
        
        # 9. Pipeline evolution
        ax9 = plt.subplot(3, 5, 9)
        evolution_data = [81, 104, 131, 139, self.final_train_data.shape[1] - 1]
        phase_names = ['Original', 'Phase 3', 'Phase 4', 'Phase 5', 'Phase 6']
        
        plt.plot(phase_names, evolution_data, marker='o', linewidth=3, 
                markersize=8, color=colors[0], markerfacecolor=colors[1])
        plt.fill_between(phase_names, evolution_data, alpha=0.3, color=colors[0])
        plt.title('Feature Count Evolution\nThrough Pipeline', fontweight='bold', fontsize=12)
        plt.ylabel('Number of Features')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # 10. Performance metrics summary
        ax10 = plt.subplot(3, 5, 10)
        if hasattr(self, 'scaler_performance') and self.best_method_name in self.scaler_performance:
            best_performance = self.scaler_performance[self.best_method_name]
            metrics_dict = {
                'CV RMSE': best_performance['cv_rmse_mean'],
                'Scale Consistency': best_performance['scale_consistency'],
                'Feature Variance': min(best_performance['feature_variance'], 2.0),  # Cap for visualization
                'Target Skewness': abs(stats.skew(self.target))
            }
            
            metric_names = list(metrics_dict.keys())
            metric_values = list(metrics_dict.values())
            
            plt.bar(metric_names, metric_values, color=colors[:4], alpha=0.8)
            plt.title('Final Performance Metrics', fontweight='bold', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Value')
            plt.grid(axis='y', alpha=0.3)
        
        # 11. Missing values check
        ax11 = plt.subplot(3, 5, 11)
        missing_train = self.validation_results['data_quality']['train_missing_values']
        missing_test = self.validation_results['data_quality']['test_missing_values']
        
        if missing_train == 0 and missing_test == 0:
            plt.text(0.5, 0.5, 'NO MISSING\nVALUES\n\n✓ SUCCESS!', 
                    ha='center', va='center', transform=ax11.transAxes,
                    fontsize=16, fontweight='bold', color=colors[0],
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        else:
            plt.bar(['Train', 'Test'], [missing_train, missing_test], 
                   color=[colors[3], colors[4]], alpha=0.8)
            plt.ylabel('Missing Values')
        
        plt.title('Missing Values Check', fontweight='bold', fontsize=12)
        ax11.set_xticks([])
        ax11.set_yticks([])
        
        # 12. Dataset readiness summary
        ax12 = plt.subplot(3, 5, 12)
        ax12.axis('off')
        
        summary_text = f"""FINAL DATASET SUMMARY

Training Samples: {self.final_train_data.shape[0]:,}
Test Samples: {self.final_test_data.shape[0]:,}
Total Features: {self.final_train_data.shape[1] - 1:,}

Feature Types:
• Scaled Numerical: {len(self.numerical_features)}
• Encoded: {len(self.encoded_features)}
• Binary: {len(self.binary_features)}

Scaling Method: {self.best_method_name}
CV RMSE: {self.scaler_performance[self.best_method_name]['cv_rmse_mean']:.4f}

Target: BoxCox Transformed
Lambda: {self.lambda_param:.3f}
Skewness: {stats.skew(self.target):.3f}

Data Quality: {'PASS' if self.validation_results['validation_passed'] else 'ISSUES'}
Missing Values: {missing_train}

✓ READY FOR MODELING!"""
        
        plt.text(0.05, 0.95, summary_text, transform=ax12.transAxes, 
                fontsize=11, fontweight='bold', va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.8', facecolor=colors[0], alpha=0.1))
        
        # 13-15. Additional visualizations for completeness
        # 13. Distribution comparison (scaled vs unscaled sample)
        ax13 = plt.subplot(3, 5, 13)
        if len(self.numerical_features) > 0:
            sample_feature = self.numerical_features[0]
            original_data = self.train_data[sample_feature]
            scaled_data = self.final_train_data[sample_feature]
            
            plt.hist(original_data, bins=30, alpha=0.6, color=colors[3], 
                    label='Original', density=True)
            plt.hist(scaled_data, bins=30, alpha=0.6, color=colors[0], 
                    label='Scaled', density=True)
            plt.title(f'Scaling Effect Example\n{sample_feature[:15]}...', fontweight='bold', fontsize=12)
            plt.xlabel('Value')
            plt.ylabel('Density')
            plt.legend(fontsize=10)
            plt.grid(alpha=0.3)
        
        # 14. Cross-validation stability
        ax14 = plt.subplot(3, 5, 14)
        if hasattr(self, 'scaler_performance'):
            methods = []
            cv_means = []
            cv_stds = []
            
            for method, results in self.scaler_performance.items():
                if results['cv_rmse_mean'] != np.inf:
                    methods.append(method)
                    cv_means.append(results['cv_rmse_mean'])
                    cv_stds.append(results.get('cv_rmse_std', 0))
            
            if methods:
                x_pos = np.arange(len(methods))
                plt.errorbar(x_pos, cv_means, yerr=cv_stds, fmt='o', 
                           capsize=5, capthick=2, markersize=8, 
                           color=colors[0], ecolor=colors[3])
                plt.xticks(x_pos, methods, rotation=45, ha='right')
                plt.title('Cross-Validation Stability\n(Mean ± Std)', fontweight='bold', fontsize=12)
                plt.ylabel('RMSE')
                plt.grid(True, alpha=0.3)
        
        # 15. Feature importance preview (correlation-based)
        ax15 = plt.subplot(3, 5, 15)
        if len(correlations) > 0:
            # Top 10 most correlated features
            top_10_corr = dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:10])
            
            feature_names = [k[:15] + '...' if len(k) > 15 else k for k in top_10_corr.keys()]
            corr_values = list(top_10_corr.values())
            
            y_pos = np.arange(len(feature_names))
            plt.barh(y_pos, corr_values, color=colors[2], alpha=0.8)
            plt.yticks(y_pos, feature_names)
            plt.title('Top 10 Feature Importance\n(Correlation-based)', fontweight='bold', fontsize=12)
            plt.xlabel('Absolute Correlation')
            plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout(pad=3.0)
        plt.savefig('scaling_final_preparation_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return self
    
    def save_final_datasets(self):
        """Save final prepared datasets and comprehensive configuration."""
        print("\nSaving final datasets and configuration...")
        
        # Save final datasets
        self.final_train_data.to_csv('final_train_prepared.csv', index=False)
        self.final_test_data.to_csv('final_test_prepared.csv', index=False)
        
        # Type conversion for JSON serialization
        def convert_types(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            return obj
        
        def recursive_convert(item):
            if isinstance(item, dict):
                return {convert_types(key): recursive_convert(value) for key, value in item.items()}
            elif isinstance(item, list):
                return [recursive_convert(element) for element in item]
            else:
                return convert_types(item)
        
        # Create comprehensive final configuration
        final_config = {
            'preprocessing_summary': {
                'pipeline_version': '6.0',
                'total_phases': 6,
                'final_feature_count': self.final_train_data.shape[1] - 1,
                'original_feature_count': 81,
                'feature_expansion_ratio': ((self.final_train_data.shape[1] - 1) / 81)
            },
            'feature_categorization': {
                'numerical_features': self.numerical_features,
                'encoded_features': self.encoded_features,
                'binary_features': self.binary_features,
                'feature_counts': {
                    'numerical': len(self.numerical_features),
                    'encoded': len(self.encoded_features),
                    'binary': len(self.binary_features)
                }
            },
            'scaling_configuration': {
                'method': self.best_method_name,
                'scaler_performance': recursive_convert(self.scaler_performance),
                'features_scaled': self.numerical_features
            },
            'target_transformation': {
                'method': 'BoxCox',
                'lambda_parameter': float(self.lambda_param),
                'original_skewness': float(stats.skew(self.target_original)),
                'transformed_skewness': float(stats.skew(self.target))
            },
            'validation_results': recursive_convert(self.validation_results),
            'dataset_info': {
                'train_shape': self.final_train_data.shape,
                'test_shape': self.final_test_data.shape,
                'missing_values': {
                    'train': int(self.final_train_data.isnull().sum().sum()),
                    'test': int(self.final_test_data.isnull().sum().sum())
                }
            }
        }
        
        with open('scaling_final_config.json', 'w') as f:
            json.dump(final_config, f, indent=4)
        
        print("SUCCESS Final training data saved to final_train_prepared.csv")
        print("SUCCESS Final test data saved to final_test_prepared.csv")
        print("SUCCESS Final configuration saved to scaling_final_config.json")
        
        return self
    
    def run_complete_pipeline(self):
        """Execute the complete state-of-the-art scaling and preparation pipeline."""
        print("PHASE 6: SCALING AND FINAL PREPARATION PIPELINE")
        print("="*70)
        
        return (self
                .load_encoded_data()
                .handle_missing_values()
                .analyze_feature_distributions()
                .evaluate_scaling_methods()
                .apply_final_scaling()
                .validate_final_dataset()
                .generate_final_visualization()
                .save_final_datasets())

if __name__ == "__main__":
    pipeline = ScalingFinalPreparationPipeline()
    pipeline.run_complete_pipeline()
    
    print("\n" + "="*70)
    print("SUCCESS PHASE 6 COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"RESULT Best scaling method: {pipeline.best_method_name}")
    print(f"RESULT Final training samples: {pipeline.final_train_data.shape[0]:,}")
    print(f"RESULT Final features: {pipeline.final_train_data.shape[1] - 1:,}")
    print(f"RESULT Feature types: {len(pipeline.numerical_features)} numerical, {len(pipeline.encoded_features)} encoded, {len(pipeline.binary_features)} binary")
    print(f"RESULT CV RMSE: {pipeline.scaler_performance[pipeline.best_method_name]['cv_rmse_mean']:.4f}")
    print(f"RESULT Target transformation: BoxCox (λ={pipeline.lambda_param:.3f})")
    print("TARGET READY FOR MACHINE LEARNING MODELING!")
    print("="*70)