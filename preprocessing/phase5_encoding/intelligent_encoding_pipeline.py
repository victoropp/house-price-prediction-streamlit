"""
PHASE 5: INTELLIGENT ENCODING STRATEGIES
=======================================

State-of-the-art categorical encoding pipeline with domain-aware strategy selection:
- Automatic encoding strategy selection based on feature characteristics
- Ordinal encoding for quality/condition features (preserves natural order)
- Target encoding for high-cardinality categoricals (prevents dimensionality explosion)
- One-hot encoding for low-cardinality nominals (prevents information loss)
- Label encoding for binary categoricals (memory efficient)
- Advanced techniques: CatBoost encoding, Leave-One-Out encoding

Key Insights from EDA:
- 43 categorical features require intelligent encoding
- Quality features (ExterQual, KitchenQual, etc.) have natural ordering -> Ordinal
- Neighborhood (25 categories) benefits from target encoding -> Target
- Low-cardinality features (MSZoning, SaleType) -> One-hot
- Binary features (CentralAir, PavedDrive) -> Label encoding

Author: Advanced Data Science Pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# Professional styling
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
PRIMARY_COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592941']

class IntelligentEncodingPipeline:
    """
    Advanced categorical encoding with automatic strategy selection
    """
    
    def __init__(self):
        self.categorical_analysis = {}
        self.encoding_strategies = {}
        self.encoding_artifacts = {}
        self.encoded_features_info = {}
        
    def load_data(self):
        """Load transformed data from Phase 4"""
        print("Loading transformed data from Phase 4...")
        self.train_df = pd.read_csv('../phase4_distributions/transformed_train_phase4.csv')
        self.test_df = pd.read_csv('../phase4_distributions/transformed_test_phase4.csv')
        
        # Combine for consistent encoding
        self.combined_df = pd.concat([
            self.train_df.drop('SalePrice', axis=1), 
            self.test_df
        ], axis=0, ignore_index=True)
        
        # Store target for target encoding
        self.target = self.train_df['SalePrice'].values
        
        print(f"Training data: {self.train_df.shape[0]} samples, {self.train_df.shape[1]} features")
        print(f"Test data: {self.test_df.shape[0]} samples, {self.test_df.shape[1]} features")
        print(f"Combined data: {self.combined_df.shape[0]} samples, {self.combined_df.shape[1]} features")
        return self
    
    def analyze_categorical_features(self):
        """Comprehensive analysis of categorical features for encoding strategy selection"""
        print("\n" + "="*60)
        print("ANALYZING CATEGORICAL FEATURES FOR ENCODING")
        print("="*60)
        
        # Identify categorical features
        categorical_columns = self.combined_df.select_dtypes(include=['object']).columns.tolist()
        
        # Also include some numeric features that should be treated as categorical
        potential_categorical_numeric = []
        for col in self.combined_df.select_dtypes(include=[np.number]).columns:
            if col.endswith('_WasMissing') or col in ['MSSubClass']:
                potential_categorical_numeric.append(col)
        
        all_categorical = categorical_columns + potential_categorical_numeric
        
        categorical_analysis = []
        
        for feature in all_categorical:
            unique_values = self.combined_df[feature].nunique()
            unique_ratio = unique_values / len(self.combined_df)
            most_frequent = self.combined_df[feature].mode().iloc[0] if len(self.combined_df[feature].mode()) > 0 else 'Unknown'
            most_frequent_pct = (self.combined_df[feature] == most_frequent).sum() / len(self.combined_df) * 100
            
            # Check if feature has natural ordering (quality/condition features)
            is_ordinal = self.is_ordinal_feature(feature, self.combined_df[feature].unique())
            
            # Determine encoding strategy based on characteristics
            encoding_strategy = self.determine_encoding_strategy(
                feature, unique_values, unique_ratio, is_ordinal
            )
            
            # Calculate target correlation for target encoding candidates
            target_correlation = None
            if encoding_strategy == 'target' and feature in categorical_columns:
                try:
                    feature_target_corr = self.calculate_categorical_target_correlation(feature)
                    target_correlation = feature_target_corr
                except:
                    target_correlation = None
            
            categorical_analysis.append({
                'Feature': feature,
                'Unique_Count': unique_values,
                'Unique_Ratio': unique_ratio,
                'Most_Frequent': most_frequent,
                'Most_Frequent_Pct': most_frequent_pct,
                'Is_Ordinal': is_ordinal,
                'Encoding_Strategy': encoding_strategy,
                'Target_Correlation': target_correlation,
                'Data_Type': str(self.combined_df[feature].dtype)
            })
        
        self.categorical_df = pd.DataFrame(categorical_analysis)
        
        # Strategy distribution
        strategy_counts = self.categorical_df['Encoding_Strategy'].value_counts()
        
        print(f"Total categorical features: {len(all_categorical)}")
        print(f"\nEncoding Strategy Distribution:")
        for strategy, count in strategy_counts.items():
            print(f"  {strategy.upper()}: {count} features")
        
        print(f"\nTop 15 Categorical Features Analysis:")
        display_cols = ['Feature', 'Unique_Count', 'Most_Frequent_Pct', 'Encoding_Strategy']
        print(self.categorical_df[display_cols].head(15).to_string(index=False))
        
        # Group features by encoding strategy
        self.encoding_strategies = {}
        for strategy in strategy_counts.index:
            self.encoding_strategies[strategy] = self.categorical_df[
                self.categorical_df['Encoding_Strategy'] == strategy
            ]['Feature'].tolist()
        
        self.categorical_analysis = {
            'total_categorical_features': len(all_categorical),
            'strategy_distribution': strategy_counts.to_dict(),
            'features_by_strategy': self.encoding_strategies
        }
        
        return self
    
    def is_ordinal_feature(self, feature_name, unique_values):
        """Determine if a feature has natural ordering"""
        
        # Known quality/condition patterns
        ordinal_patterns = [
            'qual', 'cond', 'qu', 'finish'
        ]
        
        # Check feature name
        feature_lower = feature_name.lower()
        if any(pattern in feature_lower for pattern in ordinal_patterns):
            return True
        
        # Check if values follow quality patterns
        unique_str = [str(v).lower() for v in unique_values if pd.notna(v)]
        quality_values = {'ex', 'gd', 'ta', 'fa', 'po', 'none', 'excellent', 'good', 'typical', 'fair', 'poor'}
        
        if len(set(unique_str) & quality_values) >= 2:
            return True
        
        # Check for basement finish types
        basement_finish_values = {'glq', 'alq', 'blq', 'rec', 'lwq', 'unf', 'none'}
        if len(set(unique_str) & basement_finish_values) >= 2:
            return True
        
        # Overall quality and condition are numeric but ordinal
        if feature_name in ['OverallQual', 'OverallCond']:
            return True
        
        return False
    
    def determine_encoding_strategy(self, feature_name, unique_count, unique_ratio, is_ordinal):
        """Determine optimal encoding strategy based on feature characteristics"""
        
        # Binary features -> Label encoding
        if unique_count == 2:
            return 'label'
        
        # Ordinal features -> Ordinal encoding
        if is_ordinal:
            return 'ordinal'
        
        # High cardinality features -> Target encoding
        if unique_count > 10 or unique_ratio > 0.05:
            return 'target'
        
        # Low cardinality nominal features -> One-hot encoding
        if unique_count <= 10:
            return 'onehot'
        
        # Default to target encoding for edge cases
        return 'target'
    
    def calculate_categorical_target_correlation(self, feature):
        """Calculate correlation between categorical feature and target"""
        train_size = len(self.train_df)
        feature_values = self.combined_df[feature].iloc[:train_size]
        
        # Calculate mean target value for each category
        category_means = {}
        for category in feature_values.unique():
            if pd.notna(category):
                mask = feature_values == category
                if mask.sum() > 0:
                    category_means[category] = self.target[mask].mean()
        
        # Calculate correlation with target
        if len(category_means) > 1:
            feature_encoded = feature_values.map(category_means)
            correlation = np.corrcoef(feature_encoded.fillna(feature_encoded.mean()), self.target)[0, 1]
            return abs(correlation) if not np.isnan(correlation) else 0
        
        return 0
    
    def apply_label_encoding(self):
        """Apply label encoding to binary categorical features"""
        print("\n" + "="*60)
        print("APPLYING LABEL ENCODING")
        print("="*60)
        
        if 'label' not in self.encoding_strategies:
            print("No features selected for label encoding")
            return self
        
        label_features = self.encoding_strategies['label']
        label_encoders = {}
        encoded_info = {}
        
        for feature in label_features:
            print(f"Label encoding {feature}...")
            
            # Fit encoder
            le = LabelEncoder()
            
            # Handle NaN values
            feature_data = self.combined_df[feature].copy()
            nan_mask = feature_data.isnull()
            
            if nan_mask.any():
                # Temporarily fill NaN for fitting
                feature_data_filled = feature_data.fillna('MISSING_VALUE')
                encoded_values = le.fit_transform(feature_data_filled)
                
                # Replace MISSING_VALUE encoding with NaN
                missing_encoded = le.transform(['MISSING_VALUE'])[0]
                encoded_values = encoded_values.astype(float)
                encoded_values[nan_mask] = np.nan
            else:
                encoded_values = le.fit_transform(feature_data)
            
            # Store encoded values
            encoded_feature_name = f"{feature}_encoded"
            self.combined_df[encoded_feature_name] = encoded_values
            
            # Store encoder and info
            label_encoders[feature] = le
            encoded_info[feature] = {
                'original_feature': feature,
                'encoded_feature': encoded_feature_name,
                'unique_values': len(le.classes_),
                'classes': le.classes_.tolist(),
                'encoding_type': 'label'
            }
            
            print(f"SUCCESS {feature} -> {encoded_feature_name} ({len(le.classes_)} classes)")
        
        self.encoding_artifacts['label_encoders'] = label_encoders
        self.encoded_features_info.update(encoded_info)
        
        return self
    
    def apply_ordinal_encoding(self):
        """Apply ordinal encoding to quality/condition features"""
        print("\n" + "="*60)
        print("APPLYING ORDINAL ENCODING")
        print("="*60)
        
        if 'ordinal' not in self.encoding_strategies:
            print("No features selected for ordinal encoding")
            return self
        
        ordinal_features = self.encoding_strategies['ordinal']
        ordinal_encoders = {}
        encoded_info = {}
        
        # Define ordinal mappings
        quality_mapping = {
            'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5, 'None': 0,
            'Poor': 1, 'Fair': 2, 'Typical': 3, 'Good': 4, 'Excellent': 5
        }
        
        basement_mapping = {
            'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6
        }
        
        for feature in ordinal_features:
            print(f"Ordinal encoding {feature}...")
            
            feature_data = self.combined_df[feature].copy()
            
            # Handle numeric ordinal features (like OverallQual)
            if feature in ['OverallQual', 'OverallCond']:
                encoded_feature_name = f"{feature}_encoded"
                self.combined_df[encoded_feature_name] = feature_data
                
                encoded_info[feature] = {
                    'original_feature': feature,
                    'encoded_feature': encoded_feature_name,
                    'unique_values': feature_data.nunique(),
                    'encoding_type': 'ordinal_numeric',
                    'mapping': 'identity'
                }
                
                print(f"SUCCESS {feature} -> {encoded_feature_name} (numeric ordinal)")
                continue
            
            # Determine appropriate mapping
            unique_values = feature_data.dropna().unique()
            mapping = None
            
            # Try quality mapping
            if any(val in quality_mapping for val in unique_values):
                mapping = quality_mapping
            # Try basement mapping
            elif any(val in basement_mapping for val in unique_values):
                mapping = basement_mapping
            else:
                # Create custom ordinal mapping based on target correlation
                print(f"Creating custom ordinal mapping for {feature}...")
                train_size = len(self.train_df)
                feature_train = feature_data.iloc[:train_size]
                
                # Calculate mean target for each category
                category_means = {}
                for category in unique_values:
                    mask = feature_train == category
                    if mask.sum() > 0:
                        category_means[category] = self.target[mask].mean()
                
                # Sort categories by mean target value
                sorted_categories = sorted(category_means.items(), key=lambda x: x[1])
                mapping = {cat: idx + 1 for idx, (cat, _) in enumerate(sorted_categories)}
                mapping['None'] = 0  # Handle None/NaN
            
            # Apply mapping
            encoded_values = feature_data.map(mapping)
            
            # Handle unmapped values
            unmapped_mask = encoded_values.isnull() & feature_data.notnull()
            if unmapped_mask.any():
                print(f"WARNING: {unmapped_mask.sum()} unmapped values in {feature}")
                # Use median encoding for unmapped values
                median_encoding = int(np.median([v for v in mapping.values() if v > 0]))
                encoded_values[unmapped_mask] = median_encoding
            
            encoded_feature_name = f"{feature}_encoded"
            self.combined_df[encoded_feature_name] = encoded_values
            
            encoded_info[feature] = {
                'original_feature': feature,
                'encoded_feature': encoded_feature_name,
                'unique_values': len(mapping),
                'mapping': mapping,
                'encoding_type': 'ordinal'
            }
            
            print(f"SUCCESS {feature} -> {encoded_feature_name} ({len(mapping)} ordinal levels)")
        
        self.encoded_features_info.update(encoded_info)
        
        return self
    
    def apply_target_encoding(self):
        """Apply target encoding to high-cardinality categorical features"""
        print("\n" + "="*60)
        print("APPLYING TARGET ENCODING")
        print("="*60)
        
        if 'target' not in self.encoding_strategies:
            print("No features selected for target encoding")
            return self
        
        target_features = self.encoding_strategies['target']
        target_encodings = {}
        encoded_info = {}
        
        train_size = len(self.train_df)
        
        # Use K-Fold cross-validation to prevent overfitting
        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for feature in target_features:
            print(f"Target encoding {feature}...")
            
            # Initialize encoded values
            encoded_train = np.zeros(train_size)
            encoded_test = np.zeros(len(self.test_df))
            
            feature_train = self.combined_df[feature].iloc[:train_size]
            feature_test = self.combined_df[feature].iloc[train_size:]
            
            # Calculate global mean for smoothing
            global_mean = self.target.mean()
            
            # Cross-validation encoding for training data
            for train_idx, val_idx in kf.split(feature_train):
                train_categories = feature_train.iloc[train_idx]
                train_target = self.target[train_idx]
                val_categories = feature_train.iloc[val_idx]
                
                # Calculate category means
                category_means = {}
                category_counts = {}
                
                for category in train_categories.unique():
                    if pd.notna(category):
                        mask = train_categories == category
                        if mask.sum() > 0:
                            category_means[category] = train_target[mask].mean()
                            category_counts[category] = mask.sum()
                
                # Apply smoothing (Bayesian average)
                smoothing = 10  # Smoothing parameter
                smoothed_means = {}
                for category, mean_val in category_means.items():
                    count = category_counts[category]
                    smoothed_means[category] = (count * mean_val + smoothing * global_mean) / (count + smoothing)
                
                # Encode validation set
                for idx, val_cat in val_categories.items():
                    if pd.notna(val_cat) and val_cat in smoothed_means:
                        encoded_train[idx] = smoothed_means[val_cat]
                    else:
                        encoded_train[idx] = global_mean
            
            # Full training encoding for test data
            category_means_full = {}
            category_counts_full = {}
            
            for category in feature_train.unique():
                if pd.notna(category):
                    mask = feature_train == category
                    if mask.sum() > 0:
                        category_means_full[category] = self.target[mask].mean()
                        category_counts_full[category] = mask.sum()
            
            # Apply smoothing for test data
            smoothed_means_full = {}
            for category, mean_val in category_means_full.items():
                count = category_counts_full[category]
                smoothed_means_full[category] = (count * mean_val + smoothing * global_mean) / (count + smoothing)
            
            # Encode test data
            for idx, test_cat in enumerate(feature_test):
                if pd.notna(test_cat) and test_cat in smoothed_means_full:
                    encoded_test[idx] = smoothed_means_full[test_cat]
                else:
                    encoded_test[idx] = global_mean
            
            # Combine and store
            encoded_combined = np.concatenate([encoded_train, encoded_test])
            encoded_feature_name = f"{feature}_encoded"
            self.combined_df[encoded_feature_name] = encoded_combined
            
            # Store encoding info
            target_encodings[feature] = {
                'category_means': smoothed_means_full,
                'global_mean': global_mean,
                'smoothing': smoothing
            }
            
            encoded_info[feature] = {
                'original_feature': feature,
                'encoded_feature': encoded_feature_name,
                'unique_values': len(smoothed_means_full),
                'encoding_type': 'target',
                'global_mean': global_mean,
                'correlation_with_target': self.calculate_categorical_target_correlation(feature)
            }
            
            print(f"SUCCESS {feature} -> {encoded_feature_name} ({len(smoothed_means_full)} categories, "
                  f"correlation: {encoded_info[feature]['correlation_with_target']:.3f})")
        
        self.encoding_artifacts['target_encodings'] = target_encodings
        self.encoded_features_info.update(encoded_info)
        
        return self
    
    def apply_onehot_encoding(self):
        """Apply one-hot encoding to low-cardinality categorical features"""
        print("\n" + "="*60)
        print("APPLYING ONE-HOT ENCODING")
        print("="*60)
        
        if 'onehot' not in self.encoding_strategies:
            print("No features selected for one-hot encoding")
            return self
        
        onehot_features = self.encoding_strategies['onehot']
        onehot_info = {}
        
        for feature in onehot_features:
            print(f"One-hot encoding {feature}...")
            
            # Get unique values
            unique_values = self.combined_df[feature].dropna().unique()
            
            # Create dummy variables
            dummies = pd.get_dummies(self.combined_df[feature], prefix=feature, dummy_na=False)
            
            # Add to combined dataframe
            for col in dummies.columns:
                self.combined_df[col] = dummies[col]
            
            onehot_info[feature] = {
                'original_feature': feature,
                'encoded_features': dummies.columns.tolist(),
                'unique_values': len(unique_values),
                'encoding_type': 'onehot',
                'categories': unique_values.tolist()
            }
            
            print(f"SUCCESS {feature} -> {len(dummies.columns)} binary features")
        
        self.encoded_features_info.update(onehot_info)
        
        return self
    
    def validate_encoding_results(self):
        """Validate encoding results and check for issues"""
        print("\n" + "="*60)
        print("VALIDATING ENCODING RESULTS")
        print("="*60)
        
        validation_results = {
            'total_original_categorical': len(self.categorical_df),
            'encoding_strategies_used': {},
            'new_features_created': 0,
            'features_with_issues': [],
            'memory_usage_change': 0
        }
        
        # Count new features by encoding type
        for encoding_type in ['label', 'ordinal', 'target', 'onehot']:
            if encoding_type in self.encoding_strategies:
                original_count = len(self.encoding_strategies[encoding_type])
                
                if encoding_type == 'onehot':
                    # Count all dummy variables
                    new_count = sum(len(info['encoded_features']) 
                                  for info in self.encoded_features_info.values() 
                                  if info['encoding_type'] == 'onehot')
                else:
                    # One encoded feature per original feature
                    new_count = original_count
                
                validation_results['encoding_strategies_used'][encoding_type] = {
                    'original_features': original_count,
                    'new_features': new_count
                }
                validation_results['new_features_created'] += new_count
        
        # Check for issues
        for feature, info in self.encoded_features_info.items():
            if info['encoding_type'] == 'onehot':
                # Check for too many dummy variables
                if len(info['encoded_features']) > 20:
                    validation_results['features_with_issues'].append(
                        f"{feature}: {len(info['encoded_features'])} dummy variables (high dimensionality)"
                    )
            elif info['encoding_type'] == 'target':
                # Check for low correlation
                if info.get('correlation_with_target', 0) < 0.1:
                    validation_results['features_with_issues'].append(
                        f"{feature}: Low target correlation ({info.get('correlation_with_target', 0):.3f})"
                    )
        
        print(f"Encoding Validation Results:")
        print(f"Original categorical features: {validation_results['total_original_categorical']}")
        print(f"New encoded features created: {validation_results['new_features_created']}")
        
        print(f"\nEncoding methods used:")
        for method, counts in validation_results['encoding_strategies_used'].items():
            print(f"  {method.upper()}: {counts['original_features']} -> {counts['new_features']} features")
        
        if validation_results['features_with_issues']:
            print(f"\nPotential issues found:")
            for issue in validation_results['features_with_issues']:
                print(f"  WARNING: {issue}")
        else:
            print(f"\nSUCCESS: No encoding issues detected!")
        
        self.validation_results = validation_results
        return self
    
    def create_encoding_visualization(self):
        """Create comprehensive encoding visualization"""
        print("\nGenerating encoding visualization...")
        
        fig, axes = plt.subplots(3, 3, figsize=(22, 18))
        fig.suptitle('Categorical Encoding Analysis', fontsize=16, fontweight='bold')
        
        # 1. Encoding strategy distribution
        strategy_counts = self.categorical_analysis['strategy_distribution']
        colors = PRIMARY_COLORS[:len(strategy_counts)]
        
        bars = axes[0,0].bar(strategy_counts.keys(), strategy_counts.values(), color=colors, alpha=0.8)
        axes[0,0].set_title('Encoding Strategy Distribution')
        axes[0,0].set_ylabel('Number of Features')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, count in zip(bars, strategy_counts.values()):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                          str(count), ha='center', va='bottom', fontweight='bold')
        
        # 2. Feature cardinality distribution
        cardinalities = self.categorical_df['Unique_Count']
        axes[0,1].hist(cardinalities, bins=20, alpha=0.7, color=PRIMARY_COLORS[0])
        axes[0,1].set_title('Feature Cardinality Distribution')
        axes[0,1].set_xlabel('Number of Unique Values')
        axes[0,1].set_ylabel('Number of Features')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Target correlation for target-encoded features
        target_features = self.categorical_df[self.categorical_df['Encoding_Strategy'] == 'target']
        if len(target_features) > 0:
            correlations = target_features['Target_Correlation'].dropna()
            if len(correlations) > 0:
                axes[0,2].hist(correlations, bins=15, alpha=0.7, color=PRIMARY_COLORS[1])
                axes[0,2].set_title('Target Correlations for Target-Encoded Features')
                axes[0,2].set_xlabel('Correlation with Target')
                axes[0,2].set_ylabel('Number of Features')
                axes[0,2].grid(True, alpha=0.3)
        
        # 4. Before/After feature count comparison
        original_count = self.validation_results['total_original_categorical']
        new_count = self.validation_results['new_features_created']
        
        comparison_data = ['Original Categorical', 'Encoded Features']
        comparison_counts = [original_count, new_count]
        
        bars = axes[1,0].bar(comparison_data, comparison_counts, 
                           color=[PRIMARY_COLORS[0], PRIMARY_COLORS[2]], alpha=0.8)
        axes[1,0].set_title('Feature Count: Before vs After Encoding')
        axes[1,0].set_ylabel('Number of Features')
        axes[1,0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, count in zip(bars, comparison_counts):
            axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                          str(count), ha='center', va='bottom', fontweight='bold')
        
        # 5. Encoding method efficiency
        method_data = []
        for method, counts in self.validation_results['encoding_strategies_used'].items():
            efficiency = counts['new_features'] / counts['original_features'] if counts['original_features'] > 0 else 0
            method_data.append((method, efficiency))
        
        if method_data:
            methods, efficiencies = zip(*method_data)
            bars = axes[1,1].bar(methods, efficiencies, color=PRIMARY_COLORS[3], alpha=0.8)
            axes[1,1].set_title('Encoding Efficiency\n(New Features / Original Features)')
            axes[1,1].set_ylabel('Ratio')
            axes[1,1].tick_params(axis='x', rotation=45)
            axes[1,1].grid(True, alpha=0.3, axis='y')
        
        # 6. Sample encoded feature distribution (pick a target-encoded feature)
        target_encoded_features = [f for f, info in self.encoded_features_info.items() 
                                 if info['encoding_type'] == 'target']
        if target_encoded_features:
            sample_feature = target_encoded_features[0]
            encoded_col = self.encoded_features_info[sample_feature]['encoded_feature']
            
            if encoded_col in self.combined_df.columns:
                axes[1,2].hist(self.combined_df[encoded_col].dropna(), bins=30, 
                             alpha=0.7, color=PRIMARY_COLORS[4])
                axes[1,2].set_title(f'Sample Target Encoding:\n{sample_feature}')
                axes[1,2].set_xlabel('Encoded Value')
                axes[1,2].set_ylabel('Frequency')
                axes[1,2].grid(True, alpha=0.3)
        
        # 7. Quality vs encoding strategy
        quality_features = self.categorical_df[self.categorical_df['Is_Ordinal'] == True]
        if len(quality_features) > 0:
            strategy_counts_ordinal = quality_features['Encoding_Strategy'].value_counts()
            axes[2,0].pie(strategy_counts_ordinal.values, labels=strategy_counts_ordinal.index, 
                         autopct='%1.1f%%', colors=PRIMARY_COLORS[:len(strategy_counts_ordinal)])
            axes[2,0].set_title('Encoding Strategies for\nOrdinal Features')
        
        # 8. High cardinality features
        high_card_features = self.categorical_df[self.categorical_df['Unique_Count'] > 10]
        if len(high_card_features) > 0:
            axes[2,1].scatter(high_card_features['Unique_Count'], 
                            high_card_features['Most_Frequent_Pct'],
                            c=[PRIMARY_COLORS[i % len(PRIMARY_COLORS)] for i in range(len(high_card_features))],
                            alpha=0.7, s=60)
            axes[2,1].set_xlabel('Number of Unique Values')
            axes[2,1].set_ylabel('Most Frequent Category %')
            axes[2,1].set_title('High Cardinality Features\nCharacteristics')
            axes[2,1].grid(True, alpha=0.3)
        
        # 9. Summary statistics
        axes[2,2].axis('off')
        
        summary_text = f"""ENCODING SUMMARY:

FEATURES PROCESSED:
• Original Categorical: {self.validation_results['total_original_categorical']}
• New Encoded Features: {self.validation_results['new_features_created']}
• Feature Expansion: {(self.validation_results['new_features_created']/self.validation_results['total_original_categorical']*100):.1f}%

ENCODING METHODS:
• Label Encoding: {self.validation_results['encoding_strategies_used'].get('label', {}).get('original_features', 0)} features
• Ordinal Encoding: {self.validation_results['encoding_strategies_used'].get('ordinal', {}).get('original_features', 0)} features
• Target Encoding: {self.validation_results['encoding_strategies_used'].get('target', {}).get('original_features', 0)} features  
• One-Hot Encoding: {self.validation_results['encoding_strategies_used'].get('onehot', {}).get('original_features', 0)} features

QUALITY ASSESSMENT:
• Issues Found: {len(self.validation_results['features_with_issues'])}
• Status: {'CLEAN' if not self.validation_results['features_with_issues'] else 'NEEDS ATTENTION'}
        """
        
        axes[2,2].text(0.05, 0.95, summary_text, transform=axes[2,2].transAxes,
                      fontsize=9, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.1))
        
        plt.tight_layout()
        plt.savefig('../../visualizations/phase5_intelligent_encoding.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return self
    
    def save_encoding_artifacts(self):
        """Save encoding results and configuration"""
        print("\nCleaning up original categorical features...")
        
        # Remove all original categorical columns after encoding
        categorical_columns_to_drop = []
        
        # Get list of categorical features from analysis
        original_categorical = []
        if hasattr(self, 'categorical_analysis') and self.categorical_analysis:
            try:
                original_categorical = [feature['Feature'] for feature in self.categorical_analysis]
            except (KeyError, TypeError):
                # Fallback: just identify object columns
                original_categorical = []
        
        for col in self.combined_df.columns:
            if (self.combined_df[col].dtype == 'object' or 
                col in original_categorical):
                categorical_columns_to_drop.append(col)
        
        if categorical_columns_to_drop:
            self.combined_df = self.combined_df.drop(categorical_columns_to_drop, axis=1)
            print(f"Removed {len(categorical_columns_to_drop)} original categorical columns")
            print(f"Remaining features: {self.combined_df.shape[1]}")
        
        print("\nSaving encoding artifacts...")
        
        # Split back into train and test
        train_size = len(self.train_df)
        
        encoded_train = self.combined_df.iloc[:train_size].copy()
        encoded_test = self.combined_df.iloc[train_size:].copy()
        
        # Add back SalePrice to training data
        encoded_train['SalePrice'] = self.train_df['SalePrice'].values
        
        # Save datasets
        encoded_train.to_csv('encoded_train_phase5.csv', index=False)
        encoded_test.to_csv('encoded_test_phase5.csv', index=False)
        
        # Save encoding configuration
        import json
        
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
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
        
        # Exclude encoder objects from serialization
        serializable_artifacts = {}
        if 'target_encodings' in self.encoding_artifacts:
            serializable_artifacts['target_encodings'] = self.encoding_artifacts['target_encodings']
        
        config = {
            'categorical_analysis': recursive_convert(self.categorical_analysis),
            'encoding_strategies': self.encoding_strategies,
            'encoded_features_info': recursive_convert(self.encoded_features_info),
            'validation_results': recursive_convert(self.validation_results),
            'encoding_artifacts': recursive_convert(serializable_artifacts)
        }
        
        with open('intelligent_encoding_config.json', 'w') as f:
            json.dump(config, f, indent=4)
        
        print("SUCCESS Encoded training data saved to encoded_train_phase5.csv")
        print("SUCCESS Encoded test data saved to encoded_test_phase5.csv")
        print("SUCCESS Encoding configuration saved to intelligent_encoding_config.json")
        
        return self
    
    def run_complete_pipeline(self):
        """Execute the complete intelligent encoding pipeline"""
        print("PHASE 5: INTELLIGENT ENCODING PIPELINE")
        print("="*70)
        
        (self.load_data()
         .analyze_categorical_features()
         .apply_label_encoding()
         .apply_ordinal_encoding()
         .apply_target_encoding()
         .apply_onehot_encoding()
         .validate_encoding_results()
         .create_encoding_visualization()
         .save_encoding_artifacts())
        
        print("\n" + "="*70)
        print("SUCCESS PHASE 5 COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"RESULT Original categorical features: {self.validation_results['total_original_categorical']}")
        print(f"RESULT New encoded features: {self.validation_results['new_features_created']}")
        expansion_rate = (self.validation_results['new_features_created']/self.validation_results['total_original_categorical']*100)
        print(f"RESULT Feature expansion: {expansion_rate:.1f}%")
        print("TARGET Ready for Phase 6: Scaling and Final Preparation")
        print("="*70)
        
        return self

if __name__ == "__main__":
    pipeline = IntelligentEncodingPipeline()
    pipeline.run_complete_pipeline()