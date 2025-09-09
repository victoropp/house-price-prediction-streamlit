"""
Comprehensive Field-by-Field Mapping Analysis
Deep analysis to map standardized values back to original user-friendly values

This script identifies ALL 224 features and their transformations to create
complete bidirectional mappings for user-friendly input/output
"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Any

def load_datasets():
    """Load both original and processed datasets"""
    
    # Original training data
    original_train = pd.read_csv(r'C:\Users\victo\Documents\Data_Science_Projects\house_price_prediction_advanced\dataset\train.csv')
    
    # Final processed training data
    processed_train = pd.read_csv(r'C:\Users\victo\Documents\Data_Science_Projects\house_price_prediction_advanced\house-price-prediction-streamlit\data\final_train_prepared.csv')
    
    print(f"Original dataset shape: {original_train.shape}")
    print(f"Processed dataset shape: {processed_train.shape}")
    
    return original_train, processed_train

def analyze_min_max_features(original_df, processed_df):
    """Analyze min-max normalized features and determine original ranges"""
    
    min_max_features = {}
    
    # Check for features that exist in both datasets with 0-1 range in processed
    for col in processed_df.columns:
        if col in original_df.columns:
            processed_vals = processed_df[col]
            original_vals = original_df[col]
            
            # Check if it looks min-max normalized (0-1 range)
            if (processed_vals.min() >= 0 and processed_vals.max() <= 1 and 
                processed_vals.max() > processed_vals.min()):
                
                min_max_features[col] = {
                    'original_min': original_vals.min(),
                    'original_max': original_vals.max(),
                    'original_mean': original_vals.mean(),
                    'original_std': original_vals.std(),
                    'processed_min': processed_vals.min(),
                    'processed_max': processed_vals.max(),
                    'processed_mean': processed_vals.mean(),
                    'sample_original': list(original_vals.head(5)),
                    'sample_processed': list(processed_vals.head(5)),
                }
                
                # Verify transformation
                expected_normalized = (original_vals - original_vals.min()) / (original_vals.max() - original_vals.min())
                correlation = np.corrcoef(processed_vals, expected_normalized)[0,1]
                min_max_features[col]['transformation_correlation'] = correlation
    
    return min_max_features

def analyze_categorical_unchanged_features(original_df, processed_df):
    """Analyze categorical features that remained unchanged"""
    
    unchanged_features = {}
    
    for col in processed_df.columns:
        if col in original_df.columns:
            # Check if values are identical or very similar
            original_vals = original_df[col]
            processed_vals = processed_df[col]
            
            # Skip if already identified as min-max
            processed_range = processed_vals.max() - processed_vals.min()
            if processed_range <= 1 and processed_vals.min() >= 0:
                continue
                
            # Check for exact match or high correlation
            if processed_vals.dtype in ['int64', 'float64']:
                try:
                    correlation = np.corrcoef(processed_vals, original_vals)[0,1]
                    if correlation > 0.95:  # High correlation indicates minimal transformation
                        unchanged_features[col] = {
                            'original_range': f"{original_vals.min()} - {original_vals.max()}",
                            'processed_range': f"{processed_vals.min()} - {processed_vals.max()}",
                            'unique_values': sorted(processed_vals.unique()),
                            'correlation': correlation,
                            'sample_original': list(original_vals.head(5)),
                            'sample_processed': list(processed_vals.head(5))
                        }
                except:
                    pass
    
    return unchanged_features

def analyze_encoded_features(processed_df):
    """Analyze features with _encoded suffix"""
    
    encoded_features = {}
    
    for col in processed_df.columns:
        if col.endswith('_encoded'):
            base_name = col.replace('_encoded', '')
            vals = processed_df[col]
            
            encoded_features[col] = {
                'base_feature': base_name,
                'range': f"{vals.min()} - {vals.max()}",
                'unique_values': sorted(vals.unique()),
                'num_unique': vals.nunique(),
                'sample_values': list(vals.head(10))
            }
    
    return encoded_features

def analyze_transformed_features(processed_df):
    """Analyze features with _transformed suffix (log transformations)"""
    
    transformed_features = {}
    
    for col in processed_df.columns:
        if col.endswith('_transformed'):
            base_name = col.replace('_transformed', '')
            vals = processed_df[col]
            
            transformed_features[col] = {
                'base_feature': base_name,
                'range': f"{vals.min():.4f} - {vals.max():.4f}",
                'mean': vals.mean(),
                'std': vals.std(),
                'contains_zero': (vals == 0).any(),
                'all_same_value': vals.nunique() == 1,
                'sample_values': list(vals.head(10))
            }
    
    return transformed_features

def analyze_onehot_features(processed_df):
    """Analyze one-hot encoded features (binary 0/1 or True/False)"""
    
    onehot_features = {}
    
    for col in processed_df.columns:
        vals = processed_df[col]
        unique_vals = sorted(vals.unique())
        
        # Check if binary
        if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, True, False}):
            onehot_features[col] = {
                'unique_values': unique_vals,
                'true_count': sum(vals == 1) if 1 in unique_vals else sum(vals == True),
                'false_count': sum(vals == 0) if 0 in unique_vals else sum(vals == False),
                'proportion_true': np.mean(vals == 1) if 1 in unique_vals else np.mean(vals == True)
            }
    
    return onehot_features

def create_reverse_mapping_functions():
    """Create functions to reverse-transform processed values back to original"""
    
    reverse_functions = {
        'min_max_denormalize': """
def denormalize_min_max(normalized_value, original_min, original_max):
    '''Convert 0-1 normalized value back to original scale'''
    return original_min + normalized_value * (original_max - original_min)
""",
        
        'log_reverse_transform': """
def reverse_log_transform(log_value, transform_type='log'):
    '''Reverse log transformation'''
    if transform_type == 'log':
        return np.exp(log_value)
    elif transform_type == 'log1p':
        return np.exp(log_value) - 1
    else:
        return log_value
""",
        
        'quality_scale_mapping': """
# Quality scale mappings (based on pattern analysis)
QUALITY_SCALES = {
    'ExterQual_encoded': {1: 'Po', 2: 'Fa', 3: 'TA', 4: 'Gd', 5: 'Ex'},
    'ExterCond_encoded': {1: 'Po', 2: 'Fa', 3: 'TA', 4: 'Gd', 5: 'Ex'},
    'BsmtQual_encoded': {1: 'Po', 2: 'Fa', 3: 'TA', 4: 'Gd', 5: 'Ex'},
    'BsmtCond_encoded': {1: 'Po', 2: 'Fa', 3: 'TA', 4: 'Gd'},
    'HeatingQC_encoded': {1: 'Po', 2: 'Fa', 3: 'TA', 4: 'Gd', 5: 'Ex'},
    'KitchenQual_encoded': {1: 'Po', 2: 'Fa', 3: 'TA', 4: 'Gd', 5: 'Ex'},
    'FireplaceQu_encoded': {1: 'Po', 2: 'Fa', 3: 'TA', 4: 'Gd', 5: 'Ex'},
    'GarageQual_encoded': {1: 'Po', 2: 'Fa', 3: 'TA', 4: 'Gd', 5: 'Ex'},
    'GarageCond_encoded': {1: 'Po', 2: 'Fa', 3: 'TA', 4: 'Gd', 5: 'Ex'},
    'PoolQC_encoded': {2: 'Fa', 3: 'TA', 4: 'Gd', 5: 'Ex'}
}

def decode_quality_feature(encoded_value, feature_name):
    '''Convert encoded quality value back to original quality rating'''
    return QUALITY_SCALES.get(feature_name, {}).get(int(encoded_value), str(encoded_value))
"""
    }
    
    return reverse_functions

def generate_comprehensive_mapping_report(original_df, processed_df):
    """Generate comprehensive report of all field mappings"""
    
    print("COMPREHENSIVE FIELD MAPPING ANALYSIS")
    print("=" * 60)
    
    # Analyze different transformation types
    min_max_features = analyze_min_max_features(original_df, processed_df)
    unchanged_features = analyze_categorical_unchanged_features(original_df, processed_df)
    encoded_features = analyze_encoded_features(processed_df)
    transformed_features = analyze_transformed_features(processed_df)
    onehot_features = analyze_onehot_features(processed_df)
    
    print(f"\nTRANSFORMATION SUMMARY:")
    print(f"   Min-Max Normalized Features: {len(min_max_features)}")
    print(f"   Unchanged Categorical Features: {len(unchanged_features)}")
    print(f"   Encoded Features: {len(encoded_features)}")
    print(f"   Log-Transformed Features: {len(transformed_features)}")
    print(f"   One-Hot Encoded Features: {len(onehot_features)}")
    print(f"   Total Features Analyzed: {len(min_max_features) + len(unchanged_features) + len(encoded_features) + len(transformed_features) + len(onehot_features)}")
    
    # Detailed analysis of key user-facing features
    key_user_features = ['YearBuilt', 'LotArea', 'GrLivArea', 'TotalBsmtSF', 
                        '1stFlrSF', '2ndFlrSF', 'OverallQual', 'OverallCond',
                        'BedroomAbvGr', 'FullBath', 'HalfBath', 'GarageCars']
    
    print(f"\nKEY USER-FACING FEATURES ANALYSIS:")
    print("-" * 50)
    
    for feature in key_user_features:
        if feature in min_max_features:
            data = min_max_features[feature]
            print(f"\n{feature} (MIN-MAX NORMALIZED):")
            print(f"  Original Range: {data['original_min']:,.0f} - {data['original_max']:,.0f}")
            orig_samples = [f'{x:,.0f}' for x in data['sample_original'][:3]]
            norm_samples = [f'{x:.4f}' for x in data['sample_processed'][:3]]
            print(f"  Sample Real Values: {orig_samples}...")
            print(f"  Sample Normalized: {norm_samples}...")
            print(f"  Transformation Quality: {data['transformation_correlation']:.4f}")
            
        elif feature in unchanged_features:
            data = unchanged_features[feature]
            print(f"\n{feature} (UNCHANGED):")
            print(f"  Range: {data['original_range']}")
            print(f"  Sample Values: {data['sample_original'][:5]}")
            
        else:
            print(f"\n{feature}: NOT FOUND or requires deeper analysis")
    
    print(f"\nENCODED FEATURES ANALYSIS:")
    print("-" * 50)
    
    quality_features = [f for f in encoded_features.keys() if 'Qual' in f or 'Cond' in f]
    for feature in quality_features[:5]:  # Show first 5
        data = encoded_features[feature]
        print(f"\n{feature}:")
        print(f"  Range: {data['range']}")
        print(f"  Unique Values: {data['unique_values']}")
        print(f"  Likely Quality Scale: Po(1) -> Fa(2) -> TA(3) -> Gd(4) -> Ex(5)")
    
    print(f"\nLOG-TRANSFORMED FEATURES ANALYSIS:")
    print("-" * 50)
    
    area_features = [f for f in transformed_features.keys() if 'Area' in f or 'SF' in f]
    for feature in area_features[:5]:  # Show first 5
        data = transformed_features[feature]
        print(f"\n{feature}:")
        print(f"  Range: {data['range']}")
        print(f"  Contains Zero: {data['contains_zero']}")
        print(f"  All Same Value: {data['all_same_value']}")
        if data['all_same_value']:
            print(f"  WARNING: This feature appears to be problematic - all values are identical!")
    
    # Generate reverse mapping functions
    reverse_functions = create_reverse_mapping_functions()
    
    return {
        'min_max_features': min_max_features,
        'unchanged_features': unchanged_features,
        'encoded_features': encoded_features,
        'transformed_features': transformed_features,
        'onehot_features': onehot_features,
        'reverse_functions': reverse_functions,
        'summary': {
            'total_features': processed_df.shape[1],
            'min_max_count': len(min_max_features),
            'unchanged_count': len(unchanged_features),
            'encoded_count': len(encoded_features),
            'transformed_count': len(transformed_features),
            'onehot_count': len(onehot_features)
        }
    }

def save_mapping_results(results, output_file):
    """Save mapping results to pickle file for later use"""
    
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nResults saved to: {output_file}")

def main():
    """Main analysis function"""
    
    print("Starting Comprehensive Field Mapping Analysis...")
    
    # Load datasets
    original_df, processed_df = load_datasets()
    
    # Generate comprehensive mapping report
    results = generate_comprehensive_mapping_report(original_df, processed_df)
    
    # Save results
    output_file = 'comprehensive_field_mappings.pkl'
    save_mapping_results(results, output_file)
    
    print(f"\nAnalysis Complete!")
    print(f"\nKey Insights:")
    print(f"  - {results['summary']['min_max_count']} features are min-max normalized (0-1 range)")
    print(f"  - {results['summary']['unchanged_count']} features remain in original scale")
    print(f"  - {results['summary']['encoded_count']} features are label-encoded")
    print(f"  - {results['summary']['transformed_count']} features are log-transformed")
    print(f"  - {results['summary']['onehot_count']} features are one-hot encoded")
    
    print(f"\nNext Steps:")
    print(f"  1. Implement bidirectional transformation functions")
    print(f"  2. Update Streamlit UI to use real-world values")
    print(f"  3. Test prediction accuracy with reverse transformations")
    print(f"  4. Validate all mappings with sample predictions")

if __name__ == "__main__":
    main()