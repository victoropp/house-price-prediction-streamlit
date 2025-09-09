import pandas as pd
import numpy as np

# Load datasets from different phases
print("Loading datasets from different preprocessing phases...")

# Original data
original_df = pd.read_csv(r'C:\Users\victo\Documents\Data_Science_Projects\house_price_prediction_advanced\dataset\train.csv')
print("Original dataset shape:", original_df.shape)

# Phase 3 (after feature engineering, before distribution transforms)
phase3_df = pd.read_csv(r'C:\Users\victo\Documents\Data_Science_Projects\house_price_prediction_advanced\preprocessing\phase3_feature_engineering\engineered_train_phase3.csv')
print("Phase 3 dataset shape:", phase3_df.shape)

# Phase 5 (after encoding, before scaling)  
phase5_df = pd.read_csv(r'C:\Users\victo\Documents\Data_Science_Projects\house_price_prediction_advanced\preprocessing\phase5_encoding\encoded_train_phase5.csv')
print("Phase 5 dataset shape:", phase5_df.shape)

# Final prepared data (our current normalized data)
final_df = pd.read_csv(r'C:\Users\victo\Documents\Data_Science_Projects\house_price_prediction_advanced\house-price-prediction-streamlit\data\final_train_prepared.csv')
print("Final dataset shape:", final_df.shape)

print("\n=== ANALYZING KEY FEATURES FOR REVERSE TRANSFORMATION ===")

# Key features users care about
key_features = ['YearBuilt', 'LotArea', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 
                'OverallQual', 'OverallCond', 'BedroomAbvGr', 'FullBath', 'HalfBath']

# Create mapping dictionary for reverse transformations
reverse_transforms = {}

for feature in key_features:
    print(f"\n--- {feature} ---")
    
    # Check original values
    if feature in original_df.columns:
        orig_min = original_df[feature].min()
        orig_max = original_df[feature].max()
        orig_mean = original_df[feature].mean()
        print(f"Original: {orig_min} - {orig_max} (mean: {orig_mean:.2f})")
    
    # Check phase 3 values (after feature engineering)
    if feature in phase3_df.columns:
        p3_min = phase3_df[feature].min()
        p3_max = phase3_df[feature].max() 
        p3_mean = phase3_df[feature].mean()
        print(f"Phase 3: {p3_min} - {p3_max} (mean: {p3_mean:.2f})")
        
        # Store for reverse transformation
        reverse_transforms[feature] = {
            'type': 'min_max',
            'original_min': p3_min,
            'original_max': p3_max,
            'original_mean': p3_mean
        }
    
    # Check final normalized values
    if feature in final_df.columns:
        final_min = final_df[feature].min()
        final_max = final_df[feature].max()
        final_mean = final_df[feature].mean()
        print(f"Final: {final_min:.4f} - {final_max:.4f} (mean: {final_mean:.4f})")
        
        # If this is min-max normalized, calculate the transformation
        if final_min == 0.0 and final_max == 1.0:
            print(f"MIN-MAX NORMALIZED: 0.0 represents {reverse_transforms.get(feature, {}).get('original_min', 'Unknown')}")
            print(f"  1.0 represents {reverse_transforms.get(feature, {}).get('original_max', 'Unknown')}")
            
            # Calculate example real values
            if feature in reverse_transforms:
                example_normalized = final_mean
                example_real = (reverse_transforms[feature]['original_min'] + 
                              example_normalized * (reverse_transforms[feature]['original_max'] - 
                                                  reverse_transforms[feature]['original_min']))
                print(f"  Example: {example_normalized:.4f} (normalized) = {example_real:.0f} (real)")

# Now analyze log-transformed features
print(f"\n\n=== LOG-TRANSFORMED FEATURES ANALYSIS ===")
log_features = ['LotArea_transformed', 'GrLivArea_transformed', 'TotalBsmtSF_transformed', '1stFlrSF_transformed']

for log_feature in log_features:
    base_feature = log_feature.replace('_transformed', '')
    print(f"\n--- {log_feature} ({base_feature}) ---")
    
    if log_feature in final_df.columns:
        log_min = final_df[log_feature].min()
        log_max = final_df[log_feature].max()
        log_mean = final_df[log_feature].mean()
        print(f"Log values: {log_min:.4f} - {log_max:.4f} (mean: {log_mean:.4f})")
        
        # Check if base feature exists in phase5 (before log transform)
        if base_feature in phase5_df.columns:
            orig_min = phase5_df[base_feature].min()
            orig_max = phase5_df[base_feature].max()
            orig_mean = phase5_df[base_feature].mean()
            print(f"Before log: {orig_min:.0f} - {orig_max:.0f} (mean: {orig_mean:.0f})")
            
            # Try to determine the log transformation type
            if log_min == 0.0:
                print("  Likely log1p transformation (handles zeros)")
                reverse_transforms[log_feature] = {
                    'type': 'log1p',
                    'formula': 'np.exp(log_value) - 1'
                }
            else:
                print("  Likely log transformation")  
                reverse_transforms[log_feature] = {
                    'type': 'log',
                    'formula': 'np.exp(log_value)'
                }

print(f"\n\n=== TYPICAL HOUSE FEATURE RANGES (for validation) ===")
typical_ranges = {
    'YearBuilt': (1872, 2010, "Years houses were built"),
    'LotArea': (1300, 215245, "Square feet of lot area"),  
    'GrLivArea': (334, 5642, "Square feet of living area above ground"),
    'TotalBsmtSF': (0, 6110, "Square feet of basement area"),
    '1stFlrSF': (334, 4692, "Square feet of first floor"),
    '2ndFlrSF': (0, 2065, "Square feet of second floor"),
    'OverallQual': (1, 10, "Overall quality rating"),
    'OverallCond': (1, 9, "Overall condition rating"),
    'BedroomAbvGr': (0, 8, "Bedrooms above ground"),
    'FullBath': (0, 3, "Full bathrooms"),
    'HalfBath': (0, 2, "Half bathrooms")
}

for feature, (min_val, max_val, desc) in typical_ranges.items():
    print(f"{feature}: {min_val:,} - {max_val:,} ({desc})")

print(f"\n\n=== REVERSE TRANSFORMATION FORMULAS ===")
for feature, transform in reverse_transforms.items():
    print(f"\n{feature}:")
    if transform['type'] == 'min_max':
        print(f"  Formula: real_value = {transform['original_min']} + normalized_value * ({transform['original_max']} - {transform['original_min']})")
        print(f"  Range: {transform['original_min']:,} - {transform['original_max']:,}")
    else:
        print(f"  Formula: {transform['formula']}")
        print(f"  Type: {transform['type']}")