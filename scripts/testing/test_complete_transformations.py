"""
Test Complete Transformations
Validate that the new transformation system works correctly
"""

import sys
import os
sys.path.append(r'C:\Users\victo\Documents\Data_Science_Projects\house_price_prediction_advanced\house-price-prediction-streamlit')

from utils.complete_transformations import CompleteHouseTransformations
import pandas as pd
import numpy as np

def test_transformations():
    """Test all transformation types"""
    
    print("Testing Complete House Transformations")
    print("=" * 50)
    
    # Initialize transformer
    transformer = CompleteHouseTransformations()
    
    # Test 1: Min-Max Transformations
    print("\n1. MIN-MAX TRANSFORMATION TESTS")
    print("-" * 30)
    
    test_cases = [
        ('YearBuilt', 2000),
        ('LotArea', 10000),
        ('GrLivArea', 1500),
        ('TotalBsmtSF', 1000),
        ('1stFlrSF', 1200),
        ('2ndFlrSF', 800)
    ]
    
    for feature, test_value in test_cases:
        normalized = transformer.normalize_min_max(test_value, feature)
        denormalized = transformer.denormalize_min_max(normalized, feature)
        
        print(f"{feature:15} | {test_value:>8} -> {normalized:.4f} -> {denormalized:>8}")
        
        # Verify round-trip accuracy
        accuracy = abs(denormalized - test_value) < 1
        if not accuracy:
            print(f"  WARNING: Round-trip error for {feature}")
    
    # Test 2: Quality Encoding
    print("\n2. QUALITY ENCODING TESTS")
    print("-" * 30)
    
    quality_tests = [
        ('ExterQual_encoded', 'Gd'),
        ('HeatingQC_encoded', 'Ex'),
        ('KitchenQual_encoded', 'TA'),
        ('BsmtQual_encoded', 'Fa')
    ]
    
    for feature, test_quality in quality_tests:
        encoded = transformer.encode_quality(test_quality, feature)
        decoded = transformer.decode_quality(encoded, feature)
        
        print(f"{feature:20} | {test_quality:>2} -> {encoded:>2} -> {decoded:>2}")
        
        if decoded != test_quality:
            print(f"  WARNING: Quality round-trip error for {feature}")
    
    # Test 3: Binary Encoding
    print("\n3. BINARY ENCODING TESTS")
    print("-" * 30)
    
    binary_tests = [
        ('Street_encoded', 'Pave'),
        ('CentralAir_encoded', 'Y'),
        ('Utilities_encoded', 'AllPub')
    ]
    
    for feature, test_binary in binary_tests:
        encoded = transformer.encode_binary(test_binary, feature)
        decoded = transformer.decode_binary(encoded, feature)
        
        print(f"{feature:20} | {test_binary:>6} -> {encoded:>2} -> {decoded:>6}")
        
        if decoded != test_binary:
            print(f"  WARNING: Binary round-trip error for {feature}")
    
    # Test 4: Feature Type Detection
    print("\n4. FEATURE TYPE DETECTION")
    print("-" * 30)
    
    feature_samples = [
        'YearBuilt',
        'OverallQual',
        'ExterQual_encoded',
        'Street_encoded',
        'LotArea_transformed',
        'MSZoning_RL',
        'UnknownFeature'
    ]
    
    for feature in feature_samples:
        feature_type = transformer.get_feature_type(feature)
        print(f"{feature:20} | {feature_type}")
    
    # Test 5: User Input Configuration
    print("\n5. USER INPUT CONFIGURATION")
    print("-" * 30)
    
    priority_features = transformer.get_priority_features()[:5]  # First 5
    
    for feature in priority_features:
        config = transformer.get_user_input_config(feature)
        print(f"\n{feature}:")
        print(f"  Type: {config['type']}")
        print(f"  Label: {config['label']}")
        if 'min_value' in config:
            print(f"  Range: {config['min_value']} - {config['max_value']}")
        if 'options' in config:
            print(f"  Options: {config['options']}")
    
    # Test 6: Validation
    print("\n6. COMPREHENSIVE VALIDATION")
    print("-" * 30)
    
    validation_results = transformer.validate_transformations()
    
    print("Min-Max Tests:")
    for feature, result in validation_results['min_max_tests'].items():
        status = "PASS" if result['match'] else "FAIL"
        print(f"  {status} {feature}: {result['input']} -> {result['normalized']:.4f} -> {result['denormalized']}")
    
    print("\nQuality Tests:")
    for feature, result in validation_results['quality_tests'].items():
        status = "PASS" if result['match'] else "FAIL"
        print(f"  {status} {feature}: {result['input']} -> {result['encoded']} -> {result['decoded']}")
    
    # Test 7: Real Dataset Sample
    print("\n7. REAL DATASET SAMPLE TEST")
    print("-" * 30)
    
    try:
        # Load a sample from the processed dataset
        processed_df = pd.read_csv(r'C:\Users\victo\Documents\Data_Science_Projects\house_price_prediction_advanced\house-price-prediction-streamlit\data\final_train_prepared.csv')
        original_df = pd.read_csv(r'C:\Users\victo\Documents\Data_Science_Projects\house_price_prediction_advanced\dataset\train.csv')
        
        # Test with first row
        sample_idx = 0
        
        print(f"Testing with row {sample_idx}:")
        
        # Test key features
        key_features = ['YearBuilt', 'LotArea', 'GrLivArea', 'OverallQual']
        
        for feature in key_features:
            if feature in processed_df.columns and feature in original_df.columns:
                processed_val = processed_df[feature].iloc[sample_idx]
                original_val = original_df[feature].iloc[sample_idx]
                
                # If it's min-max normalized, reverse it
                if transformer.get_feature_type(feature) == 'min_max':
                    reversed_val = transformer.denormalize_min_max(processed_val, feature)
                    print(f"  {feature:15} | Original: {original_val:>8} | Processed: {processed_val:.4f} | Reversed: {reversed_val:>8}")
                    
                    # Check if reversal matches original
                    match = abs(reversed_val - original_val) < 1
                    if not match:
                        print(f"    WARNING: Reversal doesn't match original!")
                else:
                    print(f"  {feature:15} | Original: {original_val:>8} | Processed: {processed_val:>8} | (unchanged)")
        
        print(f"\nDataset shapes: Original {original_df.shape}, Processed {processed_df.shape}")
        
    except Exception as e:
        print(f"Could not load datasets: {e}")
    
    print("\n" + "=" * 50)
    print("TRANSFORMATION TESTING COMPLETE")
    print("Review any warnings above and fix if necessary")

if __name__ == "__main__":
    test_transformations()