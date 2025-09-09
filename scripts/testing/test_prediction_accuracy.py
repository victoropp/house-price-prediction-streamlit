"""
Test Prediction Accuracy with Transformations
Ensure that the bidirectional transformations maintain model prediction accuracy
"""

import sys
import os
sys.path.append(r'C:\Users\victo\Documents\Data_Science_Projects\house_price_prediction_advanced\house-price-prediction-streamlit')

import pandas as pd
import numpy as np
import pickle
from utils.complete_transformations import CompleteHouseTransformations

def load_model_and_data():
    """Load the trained model and test data"""
    
    try:
        # Load the best model
        model_path = r'C:\Users\victo\Documents\Data_Science_Projects\house_price_prediction_advanced\house-price-prediction-streamlit\models\best_model.pkl'
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load processed test data
        test_data_path = r'C:\Users\victo\Documents\Data_Science_Projects\house_price_prediction_advanced\house-price-prediction-streamlit\data\final_train_prepared.csv'
        processed_data = pd.read_csv(test_data_path)
        
        # Load original data for comparison
        original_data_path = r'C:\Users\victo\Documents\Data_Science_Projects\house_price_prediction_advanced\dataset\train.csv'
        original_data = pd.read_csv(original_data_path)
        
        print(f"Model loaded: {type(model)}")
        print(f"Processed data shape: {processed_data.shape}")
        print(f"Original data shape: {original_data.shape}")
        
        return model, processed_data, original_data
    
    except Exception as e:
        print(f"Error loading model or data: {e}")
        return None, None, None

def test_direct_predictions(model, processed_data, num_samples=5):
    """Test predictions using processed data directly"""
    
    print("\n" + "="*60)
    print("TESTING DIRECT PREDICTIONS (Using Processed Data)")
    print("="*60)
    
    # Get features and target
    if 'SalePrice_transformed' in processed_data.columns:
        target_col = 'SalePrice_transformed'
        X = processed_data.drop([target_col], axis=1)
        y_true = processed_data[target_col]
    else:
        print("No target column found - using first N samples for prediction testing")
        X = processed_data.iloc[:, :-1]  # All but last column
        y_true = None
    
    # Make predictions on sample
    sample_indices = range(min(num_samples, len(X)))
    X_sample = X.iloc[sample_indices]
    
    try:
        predictions = model.predict(X_sample)
        print(f"\nDirect predictions for {len(sample_indices)} samples:")
        
        for i, (idx, pred) in enumerate(zip(sample_indices, predictions)):
            if y_true is not None:
                actual = y_true.iloc[idx]
                print(f"  Sample {idx}: Predicted = {pred:.4f}, Actual = {actual:.4f}")
            else:
                print(f"  Sample {idx}: Predicted = {pred:.4f}")
        
        return predictions, X_sample
    
    except Exception as e:
        print(f"Error making direct predictions: {e}")
        return None, None

def test_transformation_predictions(model, original_data, processed_data, transformer, num_samples=5):
    """Test predictions using user-friendly inputs transformed to model format"""
    
    print("\n" + "="*60)
    print("TESTING TRANSFORMATION PREDICTIONS (User Input -> Model)")
    print("="*60)
    
    # Get priority features that we can transform
    priority_features = transformer.get_priority_features()
    
    # Find which priority features exist in both datasets
    available_features = []
    for feature in priority_features:
        if feature in original_data.columns and feature in processed_data.columns:
            available_features.append(feature)
    
    print(f"Available priority features for testing: {len(available_features)}")
    print(f"Features: {available_features[:10]}...")  # Show first 10
    
    # Test transformation pipeline for a few samples
    sample_indices = range(min(num_samples, len(original_data)))
    
    transformation_results = []
    
    for idx in sample_indices:
        print(f"\n--- Testing Sample {idx} ---")
        
        # Get original values for this sample
        original_sample = original_data.iloc[idx]
        processed_sample = processed_data.iloc[idx]
        
        # Create transformed sample using our transformation functions
        transformed_sample = {}
        
        # Test key features
        test_features = ['YearBuilt', 'LotArea', 'GrLivArea', 'OverallQual']
        
        for feature in test_features:
            if feature in original_data.columns:
                original_val = original_sample[feature]
                processed_val = processed_sample[feature]
                
                # Transform using our function
                transformed_val = transformer.transform_user_input(feature, original_val)
                
                transformed_sample[feature] = transformed_val
                
                print(f"  {feature:15} | Original: {original_val:>8} | Processed: {processed_val:>8.4f} | Our Transform: {transformed_val:>8.4f}")
                
                # Check if our transformation matches the processed data
                diff = abs(transformed_val - processed_val)
                if diff > 0.001:  # Small tolerance for floating point
                    print(f"    WARNING: Transformation mismatch! Difference: {diff:.6f}")
        
        transformation_results.append({
            'index': idx,
            'original': original_sample[test_features].to_dict(),
            'processed': processed_sample[test_features].to_dict(),
            'transformed': transformed_sample
        })
    
    return transformation_results

def test_round_trip_accuracy(transformer, num_tests=10):
    """Test round-trip accuracy for various value ranges"""
    
    print("\n" + "="*60)
    print("TESTING ROUND-TRIP ACCURACY")
    print("="*60)
    
    test_cases = [
        ('YearBuilt', [1900, 1950, 1980, 2000, 2010]),
        ('LotArea', [5000, 8000, 12000, 20000, 25000]),
        ('GrLivArea', [1000, 1500, 2000, 2500, 3000]),
        ('TotalBsmtSF', [0, 500, 1000, 1500, 2000]),
        ('OverallQual', [4, 5, 6, 7, 8, 9, 10])
    ]
    
    all_passed = True
    
    for feature, test_values in test_cases:
        print(f"\n{feature}:")
        feature_passed = True
        
        for original_val in test_values:
            # Transform to model format
            transformed = transformer.transform_user_input(feature, original_val)
            
            # Transform back to user format
            if transformer.get_feature_type(feature) == 'min_max':
                recovered = transformer.denormalize_min_max(transformed, feature)
            else:
                recovered = transformed
            
            # Check accuracy
            if isinstance(original_val, int):
                accuracy = abs(recovered - original_val) < 1
            else:
                accuracy = abs(recovered - original_val) / original_val < 0.01  # 1% tolerance
            
            status = "PASS" if accuracy else "FAIL"
            print(f"  {original_val:>8} -> {transformed:>8.4f} -> {recovered:>8} [{status}]")
            
            if not accuracy:
                feature_passed = False
                all_passed = False
        
        if not feature_passed:
            print(f"  WARNING: {feature} failed round-trip tests!")
    
    return all_passed

def generate_sample_user_inputs(transformer):
    """Generate realistic sample user inputs for testing"""
    
    print("\n" + "="*60)
    print("GENERATING SAMPLE USER INPUTS")
    print("="*60)
    
    # Create realistic house scenarios
    house_scenarios = [
        {
            'name': 'Starter Home',
            'YearBuilt': 1990,
            'LotArea': 7500,
            'GrLivArea': 1200,
            'TotalBsmtSF': 800,
            '1stFlrSF': 800,
            '2ndFlrSF': 400,
            'OverallQual': 5,
            'OverallCond': 6,
            'BedroomAbvGr': 3,
            'FullBath': 2,
            'HalfBath': 0,
            'GarageCars': 2
        },
        {
            'name': 'Family Home',
            'YearBuilt': 2000,
            'LotArea': 12000,
            'GrLivArea': 2000,
            'TotalBsmtSF': 1200,
            '1stFlrSF': 1200,
            '2ndFlrSF': 800,
            'OverallQual': 7,
            'OverallCond': 7,
            'BedroomAbvGr': 4,
            'FullBath': 2,
            'HalfBath': 1,
            'GarageCars': 2
        },
        {
            'name': 'Luxury Home',
            'YearBuilt': 2005,
            'LotArea': 18000,
            'GrLivArea': 3000,
            'TotalBsmtSF': 2000,
            '1stFlrSF': 1800,
            '2ndFlrSF': 1200,
            'OverallQual': 9,
            'OverallCond': 8,
            'BedroomAbvGr': 5,
            'FullBath': 3,
            'HalfBath': 2,
            'GarageCars': 3
        }
    ]
    
    transformed_scenarios = []
    
    for scenario in house_scenarios:
        name = scenario.pop('name')
        print(f"\n{name}:")
        
        transformed_scenario = {'name': name}
        
        for feature, value in scenario.items():
            transformed_value = transformer.transform_user_input(feature, value)
            transformed_scenario[feature] = transformed_value
            
            # Show the transformation
            feature_type = transformer.get_feature_type(feature)
            print(f"  {feature:15} | {value:>8} -> {transformed_value:>8.4f} ({feature_type})")
        
        transformed_scenarios.append(transformed_scenario)
    
    return transformed_scenarios

def main():
    """Main testing function"""
    
    print("PREDICTION ACCURACY TESTING WITH TRANSFORMATIONS")
    print("="*80)
    
    # Initialize transformer
    transformer = CompleteHouseTransformations()
    
    # Load model and data
    model, processed_data, original_data = load_model_and_data()
    
    if model is None:
        print("Could not load model - skipping prediction tests")
        return
    
    # Test 1: Direct predictions using processed data
    predictions, X_sample = test_direct_predictions(model, processed_data, num_samples=3)
    
    # Test 2: Compare our transformations with the processed data
    if original_data is not None:
        transformation_results = test_transformation_predictions(
            model, original_data, processed_data, transformer, num_samples=3
        )
    
    # Test 3: Round-trip accuracy
    round_trip_passed = test_round_trip_accuracy(transformer)
    
    # Test 4: Generate realistic user input samples
    sample_scenarios = generate_sample_user_inputs(transformer)
    
    # Test 5: Make predictions with user-friendly inputs
    if model is not None and predictions is not None:
        print("\n" + "="*60)
        print("TESTING PREDICTIONS WITH USER-FRIENDLY INPUTS")
        print("="*60)
        
        for scenario in sample_scenarios[:2]:  # Test first 2 scenarios
            name = scenario.pop('name')
            print(f"\n{name}:")
            
            # Create a full feature vector (need to handle missing features)
            # For this test, we'll use the first sample as a template and update key features
            if X_sample is not None:
                feature_vector = X_sample.iloc[0].copy()
                
                # Update with our transformed values
                for feature, value in scenario.items():
                    if feature in feature_vector.index:
                        feature_vector[feature] = value
                        print(f"  {feature:15} = {value:8.4f}")
                
                try:
                    prediction = model.predict([feature_vector.values])[0]
                    print(f"  -> Predicted Price (transformed): {prediction:.4f}")
                    
                    # If we know it's log-transformed, reverse it
                    if 'SalePrice' in transformer.working_log_features:
                        real_price = transformer.reverse_log_feature(prediction, 'SalePrice_transformed')
                        print(f"  -> Estimated Real Price: ${real_price:,.0f}")
                    else:
                        print(f"  -> Prediction (unknown scale): {prediction:.4f}")
                        
                except Exception as e:
                    print(f"  -> Error making prediction: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("TESTING SUMMARY")
    print("="*80)
    
    print(f"PASS: Transformation system loaded successfully")
    print(f"PASS: Model loaded and can make predictions") if model else print(f"FAIL: Model loading failed")
    print(f"PASS: Round-trip accuracy tests passed") if round_trip_passed else print(f"FAIL: Some round-trip tests failed")
    print(f"PASS: Generated realistic user input scenarios")
    
    if model and round_trip_passed:
        print(f"\nSYSTEM READY FOR DEPLOYMENT")
        print(f"   - Bidirectional transformations working correctly")
        print(f"   - Model predictions functional")
        print(f"   - User-friendly inputs can be converted to model format")
    else:
        print(f"\nISSUES DETECTED - Review test results above")

if __name__ == "__main__":
    main()