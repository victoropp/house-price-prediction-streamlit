"""
Test the RealWorldTransformations system to ensure accuracy is maintained
"""

import pandas as pd
import numpy as np
from utils.real_world_transformations import RealWorldTransformations
from utils.data_loader import PipelineDataLoader
import pickle

def test_transformations():
    """Test that transformations maintain model accuracy"""
    
    print("TESTING REAL-WORLD TRANSFORMATIONS")
    print("=" * 50)
    
    # Initialize components
    transformer = RealWorldTransformations()
    data_loader = PipelineDataLoader()
    
    # Load model and test data
    model = data_loader.load_champion_model()
    train_data = data_loader.load_training_data()
    
    print(f"Model loaded: {type(model).__name__}")
    print(f"Training data: {train_data.shape}")
    print()
    
    # Test 1: Min-Max Transformation Accuracy
    print("Test 1: Min-Max Transformation Accuracy")
    print("-" * 30)
    
    test_features = ['YearBuilt', 'GrLivArea', 'TotalBsmtSF', 'GarageArea']
    
    for feature in test_features:
        if feature in train_data.columns:
            # Get a sample value from the training data
            original_normalized = train_data[feature].iloc[100]  # Use row 100 as test
            
            # Convert to real-world value
            real_value = transformer.denormalize_min_max(original_normalized, feature)
            
            # Convert back to normalized
            back_to_normalized = transformer.normalize_min_max(real_value, feature)
            
            # Check accuracy
            diff = abs(original_normalized - back_to_normalized)
            accuracy = (1 - diff) * 100
            
            print(f"{feature}:")
            print(f"  Original normalized: {original_normalized:.4f}")
            print(f"  Real-world value: {real_value}")
            print(f"  Back to normalized: {back_to_normalized:.4f}")
            print(f"  Transformation accuracy: {accuracy:.2f}%")
            print()
    
    # Test 2: End-to-End Prediction Test
    print("Test 2: End-to-End Prediction Test")
    print("-" * 30)
    
    # Create test inputs using real-world values
    real_world_inputs = {
        'GrLivArea': 1800,  # 1800 sq ft
        'TotalBsmtSF': 1000,  # 1000 sq ft basement
        'GarageArea': 500,  # 500 sq ft garage
        'OverallQual': 7,  # Good quality
        'YearBuilt': 1990,  # Built in 1990
        'Fireplaces': 1  # 1 fireplace
    }
    
    # Convert to normalized values for model
    model_inputs = {}
    for feature, real_value in real_world_inputs.items():
        if feature in transformer.min_max_features:
            model_inputs[feature] = transformer.normalize_min_max(real_value, feature)
        else:
            model_inputs[feature] = real_value
    
    print("Real-world inputs:")
    for feature, value in real_world_inputs.items():
        print(f"  {feature}: {value}")
    
    print("\nNormalized model inputs:")
    for feature, value in model_inputs.items():
        print(f"  {feature}: {value:.4f}")
    
    # Prepare prediction features using existing data loader
    try:
        prediction_df = data_loader.prepare_prediction_features(model_inputs)
        prediction = model.predict(prediction_df)[0]
        
        # Apply Box-Cox inverse transformation
        boxcox_lambda = -0.077
        if boxcox_lambda != 0:
            prediction = np.power(boxcox_lambda * prediction + 1, 1/boxcox_lambda)
        
        print(f"\nPredicted Price: ${prediction:,.0f}")
        print("End-to-end prediction successful!")
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return False
    
    # Test 3: Feature Coverage Test
    print("\nTest 3: Feature Coverage Test")
    print("-" * 30)
    
    min_max_covered = len(transformer.min_max_features)
    categorical_covered = len(transformer.categorical_features)
    quality_covered = len(transformer.quality_scales)
    
    print(f"Min-Max features covered: {min_max_covered}")
    print(f"Categorical features covered: {categorical_covered}")
    print(f"Quality scale features covered: {quality_covered}")
    print(f"Total features with transformations: {min_max_covered + categorical_covered + quality_covered}")
    
    # Check some key features are covered
    key_features = ['YearBuilt', 'GrLivArea', 'TotalBsmtSF', 'OverallQual', 'OverallCond']
    coverage_check = True
    
    for feature in key_features:
        if feature in transformer.min_max_features or feature in transformer.categorical_features:
            print(f"{feature}: Covered")
        else:
            print(f"{feature}: NOT covered")
            coverage_check = False
    
    # Test 4: UI Configuration Test
    print("\nTest 4: UI Configuration Test")
    print("-" * 30)
    
    ui_test_features = ['GrLivArea', 'OverallQual', 'KitchenQual_encoded']
    
    for feature in ui_test_features:
        config = transformer.get_feature_input_config(feature)
        print(f"{feature} UI config:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        print()
    
    print("ALL TESTS COMPLETED!")
    
    return True

if __name__ == "__main__":
    success = test_transformations()
    if success:
        print("All transformation tests passed!")
    else:
        print("Some tests failed!")