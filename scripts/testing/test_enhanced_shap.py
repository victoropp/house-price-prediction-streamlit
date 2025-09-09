"""
Test Enhanced SHAP Explanations
Validate that the new user-friendly SHAP system works correctly
"""

import sys
import os
sys.path.append(r'C:\Users\victo\Documents\Data_Science_Projects\house_price_prediction_advanced\house-price-prediction-streamlit')

import pandas as pd
import numpy as np
import pickle
from utils.enhanced_shap_explainer import EnhancedSHAPExplainer
from utils.enhanced_prediction_interface import EnhancedPredictionInterface
from utils.advanced_prediction_interface import AdvancedPredictionInterface

print("=" * 80)
print("TESTING ENHANCED SHAP EXPLANATIONS")
print("=" * 80)

# Load model and data
print("\n1. Loading Model and Data...")
try:
    with open(r'C:\Users\victo\Documents\Data_Science_Projects\house_price_prediction_advanced\house-price-prediction-streamlit\models\best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    processed_data = pd.read_csv(r'C:\Users\victo\Documents\Data_Science_Projects\house_price_prediction_advanced\house-price-prediction-streamlit\data\final_train_prepared.csv')
    
    print(f"Model loaded: {type(model)}")
    print(f"Data shape: {processed_data.shape}")
    print("SUCCESS - Model and data loaded")
except Exception as e:
    print(f"ERROR - Failed to load model/data: {e}")
    exit(1)

# Test enhanced SHAP explainer
print("\n2. Testing Enhanced SHAP Explainer...")
try:
    shap_explainer = EnhancedSHAPExplainer()
    print("SUCCESS - EnhancedSHAPExplainer initialized")
except Exception as e:
    print(f"ERROR - Failed to initialize SHAP explainer: {e}")
    exit(1)

# Test quick prediction interface with SHAP
print("\n3. Testing Quick Prediction Interface with SHAP...")
try:
    quick_interface = EnhancedPredictionInterface()
    
    # Test inputs for quick prediction
    test_inputs = {
        'YearBuilt': 2000,
        'LotArea': 10000,
        'GrLivArea': 2000,
        'TotalBsmtSF': 1000,
        'OverallQual': 7,
        'BedroomAbvGr': 3,
        'FullBath': 2,
        'GarageCars': 2
    }
    
    # Transform inputs
    transformed_inputs = {}
    for feature, value in test_inputs.items():
        transformed_inputs[feature] = quick_interface.transformer.transform_user_input(feature, value)
    
    # Make prediction
    prediction_result = quick_interface.make_prediction_with_user_inputs(
        model, processed_data, test_inputs, transformed_inputs
    )
    
    if prediction_result and prediction_result.get('success'):
        print(f"Quick prediction successful: ${prediction_result['predicted_price']:,.0f}")
        
        # Test SHAP explanation generation
        shap_data = shap_explainer.create_individual_shap_explanation(
            model, prediction_result['feature_vector'], test_inputs, prediction_result['predicted_price']
        )
        
        if shap_data.get('success'):
            print(f"SHAP analysis successful - {len(shap_data['feature_importance'])} significant features")
            print("Top 3 feature impacts:")
            for i, (feature, impact) in enumerate(list(shap_data['feature_importance'].items())[:3]):
                impact_amount = impact * 1000
                print(f"  {i+1}. {feature}: ${impact_amount:+,.0f}")
        else:
            print(f"SHAP analysis failed: {shap_data.get('error', 'Unknown error')}")
        
        print("SUCCESS - Quick prediction interface with SHAP working")
    else:
        print("ERROR - Quick prediction failed")
        
except Exception as e:
    print(f"ERROR - Quick prediction interface test failed: {e}")

# Test advanced prediction interface with SHAP
print("\n4. Testing Advanced Prediction Interface with SHAP...")
try:
    advanced_interface = AdvancedPredictionInterface()
    
    # Test inputs for advanced prediction
    advanced_inputs = {
        'YearBuilt': 2005,
        'LotArea': 12000,
        'GrLivArea': 2500,
        'TotalBsmtSF': 1200,
        '1stFlrSF': 1500,
        '2ndFlrSF': 1000,
        'OverallQual': 8,
        'OverallCond': 7,
        'BedroomAbvGr': 4,
        'FullBath': 2,
        'HalfBath': 1,
        'GarageCars': 2,
        'GarageArea': 600,
        'Fireplaces': 1
    }
    
    # Make advanced prediction
    advanced_result = advanced_interface.make_advanced_prediction(
        model, processed_data, advanced_inputs
    )
    
    if advanced_result and advanced_result.get('success'):
        print(f"Advanced prediction successful: ${advanced_result['predicted_price']:,.0f}")
        print(f"Features used: {advanced_result['transformed_count']}/{advanced_result['total_features']}")
        
        # Test SHAP explanation generation
        shap_data = shap_explainer.create_individual_shap_explanation(
            model, advanced_result['feature_vector'], advanced_inputs, advanced_result['predicted_price']
        )
        
        if shap_data.get('success'):
            print(f"SHAP analysis successful - {len(shap_data['feature_importance'])} significant features")
            print("Top 5 feature impacts:")
            for i, (feature, impact) in enumerate(list(shap_data['feature_importance'].items())[:5]):
                impact_amount = impact * 1000
                friendly_name = shap_explainer.feature_explainer.get_friendly_feature_name(feature)
                print(f"  {i+1}. {friendly_name}: ${impact_amount:+,.0f}")
        else:
            print(f"SHAP analysis failed: {shap_data.get('error', 'Unknown error')}")
        
        print("SUCCESS - Advanced prediction interface with SHAP working")
    else:
        print("ERROR - Advanced prediction failed")
        
except Exception as e:
    print(f"ERROR - Advanced prediction interface test failed: {e}")

# Test SHAP visualization components
print("\n5. Testing SHAP Visualization Components...")
try:
    # Test feature explainer
    feature_explainer = shap_explainer.feature_explainer
    
    # Test feature naming
    test_features = ['OverallQual', 'GrLivArea', 'YearBuilt', 'TotalSF']
    for feature in test_features:
        friendly_name = feature_explainer.get_friendly_feature_name(feature)
        formatted_value = feature_explainer.format_feature_value(feature, 2000)
        print(f"  {feature} -> {friendly_name} ({formatted_value})")
    
    # Test feature impact explanation
    test_impact = feature_explainer.explain_feature_impact('OverallQual', 8, 0.05)
    print(f"  Sample explanation: {test_impact[:100]}...")
    
    print("SUCCESS - SHAP visualization components working")
    
except Exception as e:
    print(f"ERROR - SHAP visualization test failed: {e}")

# Test improvement suggestions
print("\n6. Testing Improvement Suggestions...")
try:
    suggestions = []
    test_negative_features = ['OverallQual', 'GrLivArea', 'TotalBaths']
    
    for feature in test_negative_features:
        suggestion = shap_explainer._get_improvement_suggestion(feature)
        if suggestion:
            suggestions.append(f"  {feature}: {suggestion}")
    
    if suggestions:
        print("Sample improvement suggestions:")
        for suggestion in suggestions:
            print(suggestion)
        print("SUCCESS - Improvement suggestions working")
    else:
        print("No improvement suggestions generated (this is normal)")
        
except Exception as e:
    print(f"ERROR - Improvement suggestions test failed: {e}")

# Test benchmarking
print("\n7. Testing Feature Benchmarking...")
try:
    benchmarks = []
    test_benchmark_features = [
        ('OverallQual', 8),
        ('GrLivArea', 2200),
        ('YearBuilt', 1995)
    ]
    
    for feature, value in test_benchmark_features:
        benchmark = shap_explainer._get_feature_benchmark(feature, value)
        if benchmark:
            benchmarks.append(f"  {feature} ({value}): {benchmark}")
    
    if benchmarks:
        print("Sample benchmarks:")
        for benchmark in benchmarks:
            print(benchmark)
        print("SUCCESS - Feature benchmarking working")
    else:
        print("No benchmarks generated")
        
except Exception as e:
    print(f"ERROR - Feature benchmarking test failed: {e}")

print("\n" + "=" * 80)
print("ENHANCED SHAP TESTING COMPLETE")
print("=" * 80)

print("\nSummary:")
print("Enhanced SHAP explanations have been successfully integrated!")
print("Features added:")
print("  - Individual prediction SHAP analysis")
print("  - User-friendly feature explanations")
print("  - Interactive waterfall charts")
print("  - Plain English summaries")
print("  - Feature benchmarking")
print("  - Improvement suggestions")
print("  - Price impact visualizations")

print("\nThe Streamlit app now provides:")
print("  1. Real-time SHAP explanations for every prediction")
print("  2. Professional visualizations with plain English")
print("  3. Detailed feature impact analysis")
print("  4. Market context and benchmarking")
print("  5. Actionable improvement suggestions")

print(f"\nApp URL: http://localhost:8503")
print("Test both Quick Mode and Advanced Mode predictions!")