"""
Test SHAP Fix - Validate realistic dollar amounts
"""

import sys
sys.path.append(r'C:\Users\victo\Documents\Data_Science_Projects\house_price_prediction_advanced\house-price-prediction-streamlit')

import pandas as pd
import pickle
from utils.enhanced_shap_explainer import EnhancedSHAPExplainer
from utils.enhanced_prediction_interface import EnhancedPredictionInterface

print("TESTING SHAP FIX - REALISTIC DOLLAR AMOUNTS")
print("=" * 60)

# Load model and data
with open(r'C:\Users\victo\Documents\Data_Science_Projects\house_price_prediction_advanced\house-price-prediction-streamlit\models\best_model.pkl', 'rb') as f:
    model = pickle.load(f)

processed_data = pd.read_csv(r'C:\Users\victo\Documents\Data_Science_Projects\house_price_prediction_advanced\house-price-prediction-streamlit\data\final_train_prepared.csv')

# Initialize interfaces
shap_explainer = EnhancedSHAPExplainer()
quick_interface = EnhancedPredictionInterface()

# Test inputs
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

print(f"Predicted Price: ${prediction_result['predicted_price']:,.0f}")

# Test SHAP explanation
shap_data = shap_explainer.create_individual_shap_explanation(
    model, prediction_result['feature_vector'], test_inputs, prediction_result['predicted_price']
)

if shap_data.get('success'):
    print(f"\nBase Market Price: ${shap_data['base_value']*1000:,.0f}")
    print(f"Predicted Price: ${shap_data['predicted_value']*1000:,.0f}")
    adjustment = shap_data['predicted_value'] - shap_data['base_value']
    print(f"Total Adjustment: ${adjustment*1000:+,.0f}")
    
    print(f"\nTop Feature Impacts:")
    for i, (feature, impact) in enumerate(list(shap_data['feature_importance'].items())[:5]):
        impact_dollars = impact * 1000
        print(f"  {i+1}. {feature}: ${impact_dollars:+,.0f}")
    
    # Validate realistic ranges
    base_value = shap_data['base_value'] * 1000
    predicted_value = shap_data['predicted_value'] * 1000
    
    print(f"\nValidation:")
    if 50000 <= base_value <= 500000:
        print(f"  ✓ Base value realistic: ${base_value:,.0f}")
    else:
        print(f"  ✗ Base value unrealistic: ${base_value:,.0f}")
    
    if 50000 <= predicted_value <= 800000:
        print(f"  ✓ Predicted value realistic: ${predicted_value:,.0f}")
    else:
        print(f"  ✗ Predicted value unrealistic: ${predicted_value:,.0f}")
    
    # Check individual impacts are reasonable
    max_impact = max(abs(impact * 1000) for impact in shap_data['feature_importance'].values())
    if max_impact < 100000:  # No single feature should impact more than $100k
        print(f"  ✓ Feature impacts reasonable (max: ${max_impact:,.0f})")
    else:
        print(f"  ⚠ Large feature impact detected: ${max_impact:,.0f}")

else:
    print(f"SHAP analysis failed: {shap_data.get('error', 'Unknown error')}")

print("\nSHAP Fix Testing Complete!")