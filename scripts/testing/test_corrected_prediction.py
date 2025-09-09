"""
Test the corrected prediction with Box-Cox transformation
"""

import sys
sys.path.append(r'C:\Users\victo\Documents\Data_Science_Projects\house_price_prediction_advanced\house-price-prediction-streamlit')

import pandas as pd
import numpy as np
import pickle
from scipy.special import inv_boxcox
from utils.complete_transformations import CompleteHouseTransformations

# Load model and data
print("Loading model and data...")
with open(r'C:\Users\victo\Documents\Data_Science_Projects\house_price_prediction_advanced\house-price-prediction-streamlit\models\best_model.pkl', 'rb') as f:
    model = pickle.load(f)

processed_data = pd.read_csv(r'C:\Users\victo\Documents\Data_Science_Projects\house_price_prediction_advanced\house-price-prediction-streamlit\data\final_train_prepared.csv')
original_data = pd.read_csv(r'C:\Users\victo\Documents\Data_Science_Projects\house_price_prediction_advanced\dataset\train.csv')

# Initialize transformer
transformer = CompleteHouseTransformations()

print(f"\nModel loaded: {type(model)}")
print(f"Data shapes: Processed {processed_data.shape}, Original {original_data.shape}")

# Test with realistic house inputs
print("\n" + "="*60)
print("TESTING CORRECTED PREDICTION WITH REALISTIC INPUTS")
print("="*60)

# Define realistic house scenarios
test_scenarios = [
    {
        'name': 'Modest Family Home',
        'YearBuilt': 1995,
        'LotArea': 8500,
        'GrLivArea': 1800,
        'TotalBsmtSF': 900,
        '1stFlrSF': 1100,
        '2ndFlrSF': 700,
        'OverallQual': 6,
        'OverallCond': 6,
        'BedroomAbvGr': 3,
        'FullBath': 2,
        'HalfBath': 1,
        'GarageCars': 2,
        'GarageArea': 500,
        'Fireplaces': 1
    },
    {
        'name': 'Modern Quality Home',
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
        'Fireplaces': 2
    }
]

lambda_param = -0.07693211157738546

for scenario in test_scenarios:
    name = scenario.pop('name')
    print(f"\n{name}:")
    print("-" * 30)
    
    # Show user inputs
    print("User Inputs:")
    for feature, value in scenario.items():
        friendly_name = transformer.get_friendly_name(feature)
        print(f"  • {friendly_name}: {value}")
    
    # Transform inputs
    print("\nTransformed Inputs (Sample):")
    feature_vector = processed_data.iloc[0].copy()
    if 'SalePrice_transformed' in feature_vector.index:
        feature_vector = feature_vector.drop('SalePrice_transformed')
    
    for feature, value in scenario.items():
        if feature in feature_vector.index:
            transformed = transformer.transform_user_input(feature, value)
            feature_vector[feature] = transformed
            print(f"  • {feature}: {value} -> {transformed:.4f}")
    
    # Make prediction
    try:
        prediction_transformed = model.predict([feature_vector.values])[0]
        
        # Apply correct Box-Cox inverse transformation
        predicted_price = inv_boxcox(prediction_transformed, lambda_param) - 1
        
        print(f"\nPrediction Results:")
        print(f"  • Model Output: {prediction_transformed:.4f}")
        print(f"  • Predicted Price: ${predicted_price:,.0f}")
        
        # Validation check - compare with original data range
        orig_min = original_data['SalePrice'].min()
        orig_max = original_data['SalePrice'].max()
        orig_mean = original_data['SalePrice'].mean()
        
        if orig_min <= predicted_price <= orig_max:
            print(f"  • ✓ Price within valid range (${orig_min:,} - ${orig_max:,})")
        else:
            print(f"  • ⚠️ Price outside expected range")
        
        if predicted_price >= 50000:  # Reasonable minimum for a house
            print(f"  • ✓ Realistic house price")
        else:
            print(f"  • ⚠️ Price seems too low for a house")
            
    except Exception as e:
        print(f"  • ❌ Prediction failed: {e}")
    
    print()

# Test with actual data sample
print("="*60)
print("VALIDATION WITH ACTUAL DATA SAMPLE")
print("="*60)

# Use first 3 rows of actual data
for i in range(3):
    actual_price = original_data['SalePrice'].iloc[i]
    processed_sample = processed_data.iloc[i]
    actual_transformed = processed_sample['SalePrice_transformed']
    
    # Remove target from features
    feature_sample = processed_sample.drop('SalePrice_transformed')
    
    # Make prediction
    predicted_transformed = model.predict([feature_sample.values])[0]
    predicted_price = inv_boxcox(predicted_transformed, lambda_param) - 1
    
    print(f"\nSample {i+1}:")
    print(f"  Actual Price: ${actual_price:,}")
    print(f"  Predicted Price: ${predicted_price:,.0f}")
    print(f"  Error: ${abs(predicted_price - actual_price):,.0f} ({abs(predicted_price - actual_price)/actual_price*100:.1f}%)")
    print(f"  Model outputs: {actual_transformed:.4f} (actual) vs {predicted_transformed:.4f} (predicted)")

print(f"\n{'='*60}")
print("CORRECTED TRANSFORMATION TESTING COMPLETE")
print(f"{'='*60}")
print("The app should now show realistic house prices!")
print("Range should be approximately $50,000 - $800,000")