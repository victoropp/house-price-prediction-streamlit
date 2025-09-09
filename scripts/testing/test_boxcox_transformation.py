import pandas as pd
import numpy as np
from scipy.special import inv_boxcox
from scipy.stats import boxcox

# Load data
orig = pd.read_csv('dataset/train.csv')
processed = pd.read_csv('house-price-prediction-streamlit/data/final_train_prepared.csv')

# Test Box-Cox transformation
lambda_param = -0.07693211157738546

print("Testing Box-Cox transformation:")
print("=" * 50)

# Test forward transformation
for i in range(5):
    original_price = orig['SalePrice'].iloc[i]
    processed_price = processed['SalePrice_transformed'].iloc[i]
    
    # Apply Box-Cox transformation
    boxcox_transformed = boxcox(original_price + 1, lmbda=lambda_param)
    
    print(f"Original: ${original_price:,}")
    print(f"Box-Cox:   {boxcox_transformed:.6f}")
    print(f"Processed: {processed_price:.6f}")
    print(f"Match: {abs(boxcox_transformed - processed_price) < 0.001}")
    print()

print("Testing reverse transformation:")
print("=" * 50)

# Test reverse transformation
test_predictions = [7.93227, 7.87793, 8.0, 8.2, 7.5]

for pred in test_predictions:
    # Apply inverse Box-Cox
    reversed_price = inv_boxcox(pred, lambda_param) - 1
    print(f"Model output: {pred:.5f} -> Price: ${reversed_price:,.0f}")

# Test with problematic prediction from app
print("\nTesting app prediction:")
print("=" * 30)
app_prediction = 7.9043  # From our test earlier
correct_price = inv_boxcox(app_prediction, lambda_param) - 1
print(f"App prediction: {app_prediction:.4f}")
print(f"Correct price: ${correct_price:,.0f}")

print(f"\nRange analysis:")
min_pred = processed['SalePrice_transformed'].min()
max_pred = processed['SalePrice_transformed'].max()
min_price = inv_boxcox(min_pred, lambda_param) - 1
max_price = inv_boxcox(max_pred, lambda_param) - 1
print(f"Model range: {min_pred:.4f} - {max_pred:.4f}")
print(f"Price range: ${min_price:,.0f} - ${max_price:,.0f}")