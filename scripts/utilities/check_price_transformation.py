import pandas as pd
import numpy as np

# Load original data
orig = pd.read_csv('dataset/train.csv')
processed = pd.read_csv('house-price-prediction-streamlit/data/final_train_prepared.csv')

print('Original SalePrice examples:')
print(orig['SalePrice'].head(5).values)
print(f'Range: ${orig["SalePrice"].min():,.0f} - ${orig["SalePrice"].max():,.0f}')

print('\nTransformed SalePrice examples:')
print(processed['SalePrice_transformed'].head(5).values)
print(f'Range: {processed["SalePrice_transformed"].min():.4f} - {processed["SalePrice_transformed"].max():.4f}')

# Test if it's log transformation
print('\nTesting log transformation:')
for i in range(5):
    original = orig['SalePrice'].iloc[i]
    transformed = processed['SalePrice_transformed'].iloc[i]
    log_original = np.log(original)
    print(f'Original: ${original:,} | Log: {log_original:.6f} | Processed: {transformed:.6f} | Match: {abs(log_original - transformed) < 0.001}')

# Test reverse transformation
print('\nTesting reverse transformation:')
sample_transformed = 7.93227570796686  # From our app
reverse_transformed = np.exp(sample_transformed)
print(f'Model output: {sample_transformed:.6f}')
print(f'Reverse (exp): ${reverse_transformed:,.0f}')

# Test with model prediction range
print('\nModel prediction range analysis:')
min_pred = processed['SalePrice_transformed'].min()
max_pred = processed['SalePrice_transformed'].max()
print(f'Min model output: {min_pred:.4f} -> ${np.exp(min_pred):,.0f}')
print(f'Max model output: {max_pred:.4f} -> ${np.exp(max_pred):,.0f}')

# Check the specific prediction that gave $2,806
problematic_pred = 1.0315  # Approximate value that would give ~$2,806
print(f'\nProblematic prediction analysis:')
print(f'Prediction: {problematic_pred:.4f} -> ${np.exp(problematic_pred):,.0f}')