import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv(r'C:\Users\victo\Documents\Data_Science_Projects\house_price_prediction_advanced\house-price-prediction-streamlit\data\final_train_prepared.csv')

print('Dataset shape:', df.shape)
print('\nColumn categories analysis:')

# Get all column names
cols = df.columns.tolist()

# Categorize features
min_max_features = []
log_transformed_features = []
encoded_categorical = []
onehot_encoded = []
regular_features = []

for col in cols:
    if col.endswith('_transformed'):
        log_transformed_features.append(col)
    elif col.endswith('_encoded'):
        encoded_categorical.append(col)
    elif any(pattern in col for pattern in ['_', ' ', '(', ')']):
        # Check if it's likely a one-hot encoded feature
        if col not in ['1stFlrSF', '2ndFlrSF', '3SsnPorch']:  # These are actual features
            onehot_encoded.append(col)
        else:
            regular_features.append(col)
    else:
        regular_features.append(col)

# Analyze regular features (likely min-max normalized)
print(f'\nRegular features ({len(regular_features)}):')
for feature in regular_features:
    min_val = df[feature].min()
    max_val = df[feature].max()
    # Check if it looks like min-max normalized (0-1 range)
    if min_val >= 0 and max_val <= 1 and max_val > min_val:
        min_max_features.append(feature)
        print(f'  {feature}: {min_val:.4f} - {max_val:.4f} [MIN-MAX NORMALIZED]')
    else:
        print(f'  {feature}: {min_val:.4f} - {max_val:.4f}')

print(f'\nLog-transformed features ({len(log_transformed_features)}):')
for feature in log_transformed_features:
    min_val = df[feature].min()
    max_val = df[feature].max()
    print(f'  {feature}: {min_val:.4f} - {max_val:.4f}')

print(f'\nEncoded categorical features ({len(encoded_categorical)}):')
for feature in encoded_categorical:
    min_val = df[feature].min()
    max_val = df[feature].max()
    unique_vals = df[feature].nunique()
    print(f'  {feature}: {min_val:.1f} - {max_val:.1f} ({unique_vals} unique values)')

print(f'\nOne-hot encoded features ({len(onehot_encoded)}):')
for feature in onehot_encoded:
    unique_vals = sorted(df[feature].unique())
    print(f'  {feature}: {unique_vals}')

print(f'\n\n=== KEY FINDINGS ===')
print(f'Min-Max Normalized Features: {len(min_max_features)}')
print(f'Log-Transformed Features: {len(log_transformed_features)}')
print(f'Encoded Categorical Features: {len(encoded_categorical)}')
print(f'One-Hot Encoded Features: {len(onehot_encoded)}')

# Focus on key features users care about
key_features = ['YearBuilt', 'LotArea', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 
                'OverallQual', 'OverallCond', 'BedroomAbvGr', 'FullBath', 'HalfBath']

print(f'\n\n=== KEY USER FEATURES ANALYSIS ===')
for feature in key_features:
    if feature in df.columns:
        min_val = df[feature].min()
        max_val = df[feature].max()
        mean_val = df[feature].mean()
        print(f'{feature}: {min_val:.4f} - {max_val:.4f} (mean: {mean_val:.4f})')
    else:
        print(f'{feature}: NOT FOUND in dataset')