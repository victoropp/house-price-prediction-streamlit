import pandas as pd
import numpy as np

# Load original and final datasets
original_df = pd.read_csv(r'C:\Users\victo\Documents\Data_Science_Projects\house_price_prediction_advanced\dataset\train.csv')
phase3_df = pd.read_csv(r'C:\Users\victo\Documents\Data_Science_Projects\house_price_prediction_advanced\preprocessing\phase3_feature_engineering\engineered_train_phase3.csv')
final_df = pd.read_csv(r'C:\Users\victo\Documents\Data_Science_Projects\house_price_prediction_advanced\house-price-prediction-streamlit\data\final_train_prepared.csv')

# Additional normalized features to analyze
additional_features = ['GarageArea', 'TotRmsAbvGrd', 'MasVnrArea', 'OpenPorchSF', 'WoodDeckSF']

print('=== ADDITIONAL NORMALIZED FEATURES ANALYSIS ===\n')

for feature in additional_features:
    print(f"--- {feature} ---")
    
    # Check original values
    if feature in original_df.columns:
        orig_min = original_df[feature].min()
        orig_max = original_df[feature].max()
        orig_mean = original_df[feature].mean()
        print(f"Original: {orig_min} - {orig_max} (mean: {orig_mean:.0f})")
    
    # Check phase 3 values (after feature engineering)
    if feature in phase3_df.columns:
        p3_min = phase3_df[feature].min()
        p3_max = phase3_df[feature].max()
        p3_mean = phase3_df[feature].mean()
        print(f"Phase 3: {p3_min} - {p3_max} (mean: {p3_mean:.0f})")
        
        # Check final normalized values
        if feature in final_df.columns:
            final_min = final_df[feature].min()
            final_max = final_df[feature].max()
            final_mean = final_df[feature].mean()
            print(f"Final: {final_min:.4f} - {final_max:.4f} (mean: {final_mean:.4f})")
            
            # If normalized, show transformation
            if final_min == 0.0 and final_max == 1.0:
                print(f"TRANSFORMATION: real_value = {p3_min} + normalized_value * ({p3_max} - {p3_min})")
                
                # Calculate practical ranges
                if feature == 'GarageArea':
                    print(f"Practical range: 0 - 800 sq ft (typical: 400-600)")
                elif feature == 'TotRmsAbvGrd':
                    print(f"Practical range: 3 - 10 rooms (typical: 5-8)")
                elif feature == 'MasVnrArea':
                    print(f"Practical range: 0 - 500 sq ft (typical: 0, 100-300)")
                elif feature == 'OpenPorchSF':
                    print(f"Practical range: 0 - 200 sq ft (typical: 0, 25-100)")
                elif feature == 'WoodDeckSF':
                    print(f"Practical range: 0 - 500 sq ft (typical: 0, 100-300)")
                    
                # Show example transformation
                example_norm = final_mean
                example_real = p3_min + example_norm * (p3_max - p3_min)
                print(f"Example: {example_norm:.4f} (norm) = {example_real:.0f} (real)")
    
    print()

# Non-normalized features that are also important
non_normalized = ['GarageCars', 'Fireplaces']
print('\n=== NON-NORMALIZED FEATURES ===\n')

for feature in non_normalized:
    if feature in final_df.columns:
        min_val = final_df[feature].min()
        max_val = final_df[feature].max()
        mean_val = final_df[feature].mean()
        print(f"{feature}: {min_val:.0f} - {max_val:.0f} (mean: {mean_val:.1f})")
        
        if feature == 'GarageCars':
            print(f"  Description: Number of cars garage can hold")
            print(f"  Practical range: 0-3 (typical: 1-2)")
        elif feature == 'Fireplaces':
            print(f"  Description: Number of fireplaces")
            print(f"  Practical range: 0-2 (typical: 0-1)")
        print()

print('\n=== RECOMMENDED UI SETTINGS FOR ADDITIONAL FEATURES ===\n')

# Provide UI recommendations
ui_settings = {
    'GarageArea': {
        'min_max_range': (0, 1488),
        'practical_range': (0, 800),
        'default': 400,
        'step': 50,
        'description': 'Garage area in square feet'
    },
    'TotRmsAbvGrd': {
        'min_max_range': (2, 14), 
        'practical_range': (4, 10),
        'default': 6,
        'step': 1,
        'description': 'Total rooms above ground (excluding bathrooms)'
    },
    'MasVnrArea': {
        'min_max_range': (0, 1600),
        'practical_range': (0, 400),
        'default': 0,
        'step': 50,
        'description': 'Masonry veneer area in square feet'
    },
    'OpenPorchSF': {
        'min_max_range': (0, 547),
        'practical_range': (0, 200),
        'default': 0,
        'step': 25,
        'description': 'Open porch area in square feet'
    },
    'WoodDeckSF': {
        'min_max_range': (0, 857),
        'practical_range': (0, 400),
        'default': 0,
        'step': 50,
        'description': 'Wood deck area in square feet'
    },
    'GarageCars': {
        'min_max_range': (0, 4),
        'practical_range': (0, 3),
        'default': 2,
        'step': 1,
        'description': 'Size of garage in car capacity'
    },
    'Fireplaces': {
        'min_max_range': (0, 3),
        'practical_range': (0, 2),
        'default': 0,
        'step': 1,
        'description': 'Number of fireplaces'
    }
}

for feature, settings in ui_settings.items():
    print(f"{feature}:")
    print(f"  UI Type: slider")
    print(f"  Range: {settings['practical_range'][0]} - {settings['practical_range'][1]}")
    print(f"  Default: {settings['default']}")
    print(f"  Step: {settings['step']}")
    print(f"  Description: {settings['description']}")
    print()