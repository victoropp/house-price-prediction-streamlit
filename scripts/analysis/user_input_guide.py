"""
HOUSE PRICE PREDICTION - USER INPUT TRANSFORMATION GUIDE
=======================================================

This guide provides the exact transformation formulas and practical input ranges 
for converting between user-friendly values and the model's normalized inputs.
"""

import numpy as np

# =============================================================================
# TRANSFORMATION MAPPINGS
# =============================================================================

# Min-Max Normalized Features (0-1 range)
MIN_MAX_FEATURES = {
    'YearBuilt': {
        'min': 1872, 'max': 2010,
        'description': 'Year the house was built',
        'practical_range': (1900, 2010),
        'typical_values': [1950, 1980, 2000, 2005],
        'formula': 'real_value = 1872 + normalized_value * (2010 - 1872)'
    },
    'LotArea': {
        'min': 1300, 'max': 215245, 
        'description': 'Lot size in square feet',
        'practical_range': (5000, 20000),
        'typical_values': [7500, 10000, 12000, 15000],
        'formula': 'real_value = 1300 + normalized_value * (215245 - 1300)'
    },
    'GrLivArea': {
        'min': 334, 'max': 5642,
        'description': 'Above ground living area in square feet', 
        'practical_range': (1000, 3000),
        'typical_values': [1200, 1500, 2000, 2500],
        'formula': 'real_value = 334 + normalized_value * (5642 - 334)'
    },
    'TotalBsmtSF': {
        'min': 0, 'max': 6110,
        'description': 'Total basement area in square feet',
        'practical_range': (0, 2000), 
        'typical_values': [0, 800, 1200, 1500],
        'formula': 'real_value = 0 + normalized_value * (6110 - 0)'
    },
    '1stFlrSF': {
        'min': 334, 'max': 4692,
        'description': 'First floor square feet',
        'practical_range': (800, 2000),
        'typical_values': [1000, 1200, 1500, 1800],
        'formula': 'real_value = 334 + normalized_value * (4692 - 334)'
    },
    '2ndFlrSF': {
        'min': 0, 'max': 2065,
        'description': 'Second floor square feet',
        'practical_range': (0, 1500),
        'typical_values': [0, 600, 800, 1000],
        'formula': 'real_value = 0 + normalized_value * (2065 - 0)'
    }
}

# Categorical Features (not normalized, original scales)
CATEGORICAL_FEATURES = {
    'OverallQual': {
        'min': 1, 'max': 10,
        'description': 'Overall material and finish quality',
        'practical_range': (4, 10),
        'typical_values': [5, 6, 7, 8],
        'scale': '1=Very Poor, 5=Average, 10=Very Excellent'
    },
    'OverallCond': {
        'min': 1, 'max': 9,  
        'description': 'Overall condition rating',
        'practical_range': (3, 8),
        'typical_values': [5, 6, 7],
        'scale': '1=Very Poor, 5=Average, 9=Very Excellent'
    },
    'BedroomAbvGr': {
        'min': 0, 'max': 8,
        'description': 'Number of bedrooms above ground',
        'practical_range': (1, 5),
        'typical_values': [2, 3, 4],
        'scale': 'Integer count'
    },
    'FullBath': {
        'min': 0, 'max': 3,
        'description': 'Number of full bathrooms',
        'practical_range': (1, 3),
        'typical_values': [1, 2, 2, 3],
        'scale': 'Integer count'
    },
    'HalfBath': {
        'min': 0, 'max': 2,
        'description': 'Number of half bathrooms', 
        'practical_range': (0, 2),
        'typical_values': [0, 1, 1, 2],
        'scale': 'Integer count'
    }
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_min_max(real_value, feature_name):
    """Convert real value to 0-1 normalized value for model input"""
    if feature_name not in MIN_MAX_FEATURES:
        return real_value
    
    feature_info = MIN_MAX_FEATURES[feature_name]
    min_val = feature_info['min']
    max_val = feature_info['max']
    
    # Clamp to valid range
    real_value = max(min_val, min(max_val, real_value))
    
    # Min-max normalize 
    normalized = (real_value - min_val) / (max_val - min_val)
    return normalized

def denormalize_min_max(normalized_value, feature_name):
    """Convert 0-1 normalized value back to real value for display"""
    if feature_name not in MIN_MAX_FEATURES:
        return normalized_value
        
    feature_info = MIN_MAX_FEATURES[feature_name]
    min_val = feature_info['min']
    max_val = feature_info['max']
    
    real_value = min_val + normalized_value * (max_val - min_val)
    return int(round(real_value))

# =============================================================================
# PRACTICAL USER INPUT RECOMMENDATIONS
# =============================================================================

def get_practical_ranges():
    """Get recommended input ranges for UI sliders/inputs"""
    
    recommendations = {}
    
    # Min-max normalized features  
    for feature, info in MIN_MAX_FEATURES.items():
        recommendations[feature] = {
            'type': 'slider',
            'min': info['practical_range'][0],
            'max': info['practical_range'][1], 
            'default': info['typical_values'][1],  # Second typical value as default
            'step': 100 if 'SF' in feature or feature == 'LotArea' else 1,
            'description': info['description'],
            'examples': info['typical_values']
        }
    
    # Categorical features
    for feature, info in CATEGORICAL_FEATURES.items():
        recommendations[feature] = {
            'type': 'slider' if 'Bedroom' in feature or 'Bath' in feature else 'selectbox',
            'min': info['practical_range'][0],
            'max': info['practical_range'][1],
            'default': info['typical_values'][0],
            'step': 1,
            'description': info['description'],
            'scale': info['scale'],
            'examples': info['typical_values']
        }
    
    return recommendations

# =============================================================================
# VALIDATION AND EXAMPLES
# =============================================================================

def validate_and_demo():
    """Demonstrate the transformations with real examples"""
    
    print("HOUSE PRICE PREDICTION - TRANSFORMATION VALIDATION")
    print("=" * 60)
    
    # Test min-max transformations
    print("\n1. MIN-MAX NORMALIZATION EXAMPLES:")
    test_cases = [
        ('YearBuilt', 1990),
        ('LotArea', 10000), 
        ('GrLivArea', 1800),
        ('TotalBsmtSF', 1000),
        ('1stFlrSF', 1200),
        ('2ndFlrSF', 800)
    ]
    
    for feature, real_value in test_cases:
        normalized = normalize_min_max(real_value, feature)
        denormalized = denormalize_min_max(normalized, feature)
        info = MIN_MAX_FEATURES[feature]
        
        print(f"\n{feature}:")
        print(f"  Real Value: {real_value:,} {info['description'].lower()}")
        print(f"  Normalized: {normalized:.4f}")
        print(f"  Back to Real: {denormalized:,}")
        print(f"  Range: {info['min']:,} - {info['max']:,}")
    
    print(f"\n\n2. CATEGORICAL FEATURES (no transformation needed):")
    for feature, info in CATEGORICAL_FEATURES.items():
        print(f"\n{feature}: {info['description']}")
        print(f"  Range: {info['min']} - {info['max']}")
        print(f"  Scale: {info['scale']}")
        print(f"  Typical: {info['typical_values']}")
    
    print(f"\n\n3. RECOMMENDED UI INPUT RANGES:")
    recommendations = get_practical_ranges()
    
    for feature, rec in recommendations.items():
        print(f"\n{feature}:")
        print(f"  Type: {rec['type']}")
        print(f"  Range: {rec['min']:,} - {rec['max']:,}")
        print(f"  Default: {rec['default']:,}")
        print(f"  Description: {rec['description']}")

if __name__ == "__main__":
    validate_and_demo()