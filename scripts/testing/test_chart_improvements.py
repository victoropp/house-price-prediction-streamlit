"""
Test Chart Improvements
Verify that the enhanced feature importance chart is working
"""

import sys
sys.path.append(r'C:\Users\victo\Documents\Data_Science_Projects\house_price_prediction_advanced\house-price-prediction-streamlit')

from utils.visualization_utils import ProfessionalVisualizations

print("Testing Chart Improvements...")

# Initialize visualizer
viz = ProfessionalVisualizations()

# Test user-friendly feature name conversion
test_features = [
    'OverallQual',
    'GrLivArea', 
    'TotalBsmtSF',
    'ExterQual_encoded',
    'YearBuilt',
    'Neighborhood_encoded'
]

print("\nUser-Friendly Name Conversion Test:")
for feature in test_features:
    friendly_name = viz._get_user_friendly_feature_name(feature)
    print(f"  {feature} -> {friendly_name}")

# Test chart creation
test_data = [
    ('OverallQual', 0.3456),
    ('GrLivArea', 0.2891),
    ('TotalBsmtSF', 0.1245),
    ('ExterQual_encoded', 0.0987),
    ('YearBuilt', 0.0756)
]

print("\nChart Creation Test:")
try:
    fig = viz.create_feature_importance_chart(
        importance_data=test_data,
        title="Test Feature Importance",
        max_features=5
    )
    print("  Chart created successfully!")
    print(f"  Chart has {len(fig.data)} data traces")
    print(f"  Title: {fig.layout.title.text}")
    
    # Check if friendly names are in the chart
    if hasattr(fig.data[0], 'y'):
        y_values = list(fig.data[0].y)
        print(f"  Y-axis labels: {y_values}")
        
        # Verify user-friendly names are being used
        friendly_found = any('Overall Quality Rating' in str(label) for label in y_values)
        if friendly_found:
            print("  SUCCESS: User-friendly names detected in chart!")
        else:
            print("  WARNING: May still be using technical names")
    
except Exception as e:
    print(f"  ERROR: Chart creation failed: {e}")

print("\nTest complete!")
print("If successful, the frontend should now show:")
print("  - User-friendly feature names (e.g., 'Overall Quality Rating')")
print("  - Color-coded importance levels")  
print("  - Rank badges (#1, #2, #3)")
print("  - Enhanced hover tooltips")
print("  - Better visual design")