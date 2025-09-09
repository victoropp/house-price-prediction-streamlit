"""
Real-World Feature Transformations
Bidirectional transformations between user-friendly values and model-ready normalized values
Based on comprehensive analysis of the house price prediction dataset
"""

import numpy as np
from typing import Dict, Any, Tuple, Union

class RealWorldTransformations:
    """
    Handles transformations between user-friendly real-world values 
    and the normalized 0-1 values the model expects
    """
    
    def __init__(self):
        # Min-Max normalization parameters for key features
        # Format: feature_name: {'min': min_value, 'max': max_value, 'practical_min': practical_min, 'practical_max': practical_max}
        self.min_max_features = {
            'YearBuilt': {'min': 1872, 'max': 2010, 'practical_min': 1900, 'practical_max': 2010, 'unit': 'year'},
            'LotArea': {'min': 1300, 'max': 215245, 'practical_min': 5000, 'practical_max': 20000, 'unit': 'sq ft'},
            'GrLivArea': {'min': 334, 'max': 5642, 'practical_min': 1000, 'practical_max': 3000, 'unit': 'sq ft'},
            'TotalBsmtSF': {'min': 0, 'max': 6110, 'practical_min': 0, 'practical_max': 2000, 'unit': 'sq ft'},
            '1stFlrSF': {'min': 334, 'max': 4692, 'practical_min': 800, 'practical_max': 2000, 'unit': 'sq ft'},
            '2ndFlrSF': {'min': 0, 'max': 2065, 'practical_min': 0, 'practical_max': 1500, 'unit': 'sq ft'},
            'GarageArea': {'min': 0, 'max': 1418, 'practical_min': 0, 'practical_max': 800, 'unit': 'sq ft'},
            'TotRmsAbvGrd': {'min': 2, 'max': 14, 'practical_min': 4, 'practical_max': 10, 'unit': 'rooms'},
            'MasVnrArea': {'min': 0, 'max': 1600, 'practical_min': 0, 'practical_max': 400, 'unit': 'sq ft'},
            'OpenPorchSF': {'min': 0, 'max': 547, 'practical_min': 0, 'practical_max': 200, 'unit': 'sq ft'},
            'WoodDeckSF': {'min': 0, 'max': 857, 'practical_min': 0, 'practical_max': 400, 'unit': 'sq ft'},
            'LotFrontage': {'min': 21, 'max': 313, 'practical_min': 50, 'practical_max': 150, 'unit': 'feet'},
            'YearRemodAdd': {'min': 1950, 'max': 2010, 'practical_min': 1950, 'practical_max': 2010, 'unit': 'year'},
            'BsmtFinSF1': {'min': 0, 'max': 5644, 'practical_min': 0, 'practical_max': 1500, 'unit': 'sq ft'},
            'BsmtFinSF2': {'min': 0, 'max': 1474, 'practical_min': 0, 'practical_max': 500, 'unit': 'sq ft'},
            'BsmtUnfSF': {'min': 0, 'max': 2336, 'practical_min': 0, 'practical_max': 1500, 'unit': 'sq ft'},
            'LowQualFinSF': {'min': 0, 'max': 572, 'practical_min': 0, 'practical_max': 200, 'unit': 'sq ft'},
            'EnclosedPorch': {'min': 0, 'max': 552, 'practical_min': 0, 'practical_max': 200, 'unit': 'sq ft'},
            '3SsnPorch': {'min': 0, 'max': 508, 'practical_min': 0, 'practical_max': 200, 'unit': 'sq ft'},
            'ScreenPorch': {'min': 0, 'max': 480, 'practical_min': 0, 'practical_max': 200, 'unit': 'sq ft'},
            'MiscVal': {'min': 0, 'max': 15500, 'practical_min': 0, 'practical_max': 5000, 'unit': 'dollars'},
            'GarageYrBlt': {'min': 1900, 'max': 2010, 'practical_min': 1950, 'practical_max': 2010, 'unit': 'year'},
            'MSSubClass': {'min': 20, 'max': 190, 'practical_min': 20, 'practical_max': 190, 'unit': 'building class'}
        }
        
        # Categorical features that are already in user-friendly ranges
        self.categorical_features = {
            'OverallQual': {'min': 1, 'max': 10, 'practical_min': 4, 'practical_max': 10, 'unit': 'rating'},
            'OverallCond': {'min': 1, 'max': 9, 'practical_min': 3, 'practical_max': 8, 'unit': 'rating'},
            'BedroomAbvGr': {'min': 0, 'max': 8, 'practical_min': 1, 'practical_max': 5, 'unit': 'bedrooms'},
            'FullBath': {'min': 0, 'max': 3, 'practical_min': 1, 'practical_max': 3, 'unit': 'bathrooms'},
            'HalfBath': {'min': 0, 'max': 2, 'practical_min': 0, 'practical_max': 2, 'unit': 'half baths'},
            'GarageCars': {'min': 0, 'max': 4, 'practical_min': 0, 'practical_max': 3, 'unit': 'cars'},
            'Fireplaces': {'min': 0, 'max': 3, 'practical_min': 0, 'practical_max': 2, 'unit': 'fireplaces'},
            'BsmtFullBath': {'min': 0, 'max': 3, 'practical_min': 0, 'practical_max': 2, 'unit': 'basement baths'},
            'BsmtHalfBath': {'min': 0, 'max': 2, 'practical_min': 0, 'practical_max': 1, 'unit': 'basement half baths'},
            'KitchenAbvGr': {'min': 0, 'max': 3, 'practical_min': 1, 'practical_max': 2, 'unit': 'kitchens'},
            'MoSold': {'min': 1, 'max': 12, 'practical_min': 1, 'practical_max': 12, 'unit': 'month'},
            'YrSold': {'min': 2006, 'max': 2010, 'practical_min': 2006, 'practical_max': 2010, 'unit': 'year'}
        }

        # Quality scale mappings for encoded features
        self.quality_scales = {
            'ExterQual_encoded': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1},
            'ExterCond_encoded': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1},
            'BsmtQual_encoded': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1},
            'BsmtCond_encoded': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
            'HeatingQC_encoded': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1},
            'KitchenQual_encoded': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1},
            'FireplaceQu_encoded': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1},
            'GarageQual_encoded': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1},
            'GarageCond_encoded': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1},
            'PoolQC_encoded': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
        }

    def normalize_min_max(self, real_value: Union[int, float], feature_name: str) -> float:
        """Convert real-world value to 0-1 normalized value for model input"""
        if feature_name not in self.min_max_features:
            return real_value  # Return as-is if not a min-max feature
        
        params = self.min_max_features[feature_name]
        min_val, max_val = params['min'], params['max']
        
        # Clamp to valid range
        real_value = max(min_val, min(max_val, real_value))
        
        # Normalize to 0-1
        return (real_value - min_val) / (max_val - min_val)
    
    def denormalize_min_max(self, normalized_value: float, feature_name: str) -> Union[int, float]:
        """Convert 0-1 normalized value back to real-world value for display"""
        if feature_name not in self.min_max_features:
            return normalized_value  # Return as-is if not a min-max feature
        
        params = self.min_max_features[feature_name]
        min_val, max_val = params['min'], params['max']
        
        # Denormalize from 0-1 to real range
        real_value = min_val + normalized_value * (max_val - min_val)
        
        # Return integer for features that should be integers
        if feature_name in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'TotRmsAbvGrd', 'MSSubClass']:
            return int(round(real_value))
        
        return round(real_value, 2)
    
    def get_feature_input_config(self, feature_name: str) -> Dict[str, Any]:
        """Get Streamlit input configuration for a feature"""
        if feature_name in self.min_max_features:
            params = self.min_max_features[feature_name]
            return {
                'type': 'number_input',
                'label': self.get_friendly_feature_name(feature_name),
                'min': params['practical_min'],
                'max': params['practical_max'],
                'default': (params['practical_min'] + params['practical_max']) // 2,
                'step': self._get_step_size(feature_name),
                'help': self.get_feature_description(feature_name),
                'format': f"%d {params['unit']}" if params['unit'] in ['year', 'rooms', 'cars'] else f"%.0f {params['unit']}"
            }
        elif feature_name in self.categorical_features:
            params = self.categorical_features[feature_name]
            return {
                'type': 'selectbox',
                'options': list(range(params['practical_min'], params['practical_max'] + 1)),
                'index': (params['practical_max'] - params['practical_min']) // 2,
                'format_func': lambda x: f"{x} {params['unit']}"
            }
        elif feature_name in self.quality_scales:
            return {
                'type': 'selectbox',
                'options': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
                'index': 2,  # Default to 'TA' (Typical/Average)
                'format_func': lambda x: f"{x} ({self._quality_description(x)})"
            }
        else:
            return {
                'type': 'number_input',
                'min_value': 0,
                'max_value': 1,
                'value': 0.5,
                'step': 0.01,
                'format': "%.3f"
            }
    
    def _get_step_size(self, feature_name: str) -> Union[int, float]:
        """Get appropriate step size for different features"""
        if 'Year' in feature_name:
            return 1
        elif feature_name in ['TotRmsAbvGrd', 'MSSubClass']:
            return 1
        elif 'Area' in feature_name or 'SF' in feature_name:
            return 50 if feature_name == 'GarageArea' else 100
        elif feature_name == 'LotFrontage':
            return 5
        elif feature_name == 'MiscVal':
            return 100
        else:
            return 1
    
    def _quality_description(self, quality: str) -> str:
        """Get description for quality ratings"""
        descriptions = {
            'Ex': 'Excellent',
            'Gd': 'Good', 
            'TA': 'Typical/Average',
            'Fa': 'Fair',
            'Po': 'Poor'
        }
        return descriptions.get(quality, quality)
    
    def get_friendly_feature_name(self, feature_name: str) -> str:
        """Get user-friendly display name for a feature"""
        friendly_names = {
            'YearBuilt': 'Year Built',
            'LotArea': 'Lot Area (sq ft)',
            'GrLivArea': 'Above Ground Living Area (sq ft)',
            'TotalBsmtSF': 'Total Basement Area (sq ft)',
            '1stFlrSF': 'First Floor Area (sq ft)',
            '2ndFlrSF': 'Second Floor Area (sq ft)',
            'GarageArea': 'Garage Area (sq ft)',
            'TotRmsAbvGrd': 'Total Rooms Above Ground',
            'MasVnrArea': 'Masonry Veneer Area (sq ft)',
            'OpenPorchSF': 'Open Porch Area (sq ft)',
            'WoodDeckSF': 'Wood Deck Area (sq ft)',
            'LotFrontage': 'Lot Frontage (feet)',
            'YearRemodAdd': 'Year Remodeled/Added',
            'BsmtFinSF1': 'Basement Finished Area 1 (sq ft)',
            'BsmtFinSF2': 'Basement Finished Area 2 (sq ft)', 
            'BsmtUnfSF': 'Basement Unfinished Area (sq ft)',
            'OverallQual': 'Overall Quality (1-10)',
            'OverallCond': 'Overall Condition (1-9)',
            'BedroomAbvGr': 'Bedrooms Above Ground',
            'FullBath': 'Full Bathrooms',
            'HalfBath': 'Half Bathrooms',
            'GarageCars': 'Garage Car Capacity',
            'Fireplaces': 'Number of Fireplaces',
            'ExterQual_encoded': 'Exterior Quality',
            'ExterCond_encoded': 'Exterior Condition',
            'BsmtQual_encoded': 'Basement Quality',
            'HeatingQC_encoded': 'Heating Quality & Condition',
            'KitchenQual_encoded': 'Kitchen Quality',
            'GarageQual_encoded': 'Garage Quality'
        }
        return friendly_names.get(feature_name, feature_name.replace('_', ' ').title())
    
    def get_feature_description(self, feature_name: str) -> str:
        """Get helpful description for a feature"""
        descriptions = {
            'YearBuilt': 'Original construction year of the house',
            'LotArea': 'Size of the property lot in square feet',
            'GrLivArea': 'Above ground living area in square feet',
            'TotalBsmtSF': 'Total basement area in square feet',
            '1stFlrSF': 'First floor area in square feet',
            '2ndFlrSF': 'Second floor area in square feet (0 if single story)',
            'GarageArea': 'Garage area in square feet',
            'TotRmsAbvGrd': 'Total number of rooms above ground (excluding bathrooms)',
            'OverallQual': 'Overall material and finish quality (1=Very Poor, 10=Very Excellent)',
            'OverallCond': 'Overall condition of the house (1=Very Poor, 9=Very Excellent)',
            'BedroomAbvGr': 'Number of bedrooms above ground level',
            'FullBath': 'Number of full bathrooms above ground',
            'GarageCars': 'Size of garage in car capacity',
            'Fireplaces': 'Number of fireplaces'
        }
        return descriptions.get(feature_name, f'Feature: {feature_name}')
    
    def process_user_input(self, feature_name: str, user_value: Any) -> float:
        """Convert user input to model-ready normalized value"""
        if feature_name in self.min_max_features:
            return self.normalize_min_max(user_value, feature_name)
        elif feature_name in self.quality_scales:
            return self.quality_scales[feature_name].get(user_value, user_value)
        else:
            return float(user_value)
    
    def prepare_prediction_input(self, user_inputs: Dict[str, Any]) -> Dict[str, float]:
        """Convert all user inputs to model-ready format"""
        model_inputs = {}
        for feature_name, user_value in user_inputs.items():
            model_inputs[feature_name] = self.process_user_input(feature_name, user_value)
        return model_inputs
    
    def format_for_display(self, feature_name: str, model_value: float) -> str:
        """Format model value for user-friendly display"""
        if feature_name in self.min_max_features:
            real_value = self.denormalize_min_max(model_value, feature_name)
            unit = self.min_max_features[feature_name]['unit']
            if unit == 'year':
                return f"{int(real_value)}"
            elif unit in ['sq ft', 'feet', 'dollars']:
                return f"{real_value:,.0f} {unit}"
            else:
                return f"{real_value} {unit}"
        else:
            return f"{model_value}"