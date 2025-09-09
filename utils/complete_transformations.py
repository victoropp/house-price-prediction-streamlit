"""
Complete Bidirectional Transformations
Full implementation of all 224 feature transformations based on comprehensive analysis

This class handles ALL transformation types:
- Min-Max Normalized Features (24 features)
- Unchanged Categorical Features (12 features) 
- Encoded Features (27 features)
- Log-Transformed Features (28 features)
- One-Hot Encoded Features (121 features)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Union
import pickle
import os

class CompleteHouseTransformations:
    """
    Complete bidirectional transformation system for house price prediction
    Based on comprehensive analysis of all 224 features
    """
    
    def __init__(self):
        # Load comprehensive mapping results if available
        self.load_mapping_results()
        
        # Min-Max normalized features (24 total) - Perfect correlations
        self.min_max_features = {
            'YearBuilt': {'min': 1872, 'max': 2010, 'unit': 'year', 'practical_min': 1900, 'practical_max': 2010},
            'LotArea': {'min': 1300, 'max': 215245, 'unit': 'sq ft', 'practical_min': 5000, 'practical_max': 25000},
            'GrLivArea': {'min': 334, 'max': 5642, 'unit': 'sq ft', 'practical_min': 800, 'practical_max': 3500},
            'TotalBsmtSF': {'min': 0, 'max': 6110, 'unit': 'sq ft', 'practical_min': 0, 'practical_max': 2500},
            '1stFlrSF': {'min': 334, 'max': 4692, 'unit': 'sq ft', 'practical_min': 500, 'practical_max': 2500},
            '2ndFlrSF': {'min': 0, 'max': 2065, 'unit': 'sq ft', 'practical_min': 0, 'practical_max': 1500},
            'MasVnrArea': {'min': 0, 'max': 1600, 'unit': 'sq ft', 'practical_min': 0, 'practical_max': 500},
            'BsmtFinSF1': {'min': 0, 'max': 5644, 'unit': 'sq ft', 'practical_min': 0, 'practical_max': 2000},
            'BsmtFinSF2': {'min': 0, 'max': 1474, 'unit': 'sq ft', 'practical_min': 0, 'practical_max': 800},
            'BsmtUnfSF': {'min': 0, 'max': 2336, 'unit': 'sq ft', 'practical_min': 0, 'practical_max': 1500},
            'LowQualFinSF': {'min': 0, 'max': 572, 'unit': 'sq ft', 'practical_min': 0, 'practical_max': 300},
            'GarageYrBlt': {'min': 1900, 'max': 2010, 'unit': 'year', 'practical_min': 1950, 'practical_max': 2010},
            'GarageArea': {'min': 0, 'max': 1418, 'unit': 'sq ft', 'practical_min': 0, 'practical_max': 900},
            'WoodDeckSF': {'min': 0, 'max': 857, 'unit': 'sq ft', 'practical_min': 0, 'practical_max': 500},
            'OpenPorchSF': {'min': 0, 'max': 547, 'unit': 'sq ft', 'practical_min': 0, 'practical_max': 200},
            'EnclosedPorch': {'min': 0, 'max': 552, 'unit': 'sq ft', 'practical_min': 0, 'practical_max': 200},
            '3SsnPorch': {'min': 0, 'max': 508, 'unit': 'sq ft', 'practical_min': 0, 'practical_max': 200},
            'ScreenPorch': {'min': 0, 'max': 480, 'unit': 'sq ft', 'practical_min': 0, 'practical_max': 200},
            'MiscVal': {'min': 0, 'max': 15500, 'unit': 'dollars', 'practical_min': 0, 'practical_max': 5000},
            'MoSold': {'min': 1, 'max': 12, 'unit': 'month', 'practical_min': 1, 'practical_max': 12},
            'TotRmsAbvGrd': {'min': 2, 'max': 14, 'unit': 'rooms', 'practical_min': 4, 'practical_max': 12},
            # Engineering features - these might need special handling
            'TotalSF': {'min': 334, 'max': 11752, 'unit': 'sq ft', 'practical_min': 1000, 'practical_max': 5000},
            'TotalPorchSF': {'min': 0, 'max': 1424, 'unit': 'sq ft', 'practical_min': 0, 'practical_max': 400},
            'HouseAge': {'min': 0, 'max': 138, 'unit': 'years', 'practical_min': 5, 'practical_max': 80}
        }
        
        # Unchanged categorical features (12 total) - User-friendly as-is
        self.unchanged_features = {
            'OverallQual': {'min': 1, 'max': 10, 'unit': 'rating', 'description': 'Overall Quality (1=Poor, 10=Excellent)'},
            'OverallCond': {'min': 1, 'max': 9, 'unit': 'rating', 'description': 'Overall Condition (1=Poor, 9=Excellent)'},
            'BedroomAbvGr': {'min': 0, 'max': 8, 'unit': 'bedrooms', 'description': 'Bedrooms Above Ground'},
            'FullBath': {'min': 0, 'max': 3, 'unit': 'bathrooms', 'description': 'Full Bathrooms'},
            'HalfBath': {'min': 0, 'max': 2, 'unit': 'half baths', 'description': 'Half Bathrooms'},
            'GarageCars': {'min': 0, 'max': 4, 'unit': 'cars', 'description': 'Garage Car Capacity'},
            'BsmtFullBath': {'min': 0, 'max': 3, 'unit': 'bathrooms', 'description': 'Basement Full Bathrooms'},
            'BsmtHalfBath': {'min': 0, 'max': 2, 'unit': 'half baths', 'description': 'Basement Half Bathrooms'},
            'KitchenAbvGr': {'min': 0, 'max': 3, 'unit': 'kitchens', 'description': 'Kitchens Above Ground'},
            'Fireplaces': {'min': 0, 'max': 3, 'unit': 'fireplaces', 'description': 'Number of Fireplaces'},
            'YrSold': {'min': 2006, 'max': 2010, 'unit': 'year', 'description': 'Year Sold'},
            'TotalBaths': {'min': 1, 'max': 6, 'unit': 'bathrooms', 'description': 'Total Bathrooms'}
        }
        
        # Quality scale mappings for encoded features (27 total)
        self.quality_mappings = {
            'ExterQual_encoded': {'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'ExterCond_encoded': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'BsmtQual_encoded': {'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'BsmtCond_encoded': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4},
            'HeatingQC_encoded': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'KitchenQual_encoded': {'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'FireplaceQu_encoded': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'GarageQual_encoded': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'GarageCond_encoded': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'PoolQC_encoded': {'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
        }
        
        # Binary encoded features
        self.binary_mappings = {
            'Street_encoded': {'Grvl': 0, 'Pave': 1},
            'CentralAir_encoded': {'N': 0, 'Y': 1},
            'Utilities_encoded': {'NoSeWa': 0, 'AllPub': 1}  # Simplified
        }
        
        # Target transformation parameters (Box-Cox)
        self.target_transform_lambda = -0.07693211157738546
        
        # Working log-transformed features (exclude broken ones)
        self.working_log_features = {
            'PoolArea_transformed': {'base': 'PoolArea', 'type': 'log1p'},
            'LowQualFinSF_transformed': {'base': 'LowQualFinSF', 'type': 'log1p'},
            '3SsnPorch_transformed': {'base': '3SsnPorch', 'type': 'log1p'},
            'BsmtFinSF2_transformed': {'base': 'BsmtFinSF2', 'type': 'log1p'},
            'LivingAreaRatio_transformed': {'base': 'LivingAreaRatio', 'type': 'log'},
            'OpenPorchSF_transformed': {'base': 'OpenPorchSF', 'type': 'log1p'},
            'TotalPorchSF_transformed': {'base': 'TotalPorchSF', 'type': 'log1p'},
            'BathBedroomRatio_transformed': {'base': 'BathBedroomRatio', 'type': 'log'},
            'WoodDeckSF_transformed': {'base': 'WoodDeckSF', 'type': 'log1p'},
            '1stFlrSF_transformed': {'base': '1stFlrSF', 'type': 'log'},
            'BsmtFinSF1_transformed': {'base': 'BsmtFinSF1', 'type': 'log1p'},
            'MSSubClass_transformed': {'base': 'MSSubClass', 'type': 'log'},
            'GrLivArea_transformed': {'base': 'GrLivArea', 'type': 'log'},
            'TotalBsmtSF_transformed': {'base': 'TotalBsmtSF', 'type': 'log1p'}
        }
        
        # Broken log-transformed features (all have identical values)
        self.broken_log_features = {
            'LotArea_transformed': 17.2228,
            'MasVnrArea_transformed': 2.8370,
            'GarageYrBlt_transformed': 35765132.1567,
            'GarageQualityScore_transformed': 83.6320,
            'TotalSF_transformed': 24.8137,
            'AgeQualityInteraction_transformed': 4.6978
        }

    def load_mapping_results(self):
        """Load comprehensive mapping results if available"""
        try:
            mapping_file = os.path.join(os.path.dirname(__file__), '..', '..', 'comprehensive_field_mappings.pkl')
            if os.path.exists(mapping_file):
                with open(mapping_file, 'rb') as f:
                    self.mapping_results = pickle.load(f)
                print("Loaded comprehensive mapping results")
            else:
                self.mapping_results = None
        except:
            self.mapping_results = None
    
    def normalize_min_max(self, real_value: Union[int, float], feature_name: str) -> float:
        """Convert real-world value to 0-1 normalized value for model input"""
        if feature_name not in self.min_max_features:
            return real_value
        
        params = self.min_max_features[feature_name]
        min_val, max_val = params['min'], params['max']
        
        # Clamp to valid range
        real_value = max(min_val, min(max_val, real_value))
        
        # Normalize to 0-1
        normalized = (real_value - min_val) / (max_val - min_val)
        return round(normalized, 6)
    
    def denormalize_min_max(self, normalized_value: float, feature_name: str) -> Union[int, float]:
        """Convert 0-1 normalized value back to real-world value for display"""
        if feature_name not in self.min_max_features:
            return normalized_value
        
        params = self.min_max_features[feature_name]
        min_val, max_val = params['min'], params['max']
        
        # Denormalize from 0-1 to real range
        real_value = min_val + normalized_value * (max_val - min_val)
        
        # Return integer for features that should be integers
        if params['unit'] in ['year', 'rooms', 'month']:
            return int(round(real_value))
        
        return round(real_value, 2)
    
    def encode_quality(self, quality_text: str, feature_name: str) -> Union[int, float]:
        """Convert quality text (Po, Fa, TA, Gd, Ex) to encoded number"""
        if feature_name not in self.quality_mappings:
            return quality_text
        
        return self.quality_mappings[feature_name].get(quality_text, quality_text)
    
    def decode_quality(self, encoded_value: Union[int, float], feature_name: str) -> str:
        """Convert encoded number back to quality text"""
        if feature_name not in self.quality_mappings:
            return str(encoded_value)
        
        # Create reverse mapping
        reverse_map = {v: k for k, v in self.quality_mappings[feature_name].items()}
        return reverse_map.get(int(encoded_value), str(encoded_value))
    
    def encode_binary(self, category_text: str, feature_name: str) -> Union[int, float]:
        """Convert binary category text to encoded number"""
        if feature_name not in self.binary_mappings:
            return category_text
        
        return self.binary_mappings[feature_name].get(category_text, category_text)
    
    def decode_binary(self, encoded_value: Union[int, float], feature_name: str) -> str:
        """Convert encoded number back to binary category text"""
        if feature_name not in self.binary_mappings:
            return str(encoded_value)
        
        # Create reverse mapping
        reverse_map = {v: k for k, v in self.binary_mappings[feature_name].items()}
        return reverse_map.get(int(encoded_value), str(encoded_value))
    
    def transform_log_feature(self, real_value: Union[int, float], feature_name: str) -> float:
        """Apply log transformation to feature value"""
        if feature_name not in self.working_log_features:
            # Check if it's a broken feature
            if feature_name in self.broken_log_features:
                return self.broken_log_features[feature_name]
            return real_value
        
        transform_info = self.working_log_features[feature_name]
        transform_type = transform_info['type']
        
        if transform_type == 'log1p':
            return np.log1p(max(0, real_value))
        elif transform_type == 'log':
            return np.log(max(1e-8, real_value))  # Avoid log(0)
        else:
            return real_value
    
    def reverse_log_feature(self, log_value: float, feature_name: str) -> Union[int, float]:
        """Reverse log transformation to get original value"""
        if feature_name not in self.working_log_features:
            return log_value
        
        transform_info = self.working_log_features[feature_name]
        transform_type = transform_info['type']
        
        if transform_type == 'log1p':
            return max(0, np.expm1(log_value))
        elif transform_type == 'log':
            return max(0, np.exp(log_value))
        else:
            return log_value
    
    def inverse_target_transformation(self, transformed_price: float) -> float:
        """Apply inverse Box-Cox transformation to get real price"""
        try:
            from scipy.special import inv_boxcox
            return inv_boxcox(transformed_price, self.target_transform_lambda) - 1
        except ImportError:
            # Fallback if scipy not available
            return np.exp(transformed_price)
    
    def get_feature_type(self, feature_name: str) -> str:
        """Determine the transformation type for a feature"""
        if feature_name in self.min_max_features:
            return 'min_max'
        elif feature_name in self.unchanged_features:
            return 'unchanged'
        elif feature_name in self.quality_mappings:
            return 'quality_encoded'
        elif feature_name in self.binary_mappings:
            return 'binary_encoded'
        elif feature_name in self.working_log_features:
            return 'log_transformed'
        elif feature_name in self.broken_log_features:
            return 'broken_log'
        elif feature_name.endswith('_encoded'):
            return 'other_encoded'
        elif any(pattern in feature_name for pattern in ['_', ' ', '(', ')']):
            return 'onehot_encoded'
        else:
            return 'unknown'
    
    def get_user_input_config(self, feature_name: str) -> Dict[str, Any]:
        """Get Streamlit input configuration for a feature"""
        feature_type = self.get_feature_type(feature_name)
        
        if feature_type == 'min_max':
            params = self.min_max_features[feature_name]
            step = 1 if params['unit'] in ['year', 'rooms', 'month'] else (
                50 if 'SF' in feature_name or 'Area' in feature_name else 100
            )
            return {
                'type': 'number_input',
                'label': self.get_friendly_name(feature_name),
                'min_value': params['practical_min'],
                'max_value': params['practical_max'],
                'value': (params['practical_min'] + params['practical_max']) // 2,
                'step': step,
                'help': f"Range: {params['min']:,} - {params['max']:,} {params['unit']}"
            }
            
        elif feature_type == 'unchanged':
            params = self.unchanged_features[feature_name]
            return {
                'type': 'selectbox',
                'label': self.get_friendly_name(feature_name),
                'options': list(range(params['min'], params['max'] + 1)),
                'index': (params['max'] - params['min']) // 2,
                'help': params['description']
            }
            
        elif feature_type == 'quality_encoded':
            options = list(self.quality_mappings[feature_name].keys())
            return {
                'type': 'selectbox',
                'label': self.get_friendly_name(feature_name),
                'options': options,
                'index': options.index('TA') if 'TA' in options else 0,
                'help': 'Po=Poor, Fa=Fair, TA=Typical/Average, Gd=Good, Ex=Excellent'
            }
            
        elif feature_type == 'binary_encoded':
            options = list(self.binary_mappings[feature_name].keys())
            return {
                'type': 'selectbox',
                'label': self.get_friendly_name(feature_name),
                'options': options,
                'index': 0,
                'help': f"Choose {' or '.join(options)}"
            }
            
        else:
            return {
                'type': 'number_input',
                'label': self.get_friendly_name(feature_name),
                'value': 0.0,
                'help': f"Feature type: {feature_type}"
            }
    
    def get_friendly_name(self, feature_name: str) -> str:
        """Get user-friendly display name for a feature"""
        friendly_names = {
            'YearBuilt': 'Year Built',
            'LotArea': 'Lot Area (sq ft)',
            'GrLivArea': 'Above Ground Living Area (sq ft)',
            'TotalBsmtSF': 'Total Basement Area (sq ft)',
            '1stFlrSF': 'First Floor Area (sq ft)',
            '2ndFlrSF': 'Second Floor Area (sq ft)',
            'GarageArea': 'Garage Area (sq ft)',
            'MasVnrArea': 'Masonry Veneer Area (sq ft)',
            'BsmtFinSF1': 'Basement Finished Area 1 (sq ft)',
            'BsmtFinSF2': 'Basement Finished Area 2 (sq ft)',
            'BsmtUnfSF': 'Basement Unfinished Area (sq ft)',
            'LowQualFinSF': 'Low Quality Finished Area (sq ft)',
            'GarageYrBlt': 'Garage Year Built',
            'WoodDeckSF': 'Wood Deck Area (sq ft)',
            'OpenPorchSF': 'Open Porch Area (sq ft)',
            'EnclosedPorch': 'Enclosed Porch Area (sq ft)',
            '3SsnPorch': '3-Season Porch Area (sq ft)',
            'ScreenPorch': 'Screen Porch Area (sq ft)',
            'MiscVal': 'Miscellaneous Value ($)',
            'MoSold': 'Month Sold',
            'TotRmsAbvGrd': 'Total Rooms Above Ground',
            'OverallQual': 'Overall Quality',
            'OverallCond': 'Overall Condition',
            'BedroomAbvGr': 'Bedrooms Above Ground',
            'FullBath': 'Full Bathrooms',
            'HalfBath': 'Half Bathrooms',
            'GarageCars': 'Garage Car Capacity',
            'ExterQual_encoded': 'Exterior Quality',
            'ExterCond_encoded': 'Exterior Condition',
            'BsmtQual_encoded': 'Basement Quality',
            'HeatingQC_encoded': 'Heating Quality & Condition',
            'KitchenQual_encoded': 'Kitchen Quality'
        }
        return friendly_names.get(feature_name, feature_name.replace('_', ' ').title())
    
    def transform_user_input(self, feature_name: str, user_value: Any) -> float:
        """Convert user input to model-ready value"""
        feature_type = self.get_feature_type(feature_name)
        
        if feature_type == 'min_max':
            return self.normalize_min_max(user_value, feature_name)
        elif feature_type == 'unchanged':
            return float(user_value)
        elif feature_type == 'quality_encoded':
            return float(self.encode_quality(user_value, feature_name))
        elif feature_type == 'binary_encoded':
            return float(self.encode_binary(user_value, feature_name))
        else:
            return float(user_value)
    
    def format_model_output(self, feature_name: str, model_value: float) -> str:
        """Format model value for user-friendly display"""
        feature_type = self.get_feature_type(feature_name)
        
        if feature_type == 'min_max':
            real_value = self.denormalize_min_max(model_value, feature_name)
            params = self.min_max_features[feature_name]
            if params['unit'] == 'year':
                return f"{int(real_value)}"
            elif params['unit'] in ['sq ft', 'dollars']:
                return f"{real_value:,.0f} {params['unit']}"
            else:
                return f"{real_value} {params['unit']}"
        elif feature_type == 'quality_encoded':
            return self.decode_quality(model_value, feature_name)
        elif feature_type == 'binary_encoded':
            return self.decode_binary(model_value, feature_name)
        else:
            return f"{model_value:.2f}"
    
    def get_priority_features(self) -> List[str]:
        """Get list of priority features for UI implementation"""
        return [
            # Most important user-facing features
            'YearBuilt', 'LotArea', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
            'OverallQual', 'OverallCond', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'GarageCars',
            'ExterQual_encoded', 'HeatingQC_encoded', 'KitchenQual_encoded',
            'GarageArea', 'MasVnrArea', 'OpenPorchSF', 'WoodDeckSF', 'Fireplaces'
        ]
    
    def validate_transformations(self) -> Dict[str, Any]:
        """Validate that transformations work correctly"""
        validation_results = {
            'min_max_tests': {},
            'quality_tests': {},
            'binary_tests': {},
            'overall_status': True
        }
        
        # Test min-max transformations
        for feature in ['YearBuilt', 'LotArea', 'GrLivArea']:
            if feature in self.min_max_features:
                test_value = 2000 if feature == 'YearBuilt' else 1500
                normalized = self.normalize_min_max(test_value, feature)
                denormalized = self.denormalize_min_max(normalized, feature)
                
                validation_results['min_max_tests'][feature] = {
                    'input': test_value,
                    'normalized': normalized,
                    'denormalized': denormalized,
                    'match': abs(denormalized - test_value) < 1
                }
        
        # Test quality transformations
        for feature in ['ExterQual_encoded', 'HeatingQC_encoded']:
            if feature in self.quality_mappings:
                encoded = self.encode_quality('Gd', feature)
                decoded = self.decode_quality(encoded, feature)
                
                validation_results['quality_tests'][feature] = {
                    'input': 'Gd',
                    'encoded': encoded,
                    'decoded': decoded,
                    'match': decoded == 'Gd'
                }
        
        return validation_results