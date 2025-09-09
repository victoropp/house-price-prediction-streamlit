"""
User-Friendly Dropdown Mappings
Converts technical feature values to user-friendly dropdown options based on the actual Ames Housing dataset
"""

class UserFriendlyDropdowns:
    """Provides user-friendly dropdown options for all feature selections based on actual dataset values"""
    
    def __init__(self):
        # Quality mappings based on actual Ames Housing dataset
        # Most quality features use: Ex=5, Gd=4, TA=3, Fa=2, Po=1, NA=0
        self.quality_mappings = {
            'KitchenQual_encoded': {
                'display_to_value': {
                    '⭐⭐⭐⭐⭐ Excellent - Premium appliances & finishes': 5.0,
                    '⭐⭐⭐⭐ Good - Quality appliances & materials': 4.0, 
                    '⭐⭐⭐ Average - Standard kitchen setup': 3.0,
                    '⭐⭐ Fair - Basic kitchen, needs updating': 2.0
                }
            },
            'ExterQual_encoded': {
                'display_to_value': {
                    '⭐⭐⭐⭐⭐ Excellent - Premium exterior materials (stone/brick)': 5,
                    '⭐⭐⭐⭐ Good - Quality materials (wood shingles, etc.)': 4,
                    '⭐⭐⭐ Average - Standard exterior materials': 3,
                    '⭐⭐ Fair - Below average materials': 2
                }
            },
            'ExterCond_encoded': {
                'display_to_value': {
                    '⭐⭐⭐⭐⭐ Excellent - Like new condition': 5,
                    '⭐⭐⭐⭐ Good - Well maintained': 4,
                    '⭐⭐⭐ Average - Normal wear & tear': 3,
                    '⭐⭐ Fair - Some maintenance needed': 2,
                    '⭐ Poor - Major maintenance required': 1
                }
            },
            'HeatingQC_encoded': {
                'display_to_value': {
                    '🔥 Excellent - Premium heating system': 5,
                    '🔥 Good - Reliable heating system': 4,
                    '🔥 Average - Standard heating': 3,
                    '🔥 Fair - Basic heating system': 2,
                    '🔥 Poor - Inadequate heating': 1
                }
            },
            'BsmtQual_encoded': {
                'display_to_value': {
                    '⭐⭐⭐⭐⭐ Excellent - 100+ inches ceiling height': 5.0,
                    '⭐⭐⭐⭐ Good - 90-99 inches ceiling height': 4.0,
                    '⭐⭐⭐ Average - 80-89 inches ceiling height': 3.0,
                    '⭐⭐ Fair - 70-79 inches ceiling height': 2.0
                }
            },
            'BsmtCond_encoded': {
                'display_to_value': {
                    '⭐⭐⭐⭐⭐ Excellent - No issues': 5.0,
                    '⭐⭐⭐⭐ Good - Minor issues': 4.0,
                    '⭐⭐⭐ Average - Some dampness/cracking': 3.0,
                    '⭐⭐ Fair - Moderate dampness/cracking': 2.0,
                    '⭐ Poor - Severe issues, needs repair': 1.0
                }
            },
            'GarageQual_encoded': {
                'display_to_value': {
                    '⭐⭐⭐⭐⭐ Excellent - Premium garage quality': 5.0,
                    '⭐⭐⭐⭐ Good - Above average garage': 4.0,
                    '⭐⭐⭐ Average - Standard garage': 3.0,
                    '⭐⭐ Fair - Below average garage': 2.0,
                    '⭐ Poor - Poor quality garage': 1.0
                }
            },
            'GarageCond_encoded': {
                'display_to_value': {
                    '⭐⭐⭐⭐⭐ Excellent - Like new condition': 5.0,
                    '⭐⭐⭐⭐ Good - Well maintained': 4.0,
                    '⭐⭐⭐ Average - Normal wear & tear': 3.0,
                    '⭐⭐ Fair - Minor repairs needed': 2.0,
                    '⭐ Poor - Major repairs needed': 1.0
                }
            },
            'FireplaceQu_encoded': {
                'display_to_value': {
                    '🔥 Excellent - Superior masonry fireplace': 5.0,
                    '🔥 Good - Masonry fireplace in main level': 4.0,
                    '🔥 Average - Prefabricated fireplace': 3.0,
                    '🔥 Fair - Fireplace in basement': 2.0,
                    '🔥 Poor - Ben Franklin stove': 1.0
                }
            },
            'PoolQC_encoded': {
                'display_to_value': {
                    '🏊‍♂️ Excellent - Premium pool': 4.0,
                    '🏊‍♂️ Good - Above average pool': 3.0,
                    '🏊‍♂️ Average - Standard pool': 2.0,
                    '🏊‍♂️ Fair - Below average pool': 1.0
                }
            }
        }
        
        # Overall Quality and Condition (1-10 scale)
        self.overall_mappings = {
            'OverallQual': {
                'display_to_value': {
                    '10 ⭐ Very Excellent - Top 1% luxury homes': 10,
                    '9 ⭐ Excellent - Premium custom homes': 9,
                    '8 ⭐ Very Good - High-end homes': 8,
                    '7 ⭐ Good - Above average quality': 7,
                    '6 ⭐ Above Average - Better than typical': 6,
                    '5 ⭐ Average - Typical quality home': 5,
                    '4 ⭐ Below Average - Some quality compromises': 4,
                    '3 ⭐ Fair - Lower quality materials/workmanship': 3,
                    '2 ⭐ Poor - Major quality issues': 2,
                    '1 ⭐ Very Poor - Significant structural problems': 1
                }
            },
            'OverallCond': {
                'display_to_value': {
                    '9 ⭐ Excellent - Exceptionally well maintained': 9,
                    '8 ⭐ Very Good - Well maintained, minor wear': 8,
                    '7 ⭐ Good - Normal maintenance, good condition': 7,
                    '6 ⭐ Above Average - Well kept with minor issues': 6,
                    '5 ⭐ Average - Normal wear for age': 5,
                    '4 ⭐ Below Average - Some deferred maintenance': 4,
                    '3 ⭐ Fair - Needs attention, multiple issues': 3,
                    '2 ⭐ Poor - Significant maintenance required': 2,
                    '1 ⭐ Very Poor - Major renovation needed': 1
                }
            }
        }
        
        # Binary feature mappings  
        self.binary_mappings = {
            'CentralAir_encoded': {
                'display_to_value': {
                    '❄️ Yes - Central Air Conditioning': 1,
                    '🌡️ No - No Central AC': 0
                }
            },
            'WasRemodeled': {
                'display_to_value': {
                    '🔨 Yes - Home has been remodeled': 1,
                    '🏠 No - Original construction': 0
                }
            }
        }
        
        # Special handling for features with original categorical values
        # These are still encoded in the final dataset but we can provide meaningful descriptions
        self.categorical_descriptions = {
            'MSZoning': 'General zoning classification',
            'LotShape': 'General shape of property', 
            'LandContour': 'Flatness of the property',
            'LotConfig': 'Lot configuration',
            'LandSlope': 'Slope of property',
            'BldgType': 'Type of dwelling',
            'HouseStyle': 'Style of dwelling',
            'RoofStyle': 'Type of roof',
            'Foundation': 'Type of foundation',
            'Heating': 'Type of heating',
            'Electrical': 'Electrical system',
            'GarageType': 'Garage location',
            'PavedDrive': 'Paved driveway',
            'SaleType': 'Type of sale',
            'SaleCondition': 'Condition of sale'
        }
    
    def get_user_friendly_options(self, feature_name, unique_values):
        """
        Get user-friendly dropdown options for a given feature
        
        Args:
            feature_name: Name of the feature
            unique_values: List of unique values from the dataset
            
        Returns:
            dict: {display_name: actual_value} mapping
        """
        # Handle neighborhood specially - already has good mapping
        if feature_name == 'Neighborhood_encoded':
            from utils.neighborhood_mapper import get_neighborhood_options_for_ui
            return get_neighborhood_options_for_ui()
        
        # Check quality mappings first
        if feature_name in self.quality_mappings:
            mapping = self.quality_mappings[feature_name]['display_to_value']
            # Filter to only include values that exist in the dataset
            available_mapping = {}
            for display, value in mapping.items():
                if value in unique_values:
                    available_mapping[display] = value
            return available_mapping
        
        # Check overall quality/condition mappings
        if feature_name in self.overall_mappings:
            mapping = self.overall_mappings[feature_name]['display_to_value']
            available_mapping = {}
            for display, value in mapping.items():
                if value in unique_values:
                    available_mapping[display] = value
            return available_mapping
        
        # Check binary mappings
        if feature_name in self.binary_mappings:
            mapping = self.binary_mappings[feature_name]['display_to_value']
            available_mapping = {}
            for display, value in mapping.items():
                if value in unique_values:
                    available_mapping[display] = value
            return available_mapping
        
        # For encoded features without specific mappings, create descriptive labels
        if '_encoded' in feature_name:
            base_name = feature_name.replace('_encoded', '')
            if base_name in self.categorical_descriptions:
                desc = self.categorical_descriptions[base_name]
                return {f"{desc} - Option {i+1} (Value: {val})": val 
                       for i, val in enumerate(sorted(unique_values))}
        
        # Default: return values with basic formatting
        if all(isinstance(val, (int, float)) for val in unique_values):
            return {f"Level {int(val)}": val for val in sorted(unique_values)}
        else:
            return {str(val): val for val in sorted(unique_values)}
    
    def get_default_selection_index(self, feature_name, options_dict):
        """Get a good default selection index for a feature"""
        if feature_name in ['OverallQual', 'OverallCond']:
            # Default to "Average" (5) or closest available
            for i, (display_name, value) in enumerate(options_dict.items()):
                if 'Average' in display_name or value == 5:
                    return i
        
        if '_encoded' in feature_name and any(qual in feature_name for qual in ['Qual', 'Cond']):
            # Default to "Good" (4) or "Average" (3) for quality features
            for i, (display_name, value) in enumerate(options_dict.items()):
                if 'Good' in display_name or 'Average' in display_name:
                    return i
        
        if feature_name == 'Neighborhood_encoded':
            # Default to a mid-tier neighborhood
            for i, (display_name, value) in enumerate(options_dict.items()):
                if 'Mid-Tier' in display_name:
                    return i
        
        # Default to middle option
        return len(options_dict) // 2
        
    def format_number_input_help(self, feature_name):
        """Get helpful descriptions for number inputs"""
        help_texts = {
            'GrLivArea': 'Above grade (ground) living area in square feet',
            'TotalBsmtSF': 'Total square feet of basement area (0 if no basement)',
            'GarageArea': 'Size of garage in square feet (0 if no garage)',
            'YearBuilt': 'Original construction date',
            'YearRemodAdd': 'Remodel date (same as YearBuilt if no remodeling)', 
            'LotArea': 'Lot size in square feet',
            'LotFrontage': 'Linear feet of street connected to property',
            'TotalPorchSF': 'Total porch area in square feet',
            'Fireplaces': 'Number of fireplaces',
            'FullBath': 'Full bathrooms above grade',
            'HalfBath': 'Half baths above grade', 
            'BedroomAbvGr': 'Number of bedrooms above basement level',
            'TotRmsAbvGrd': 'Total rooms above grade (not including bathrooms)',
            'GarageCars': 'Size of garage in car capacity',
            '1stFlrSF': 'First floor square feet',
            '2ndFlrSF': 'Second floor square feet',
            'WoodDeckSF': 'Wood deck area in square feet',
            'OpenPorchSF': 'Open porch area in square feet',
            'EnclosedPorch': 'Enclosed porch area in square feet',
            'ScreenPorch': 'Screen porch area in square feet',
            'PoolArea': 'Pool area in square feet',
            'MiscVal': 'Value of miscellaneous feature ($)'
        }
        
        return help_texts.get(feature_name, f'Enter {feature_name.replace("_", " ").lower()}')
        
    def get_feature_display_name(self, feature_name):
        """Get user-friendly display name for feature"""
        display_names = {
            'OverallQual': 'Overall Quality (Material & Finish)',
            'OverallCond': 'Overall Condition Rating', 
            'KitchenQual_encoded': 'Kitchen Quality',
            'ExterQual_encoded': 'Exterior Material Quality',
            'ExterCond_encoded': 'Exterior Condition',
            'HeatingQC_encoded': 'Heating Quality & Condition',
            'BsmtQual_encoded': 'Basement Quality (Height)',
            'BsmtCond_encoded': 'Basement Condition',
            'GarageQual_encoded': 'Garage Quality',
            'GarageCond_encoded': 'Garage Condition',
            'FireplaceQu_encoded': 'Fireplace Quality',
            'PoolQC_encoded': 'Pool Quality',
            'CentralAir_encoded': 'Central Air Conditioning',
            'WasRemodeled': 'Has Been Remodeled',
            'GrLivArea': 'Above Ground Living Area (sq ft)',
            'TotalBsmtSF': 'Total Basement Area (sq ft)',
            'GarageArea': 'Garage Area (sq ft)',
            'YearBuilt': 'Year Built',
            'Fireplaces': 'Number of Fireplaces'
        }
        
        return display_names.get(feature_name, feature_name.replace('_', ' ').title())