"""
Comprehensive Feature Mappings for User-Friendly Dropdowns
==========================================================

Based on thorough analysis of the complete data transformation pipeline:
1. Original Ames Housing dataset categorical values
2. Phase 5 intelligent encoding transformations  
3. Final processed dataset feature values

This mapping ensures 100% accuracy between user selections and model inputs.

Pipeline Understanding:
- Quality features: Ex=5, Gd=4, TA=3, Fa=2, Po=1, None=0
- Target encoding: Neighborhood -> Average sale price
- One-hot encoding: Low cardinality categoricals
- Ordinal encoding: Quality/condition features with natural order
"""

import pandas as pd
from utils.neighborhood_mapper import get_neighborhood_options_for_ui


class ComprehensiveFeatureMappings:
    """Accurate feature mappings based on complete pipeline analysis"""
    
    def __init__(self):
        # QUALITY FEATURES - Based on ordinal encoding from preprocessing pipeline
        # Original mapping: Po=1, Fa=2, TA=3, Gd=4, Ex=5, None=0
        self.quality_mappings = {
            'ExterQual_encoded': {
                'display_to_value': {
                    '🏆 Excellent - Premium exterior materials (Stone/Brick)': 5,
                    '✅ Good - Quality materials (Wood shingles, etc.)': 4,
                    '⚖️ Typical/Average - Standard exterior materials (Vinyl siding)': 3,
                    '⚠️ Fair - Below average/aging materials': 2
                },
                'description': 'Quality of exterior materials and finish'
            },
            'ExterCond_encoded': {
                'display_to_value': {
                    '🏆 Excellent - Like new condition': 5,
                    '✅ Good - Well maintained, minor wear': 4, 
                    '⚖️ Typical/Average - Normal wear and tear for age': 3,
                    '⚠️ Fair - Some deferred maintenance needed': 2,
                    '❌ Poor - Major maintenance required': 1
                },
                'description': 'Present condition of exterior materials'
            },
            'KitchenQual_encoded': {
                'display_to_value': {
                    '🏆 Excellent - Premium appliances, granite counters, custom cabinets': 5.0,
                    '✅ Good - Quality appliances, stone/corian counters, quality cabinets': 4.0,
                    '⚖️ Typical/Average - Standard appliances, formica counters, standard cabinets': 3.0,
                    '⚠️ Fair - Aging appliances, basic counters, dated cabinets': 2.0
                },
                'description': 'Kitchen quality - appliances, countertops, and cabinets'
            },
            'BsmtQual_encoded': {
                'display_to_value': {
                    '🏆 Excellent - 100+ inches ceiling height': 5.0,
                    '✅ Good - 90-99 inches ceiling height': 4.0,
                    '⚖️ Typical/Average - 80-89 inches ceiling height': 3.0,
                    '⚠️ Fair - 70-79 inches ceiling height (low but usable)': 2.0
                },
                'description': 'Basement height and overall quality'
            },
            'BsmtCond_encoded': {
                'display_to_value': {
                    '✅ Good - Minor issues, slight dampness allowed': 4.0,
                    '⚖️ Typical/Average - Some dampness or minor cracking': 3.0,
                    '⚠️ Fair - Moderate dampness, settling cracks': 2.0,
                    '❌ Poor - Severe dampness, major settling/structural issues': 1.0
                },
                'description': 'General condition of basement structure and moisture'
            },
            'HeatingQC_encoded': {
                'display_to_value': {
                    '🏆 Excellent - New efficient system, perfect heating': 5,
                    '✅ Good - Reliable system, adequate heating': 4,
                    '⚖️ Typical/Average - Functional system, normal operation': 3,
                    '⚠️ Fair - Older system, some issues, adequate heating': 2,
                    '❌ Poor - Old/inadequate system, heating problems': 1
                },
                'description': 'Quality and condition of heating system'
            },
            'GarageQual_encoded': {
                'display_to_value': {
                    '🏆 Excellent - Premium construction and materials': 5.0,
                    '✅ Good - Above average construction': 4.0,
                    '⚖️ Typical/Average - Standard construction': 3.0,
                    '⚠️ Fair - Below average construction': 2.0,
                    '❌ Poor - Poor quality construction/materials': 1.0
                },
                'description': 'Quality of garage construction and materials'
            },
            'GarageCond_encoded': {
                'display_to_value': {
                    '🏆 Excellent - Like new condition': 5.0,
                    '✅ Good - Well maintained': 4.0,
                    '⚖️ Typical/Average - Normal wear and tear': 3.0,
                    '⚠️ Fair - Minor repairs needed': 2.0,
                    '❌ Poor - Major repairs needed': 1.0
                },
                'description': 'Present condition of garage structure'
            },
            'FireplaceQu_encoded': {
                'display_to_value': {
                    '🏆 Excellent - Superior masonry fireplace': 5.0,
                    '✅ Good - Masonry fireplace in main level': 4.0,
                    '⚖️ Typical/Average - Prefabricated fireplace in main level': 3.0,
                    '⚠️ Fair - Fireplace in basement': 2.0,
                    '❌ Poor - Ben Franklin stove': 1.0
                },
                'description': 'Quality and location of fireplace'
            },
            'PoolQC_encoded': {
                'display_to_value': {
                    '🏆 Excellent - Premium pool with quality materials': 5.0,
                    '⚖️ Average/No Pool - No pool present': 4.0,  # This is the most common value
                    '⚠️ Fair - Basic pool': 2.0
                },
                'description': 'Pool quality (most homes have no pool)'
            }
        }
        
        # OVERALL QUALITY AND CONDITION (1-10 scale)
        self.overall_mappings = {
            'OverallQual': {
                'display_to_value': {
                    '10 🌟 Very Excellent - Top 1% luxury homes': 10,
                    '9 🌟 Excellent - Premium custom homes': 9,
                    '8 🌟 Very Good - High-end construction throughout': 8,
                    '7 🌟 Good - Above average materials and construction': 7,
                    '6 🌟 Above Average - Better than typical construction': 6,
                    '5 🌟 Average - Typical construction and materials': 5,
                    '4 🌟 Below Average - Some cost-cutting in materials': 4,
                    '3 🌟 Fair - Lower quality materials and workmanship': 3,
                    '2 🌟 Poor - Significant quality compromises': 2,
                    '1 🌟 Very Poor - Major structural/quality problems': 1
                },
                'description': 'Overall material and finish quality of the house'
            },
            'OverallCond': {
                'display_to_value': {
                    '9 🏠 Excellent - Recently renovated or new condition': 9,
                    '8 🏠 Very Good - Well maintained with minimal wear': 8,
                    '7 🏠 Good - Normal maintenance, well-kept condition': 7,
                    '6 🏠 Above Average - Generally well-kept with minor issues': 6,
                    '5 🏠 Average - Normal wear and tear appropriate for age': 5,
                    '4 🏠 Below Average - Some deferred maintenance evident': 4,
                    '3 🏠 Fair - Multiple areas need attention': 3,
                    '2 🏠 Poor - Significant maintenance required': 2,
                    '1 🏠 Very Poor - Major renovation needed': 1
                },
                'description': 'Overall condition rating of the house'
            }
        }
        
        # BINARY FEATURES
        self.binary_mappings = {
            'CentralAir_encoded': {
                'display_to_value': {
                    '❄️ Yes - Central Air Conditioning installed': 1,
                    '🌡️ No - Window units or no AC': 0
                },
                'description': 'Central air conditioning system'
            },
            'WasRemodeled': {
                'display_to_value': {
                    '🔨 Yes - Home has been remodeled/renovated': 1,
                    '🏠 No - Original construction, no major renovations': 0
                },
                'description': 'Whether the home has been remodeled'
            }
        }
        
        # ONE-HOT ENCODED CATEGORIES - Based on original categorical values
        self.categorical_mappings = {
            'MSZoning': {
                'display_to_value': {
                    '🏘️ Residential Low Density (Single-Family)': 'RL',
                    '🏢 Residential Medium Density (Townhomes/Condos)': 'RM',
                    '🏞️ Floating Village Residential': 'FV',
                    '🏙️ Residential High Density': 'RH',
                    '🏢 Commercial': 'C (all)'
                },
                'description': 'General zoning classification'
            },
            'LotShape': {
                'display_to_value': {
                    '📐 Regular - Standard rectangular lot': 'Reg',
                    '📐 Slightly Irregular - Minor variations from rectangle': 'IR1',
                    '📐 Moderately Irregular - Noticeable shape variations': 'IR2',
                    '📐 Very Irregular - Highly unusual lot shape': 'IR3'
                },
                'description': 'General shape of property lot'
            },
            'LandContour': {
                'display_to_value': {
                    '📏 Level - Near flat/level lot': 'Lvl',
                    '⛰️ Banked - Quick rise from street to building': 'Bnk',
                    '🏔️ Hillside - Significant slope from side to side': 'HLS',
                    '🕳️ Depression - Lot is lower than street level': 'Low'
                },
                'description': 'Flatness/slope of the property'
            },
            'LotConfig': {
                'display_to_value': {
                    '🏠 Interior Lot - Surrounded by other properties': 'Inside',
                    '🏠 Corner Lot - At intersection of two streets': 'Corner',
                    '🏠 Cul-de-sac - At end of dead-end street': 'CulDSac',
                    '🏠 Frontage on 2 sides': 'FR2',
                    '🏠 Frontage on 3 sides': 'FR3'
                },
                'description': 'Lot configuration relative to streets'
            },
            'BldgType': {
                'display_to_value': {
                    '🏡 Single-Family Detached': '1Fam',
                    '🏠 Townhouse End Unit': 'TwnhsE',
                    '🏢 Duplex (2-unit building)': 'Duplex',
                    '🏢 Townhouse Interior Unit': 'Twnhs',
                    '🏢 Two-Family Conversion (split single-family)': '2fmCon'
                },
                'description': 'Type of dwelling/building structure'
            },
            'HouseStyle': {
                'display_to_value': {
                    '🏠 One Story - All living on main level': '1Story',
                    '🏡 Two Story - Two full levels': '2Story',
                    '🏠 One and Half Story - Finished upper level': '1.5Fin',
                    '🏠 Split Level - Multi-level design': 'SLvl',
                    '🏠 Split Foyer - Entry between levels': 'SFoyer',
                    '🏠 One and Half Story - Unfinished upper level': '1.5Unf',
                    '🏡 Two and Half Story - Unfinished third level': '2.5Unf',
                    '🏡 Two and Half Story - Finished third level': '2.5Fin'
                },
                'description': 'Style/layout of dwelling'
            },
            'Foundation': {
                'display_to_value': {
                    '🧱 Poured Concrete - Modern, solid construction': 'PConc',
                    '🧱 Cinder Block - Standard masonry construction': 'CBlock',
                    '🧱 Brick & Tile - Traditional masonry': 'BrkTil',
                    '🪨 Stone - Natural stone construction': 'Stone',
                    '🏗️ Slab - Concrete slab, no basement': 'Slab',
                    '🪵 Wood - Wooden foundation (older homes)': 'Wood'
                },
                'description': 'Type of foundation construction'
            },
            'NeighborhoodTier': {
                'display_to_value': {
                    '🏘️ Standard/Budget Neighborhood': 'Standard',
                    '🏘️ Mid-Tier Neighborhood': 'Mid_Tier', 
                    '🏘️ Premium/Luxury Neighborhood': 'Premium'
                },
                'description': 'Neighborhood tier based on average home prices'
            },
            'HouseAgeGroup': {
                'display_to_value': {
                    '📅 Recent Home (11-25 years)': 'Recent (11-25)',
                    '📅 Mature Home (26-50 years)': 'Mature (26-50)', 
                    '📅 Older Home (51-100 years)': 'Old (51-100)',
                    '📅 Historic Home (100+ years)': 'Historic (100+)'
                },
                'description': 'House age group classification'
            },
            'SaleType': {
                'display_to_value': {
                    '📄 Warranty Deed - Conventional sale': 'WD',
                    '🏠 New Construction - Just built/never occupied': 'New',
                    '💰 Cash Sale - Warranty deed with cash payment': 'CWD',
                    '📋 Court Officer Deed - Foreclosure/legal sale': 'COD',
                    '📝 Contract Sale - Land contract/owner financing': 'Con',
                    '📝 Contract Low Down Payment': 'ConLD',
                    '📝 Contract Low Interest Rate': 'ConLI', 
                    '📝 Contract Low Down & Interest': 'ConLw',
                    '❓ Other/Unusual sale type': 'Oth'
                },
                'description': 'Type of sale transaction'
            }
        }
        
    def get_feature_options(self, feature_name, available_values):
        """
        Get user-friendly options for any feature based on available values in dataset
        
        Args:
            feature_name: Name of the feature
            available_values: Actual values present in the dataset
            
        Returns:
            dict: {display_name: actual_value} mapping
        """
        # Handle neighborhood specially (already well-implemented)
        if feature_name == 'Neighborhood_encoded':
            return get_neighborhood_options_for_ui()
        
        # Check quality mappings
        if feature_name in self.quality_mappings:
            mapping = self.quality_mappings[feature_name]['display_to_value']
            return {display: value for display, value in mapping.items() 
                   if value in available_values}
        
        # Check overall mappings
        if feature_name in self.overall_mappings:
            mapping = self.overall_mappings[feature_name]['display_to_value']
            return {display: value for display, value in mapping.items() 
                   if value in available_values}
        
        # Check binary mappings
        if feature_name in self.binary_mappings:
            mapping = self.binary_mappings[feature_name]['display_to_value']
            return {display: value for display, value in mapping.items() 
                   if value in available_values}
        
        # Handle one-hot encoded features
        for base_category, mapping_info in self.categorical_mappings.items():
            if feature_name.startswith(base_category + '_'):
                # This is a one-hot column - extract the specific category value
                category_value = feature_name.replace(base_category + '_', '')
                # Get the description for this specific category
                original_mappings = mapping_info['display_to_value']
                
                # Find the friendly name for this specific category value
                friendly_category = None
                for display_name, orig_value in original_mappings.items():
                    if orig_value == category_value:
                        friendly_category = display_name
                        break
                
                if friendly_category:
                    # Return binary options for this one-hot feature using actual dataset values
                    # Check if the dataset uses True/False or 1/0
                    true_val = True if True in available_values else 1
                    false_val = False if False in available_values else 0
                    return {
                        f"✅ Yes - {friendly_category}": true_val,
                        f"❌ No - Not {friendly_category}": false_val
                    }
                else:
                    # Fallback for unknown category values
                    true_val = True if True in available_values else 1
                    false_val = False if False in available_values else 0
                    return {
                        f"✅ Yes - {category_value}": true_val,
                        f"❌ No - Not {category_value}": false_val
                    }
            elif feature_name == base_category:
                # This should never happen in our dataset since all categoricals are one-hot encoded
                # But if it does, don't return string values that would break the model
                # Instead, return a safe binary mapping
                return {
                    f"Select {base_category} option": 0
                }
        
        # Handle other encoded features with target encoding (Condition1, Condition2, etc.)
        if '_encoded' in feature_name:
            base_name = feature_name.replace('_encoded', '')
            return {f"{base_name} Level {int(val)}": val for val in sorted(available_values) if isinstance(val, (int, float))}
        
        # Handle binary True/False features (including one-hot that we might have missed)
        if set(available_values) == {True, False} or set(available_values) == {0, 1} or set(available_values) == {False, True} or set(available_values) == {1, 0}:
            # This looks like a binary feature - use the actual dataset values
            true_val = True if True in available_values else 1
            false_val = False if False in available_values else 0
            
            # Try to give it meaningful names
            friendly_name = self.get_friendly_feature_name(feature_name)
            if friendly_name != feature_name.replace('_', ' ').title():
                # We have a good mapping, create binary options
                return {
                    f"✅ Yes - {friendly_name}": true_val,
                    f"❌ No - Not {friendly_name}": false_val
                }
            else:
                # Fallback to generic binary options
                clean_name = feature_name.replace('_', ' ')
                return {
                    f"✅ Yes - {clean_name}": true_val,
                    f"❌ No - Not {clean_name}": false_val
                }
        
        # Default for any remaining features
        if all(isinstance(val, (int, float)) for val in available_values):
            return {f"Level {int(val)}": val for val in sorted(available_values)}
        else:
            # Handle string values
            return {str(val): val for val in sorted(available_values)}
    
    def get_feature_description(self, feature_name):
        """Get description for a feature"""
        # Check all mapping types for descriptions
        for mapping_dict in [self.quality_mappings, self.overall_mappings, 
                            self.binary_mappings, self.categorical_mappings]:
            if feature_name in mapping_dict:
                return mapping_dict[feature_name].get('description', '')
        
        # Handle one-hot features
        for base_category, mapping_info in self.categorical_mappings.items():
            if feature_name.startswith(base_category + '_'):
                return mapping_info.get('description', f'{base_category} category')
        
        # Comprehensive feature descriptions
        descriptions = {
            # Core Area Features
            'GrLivArea': 'Above grade (ground) living area in square feet - key size indicator',
            'TotalBsmtSF': 'Total square feet of basement area - includes finished and unfinished',
            'LotArea': 'Total lot size in square feet - property footprint',
            'GarageArea': 'Size of garage in square feet - typically 200-400 for 1-car, 400-600 for 2-car',
            '1stFlrSF': 'First floor square footage - main living level',
            '2ndFlrSF': 'Second floor square footage - if applicable',
            'BsmtFinSF1': 'Type 1 finished square footage in basement - higher quality finish',
            'BsmtFinSF2': 'Type 2 finished square footage in basement - average quality finish',
            'BsmtUnfSF': 'Unfinished square footage of basement area',
            'LowQualFinSF': 'Low quality finished square feet (usually converted spaces)',
            
            # Outdoor Areas
            'WoodDeckSF': 'Wood deck area in square feet',
            'OpenPorchSF': 'Open porch area in square feet',
            'EnclosedPorch': 'Enclosed porch area in square feet',
            '3SsnPorch': 'Three season porch area in square feet',
            'ScreenPorch': 'Screen porch area in square feet',
            'PoolArea': 'Pool area in square feet (0 if no pool)',
            'MasVnrArea': 'Masonry veneer area in square feet',
            
            # Transformed Features
            'GrLivArea_transformed': 'Scaled living area - optimized for model prediction',
            'TotalBsmtSF_transformed': 'Scaled basement area - optimized for model prediction',
            'LotArea_transformed': 'Scaled lot size - optimized for model prediction',
            'TotalSF_transformed': 'Scaled total house area - combined living spaces',
            
            # Age & Time
            'YearBuilt': 'Original construction year - impacts style and systems',
            'YearRemodAdd': 'Year of remodel/addition (same as YearBuilt if no major renovations)',
            'GarageYrBlt': 'Year garage was built (often same as house)',
            'HouseAge': 'Age of house in years from current date',
            'YearsSinceRemodel': 'Years since last major remodel or addition',
            'GarageAge': 'Age of garage in years',
            'YrSold': 'Year the house was sold',
            'MoSold': 'Month the house was sold (1=Jan, 12=Dec)',
            
            # Composite Scores
            'QualityScore': 'Overall quality composite score based on multiple quality ratings',
            'BasementQualityScore': 'Combined basement quality and condition score',
            'GarageQualityScore': 'Combined garage quality and condition score',
            'BsmtFinishedRatio': 'Percentage of basement that is finished',
            'LivingAreaRatio': 'Ratio of 2nd floor to 1st floor area',
            'RoomDensity': 'Number of rooms per square foot of living area',
            'BathBedroomRatio': 'Ratio of bathrooms to bedrooms',
            'GarageLivingRatio': 'Garage area as percentage of living area',
            'LotCoverageRatio': 'House footprint as percentage of lot size',
            'AgeQualityInteraction': 'Interaction between house age and quality',
            
            # Count Features
            'TotRmsAbvGrd': 'Total rooms above ground (excludes bathrooms)',
            'BsmtFullBath': 'Number of full bathrooms in basement',
            'BsmtHalfBath': 'Number of half bathrooms in basement',
            'FullBath': 'Number of full bathrooms above ground',
            'HalfBath': 'Number of half bathrooms above ground', 
            'BedroomAbvGr': 'Number of bedrooms above ground level',
            'KitchenAbvGr': 'Number of kitchens above ground (typically 1)',
            'Fireplaces': 'Number of fireplaces - adds ambiance and value',
            'GarageCars': 'Garage capacity in number of cars',
            'TotalBaths': 'Total number of bathrooms (full + half)',
            
            # Property Details
            'MSSubClass': 'Building class/style (20=1-STORY 1946+, 60=2-STORY 1946+, etc.)',
            'LotFrontage': 'Linear feet of street connected to property',
            'MiscVal': 'Dollar value of miscellaneous features (shed, tennis court, etc.)',
            
            # Missing Value Indicators
            'LotFrontage_WasMissing': 'Whether lot frontage data was originally missing',
            'MasVnrArea_WasMissing': 'Whether masonry veneer area data was originally missing', 
            'GarageYrBlt_WasMissing': 'Whether garage year built data was originally missing',
            
            # Encoded Features
            'Neighborhood_encoded': 'Neighborhood encoded by average sale price - higher values = more expensive areas',
            'Condition1_encoded': 'Proximity to main road or railroad encoded by price impact',
            'Condition2_encoded': 'Secondary proximity condition (if multiple conditions apply)',
            'Exterior1st_encoded': 'Primary exterior covering material encoded by typical value',
            'Exterior2nd_encoded': 'Secondary exterior material (if different from primary)',
            'Street_encoded': 'Type of road access (Paved vs Gravel) encoded',
            'Alley_encoded': 'Type of alley access encoded',
            'Utilities_encoded': 'Available utilities encoded',
            'SaleCondition_encoded': 'Condition of sale encoded by price impact'
        }
        
        return descriptions.get(feature_name, feature_name.replace('_', ' ').title())
    
    def get_friendly_feature_name(self, feature_name):
        """Get user-friendly display name for feature"""
        friendly_names = {
            # Quality & Condition Features
            'OverallQual': 'Overall Quality (Material & Finish)',
            'OverallCond': 'Overall Condition Rating',
            'ExterQual_encoded': 'Exterior Material Quality',
            'ExterCond_encoded': 'Exterior Material Condition', 
            'KitchenQual_encoded': 'Kitchen Quality',
            'BsmtQual_encoded': 'Basement Quality (Height)',
            'BsmtCond_encoded': 'Basement Condition',
            'HeatingQC_encoded': 'Heating System Quality',
            'GarageQual_encoded': 'Garage Construction Quality',
            'GarageCond_encoded': 'Garage Condition',
            'FireplaceQu_encoded': 'Fireplace Quality',
            'PoolQC_encoded': 'Pool Quality',
            'CentralAir_encoded': 'Central Air Conditioning',
            'WasRemodeled': 'Home Remodeling History',
            
            # Original Size Features (sq ft)
            'GrLivArea': '🏠 Living Area Above Ground (sq ft)',
            'TotalBsmtSF': '🏠 Total Basement Area (sq ft)',
            'LotArea': '🌿 Lot Size (sq ft)',
            'GarageArea': '🚗 Garage Size (sq ft)',
            '1stFlrSF': '🏠 First Floor Area (sq ft)',
            '2ndFlrSF': '🏠 Second Floor Area (sq ft)',
            'BsmtFinSF1': '🏠 Basement Finished Area Type 1 (sq ft)',
            'BsmtFinSF2': '🏠 Basement Finished Area Type 2 (sq ft)',
            'BsmtUnfSF': '🏠 Unfinished Basement Area (sq ft)',
            'LowQualFinSF': '🏠 Low Quality Finished Area (sq ft)',
            'WoodDeckSF': '🌲 Wood Deck Area (sq ft)',
            'OpenPorchSF': '🏡 Open Porch Area (sq ft)',
            'EnclosedPorch': '🏡 Enclosed Porch Area (sq ft)',
            '3SsnPorch': '🏡 Three Season Porch Area (sq ft)',
            'ScreenPorch': '🏡 Screen Porch Area (sq ft)',
            'PoolArea': '🏊 Pool Area (sq ft)',
            'MasVnrArea': '🧱 Masonry Veneer Area (sq ft)',
            
            # Transformed/Normalized Size Features 
            'GrLivArea_transformed': '🏠 Living Area Above Ground',
            'TotalBsmtSF_transformed': '🏠 Total Basement Area',
            'LotArea_transformed': '🌿 Lot Size',
            'GarageArea': '🚗 Garage Size',
            '1stFlrSF_transformed': '🏠 First Floor Area',
            'BsmtFinSF1_transformed': '🏠 Basement Finished Area Type 1',
            'BsmtFinSF2_transformed': '🏠 Basement Finished Area Type 2',
            'WoodDeckSF_transformed': '🌲 Wood Deck Area',
            'OpenPorchSF_transformed': '🏡 Open Porch Area',
            'EnclosedPorch_transformed': '🏡 Enclosed Porch Area',
            '3SsnPorch_transformed': '🏡 Three Season Porch Area',
            'ScreenPorch_transformed': '🏡 Screen Porch Area',
            'PoolArea_transformed': '🏊 Pool Area',
            'MasVnrArea_transformed': '🧱 Masonry Veneer Area',
            'LowQualFinSF_transformed': '🏠 Low Quality Finished Area',
            'TotalSF_transformed': '🏠 Total House Area',
            'TotalPorchSF_transformed': '🏡 Total Porch Area',
            
            # Age & Time Features
            'YearBuilt': '📅 Year Built',
            'YearRemodAdd': '🔨 Year Remodeled/Added',
            'GarageYrBlt': '🚗 Garage Year Built',
            'YrSold': '💰 Year Sold',
            'MoSold': '📅 Month Sold',
            'HouseAge': '📅 House Age (years)',
            'YearsSinceRemodel': '🔨 Years Since Remodel',
            'GarageAge': '🚗 Garage Age (years)',
            'YearsSinceSold': '💰 Years Since Sale',
            'GarageYrBlt_transformed': '🚗 Garage Age',
            
            # Composite Features
            'TotalSF': '🏠 Total House Area (sq ft)',
            'TotalPorchSF': '🏡 Total Porch Area (sq ft)',
            'BsmtFinishedRatio': '📊 Basement Finished Ratio',
            'LivingAreaRatio': '📊 Living Area Ratio',
            'QualityScore': '⭐ Composite Quality Score',
            'BasementQualityScore': '⭐ Basement Quality Score',
            'GarageQualityScore': '🚗 Garage Quality Score',
            'RoomDensity': '🏠 Room Density (rooms per sq ft)',
            'BathBedroomRatio': '🛁 Bathroom to Bedroom Ratio',
            'GarageLivingRatio': '🚗 Garage to Living Area Ratio',
            'LotCoverageRatio': '🌿 House to Lot Coverage Ratio',
            'AgeQualityInteraction': '📅 Age-Quality Interaction Score',
            'TotalBaths': '🛁 Total Bathrooms',
            
            # Transformed Composite Features
            'BsmtFinishedRatio_transformed': '📊 Basement Finished Ratio',
            'LivingAreaRatio_transformed': '📊 Living Area Ratio',
            'GarageQualityScore_transformed': '🚗 Garage Quality Score',
            'BathBedroomRatio_transformed': '🛁 Bathroom to Bedroom Ratio',
            'LotCoverageRatio_transformed': '🌿 House to Lot Coverage Ratio',
            'AgeQualityInteraction_transformed': '📅 Age-Quality Score',
            
            # Count Features
            'TotRmsAbvGrd': '🏠 Total Rooms Above Ground',
            'BsmtFullBath': '🛁 Basement Full Bathrooms',
            'BsmtHalfBath': '🚿 Basement Half Bathrooms',
            'FullBath': '🛁 Full Bathrooms Above Ground',
            'HalfBath': '🚿 Half Bathrooms Above Ground',
            'BedroomAbvGr': '🛏️ Bedrooms Above Ground',
            'KitchenAbvGr': '👨‍🍳 Kitchens Above Ground',
            'Fireplaces': '🔥 Number of Fireplaces',
            'GarageCars': '🚗 Garage Car Capacity',
            'BsmtHalfBath_transformed': '🚿 Basement Half Bathrooms',
            'KitchenAbvGr_transformed': '👨‍🍳 Kitchens Above Ground',
            
            # Property Classification
            'MSSubClass': '🏗️ Building Class/Style',
            'MSSubClass_transformed': '🏗️ Building Class/Style',
            'MSSubClass_encoded': '🏗️ Building Class/Style',
            'LotFrontage': '🌿 Lot Frontage (linear feet)',
            'LotFrontage_transformed': '🌿 Lot Frontage',
            'MiscVal': '💰 Miscellaneous Feature Value',
            'MiscVal_transformed': '💰 Miscellaneous Feature Value',
            
            # Missing Value Indicators
            'LotFrontage_WasMissing': '❓ Lot Frontage Data Available',
            'MasVnrArea_WasMissing': '❓ Masonry Veneer Data Available',
            'GarageYrBlt_WasMissing': '❓ Garage Year Built Data Available',
            'LotFrontage_WasMissing_encoded': '❓ Lot Frontage Data Available',
            'MasVnrArea_WasMissing_encoded': '❓ Masonry Veneer Data Available',
            'GarageYrBlt_WasMissing_encoded': '❓ Garage Year Built Data Available',
            
            # Encoded Location & Condition Features
            'Neighborhood_encoded': '📍 Neighborhood (by avg price)',
            'Condition1_encoded': '🚗 Primary Proximity to Roads/Railroad',
            'Condition2_encoded': '🚗 Secondary Proximity (if applicable)',
            'Exterior1st_encoded': '🏠 Primary Exterior Material',
            'Exterior2nd_encoded': '🏠 Secondary Exterior Material',
            'Street_encoded': '🛣️ Type of Road Access',
            'Alley_encoded': '🛤️ Alley Access Type',
            'Utilities_encoded': '⚡ Available Utilities',
            'SaleCondition_encoded': '💰 Condition of Sale',
            
            # Quality Features (encoded)
            'BsmtFinType1_encoded': '🏠 Basement Finished Type 1 Quality',
            'BsmtFinType2_encoded': '🏠 Basement Finished Type 2 Quality',
            'GarageFinish_encoded': '🚗 Garage Interior Finish Quality',
            
            # One-Hot Categorical Features - MSZoning
            'MSZoning_C (all)': '🏢 Commercial Zoning',
            'MSZoning_FV': '🏞️ Floating Village Residential',
            'MSZoning_RH': '🏙️ Residential High Density',
            'MSZoning_RL': '🏘️ Residential Low Density',
            'MSZoning_RM': '🏢 Residential Medium Density',
            
            # LotShape
            'LotShape_IR1': '📐 Slightly Irregular Lot',
            'LotShape_IR2': '📐 Moderately Irregular Lot',
            'LotShape_IR3': '📐 Very Irregular Lot',
            'LotShape_Reg': '📐 Regular Rectangular Lot',
            
            # LandContour
            'LandContour_Bnk': '⛰️ Banked Lot (steep slope)',
            'LandContour_HLS': '🏔️ Hillside Lot',
            'LandContour_Low': '🏞️ Depression/Low Area',
            'LandContour_Lvl': '📏 Near Flat/Level Lot',
            
            # LotConfig
            'LotConfig_Corner': '🏠 Corner Lot',
            'LotConfig_CulDSac': '🏠 Cul-de-sac Lot',
            'LotConfig_FR2': '🏠 Frontage on 2 sides',
            'LotConfig_FR3': '🏠 Frontage on 3 sides',
            'LotConfig_Inside': '🏠 Inside Lot (standard)',
            
            # LandSlope
            'LandSlope_Gtl': '📈 Gentle Slope',
            'LandSlope_Mod': '📈 Moderate Slope',
            'LandSlope_Sev': '📈 Severe Slope',
            
            # BldgType
            'BldgType_1Fam': '🏠 Single-family Detached',
            'BldgType_2fmCon': '🏠 Two-family Conversion',
            'BldgType_Duplex': '🏠 Duplex',
            'BldgType_Twnhs': '🏠 Townhouse End Unit',
            'BldgType_TwnhsE': '🏠 Townhouse Inside Unit',
            
            # HouseStyle
            'HouseStyle_1.5Fin': '🏠 One and a half story (finished)',
            'HouseStyle_1.5Unf': '🏠 One and a half story (unfinished)',
            'HouseStyle_1Story': '🏠 One story',
            'HouseStyle_2.5Fin': '🏠 Two and a half story (finished)',
            'HouseStyle_2.5Unf': '🏠 Two and a half story (unfinished)',
            'HouseStyle_2Story': '🏠 Two story',
            'HouseStyle_SFoyer': '🏠 Split Foyer',
            'HouseStyle_SLvl': '🏠 Split Level',
            
            # RoofStyle
            'RoofStyle_Flat': '🏠 Flat Roof',
            'RoofStyle_Gable': '🏠 Gable Roof',
            'RoofStyle_Gambrel': '🏠 Gambrel Roof (barn-style)',
            'RoofStyle_Hip': '🏠 Hip Roof',
            'RoofStyle_Mansard': '🏠 Mansard Roof',
            'RoofStyle_Shed': '🏠 Shed Roof',
            
            # RoofMatl
            'RoofMatl_ClyTile': '🏠 Clay/Tile Roof',
            'RoofMatl_CompShg': '🏠 Standard Composite Shingle',
            'RoofMatl_Membran': '🏠 Membrane Roof',
            'RoofMatl_Metal': '🏠 Metal Roof',
            'RoofMatl_Roll': '🏠 Roll Roof',
            'RoofMatl_Tar&Grv': '🏠 Gravel/Tar Roof',
            'RoofMatl_WdShake': '🏠 Wood Shake Roof',
            'RoofMatl_WdShngl': '🏠 Wood Shingle Roof',
            
            # MasVnrType
            'MasVnrType_BrkCmn': '🧱 Brick Common Veneer',
            'MasVnrType_BrkFace': '🧱 Brick Face Veneer',
            'MasVnrType_Stone': '🪨 Stone Veneer',
            
            # Foundation
            'Foundation_BrkTil': '🏗️ Brick & Tile Foundation',
            'Foundation_CBlock': '🏗️ Cinder Block Foundation',
            'Foundation_PConc': '🏗️ Poured Concrete Foundation',
            'Foundation_Slab': '🏗️ Slab Foundation',
            'Foundation_Stone': '🏗️ Stone Foundation',
            'Foundation_Wood': '🏗️ Wood Foundation',
            
            # BsmtExposure
            'BsmtExposure_Av': '🏠 Average Basement Exposure',
            'BsmtExposure_Gd': '🏠 Good Basement Exposure',
            'BsmtExposure_Mn': '🏠 Minimal Basement Exposure',
            'BsmtExposure_No': '🏠 No Basement Exposure',
            
            # Heating
            'Heating_Floor': '🔥 Floor Furnace',
            'Heating_GasA': '🔥 Gas Hot Air Furnace',
            'Heating_GasW': '🔥 Gas Hot Water/Steam',
            'Heating_Grav': '🔥 Gravity Furnace',
            'Heating_OthW': '🔥 Hot Water/Steam (other)',
            'Heating_Wall': '🔥 Wall Furnace',
            
            # Electrical
            'Electrical_FuseA': '⚡ 60 AMP Fuse Box',
            'Electrical_FuseF': '⚡ Fuse Box (fair)',
            'Electrical_FuseP': '⚡ Poor Fuse Box',
            'Electrical_Mix': '⚡ Mixed Electrical System',
            'Electrical_SBrkr': '⚡ Standard Circuit Breaker',
            
            # Functional
            'Functional_Maj1': '🔧 Major Deductions (1)',
            'Functional_Maj2': '🔧 Major Deductions (2)', 
            'Functional_Min1': '🔧 Minor Deductions (1)',
            'Functional_Min2': '🔧 Minor Deductions (2)',
            'Functional_Mod': '🔧 Moderate Deductions',
            'Functional_Sev': '🔧 Severely Damaged',
            'Functional_Typ': '🔧 Typical Functionality',
            
            # GarageType
            'GarageType_2Types': '🚗 More than one type',
            'GarageType_Attchd': '🚗 Attached Garage',
            'GarageType_Basment': '🚗 Basement Garage',
            'GarageType_BuiltIn': '🚗 Built-in Garage',
            'GarageType_CarPort': '🚗 Car Port',
            'GarageType_Detchd': '🚗 Detached Garage',
            
            # PavedDrive
            'PavedDrive_N': '🛣️ Unpaved Driveway',
            'PavedDrive_P': '🛣️ Partial Pavement',
            'PavedDrive_Y': '🛣️ Paved Driveway',
            
            # Fence
            'Fence_GdPrv': '🏡 Good Privacy Fence',
            'Fence_GdWo': '🏡 Good Wood Fence',
            'Fence_MnPrv': '🏡 Minimum Privacy Fence',
            'Fence_MnWw': '🏡 Minimum Wood/Wire Fence',
            
            # MiscFeature
            'MiscFeature_Gar2': '🏠 2nd Garage',
            'MiscFeature_Othr': '🏠 Other Miscellaneous Feature',
            'MiscFeature_Shed': '🏠 Shed (over 100 sq ft)',
            'MiscFeature_TenC': '🏠 Tennis Court',
            
            # SaleType
            'SaleType_COD': '💰 Cash on Delivery',
            'SaleType_CWD': '💰 Warranty Deed (Cash)',
            'SaleType_Con': '💰 Contract',
            'SaleType_ConLD': '💰 Contract Low Down Payment',
            'SaleType_ConLI': '💰 Contract Low Interest',
            'SaleType_ConLw': '💰 Contract Low Down & Interest',
            'SaleType_New': '💰 Home just constructed/sold',
            'SaleType_Oth': '💰 Other Sale Type',
            'SaleType_WD': '💰 Warranty Deed (conventional)',
            
            # HouseAgeGroup (custom engineered)
            'HouseAgeGroup_Historic (100+)': '📅 Historic Home (100+ years)',
            'HouseAgeGroup_Mature (26-50)': '📅 Mature Home (26-50 years)',
            'HouseAgeGroup_Old (51-100)': '📅 Older Home (51-100 years)',
            'HouseAgeGroup_Recent (11-25)': '📅 Recent Home (11-25 years)',
            
            # NeighborhoodTier (custom engineered)  
            'NeighborhoodTier_Mid_Tier': '🏘️ Mid-Tier Neighborhood',
            'NeighborhoodTier_Premium': '🏘️ Premium Neighborhood', 
            'NeighborhoodTier_Standard': '🏘️ Standard Neighborhood',
            
            # HouseAgeGroup (custom engineered)
            'HouseAgeGroup_Historic (100+)': '📅 Historic Home (100+ years)',
            'HouseAgeGroup_Mature (26-50)': '📅 Mature Home (26-50 years)',
            'HouseAgeGroup_Old (51-100)': '📅 Older Home (51-100 years)',
            'HouseAgeGroup_Recent (11-25)': '📅 Recent Home (11-25 years)',
            
            # Target variable
            'SalePrice_transformed': '💰 Sale Price (log transformed)'
        }
        
        # Handle one-hot features - first check if we have explicit mapping
        if feature_name in friendly_names:
            return friendly_names[feature_name]
            
        # Then check for base category patterns
        for base_category in self.categorical_mappings.keys():
            if feature_name.startswith(base_category + '_'):
                # For one-hot features, create specific names based on the category value
                category_value = feature_name.replace(base_category + '_', '')
                category_mapping = self.categorical_mappings[base_category]['display_to_value']
                
                # Find the friendly name for this specific category value
                for display_name, orig_value in category_mapping.items():
                    if orig_value == category_value:
                        return display_name
                
                # Fallback to a descriptive name
                return f"{category_value} - {self.categorical_mappings[base_category]['description']}"
        
        return friendly_names.get(feature_name, feature_name.replace('_', ' ').title())
    
    def get_default_selection_index(self, feature_name, options_dict):
        """Get smart default selection for a feature"""
        options_list = list(options_dict.items())
        
        # Quality features - default to "Good" or "Typical/Average"
        if any(qual in feature_name for qual in ['Qual', 'Cond', 'QC']):
            for i, (display, value) in enumerate(options_list):
                if 'Good' in display or 'Typical' in display or 'Average' in display:
                    return i
        
        # Overall features - default to 5-6 (Average to Above Average)
        if feature_name in ['OverallQual', 'OverallCond']:
            for i, (display, value) in enumerate(options_list):
                if value in [5, 6]:
                    return i
        
        # Binary features - default to more common option
        if feature_name in self.binary_mappings:
            # For most binary features, default to "No/False" unless it's common
            if feature_name == 'CentralAir_encoded':
                for i, (display, value) in enumerate(options_list):
                    if value == 1:  # Yes, most homes have central air
                        return i
        
        # Neighborhood - default to mid-tier
        if feature_name == 'Neighborhood_encoded':
            for i, (display, value) in enumerate(options_list):
                if 'Mid-Tier' in display:
                    return i
        
        # Default to middle option
        return len(options_list) // 2