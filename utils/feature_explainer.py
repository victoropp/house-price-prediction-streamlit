"""
Feature Explanation Utilities
Converts technical feature names to user-friendly explanations for SHAP analysis
"""

from utils.neighborhood_mapper import get_neighborhood_info, get_neighborhood_impact_explanation


class FeatureExplainer:
    """Converts technical feature names and values to user-friendly explanations"""
    
    def __init__(self):
        self.feature_descriptions = {
            # Numerical features
            'OverallQual': 'Overall Material/Finish Quality',
            'GrLivArea': 'Above Ground Living Area (sq ft)',
            'TotalSF': 'Total Living Space (sq ft)', 
            'GarageArea': 'Garage Size (sq ft)',
            'TotalBsmtSF': 'Total Basement Area (sq ft)',
            '1stFlrSF': 'First Floor Area (sq ft)',
            '2ndFlrSF': 'Second Floor Area (sq ft)',
            'YearBuilt': 'Year Built',
            'YearRemodAdd': 'Year Remodeled/Added',
            'LotArea': 'Lot Size (sq ft)',
            'LotFrontage': 'Lot Frontage (linear feet)',
            'MasVnrArea': 'Masonry Veneer Area (sq ft)',
            'BsmtFinSF1': 'Finished Basement Area Type 1 (sq ft)',
            'WoodDeckSF': 'Wood Deck Area (sq ft)',
            'OpenPorchSF': 'Open Porch Area (sq ft)',
            'EnclosedPorch': 'Enclosed Porch Area (sq ft)',
            'ScreenPorch': 'Screen Porch Area (sq ft)',
            'PoolArea': 'Pool Area (sq ft)',
            'GarageYrBlt': 'Garage Year Built',
            'MiscVal': 'Miscellaneous Feature Value ($)',
            'TotRmsAbvGrd': 'Total Rooms Above Ground',
            'Fireplaces': 'Number of Fireplaces',
            'FullBath': 'Full Bathrooms Above Grade',
            'HalfBath': 'Half Bathrooms Above Grade', 
            'BedroomAbvGr': 'Bedrooms Above Grade',
            'KitchenAbvGr': 'Kitchens Above Grade',
            'GarageCars': 'Garage Car Capacity',
            
            # Condition/Quality scores (encoded)
            'ExterQual_encoded': 'Exterior Material Quality',
            'ExterCond_encoded': 'Exterior Material Condition',
            'BsmtQual_encoded': 'Basement Quality',
            'BsmtCond_encoded': 'Basement Condition',
            'HeatingQC_encoded': 'Heating Quality & Condition',
            'KitchenQual_encoded': 'Kitchen Quality',
            'FireplaceQu_encoded': 'Fireplace Quality',
            'GarageQual_encoded': 'Garage Quality',
            'GarageCond_encoded': 'Garage Condition',
            'PoolQC_encoded': 'Pool Quality',
            
            # Categorical features (encoded)
            'Neighborhood_encoded': 'Neighborhood Location',
            'MSSubClass_encoded': 'Building Class/Type',
            'Exterior1st_encoded': 'Exterior Covering',
            'Exterior2nd_encoded': 'Exterior Covering (2nd)',
            'Condition1_encoded': 'Proximity to Various Conditions',
            'SaleCondition_encoded': 'Sale Condition',
            'SaleType_encoded': 'Type of Sale',
            
            # Engineered features
            'HouseAge': 'House Age (years)',
            'YearsSinceRemodel': 'Years Since Last Remodel',
            'GarageAge': 'Garage Age (years)',
            'TotalBaths': 'Total Bathrooms',
            'QualityScore': 'Overall Quality Index',
            'BasementQualityScore': 'Basement Quality Index',
            'GarageQualityScore': 'Garage Quality Index',
            'AgeQualityInteraction': 'Age-Quality Balance',
            'LivingAreaRatio': 'Living Area Distribution',
            'BathBedroomRatio': 'Bath-to-Bedroom Ratio',
            'RoomDensity': 'Room Density Index',
            'LotCoverageRatio': 'House-to-Lot Coverage',
            'GarageLivingRatio': 'Garage-to-Living Ratio',
            'BsmtFinishedRatio': 'Basement Finished Percentage',
            'TotalPorchSF': 'Total Porch Area (sq ft)',
            'WasRemodeled': 'Has Been Remodeled',
            'YearsSinceSold': 'Years Since Sale'
        }
        
        # Quality/condition scale mappings
        self.quality_scales = {
            'ExterQual_encoded': {1: 'Poor', 2: 'Fair', 3: 'Average', 4: 'Good', 5: 'Excellent'},
            'ExterCond_encoded': {1: 'Poor', 2: 'Fair', 3: 'Average', 4: 'Good', 5: 'Excellent'},
            'BsmtQual_encoded': {0: 'No Basement', 1: 'Poor', 2: 'Fair', 3: 'Average', 4: 'Good', 5: 'Excellent'},
            'BsmtCond_encoded': {0: 'No Basement', 1: 'Poor', 2: 'Fair', 3: 'Average', 4: 'Good', 5: 'Excellent'},
            'HeatingQC_encoded': {1: 'Poor', 2: 'Fair', 3: 'Average', 4: 'Good', 5: 'Excellent'},
            'KitchenQual_encoded': {1: 'Poor', 2: 'Fair', 3: 'Average', 4: 'Good', 5: 'Excellent'},
            'FireplaceQu_encoded': {0: 'No Fireplace', 1: 'Poor', 2: 'Fair', 3: 'Average', 4: 'Good', 5: 'Excellent'},
            'GarageQual_encoded': {0: 'No Garage', 1: 'Poor', 2: 'Fair', 3: 'Average', 4: 'Good', 5: 'Excellent'},
            'GarageCond_encoded': {0: 'No Garage', 1: 'Poor', 2: 'Fair', 3: 'Average', 4: 'Good', 5: 'Excellent'},
            'PoolQC_encoded': {0: 'No Pool', 1: 'Poor', 2: 'Fair', 3: 'Average', 4: 'Good', 5: 'Excellent'},
            'OverallQual': {1: 'Very Poor', 2: 'Poor', 3: 'Fair', 4: 'Below Average', 5: 'Average', 
                          6: 'Above Average', 7: 'Good', 8: 'Very Good', 9: 'Excellent', 10: 'Very Excellent'},
            'OverallCond': {1: 'Very Poor', 2: 'Poor', 3: 'Fair', 4: 'Below Average', 5: 'Average',
                          6: 'Above Average', 7: 'Good', 8: 'Very Good', 9: 'Excellent', 10: 'Very Excellent'}
        }
    
    def get_friendly_feature_name(self, feature_name: str) -> str:
        """Get user-friendly feature name"""
        return self.feature_descriptions.get(feature_name, feature_name.replace('_', ' ').title())
    
    def format_feature_value(self, feature_name: str, value) -> str:
        """Format feature value with appropriate units and descriptions"""
        try:
            # Handle neighborhood specially
            if feature_name == 'Neighborhood_encoded':
                neighborhood_info = get_neighborhood_info(float(value))
                return f"{neighborhood_info['name']} ({neighborhood_info['tier']})"
            
            # Handle quality/condition scales
            if feature_name in self.quality_scales:
                scale = self.quality_scales[feature_name]
                int_value = int(round(float(value)))
                scale_desc = scale.get(int_value, f"Level {int_value}")
                return f"{scale_desc} ({int_value}/10)" if 'Overall' in feature_name else f"{scale_desc}"
            
            # Handle boolean features
            if feature_name == 'WasRemodeled':
                return "Yes" if value > 0.5 else "No"
            
            # Handle area measurements
            if any(area_term in feature_name for area_term in ['SF', 'Area']):
                return f"{int(value):,} sq ft" if value > 0 else "None"
            
            # Handle counts
            if any(count_term in feature_name for count_term in ['Bath', 'Bedroom', 'Room', 'Fireplace', 'Car', 'Kitchen']):
                return f"{int(value)}"
            
            # Handle years
            if 'Year' in feature_name and not 'Since' in feature_name and not 'Age' in feature_name:
                return f"{int(value)}"
            
            # Handle age features
            if 'Age' in feature_name or 'Since' in feature_name:
                return f"{int(value)} years"
            
            # Handle ratios and percentages
            if 'Ratio' in feature_name:
                return f"{value:.2f}"
            
            # Handle scores and indices
            if any(score_term in feature_name for score_term in ['Score', 'Index', 'Interaction']):
                return f"{value:.1f}"
            
            # Handle dollar values
            if 'Val' in feature_name:
                return f"${int(value):,}"
            
            # Handle frontage
            if 'Frontage' in feature_name:
                return f"{int(value)} ft" if value > 0 else "None"
            
            # Default formatting
            if isinstance(value, float):
                return f"{value:.1f}"
            else:
                return str(value)
                
        except (ValueError, TypeError):
            return str(value)
    
    def explain_feature_impact(self, feature_name: str, feature_value, impact_score: float) -> str:
        """Create comprehensive explanation of feature impact on price prediction"""
        
        friendly_name = self.get_friendly_feature_name(feature_name)
        formatted_value = self.format_feature_value(feature_name, feature_value)
        impact_pct = abs(impact_score) * 100
        
        # Direction indicators
        if impact_score > 0:
            direction = "increases"
            icon = "📈"
            trend = "positive"
        else:
            direction = "decreases"
            icon = "📉" 
            trend = "negative"
        
        # Special handling for neighborhood
        if feature_name == 'Neighborhood_encoded':
            return get_neighborhood_impact_explanation(float(feature_value), impact_score)
        
        # Impact magnitude descriptions
        if impact_pct >= 3:
            magnitude = "strongly"
        elif impact_pct >= 1:
            magnitude = "moderately"
        else:
            magnitude = "slightly"
        
        # Feature-specific insights
        insights = self._get_feature_insight(feature_name, feature_value, trend)
        
        explanation = f"{icon} **{friendly_name}**: {formatted_value} - {magnitude} {direction} price by {impact_pct:.1f}%"
        
        if insights:
            explanation += f" • {insights}"
        
        return explanation
    
    def _get_feature_insight(self, feature_name: str, value, trend: str) -> str:
        """Get contextual insights for specific features"""
        try:
            insights = {
                'OverallQual': {
                    'positive': "Higher quality materials and finishes command premium prices",
                    'negative': "Lower quality construction reduces market value"
                },
                'GrLivArea': {
                    'positive': "Larger living spaces are highly valued by buyers",
                    'negative': "Smaller living areas reduce overall appeal"
                },
                'TotalSF': {
                    'positive': "More total square footage increases property value",
                    'negative': "Limited space constrains market value"
                },
                'YearBuilt': {
                    'positive': "Newer construction appeals to modern buyers",
                    'negative': "Older homes may need more updates"
                },
                'GarageArea': {
                    'positive': "Ample parking/storage space adds convenience",
                    'negative': "Limited garage space reduces utility"
                },
                'HouseAge': {
                    'positive': "Well-maintained older homes can have character value",
                    'negative': "Older homes typically need more maintenance"
                },
                'TotalBaths': {
                    'positive': "More bathrooms increase convenience and value",
                    'negative': "Fewer bathrooms can limit buyer appeal"
                },
                'Fireplaces': {
                    'positive': "Fireplaces add comfort and aesthetic appeal",
                    'negative': "No fireplaces means missing this desirable feature"
                }
            }
            
            feature_insights = insights.get(feature_name, {})
            return feature_insights.get(trend, "")
            
        except:
            return ""
    
    def create_enhanced_shap_explanation(self, top_features, input_data) -> str:
        """Create enhanced SHAP explanation with user-friendly descriptions"""
        explanations = []
        
        for i, (feature, importance) in enumerate(top_features[:5]):
            if feature in input_data:
                feature_value = input_data[feature]
                explanation = self.explain_feature_impact(feature, feature_value, importance)
                explanations.append(f"**{i+1}.** {explanation}")
        
        return "\n\n".join(explanations)