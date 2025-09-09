"""
PHASE 3: ADVANCED FEATURE ENGINEERING
====================================

State-of-the-art feature engineering pipeline based on EDA findings:
- Composite area features (total living space, overall area)
- House age and temporal features from year data
- Neighborhood tier classification based on price analysis
- Quality score composites from multiple quality ratings
- Bathroom and room ratio features
- Lot size categories and area ratios

Key Insights from EDA:
- Area features (GrLivArea, TotalBsmtSF, 1stFlrSF) are highly correlated - combine them
- Year features (YearBuilt, YearRemodAdd) should become age features
- Neighborhood tiers: Premium (NoRidge, NridgHt, StoneBr), Mid-tier, Standard
- Quality features can be combined into overall quality scores
- Ratios often more predictive than raw counts

Author: Advanced Data Science Pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Professional styling
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
PRIMARY_COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592941']

class AdvancedFeatureEngineeringPipeline:
    """
    Comprehensive feature engineering pipeline with domain expertise
    """
    
    def __init__(self):
        self.feature_engineering_log = {}
        self.new_features = {}
        self.neighborhood_tiers = {}
        
    def load_data(self):
        """Load processed data from Phase 2"""
        print("Loading processed data from Phase 2...")
        self.train_df = pd.read_csv('../phase2_missing_values/processed_train_phase2.csv')
        self.test_df = pd.read_csv('../phase2_missing_values/processed_test_phase2.csv')
        
        # Combine for consistent feature engineering
        self.combined_df = pd.concat([
            self.train_df.drop('SalePrice', axis=1), 
            self.test_df
        ], axis=0, ignore_index=True)
        
        print(f"Training data: {self.train_df.shape[0]} samples, {self.train_df.shape[1]} features")
        print(f"Test data: {self.test_df.shape[0]} samples, {self.test_df.shape[1]} features")
        print(f"Combined data: {self.combined_df.shape[0]} samples, {self.combined_df.shape[1]} features")
        return self
    
    def create_composite_area_features(self):
        """Create composite area features based on EDA correlation analysis"""
        print("\n" + "="*60)
        print("CREATING COMPOSITE AREA FEATURES")
        print("="*60)
        
        area_features_created = []
        
        # 1. Total Square Footage (combine all area features)
        area_components = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']
        if all(col in self.combined_df.columns for col in area_components):
            self.combined_df['TotalSF'] = (
                self.combined_df['TotalBsmtSF'] + 
                self.combined_df['1stFlrSF'] + 
                self.combined_df['2ndFlrSF']
            )
            area_features_created.append({
                'feature': 'TotalSF',
                'description': 'Total square footage (basement + 1st + 2nd floor)',
                'components': area_components,
                'min_value': self.combined_df['TotalSF'].min(),
                'max_value': self.combined_df['TotalSF'].max(),
                'mean_value': self.combined_df['TotalSF'].mean()
            })
            print(f"SUCCESS Created TotalSF: mean={self.combined_df['TotalSF'].mean():.0f} sq ft")
        
        # 2. Total Porch Area
        porch_components = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
        if all(col in self.combined_df.columns for col in porch_components):
            self.combined_df['TotalPorchSF'] = (
                self.combined_df['OpenPorchSF'] + 
                self.combined_df['EnclosedPorch'] + 
                self.combined_df['3SsnPorch'] + 
                self.combined_df['ScreenPorch']
            )
            area_features_created.append({
                'feature': 'TotalPorchSF',
                'description': 'Total porch area (all porch types)',
                'components': porch_components,
                'min_value': self.combined_df['TotalPorchSF'].min(),
                'max_value': self.combined_df['TotalPorchSF'].max(),
                'mean_value': self.combined_df['TotalPorchSF'].mean()
            })
            print(f"SUCCESS Created TotalPorchSF: mean={self.combined_df['TotalPorchSF'].mean():.0f} sq ft")
        
        # 3. Total Bathroom Count
        bathroom_components = ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']
        if all(col in self.combined_df.columns for col in bathroom_components):
            self.combined_df['TotalBaths'] = (
                self.combined_df['FullBath'] + 
                self.combined_df['HalfBath'] * 0.5 +  # Half baths count as 0.5
                self.combined_df['BsmtFullBath'] + 
                self.combined_df['BsmtHalfBath'] * 0.5
            )
            area_features_created.append({
                'feature': 'TotalBaths',
                'description': 'Total bathroom count (half baths = 0.5)',
                'components': bathroom_components,
                'min_value': self.combined_df['TotalBaths'].min(),
                'max_value': self.combined_df['TotalBaths'].max(),
                'mean_value': self.combined_df['TotalBaths'].mean()
            })
            print(f"SUCCESS Created TotalBaths: mean={self.combined_df['TotalBaths'].mean():.1f} baths")
        
        # 4. Finished Basement Ratio
        if all(col in self.combined_df.columns for col in ['BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF']):
            # Avoid division by zero
            basement_mask = self.combined_df['TotalBsmtSF'] > 0
            self.combined_df['BsmtFinishedRatio'] = 0.0
            self.combined_df.loc[basement_mask, 'BsmtFinishedRatio'] = (
                (self.combined_df.loc[basement_mask, 'BsmtFinSF1'] + 
                 self.combined_df.loc[basement_mask, 'BsmtFinSF2']) / 
                self.combined_df.loc[basement_mask, 'TotalBsmtSF']
            )
            area_features_created.append({
                'feature': 'BsmtFinishedRatio',
                'description': 'Ratio of finished basement to total basement area',
                'components': ['BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF'],
                'min_value': self.combined_df['BsmtFinishedRatio'].min(),
                'max_value': self.combined_df['BsmtFinishedRatio'].max(),
                'mean_value': self.combined_df['BsmtFinishedRatio'].mean()
            })
            print(f"SUCCESS Created BsmtFinishedRatio: mean={self.combined_df['BsmtFinishedRatio'].mean():.2f}")
        
        # 5. Living Area to Lot Area Ratio
        if all(col in self.combined_df.columns for col in ['GrLivArea', 'LotArea']):
            self.combined_df['LivingAreaRatio'] = self.combined_df['GrLivArea'] / self.combined_df['LotArea']
            area_features_created.append({
                'feature': 'LivingAreaRatio',
                'description': 'Ratio of above ground living area to lot area',
                'components': ['GrLivArea', 'LotArea'],
                'min_value': self.combined_df['LivingAreaRatio'].min(),
                'max_value': self.combined_df['LivingAreaRatio'].max(),
                'mean_value': self.combined_df['LivingAreaRatio'].mean()
            })
            print(f"SUCCESS Created LivingAreaRatio: mean={self.combined_df['LivingAreaRatio'].mean():.3f}")
        
        self.feature_engineering_log['composite_area_features'] = area_features_created
        return self
    
    def create_temporal_features(self):
        """Create age and temporal features from year data"""
        print("\n" + "="*60)
        print("CREATING TEMPORAL FEATURES")
        print("="*60)
        
        temporal_features_created = []
        current_year = 2023  # Reference year for age calculation
        
        # 1. House Age (from YearBuilt)
        if 'YearBuilt' in self.combined_df.columns:
            self.combined_df['HouseAge'] = current_year - self.combined_df['YearBuilt']
            temporal_features_created.append({
                'feature': 'HouseAge',
                'description': f'Age of house as of {current_year}',
                'source': 'YearBuilt',
                'min_value': self.combined_df['HouseAge'].min(),
                'max_value': self.combined_df['HouseAge'].max(),
                'mean_value': self.combined_df['HouseAge'].mean()
            })
            print(f"SUCCESS Created HouseAge: mean={self.combined_df['HouseAge'].mean():.1f} years")
        
        # 2. Years Since Remodel
        if all(col in self.combined_df.columns for col in ['YearRemodAdd', 'YearBuilt']):
            self.combined_df['YearsSinceRemodel'] = current_year - self.combined_df['YearRemodAdd']
            
            # Create remodel indicator (was house ever remodeled?)
            self.combined_df['WasRemodeled'] = (
                self.combined_df['YearRemodAdd'] != self.combined_df['YearBuilt']
            ).astype(int)
            
            temporal_features_created.extend([
                {
                    'feature': 'YearsSinceRemodel',
                    'description': f'Years since last remodel as of {current_year}',
                    'source': 'YearRemodAdd',
                    'min_value': self.combined_df['YearsSinceRemodel'].min(),
                    'max_value': self.combined_df['YearsSinceRemodel'].max(),
                    'mean_value': self.combined_df['YearsSinceRemodel'].mean()
                },
                {
                    'feature': 'WasRemodeled',
                    'description': 'Binary indicator if house was ever remodeled',
                    'source': 'YearRemodAdd vs YearBuilt',
                    'min_value': self.combined_df['WasRemodeled'].min(),
                    'max_value': self.combined_df['WasRemodeled'].max(),
                    'mean_value': self.combined_df['WasRemodeled'].mean()
                }
            ])
            print(f"SUCCESS Created YearsSinceRemodel: mean={self.combined_df['YearsSinceRemodel'].mean():.1f} years")
            print(f"SUCCESS Created WasRemodeled: {self.combined_df['WasRemodeled'].sum()} houses remodeled ({self.combined_df['WasRemodeled'].mean()*100:.1f}%)")
        
        # 3. Garage Age
        if 'GarageYrBlt' in self.combined_df.columns:
            # Only calculate for houses with garages (GarageYrBlt > 0)
            garage_mask = self.combined_df['GarageYrBlt'] > 0
            self.combined_df['GarageAge'] = 0
            self.combined_df.loc[garage_mask, 'GarageAge'] = current_year - self.combined_df.loc[garage_mask, 'GarageYrBlt']
            
            temporal_features_created.append({
                'feature': 'GarageAge',
                'description': f'Age of garage as of {current_year} (0 if no garage)',
                'source': 'GarageYrBlt',
                'min_value': self.combined_df['GarageAge'].min(),
                'max_value': self.combined_df['GarageAge'].max(),
                'mean_value': self.combined_df['GarageAge'].mean()
            })
            print(f"SUCCESS Created GarageAge: mean={self.combined_df['GarageAge'].mean():.1f} years")
        
        # 4. Years Since Sold (for temporal modeling)
        if 'YrSold' in self.combined_df.columns:
            self.combined_df['YearsSinceSold'] = current_year - self.combined_df['YrSold']
            temporal_features_created.append({
                'feature': 'YearsSinceSold',
                'description': f'Years since sale as of {current_year}',
                'source': 'YrSold',
                'min_value': self.combined_df['YearsSinceSold'].min(),
                'max_value': self.combined_df['YearsSinceSold'].max(),
                'mean_value': self.combined_df['YearsSinceSold'].mean()
            })
            print(f"SUCCESS Created YearsSinceSold: mean={self.combined_df['YearsSinceSold'].mean():.1f} years")
        
        # 5. House Age Groups (categorical)
        if 'HouseAge' in self.combined_df.columns:
            age_bins = [0, 10, 25, 50, 100, float('inf')]
            age_labels = ['New (0-10)', 'Recent (11-25)', 'Mature (26-50)', 'Old (51-100)', 'Historic (100+)']
            self.combined_df['HouseAgeGroup'] = pd.cut(
                self.combined_df['HouseAge'], 
                bins=age_bins, 
                labels=age_labels, 
                right=False
            )
            
            temporal_features_created.append({
                'feature': 'HouseAgeGroup',
                'description': 'Categorical house age groups',
                'source': 'HouseAge',
                'categories': age_labels,
                'distribution': self.combined_df['HouseAgeGroup'].value_counts().to_dict()
            })
            print(f"SUCCESS Created HouseAgeGroup: {len(age_labels)} categories")
        
        self.feature_engineering_log['temporal_features'] = temporal_features_created
        return self
    
    def create_neighborhood_tiers(self):
        """Create neighborhood tier classification based on EDA analysis"""
        print("\n" + "="*60)
        print("CREATING NEIGHBORHOOD TIER CLASSIFICATION")
        print("="*60)
        
        if 'Neighborhood' not in self.combined_df.columns:
            print("WARNING: Neighborhood column not found, skipping neighborhood tiers")
            return self
        
        # Based on EDA analysis: Premium neighborhoods command higher prices
        # Calculate mean price per neighborhood from training data
        neighborhood_prices = self.train_df.groupby('Neighborhood')['SalePrice'].mean().sort_values(ascending=False)
        
        print("Neighborhood price ranking (from training data):")
        for i, (neighborhood, price) in enumerate(neighborhood_prices.head(10).items(), 1):
            print(f"  {i:2d}. {neighborhood}: ${price:,.0f}")
        
        # Define tiers based on price percentiles
        price_75th = neighborhood_prices.quantile(0.75)
        price_25th = neighborhood_prices.quantile(0.25)
        
        # Tier classification
        premium_neighborhoods = neighborhood_prices[neighborhood_prices >= price_75th].index.tolist()
        mid_tier_neighborhoods = neighborhood_prices[
            (neighborhood_prices >= price_25th) & (neighborhood_prices < price_75th)
        ].index.tolist()
        standard_neighborhoods = neighborhood_prices[neighborhood_prices < price_25th].index.tolist()
        
        # Apply tier classification
        def assign_neighborhood_tier(neighborhood):
            if neighborhood in premium_neighborhoods:
                return 'Premium'
            elif neighborhood in mid_tier_neighborhoods:
                return 'Mid_Tier'
            else:
                return 'Standard'
        
        self.combined_df['NeighborhoodTier'] = self.combined_df['Neighborhood'].apply(assign_neighborhood_tier)
        
        # Store tier information
        self.neighborhood_tiers = {
            'Premium': {
                'neighborhoods': premium_neighborhoods,
                'count': len(premium_neighborhoods),
                'price_threshold': price_75th,
                'avg_price': neighborhood_prices[neighborhood_prices >= price_75th].mean()
            },
            'Mid_Tier': {
                'neighborhoods': mid_tier_neighborhoods,
                'count': len(mid_tier_neighborhoods),
                'price_range': [price_25th, price_75th],
                'avg_price': neighborhood_prices[
                    (neighborhood_prices >= price_25th) & (neighborhood_prices < price_75th)
                ].mean()
            },
            'Standard': {
                'neighborhoods': standard_neighborhoods,
                'count': len(standard_neighborhoods),
                'price_threshold': price_25th,
                'avg_price': neighborhood_prices[neighborhood_prices < price_25th].mean()
            }
        }
        
        # Distribution of tiers
        tier_distribution = self.combined_df['NeighborhoodTier'].value_counts()
        
        print(f"\nNeighborhood Tier Classification:")
        print(f"Premium Tier ({len(premium_neighborhoods)} neighborhoods): ${self.neighborhood_tiers['Premium']['avg_price']:,.0f} avg")
        print(f"  - {premium_neighborhoods}")
        print(f"Mid Tier ({len(mid_tier_neighborhoods)} neighborhoods): ${self.neighborhood_tiers['Mid_Tier']['avg_price']:,.0f} avg")
        print(f"  - {mid_tier_neighborhoods}")
        print(f"Standard Tier ({len(standard_neighborhoods)} neighborhoods): ${self.neighborhood_tiers['Standard']['avg_price']:,.0f} avg")
        print(f"  - {standard_neighborhoods}")
        
        print(f"\nTier Distribution in Combined Data:")
        for tier, count in tier_distribution.items():
            pct = (count / len(self.combined_df)) * 100
            print(f"  {tier}: {count} properties ({pct:.1f}%)")
        
        self.feature_engineering_log['neighborhood_tiers'] = {
            'tier_definitions': self.neighborhood_tiers,
            'tier_distribution': tier_distribution.to_dict()
        }
        
        return self
    
    def create_quality_composite_features(self):
        """Create composite quality scores from multiple quality ratings"""
        print("\n" + "="*60)
        print("CREATING QUALITY COMPOSITE FEATURES")
        print("="*60)
        
        quality_features_created = []
        
        # Quality feature mapping (ordinal encoding)
        quality_mapping = {
            'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0,
            'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1
        }
        
        # 1. Overall Quality Score (combine multiple quality features)
        quality_features = ['OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 
                          'BsmtQual', 'BsmtCond', 'KitchenQual']
        
        available_quality_features = [f for f in quality_features if f in self.combined_df.columns]
        
        if len(available_quality_features) >= 3:  # Need at least 3 quality features
            # Convert categorical quality to numeric
            quality_scores = pd.DataFrame()
            
            for feature in available_quality_features:
                if feature in ['OverallQual', 'OverallCond']:
                    # These are already numeric
                    quality_scores[feature] = self.combined_df[feature]
                else:
                    # Convert categorical to numeric
                    quality_scores[feature] = self.combined_df[feature].map(quality_mapping).fillna(0)
            
            # Calculate composite quality score (weighted average)
            weights = {
                'OverallQual': 0.3,    # Most important
                'OverallCond': 0.2,    # Condition matters
                'ExterQual': 0.2,      # Exterior quality
                'ExterCond': 0.1,      # Exterior condition
                'BsmtQual': 0.1,       # Basement quality
                'BsmtCond': 0.05,      # Basement condition
                'KitchenQual': 0.05    # Kitchen quality
            }
            
            # Apply weights only to available features
            available_weights = {f: weights.get(f, 0.1) for f in available_quality_features}
            weight_sum = sum(available_weights.values())
            normalized_weights = {f: w/weight_sum for f, w in available_weights.items()}
            
            self.combined_df['QualityScore'] = sum(
                quality_scores[feature] * weight 
                for feature, weight in normalized_weights.items()
            )
            
            quality_features_created.append({
                'feature': 'QualityScore',
                'description': 'Weighted composite quality score',
                'components': available_quality_features,
                'weights': normalized_weights,
                'min_value': self.combined_df['QualityScore'].min(),
                'max_value': self.combined_df['QualityScore'].max(),
                'mean_value': self.combined_df['QualityScore'].mean()
            })
            print(f"SUCCESS Created QualityScore: mean={self.combined_df['QualityScore'].mean():.2f}")
        
        # 2. Bathroom Quality Score
        bathroom_quality_features = ['KitchenQual']  # Can expand if more bathroom-specific quality features exist
        
        # 3. Basement Quality Score
        basement_quality_features = ['BsmtQual', 'BsmtCond', 'BsmtExposure']
        available_basement_features = [f for f in basement_quality_features if f in self.combined_df.columns]
        
        if len(available_basement_features) >= 2:
            basement_scores = pd.DataFrame()
            
            for feature in available_basement_features:
                basement_scores[feature] = self.combined_df[feature].map(quality_mapping).fillna(0)
            
            self.combined_df['BasementQualityScore'] = basement_scores.mean(axis=1)
            
            quality_features_created.append({
                'feature': 'BasementQualityScore',
                'description': 'Average basement quality score',
                'components': available_basement_features,
                'min_value': self.combined_df['BasementQualityScore'].min(),
                'max_value': self.combined_df['BasementQualityScore'].max(),
                'mean_value': self.combined_df['BasementQualityScore'].mean()
            })
            print(f"SUCCESS Created BasementQualityScore: mean={self.combined_df['BasementQualityScore'].mean():.2f}")
        
        # 4. Garage Quality Score
        garage_quality_features = ['GarageQual', 'GarageCond']
        available_garage_features = [f for f in garage_quality_features if f in self.combined_df.columns]
        
        if len(available_garage_features) >= 2:
            garage_scores = pd.DataFrame()
            
            for feature in available_garage_features:
                garage_scores[feature] = self.combined_df[feature].map(quality_mapping).fillna(0)
            
            self.combined_df['GarageQualityScore'] = garage_scores.mean(axis=1)
            
            quality_features_created.append({
                'feature': 'GarageQualityScore',
                'description': 'Average garage quality score',
                'components': available_garage_features,
                'min_value': self.combined_df['GarageQualityScore'].min(),
                'max_value': self.combined_df['GarageQualityScore'].max(),
                'mean_value': self.combined_df['GarageQualityScore'].mean()
            })
            print(f"SUCCESS Created GarageQualityScore: mean={self.combined_df['GarageQualityScore'].mean():.2f}")
        
        self.feature_engineering_log['quality_features'] = quality_features_created
        return self
    
    def create_ratio_and_interaction_features(self):
        """Create ratio and interaction features"""
        print("\n" + "="*60)
        print("CREATING RATIO AND INTERACTION FEATURES")
        print("="*60)
        
        ratio_features_created = []
        
        # 1. Rooms per Square Foot (room density)
        if all(col in self.combined_df.columns for col in ['TotRmsAbvGrd', 'GrLivArea']):
            # Avoid division by zero
            living_area_mask = self.combined_df['GrLivArea'] > 0
            self.combined_df['RoomDensity'] = 0.0
            self.combined_df.loc[living_area_mask, 'RoomDensity'] = (
                self.combined_df.loc[living_area_mask, 'TotRmsAbvGrd'] / 
                self.combined_df.loc[living_area_mask, 'GrLivArea'] * 1000  # Per 1000 sq ft
            )
            ratio_features_created.append({
                'feature': 'RoomDensity',
                'description': 'Rooms per 1000 sq ft of living area',
                'components': ['TotRmsAbvGrd', 'GrLivArea'],
                'mean_value': self.combined_df['RoomDensity'].mean()
            })
            print(f"SUCCESS Created RoomDensity: mean={self.combined_df['RoomDensity'].mean():.2f} rooms/1000sqft")
        
        # 2. Bath to Bedroom Ratio
        if all(col in self.combined_df.columns for col in ['TotalBaths', 'BedroomAbvGr']):
            # Avoid division by zero, use 1 as minimum bedroom count
            bedroom_count = self.combined_df['BedroomAbvGr'].clip(lower=1)
            self.combined_df['BathBedroomRatio'] = self.combined_df['TotalBaths'] / bedroom_count
            
            ratio_features_created.append({
                'feature': 'BathBedroomRatio',
                'description': 'Ratio of total bathrooms to bedrooms',
                'components': ['TotalBaths', 'BedroomAbvGr'],
                'mean_value': self.combined_df['BathBedroomRatio'].mean()
            })
            print(f"SUCCESS Created BathBedroomRatio: mean={self.combined_df['BathBedroomRatio'].mean():.2f}")
        
        # 3. Garage to Living Area Ratio
        if all(col in self.combined_df.columns for col in ['GarageArea', 'GrLivArea']):
            living_area_mask = self.combined_df['GrLivArea'] > 0
            self.combined_df['GarageLivingRatio'] = 0.0
            self.combined_df.loc[living_area_mask, 'GarageLivingRatio'] = (
                self.combined_df.loc[living_area_mask, 'GarageArea'] / 
                self.combined_df.loc[living_area_mask, 'GrLivArea']
            )
            ratio_features_created.append({
                'feature': 'GarageLivingRatio',
                'description': 'Ratio of garage area to living area',
                'components': ['GarageArea', 'GrLivArea'],
                'mean_value': self.combined_df['GarageLivingRatio'].mean()
            })
            print(f"SUCCESS Created GarageLivingRatio: mean={self.combined_df['GarageLivingRatio'].mean():.3f}")
        
        # 4. Lot Coverage Ratio
        if all(col in self.combined_df.columns for col in ['TotalSF', 'LotArea']):
            lot_area_mask = self.combined_df['LotArea'] > 0
            self.combined_df['LotCoverageRatio'] = 0.0
            self.combined_df.loc[lot_area_mask, 'LotCoverageRatio'] = (
                self.combined_df.loc[lot_area_mask, 'TotalSF'] / 
                self.combined_df.loc[lot_area_mask, 'LotArea']
            )
            ratio_features_created.append({
                'feature': 'LotCoverageRatio',
                'description': 'Ratio of total building area to lot area',
                'components': ['TotalSF', 'LotArea'],
                'mean_value': self.combined_df['LotCoverageRatio'].mean()
            })
            print(f"SUCCESS Created LotCoverageRatio: mean={self.combined_df['LotCoverageRatio'].mean():.3f}")
        
        # 5. Age-Quality Interaction
        if all(col in self.combined_df.columns for col in ['HouseAge', 'QualityScore']):
            self.combined_df['AgeQualityInteraction'] = self.combined_df['HouseAge'] * self.combined_df['QualityScore']
            ratio_features_created.append({
                'feature': 'AgeQualityInteraction',
                'description': 'Interaction between house age and quality score',
                'components': ['HouseAge', 'QualityScore'],
                'mean_value': self.combined_df['AgeQualityInteraction'].mean()
            })
            print(f"SUCCESS Created AgeQualityInteraction: mean={self.combined_df['AgeQualityInteraction'].mean():.1f}")
        
        self.feature_engineering_log['ratio_features'] = ratio_features_created
        return self
    
    def create_feature_engineering_validation(self):
        """Validate all engineered features"""
        print("\n" + "="*60)
        print("FEATURE ENGINEERING VALIDATION")
        print("="*60)
        
        # Get list of all new features created
        original_features = set(pd.read_csv('../../dataset/train.csv').columns)
        current_features = set(self.combined_df.columns)
        new_features = current_features - original_features
        
        validation_results = {
            'original_feature_count': len(original_features),
            'new_feature_count': len(new_features),
            'total_feature_count': len(current_features),
            'feature_increase_pct': (len(new_features) / len(original_features)) * 100
        }
        
        print(f"Original features: {validation_results['original_feature_count']}")
        print(f"New features created: {validation_results['new_feature_count']}")
        print(f"Total features: {validation_results['total_feature_count']}")
        print(f"Feature increase: {validation_results['feature_increase_pct']:.1f}%")
        
        print(f"\nNew Features Created:")
        new_features_list = sorted(list(new_features))
        for i, feature in enumerate(new_features_list, 1):
            print(f"  {i:2d}. {feature}")
        
        # Check for any issues
        issues = []
        
        # Check for infinite values
        for feature in new_features_list:
            if self.combined_df[feature].dtype in ['float64', 'int64']:
                inf_count = np.isinf(self.combined_df[feature]).sum()
                if inf_count > 0:
                    issues.append(f"{feature}: {inf_count} infinite values")
        
        # Check for null values in new features
        for feature in new_features_list:
            null_count = self.combined_df[feature].isnull().sum()
            if null_count > 0:
                issues.append(f"{feature}: {null_count} null values")
        
        if issues:
            print(f"\nISSUES FOUND:")
            for issue in issues:
                print(f"  WARNING: {issue}")
        else:
            print(f"\nSUCCESS: No data quality issues found in new features!")
        
        validation_results['issues'] = issues
        validation_results['new_features_list'] = new_features_list
        
        self.feature_engineering_log['validation'] = validation_results
        return self
    
    def create_feature_engineering_visualization(self):
        """Create comprehensive feature engineering visualization"""
        print("\nGenerating feature engineering visualization...")
        
        fig, axes = plt.subplots(3, 3, figsize=(22, 18))
        fig.suptitle('Feature Engineering Analysis', fontsize=16, fontweight='bold')
        
        # 1. Feature count comparison
        original_count = self.feature_engineering_log['validation']['original_feature_count']
        new_count = self.feature_engineering_log['validation']['new_feature_count']
        
        categories = ['Original Features', 'New Features']
        counts = [original_count, new_count]
        colors = [PRIMARY_COLORS[0], PRIMARY_COLORS[2]]
        
        bars = axes[0,0].bar(categories, counts, color=colors, alpha=0.8)
        axes[0,0].set_title('Feature Count: Before vs After')
        axes[0,0].set_ylabel('Number of Features')
        axes[0,0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            axes[0,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                          str(count), ha='center', va='bottom', fontweight='bold')
        
        # 2. New features by category
        feature_categories = {
            'Area/Space': ['TotalSF', 'TotalPorchSF', 'TotalBaths', 'BsmtFinishedRatio', 'LivingAreaRatio'],
            'Temporal': ['HouseAge', 'YearsSinceRemodel', 'WasRemodeled', 'GarageAge', 'YearsSinceSold', 'HouseAgeGroup'],
            'Quality': ['QualityScore', 'BasementQualityScore', 'GarageQualityScore'],
            'Ratios': ['RoomDensity', 'BathBedroomRatio', 'GarageLivingRatio', 'LotCoverageRatio'],
            'Interactions': ['AgeQualityInteraction'],
            'Location': ['NeighborhoodTier'],
            'Indicators': [col for col in self.combined_df.columns if 'WasMissing' in col]
        }
        
        category_counts = {}
        for category, features in feature_categories.items():
            available_features = [f for f in features if f in self.combined_df.columns]
            category_counts[category] = len(available_features)
        
        # Filter out empty categories
        category_counts = {k: v for k, v in category_counts.items() if v > 0}
        
        axes[0,1].pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%',
                     colors=PRIMARY_COLORS[:len(category_counts)])
        axes[0,1].set_title('New Features by Category')
        
        # 3. Sample feature distributions (TotalSF)
        if 'TotalSF' in self.combined_df.columns:
            axes[0,2].hist(self.combined_df['TotalSF'], bins=50, alpha=0.7, color=PRIMARY_COLORS[0])
            axes[0,2].set_title('Total Square Footage Distribution')
            axes[0,2].set_xlabel('Total SF')
            axes[0,2].set_ylabel('Frequency')
            axes[0,2].grid(True, alpha=0.3)
        
        # 4. House Age Distribution
        if 'HouseAge' in self.combined_df.columns:
            axes[1,0].hist(self.combined_df['HouseAge'], bins=30, alpha=0.7, color=PRIMARY_COLORS[1])
            axes[1,0].set_title('House Age Distribution')
            axes[1,0].set_xlabel('Age (Years)')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].grid(True, alpha=0.3)
        
        # 5. Neighborhood Tier Distribution
        if 'NeighborhoodTier' in self.combined_df.columns:
            tier_counts = self.combined_df['NeighborhoodTier'].value_counts()
            bars = axes[1,1].bar(tier_counts.index, tier_counts.values, 
                               color=PRIMARY_COLORS[2], alpha=0.8)
            axes[1,1].set_title('Neighborhood Tier Distribution')
            axes[1,1].set_ylabel('Number of Properties')
            axes[1,1].grid(True, alpha=0.3, axis='y')
        
        # 6. Quality Score Distribution
        if 'QualityScore' in self.combined_df.columns:
            axes[1,2].hist(self.combined_df['QualityScore'], bins=30, alpha=0.7, color=PRIMARY_COLORS[3])
            axes[1,2].set_title('Quality Score Distribution')
            axes[1,2].set_xlabel('Quality Score')
            axes[1,2].set_ylabel('Frequency')
            axes[1,2].grid(True, alpha=0.3)
        
        # 7. Bath to Bedroom Ratio
        if 'BathBedroomRatio' in self.combined_df.columns:
            axes[2,0].hist(self.combined_df['BathBedroomRatio'], bins=30, alpha=0.7, color=PRIMARY_COLORS[4])
            axes[2,0].set_title('Bath to Bedroom Ratio')
            axes[2,0].set_xlabel('Ratio')
            axes[2,0].set_ylabel('Frequency')
            axes[2,0].grid(True, alpha=0.3)
        
        # 8. Living Area Ratio
        if 'LivingAreaRatio' in self.combined_df.columns:
            axes[2,1].scatter(self.combined_df['LotArea'], self.combined_df['LivingAreaRatio'], 
                            alpha=0.6, color=PRIMARY_COLORS[0], s=20)
            axes[2,1].set_title('Living Area Ratio vs Lot Area')
            axes[2,1].set_xlabel('Lot Area')
            axes[2,1].set_ylabel('Living Area Ratio')
            axes[2,1].grid(True, alpha=0.3)
        
        # 9. Summary statistics
        axes[2,2].axis('off')
        
        # Create summary text
        summary_stats = self.feature_engineering_log['validation']
        summary_text = f"""FEATURE ENGINEERING SUMMARY:

FEATURES CREATED:
• Original Features: {summary_stats['original_feature_count']}
• New Features: {summary_stats['new_feature_count']}  
• Total Features: {summary_stats['total_feature_count']}
• Increase: {summary_stats['feature_increase_pct']:.1f}%

CATEGORIES:
• Area/Space Features: {category_counts.get('Area/Space', 0)}
• Temporal Features: {category_counts.get('Temporal', 0)}
• Quality Features: {category_counts.get('Quality', 0)}
• Ratio Features: {category_counts.get('Ratios', 0)}
• Interaction Features: {category_counts.get('Interactions', 0)}
• Location Features: {category_counts.get('Location', 0)}
• Indicator Features: {category_counts.get('Indicators', 0)}

DATA QUALITY:
• Issues Found: {len(summary_stats['issues'])}
• Status: {'CLEAN' if not summary_stats['issues'] else 'NEEDS ATTENTION'}
        """
        
        axes[2,2].text(0.05, 0.95, summary_text, transform=axes[2,2].transAxes,
                      fontsize=10, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.1))
        
        plt.tight_layout()
        plt.savefig('../../visualizations/phase3_feature_engineering.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return self
    
    def save_feature_engineering_artifacts(self):
        """Save feature engineering results and configuration"""
        print("\nSaving feature engineering artifacts...")
        
        # Split back into train and test
        train_size = len(self.train_df)
        
        engineered_train = self.combined_df.iloc[:train_size].copy()
        engineered_test = self.combined_df.iloc[train_size:].copy()
        
        # Add back SalePrice to training data
        engineered_train['SalePrice'] = self.train_df['SalePrice'].values
        
        # Save datasets
        engineered_train.to_csv('engineered_train_phase3.csv', index=False)
        engineered_test.to_csv('engineered_test_phase3.csv', index=False)
        
        # Save feature engineering configuration
        import json
        
        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            return obj
        
        def recursive_convert(item):
            if isinstance(item, dict):
                return {key: recursive_convert(value) for key, value in item.items()}
            elif isinstance(item, list):
                return [recursive_convert(element) for element in item]
            else:
                return convert_types(item)
        
        config = {
            'feature_engineering_log': recursive_convert(self.feature_engineering_log),
            'neighborhood_tiers': recursive_convert(self.neighborhood_tiers),
            'new_features_created': self.feature_engineering_log['validation']['new_features_list']
        }
        
        with open('feature_engineering_config.json', 'w') as f:
            json.dump(config, f, indent=4)
        
        print("SUCCESS Engineered training data saved to engineered_train_phase3.csv")
        print("SUCCESS Engineered test data saved to engineered_test_phase3.csv")
        print("SUCCESS Feature engineering config saved to feature_engineering_config.json")
        
        return self
    
    def run_complete_pipeline(self):
        """Execute the complete feature engineering pipeline"""
        print("PHASE 3: ADVANCED FEATURE ENGINEERING PIPELINE")
        print("="*70)
        
        (self.load_data()
         .create_composite_area_features()
         .create_temporal_features()
         .create_neighborhood_tiers()
         .create_quality_composite_features()
         .create_ratio_and_interaction_features()
         .create_feature_engineering_validation()
         .create_feature_engineering_visualization()
         .save_feature_engineering_artifacts())
        
        print("\n" + "="*70)
        print("SUCCESS PHASE 3 COMPLETED SUCCESSFULLY!")
        print("="*70)
        validation = self.feature_engineering_log['validation']
        print(f"RESULT Original features: {validation['original_feature_count']}")
        print(f"RESULT New features created: {validation['new_feature_count']}")
        print(f"RESULT Feature increase: {validation['feature_increase_pct']:.1f}%")
        print("TARGET Ready for Phase 4: Distribution Transformations")
        print("="*70)
        
        return self

if __name__ == "__main__":
    pipeline = AdvancedFeatureEngineeringPipeline()
    pipeline.run_complete_pipeline()