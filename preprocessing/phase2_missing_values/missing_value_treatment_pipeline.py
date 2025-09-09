"""
PHASE 2: STRATEGIC MISSING VALUE TREATMENT
==========================================

State-of-the-art missing value treatment pipeline based on EDA findings:
- Domain-knowledge driven imputation strategies
- Systematic vs random missing pattern analysis  
- Advanced imputation techniques (KNN, MICE, Domain-specific)
- Missing value indicator feature engineering

Key Findings from EDA:
- PoolQC (99.5% missing) - Indicates "No Pool"
- MiscFeature (96.3% missing) - Indicates "No Misc Feature"  
- Alley (93.8% missing) - Indicates "No Alley Access"
- LotFrontage (17.7% missing) - Needs intelligent imputation
- Garage features (5.5% missing) - Systematic missing pattern

Author: Advanced Data Science Pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Professional styling
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
PRIMARY_COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592941']

class MissingValueTreatmentPipeline:
    """
    Advanced missing value treatment with domain knowledge integration
    """
    
    def __init__(self):
        self.missing_patterns = {}
        self.imputation_strategies = {}
        self.feature_engineering_log = {}
        
    def load_data(self):
        """Load both training and test datasets"""
        print("Loading training and test datasets...")
        self.train_df = pd.read_csv('../../dataset/train.csv')
        self.test_df = pd.read_csv('../../dataset/test.csv')
        
        # Combine for consistent missing value treatment
        self.combined_df = pd.concat([
            self.train_df.drop('SalePrice', axis=1), 
            self.test_df
        ], axis=0, ignore_index=True)
        
        print(f"Training data: {self.train_df.shape[0]} samples")
        print(f"Test data: {self.test_df.shape[0]} samples") 
        print(f"Combined data: {self.combined_df.shape[0]} samples")
        return self
    
    def analyze_missing_patterns(self):
        """Comprehensive missing value pattern analysis"""
        print("\n" + "="*60)
        print("MISSING VALUE PATTERN ANALYSIS")
        print("="*60)
        
        missing_info = []
        
        for col in self.combined_df.columns:
            missing_count = self.combined_df[col].isnull().sum()
            missing_pct = (missing_count / len(self.combined_df)) * 100
            
            if missing_count > 0:
                # Determine missing pattern type
                if missing_pct > 90:
                    pattern_type = "Systematic_High" 
                elif missing_pct > 50:
                    pattern_type = "Systematic_Moderate"
                elif missing_pct > 15:
                    pattern_type = "Random_Moderate"
                else:
                    pattern_type = "Random_Low"
                
                missing_info.append({
                    'Feature': col,
                    'Missing_Count': missing_count,
                    'Missing_Percentage': missing_pct,
                    'Pattern_Type': pattern_type,
                    'Data_Type': str(self.combined_df[col].dtype)
                })
        
        self.missing_df = pd.DataFrame(missing_info).sort_values('Missing_Percentage', ascending=False)
        
        print(f"Features with missing values: {len(self.missing_df)}")
        print("\nMissing Value Summary by Pattern Type:")
        pattern_summary = self.missing_df.groupby('Pattern_Type').agg({
            'Feature': 'count',
            'Missing_Percentage': ['min', 'max', 'mean']
        }).round(2)
        print(pattern_summary)
        
        print(f"\nTop 10 Missing Value Features:")
        display_cols = ['Feature', 'Missing_Count', 'Missing_Percentage', 'Pattern_Type']
        print(self.missing_df[display_cols].head(10).to_string(index=False))
        
        return self
    
    def design_imputation_strategies(self):
        """Design domain-knowledge driven imputation strategies"""
        print("\n" + "="*60)
        print("IMPUTATION STRATEGY DESIGN")
        print("="*60)
        
        # Strategy 1: High Missing Categorical (>90%) - Domain Knowledge
        high_missing_categorical = {
            'PoolQC': 'None',  # No pool
            'MiscFeature': 'None',  # No misc feature
            'Alley': 'None',  # No alley access
            'Fence': 'None',  # No fence
            'FireplaceQu': 'None'  # No fireplace
        }
        
        # Strategy 2: Garage Feature Group (Systematic Missing)
        garage_features = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
        garage_numerical = ['GarageYrBlt', 'GarageArea', 'GarageCars']
        
        # Strategy 3: Basement Feature Group (Systematic Missing)
        basement_features = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
        basement_numerical = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']
        
        # Strategy 4: Moderate Missing - Advanced Imputation
        moderate_missing = ['LotFrontage', 'MasVnrType', 'MasVnrArea', 'Electrical']
        
        self.imputation_strategies = {
            'high_missing_categorical': high_missing_categorical,
            'garage_group': {'categorical': garage_features, 'numerical': garage_numerical},
            'basement_group': {'categorical': basement_features, 'numerical': basement_numerical},
            'moderate_missing': moderate_missing
        }
        
        print("DESIGNED IMPUTATION STRATEGIES:")
        print("1. High Missing Categorical (>90%): Domain 'None' replacement")
        for feature, replacement in high_missing_categorical.items():
            if feature in self.combined_df.columns:
                print(f"   - {feature}: '{replacement}'")
        
        print("2. Garage Feature Group: Systematic imputation")
        print(f"   - Categorical: {garage_features}")
        print(f"   - Numerical: {garage_numerical}")
        
        print("3. Basement Feature Group: Systematic imputation") 
        print(f"   - Categorical: {basement_features}")
        print(f"   - Numerical: {basement_numerical}")
        
        print("4. Moderate Missing: Advanced imputation (KNN/MICE)")
        print(f"   - Features: {moderate_missing}")
        
        return self
    
    def apply_domain_knowledge_imputation(self):
        """Apply domain knowledge based imputation"""
        print("\n" + "="*60)
        print("APPLYING DOMAIN KNOWLEDGE IMPUTATION")
        print("="*60)
        
        # Create working copy
        self.processed_df = self.combined_df.copy()
        imputation_log = {}
        
        # Strategy 1: High missing categorical features
        for feature, replacement_value in self.imputation_strategies['high_missing_categorical'].items():
            if feature in self.processed_df.columns:
                before_missing = self.processed_df[feature].isnull().sum()
                self.processed_df[feature] = self.processed_df[feature].fillna(replacement_value)
                after_missing = self.processed_df[feature].isnull().sum()
                imputation_log[feature] = {
                    'strategy': 'Domain_Knowledge',
                    'before': before_missing,
                    'after': after_missing,
                    'value': replacement_value
                }
                print(f"SUCCESS {feature}: {before_missing} -> {after_missing} (filled with '{replacement_value}')")
        
        # Strategy 2: Garage features (systematic missing)
        garage_features = self.imputation_strategies['garage_group']
        
        # Identify records with no garage (all garage features missing)
        garage_missing_mask = self.processed_df['GarageType'].isnull()
        
        # Categorical garage features
        for feature in garage_features['categorical']:
            if feature in self.processed_df.columns:
                before_missing = self.processed_df[feature].isnull().sum()
                self.processed_df.loc[garage_missing_mask, feature] = 'None'
                after_missing = self.processed_df[feature].isnull().sum()
                imputation_log[feature] = {
                    'strategy': 'Systematic_None',
                    'before': before_missing,
                    'after': after_missing,
                    'value': 'None'
                }
                print(f"SUCCESS {feature}: {before_missing} -> {after_missing} (systematic 'None')")
        
        # Numerical garage features
        for feature in garage_features['numerical']:
            if feature in self.processed_df.columns:
                before_missing = self.processed_df[feature].isnull().sum()
                if feature == 'GarageYrBlt':
                    # Set garage year built to 0 for no garage
                    self.processed_df.loc[garage_missing_mask, feature] = 0
                else:
                    # Set garage area/cars to 0 for no garage
                    self.processed_df.loc[garage_missing_mask, feature] = 0
                after_missing = self.processed_df[feature].isnull().sum()
                imputation_log[feature] = {
                    'strategy': 'Systematic_Zero',
                    'before': before_missing,
                    'after': after_missing,
                    'value': 0
                }
                print(f"SUCCESS {feature}: {before_missing} -> {after_missing} (systematic 0)")
        
        # Strategy 3: Basement features (systematic missing)
        basement_features = self.imputation_strategies['basement_group']
        
        # Identify records with no basement
        basement_missing_mask = (
            self.processed_df['BsmtQual'].isnull() & 
            self.processed_df['BsmtCond'].isnull() &
            self.processed_df['BsmtExposure'].isnull()
        )
        
        # Categorical basement features
        for feature in basement_features['categorical']:
            if feature in self.processed_df.columns:
                before_missing = self.processed_df[feature].isnull().sum()
                self.processed_df.loc[basement_missing_mask, feature] = 'None'
                after_missing = self.processed_df[feature].isnull().sum()
                imputation_log[feature] = {
                    'strategy': 'Systematic_None',
                    'before': before_missing,
                    'after': after_missing,
                    'value': 'None'
                }
                print(f"SUCCESS {feature}: {before_missing} -> {after_missing} (systematic 'None')")
        
        # Numerical basement features  
        for feature in basement_features['numerical']:
            if feature in self.processed_df.columns:
                before_missing = self.processed_df[feature].isnull().sum()
                self.processed_df.loc[basement_missing_mask, feature] = 0
                after_missing = self.processed_df[feature].isnull().sum()
                imputation_log[feature] = {
                    'strategy': 'Systematic_Zero',
                    'before': before_missing,
                    'after': after_missing,
                    'value': 0
                }
                print(f"SUCCESS {feature}: {before_missing} -> {after_missing} (systematic 0)")
        
        self.domain_imputation_log = imputation_log
        return self
    
    def apply_advanced_imputation(self):
        """Apply advanced imputation for moderate missing values"""
        print("\n" + "="*60)
        print("APPLYING ADVANCED IMPUTATION TECHNIQUES")
        print("="*60)
        
        advanced_log = {}
        
        # LotFrontage: Use KNN based on similar properties
        if 'LotFrontage' in self.processed_df.columns:
            print("Applying KNN imputation for LotFrontage...")
            
            # Features for KNN imputation of LotFrontage
            knn_features = ['LotArea', 'LotShape', 'LotConfig', 'Neighborhood']
            
            # Prepare data for KNN
            lot_data = self.processed_df[knn_features + ['LotFrontage']].copy()
            
            # Encode categorical features for KNN
            for col in ['LotShape', 'LotConfig', 'Neighborhood']:
                if col in lot_data.columns:
                    lot_data[col] = pd.Categorical(lot_data[col]).codes
            
            # Apply KNN imputation
            knn_imputer = KNNImputer(n_neighbors=5, weights='uniform')
            before_missing = self.processed_df['LotFrontage'].isnull().sum()
            
            lot_data_imputed = knn_imputer.fit_transform(lot_data)
            self.processed_df['LotFrontage'] = lot_data_imputed[:, -1]  # Last column is LotFrontage
            
            after_missing = self.processed_df['LotFrontage'].isnull().sum()
            advanced_log['LotFrontage'] = {
                'strategy': 'KNN_Imputation',
                'before': before_missing,
                'after': after_missing,
                'parameters': 'n_neighbors=5'
            }
            print(f"SUCCESS LotFrontage: {before_missing} -> {after_missing} (KNN imputation)")
        
        # MasVnrType and MasVnrArea: Paired imputation
        if 'MasVnrType' in self.processed_df.columns and 'MasVnrArea' in self.processed_df.columns:
            print("Applying paired imputation for MasVnrType and MasVnrArea...")
            
            # Most houses without masonry veneer type also have area = 0
            masonry_missing_mask = self.processed_df['MasVnrType'].isnull()
            
            before_missing_type = self.processed_df['MasVnrType'].isnull().sum()
            before_missing_area = self.processed_df['MasVnrArea'].isnull().sum()
            
            # Fill missing MasVnrType with 'None' and MasVnrArea with 0
            self.processed_df.loc[masonry_missing_mask, 'MasVnrType'] = 'None'
            self.processed_df.loc[masonry_missing_mask, 'MasVnrArea'] = 0
            
            after_missing_type = self.processed_df['MasVnrType'].isnull().sum()
            after_missing_area = self.processed_df['MasVnrArea'].isnull().sum()
            
            advanced_log['MasVnrType'] = {
                'strategy': 'Paired_Domain',
                'before': before_missing_type,
                'after': after_missing_type,
                'value': 'None'
            }
            advanced_log['MasVnrArea'] = {
                'strategy': 'Paired_Domain',  
                'before': before_missing_area,
                'after': after_missing_area,
                'value': 0
            }
            print(f"SUCCESS MasVnrType: {before_missing_type} -> {after_missing_type} (paired with area)")
            print(f"SUCCESS MasVnrArea: {before_missing_area} -> {after_missing_area} (paired with type)")
        
        # Electrical: Mode imputation (most common value)
        if 'Electrical' in self.processed_df.columns:
            before_missing = self.processed_df['Electrical'].isnull().sum()
            if before_missing > 0:
                mode_value = self.processed_df['Electrical'].mode().iloc[0]
                self.processed_df['Electrical'] = self.processed_df['Electrical'].fillna(mode_value)
                after_missing = self.processed_df['Electrical'].isnull().sum()
                advanced_log['Electrical'] = {
                    'strategy': 'Mode_Imputation',
                    'before': before_missing,
                    'after': after_missing,
                    'value': mode_value
                }
                print(f"SUCCESS Electrical: {before_missing} -> {after_missing} (mode: {mode_value})")
        
        self.advanced_imputation_log = advanced_log
        return self
    
    def create_missing_value_indicators(self):
        """Create indicator features for originally missing values"""
        print("\n" + "="*60)
        print("CREATING MISSING VALUE INDICATORS")
        print("="*60)
        
        # Features where missingness might be informative
        indicator_features = ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
        
        indicators_created = []
        
        for feature in indicator_features:
            if feature in self.combined_df.columns:  # Check original data
                indicator_name = f'{feature}_WasMissing'
                # Create indicator based on original missing values
                self.processed_df[indicator_name] = self.combined_df[feature].isnull().astype(int)
                
                missing_count = self.processed_df[indicator_name].sum()
                indicators_created.append({
                    'Original_Feature': feature,
                    'Indicator_Feature': indicator_name,
                    'Missing_Count': missing_count,
                    'Missing_Percentage': (missing_count / len(self.processed_df)) * 100
                })
                
                print(f"SUCCESS Created {indicator_name}: {missing_count} missing indicators")
        
        self.missing_indicators = pd.DataFrame(indicators_created)
        self.feature_engineering_log['missing_indicators'] = indicators_created
        
        return self
    
    def validate_imputation_results(self):
        """Comprehensive validation of imputation results"""
        print("\n" + "="*60)
        print("IMPUTATION RESULTS VALIDATION")  
        print("="*60)
        
        # Check remaining missing values
        remaining_missing = self.processed_df.isnull().sum()
        remaining_missing = remaining_missing[remaining_missing > 0]
        
        print(f"Remaining missing values: {len(remaining_missing)} features")
        if len(remaining_missing) > 0:
            print("Features still with missing values:")
            for feature, count in remaining_missing.items():
                pct = (count / len(self.processed_df)) * 100
                print(f"  - {feature}: {count} ({pct:.2f}%)")
        else:
            print("SUCCESS: No missing values remaining!")
        
        # Validation metrics
        original_missing_total = self.combined_df.isnull().sum().sum()
        processed_missing_total = self.processed_df.isnull().sum().sum()
        
        self.validation_results = {
            'original_missing_total': original_missing_total,
            'processed_missing_total': processed_missing_total,
            'missing_reduction': original_missing_total - processed_missing_total,
            'missing_reduction_pct': ((original_missing_total - processed_missing_total) / original_missing_total) * 100,
            'features_with_remaining_missing': len(remaining_missing),
            'indicators_created': len(self.missing_indicators)
        }
        
        print(f"\nVALIDATION SUMMARY:")
        print(f"Original missing values: {original_missing_total:,}")
        print(f"Processed missing values: {processed_missing_total:,}")
        print(f"Missing reduction: {self.validation_results['missing_reduction']:,} ({self.validation_results['missing_reduction_pct']:.1f}%)")
        print(f"Missing value indicators created: {self.validation_results['indicators_created']}")
        
        return self
    
    def create_imputation_visualization(self):
        """Create comprehensive imputation visualization"""
        print("\nGenerating imputation visualization...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Missing Value Treatment Analysis', fontsize=16, fontweight='bold')
        
        # 1. Before/After missing value comparison
        before_missing = self.combined_df.isnull().sum()
        after_missing = self.processed_df.isnull().sum()
        
        # Get features that had missing values
        had_missing = before_missing[before_missing > 0].index[:15]  # Top 15
        
        x = np.arange(len(had_missing))
        width = 0.35
        
        bars1 = axes[0,0].bar(x - width/2, before_missing[had_missing], width, 
                             label='Before', color=PRIMARY_COLORS[0], alpha=0.8)
        bars2 = axes[0,0].bar(x + width/2, after_missing[had_missing], width,
                             label='After', color=PRIMARY_COLORS[2], alpha=0.8)
        
        axes[0,0].set_xlabel('Features')
        axes[0,0].set_ylabel('Missing Count')  
        axes[0,0].set_title('Before vs After Imputation')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(had_missing, rotation=45, ha='right')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3, axis='y')
        
        # 2. Imputation strategy distribution
        all_logs = {**self.domain_imputation_log, **self.advanced_imputation_log}
        strategies = [log['strategy'] for log in all_logs.values()]
        strategy_counts = pd.Series(strategies).value_counts()
        
        axes[0,1].pie(strategy_counts.values, labels=strategy_counts.index, autopct='%1.1f%%',
                     colors=PRIMARY_COLORS[:len(strategy_counts)])
        axes[0,1].set_title('Imputation Strategy Distribution')
        
        # 3. Missing value reduction by feature
        reduction_data = []
        for feature in had_missing:
            reduction = before_missing[feature] - after_missing[feature]
            reduction_pct = (reduction / before_missing[feature]) * 100 if before_missing[feature] > 0 else 0
            reduction_data.append(reduction_pct)
        
        bars3 = axes[0,2].bar(range(len(had_missing)), reduction_data, 
                             color=PRIMARY_COLORS[3], alpha=0.8)
        axes[0,2].set_xlabel('Features')
        axes[0,2].set_ylabel('Missing Reduction (%)')
        axes[0,2].set_title('Missing Value Reduction by Feature')
        axes[0,2].set_xticks(range(len(had_missing)))
        axes[0,2].set_xticklabels(had_missing, rotation=45, ha='right')
        axes[0,2].grid(True, alpha=0.3, axis='y')
        
        # 4. Missing pattern heatmap (before)
        missing_matrix_before = self.combined_df[had_missing].isnull().iloc[:200]  # Sample 200 rows
        sns.heatmap(missing_matrix_before.T, cbar=True, ax=axes[1,0], 
                   cmap='viridis_r', cbar_kws={"shrink": .8})
        axes[1,0].set_title('Missing Pattern Before (Sample)')
        axes[1,0].set_xlabel('Sample Index')
        
        # 5. Missing pattern heatmap (after)
        missing_matrix_after = self.processed_df[had_missing].isnull().iloc[:200]  # Sample 200 rows
        sns.heatmap(missing_matrix_after.T, cbar=True, ax=axes[1,1],
                   cmap='viridis_r', cbar_kws={"shrink": .8})
        axes[1,1].set_title('Missing Pattern After (Sample)')
        axes[1,1].set_xlabel('Sample Index')
        
        # 6. Summary statistics
        axes[1,2].axis('off')
        
        summary_text = f"""IMPUTATION SUMMARY:

Original Missing: {self.validation_results['original_missing_total']:,}
Remaining Missing: {self.validation_results['processed_missing_total']:,}
Reduction: {self.validation_results['missing_reduction_pct']:.1f}%

STRATEGIES APPLIED:
• Domain Knowledge: {sum(1 for log in all_logs.values() if 'Domain' in log['strategy'])} features
• Systematic: {sum(1 for log in all_logs.values() if 'Systematic' in log['strategy'])} features  
• KNN Imputation: {sum(1 for log in all_logs.values() if 'KNN' in log['strategy'])} features
• Mode Imputation: {sum(1 for log in all_logs.values() if 'Mode' in log['strategy'])} features

INDICATORS CREATED:
• {len(self.missing_indicators)} missing value indicators
• Preserve information about original missingness
        """
        
        axes[1,2].text(0.05, 0.95, summary_text, transform=axes[1,2].transAxes,
                      fontsize=10, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.1))
        
        plt.tight_layout()
        plt.savefig('../../visualizations/phase2_missing_value_treatment.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return self
    
    def save_imputation_artifacts(self):
        """Save imputation results and configuration"""
        print("\nSaving imputation artifacts...")
        
        # Save processed dataset
        # Split back into train and test
        train_size = len(self.train_df)
        
        processed_train = self.processed_df.iloc[:train_size].copy()
        processed_test = self.processed_df.iloc[train_size:].copy()
        
        # Add back SalePrice to training data
        processed_train['SalePrice'] = self.train_df['SalePrice'].values
        
        processed_train.to_csv('processed_train_phase2.csv', index=False)
        processed_test.to_csv('processed_test_phase2.csv', index=False)
        
        # Save imputation configuration
        imputation_config = {
            'strategies_applied': self.imputation_strategies,
            'domain_imputation_log': self.domain_imputation_log,
            'advanced_imputation_log': self.advanced_imputation_log,
            'validation_results': self.validation_results,
            'missing_indicators': self.missing_indicators.to_dict('records')
        }
        
        import json
        with open('imputation_config.json', 'w') as f:
            # Convert numpy types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            def recursive_convert(item):
                if isinstance(item, dict):
                    return {key: recursive_convert(value) for key, value in item.items()}
                elif isinstance(item, list):
                    return [recursive_convert(element) for element in item]
                else:
                    return convert_numpy_types(item)
            
            json.dump(recursive_convert(imputation_config), f, indent=4)
        
        print("SUCCESS Processed training data saved to processed_train_phase2.csv")
        print("SUCCESS Processed test data saved to processed_test_phase2.csv") 
        print("SUCCESS Imputation configuration saved to imputation_config.json")
        
        return self
    
    def run_complete_pipeline(self):
        """Execute the complete missing value treatment pipeline"""
        print("PHASE 2: MISSING VALUE TREATMENT PIPELINE")
        print("="*70)
        
        (self.load_data()
         .analyze_missing_patterns()
         .design_imputation_strategies()
         .apply_domain_knowledge_imputation()
         .apply_advanced_imputation()
         .create_missing_value_indicators()
         .validate_imputation_results()
         .create_imputation_visualization()
         .save_imputation_artifacts())
        
        print("\n" + "="*70)
        print("SUCCESS PHASE 2 COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"RESULT Missing value reduction: {self.validation_results['missing_reduction_pct']:.1f}%")
        print(f"RESULT Features with indicators: {len(self.missing_indicators)}")
        print("TARGET Ready for Phase 3: Feature Engineering")
        print("="*70)
        
        return self

if __name__ == "__main__":
    pipeline = MissingValueTreatmentPipeline()
    pipeline.run_complete_pipeline()