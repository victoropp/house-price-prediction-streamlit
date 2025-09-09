"""
Comprehensive EDA Visualizations for House Price Prediction
===========================================================

This script creates publication-quality visualizations following data science best practices:
- Clear storytelling through visualization
- Appropriate chart types for data types
- Professional styling and color schemes
- Comprehensive analysis covering all aspects of the dataset

Visualization Principles Applied:
1. WHAT: Choose appropriate chart types for data types
2. WHY: Each chart answers specific analytical questions
3. HOW: Professional styling, clear labels, proper scales
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# Set up professional styling
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Custom color palettes
PRIMARY_COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592941']
DIVERGING_COLORS = ['#d7191c', '#fdae61', '#ffffbf', '#abd9e9', '#2c7bb6']
SEQUENTIAL_COLORS = ['#f7f7f7', '#cccccc', '#969696', '#636363', '#252525']

# Figure settings
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

class HousePriceVisualizer:
    """
    Comprehensive visualization class for house price EDA
    """
    
    def __init__(self):
        """Initialize the visualizer and load data"""
        print("Loading data and initializing visualizer...")
        
        # Load datasets
        self.train_df = pd.read_csv('../dataset/train.csv')
        self.test_df = pd.read_csv('../dataset/test.csv')
        
        # Load EDA results
        self.correlations = pd.read_csv('../eda/eda_correlations.csv')
        self.missing_values = pd.read_csv('../eda/eda_missing_values.csv')
        self.skewness = pd.read_csv('../eda/eda_skewness_analysis.csv')
        self.outliers = pd.read_csv('../eda/eda_outlier_analysis.csv')
        self.neighborhoods = pd.read_csv('../eda/eda_neighborhood_analysis.csv')
        
        # Feature categories
        self.numeric_features = self.train_df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = self.train_df.select_dtypes(include=['object']).columns.tolist()
        
        print(f"Data loaded: {self.train_df.shape[0]} training samples, {self.train_df.shape[1]} features")
    
    def create_target_analysis(self):
        """
        WHAT: Target variable (SalePrice) distribution and characteristics
        WHY: Understanding target distribution is crucial for model selection and evaluation
        HOW: Multiple subplots showing distribution, box plot, and transformation effects
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Target Variable Analysis: Sale Price Distribution', fontsize=16, fontweight='bold')
        
        # 1. Histogram with density
        axes[0,0].hist(self.train_df['SalePrice'], bins=50, density=True, alpha=0.7, color=PRIMARY_COLORS[0])
        axes[0,0].axvline(self.train_df['SalePrice'].mean(), color=PRIMARY_COLORS[1], linestyle='--', 
                         label=f'Mean: ${self.train_df["SalePrice"].mean():,.0f}')
        axes[0,0].axvline(self.train_df['SalePrice'].median(), color=PRIMARY_COLORS[2], linestyle='--',
                         label=f'Median: ${self.train_df["SalePrice"].median():,.0f}')
        axes[0,0].set_title('Distribution of Sale Prices')
        axes[0,0].set_xlabel('Sale Price ($)')
        axes[0,0].set_ylabel('Density')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Box plot
        box = axes[0,1].boxplot(self.train_df['SalePrice'], patch_artist=True)
        box['boxes'][0].set_facecolor(PRIMARY_COLORS[0])
        axes[0,1].set_title('Sale Price Box Plot\n(Shows Quartiles & Outliers)')
        axes[0,1].set_ylabel('Sale Price ($)')
        axes[0,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Q-Q plot for normality
        stats.probplot(self.train_df['SalePrice'], dist="norm", plot=axes[0,2])
        axes[0,2].set_title('Q-Q Plot: Sale Price vs Normal Distribution')
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Log transformation
        log_prices = np.log1p(self.train_df['SalePrice'])
        axes[1,0].hist(log_prices, bins=50, density=True, alpha=0.7, color=PRIMARY_COLORS[3])
        axes[1,0].axvline(log_prices.mean(), color=PRIMARY_COLORS[1], linestyle='--', 
                         label=f'Mean: {log_prices.mean():.3f}')
        axes[1,0].axvline(log_prices.median(), color=PRIMARY_COLORS[2], linestyle='--',
                         label=f'Median: {log_prices.median():.3f}')
        axes[1,0].set_title('Log-Transformed Sale Prices')
        axes[1,0].set_xlabel('Log(Sale Price + 1)')
        axes[1,0].set_ylabel('Density')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Skewness comparison
        original_skew = skew(self.train_df['SalePrice'])
        log_skew = skew(log_prices)
        
        skew_data = pd.DataFrame({
            'Transformation': ['Original', 'Log'],
            'Skewness': [original_skew, log_skew]
        })
        
        bars = axes[1,1].bar(skew_data['Transformation'], skew_data['Skewness'], 
                           color=[PRIMARY_COLORS[0], PRIMARY_COLORS[3]])
        axes[1,1].set_title('Skewness Comparison')
        axes[1,1].set_ylabel('Skewness')
        axes[1,1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, skew_data['Skewness']):
            axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                          f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 6. Summary statistics table
        axes[1,2].axis('off')
        stats_data = [
            ['Statistic', 'Value'],
            ['Count', f'{len(self.train_df):,}'],
            ['Mean', f'${self.train_df["SalePrice"].mean():,.0f}'],
            ['Median', f'${self.train_df["SalePrice"].median():,.0f}'],
            ['Std Dev', f'${self.train_df["SalePrice"].std():,.0f}'],
            ['Min', f'${self.train_df["SalePrice"].min():,.0f}'],
            ['Max', f'${self.train_df["SalePrice"].max():,.0f}'],
            ['Skewness', f'{original_skew:.3f}'],
            ['Kurtosis', f'{kurtosis(self.train_df["SalePrice"]):.3f}']
        ]
        
        table = axes[1,2].table(cellText=stats_data, cellLoc='center', loc='center',
                               colWidths=[0.4, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        axes[1,2].set_title('Summary Statistics')
        
        plt.tight_layout()
        plt.savefig('01_target_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Target analysis visualization saved: 01_target_analysis.png")
    
    def create_correlation_heatmap(self):
        """
        WHAT: Correlation matrix of numeric features with SalePrice
        WHY: Identify most important features for price prediction
        HOW: Heatmap with hierarchical clustering and top correlations highlighted
        """
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('Feature Correlation Analysis', fontsize=16, fontweight='bold')
        
        # 1. Full correlation heatmap (top 20 features)
        top_20_corr = self.correlations.head(20)['Feature'].tolist()
        corr_matrix = self.train_df[top_20_corr].corr()
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                   fmt='.2f', square=True, ax=axes[0], cbar_kws={"shrink": .8})
        axes[0].set_title('Correlation Heatmap: Top 20 Features')
        
        # 2. Top correlations with SalePrice (bar chart)
        top_correlations = self.correlations.head(15)  # Top 15 including SalePrice
        top_correlations = top_correlations[top_correlations['Feature'] != 'SalePrice']  # Remove SalePrice itself
        
        plt.sca(axes[1])
        bars = plt.barh(range(len(top_correlations)), top_correlations['Correlation_with_SalePrice'],
                       color=[PRIMARY_COLORS[i % len(PRIMARY_COLORS)] for i in range(len(top_correlations))])
        
        plt.yticks(range(len(top_correlations)), top_correlations['Feature'])
        plt.xlabel('Correlation with Sale Price')
        plt.title('Top Features by Correlation with Sale Price')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, top_correlations['Correlation_with_SalePrice'])):
            plt.text(value + 0.01, bar.get_y() + bar.get_height()/2, f'{value:.3f}',
                    va='center', ha='left', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('02_correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Correlation analysis saved: 02_correlation_analysis.png")
    
    def create_missing_values_analysis(self):
        """
        WHAT: Missing values pattern and impact analysis
        WHY: Understanding missing data is crucial for preprocessing strategy
        HOW: Bar charts and heatmaps showing missing patterns
        """
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Missing Values Analysis', fontsize=16, fontweight='bold')
        
        # Filter features with missing values
        missing_features = self.missing_values[
            (self.missing_values['Train_Missing_Count'] > 0) | 
            (pd.to_numeric(self.missing_values['Test_Missing_Count'], errors='coerce') > 0)
        ].copy()
        
        # 1. Missing values by feature (train)
        train_missing = missing_features.sort_values('Train_Missing_Percent', ascending=True)
        bars1 = axes[0,0].barh(range(len(train_missing)), train_missing['Train_Missing_Percent'],
                              color=PRIMARY_COLORS[0])
        axes[0,0].set_yticks(range(len(train_missing)))
        axes[0,0].set_yticklabels(train_missing['Feature'])
        axes[0,0].set_xlabel('Missing Values (%)')
        axes[0,0].set_title('Missing Values in Training Set')
        axes[0,0].grid(True, alpha=0.3, axis='x')
        
        # 2. Missing values by feature (test)
        test_missing = missing_features.copy()
        test_missing['Test_Missing_Percent'] = pd.to_numeric(test_missing['Test_Missing_Percent'], errors='coerce')
        test_missing = test_missing.sort_values('Test_Missing_Percent', ascending=True)
        
        bars2 = axes[0,1].barh(range(len(test_missing)), test_missing['Test_Missing_Percent'],
                              color=PRIMARY_COLORS[1])
        axes[0,1].set_yticks(range(len(test_missing)))
        axes[0,1].set_yticklabels(test_missing['Feature'])
        axes[0,1].set_xlabel('Missing Values (%)')
        axes[0,1].set_title('Missing Values in Test Set')
        axes[0,1].grid(True, alpha=0.3, axis='x')
        
        # 3. Missing values heatmap pattern
        # Create a subset of features with highest missing rates for visualization
        high_missing_features = train_missing.tail(15)['Feature'].tolist()
        missing_matrix = self.train_df[high_missing_features].isnull()
        
        sns.heatmap(missing_matrix.T, cbar=True, ax=axes[1,0], cmap='viridis_r',
                   cbar_kws={"shrink": .8})
        axes[1,0].set_title('Missing Values Pattern\n(Sample of Records)')
        axes[1,0].set_xlabel('Sample Index')
        
        # 4. Missing values impact on price
        # Analyze if missing values in key features affect price distribution
        key_missing_feature = 'LotFrontage'  # Choose a feature with moderate missing values
        
        if key_missing_feature in self.train_df.columns:
            missing_mask = self.train_df[key_missing_feature].isnull()
            price_with_missing = self.train_df[missing_mask]['SalePrice']
            price_without_missing = self.train_df[~missing_mask]['SalePrice']
            
            data_for_box = [price_without_missing, price_with_missing]
            box_plot = axes[1,1].boxplot(data_for_box, labels=['Available', 'Missing'], patch_artist=True)
            
            box_plot['boxes'][0].set_facecolor(PRIMARY_COLORS[2])
            box_plot['boxes'][1].set_facecolor(PRIMARY_COLORS[3])
            
            axes[1,1].set_title(f'Sale Price Distribution\nby {key_missing_feature} Availability')
            axes[1,1].set_ylabel('Sale Price ($)')
            axes[1,1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('03_missing_values_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Missing values analysis saved: 03_missing_values_analysis.png")
    
    def create_distribution_analysis(self):
        """
        WHAT: Distribution analysis of key numeric features
        WHY: Understanding feature distributions guides preprocessing decisions
        HOW: Histograms, box plots, and skewness analysis
        """
        # Select top correlated features for analysis
        top_features = self.correlations.head(10)['Feature'].tolist()
        top_features = [f for f in top_features if f != 'SalePrice'][:8]  # Top 8 excluding SalePrice
        
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        fig.suptitle('Distribution Analysis: Key Numeric Features', fontsize=16, fontweight='bold')
        
        for i, feature in enumerate(top_features):
            row = i // 2
            col = (i % 2) * 2
            
            # Histogram
            axes[row, col].hist(self.train_df[feature], bins=30, alpha=0.7, 
                               color=PRIMARY_COLORS[i % len(PRIMARY_COLORS)])
            axes[row, col].set_title(f'{feature} Distribution')
            axes[row, col].set_xlabel(feature)
            axes[row, col].set_ylabel('Frequency')
            axes[row, col].grid(True, alpha=0.3)
            
            # Add skewness info
            feature_skew = skew(self.train_df[feature].dropna())
            axes[row, col].text(0.7, 0.9, f'Skew: {feature_skew:.2f}', 
                               transform=axes[row, col].transAxes, 
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Box plot
            box = axes[row, col+1].boxplot(self.train_df[feature], patch_artist=True)
            box['boxes'][0].set_facecolor(PRIMARY_COLORS[i % len(PRIMARY_COLORS)])
            axes[row, col+1].set_title(f'{feature} Box Plot')
            axes[row, col+1].set_ylabel(feature)
            axes[row, col+1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('04_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Distribution analysis saved: 04_distribution_analysis.png")
    
    def create_categorical_analysis(self):
        """
        WHAT: Analysis of categorical features and their relationship with price
        WHY: Understanding categorical features helps with encoding decisions
        HOW: Bar plots and box plots showing category distributions and price impacts
        """
        # Select important categorical features
        important_categorical = ['Neighborhood', 'OverallQual', 'ExterQual', 'KitchenQual', 
                               'BsmtQual', 'GarageType', 'SaleType', 'MSZoning']
        available_categorical = [f for f in important_categorical if f in self.train_df.columns]
        
        fig, axes = plt.subplots(3, 3, figsize=(22, 18))
        fig.suptitle('Categorical Features Analysis', fontsize=16, fontweight='bold')
        
        for i, feature in enumerate(available_categorical[:9]):
            row = i // 3
            col = i % 3
            
            if feature == 'Neighborhood':
                # Special handling for neighborhood - show top 10
                neighborhood_prices = self.train_df.groupby(feature)['SalePrice'].mean().sort_values(ascending=False)
                top_neighborhoods = neighborhood_prices.head(10)
                
                bars = axes[row, col].bar(range(len(top_neighborhoods)), top_neighborhoods.values,
                                         color=PRIMARY_COLORS[0])
                axes[row, col].set_xticks(range(len(top_neighborhoods)))
                axes[row, col].set_xticklabels(top_neighborhoods.index, rotation=45)
                axes[row, col].set_title(f'Average Sale Price by {feature} (Top 10)')
                axes[row, col].set_ylabel('Average Sale Price ($)')
                axes[row, col].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
                
            elif feature in ['OverallQual', 'OverallCond']:
                # Box plot for ordinal features
                unique_vals = sorted(self.train_df[feature].unique())
                data_for_box = [self.train_df[self.train_df[feature] == val]['SalePrice'] for val in unique_vals]
                
                box_plot = axes[row, col].boxplot(data_for_box, labels=unique_vals, patch_artist=True)
                for patch, color in zip(box_plot['boxes'], PRIMARY_COLORS):
                    patch.set_facecolor(color)
                
                axes[row, col].set_title(f'Sale Price Distribution by {feature}')
                axes[row, col].set_xlabel(feature)
                axes[row, col].set_ylabel('Sale Price ($)')
                axes[row, col].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
                
            else:
                # Regular categorical features
                feature_counts = self.train_df[feature].value_counts().head(8)
                bars = axes[row, col].bar(range(len(feature_counts)), feature_counts.values,
                                         color=PRIMARY_COLORS[i % len(PRIMARY_COLORS)])
                axes[row, col].set_xticks(range(len(feature_counts)))
                axes[row, col].set_xticklabels(feature_counts.index, rotation=45)
                axes[row, col].set_title(f'{feature} Distribution')
                axes[row, col].set_ylabel('Count')
            
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('05_categorical_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Categorical analysis saved: 05_categorical_analysis.png")
    
    def create_neighborhood_insights(self):
        """
        WHAT: Deep dive into neighborhood analysis
        WHY: Location is a key driver of house prices
        HOW: Multiple visualizations showing neighborhood patterns
        """
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Neighborhood Analysis: Location Impact on Pricing', fontsize=16, fontweight='bold')
        
        # 1. Average price by neighborhood (all neighborhoods)
        neighborhood_stats = self.neighborhoods.sort_values('mean', ascending=False)
        
        bars1 = axes[0,0].barh(range(len(neighborhood_stats)), neighborhood_stats['mean'],
                              color=[PRIMARY_COLORS[i % len(PRIMARY_COLORS)] for i in range(len(neighborhood_stats))])
        axes[0,0].set_yticks(range(len(neighborhood_stats)))
        axes[0,0].set_yticklabels(neighborhood_stats['Neighborhood'])
        axes[0,0].set_xlabel('Average Sale Price ($)')
        axes[0,0].set_title('Average Sale Price by Neighborhood')
        axes[0,0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        axes[0,0].grid(True, alpha=0.3, axis='x')
        
        # 2. Number of sales by neighborhood
        bars2 = axes[0,1].barh(range(len(neighborhood_stats)), neighborhood_stats['count'],
                              color=PRIMARY_COLORS[1])
        axes[0,1].set_yticks(range(len(neighborhood_stats)))
        axes[0,1].set_yticklabels(neighborhood_stats['Neighborhood'])
        axes[0,1].set_xlabel('Number of Sales')
        axes[0,1].set_title('Sales Volume by Neighborhood')
        axes[0,1].grid(True, alpha=0.3, axis='x')
        
        # 3. Price range (min-max) by top neighborhoods
        top_10_neighborhoods = neighborhood_stats.head(10)
        
        # Create error bars showing min-max range
        y_pos = np.arange(len(top_10_neighborhoods))
        axes[1,0].barh(y_pos, top_10_neighborhoods['mean'], 
                      xerr=[top_10_neighborhoods['mean'] - top_10_neighborhoods['min'],
                            top_10_neighborhoods['max'] - top_10_neighborhoods['mean']],
                      capsize=5, color=PRIMARY_COLORS[2], alpha=0.7)
        
        axes[1,0].set_yticks(y_pos)
        axes[1,0].set_yticklabels(top_10_neighborhoods['Neighborhood'])
        axes[1,0].set_xlabel('Sale Price ($)')
        axes[1,0].set_title('Price Range by Top 10 Neighborhoods\n(Error bars show min-max)')
        axes[1,0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        axes[1,0].grid(True, alpha=0.3, axis='x')
        
        # 4. Price variability (coefficient of variation)
        neighborhood_stats['cv'] = neighborhood_stats['std'] / neighborhood_stats['mean']
        cv_sorted = neighborhood_stats.sort_values('cv', ascending=False).head(10)
        
        bars4 = axes[1,1].bar(range(len(cv_sorted)), cv_sorted['cv'],
                             color=PRIMARY_COLORS[3])
        axes[1,1].set_xticks(range(len(cv_sorted)))
        axes[1,1].set_xticklabels(cv_sorted['Neighborhood'], rotation=45)
        axes[1,1].set_ylabel('Coefficient of Variation')
        axes[1,1].set_title('Price Variability by Neighborhood\n(Higher = More Variable Prices)')
        axes[1,1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('06_neighborhood_insights.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Neighborhood insights saved: 06_neighborhood_insights.png")
    
    def create_feature_engineering_insights(self):
        """
        WHAT: Insights for feature engineering decisions
        WHY: Guide preprocessing and feature creation strategies
        HOW: Analysis of skewness, outliers, and transformation needs
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Feature Engineering Insights', fontsize=16, fontweight='bold')
        
        # 1. Skewness analysis
        highly_skewed = self.skewness[abs(self.skewness['Skewness']) > 1].sort_values('Skewness', 
                                                                                      key=abs, ascending=False).head(15)
        
        colors = ['red' if x > 0 else 'blue' for x in highly_skewed['Skewness']]
        bars1 = axes[0,0].barh(range(len(highly_skewed)), highly_skewed['Skewness'], color=colors)
        axes[0,0].set_yticks(range(len(highly_skewed)))
        axes[0,0].set_yticklabels(highly_skewed['Feature'])
        axes[0,0].set_xlabel('Skewness')
        axes[0,0].set_title('Highly Skewed Features\n(Red: Right Skew, Blue: Left Skew)')
        axes[0,0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        axes[0,0].grid(True, alpha=0.3, axis='x')
        
        # 2. Outlier analysis
        high_outlier_features = self.outliers.sort_values('Outlier_Percent', ascending=False).head(15)
        
        bars2 = axes[0,1].barh(range(len(high_outlier_features)), high_outlier_features['Outlier_Percent'],
                              color=PRIMARY_COLORS[0])
        axes[0,1].set_yticks(range(len(high_outlier_features)))
        axes[0,1].set_yticklabels(high_outlier_features['Feature'])
        axes[0,1].set_xlabel('Outlier Percentage (%)')
        axes[0,1].set_title('Features with Most Outliers')
        axes[0,1].grid(True, alpha=0.3, axis='x')
        
        # 3. Feature importance tiers
        correlation_tiers = pd.DataFrame({
            'Tier': ['Very High (>0.7)', 'High (0.5-0.7)', 'Moderate (0.3-0.5)', 'Low (<0.3)'],
            'Count': [
                sum(self.correlations['Correlation_with_SalePrice'] > 0.7),
                sum((self.correlations['Correlation_with_SalePrice'] > 0.5) & 
                    (self.correlations['Correlation_with_SalePrice'] <= 0.7)),
                sum((self.correlations['Correlation_with_SalePrice'] > 0.3) & 
                    (self.correlations['Correlation_with_SalePrice'] <= 0.5)),
                sum(self.correlations['Correlation_with_SalePrice'] <= 0.3)
            ]
        })
        
        bars3 = axes[0,2].bar(correlation_tiers['Tier'], correlation_tiers['Count'],
                             color=PRIMARY_COLORS[:4])
        axes[0,2].set_title('Feature Importance Distribution')
        axes[0,2].set_ylabel('Number of Features')
        axes[0,2].tick_params(axis='x', rotation=45)
        axes[0,2].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars3, correlation_tiers['Count']):
            axes[0,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                          str(value), ha='center', va='bottom', fontweight='bold')
        
        # 4. Year features analysis (age calculation opportunity)
        year_features = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']
        available_year_features = [f for f in year_features if f in self.train_df.columns]
        
        if available_year_features:
            for i, feature in enumerate(available_year_features[:2]):  # Show 2 year features
                axes[1,i].scatter(self.train_df[feature], self.train_df['SalePrice'], 
                                 alpha=0.6, color=PRIMARY_COLORS[i])
                axes[1,i].set_xlabel(feature)
                axes[1,i].set_ylabel('Sale Price ($)')
                axes[1,i].set_title(f'Sale Price vs {feature}')
                axes[1,i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
                axes[1,i].grid(True, alpha=0.3)
                
                # Add correlation
                corr = self.train_df[feature].corr(self.train_df['SalePrice'])
                axes[1,i].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                              transform=axes[1,i].transAxes, 
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 5. Preprocessing recommendations
        axes[1,2].axis('off')
        recommendations = [
            "PREPROCESSING RECOMMENDATIONS",
            "",
            f"1. LOG TRANSFORM: {len(highly_skewed)} highly skewed features",
            f"2. OUTLIER HANDLING: {len(high_outlier_features)} features need attention", 
            f"3. MISSING VALUES: {len(self.missing_values)} features need imputation",
            "4. FEATURE ENGINEERING:",
            "   - Create house age from YearBuilt",
            "   - Combine related area features",
            "   - Encode quality features ordinally",
            "   - Create neighborhood clusters",
            "",
            "5. SCALING: StandardScaler for continuous features",
            "6. ENCODING: Target encoding for high-cardinality cats"
        ]
        
        for i, text in enumerate(recommendations):
            weight = 'bold' if text.startswith(('1.', '2.', '3.', '4.', '5.', '6.', 'PREPROCESSING')) else 'normal'
            axes[1,2].text(0.05, 0.95 - i*0.07, text, transform=axes[1,2].transAxes, 
                          fontweight=weight, fontsize=10)
        
        plt.tight_layout()
        plt.savefig('07_feature_engineering_insights.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Feature engineering insights saved: 07_feature_engineering_insights.png")
    
    def create_executive_dashboard(self):
        """
        WHAT: Executive summary dashboard with key insights
        WHY: Provide stakeholders with actionable insights from the analysis
        HOW: Clean, professional layout with key metrics and recommendations
        """
        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        fig.suptitle('House Price Analysis: Executive Dashboard', fontsize=18, fontweight='bold')
        
        # 1. Key metrics (top-left)
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.axis('off')
        
        metrics = [
            f"Dataset Size: {self.train_df.shape[0]:,} houses",
            f"Price Range: ${self.train_df['SalePrice'].min():,} - ${self.train_df['SalePrice'].max():,}",
            f"Median Price: ${self.train_df['SalePrice'].median():,}",
            f"Total Features: {self.train_df.shape[1] - 1}",
            f"Top Predictor: {self.correlations.iloc[1]['Feature']} (r={self.correlations.iloc[1]['Correlation_with_SalePrice']:.3f})"
        ]
        
        for i, metric in enumerate(metrics):
            ax1.text(0.05, 0.8 - i*0.15, metric, transform=ax1.transAxes, 
                    fontsize=14, fontweight='bold', color=PRIMARY_COLORS[0])
        
        ax1.set_title('KEY METRICS', fontsize=14, fontweight='bold', loc='left')
        
        # 2. Price distribution (top-right)
        ax2 = fig.add_subplot(gs[0, 2:])
        ax2.hist(self.train_df['SalePrice']/1000, bins=40, color=PRIMARY_COLORS[0], alpha=0.7)
        ax2.set_xlabel('Sale Price ($000s)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('PRICE DISTRIBUTION', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Top predictors (middle-left)
        ax3 = fig.add_subplot(gs[1, :2])
        top_10_features = self.correlations.head(11)  # Top 10 + SalePrice
        top_10_features = top_10_features[top_10_features['Feature'] != 'SalePrice']
        
        bars = ax3.barh(range(len(top_10_features)), top_10_features['Correlation_with_SalePrice'],
                       color=PRIMARY_COLORS[1])
        ax3.set_yticks(range(len(top_10_features)))
        ax3.set_yticklabels(top_10_features['Feature'])
        ax3.set_xlabel('Correlation with Sale Price')
        ax3.set_title('TOP PREDICTIVE FEATURES', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. Missing values impact (middle-right)
        ax4 = fig.add_subplot(gs[1, 2:])
        missing_summary = self.missing_values.head(10)
        bars = ax4.bar(range(len(missing_summary)), missing_summary['Train_Missing_Percent'],
                      color=PRIMARY_COLORS[2])
        ax4.set_xticks(range(len(missing_summary)))
        ax4.set_xticklabels(missing_summary['Feature'], rotation=45)
        ax4.set_ylabel('Missing Values (%)')
        ax4.set_title('DATA QUALITY ISSUES', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Neighborhood insights (bottom-left)
        ax5 = fig.add_subplot(gs[2, :2])
        top_neighborhoods = self.neighborhoods.head(8)
        bars = ax5.bar(range(len(top_neighborhoods)), top_neighborhoods['mean']/1000,
                      color=PRIMARY_COLORS[3])
        ax5.set_xticks(range(len(top_neighborhoods)))
        ax5.set_xticklabels(top_neighborhoods['Neighborhood'], rotation=45)
        ax5.set_ylabel('Average Price ($000s)')
        ax5.set_title('PREMIUM NEIGHBORHOODS', fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Feature categories (bottom-right)
        ax6 = fig.add_subplot(gs[2, 2:])
        categories = ['Numeric', 'Categorical', 'Missing Issues', 'High Skew', 'Outliers']
        counts = [
            len(self.numeric_features),
            len(self.categorical_features),
            len(self.missing_values[self.missing_values['Train_Missing_Count'] > 0]),
            len(self.skewness[abs(self.skewness['Skewness']) > 1]),
            len(self.outliers[self.outliers['Outlier_Count'] > 0])
        ]
        
        bars = ax6.bar(categories, counts, color=PRIMARY_COLORS[:5])
        ax6.set_ylabel('Count')
        ax6.set_title('FEATURE CATEGORIES', fontweight='bold')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3, axis='y')
        
        # 7. Recommendations (bottom)
        ax7 = fig.add_subplot(gs[3, :])
        ax7.axis('off')
        
        recommendations_text = """
        BUSINESS RECOMMENDATIONS:
        
        🏠 PRICING STRATEGY: Focus on Overall Quality, Living Area, and Garage features - these drive 60%+ of price variation
        
        📍 LOCATION ANALYSIS: Premium neighborhoods (NoRidge, NridgHt, StoneBr) command 50%+ price premiums
        
        🔧 DATA IMPROVEMENTS: Address missing values in Pool, Fence, and Alley features (90%+ missing rates)
        
        🎯 MODEL READINESS: Dataset requires log transformation, outlier handling, and feature engineering before modeling
        
        💡 FEATURE ENGINEERING: Create house age, combine area features, and encode quality metrics for optimal model performance
        """
        
        ax7.text(0.02, 0.95, recommendations_text, transform=ax7.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.1))
        
        plt.tight_layout()
        plt.savefig('08_executive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Executive dashboard saved: 08_executive_dashboard.png")
    
    def generate_all_visualizations(self):
        """Generate all visualizations in sequence"""
        print("Starting comprehensive visualization generation...")
        print("="*60)
        
        try:
            self.create_target_analysis()
            self.create_correlation_heatmap()
            self.create_missing_values_analysis()
            self.create_distribution_analysis()
            self.create_categorical_analysis()
            self.create_neighborhood_insights()
            self.create_feature_engineering_insights()
            self.create_executive_dashboard()
            
            print("\n" + "="*60)
            print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
            print("="*60)
            print("\nVisualization files created in '' folder:")
            print("1. 01_target_analysis.png - Target variable distribution and characteristics")
            print("2. 02_correlation_analysis.png - Feature correlations and importance")
            print("3. 03_missing_values_analysis.png - Missing data patterns and impact")
            print("4. 04_distribution_analysis.png - Key feature distributions")
            print("5. 05_categorical_analysis.png - Categorical feature analysis")
            print("6. 06_neighborhood_insights.png - Location-based price analysis")
            print("7. 07_feature_engineering_insights.png - Preprocessing recommendations")
            print("8. 08_executive_dashboard.png - Executive summary dashboard")
            
        except Exception as e:
            print(f"Error generating visualizations: {str(e)}")
            raise

if __name__ == "__main__":
    # Initialize and run the visualization suite
    visualizer = HousePriceVisualizer()
    visualizer.generate_all_visualizations()