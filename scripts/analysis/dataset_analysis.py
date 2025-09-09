import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("="*80)
print("COMPREHENSIVE HOUSE PRICE DATASET ANALYSIS")
print("="*80)

# Load the datasets
train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')

print(f"\nDATASET OVERVIEW")
print("-" * 50)
print(f"Training set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")
print(f"Total features: {train_df.shape[1] - 1}")  # Excluding target variable
print(f"Target variable: SalePrice")

# Basic info about the datasets
print(f"\nTARGET VARIABLE ANALYSIS (SalePrice)")
print("-" * 50)
print(f"Mean Sale Price: ${train_df['SalePrice'].mean():,.2f}")
print(f"Median Sale Price: ${train_df['SalePrice'].median():,.2f}")
print(f"Min Sale Price: ${train_df['SalePrice'].min():,.2f}")
print(f"Max Sale Price: ${train_df['SalePrice'].max():,.2f}")
print(f"Standard Deviation: ${train_df['SalePrice'].std():,.2f}")
print(f"Skewness: {skew(train_df['SalePrice']):.3f}")
print(f"Kurtosis: {kurtosis(train_df['SalePrice']):.3f}")

# Data types analysis
print(f"\nDATA TYPES ANALYSIS")
print("-" * 50)
numeric_features = train_df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()

print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")

print(f"\nNumeric Features:")
for i, feature in enumerate(numeric_features, 1):
    print(f"{i:2d}. {feature}")

print(f"\nCategorical Features:")
for i, feature in enumerate(categorical_features, 1):
    print(f"{i:2d}. {feature}")

# Missing values analysis
print(f"\nMISSING VALUES ANALYSIS")
print("-" * 50)
train_missing = train_df.isnull().sum()
test_missing = test_df.isnull().sum()

train_missing_pct = (train_missing / len(train_df)) * 100
test_missing_pct = (test_missing / len(test_df)) * 100

# Get common columns between train and test
common_columns = list(set(train_df.columns) & set(test_df.columns))
train_missing_common = train_missing[common_columns]
test_missing_common = test_missing[common_columns]
train_missing_pct_common = train_missing_pct[common_columns]
test_missing_pct_common = test_missing_pct[common_columns]

missing_data = pd.DataFrame({
    'Feature': common_columns,
    'Train_Missing': train_missing_common.values,
    'Train_Missing_%': train_missing_pct_common.values,
    'Test_Missing': test_missing_common.values,
    'Test_Missing_%': test_missing_pct_common.values
})

# Also show train-only columns (like SalePrice)
train_only_cols = set(train_df.columns) - set(test_df.columns)
if train_only_cols:
    print(f"Train-only columns: {list(train_only_cols)}")
    for col in train_only_cols:
        train_missing_val = train_missing[col]
        train_missing_pct_val = train_missing_pct[col]
        print(f"  {col}: {train_missing_val} missing ({train_missing_pct_val:.1f}%)")

# Filter features with missing values
missing_data = missing_data[(missing_data['Train_Missing'] > 0) | (missing_data['Test_Missing'] > 0)]
missing_data = missing_data.sort_values('Train_Missing_%', ascending=False)

print(f"Features with missing values: {len(missing_data)}")
print("\nTop missing value features:")
print(missing_data.head(15).to_string(index=False))

# Analyze categorical features
print(f"\nCATEGORICAL FEATURES ANALYSIS")
print("-" * 50)
for feature in categorical_features[:10]:  # Show first 10 categorical features
    unique_count = train_df[feature].nunique()
    most_frequent = train_df[feature].mode().iloc[0] if not train_df[feature].isna().all() else "N/A"
    print(f"{feature}: {unique_count} unique values, Most frequent: {most_frequent}")

# Analyze numeric features
print(f"\nNUMERIC FEATURES ANALYSIS")
print("-" * 50)
numeric_stats = train_df[numeric_features].describe().T
print("Basic statistics for numeric features:")
print(numeric_stats.round(2))

# Correlation analysis with target variable
print(f"\nCORRELATION WITH TARGET VARIABLE")
print("-" * 50)
correlations = train_df[numeric_features].corr()['SalePrice'].abs().sort_values(ascending=False)
print("Top 20 features most correlated with SalePrice:")
for i, (feature, corr) in enumerate(correlations.head(20).items(), 1):
    print(f"{i:2d}. {feature:<20}: {corr:.3f}")

# Feature distributions analysis
print(f"\nFEATURE DISTRIBUTIONS")
print("-" * 50)
highly_skewed = []
for feature in numeric_features:
    if feature != 'SalePrice':
        skew_val = skew(train_df[feature].dropna())
        if abs(skew_val) > 1:
            highly_skewed.append((feature, skew_val))

highly_skewed = sorted(highly_skewed, key=lambda x: abs(x[1]), reverse=True)
print(f"Highly skewed features (|skew| > 1): {len(highly_skewed)}")
for feature, skew_val in highly_skewed[:10]:
    print(f"  {feature}: {skew_val:.3f}")

# Outlier analysis
print(f"\nOUTLIER ANALYSIS")
print("-" * 50)
outlier_features = []
for feature in numeric_features:
    if feature != 'SalePrice' and train_df[feature].dtype in ['int64', 'float64']:
        Q1 = train_df[feature].quantile(0.25)
        Q3 = train_df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = train_df[(train_df[feature] < lower_bound) | (train_df[feature] > upper_bound)][feature]
        outlier_count = len(outliers)
        
        if outlier_count > 0:
            outlier_features.append((feature, outlier_count, (outlier_count/len(train_df))*100))

outlier_features = sorted(outlier_features, key=lambda x: x[2], reverse=True)
print(f"Features with outliers: {len(outlier_features)}")
for feature, count, pct in outlier_features[:10]:
    print(f"  {feature}: {count} outliers ({pct:.1f}%)")

# Year-based features analysis
print(f"\nYEAR-BASED FEATURES ANALYSIS")
print("-" * 50)
year_features = [col for col in numeric_features if 'Year' in col or 'Yr' in col]
print(f"Year-based features: {year_features}")
for feature in year_features:
    print(f"{feature}: Range {train_df[feature].min():.0f} - {train_df[feature].max():.0f}")

# Quality and condition features
print(f"\nQUALITY AND CONDITION FEATURES")
print("-" * 50)
quality_features = [col for col in train_df.columns if 'Qual' in col or 'Cond' in col]
print(f"Quality/Condition features: {quality_features}")
for feature in quality_features:
    if train_df[feature].dtype == 'object':
        print(f"{feature}: {train_df[feature].value_counts().to_dict()}")

# Area-based features
print(f"\nAREA-BASED FEATURES")
print("-" * 50)
area_features = [col for col in numeric_features if 'SF' in col or 'Area' in col]
print(f"Area-based features: {area_features}")
for feature in area_features:
    mean_val = train_df[feature].mean()
    max_val = train_df[feature].max()
    print(f"{feature}: Mean={mean_val:.0f}, Max={max_val:.0f}")

# Neighborhood analysis
print(f"\nNEIGHBORHOOD ANALYSIS")
print("-" * 50)
if 'Neighborhood' in train_df.columns:
    neighborhood_stats = train_df.groupby('Neighborhood')['SalePrice'].agg(['count', 'mean', 'median']).round(0)
    neighborhood_stats = neighborhood_stats.sort_values('mean', ascending=False)
    print("Top 10 neighborhoods by average sale price:")
    print(neighborhood_stats.head(10))

# Feature importance insights
print(f"\nKEY INSIGHTS FOR MODEL BUILDING")
print("-" * 50)
print("1. MOST IMPORTANT NUMERIC FEATURES (by correlation):")
top_corr_features = correlations.head(10).index.tolist()
for i, feature in enumerate(top_corr_features[1:], 1):  # Skip SalePrice itself
    print(f"   {i}. {feature}")

print("\n2. FEATURES REQUIRING PREPROCESSING:")
print(f"   - Missing values: {len(missing_data)} features need imputation")
print(f"   - Highly skewed: {len(highly_skewed)} features may need transformation")
print(f"   - Outliers: {len(outlier_features)} features have significant outliers")

print("\n3. FEATURE CATEGORIES:")
print(f"   - Year features: {len(year_features)} (may need age calculation)")
print(f"   - Area features: {len(area_features)} (physical dimensions)")
print(f"   - Quality features: {len(quality_features)} (ordinal encoding needed)")

print("\n4. DATA QUALITY:")
total_missing = train_df.isnull().sum().sum()
print(f"   - Total missing values: {total_missing}")
print(f"   - Clean features: {len(train_df.columns) - len(missing_data)} features with no missing values")

print("\n" + "="*80)
print("ANALYSIS COMPLETE - DATASET FULLY UNDERSTOOD")
print("="*80)