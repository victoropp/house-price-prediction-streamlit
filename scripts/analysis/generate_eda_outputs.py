import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

print("Generating EDA outputs to CSV files...")

# Load the datasets
train_df = pd.read_csv('dataset/train.csv')
test_df = pd.read_csv('dataset/test.csv')

# 1. Basic Dataset Information
dataset_info = pd.DataFrame({
    'Metric': ['Training_Samples', 'Test_Samples', 'Total_Features', 'Numeric_Features', 'Categorical_Features'],
    'Value': [
        train_df.shape[0],
        test_df.shape[0], 
        train_df.shape[1] - 1,  # Excluding target
        len(train_df.select_dtypes(include=[np.number]).columns),
        len(train_df.select_dtypes(include=['object']).columns)
    ]
})
dataset_info.to_csv('eda_dataset_info.csv', index=False)
print("Dataset info saved to: eda_dataset_info.csv")

# 2. Target Variable Statistics
target_stats = pd.DataFrame({
    'Statistic': ['Mean', 'Median', 'Min', 'Max', 'Std_Dev', 'Skewness', 'Kurtosis'],
    'Value': [
        train_df['SalePrice'].mean(),
        train_df['SalePrice'].median(),
        train_df['SalePrice'].min(),
        train_df['SalePrice'].max(),
        train_df['SalePrice'].std(),
        skew(train_df['SalePrice']),
        kurtosis(train_df['SalePrice'])
    ]
})
target_stats.to_csv('eda_target_statistics.csv', index=False)
print("- Target statistics saved to: eda_target_statistics.csv")

# 3. Feature Lists
numeric_features = train_df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()

feature_lists = pd.DataFrame({
    'Feature_Type': ['Numeric'] * len(numeric_features) + ['Categorical'] * len(categorical_features),
    'Feature_Name': numeric_features + categorical_features
})
feature_lists.to_csv('eda_feature_lists.csv', index=False)
print("- Feature lists saved to: eda_feature_lists.csv")

# 4. Missing Values Analysis
train_missing = train_df.isnull().sum()
test_missing = test_df.isnull().sum()
train_missing_pct = (train_missing / len(train_df)) * 100
test_missing_pct = (test_missing / len(test_df)) * 100

# Get common columns
common_columns = list(set(train_df.columns) & set(test_df.columns))
train_missing_common = train_missing[common_columns]
test_missing_common = test_missing[common_columns]
train_missing_pct_common = train_missing_pct[common_columns]
test_missing_pct_common = test_missing_pct[common_columns]

missing_analysis = pd.DataFrame({
    'Feature': common_columns,
    'Train_Missing_Count': train_missing_common.values,
    'Train_Missing_Percent': train_missing_pct_common.values,
    'Test_Missing_Count': test_missing_common.values,
    'Test_Missing_Percent': test_missing_pct_common.values
})

# Add SalePrice info
missing_analysis = pd.concat([
    missing_analysis,
    pd.DataFrame({
        'Feature': ['SalePrice'],
        'Train_Missing_Count': [0],
        'Train_Missing_Percent': [0.0],
        'Test_Missing_Count': ['N/A'],
        'Test_Missing_Percent': ['N/A']
    })
], ignore_index=True)

# Filter and sort by missing percentage
missing_analysis_filtered = missing_analysis[
    (missing_analysis['Train_Missing_Count'] > 0) | 
    (missing_analysis['Test_Missing_Count'] == 'N/A') |
    (pd.to_numeric(missing_analysis['Test_Missing_Count'], errors='coerce') > 0)
].sort_values('Train_Missing_Percent', ascending=False)

missing_analysis_filtered.to_csv('eda_missing_values.csv', index=False)
print("- Missing values analysis saved to: eda_missing_values.csv")

# 5. Correlation Analysis
correlations = train_df[numeric_features].corr()['SalePrice'].abs().sort_values(ascending=False)
correlation_analysis = pd.DataFrame({
    'Feature': correlations.index,
    'Correlation_with_SalePrice': correlations.values
})
correlation_analysis.to_csv('eda_correlations.csv', index=False)
print("- Correlation analysis saved to: eda_correlations.csv")

# 6. Numeric Features Statistics
numeric_stats = train_df[numeric_features].describe().T
numeric_stats.insert(0, 'Feature', numeric_stats.index)
numeric_stats.reset_index(drop=True, inplace=True)
numeric_stats.to_csv('eda_numeric_statistics.csv', index=False)
print("- Numeric statistics saved to: eda_numeric_statistics.csv")

# 7. Categorical Features Analysis
categorical_analysis = []
for feature in categorical_features:
    unique_count = train_df[feature].nunique()
    most_frequent = train_df[feature].mode().iloc[0] if not train_df[feature].isna().all() else "N/A"
    most_frequent_count = train_df[feature].value_counts().iloc[0] if not train_df[feature].isna().all() else 0
    most_frequent_pct = (most_frequent_count / len(train_df)) * 100 if most_frequent_count > 0 else 0
    
    categorical_analysis.append({
        'Feature': feature,
        'Unique_Values': unique_count,
        'Most_Frequent_Value': most_frequent,
        'Most_Frequent_Count': most_frequent_count,
        'Most_Frequent_Percent': most_frequent_pct
    })

categorical_df = pd.DataFrame(categorical_analysis)
categorical_df.to_csv('eda_categorical_analysis.csv', index=False)
print("- Categorical analysis saved to: eda_categorical_analysis.csv")

# 8. Skewness Analysis
skewness_analysis = []
for feature in numeric_features:
    if feature != 'Id':  # Skip ID column
        skew_val = skew(train_df[feature].dropna())
        skewness_analysis.append({
            'Feature': feature,
            'Skewness': skew_val,
            'Highly_Skewed': abs(skew_val) > 1
        })

skewness_df = pd.DataFrame(skewness_analysis).sort_values('Skewness', key=abs, ascending=False)
skewness_df.to_csv('eda_skewness_analysis.csv', index=False)
print("- Skewness analysis saved to: eda_skewness_analysis.csv")

# 9. Outlier Analysis
outlier_analysis = []
for feature in numeric_features:
    if feature not in ['Id', 'SalePrice'] and train_df[feature].dtype in ['int64', 'float64']:
        Q1 = train_df[feature].quantile(0.25)
        Q3 = train_df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = train_df[(train_df[feature] < lower_bound) | (train_df[feature] > upper_bound)][feature]
        outlier_count = len(outliers)
        outlier_percent = (outlier_count / len(train_df)) * 100
        
        outlier_analysis.append({
            'Feature': feature,
            'Outlier_Count': outlier_count,
            'Outlier_Percent': outlier_percent,
            'Lower_Bound': lower_bound,
            'Upper_Bound': upper_bound,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR
        })

outlier_df = pd.DataFrame(outlier_analysis).sort_values('Outlier_Percent', ascending=False)
outlier_df.to_csv('eda_outlier_analysis.csv', index=False)
print("- Outlier analysis saved to: eda_outlier_analysis.csv")

# 10. Feature Categories Analysis
year_features = [col for col in numeric_features if 'Year' in col or 'Yr' in col]
area_features = [col for col in numeric_features if 'SF' in col or 'Area' in col]
quality_features = [col for col in train_df.columns if 'Qual' in col or 'Cond' in col]

feature_categories = pd.DataFrame({
    'Category': ['Year_Features', 'Area_Features', 'Quality_Features'],
    'Count': [len(year_features), len(area_features), len(quality_features)],
    'Features': [', '.join(year_features), ', '.join(area_features), ', '.join(quality_features)]
})
feature_categories.to_csv('eda_feature_categories.csv', index=False)
print("- Feature categories saved to: eda_feature_categories.csv")

# 11. Neighborhood Analysis
if 'Neighborhood' in train_df.columns:
    neighborhood_stats = train_df.groupby('Neighborhood')['SalePrice'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(2)
    neighborhood_stats.insert(0, 'Neighborhood', neighborhood_stats.index)
    neighborhood_stats.reset_index(drop=True, inplace=True)
    neighborhood_stats = neighborhood_stats.sort_values('mean', ascending=False)
    neighborhood_stats.to_csv('eda_neighborhood_analysis.csv', index=False)
    print("- Neighborhood analysis saved to: eda_neighborhood_analysis.csv")

# 12. Year Features Analysis
year_analysis = []
for feature in year_features:
    year_analysis.append({
        'Feature': feature,
        'Min_Year': train_df[feature].min(),
        'Max_Year': train_df[feature].max(),
        'Range': train_df[feature].max() - train_df[feature].min(),
        'Mean_Year': train_df[feature].mean(),
        'Missing_Count': train_df[feature].isnull().sum()
    })

if year_analysis:
    year_df = pd.DataFrame(year_analysis)
    year_df.to_csv('eda_year_features.csv', index=False)
    print("- Year features analysis saved to: eda_year_features.csv")

# 13. Summary Report
summary_stats = {
    'Total_Features': train_df.shape[1] - 1,
    'Features_with_Missing_Values': len(missing_analysis_filtered) - 1,  # Exclude SalePrice
    'Highly_Skewed_Features': len(skewness_df[skewness_df['Highly_Skewed']]),
    'Features_with_Outliers': len(outlier_df[outlier_df['Outlier_Count'] > 0]),
    'Most_Correlated_Feature': correlation_analysis.iloc[1]['Feature'],  # Skip SalePrice itself
    'Highest_Correlation': correlation_analysis.iloc[1]['Correlation_with_SalePrice'],
    'Neighborhoods': train_df['Neighborhood'].nunique() if 'Neighborhood' in train_df.columns else 0,
    'Year_Range': f"{min([train_df[col].min() for col in year_features]):.0f}-{max([train_df[col].max() for col in year_features]):.0f}" if year_features else "N/A"
}

summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
summary_df.to_csv('eda_summary_report.csv', index=False)
print("- Summary report saved to: eda_summary_report.csv")

print(f"\n{'='*60}")
print("EDA CSV GENERATION COMPLETE!")
print(f"{'='*60}")
print(f"Generated {len([f for f in ['eda_dataset_info.csv', 'eda_target_statistics.csv', 'eda_feature_lists.csv', 'eda_missing_values.csv', 'eda_correlations.csv', 'eda_numeric_statistics.csv', 'eda_categorical_analysis.csv', 'eda_skewness_analysis.csv', 'eda_outlier_analysis.csv', 'eda_feature_categories.csv', 'eda_neighborhood_analysis.csv', 'eda_year_features.csv', 'eda_summary_report.csv']])} CSV files with comprehensive EDA results.")
print("\nFiles created:")
print("- eda_dataset_info.csv - Basic dataset information")
print("- eda_target_statistics.csv - SalePrice statistics")
print("- eda_feature_lists.csv - All feature names by type")
print("- eda_missing_values.csv - Missing value analysis")
print("- eda_correlations.csv - Feature correlations with target")
print("- eda_numeric_statistics.csv - Descriptive statistics for numeric features")
print("- eda_categorical_analysis.csv - Categorical feature analysis")
print("- eda_skewness_analysis.csv - Feature skewness analysis")
print("- eda_outlier_analysis.csv - Outlier detection results")
print("- eda_feature_categories.csv - Feature grouping by type")
print("- eda_neighborhood_analysis.csv - Neighborhood price analysis")
print("- eda_year_features.csv - Year-based feature analysis")
print("- eda_summary_report.csv - Overall EDA summary")