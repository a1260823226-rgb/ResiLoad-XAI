# -*- coding: utf-8 -*-
"""
多模态特征相关性分析脚本 v2
对居民负荷数据进行五种相关性分析方法
"""
import pandas as pd
import numpy as np
import warnings
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
import os

warnings.filterwarnings('ignore')

def load_and_prepare_data(data_path='data/df_merged.csv', sample_size=100000):
    """加载和准备数据"""
    print("=" * 70)
    print("Step 1: Load Data")
    print("=" * 70)
    
    print("Loading data...")
    df = pd.read_csv(data_path, parse_dates=['DATETIME'])
    print("Loaded: {} rows, {} columns".format(len(df), len(df.columns)))
    
    # 选择数值特征
    exclude_cols = ['TRANSFORMER_ID', 'DATETIME', 'HOLIDAY', 'STATION_ID', 'LOAD']
    numeric_features = [col for col in df.columns 
                       if col not in exclude_cols and df[col].dtype in ['float64', 'int64']]
    
    print("Numeric features: {}".format(len(numeric_features)))
    print("Target variable: LOAD")
    print()
    
    # 准备数据
    print("=" * 70)
    print("Step 2: Prepare Data")
    print("=" * 70)
    
    df_clean = df[numeric_features + ['LOAD']].dropna()
    print("Original data: {} rows".format(len(df)))
    print("Cleaned data: {} rows".format(len(df_clean)))
    print("Removed rows: {}".format(len(df) - len(df_clean)))
    
    # 采样
    if len(df_clean) > sample_size:
        print("Sampling {} rows for faster computation...".format(sample_size))
        df_clean = df_clean.sample(n=sample_size, random_state=42)
        print("Sampled data: {} rows".format(len(df_clean)))
    
    X = df_clean[numeric_features].values
    y = df_clean['LOAD'].values
    
    print("Feature matrix shape: {}".format(X.shape))
    print("Target vector shape: {}".format(y.shape))
    print()
    
    return X, y, numeric_features

def pearson_correlation(X, y, features):
    """Pearson相关系数"""
    print("=" * 70)
    print("Method 1: Pearson Correlation Coefficient")
    print("=" * 70)
    print("Description: Measures linear correlation between variables")
    print("Application: Continuous data, feature selection before fusion")
    print()
    
    results = []
    for i, feature in enumerate(features):
        try:
            corr, p_value = pearsonr(X[:, i], y)
            results.append({
                'Feature': feature,
                'Pearson_Correlation': corr,
                'P_Value': p_value,
                'Significant': 'Yes' if p_value < 0.05 else 'No'
            })
        except Exception as e:
            results.append({
                'Feature': feature,
                'Pearson_Correlation': np.nan,
                'P_Value': np.nan,
                'Significant': 'Error'
            })
    
    df_result = pd.DataFrame(results)
    df_result = df_result.sort_values('Pearson_Correlation', key=abs, ascending=False)
    
    print("Top 10 correlated features:")
    print(df_result.head(10).to_string(index=False))
    print()
    
    return df_result

def spearman_correlation(X, y, features):
    """Spearman秩相关系数"""
    print("=" * 70)
    print("Method 2: Spearman Rank Correlation")
    print("=" * 70)
    print("Description: Measures monotonic correlation, distribution-free")
    print("Application: Non-normal multimodal features")
    print()
    
    results = []
    for i, feature in enumerate(features):
        try:
            corr, p_value = spearmanr(X[:, i], y)
            results.append({
                'Feature': feature,
                'Spearman_Correlation': corr,
                'P_Value': p_value,
                'Significant': 'Yes' if p_value < 0.05 else 'No'
            })
        except Exception as e:
            results.append({
                'Feature': feature,
                'Spearman_Correlation': np.nan,
                'P_Value': np.nan,
                'Significant': 'Error'
            })
    
    df_result = pd.DataFrame(results)
    df_result = df_result.sort_values('Spearman_Correlation', key=abs, ascending=False)
    
    print("Top 10 correlated features:")
    print(df_result.head(10).to_string(index=False))
    print()
    
    return df_result

def kendall_correlation(X, y, features):
    """Kendall tau相关系数"""
    print("=" * 70)
    print("Method 3: Kendall Tau Correlation")
    print("=" * 70)
    print("Description: Rank-based concordance measure")
    print("Application: Time series ordinal correlation")
    print()
    
    results = []
    for i, feature in enumerate(features):
        try:
            corr, p_value = kendalltau(X[:, i], y)
            results.append({
                'Feature': feature,
                'Kendall_Tau': corr,
                'P_Value': p_value,
                'Significant': 'Yes' if p_value < 0.05 else 'No'
            })
        except Exception as e:
            results.append({
                'Feature': feature,
                'Kendall_Tau': np.nan,
                'P_Value': np.nan,
                'Significant': 'Error'
            })
    
    df_result = pd.DataFrame(results)
    df_result = df_result.sort_values('Kendall_Tau', key=abs, ascending=False)
    
    print("Top 10 correlated features:")
    print(df_result.head(10).to_string(index=False))
    print()
    
    return df_result

def mutual_information(X, y, features):
    """互信息"""
    print("=" * 70)
    print("Method 5: Mutual Information")
    print("=" * 70)
    print("Description: Quantifies dependency (including non-linear)")
    print("Application: Multimodal feature selection")
    print()
    
    try:
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        results = []
        for feature, score in zip(features, mi_scores):
            results.append({
                'Feature': feature,
                'Mutual_Information': score
            })
        
        df_result = pd.DataFrame(results)
        df_result = df_result.sort_values('Mutual_Information', ascending=False)
        
        print("Top 10 mutual information features:")
        print(df_result.head(10).to_string(index=False))
        print()
        
        return df_result
    except Exception as e:
        print("Error computing mutual information: {}".format(str(e)))
        return None

def generate_summary(pearson_df, spearman_df, kendall_df, mi_df, features):
    """生成综合总结"""
    print("=" * 70)
    print("Comprehensive Correlation Analysis Summary")
    print("=" * 70)
    print()
    
    summary_data = []
    
    for feature in features:
        row = {'Feature': feature}
        
        # Pearson
        if pearson_df is not None:
            p_row = pearson_df[pearson_df['Feature'] == feature]
            if not p_row.empty:
                row['Pearson'] = p_row['Pearson_Correlation'].values[0]
        
        # Spearman
        if spearman_df is not None:
            s_row = spearman_df[spearman_df['Feature'] == feature]
            if not s_row.empty:
                row['Spearman'] = s_row['Spearman_Correlation'].values[0]
        
        # Kendall
        if kendall_df is not None:
            k_row = kendall_df[kendall_df['Feature'] == feature]
            if not k_row.empty:
                row['Kendall'] = k_row['Kendall_Tau'].values[0]
        
        # Mutual Information
        if mi_df is not None:
            m_row = mi_df[mi_df['Feature'] == feature]
            if not m_row.empty:
                row['MutualInfo'] = m_row['Mutual_Information'].values[0]
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # 计算平均相关性
    corr_cols = ['Pearson', 'Spearman', 'Kendall']
    summary_df['Avg_Correlation'] = summary_df[corr_cols].abs().mean(axis=1)
    summary_df = summary_df.sort_values('Avg_Correlation', ascending=False)
    
    print("Feature Comprehensive Correlation Ranking (Top 15):")
    print(summary_df.head(15).to_string(index=False))
    print()
    
    return summary_df

def save_results(pearson_df, spearman_df, kendall_df, mi_df, summary_df, output_dir='data'):
    """保存结果"""
    print("=" * 70)
    print("Step 3: Save Results")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    if pearson_df is not None:
        path = os.path.join(output_dir, 'correlation_Pearson.csv')
        pearson_df.to_csv(path, index=False, encoding='utf-8-sig')
        print("Saved Pearson: {}".format(path))
    
    if spearman_df is not None:
        path = os.path.join(output_dir, 'correlation_Spearman.csv')
        spearman_df.to_csv(path, index=False, encoding='utf-8-sig')
        print("Saved Spearman: {}".format(path))
    
    if kendall_df is not None:
        path = os.path.join(output_dir, 'correlation_Kendall.csv')
        kendall_df.to_csv(path, index=False, encoding='utf-8-sig')
        print("Saved Kendall: {}".format(path))
    
    if mi_df is not None:
        path = os.path.join(output_dir, 'correlation_MutualInfo.csv')
        mi_df.to_csv(path, index=False, encoding='utf-8-sig')
        print("Saved Mutual Information: {}".format(path))
    
    path = os.path.join(output_dir, 'correlation_summary.csv')
    summary_df.to_csv(path, index=False, encoding='utf-8-sig')
    print("Saved Summary: {}".format(path))
    
    print()

def main():
    """主函数"""
    print("\n")
    print("*" * 70)
    print("Multimodal Feature Correlation Analysis")
    print("*" * 70)
    print()
    
    # 加载和准备数据
    X, y, features = load_and_prepare_data(sample_size=100000)
    
    # 执行五种相关性分析
    print("=" * 70)
    print("Execute Correlation Analysis")
    print("=" * 70)
    print()
    
    pearson_df = pearson_correlation(X, y, features)
    spearman_df = spearman_correlation(X, y, features)
    kendall_df = kendall_correlation(X, y, features)
    mi_df = mutual_information(X, y, features)
    
    # 生成综合总结
    summary_df = generate_summary(pearson_df, spearman_df, kendall_df, mi_df, features)
    
    # 保存结果
    save_results(pearson_df, spearman_df, kendall_df, mi_df, summary_df)
    
    print("\n")
    print("*" * 70)
    print("Analysis Complete!")
    print("*" * 70)
    print()
    
    return summary_df

if __name__ == "__main__":
    summary_df = main()
    print("Final Comprehensive Correlation Ranking:")
    print(summary_df.head(20))
