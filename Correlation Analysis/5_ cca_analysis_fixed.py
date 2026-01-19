# -*- coding: utf-8 -*-
"""
典型相关分析 (Canonical Correlation Analysis, CCA) - 修复版
用于多模态特征融合分析
"""
import pandas as pd
import numpy as np
import warnings
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
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
    
    # 清洗数据
    df_clean = df.dropna()
    print("Cleaned data: {} rows".format(len(df_clean)))
    
    # 采样
    if len(df_clean) > sample_size:
        print("Sampling {} rows for faster computation...".format(sample_size))
        df_clean = df_clean.sample(n=sample_size, random_state=42)
        print("Sampled data: {} rows".format(len(df_clean)))
    
    print()
    return df_clean

def cca_weather_vs_extreme(df):
    """CCA: 气象特征 vs 极端天气特征"""
    print("=" * 70)
    print("CCA Analysis 1: Weather Features vs Extreme Weather Features")
    print("=" * 70)
    print()
    
    # 第一组: 气象特征
    weather_features = ['TEMP', 'RH', 'WDSP', 'PRCP', 'DEWP', 'SLP', 'STP', 'VISIB', 'MXSPD', 'GUST', 'MAX', 'MIN']
    
    # 第二组: 极端天气特征
    extreme_features = ['HIGH_TEMPERATURE', 'LOW_TEMPERATURE', 'HIGH_HUMIDITY',
                       'HEAT_INDEX_CAUTION', 'HEAT_INDEX_EXTREME_CAUTION', 'HEAT_INDEX_DANGER',
                       'WIND_LEVEL_1', 'WIND_LEVEL_2', 'WIND_LEVEL_3', 'WIND_LEVEL_4',
                       'WIND_LEVEL_5', 'WIND_LEVEL_6', 'WIND_LEVEL_7', 'WIND_LEVEL_8',
                       'PRECIPITATION_50', 'PRECIPITATION_100']
    
    # 获取可用的特征
    available_weather = [f for f in weather_features if f in df.columns]
    available_extreme = [f for f in extreme_features if f in df.columns]
    
    print("Weather features: {} features".format(len(available_weather)))
    print("  {}".format(', '.join(available_weather)))
    print()
    print("Extreme weather features: {} features".format(len(available_extreme)))
    print("  {}".format(', '.join(available_extreme)))
    print()
    
    # 提取数据
    X_weather = df[available_weather].values
    X_extreme = df[available_extreme].values
    
    # 标准化
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    X_weather_scaled = scaler1.fit_transform(X_weather)
    X_extreme_scaled = scaler2.fit_transform(X_extreme)
    
    print("Weather features shape: {}".format(X_weather_scaled.shape))
    print("Extreme features shape: {}".format(X_extreme_scaled.shape))
    print()
    
    # 执行 CCA
    n_components = min(X_weather_scaled.shape[1], X_extreme_scaled.shape[1], 8)
    print("Number of canonical components: {}".format(n_components))
    print()
    
    cca = CCA(n_components=n_components)
    cca.fit(X_weather_scaled, X_extreme_scaled)
    
    # 获取典型变量
    U = cca.transform(X_weather_scaled)
    V = cca.transform(X_extreme_scaled)
    
    # 计算典型相关系数
    canonical_corrs = []
    for i in range(n_components):
        corr = np.corrcoef(U[:, i], V[:, i])[0, 1]
        canonical_corrs.append(corr)
    
    # 创建结果数据框
    cca_results = pd.DataFrame({
        'Component': ['CC{}'.format(i+1) for i in range(n_components)],
        'Canonical_Correlation': canonical_corrs,
        'Squared_Correlation': np.array(canonical_corrs)**2
    })
    
    print("Canonical Correlation Results:")
    print(cca_results.to_string(index=False))
    print()
    
    # 获取典型变量的权重
    print("=" * 70)
    print("Canonical Loadings (Weather Features)")
    print("=" * 70)
    
    weather_loadings = pd.DataFrame(
        cca.x_weights_,
        columns=['CC{}'.format(i+1) for i in range(n_components)],
        index=available_weather
    )
    
    print(weather_loadings.to_string())
    print()
    
    print("=" * 70)
    print("Canonical Loadings (Extreme Weather Features)")
    print("=" * 70)
    
    extreme_loadings = pd.DataFrame(
        cca.y_weights_,
        columns=['CC{}'.format(i+1) for i in range(n_components)],
        index=available_extreme
    )
    
    print(extreme_loadings.to_string())
    print()
    
    # 计算特征对典型变量的贡献
    print("=" * 70)
    print("Feature Contributions to First Canonical Variate (CC1)")
    print("=" * 70)
    
    weather_contrib = pd.DataFrame({
        'Feature': available_weather,
        'CC1_Loading': cca.x_weights_[:, 0],
        'CC1_Loading_Abs': np.abs(cca.x_weights_[:, 0])
    }).sort_values('CC1_Loading_Abs', ascending=False)
    
    extreme_contrib = pd.DataFrame({
        'Feature': available_extreme,
        'CC1_Loading': cca.y_weights_[:, 0],
        'CC1_Loading_Abs': np.abs(cca.y_weights_[:, 0])
    }).sort_values('CC1_Loading_Abs', ascending=False)
    
    print("\nWeather Features Contribution to CC1:")
    print(weather_contrib.head(10).to_string(index=False))
    
    print("\nExtreme Weather Features Contribution to CC1:")
    print(extreme_contrib.head(10).to_string(index=False))
    print()
    
    return cca_results, weather_loadings, extreme_loadings, weather_contrib, extreme_contrib

def cca_weather_vs_load(df):
    """CCA: 气象特征 vs 负荷"""
    print("=" * 70)
    print("CCA Analysis 2: Weather Features vs Load")
    print("=" * 70)
    print()
    
    # 第一组: 气象特征
    weather_features = ['TEMP', 'RH', 'WDSP', 'PRCP', 'DEWP', 'SLP', 'STP', 'VISIB', 'MXSPD', 'GUST', 'MAX', 'MIN']
    
    # 第二组: 负荷和位置特征
    load_features = ['LOAD', 'CLOSEST_STATION']
    
    available_weather = [f for f in weather_features if f in df.columns]
    available_load = [f for f in load_features if f in df.columns]
    
    if len(available_load) < 2:
        print("Not enough load-related features for CCA")
        return None, None, None, None
    
    print("Weather features: {} features".format(len(available_weather)))
    print("Load-related features: {} features".format(len(available_load)))
    print()
    
    X_weather = df[available_weather].values
    X_load = df[available_load].values
    
    # 标准化
    scaler1 = StandardScaler()
    scaler2 = StandardScaler()
    X_weather_scaled = scaler1.fit_transform(X_weather)
    X_load_scaled = scaler2.fit_transform(X_load)
    
    # 执行 CCA
    n_components = min(X_weather_scaled.shape[1], X_load_scaled.shape[1], 5)
    cca = CCA(n_components=n_components)
    cca.fit(X_weather_scaled, X_load_scaled)
    
    # 获取典型变量
    U = cca.transform(X_weather_scaled)
    V = cca.transform(X_load_scaled)
    
    # 计算典型相关系数
    canonical_corrs = []
    for i in range(n_components):
        corr = np.corrcoef(U[:, i], V[:, i])[0, 1]
        canonical_corrs.append(corr)
    
    cca_results = pd.DataFrame({
        'Component': ['CC{}'.format(i+1) for i in range(n_components)],
        'Canonical_Correlation': canonical_corrs,
        'Squared_Correlation': np.array(canonical_corrs)**2
    })
    
    print("Canonical Correlation Results:")
    print(cca_results.to_string(index=False))
    print()
    
    # 特征贡献
    weather_contrib = pd.DataFrame({
        'Feature': available_weather,
        'CC1_Loading': cca.x_weights_[:, 0],
        'CC1_Loading_Abs': np.abs(cca.x_weights_[:, 0])
    }).sort_values('CC1_Loading_Abs', ascending=False)
    
    load_contrib = pd.DataFrame({
        'Feature': available_load,
        'CC1_Loading': cca.y_weights_[:, 0],
        'CC1_Loading_Abs': np.abs(cca.y_weights_[:, 0])
    }).sort_values('CC1_Loading_Abs', ascending=False)
    
    print("Weather Features Contribution to CC1:")
    print(weather_contrib.to_string(index=False))
    print()
    
    print("Load-Related Features Contribution to CC1:")
    print(load_contrib.to_string(index=False))
    print()
    
    return cca_results, weather_contrib, load_contrib, cca

def save_cca_results(cca_results1, weather_loadings1, extreme_loadings1, 
                     weather_contrib1, extreme_contrib1,
                     cca_results2, weather_contrib2, load_contrib2,
                     output_dir='data'):
    """保存 CCA 结果"""
    print("=" * 70)
    print("Step 2: Save Results")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存 CCA 1 结果
    path = os.path.join(output_dir, 'cca_weather_vs_extreme_correlations.csv')
    cca_results1.to_csv(path, index=False, encoding='utf-8-sig')
    print("Saved CCA 1 correlations: {}".format(path))
    
    path = os.path.join(output_dir, 'cca_weather_loadings.csv')
    weather_loadings1.to_csv(path, encoding='utf-8-sig')
    print("Saved weather loadings: {}".format(path))
    
    path = os.path.join(output_dir, 'cca_extreme_loadings.csv')
    extreme_loadings1.to_csv(path, encoding='utf-8-sig')
    print("Saved extreme weather loadings: {}".format(path))
    
    path = os.path.join(output_dir, 'cca_weather_contributions.csv')
    weather_contrib1.to_csv(path, index=False, encoding='utf-8-sig')
    print("Saved weather contributions: {}".format(path))
    
    path = os.path.join(output_dir, 'cca_extreme_contributions.csv')
    extreme_contrib1.to_csv(path, index=False, encoding='utf-8-sig')
    print("Saved extreme weather contributions: {}".format(path))
    
    # 保存 CCA 2 结果
    if cca_results2 is not None:
        path = os.path.join(output_dir, 'cca_weather_vs_load_correlations.csv')
        cca_results2.to_csv(path, index=False, encoding='utf-8-sig')
        print("Saved CCA 2 correlations: {}".format(path))
        
        path = os.path.join(output_dir, 'cca_weather_load_contributions.csv')
        weather_contrib2.to_csv(path, index=False, encoding='utf-8-sig')
        print("Saved weather-load contributions: {}".format(path))
        
        path = os.path.join(output_dir, 'cca_load_contributions.csv')
        load_contrib2.to_csv(path, index=False, encoding='utf-8-sig')
        print("Saved load contributions: {}".format(path))
    
    print()

def main():
    """主函数"""
    print("\n")
    print("*" * 70)
    print("Canonical Correlation Analysis (CCA)")
    print("*" * 70)
    print()
    
    # 加载数据
    df = load_and_prepare_data(sample_size=100000)
    
    # 执行 CCA 分析 1: 气象 vs 极端天气
    cca_results1, weather_loadings1, extreme_loadings1, weather_contrib1, extreme_contrib1 = cca_weather_vs_extreme(df)
    
    # 执行 CCA 分析 2: 气象 vs 负荷
    print()
    cca_results2, weather_contrib2, load_contrib2, cca2 = cca_weather_vs_load(df)
    
    # 保存结果
    save_cca_results(cca_results1, weather_loadings1, extreme_loadings1, 
                     weather_contrib1, extreme_contrib1,
                     cca_results2, weather_contrib2, load_contrib2)
    
    print("\n")
    print("*" * 70)
    print("CCA Analysis Complete!")
    print("*" * 70)
    print()
    
    return cca_results1, cca_results2

if __name__ == "__main__":
    cca_results1, cca_results2 = main()
    
    print("Summary of CCA Results:")
    print("\nCCA 1: Weather vs Extreme Weather")
    print(cca_results1)
    
    if cca_results2 is not None:
        print("\nCCA 2: Weather vs Load")
        print(cca_results2)
