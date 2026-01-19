# -*- coding: utf-8 -*-
"""
多模态特征相关性分析脚本
对居民负荷数据进行五种相关性分析方法：
1. Pearson相关系数
2. Spearman秩相关系数
3. Kendall tau相关系数
4. 典型相关分析 (CCA)
5. 互信息 (Mutual Information)
"""
import pandas as pd
import numpy as np
import warnings
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
import os

warnings.filterwarnings('ignore')

class MultimodalCorrelationAnalysis:
    """多模态特征相关性分析器"""
    
    def __init__(self, data_path='data/df_merged.csv'):
        """初始化"""
        self.data_path = data_path
        self.df = None
        self.results = {}
        self.numeric_features = None
        self.target = 'LOAD'
        
    def load_data(self):
        """加载数据"""
        print("=" * 70)
        print("步骤 1: 加载数据")
        print("=" * 70)
        
        print(f"加载 {self.data_path}...")
        self.df = pd.read_csv(self.data_path, parse_dates=['DATETIME'])
        print(f"✓ 加载完成: {len(self.df)} 行, {len(self.df.columns)} 列")
        
        # 选择数值特征（排除 TRANSFORMER_ID, DATETIME, HOLIDAY）
        exclude_cols = ['TRANSFORMER_ID', 'DATETIME', 'HOLIDAY', 'STATION_ID']
        self.numeric_features = [col for col in self.df.columns 
                                if col not in exclude_cols and self.df[col].dtype in ['float64', 'int64']]
        
        print(f"✓ 数值特征数: {len(self.numeric_features)}")
        print(f"✓ 目标变量: {self.target}")
        print()
        
    def prepare_data(self):
        """准备数据"""
        print("=" * 70)
        print("步骤 2: 数据准备")
        print("=" * 70)
        
        # 移除包含 NaN 的行
        df_clean = self.df[self.numeric_features + [self.target]].dropna()
        
        print(f"原始数据: {len(self.df)} 行")
        print(f"清洗后: {len(df_clean)} 行")
        print(f"移除行数: {len(self.df) - len(df_clean)}")
        
        self.df_clean = df_clean
        self.X = df_clean[self.numeric_features].values
        self.y = df_clean[self.target].values
        
        print(f"✓ 特征矩阵形状: {self.X.shape}")
        print(f"✓ 目标向量形状: {self.y.shape}")
        print()
        
    def pearson_correlation(self):
        """1. Pearson相关系数"""
        print("=" * 70)
        print("方法 1: Pearson相关系数")
        print("=" * 70)
        print("描述: 测量变量间的线性相关性")
        print("适用: 连续数据，多模态特征融合前筛选")
        print()
        
        pearson_results = []
        
        for i, feature in enumerate(self.numeric_features):
            try:
                # 计算 Pearson 相关系数和 p 值
                corr, p_value = pearsonr(self.X[:, i], self.y)
                pearson_results.append({
                    'Feature': feature,
                    'Pearson_Correlation': corr,
                    'P_Value': p_value,
                    'Significant': 'Yes' if p_value < 0.05 else 'No'
                })
            except Exception as e:
                print(f"  ⚠ {feature}: {str(e)}")
                pearson_results.append({
                    'Feature': feature,
                    'Pearson_Correlation': np.nan,
                    'P_Value': np.nan,
                    'Significant': 'Error'
                })
        
        pearson_df = pd.DataFrame(pearson_results)
        pearson_df = pearson_df.sort_values('Pearson_Correlation', key=abs, ascending=False)
        
        print("Top 10 相关特征:")
        print(pearson_df.head(10).to_string(index=False))
        print()
        
        self.results['Pearson'] = pearson_df
        return pearson_df
    
    def spearman_correlation(self):
        """2. Spearman秩相关系数"""
        print("=" * 70)
        print("方法 2: Spearman秩相关系数")
        print("=" * 70)
        print("描述: 测量变量间的单调相关性，不受分布影响")
        print("适用: 非正态分布的多模态特征")
        print()
        
        spearman_results = []
        
        for i, feature in enumerate(self.numeric_features):
            try:
                # 计算 Spearman 相关系数和 p 值
                corr, p_value = spearmanr(self.X[:, i], self.y)
                spearman_results.append({
                    'Feature': feature,
                    'Spearman_Correlation': corr,
                    'P_Value': p_value,
                    'Significant': 'Yes' if p_value < 0.05 else 'No'
                })
            except Exception as e:
                print(f"  ⚠ {feature}: {str(e)}")
                spearman_results.append({
                    'Feature': feature,
                    'Spearman_Correlation': np.nan,
                    'P_Value': np.nan,
                    'Significant': 'Error'
                })
        
        spearman_df = pd.DataFrame(spearman_results)
        spearman_df = spearman_df.sort_values('Spearman_Correlation', key=abs, ascending=False)
        
        print("Top 10 相关特征:")
        print(spearman_df.head(10).to_string(index=False))
        print()
        
        self.results['Spearman'] = spearman_df
        return spearman_df
    
    def kendall_correlation(self):
        """3. Kendall tau相关系数"""
        print("=" * 70)
        print("方法 3: Kendall tau相关系数")
        print("=" * 70)
        print("描述: 基于排序的一致性测量，适用于小样本或异常值")
        print("适用: 时间序列特征的序数相关")
        print()
        
        kendall_results = []
        
        for i, feature in enumerate(self.numeric_features):
            try:
                # 计算 Kendall tau 相关系数和 p 值
                corr, p_value = kendalltau(self.X[:, i], self.y)
                kendall_results.append({
                    'Feature': feature,
                    'Kendall_Tau': corr,
                    'P_Value': p_value,
                    'Significant': 'Yes' if p_value < 0.05 else 'No'
                })
            except Exception as e:
                print(f"  ⚠ {feature}: {str(e)}")
                kendall_results.append({
                    'Feature': feature,
                    'Kendall_Tau': np.nan,
                    'P_Value': np.nan,
                    'Significant': 'Error'
                })
        
        kendall_df = pd.DataFrame(kendall_results)
        kendall_df = kendall_df.sort_values('Kendall_Tau', key=abs, ascending=False)
        
        print("Top 10 相关特征:")
        print(kendall_df.head(10).to_string(index=False))
        print()
        
        self.results['Kendall'] = kendall_df
        return kendall_df
    
    def canonical_correlation_analysis(self):
        """4. 典型相关分析 (CCA)"""
        print("=" * 70)
        print("方法 4: 典型相关分析 (CCA)")
        print("=" * 70)
        print("描述: 测量两组变量集间的最大相关性")
        print("适用: 多模态特征融合，跨模态影响分析")
        print()
        
        try:
            # 将特征分为两组：气象特征和其他特征
            weather_features = ['TEMP', 'RH', 'WDSP', 'PRCP', 'DEWP', 'SLP', 'STP', 'VISIB', 'MXSPD', 'GUST', 'MAX', 'MIN']
            other_features = [f for f in self.numeric_features if f not in weather_features]
            
            # 获取两组特征的索引
            weather_idx = [self.numeric_features.index(f) for f in weather_features if f in self.numeric_features]
            other_idx = [self.numeric_features.index(f) for f in other_features if f in self.numeric_features]
            
            if len(weather_idx) > 0 and len(other_idx) > 0:
                X_weather = self.X[:, weather_idx]
                X_other = self.X[:, other_idx]
                
                # 标准化数据
                scaler1 = StandardScaler()
                scaler2 = StandardScaler()
                X_weather_scaled = scaler1.fit_transform(X_weather)
                X_other_scaled = scaler2.fit_transform(X_other)
                
                # 执行 CCA
                n_components = min(X_weather_scaled.shape[1], X_other_scaled.shape[1], 5)
                cca = CCA(n_components=n_components)
                cca.fit(X_weather_scaled, X_other_scaled)
                
                # 获取典型相关系数
                canonical_corrs = []
                for i in range(n_components):
                    U = cca.transform(X_weather_scaled)[:, i]
                    V = cca.transform(X_other_scaled)[:, i]
                    corr = np.corrcoef(U, V)[0, 1]
                    canonical_corrs.append(corr)
                
                cca_results = pd.DataFrame({
                    'Component': [f'CC{i+1}' for i in range(n_components)],
                    'Canonical_Correlation': canonical_corrs,
                    'Weather_Features': len(weather_idx),
                    'Other_Features': len(other_idx)
                })
                
                print(f"气象特征组: {len(weather_idx)} 个")
                print(f"其他特征组: {len(other_idx)} 个")
                print(f"典型相关分量: {n_components}")
                print()
                print("典型相关系数:")
                print(cca_results.to_string(index=False))
                print()
                
                self.results['CCA'] = cca_results
                return cca_results
            else:
                print("⚠ 特征分组不足，跳过 CCA 分析")
                return None
                
        except Exception as e:
            print(f"✗ CCA 分析失败: {str(e)}")
            return None
    
    def mutual_information(self):
        """5. 互信息 (Mutual Information)"""
        print("=" * 70)
        print("方法 5: 互信息 (Mutual Information)")
        print("=" * 70)
        print("描述: 量化变量间的依赖性（包括非线性）")
        print("适用: 多模态特征选择，识别无关特征")
        print()
        
        try:
            # 计算互信息
            mi_scores = mutual_info_regression(self.X, self.y, random_state=42)
            
            mi_results = []
            for feature, score in zip(self.numeric_features, mi_scores):
                mi_results.append({
                    'Feature': feature,
                    'Mutual_Information': score
                })
            
            mi_df = pd.DataFrame(mi_results)
            mi_df = mi_df.sort_values('Mutual_Information', ascending=False)
            
            print("Top 10 互信息特征:")
            print(mi_df.head(10).to_string(index=False))
            print()
            
            self.results['MutualInfo'] = mi_df
            return mi_df
            
        except Exception as e:
            print(f"✗ 互信息计算失败: {str(e)}")
            return None
    
    def generate_summary_report(self):
        """生成综合总结报告"""
        print("=" * 70)
        print("综合相关性分析总结")
        print("=" * 70)
        print()
        
        # 创建综合排名表
        summary_data = []
        
        for feature in self.numeric_features:
            row = {'Feature': feature}
            
            # Pearson
            if 'Pearson' in self.results:
                pearson_row = self.results['Pearson'][self.results['Pearson']['Feature'] == feature]
                if not pearson_row.empty:
                    row['Pearson'] = pearson_row['Pearson_Correlation'].values[0]
            
            # Spearman
            if 'Spearman' in self.results:
                spearman_row = self.results['Spearman'][self.results['Spearman']['Feature'] == feature]
                if not spearman_row.empty:
                    row['Spearman'] = spearman_row['Spearman_Correlation'].values[0]
            
            # Kendall
            if 'Kendall' in self.results:
                kendall_row = self.results['Kendall'][self.results['Kendall']['Feature'] == feature]
                if not kendall_row.empty:
                    row['Kendall'] = kendall_row['Kendall_Tau'].values[0]
            
            # Mutual Information
            if 'MutualInfo' in self.results:
                mi_row = self.results['MutualInfo'][self.results['MutualInfo']['Feature'] == feature]
                if not mi_row.empty:
                    row['MutualInfo'] = mi_row['Mutual_Information'].values[0]
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # 计算平均相关性（用于排名）
        corr_cols = ['Pearson', 'Spearman', 'Kendall']
        summary_df['Avg_Correlation'] = summary_df[corr_cols].abs().mean(axis=1)
        summary_df = summary_df.sort_values('Avg_Correlation', ascending=False)
        
        print("特征综合相关性排名 (Top 15):")
        print(summary_df.head(15).to_string(index=False))
        print()
        
        return summary_df
    
    def save_results(self, output_dir='data'):
        """保存所有结果到 CSV 文件"""
        print("=" * 70)
        print("步骤 3: 保存结果")
        print("=" * 70)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存各方法的详细结果
        for method_name, result_df in self.results.items():
            if result_df is not None:
                output_path = os.path.join(output_dir, f'correlation_{method_name}.csv')
                result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
                print(f"✓ 保存 {method_name}: {output_path}")
        
        # 保存综合总结
        summary_df = self.generate_summary_report()
        summary_path = os.path.join(output_dir, 'correlation_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"✓ 保存综合总结: {summary_path}")
        
        print()
        return summary_df
    
    def run(self):
        """执行完整分析流程"""
        print("\n")
        print("*" * 70)
        print("多模态特征相关性分析")
        print("*" * 70)
        print()
        
        # 步骤 1: 加载数据
        self.load_data()
        
        # 步骤 2: 准备数据
        self.prepare_data()
        
        # 步骤 3: 执行五种相关性分析
        print("=" * 70)
        print("执行相关性分析")
        print("=" * 70)
        print()
        
        self.pearson_correlation()
        self.spearman_correlation()
        self.kendall_correlation()
        self.canonical_correlation_analysis()
        self.mutual_information()
        
        # 步骤 4: 保存结果
        summary_df = self.save_results()
        
        print("\n")
        print("*" * 70)
        print("分析完成！")
        print("*" * 70)
        print()
        
        return summary_df


def main():
    """主函数"""
    analyzer = MultimodalCorrelationAnalysis(data_path='data/df_merged.csv')
    summary_df = analyzer.run()
    return summary_df


if __name__ == "__main__":
    summary_df = main()
    
    # 显示最终结果
    print("\n最终综合相关性排名:")
    print(summary_df.head(20))
