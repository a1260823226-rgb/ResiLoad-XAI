# -*- coding: utf-8 -*-
"""
数据合并与清洗脚本 v2（优化版）
按 "变压器 ID + 时间 + 气象站 ID" 关联所有数据，处理缺失值和异常值
修复了 FutureWarning 并增强了缺失值处理
"""
import pandas as pd
import numpy as np
import os
from pathlib import Path

class DataMerger:
    """数据合并与清洗处理器"""
    
    def __init__(self, data_dir='data'):
        """初始化"""
        self.data_dir = data_dir
        self.df_merged = None
        
    def load_data(self):
        """加载所有预处理后的数据"""
        print("=" * 70)
        print("步骤 1: 加载数据")
        print("=" * 70)
        
        # 加载变压器元数据（包含 CLOSEST_STATION 字段）
        print("加载 transformer_meta...")
        self.transformer_meta = pd.read_csv(
            os.path.join(self.data_dir, 'transformer_meta_preprocessed.csv')
        )
        print(f"  ✓ 加载 {len(self.transformer_meta)} 条变压器元数据")
        
        # 加载变压器负荷数据（核心数据）
        print("加载 transformer_raw...")
        # 由于文件较大，分块读取
        chunk_size = 100000
        chunks = []
        for chunk in pd.read_csv(
            os.path.join(self.data_dir, 'transformer_raw_preprocessed.csv'),
            chunksize=chunk_size
        ):
            chunk['DATETIME'] = pd.to_datetime(chunk['DATETIME'])
            chunks.append(chunk)
        self.transformer_raw = pd.concat(chunks, ignore_index=True)
        print(f"  ✓ 加载 {len(self.transformer_raw)} 条负荷数据")
        
        # 加载气象数据
        print("加载 weather...")
        self.weather = pd.read_csv(
            os.path.join(self.data_dir, 'weather_preprocessed.csv')
        )
        self.weather['DATETIME'] = pd.to_datetime(self.weather['DATETIME'])
        print(f"  ✓ 加载 {len(self.weather)} 条气象数据")
        
        # 加载极端天气数据
        print("加载 extreme_weather_calculated...")
        self.extreme_weather = pd.read_csv(
            os.path.join(self.data_dir, 'extreme_weather_calculated_preprocessed.csv')
        )
        self.extreme_weather['DATETIME'] = pd.to_datetime(self.extreme_weather['DATETIME'])
        print(f"  ✓ 加载 {len(self.extreme_weather)} 条极端天气数据")
        
        # 加载节假日数据
        print("加载 holiday...")
        self.holiday = pd.read_csv(
            os.path.join(self.data_dir, 'holiday_preprocessed.csv')
        )
        self.holiday['DATETIME'] = pd.to_datetime(self.holiday['DATETIME'])
        print(f"  ✓ 加载 {len(self.holiday)} 条节假日数据")
        
        print()
    
    def merge_data(self):
        """合并所有数据到 df_merged"""
        print("=" * 70)
        print("步骤 2: 合并数据")
        print("=" * 70)
        
        # 2.1 为 transformer_raw 添加 CLOSEST_STATION 字段
        print("2.1 绑定变压器与最近气象站...")
        self.transformer_raw = self.transformer_raw.merge(
            self.transformer_meta[['TRANSFORMER_ID', 'CLOSEST_STATION']],
            on='TRANSFORMER_ID',
            how='left'
        )
        print(f"  ✓ 已为 {len(self.transformer_raw)} 条负荷数据绑定气象站")
        
        # 2.2 合并气象数据（基于 DATETIME + CLOSEST_STATION）
        print("2.2 合并气象数据...")
        self.df_merged = self.transformer_raw.merge(
            self.weather,
            left_on=['DATETIME', 'CLOSEST_STATION'],
            right_on=['DATETIME', 'STATION_ID'],
            how='left',
            suffixes=('', '_weather')
        )
        # 删除重复的 STATION_ID 列
        if 'STATION_ID_weather' in self.df_merged.columns:
            self.df_merged.drop('STATION_ID_weather', axis=1, inplace=True)
        print(f"  ✓ 合并后数据量: {len(self.df_merged)} 条")
        
        # 2.3 合并极端天气数据（基于 DATETIME + CLOSEST_STATION）
        print("2.3 合并极端天气数据...")
        self.df_merged = self.df_merged.merge(
            self.extreme_weather,
            left_on=['DATETIME', 'CLOSEST_STATION'],
            right_on=['DATETIME', 'STATION_ID'],
            how='left',
            suffixes=('', '_extreme')
        )
        # 删除重复的 STATION_ID 列
        if 'STATION_ID_extreme' in self.df_merged.columns:
            self.df_merged.drop('STATION_ID_extreme', axis=1, inplace=True)
        print(f"  ✓ 合并后数据量: {len(self.df_merged)} 条")
        
        # 2.4 合并节假日数据（基于 DATETIME）
        print("2.4 合并节假日数据...")
        # 只保留日期部分进行匹配
        self.df_merged['DATE'] = self.df_merged['DATETIME'].dt.date
        self.holiday['DATE'] = self.holiday['DATETIME'].dt.date
        
        self.df_merged = self.df_merged.merge(
            self.holiday[['DATE', 'HOLIDAY']],
            on='DATE',
            how='left'
        )
        self.df_merged.drop('DATE', axis=1, inplace=True)
        print(f"  ✓ 合并后数据量: {len(self.df_merged)} 条")
        
        print(f"\n合并完成！最终数据集包含 {len(self.df_merged)} 条记录")
        print(f"字段数量: {len(self.df_merged.columns)}")
        print()
    
    def handle_load_missing_values(self):
        """处理负荷缺失值：按变压器分组线性插值（<3 小时），删除长时缺失"""
        print("=" * 70)
        print("步骤 3: 处理负荷缺失值")
        print("=" * 70)
        
        # 统计初始缺失情况
        initial_missing = self.df_merged['LOAD'].isnull().sum()
        print(f"初始 LOAD 缺失值: {initial_missing} ({initial_missing/len(self.df_merged)*100:.2f}%)")
        
        # 按变压器分组处理
        print("\n按变压器分组进行线性插值...")
        
        def interpolate_load(group):
            """对单个变压器的负荷数据进行插值"""
            # 按时间排序
            group = group.sort_values('DATETIME').copy()
            
            # 标记连续缺失的长度
            group['is_missing'] = group['LOAD'].isnull()
            group['missing_group'] = (group['is_missing'] != group['is_missing'].shift()).cumsum()
            
            # 计算每个缺失组的长度
            missing_lengths = group[group['is_missing']].groupby('missing_group').size()
            
            # 只对缺失长度 <= 3 的进行插值
            for missing_group_id, length in missing_lengths.items():
                if length <= 3:
                    mask = (group['missing_group'] == missing_group_id) & group['is_missing']
                    # 使用 loc 避免 SettingWithCopyWarning
                    group.loc[mask, 'LOAD'] = group.loc[mask, 'LOAD'].interpolate(method='linear')
            
            # 删除辅助列
            group = group.drop(['is_missing', 'missing_group'], axis=1)
            
            return group
        
        # 使用 transform 和自定义函数来避免 include_groups 问题
        result_list = []
        for transformer_id, group in self.df_merged.groupby('TRANSFORMER_ID'):
            result_list.append(interpolate_load(group))
        self.df_merged = pd.concat(result_list, ignore_index=True)
        
        # 删除剩余的长时缺失数据
        remaining_missing = self.df_merged['LOAD'].isnull().sum()
        print(f"插值后剩余缺失值: {remaining_missing} ({remaining_missing/len(self.df_merged)*100:.2f}%)")
        
        print("删除长时缺失数据（>3 小时）...")
        self.df_merged = self.df_merged.dropna(subset=['LOAD'])
        
        final_count = len(self.df_merged)
        print(f"  ✓ 删除后数据量: {final_count} 条")
        print(f"  ✓ LOAD 缺失值: {self.df_merged['LOAD'].isnull().sum()}")
        print()
    
    def handle_weather_missing_values(self):
        """处理气象缺失值：核心字段用前/后值填充，PRCP 填充 0"""
        print("=" * 70)
        print("步骤 4: 处理气象缺失值")
        print("=" * 70)
        
        # 定义核心气象字段
        core_weather_fields = ['TEMP', 'RH', 'WDSP']
        # 其他气象字段
        other_weather_fields = ['DEWP', 'SLP', 'STP', 'VISIB', 'MXSPD', 'GUST', 'MAX', 'MIN']
        
        # 统计初始缺失情况
        print("初始气象字段缺失情况:")
        for field in core_weather_fields + ['PRCP'] + other_weather_fields:
            if field in self.df_merged.columns:
                missing = self.df_merged[field].isnull().sum()
                if missing > 0:
                    print(f"  {field}: {missing} ({missing/len(self.df_merged)*100:.2f}%)")
        
        # 4.1 PRCP 填充 0（降水量缺失通常表示无降水）
        print("\n4.1 PRCP 缺失值填充 0...")
        if 'PRCP' in self.df_merged.columns:
            prcp_missing = self.df_merged['PRCP'].isnull().sum()
            self.df_merged['PRCP'] = self.df_merged['PRCP'].fillna(0)
            print(f"  ✓ 填充 {prcp_missing} 个 PRCP 缺失值")
        
        # 4.2 核心字段使用前/后值填充（时间序列优化方法）
        print("\n4.2 核心字段（TEMP, RH, WDSP）使用前/后值填充...")
        
        # 按气象站和时间排序
        self.df_merged = self.df_merged.sort_values(['CLOSEST_STATION', 'DATETIME'])
        
        for field in core_weather_fields:
            if field not in self.df_merged.columns:
                continue
            
            initial_missing = self.df_merged[field].isnull().sum()
            if initial_missing == 0:
                continue
            
            # 按气象站分组，使用前向填充 + 后向填充
            self.df_merged[field] = self.df_merged.groupby('CLOSEST_STATION')[field].transform(
                lambda x: x.ffill().bfill()
            )
            
            # 如果还有缺失（整个气象站都缺失），用全局中位数填充
            remaining_missing = self.df_merged[field].isnull().sum()
            if remaining_missing > 0:
                global_median = self.df_merged[field].median()
                self.df_merged[field] = self.df_merged[field].fillna(global_median)
                print(f"  {field}: 前/后值填充 {initial_missing - remaining_missing} 个，"
                      f"全局中位数填充 {remaining_missing} 个")
            else:
                print(f"  {field}: 前/后值填充 {initial_missing} 个")
        
        # 4.3 其他气象字段也使用前/后值填充
        print("\n4.3 其他气象字段使用前/后值填充...")
        for field in other_weather_fields:
            if field not in self.df_merged.columns:
                continue
            
            initial_missing = self.df_merged[field].isnull().sum()
            if initial_missing == 0:
                continue
            
            # 按气象站分组，使用前向填充 + 后向填充
            self.df_merged[field] = self.df_merged.groupby('CLOSEST_STATION')[field].transform(
                lambda x: x.ffill().bfill()
            )
            
            # 如果还有缺失，用全局中位数填充
            remaining_missing = self.df_merged[field].isnull().sum()
            if remaining_missing > 0:
                global_median = self.df_merged[field].median()
                self.df_merged[field] = self.df_merged[field].fillna(global_median)
            
            if initial_missing > 0:
                print(f"  {field}: 填充 {initial_missing} 个缺失值")
        
        # 4.4 SNDP（积雪深度）填充 0
        if 'SNDP' in self.df_merged.columns:
            sndp_missing = self.df_merged['SNDP'].isnull().sum()
            if sndp_missing > 0:
                self.df_merged['SNDP'] = self.df_merged['SNDP'].fillna(0)
                print(f"\n4.4 SNDP（积雪深度）填充 0: {sndp_missing} 个")
        
        # 4.5 极端天气字段填充 0（缺失表示无极端天气）
        print("\n4.5 极端天气字段填充 0（缺失表示无极端天气）...")
        extreme_weather_cols = [col for col in self.df_merged.columns 
                               if any(x in col for x in ['HIGH_TEMPERATURE', 'LOW_TEMPERATURE', 
                                                          'HIGH_HUMIDITY', 'HEAT_INDEX', 'WIND_CHILL',
                                                          'WIND_LEVEL', 'PRECIPITATION'])]
        
        filled_count = 0
        for col in extreme_weather_cols:
            missing = self.df_merged[col].isnull().sum()
            if missing > 0:
                self.df_merged[col] = self.df_merged[col].fillna(0)
                filled_count += missing
        
        if filled_count > 0:
            print(f"  ✓ 填充 {len(extreme_weather_cols)} 个极端天气字段，共 {filled_count} 个缺失值")
        
        # 4.6 STATION_ID 填充（使用 CLOSEST_STATION）
        if 'STATION_ID' in self.df_merged.columns:
            station_missing = self.df_merged['STATION_ID'].isnull().sum()
            if station_missing > 0:
                self.df_merged['STATION_ID'] = self.df_merged['STATION_ID'].fillna(
                    self.df_merged['CLOSEST_STATION']
                )
                print(f"\n4.6 STATION_ID 使用 CLOSEST_STATION 填充: {station_missing} 个")
        
        print(f"\n  ✓ 时间序列填充完成（高效方法）")
        
        # 验证插补结果
        print("\n插补后缺失值统计:")
        missing_summary = self.df_merged.isnull().sum()
        missing_summary = missing_summary[missing_summary > 0]
        if len(missing_summary) > 0:
            for field, count in missing_summary.items():
                print(f"  {field}: {count} ({count/len(self.df_merged)*100:.2f}%)")
        else:
            print("  ✓ 所有字段无缺失值")
        
        print()
    
    def remove_load_outliers(self):
        """用 3σ 原则过滤 LOAD 的异常值，保留极端天气导致的合理波动"""
        print("=" * 70)
        print("步骤 5: 过滤负荷异常值（3σ 原则）")
        print("=" * 70)
        
        initial_count = len(self.df_merged)
        
        # 按变压器分组计算 3σ 阈值
        print("按变压器分组计算 3σ 阈值...")
        
        def filter_outliers(group):
            """对单个变压器的负荷数据过滤异常值"""
            group = group.copy()
            mean = group['LOAD'].mean()
            std = group['LOAD'].std()
            
            # 3σ 阈值
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            
            # 标记异常值
            group['is_outlier'] = (group['LOAD'] < lower_bound) | (group['LOAD'] > upper_bound)
            
            # 检查是否为极端天气期间（保留这些异常值）
            # 只有当极端天气字段值 > 0 时才认为是真正的极端天气
            extreme_weather_cols = [col for col in group.columns if 'HIGH_TEMPERATURE' in col 
                                   or 'LOW_TEMPERATURE' in col or 'HIGH_HUMIDITY' in col
                                   or 'WIND_LEVEL' in col or 'PRECIPITATION' in col]
            
            if extreme_weather_cols:
                # 检查是否有任何极端天气标记 > 0（不是填充的 0）
                has_extreme_weather = (group[extreme_weather_cols] > 0).any(axis=1)
                group['is_outlier'] = group['is_outlier'] & ~has_extreme_weather
            
            return group
        
        # 使用循环来避免 include_groups 问题
        result_list = []
        for transformer_id, group in self.df_merged.groupby('TRANSFORMER_ID'):
            result_list.append(filter_outliers(group))
        self.df_merged = pd.concat(result_list, ignore_index=True)
        
        # 统计异常值数量
        outlier_count = self.df_merged['is_outlier'].sum()
        print(f"检测到 {outlier_count} 个异常值 ({outlier_count/initial_count*100:.2f}%)")
        
        # 删除异常值
        self.df_merged = self.df_merged[~self.df_merged['is_outlier']]
        self.df_merged = self.df_merged.drop('is_outlier', axis=1)
        
        final_count = len(self.df_merged)
        removed_count = initial_count - final_count
        print(f"  ✓ 删除 {removed_count} 条异常数据")
        print(f"  ✓ 保留 {final_count} 条数据")
        print()
    
    def save_merged_data(self, output_path='data/df_merged.csv'):
        """保存合并后的数据集"""
        print("=" * 70)
        print("步骤 6: 保存合并数据集")
        print("=" * 70)
        
        # 保存数据
        print(f"保存到 {output_path}...")
        self.df_merged.to_csv(output_path, index=False)
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"  ✓ 保存成功")
        print(f"  ✓ 文件大小: {file_size:.2f} MB")
        print(f"  ✓ 数据量: {len(self.df_merged)} 条")
        print(f"  ✓ 字段数: {len(self.df_merged.columns)}")
        print()
    
    def print_summary(self):
        """打印数据摘要"""
        print("=" * 70)
        print("数据摘要")
        print("=" * 70)
        
        print(f"\n数据集形状: {self.df_merged.shape}")
        print(f"时间范围: {self.df_merged['DATETIME'].min()} 至 {self.df_merged['DATETIME'].max()}")
        print(f"变压器数量: {self.df_merged['TRANSFORMER_ID'].nunique()}")
        print(f"气象站数量: {self.df_merged['CLOSEST_STATION'].nunique()}")
        
        print("\n主要字段统计:")
        print(f"  LOAD - 均值: {self.df_merged['LOAD'].mean():.2f}, "
              f"标准差: {self.df_merged['LOAD'].std():.2f}, "
              f"范围: [{self.df_merged['LOAD'].min():.2f}, {self.df_merged['LOAD'].max():.2f}]")
        
        if 'TEMP' in self.df_merged.columns:
            print(f"  TEMP - 均值: {self.df_merged['TEMP'].mean():.2f}, "
                  f"标准差: {self.df_merged['TEMP'].std():.2f}, "
                  f"范围: [{self.df_merged['TEMP'].min():.2f}, {self.df_merged['TEMP'].max():.2f}]")
        
        if 'RH' in self.df_merged.columns:
            print(f"  RH - 均值: {self.df_merged['RH'].mean():.2f}, "
                  f"标准差: {self.df_merged['RH'].std():.2f}, "
                  f"范围: [{self.df_merged['RH'].min():.2f}, {self.df_merged['RH'].max():.2f}]")
        
        print("\n缺失值统计:")
        missing = self.df_merged.isnull().sum()
        if missing.any():
            for col, count in missing[missing > 0].items():
                print(f"  {col}: {count} ({count/len(self.df_merged)*100:.2f}%)")
        else:
            print("  ✓ 无缺失值")
        
        print("\n字段列表:")
        for i, col in enumerate(self.df_merged.columns, 1):
            print(f"  {i}. {col} ({self.df_merged[col].dtype})")
        
        print("\n" + "=" * 70)
    
    def run(self):
        """执行完整的数据合并与清洗流程"""
        print("\n")
        print("*" * 70)
        print("数据合并与清洗流程 v2（优化版）")
        print("*" * 70)
        print()
        
        # 步骤 1: 加载数据
        self.load_data()
        
        # 步骤 2: 合并数据
        self.merge_data()
        
        # 步骤 3: 处理负荷缺失值
        self.handle_load_missing_values()
        
        # 步骤 4: 处理气象缺失值
        self.handle_weather_missing_values()
        
        # 步骤 5: 过滤异常值
        self.remove_load_outliers()
        
        # 步骤 6: 保存数据
        self.save_merged_data()
        
        # 打印摘要
        self.print_summary()
        
        print("\n")
        print("*" * 70)
        print("数据处理完成！")
        print("*" * 70)
        print()
        
        return self.df_merged


def main():
    """主函数"""
    # 创建数据合并器
    merger = DataMerger(data_dir='data')
    
    # 执行完整流程
    df_merged = merger.run()
    
    # 返回合并后的数据集
    return df_merged


if __name__ == "__main__":
    df_merged = main()
    
    # 示例：查看前几行数据
    print("\n数据预览（前 5 行）:")
    print(df_merged.head())
    
    print("\n数据预览（后 5 行）:")
    print(df_merged.tail())
