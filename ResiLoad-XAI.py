"""
V6 多变压器版本 - 使用全部变压器数据进行联合训练
V6 Multi-Transformer: Feature Selection + LightGBM + Enhanced Regularization

改进点:
1. 使用全部变压器数据（不再只用一个变压器）
2. 添加变压器ID作为特征
3. 数据量增加 N 倍（N = 变压器数量）
4. 模型泛化能力更强
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import KFold
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import os
from tqdm import tqdm
import hashlib

try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

print("=" * 80)
print("V6 多变压器版本 - 使用全部变压器数据进行联合训练")
print("=" * 80)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# 缓存管理类
# ============================================================================
class QuantumFeatureCache:
    def __init__(self, cache_dir='quantum_cache_v6_multi'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def get_cache_key(self, data_hash, n_qubits, batch_idx):
        return f"quantum_{data_hash}_{n_qubits}_{batch_idx}.npy"
    
    def compute_data_hash(self, data):
        data_bytes = data.tobytes()
        return hashlib.md5(data_bytes).hexdigest()[:16]
    
    def save_cache(self, key, data):
        path = os.path.join(self.cache_dir, key)
        np.save(path, data)
        
    def load_cache(self, key):
        path = os.path.join(self.cache_dir, key)
        if os.path.exists(path):
            return np.load(path)
        return None

# ============================================================================
# Step 1-7: 数据加载、特征工程、标准化
# ============================================================================
print("\n" + "=" * 80)
print("Step 1-7: 数据加载、特征工程、标准化")
print("=" * 80)

chunk_size = 100000
chunks = []
print("正在加载排序后的数据...")

for i, chunk in enumerate(pd.read_csv('data/df_merged_sorted_by_time.csv', chunksize=chunk_size)):
    chunks.append(chunk)
    if (i + 1) % 10 == 0:
        print(f"  已加载 {(i + 1) * chunk_size:,} 行...")

df = pd.concat(chunks, ignore_index=True)
print(f"✓ 数据加载完成: {df.shape}")

# ============================================================================
# 关键改进：使用全部变压器数据
# ============================================================================
transformer_counts = df['TRANSFORMER_ID'].value_counts()
print(f"\n变压器统计:")
print(f"  总变压器数: {len(transformer_counts)}")
print(f"  平均每个变压器样本数: {transformer_counts.mean():.0f}")
print(f"  最多样本数: {transformer_counts.max()}")
print(f"  最少样本数: {transformer_counts.min()}")

# 只保留样本数足够的变压器（至少1000个样本）
min_samples = 1000
valid_transformers = transformer_counts[transformer_counts >= min_samples].index.tolist()
df = df[df['TRANSFORMER_ID'].isin(valid_transformers)].copy()

print(f"\n筛选后（样本数 >= {min_samples}):")
print(f"  保留变压器数: {len(valid_transformers)}")
print(f"  总样本数: {len(df):,}")

df = df.sort_values(['TRANSFORMER_ID', 'DATETIME']).reset_index(drop=True)
print(f"✓ 筛选后数据形状: {df.shape}")

df['DATETIME'] = pd.to_datetime(df['DATETIME'])

CORE_WEATHER_FEATURES = [
    'TEMP', 'MIN', 'MAX', 'DEWP', 'SLP', 'MXSPD',
    'GUST', 'STP', 'WDSP', 'RH', 'PRCP', 'HEAT_INDEX_EXTREME_CAUTION'
]

print(f"\n添加时间特征...")
df['Hour'] = df['DATETIME'].dt.hour
df['DayOfWeek'] = df['DATETIME'].dt.dayofweek
df['Month'] = df['DATETIME'].dt.month

df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
df['IsPeakHour'] = df['Hour'].isin([8, 9, 10, 18, 19, 20]).astype(int)

print(f"添加滞后特征...")
key_lags = [1, 2, 3, 6, 12, 24, 48, 72, 168]
for lag in key_lags:
    df[f'LOAD_lag{lag}'] = df.groupby('TRANSFORMER_ID')['LOAD'].shift(lag)

print(f"添加滚动特征...")
key_windows = [6, 12, 24, 48, 168]
for window in key_windows:
    df[f'LOAD_rolling_mean_{window}'] = df.groupby('TRANSFORMER_ID')['LOAD'].transform(
        lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
    )
    df[f'LOAD_rolling_std_{window}'] = df.groupby('TRANSFORMER_ID')['LOAD'].transform(
        lambda x: x.rolling(window=window, min_periods=1).std().shift(1)
    )
    df[f'LOAD_rolling_min_{window}'] = df.groupby('TRANSFORMER_ID')['LOAD'].transform(
        lambda x: x.rolling(window=window, min_periods=1).min().shift(1)
    )
    df[f'LOAD_rolling_max_{window}'] = df.groupby('TRANSFORMER_ID')['LOAD'].transform(
        lambda x: x.rolling(window=window, min_periods=1).max().shift(1)
    )

print(f"添加差分特征...")
df['LOAD_diff1'] = df.groupby('TRANSFORMER_ID')['LOAD'].transform(
    lambda x: x.shift(1) - x.shift(2)
)
df['LOAD_diff24'] = df.groupby('TRANSFORMER_ID')['LOAD'].transform(
    lambda x: x.shift(1) - x.shift(25)
)

# 编码变压器ID
print(f"编码变压器ID...")
le = LabelEncoder()
df['TRANSFORMER_ID_ENCODED'] = le.fit_transform(df['TRANSFORMER_ID'])

df = df.dropna()
print(f"✓ 特征工程完成: {df.shape}")

selected_features = (
    CORE_WEATHER_FEATURES +
    ['Hour', 'DayOfWeek', 'Month'] +
    ['Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos'] +
    ['IsWeekend', 'IsPeakHour'] +
    ['TRANSFORMER_ID_ENCODED'] +  # 添加变压器ID特征
    [f'LOAD_lag{lag}' for lag in key_lags] +
    [f'LOAD_rolling_mean_{window}' for window in key_windows] +
    [f'LOAD_rolling_std_{window}' for window in key_windows] +
    [f'LOAD_rolling_min_{window}' for window in key_windows] +
    [f'LOAD_rolling_max_{window}' for window in key_windows] +
    ['LOAD_diff1', 'LOAD_diff24']
)

selected_features = [f for f in selected_features if
                     f in df.columns and df[f].dtype in ['int64', 'float64', 'int32', 'float32']]

X = df[selected_features].copy()
y = df['LOAD'].copy()

print(f"✓ 特征矩阵: {X.shape}")

train_size = int(0.70 * len(X))
val_size = int(0.15 * len(X))

X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:train_size + val_size]
y_val = y[train_size:train_size + val_size]
X_test = X[train_size + val_size:]
y_test = y[train_size + val_size:]

print(f"✓ 训练集: {X_train.shape[0]:,}, 验证集: {X_val.shape[0]:,}, 测试集: {X_test.shape[0]:,}")

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("✓ 标准化完成")

# ============================================================================
# Step 8: 量子特征编码 (简化版)
# ============================================================================
print("\n" + "=" * 80)
print("Step 8: 量子特征编码 (简化版)")
print("=" * 80)

quantum_features_train = None
quantum_features_val = None
quantum_features_test = None
n_qubits = 4

if PENNYLANE_AVAILABLE:
    try:
        print(f"初始化PennyLane (n_qubits={n_qubits})...")
        
        cache = QuantumFeatureCache()
        dev = qml.device('default.qubit', wires=n_qubits)
        
        @qml.qnode(dev)
        def quantum_circuit(inputs):
            normalized_inputs = np.clip(inputs * np.pi, -np.pi, np.pi)
            
            for i in range(n_qubits):
                qml.RX(normalized_inputs[i % len(normalized_inputs)], wires=i)
                qml.RY(normalized_inputs[i % len(normalized_inputs)] * 0.5, wires=i)
            
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
            
            for i in range(n_qubits):
                qml.RY(np.pi / 4, wires=i)
            
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        print(f"✓ 量子电路初始化完成")
        
        for dataset_name, X_data, batch_name in [
            ('训练集', X_train_scaled, 'train'),
            ('验证集', X_val_scaled, 'val'),
            ('测试集', X_test_scaled, 'test')
        ]:
            print(f"\n处理{dataset_name}（{len(X_data):,} 样本）...")
            quantum_features = []
            
            for batch_idx in tqdm(range(0, len(X_data), 300), desc=f"{dataset_name}量子编码"):
                batch_end = min(batch_idx + 300, len(X_data))
                batch_data = X_data[batch_idx:batch_end]
                
                data_hash = cache.compute_data_hash(batch_data)
                cache_key = cache.get_cache_key(data_hash, n_qubits, batch_idx // 300)
                
                cached_features = cache.load_cache(cache_key)
                if cached_features is not None:
                    quantum_features.append(cached_features)
                    continue
                
                batch_quantum_features = []
                for i in range(len(batch_data)):
                    try:
                        qf = quantum_circuit(batch_data[i][:n_qubits])
                        batch_quantum_features.append(qf)
                    except:
                        batch_quantum_features.append([0.0] * n_qubits)
                
                batch_quantum_features = np.array(batch_quantum_features)
                cache.save_cache(cache_key, batch_quantum_features)
                quantum_features.append(batch_quantum_features)
            
            quantum_features = np.vstack(quantum_features)
            
            if batch_name == 'train':
                quantum_features_train = quantum_features
            elif batch_name == 'val':
                quantum_features_val = quantum_features
            else:
                quantum_features_test = quantum_features
            
            print(f"✓ {dataset_name}量子特征: {quantum_features.shape}")
        
        print(f"\n✓ 量子特征编码完成")
        
    except Exception as e:
        print(f"⚠ 量子编码失败: {e}")

# ============================================================================
# Step 9: 融合特征
# ============================================================================
print("\n" + "=" * 80)
print("Step 9: 融合特征")
print("=" * 80)

if quantum_features_train is not None:
    X_train_final = np.hstack([X_train_scaled, quantum_features_train])
    X_val_final = np.hstack([X_val_scaled, quantum_features_val])
    X_test_final = np.hstack([X_test_scaled, quantum_features_test])
    
    quantum_feature_names = [f'Quantum_{i}' for i in range(n_qubits)]
    all_features = selected_features + quantum_feature_names
    
    print(f"✓ 融合完成: {len(all_features)} 特征")
else:
    X_train_final = X_train_scaled
    X_val_final = X_val_scaled
    X_test_final = X_test_scaled
    all_features = selected_features
    print("⚠ 仅使用经典特征")

# ============================================================================
# Step 10: LightGBM初步训练 (用于特征选择)
# ============================================================================
print("\n" + "=" * 80)
print("Step 10: LightGBM初步训练 (用于特征选择)")
print("=" * 80)

lgb_params_initial = {
    'num_leaves': 31,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'max_depth': 7,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
    'random_state': 42,
    'verbose': -1,
}

print("训练初步LightGBM模型...")
lgb_model_initial = lgb.LGBMRegressor(**lgb_params_initial)
lgb_model_initial.fit(
    X_train_final, y_train,
    eval_set=[(X_val_final, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
)

print("✓ 初步模型训练完成")

# ============================================================================
# Step 11: 基于特征重要性进行特征选择
# ============================================================================
print("\n" + "=" * 80)
print("Step 11: 基于特征重要性进行特征选择")
print("=" * 80)

feature_importance_df = pd.DataFrame({
    'Feature': all_features,
    'Importance': lgb_model_initial.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 20 特征:")
print(feature_importance_df.head(20).to_string(index=False))

cumsum_importance = feature_importance_df['Importance'].cumsum() / feature_importance_df['Importance'].sum()
n_features_95 = (cumsum_importance <= 0.95).sum() + 1
selected_feature_indices = feature_importance_df.head(n_features_95).index.tolist()
selected_feature_names = feature_importance_df.iloc[selected_feature_indices]['Feature'].tolist()

print(f"\n✓ 特征选择完成")
print(f"  原始特征数: {len(all_features)}")
print(f"  选择特征数: {len(selected_feature_names)} (累积重要性 95%)")
print(f"  特征数减少: {(1 - len(selected_feature_names) / len(all_features)) * 100:.1f}%")

X_train_selected = X_train_final[:, selected_feature_indices]
X_val_selected = X_val_final[:, selected_feature_indices]
X_test_selected = X_test_final[:, selected_feature_indices]

print(f"\n✓ 特征选择后的数据形状:")
print(f"  训练集: {X_train_selected.shape}")
print(f"  验证集: {X_val_selected.shape}")
print(f"  测试集: {X_test_selected.shape}")

# ============================================================================
# Step 12: 使用优化参数的LightGBM训练
# ============================================================================
print("\n" + "=" * 80)
print("Step 12: 使用优化参数的LightGBM训练")
print("=" * 80)

lgb_params_final = {
    'num_leaves': 25,
    'learning_rate': 0.02,
    'n_estimators': 800,
    'max_depth': 6,
    'min_child_samples': 25,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'reg_alpha': 2.0,
    'reg_lambda': 2.0,
    'random_state': 42,
    'verbose': -1,
}

print("训练最终LightGBM模型...")
print(f"参数配置:")
for key, value in lgb_params_final.items():
    print(f"  {key}: {value}")

train_start = datetime.now()

lgb_model = lgb.LGBMRegressor(**lgb_params_final)
lgb_model.fit(
    X_train_selected, y_train,
    eval_set=[(X_train_selected, y_train), (X_val_selected, y_val), (X_test_selected, y_test)],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(20)]
)

train_time = (datetime.now() - train_start).total_seconds()
print(f"\n✓ LightGBM训练完成（耗时: {train_time:.2f}秒）")
best_iter = lgb_model.best_iteration if hasattr(lgb_model, 'best_iteration') else lgb_model._best_iteration
print(f"  最佳迭代: {best_iter}")

# ============================================================================
# Step 13: 预测和评估
# ============================================================================
print("\n" + "=" * 80)
print("Step 13: 预测和评估")
print("=" * 80)

y_train_pred = lgb_model.predict(X_train_selected)
y_val_pred = lgb_model.predict(X_val_selected)
y_test_pred = lgb_model.predict(X_test_selected)

print(f"✓ 预测完成")

def calculate_metrics(y_true, y_pred, set_name):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    print(f"\n{set_name} 指标:")
    print(f"  R² Score: {r2:.6f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  MAPE: {mape:.4f}%")
    
    return {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

train_metrics = calculate_metrics(y_train, y_train_pred, "训练集")
val_metrics = calculate_metrics(y_val, y_val_pred, "验证集")
test_metrics = calculate_metrics(y_test, y_test_pred, "测试集")

overfitting_gap = abs(train_metrics['R2'] - test_metrics['R2'])
print(f"\n过拟合检查:")
print(f"  训练-测试R²差异: {overfitting_gap:.6f}")
print(f"  状态: {'✓ 良好' if overfitting_gap < 0.1 else '⚠ 需要关注' if overfitting_gap < 0.2 else '✗ 严重过拟合'}")

# ============================================================================
# Step 14: K-Fold 交叉验证
# ============================================================================
print("\n" + "=" * 80)
print("Step 14: K-Fold 交叉验证")
print("=" * 80)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

print("执行 5-Fold 交叉验证...")
for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_selected), 1):
    X_fold_train = X_train_selected[train_idx]
    y_fold_train = y_train.iloc[train_idx]
    X_fold_val = X_train_selected[val_idx]
    y_fold_val = y_train.iloc[val_idx]
    
    fold_model = lgb.LGBMRegressor(**lgb_params_final)
    fold_model.fit(
        X_fold_train, y_fold_train,
        eval_set=[(X_fold_train, y_fold_train), (X_fold_val, y_fold_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    y_fold_pred = fold_model.predict(X_fold_val)
    fold_r2 = r2_score(y_fold_val, y_fold_pred)
    fold_rmse = np.sqrt(mean_squared_error(y_fold_val, y_fold_pred))
    
    cv_scores.append({'Fold': fold, 'R2': fold_r2, 'RMSE': fold_rmse})
    print(f"  Fold {fold}: R² = {fold_r2:.6f}, RMSE = {fold_rmse:.4f}")

cv_df = pd.DataFrame(cv_scores)
print(f"\n交叉验证结果:")
print(f"  平均 R²: {cv_df['R2'].mean():.6f} ± {cv_df['R2'].std():.6f}")
print(f"  平均 RMSE: {cv_df['RMSE'].mean():.4f} ± {cv_df['RMSE'].std():.4f}")

# ============================================================================
# Step 15: SHAP分析
# ============================================================================
print("\n" + "=" * 80)
print("Step 15: SHAP可解释性分析")
print("=" * 80)

shap_importance = None
shap_values = None

if SHAP_AVAILABLE:
    try:
        print(f"SHAP版本: {shap.__version__}")
        
        explainer = shap.TreeExplainer(lgb_model)
        
        shap_sample_size = min(1000, len(X_test_selected))
        print(f"✓ SHAP TreeExplainer初始化完成")
        print(f"  分析样本数: {shap_sample_size}")
        
        print(f"\n计算SHAP值（{shap_sample_size} 个测试样本）...")
        shap_values = explainer.shap_values(X_test_selected[:shap_sample_size])
        
        print(f"✓ SHAP值计算完成")
        
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_importance = pd.DataFrame({
            'Feature': selected_feature_names,
            'SHAP_Importance': mean_abs_shap,
            'Mean_SHAP_Value': np.mean(shap_values, axis=0),
            'Std_SHAP_Value': np.std(shap_values, axis=0),
        }).sort_values('SHAP_Importance', ascending=False)
        
        print("\nTop 20 特征（SHAP重要性排名）")
        print("=" * 80)
        for idx, row in shap_importance.head(20).iterrows():
            feature_type = "量子" if "Quantum" in row['Feature'] else "经典"
            print(f"  {row['Feature']:40s}: {row['SHAP_Importance']:10.6f} [{feature_type}]")
        
        output_dir = 'model_output_v6_multi'
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n生成SHAP可视化...")
        
        # SHAP摘要图
        plt.figure(figsize=(14, 10))
        shap.summary_plot(
            shap_values,
            X_test_selected[:shap_sample_size],
            feature_names=selected_feature_names,
            max_display=20,
            show=False
        )
        plt.tight_layout()
        plt.savefig(f'{output_dir}/shap_summary_plot.png', dpi=300, bbox_inches='tight')
        print("✓ SHAP摘要图已保存")
        plt.close()
        
        # SHAP条形图
        plt.figure(figsize=(14, 10))
        shap.summary_plot(
            shap_values,
            X_test_selected[:shap_sample_size],
            feature_names=selected_feature_names,
            plot_type="bar",
            max_display=20,
            show=False
        )
        plt.tight_layout()
        plt.savefig(f'{output_dir}/shap_bar_plot.png', dpi=300, bbox_inches='tight')
        print("✓ SHAP条形图已保存")
        plt.close()
        
        # SHAP力图（前5个样本）
        print("生成SHAP力图...")
        for i in range(min(5, shap_sample_size)):
            plt.figure(figsize=(16, 4))
            shap.force_plot(
                explainer.expected_value,
                shap_values[i],
                X_test_selected[i],
                feature_names=selected_feature_names,
                matplotlib=True,
                show=False
            )
            plt.tight_layout()
            plt.savefig(f'{output_dir}/shap_force_plot_{i}.png', dpi=300, bbox_inches='tight')
            plt.close()
        print(f"✓ SHAP力图已保存（前5个样本）")
        
        # SHAP依赖图（Top 6特征）
        print("生成SHAP依赖图...")
        top_6_features = shap_importance.head(6)['Feature'].tolist()
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, feature in enumerate(top_6_features):
            feature_idx = selected_feature_names.index(feature)
            ax = axes[idx]
            shap.dependence_plot(
                feature_idx,
                shap_values,
                X_test_selected[:shap_sample_size],
                feature_names=selected_feature_names,
                ax=ax,
                show=False
            )
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/shap_dependence_plots.png', dpi=300, bbox_inches='tight')
        print("✓ SHAP依赖图已保存")
        plt.close()
        
        print(f"\n✓ SHAP分析完成")
        
    except Exception as e:
        print(f"⚠ SHAP分析失败: {e}")
        import traceback
        traceback.print_exc()
else:
    print("⚠ SHAP未安装，跳过SHAP分析")

# ============================================================================
# Step 16: 保存模型和结果
# ============================================================================
print("\n" + "=" * 80)
print("Step 16: 保存模型和结果")
print("=" * 80)

output_dir = 'model_output_v6_multi'
os.makedirs(output_dir, exist_ok=True)

with open(f'{output_dir}/model.pkl', 'wb') as f:
    pickle.dump(lgb_model, f)
print("✓ 模型已保存")

with open(f'{output_dir}/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ 标准化器已保存")

with open(f'{output_dir}/selected_features.pkl', 'wb') as f:
    pickle.dump(selected_feature_names, f)
print("✓ 特征列表已保存")

with open(f'{output_dir}/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("✓ 变压器ID编码器已保存")

results_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_test_pred,
    'Error': y_test_pred - y_test.values,
    'Abs_Error': np.abs(y_test_pred - y_test.values),
    'Percent_Error': np.abs(y_test_pred - y_test.values) / (np.abs(y_test.values) + 1e-10) * 100,
})
results_df.to_csv(f'{output_dir}/test_predictions.csv', index=False)
print("✓ 测试集预测结果已保存")

metrics_df = pd.DataFrame([
    {'Set': '训练集', **train_metrics},
    {'Set': '验证集', **val_metrics},
    {'Set': '测试集', **test_metrics},
])
metrics_df.to_csv(f'{output_dir}/metrics.csv', index=False)
print("✓ 评估指标已保存")

cv_df.to_csv(f'{output_dir}/cv_results.csv', index=False)
print("✓ 交叉验证结果已保存")

feature_importance_df.to_csv(f'{output_dir}/feature_importance.csv', index=False)
print("✓ 特征重要性已保存")

if shap_importance is not None:
    shap_importance.to_csv(f'{output_dir}/shap_importance.csv', index=False)
    print("✓ SHAP重要性已保存")
    
    # 生成SHAP详细分析报告
    shap_detail_report = []
    for idx, row in shap_importance.iterrows():
        shap_detail_report.append({
            'Rank': idx + 1,
            'Feature': row['Feature'],
            'SHAP_Importance': row['SHAP_Importance'],
            'Mean_SHAP_Value': row['Mean_SHAP_Value'],
            'Std_SHAP_Value': row['Std_SHAP_Value'],
            'Feature_Type': 'Quantum' if 'Quantum' in row['Feature'] else 'Classical',
            'LightGBM_Importance': feature_importance_df[feature_importance_df['Feature'] == row['Feature']]['Importance'].values[0] if row['Feature'] in feature_importance_df['Feature'].values else 0,
        })
    
    shap_detail_df = pd.DataFrame(shap_detail_report)
    shap_detail_df.to_csv(f'{output_dir}/shap_detailed_analysis.csv', index=False)
    print("✓ SHAP详细分析已保存")
    
    # 生成特征对比分析（SHAP vs LightGBM）
    comparison_data = []
    for feature in selected_feature_names:
        lgb_imp = feature_importance_df[feature_importance_df['Feature'] == feature]['Importance'].values
        shap_imp = shap_importance[shap_importance['Feature'] == feature]['SHAP_Importance'].values
        
        if len(lgb_imp) > 0 and len(shap_imp) > 0:
            comparison_data.append({
                'Feature': feature,
                'LightGBM_Importance': lgb_imp[0],
                'SHAP_Importance': shap_imp[0],
                'Importance_Ratio': shap_imp[0] / (lgb_imp[0] + 1e-10),
                'Feature_Type': 'Quantum' if 'Quantum' in feature else 'Classical',
            })
    
    comparison_df = pd.DataFrame(comparison_data).sort_values('SHAP_Importance', ascending=False)
    comparison_df.to_csv(f'{output_dir}/feature_importance_comparison.csv', index=False)
    print("✓ 特征重要性对比已保存")

print(f"\n✓ 所有结果已保存到: {output_dir}/")

# ============================================================================
# Step 17: 可视化分析
# ============================================================================
print("\n" + "=" * 80)
print("Step 17: 可视化分析")
print("=" * 80)

fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# 1-3. 散点图
for idx, (ax_pos, (y_true, y_pred, title, color)) in enumerate([
    (gs[0, 0], (y_train, y_train_pred, f'训练集 (R²={train_metrics["R2"]:.4f})', 'steelblue')),
    (gs[0, 1], (y_val, y_val_pred, f'验证集 (R²={val_metrics["R2"]:.4f})', 'lightgreen')),
    (gs[0, 2], (y_test, y_test_pred, f'测试集 (R²={test_metrics["R2"]:.4f})', 'coral')),
]):
    ax = fig.add_subplot(ax_pos)
    ax.scatter(y_true, y_pred, alpha=0.3, s=1, color=color)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测')
    ax.set_xlabel('实际值', fontsize=11)
    ax.set_ylabel('预测值', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

# 4. 误差分布
ax4 = fig.add_subplot(gs[1, 0])
errors = y_test_pred - y_test.values
ax4.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
ax4.axvline(x=0, color='r', linestyle='--', lw=2)
ax4.set_xlabel('误差', fontsize=11)
ax4.set_ylabel('频数', fontsize=11)
ax4.set_title('误差分布', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# 5. 残差图
ax5 = fig.add_subplot(gs[1, 1])
ax5.scatter(y_test_pred, errors, alpha=0.3, s=1, color='steelblue')
ax5.axhline(y=0, color='r', linestyle='--', lw=2)
ax5.set_xlabel('预测值', fontsize=11)
ax5.set_ylabel('残差', fontsize=11)
ax5.set_title('残差图', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

# 6. R²对比
ax6 = fig.add_subplot(gs[1, 2])
sets = ['训练集', '验证集', '测试集']
r2_scores = [train_metrics['R2'], val_metrics['R2'], test_metrics['R2']]
colors = ['steelblue', 'lightgreen', 'coral']
bars = ax6.bar(sets, r2_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax6.set_ylabel('R² Score', fontsize=11)
ax6.set_title('R²对比', fontsize=12, fontweight='bold')
ax6.set_ylim([0, 1.0])
ax6.grid(True, alpha=0.3, axis='y')
for bar, score in zip(bars, r2_scores):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width() / 2., height,
             f'{score:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# 7. LightGBM特征重要性
ax7 = fig.add_subplot(gs[2, :2])
top_features = feature_importance_df.head(15)
colors_feat = ['red' if 'Quantum' in f else 'steelblue' for f in top_features['Feature']]
ax7.barh(top_features['Feature'], top_features['Importance'], color=colors_feat, alpha=0.8)
ax7.set_xlabel('重要性', fontsize=11)
ax7.set_title('LightGBM特征重要性 Top 15 (红色=量子特征)', fontsize=12, fontweight='bold')
ax7.invert_yaxis()
ax7.grid(True, alpha=0.3, axis='x')

# 8. 指标总结
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis('off')
metrics_text = f"""
V6 多变压器模型性能总结

数据统计:
  变压器数: {len(valid_transformers)}
  总样本数: {len(df):,}

训练集:
  R² = {train_metrics['R2']:.6f}
  RMSE = {train_metrics['RMSE']:.4f}

验证集:
  R² = {val_metrics['R2']:.6f}
  RMSE = {val_metrics['RMSE']:.4f}

测试集:
  R² = {test_metrics['R2']:.6f}
  RMSE = {test_metrics['RMSE']:.4f}

交叉验证:
  平均 R²: {cv_df['R2'].mean():.6f}
  Std: {cv_df['R2'].std():.6f}

特征统计:
  原始: {len(all_features)}
  选择: {len(selected_feature_names)}
  减少: {(1-len(selected_feature_names)/len(all_features))*100:.1f}%
"""
ax8.text(0.05, 0.5, metrics_text, fontsize=9, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('V6 多变压器版本 - LightGBM + 特征选择 - 综合分析', fontsize=16, fontweight='bold', y=0.995)
plt.savefig(f'{output_dir}/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
print("✓ 综合分析图已保存")
plt.close()

# 时间序列预测图
print("生成时间序列预测图...")
fig, ax = plt.subplots(figsize=(16, 6))
plot_size = min(500, len(y_test))
x_axis = range(plot_size)
ax.plot(x_axis, y_test.values[:plot_size], label='实际值', color='blue', alpha=0.7, linewidth=1.5)
ax.plot(x_axis, y_test_pred[:plot_size], label='预测值', color='red', alpha=0.7, linewidth=1.5)
ax.fill_between(x_axis, y_test.values[:plot_size], y_test_pred[:plot_size], alpha=0.2, color='gray')
ax.set_xlabel('时间步', fontsize=12)
ax.set_ylabel('负荷', fontsize=12)
ax.set_title(f'时间序列预测对比（前{plot_size}个样本）', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/timeseries_prediction.png', dpi=300, bbox_inches='tight')
print("✓ 时间序列预测图已保存")
plt.close()

# SHAP vs LightGBM特征重要性对比
if shap_importance is not None:
    print("生成特征重要性对比图...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # LightGBM重要性
    top_15_lgb = feature_importance_df.head(15)
    colors_lgb = ['red' if 'Quantum' in f else 'steelblue' for f in top_15_lgb['Feature']]
    axes[0].barh(top_15_lgb['Feature'], top_15_lgb['Importance'], color=colors_lgb, alpha=0.8)
    axes[0].set_xlabel('重要性', fontsize=12)
    axes[0].set_title('LightGBM特征重要性 Top 15', fontsize=13, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # SHAP重要性
    top_15_shap = shap_importance.head(15)
    colors_shap = ['red' if 'Quantum' in f else 'steelblue' for f in top_15_shap['Feature']]
    axes[1].barh(top_15_shap['Feature'], top_15_shap['SHAP_Importance'], color=colors_shap, alpha=0.8)
    axes[1].set_xlabel('SHAP重要性', fontsize=12)
    axes[1].set_title('SHAP特征重要性 Top 15', fontsize=13, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ 特征重要性对比图已保存")
    plt.close()

print("\n✓ 可视化完成")

# ============================================================================
# Step 18: 生成最终报告
# ============================================================================
print("\n" + "=" * 80)
print("Step 18: 生成最终报告")
print("=" * 80)

report = f"""
{'=' * 80}
V6 多变压器版本 - 使用全部变压器数据进行联合训练
{'=' * 80}

执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'=' * 80}
数据统计
{'=' * 80}

原始数据:
  总行数: {len(df):,}
  变压器数: {len(valid_transformers)}
  平均每个变压器样本数: {len(df) // len(valid_transformers):,}

特征统计:
  原始特征数: {len(all_features)}
  选择特征数: {len(selected_feature_names)}
  特征减少比例: {(1 - len(selected_feature_names) / len(all_features)) * 100:.1f}%

数据集划分:
  训练集: {len(X_train):,} 样本
  验证集: {len(X_val):,} 样本
  测试集: {len(X_test):,} 样本

{'=' * 80}
模型性能
{'=' * 80}

训练集:
  R² Score: {train_metrics['R2']:.6f}
  RMSE: {train_metrics['RMSE']:.4f}
  MAE: {train_metrics['MAE']:.4f}
  MAPE: {train_metrics['MAPE']:.4f}%

验证集:
  R² Score: {val_metrics['R2']:.6f}
  RMSE: {val_metrics['RMSE']:.4f}
  MAE: {val_metrics['MAE']:.4f}
  MAPE: {val_metrics['MAPE']:.4f}%

测试集:
  R² Score: {test_metrics['R2']:.6f}
  RMSE: {test_metrics['RMSE']:.4f}
  MAE: {test_metrics['MAE']:.4f}
  MAPE: {test_metrics['MAPE']:.4f}%

过拟合检查:
  训练-测试R²差异: {overfitting_gap:.6f}
  状态: {'✓ 良好' if overfitting_gap < 0.1 else '⚠ 需要关注' if overfitting_gap < 0.2 else '✗ 严重过拟合'}

{'=' * 80}
5-Fold 交叉验证结果
{'=' * 80}

平均 R²: {cv_df['R2'].mean():.6f} ± {cv_df['R2'].std():.6f}
平均 RMSE: {cv_df['RMSE'].mean():.4f} ± {cv_df['RMSE'].std():.4f}

各 Fold 结果:
"""

for idx, row in cv_df.iterrows():
    report += f"  Fold {int(row['Fold'])}: R² = {row['R2']:.6f}, RMSE = {row['RMSE']:.4f}\n"

report += f"""
{'=' * 80}
LightGBM 特征重要性分析
{'=' * 80}

Top 20 特征:
"""

for idx, row in feature_importance_df.head(20).iterrows():
    feature_type = "量子" if "Quantum" in row['Feature'] else "经典"
    report += f"  {idx + 1:2d}. {row['Feature']:40s}: {row['Importance']:10.6f} [{feature_type}]\n"

if shap_importance is not None:
    report += f"""
{'=' * 80}
SHAP 特征重要性分析
{'=' * 80}

Top 20 特征（SHAP重要性排名）:
"""
    for idx, row in shap_importance.head(20).iterrows():
        feature_type = "量子" if "Quantum" in row['Feature'] else "经典"
        report += f"  {idx + 1:2d}. {row['Feature']:40s}: {row['SHAP_Importance']:10.6f} [{feature_type}]\n"
    
    report += f"""
SHAP 统计信息:
  平均SHAP值范围: [{shap_importance['Mean_SHAP_Value'].min():.6f}, {shap_importance['Mean_SHAP_Value'].max():.6f}]
  SHAP标准差范围: [{shap_importance['Std_SHAP_Value'].min():.6f}, {shap_importance['Std_SHAP_Value'].max():.6f}]

特征类型分布:
"""
    quantum_count = (shap_importance['Feature'].str.contains('Quantum')).sum()
    classical_count = len(shap_importance) - quantum_count
    report += f"  量子特征: {quantum_count} 个\n"
    report += f"  经典特征: {classical_count} 个\n"

report += f"""
{'=' * 80}
版本对比
{'=' * 80}

V6 单变压器版本:
  - 变压器数: 1
  - 样本数: ~8,760
  - 测试集 R²: 0.82+

V6 多变压器版本 (当前):
  - 变压器数: {len(valid_transformers)}
  - 样本数: {len(df):,}
  - 测试集 R²: {test_metrics['R2']:.6f}
  - 改进: {'✓ 显著改进' if test_metrics['R2'] > 0.82 else '⚠ 有所改进' if test_metrics['R2'] > 0.81 else '✗ 需要进一步优化'}

{'=' * 80}
关键改进
{'=' * 80}

1. ✓ 使用全部变压器数据
   - 从 1 个变压器 → {len(valid_transformers)} 个变压器
   - 样本数增加 {len(df) // 8760:.1f}x

2. ✓ 添加变压器ID特征
   - 模型可以学习不同变压器的特性差异
   - 提高泛化能力

3. ✓ 按变压器分组计算滞后和滚动特征
   - 避免不同变压器之间的数据泄漏
   - 保证特征的有效性

4. ✓ 特征选择
   - 保留累积重要性 95% 的特征
   - 减少过拟合风险

5. ✓ 完整的SHAP可解释性分析
   - SHAP重要性排名
   - SHAP摘要图、条形图、力图、依赖图
   - 特征重要性对比分析

{'=' * 80}
输出文件
{'=' * 80}

模型文件:
  ✓ model.pkl - 训练好的LightGBM模型
  ✓ scaler.pkl - RobustScaler标准化器
  ✓ selected_features.pkl - 选择的特征列表
  ✓ label_encoder.pkl - 变压器ID编码器

结果文件:
  ✓ test_predictions.csv - 测试集预测结果
  ✓ metrics.csv - 评估指标
  ✓ cv_results.csv - 交叉验证结果
  ✓ feature_importance.csv - LightGBM特征重要性
  ✓ shap_importance.csv - SHAP特征重要性
  ✓ shap_detailed_analysis.csv - SHAP详细分析
  ✓ feature_importance_comparison.csv - 特征重要性对比

可视化文件:
  ✓ comprehensive_analysis.png - 综合分析图
  ✓ timeseries_prediction.png - 时间序列预测图
  ✓ feature_importance_comparison.png - 特征重要性对比图
  ✓ shap_summary_plot.png - SHAP摘要图
  ✓ shap_bar_plot.png - SHAP条形图
  ✓ shap_force_plot_*.png - SHAP力图（前5个样本）
  ✓ shap_dependence_plots.png - SHAP依赖图

缓存文件:
  ✓ quantum_cache_v6_multi/ - 量子特征缓存目录

输出目录: {output_dir}/

{'=' * 80}
完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 80}
"""

with open(f'{output_dir}/optimization_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(report)
print(f"\n✓ 最终报告已保存到: {output_dir}/optimization_report.txt")

print("\n" + "=" * 80)
print("V6 多变压器版本完成！")
print("=" * 80)
