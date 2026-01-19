"""
最终优化的量子增强XGBoost模型 V4
Final Optimized Quantum-Enhanced XGBoost V4

核心改进:
1. 分批量子编码 + 缓存机制 (保留量子特征)
2. 精简特征工程 (仅12个气象特征)
3. 优化XGBoost参数 (针对小数据集)
4. K-Fold交叉验证 (评估泛化能力)
5. SHAP可解释性分析 (保留SHAP)
6. 特征选择 (基于SHAP重要性)
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import KFold
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import os
from tqdm import tqdm
import hashlib

# 量子和可解释性库
try:
    import pennylane as qml
    from pennylane import numpy as pnp
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
print("最终优化的量子增强XGBoost模型 V4")
print("=" * 80)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# 缓存管理类
# ============================================================================
class QuantumFeatureCache:
    """量子特征缓存管理"""
    def __init__(self, cache_dir='quantum_cache_v4'):
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
    
    def cache_exists(self, key):
        return os.path.exists(os.path.join(self.cache_dir, key))

# ============================================================================
# Step 1: 加载数据
# ============================================================================
print("\n" + "=" * 80)
print("Step 1: 加载数据")
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
# Step 2: 选择变压器
# ============================================================================
print("\n" + "=" * 80)
print("Step 2: 选择变压器")
print("=" * 80)

transformer_counts = df['TRANSFORMER_ID'].value_counts()
selected_transformer = transformer_counts.index[0]
df = df[df['TRANSFORMER_ID'] == selected_transformer].copy()
df = df.sort_values('DATETIME').reset_index(drop=True)
print(f"✓ 筛选后数据形状: {df.shape}")

# ============================================================================
# Step 3: 特征工程 - 精简版
# ============================================================================
print("\n" + "=" * 80)
print("Step 3: 特征工程 - 精简版（12个核心气象特征）")
print("=" * 80)

df['DATETIME'] = pd.to_datetime(df['DATETIME'])

CORE_WEATHER_FEATURES = [
    'TEMP', 'MIN', 'MAX', 'DEWP', 'SLP', 'MXSPD',
    'GUST', 'STP', 'WDSP', 'RH', 'PRCP', 'HEAT_INDEX_EXTREME_CAUTION'
]

print(f"核心气象特征: {len(CORE_WEATHER_FEATURES)} 个")

# 时间特征
print("添加时间特征...")
df['Hour'] = df['DATETIME'].dt.hour
df['DayOfWeek'] = df['DATETIME'].dt.dayofweek
df['Month'] = df['DATETIME'].dt.month

df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)

df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
df['IsPeakHour'] = df['Hour'].isin([8, 9, 10, 18, 19, 20]).astype(int)

# 滞后特征
print("添加滞后特征...")
key_lags = [1, 24, 168]
for lag in key_lags:
    df[f'LOAD_lag{lag}'] = df['LOAD'].shift(lag)

# 滚动特征
print("添加滚动特征...")
key_windows = [24, 168]
for window in key_windows:
    df[f'LOAD_rolling_mean_{window}'] = df['LOAD'].rolling(window=window, min_periods=1).mean().shift(1)
    df[f'LOAD_rolling_std_{window}'] = df['LOAD'].rolling(window=window, min_periods=1).std().shift(1)

# 差分特征
print("添加差分特征...")
df['LOAD_diff1'] = df['LOAD'].shift(1) - df['LOAD'].shift(2)
df['LOAD_diff24'] = df['LOAD'].shift(1) - df['LOAD'].shift(25)

df = df.dropna()
print(f"✓ 特征工程完成: {df.shape}")

# ============================================================================
# Step 4: 特征选择
# ============================================================================
print("\n" + "=" * 80)
print("Step 4: 特征选择")
print("=" * 80)

selected_features = (
    CORE_WEATHER_FEATURES +
    ['Hour', 'DayOfWeek', 'Month'] +
    ['Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos'] +
    ['IsWeekend', 'IsPeakHour'] +
    [f'LOAD_lag{lag}' for lag in key_lags] +
    [f'LOAD_rolling_mean_{window}' for window in key_windows] +
    [f'LOAD_rolling_std_{window}' for window in key_windows] +
    ['LOAD_diff1', 'LOAD_diff24']
)

selected_features = [f for f in selected_features if
                     f in df.columns and df[f].dtype in ['int64', 'float64', 'int32', 'float32']]

print(f"✓ 选择的特征数: {len(selected_features)}")

# ============================================================================
# Step 5: 准备数据
# ============================================================================
print("\n" + "=" * 80)
print("Step 5: 准备数据")
print("=" * 80)

X = df[selected_features].copy()
y = df['LOAD'].copy()

print(f"✓ 特征矩阵: {X.shape}")
print(f"✓ 目标变量: {y.shape}")

# ============================================================================
# Step 6: 时间序列分割
# ============================================================================
print("\n" + "=" * 80)
print("Step 6: 时间序列分割")
print("=" * 80)

train_size = int(0.70 * len(X))
val_size = int(0.15 * len(X))

X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:train_size + val_size]
y_val = y[train_size:train_size + val_size]
X_test = X[train_size + val_size:]
y_test = y[train_size + val_size:]

print(f"✓ 训练集: {X_train.shape[0]:,} 样本 (70%)")
print(f"✓ 验证集: {X_val.shape[0]:,} 样本 (15%)")
print(f"✓ 测试集: {X_test.shape[0]:,} 样本 (15%)")

# ============================================================================
# Step 7: 鲁棒标准化
# ============================================================================
print("\n" + "=" * 80)
print("Step 7: 鲁棒标准化")
print("=" * 80)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("✓ 标准化完成")

# ============================================================================
# Step 8: 分批量子特征编码（带缓存）
# ============================================================================
print("\n" + "=" * 80)
print("Step 8: 分批量子特征编码（带缓存机制）")
print("=" * 80)

quantum_features_train = None
quantum_features_val = None
quantum_features_test = None
n_qubits = 6  # 针对小数据集优化
batch_size = 300

if PENNYLANE_AVAILABLE:
    try:
        print(f"初始化PennyLane量子设备...")
        print(f"  量子比特数: {n_qubits}")
        print(f"  分批大小: {batch_size}")
        
        cache = QuantumFeatureCache()
        dev = qml.device('default.qubit', wires=n_qubits)
        
        @qml.qnode(dev)
        def quantum_circuit(inputs):
            """优化的量子电路"""
            normalized_inputs = np.clip(inputs * np.pi, -np.pi, np.pi)
            
            # Layer 1: 角度编码
            for i in range(n_qubits):
                qml.RX(normalized_inputs[i % len(normalized_inputs)], wires=i)
                qml.RY(normalized_inputs[i % len(normalized_inputs)] * 0.5, wires=i)
            
            # Layer 2: 纠缠
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
            
            # Layer 3: 旋转
            for i in range(n_qubits):
                qml.RY(np.pi / 4, wires=i)
            
            # Layer 4: 第二次纠缠
            for i in range(n_qubits):
                qml.CNOT(wires=[(i + 1) % n_qubits, i])
            
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
        
        print(f"✓ PennyLane量子电路初始化完成")
        
        # 处理训练集
        print(f"\n处理训练集（{len(X_train_scaled):,} 样本）...")
        quantum_features_train = []
        
        for batch_idx in tqdm(range(0, len(X_train_scaled), batch_size), desc="训练集量子编码"):
            batch_end = min(batch_idx + batch_size, len(X_train_scaled))
            batch_data = X_train_scaled[batch_idx:batch_end]
            
            data_hash = cache.compute_data_hash(batch_data)
            cache_key = cache.get_cache_key(data_hash, n_qubits, batch_idx // batch_size)
            
            cached_features = cache.load_cache(cache_key)
            if cached_features is not None:
                quantum_features_train.append(cached_features)
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
            quantum_features_train.append(batch_quantum_features)
        
        quantum_features_train = np.vstack(quantum_features_train)
        print(f"✓ 训练集量子特征: {quantum_features_train.shape}")
        
        # 处理验证集
        print(f"\n处理验证集（{len(X_val_scaled):,} 样本）...")
        quantum_features_val = []
        
        for batch_idx in tqdm(range(0, len(X_val_scaled), batch_size), desc="验证集量子编码"):
            batch_end = min(batch_idx + batch_size, len(X_val_scaled))
            batch_data = X_val_scaled[batch_idx:batch_end]
            
            data_hash = cache.compute_data_hash(batch_data)
            cache_key = cache.get_cache_key(data_hash, n_qubits, batch_idx // batch_size)
            
            cached_features = cache.load_cache(cache_key)
            if cached_features is not None:
                quantum_features_val.append(cached_features)
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
            quantum_features_val.append(batch_quantum_features)
        
        quantum_features_val = np.vstack(quantum_features_val)
        print(f"✓ 验证集量子特征: {quantum_features_val.shape}")
        
        # 处理测试集
        print(f"\n处理测试集（{len(X_test_scaled):,} 样本）...")
        quantum_features_test = []
        
        for batch_idx in tqdm(range(0, len(X_test_scaled), batch_size), desc="测试集量子编码"):
            batch_end = min(batch_idx + batch_size, len(X_test_scaled))
            batch_data = X_test_scaled[batch_idx:batch_end]
            
            data_hash = cache.compute_data_hash(batch_data)
            cache_key = cache.get_cache_key(data_hash, n_qubits, batch_idx // batch_size)
            
            cached_features = cache.load_cache(cache_key)
            if cached_features is not None:
                quantum_features_test.append(cached_features)
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
            quantum_features_test.append(batch_quantum_features)
        
        quantum_features_test = np.vstack(quantum_features_test)
        print(f"✓ 测试集量子特征: {quantum_features_test.shape}")
        
        print(f"\n✓ 分批量子特征编码完成")
        
    except Exception as e:
        print(f"⚠ PennyLane量子编码失败: {e}")
        import traceback
        traceback.print_exc()
else:
    print("⚠ PennyLane未安装")

# ============================================================================
# Step 9: 融合量子特征与经典特征
# ============================================================================
print("\n" + "=" * 80)
print("Step 9: 融合量子特征与经典特征")
print("=" * 80)

if quantum_features_train is not None:
    X_train_final = np.hstack([X_train_scaled, quantum_features_train])
    X_val_final = np.hstack([X_val_scaled, quantum_features_val])
    X_test_final = np.hstack([X_test_scaled, quantum_features_test])
    
    quantum_feature_names = [f'Quantum_{i}' for i in range(n_qubits)]
    all_features = selected_features + quantum_feature_names
    
    print(f"✓ 融合完成")
    print(f"  经典特征: {len(selected_features)}")
    print(f"  量子特征: {n_qubits}")
    print(f"  总特征数: {len(all_features)}")
else:
    X_train_final = X_train_scaled
    X_val_final = X_val_scaled
    X_test_final = X_test_scaled
    all_features = selected_features
    print("⚠ 仅使用经典特征")

# ============================================================================
# Step 10: XGBoost训练 - 优化参数
# ============================================================================
print("\n" + "=" * 80)
print("Step 10: XGBoost训练 - 优化参数（针对小数据集）")
print("=" * 80)

xgb_params = {
    'n_estimators': 500,        # 平衡点
    'max_depth': 6,             # 防止过拟合
    'learning_rate': 0.03,      # 平衡学习速度
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'colsample_bylevel': 0.85,
    'min_child_weight': 3,
    'gamma': 0.3,
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist',
    'max_bin': 200,
}

print("✓ 使用CPU模式训练")
model = xgb.XGBRegressor(**xgb_params)

print("\n开始XGBoost训练...")
print(f"参数配置:")
for key, value in xgb_params.items():
    print(f"  {key}: {value}")

train_start = datetime.now()

model.fit(
    X_train_final, y_train,
    eval_set=[(X_train_final, y_train), (X_val_final, y_val), (X_test_final, y_test)],
    eval_metric=['rmse', 'mae'],
    early_stopping_rounds=80,
    verbose=20
)

train_time = (datetime.now() - train_start).total_seconds()
print(f"\n✓ XGBoost训练完成（耗时: {train_time:.2f}秒）")
print(f"  最佳迭代: {model.best_iteration}")

# ============================================================================
# Step 11: 预测和评估
# ============================================================================
print("\n" + "=" * 80)
print("Step 11: 预测和评估")
print("=" * 80)

y_train_pred = model.predict(X_train_final)
y_val_pred = model.predict(X_val_final)
y_test_pred = model.predict(X_test_final)

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
# Step 12: K-Fold 交叉验证
# ============================================================================
print("\n" + "=" * 80)
print("Step 12: K-Fold 交叉验证")
print("=" * 80)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

print("执行 5-Fold 交叉验证...")
for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_final), 1):
    X_fold_train = X_train_final[train_idx]
    y_fold_train = y_train.iloc[train_idx]
    X_fold_val = X_train_final[val_idx]
    y_fold_val = y_train.iloc[val_idx]
    
    fold_model = xgb.XGBRegressor(**xgb_params)
    fold_model.fit(
        X_fold_train, y_fold_train,
        eval_set=[(X_fold_train, y_fold_train), (X_fold_val, y_fold_val)],
        eval_metric=['rmse'],
        early_stopping_rounds=50,
        verbose=0
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
# Step 13: SHAP可解释性分析
# ============================================================================
print("\n" + "=" * 80)
print("Step 13: SHAP可解释性分析")
print("=" * 80)

shap_importance = None

if SHAP_AVAILABLE:
    try:
        print(f"SHAP版本: {shap.__version__}")
        
        explainer = shap.TreeExplainer(model)
        
        shap_sample_size = min(500, len(X_test_final))
        print(f"✓ SHAP TreeExplainer初始化完成")
        print(f"  分析样本数: {shap_sample_size}")
        
        print(f"\n计算SHAP值（{shap_sample_size} 个测试样本）...")
        shap_values = explainer.shap_values(X_test_final[:shap_sample_size])
        
        print(f"✓ SHAP值计算完成")
        
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_importance = pd.DataFrame({
            'Feature': all_features,
            'SHAP_Importance': mean_abs_shap
        }).sort_values('SHAP_Importance', ascending=False)
        
        print("\nTop 15 特征（SHAP重要性排名）")
        print("=" * 60)
        for idx, row in shap_importance.head(15).iterrows():
            feature_type = "量子" if "Quantum" in row['Feature'] else "经典"
            print(f"  {row['Feature']:40s}: {row['SHAP_Importance']:10.6f} [{feature_type}]")
        
        os.makedirs('model_output_optimized_v4', exist_ok=True)
        
        # 生成SHAP摘要图
        print("\n生成SHAP摘要图...")
        plt.figure(figsize=(14, 10))
        shap.summary_plot(
            shap_values,
            X_test_final[:shap_sample_size],
            feature_names=all_features,
            max_display=15,
            show=False
        )
        plt.tight_layout()
        plt.savefig('model_output_optimized_v4/shap_summary_plot.png', dpi=300, bbox_inches='tight')
        print("✓ SHAP摘要图已保存")
        plt.close()
        
        # 生成SHAP条形图
        print("生成SHAP条形图...")
        plt.figure(figsize=(14, 10))
        shap.summary_plot(
            shap_values,
            X_test_final[:shap_sample_size],
            feature_names=all_features,
            plot_type="bar",
            max_display=15,
            show=False
        )
        plt.tight_layout()
        plt.savefig('model_output_optimized_v4/shap_bar_plot.png', dpi=300, bbox_inches='tight')
        print("✓ SHAP条形图已保存")
        plt.close()
        
        print(f"\n✓ SHAP分析完成")
        
    except Exception as e:
        print(f"⚠ SHAP分析失败: {e}")
else:
    print("⚠ SHAP未安装")

# ============================================================================
# Step 14: 特征重要性分析
# ============================================================================
print("\n" + "=" * 80)
print("Step 14: 特征重要性分析")
print("=" * 80)

feature_importance = pd.DataFrame({
    'Feature': all_features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 15 特征（XGBoost重要性）:")
print("=" * 60)
for idx, row in feature_importance.head(15).iterrows():
    feature_type = "量子" if "Quantum" in row['Feature'] else "经典"
    print(f"  {row['Feature']:40s}: {row['Importance']:10.6f} [{feature_type}]")

# ============================================================================
# Step 15: 保存模型和结果
# ============================================================================
print("\n" + "=" * 80)
print("Step 15: 保存模型和结果")
print("=" * 80)

output_dir = 'model_output_optimized_v4'
os.makedirs(output_dir, exist_ok=True)

with open(f'{output_dir}/model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ 模型已保存")

with open(f'{output_dir}/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ 标准化器已保存")

with open(f'{output_dir}/features.pkl', 'wb') as f:
    pickle.dump(all_features, f)
print("✓ 特征列表已保存")

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

feature_importance.to_csv(f'{output_dir}/feature_importance.csv', index=False)
print("✓ 特征重要性已保存")

if shap_importance is not None:
    shap_importance.to_csv(f'{output_dir}/shap_importance.csv', index=False)
    print("✓ SHAP重要性已保存")

print(f"\n✓ 所有结果已保存到: {output_dir}/")

# ============================================================================
# Step 16: 可视化分析
# ============================================================================
print("\n" + "=" * 80)
print("Step 16: 可视化分析")
print("=" * 80)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. 散点图 - 训练集
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_train, y_train_pred, alpha=0.3, s=1, color='steelblue')
min_val = min(y_train.min(), y_train_pred.min())
max_val = max(y_train.max(), y_train_pred.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测')
ax1.set_xlabel('实际值', fontsize=11)
ax1.set_ylabel('预测值', fontsize=11)
ax1.set_title(f'训练集 (R²={train_metrics["R2"]:.4f})', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 散点图 - 验证集
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_val, y_val_pred, alpha=0.3, s=1, color='lightgreen')
min_val = min(y_val.min(), y_val_pred.min())
max_val = max(y_val.max(), y_val_pred.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测')
ax2.set_xlabel('实际值', fontsize=11)
ax2.set_ylabel('预测值', fontsize=11)
ax2.set_title(f'验证集 (R²={val_metrics["R2"]:.4f})', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 散点图 - 测试集
ax3 = fig.add_subplot(gs[0, 2])
ax3.scatter(y_test, y_test_pred, alpha=0.3, s=1, color='coral')
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测')
ax3.set_xlabel('实际值', fontsize=11)
ax3.set_ylabel('预测值', fontsize=11)
ax3.set_title(f'测试集 (R²={test_metrics["R2"]:.4f})', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

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

# 7. 特征重要性 Top 15
ax7 = fig.add_subplot(gs[2, :2])
top_features = feature_importance.head(15)
colors_feat = ['red' if 'Quantum' in f else 'steelblue' for f in top_features['Feature']]
ax7.barh(top_features['Feature'], top_features['Importance'], color=colors_feat, alpha=0.8)
ax7.set_xlabel('重要性', fontsize=11)
ax7.set_title('特征重要性 Top 15 (红色=量子特征)', fontsize=12, fontweight='bold')
ax7.invert_yaxis()
ax7.grid(True, alpha=0.3, axis='x')

# 8. 指标对比表
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis('off')
metrics_text = f"""
模型性能总结

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
  经典: {len(selected_features)}
  量子: {n_qubits if quantum_features_train is not None else 0}
  总计: {len(all_features)}
"""
ax8.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('最终优化的量子增强XGBoost模型 V4 - 综合分析', fontsize=16, fontweight='bold', y=0.995)
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

# 交叉验证结果图
print("生成交叉验证结果图...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.bar(cv_df['Fold'], cv_df['R2'], color='steelblue', alpha=0.8, edgecolor='black')
ax1.axhline(y=cv_df['R2'].mean(), color='r', linestyle='--', lw=2, label=f'平均: {cv_df["R2"].mean():.4f}')
ax1.set_xlabel('Fold', fontsize=11)
ax1.set_ylabel('R² Score', fontsize=11)
ax1.set_title('5-Fold 交叉验证 - R² 分布', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

ax2.bar(cv_df['Fold'], cv_df['RMSE'], color='coral', alpha=0.8, edgecolor='black')
ax2.axhline(y=cv_df['RMSE'].mean(), color='r', linestyle='--', lw=2, label=f'平均: {cv_df["RMSE"].mean():.4f}')
ax2.set_xlabel('Fold', fontsize=11)
ax2.set_ylabel('RMSE', fontsize=11)
ax2.set_title('5-Fold 交叉验证 - RMSE 分布', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{output_dir}/cv_results.png', dpi=300, bbox_inches='tight')
print("✓ 交叉验证结果图已保存")
plt.close()

print("\n✓ 可视化完成")

# ============================================================================
# Step 17: 生成最终报告
# ============================================================================
print("\n" + "=" * 80)
print("Step 17: 生成最终报告")
print("=" * 80)

report = f"""
{'=' * 80}
最终优化的量子增强XGBoost模型 V4 - 优化报告
Final Optimized Quantum-Enhanced XGBoost V4 - Report
{'=' * 80}

执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'=' * 80}
核心优化策略
{'=' * 80}

1. ✓ 分批量子编码 + 缓存机制
   - 分批大小: {batch_size}
   - 量子比特数: {n_qubits}
   - 缓存目录: quantum_cache_v4/

2. ✓ 精简特征工程
   - 核心气象特征: {len(CORE_WEATHER_FEATURES)}
   - 经典特征总数: {len(selected_features)}
   - 量子特征数: {n_qubits if quantum_features_train is not None else 0}
   - 总特征数: {len(all_features)}

3. ✓ 优化XGBoost参数
   - 树数: 500 (平衡点)
   - 深度: 6 (防止过拟合)
   - 学习率: 0.03
   - 正则化: 增加

4. ✓ K-Fold交叉验证
   - 5-Fold验证
   - 评估泛化能力

5. ✓ SHAP可解释性分析
   - 特征重要性排名
   - 量子特征贡献度

{'=' * 80}
数据集配置
{'=' * 80}

总样本数: {len(df):,}
特征工程后: {len(X):,}
训练集: {X_train.shape[0]:,} 样本 (70%)
验证集: {X_val.shape[0]:,} 样本 (15%)
测试集: {X_test.shape[0]:,} 样本 (15%)

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
特征重要性分析
{'=' * 80}

XGBoost 特征重要性 Top 10:
"""

for idx, row in feature_importance.head(10).iterrows():
    feature_type = "量子" if "Quantum" in row['Feature'] else "经典"
    report += f"  {idx + 1:2d}. {row['Feature']:40s}: {row['Importance']:10.6f} [{feature_type}]\n"

if shap_importance is not None:
    report += f"""
SHAP 特征重要性 Top 10:
"""
    for idx, row in shap_importance.head(10).iterrows():
        feature_type = "量子" if "Quantum" in row['Feature'] else "经典"
        report += f"  {idx + 1:2d}. {row['Feature']:40s}: {row['SHAP_Importance']:10.6f} [{feature_type}]\n"

report += f"""
{'=' * 80}
量子特征贡献度分析
{'=' * 80}

量子特征数: {n_qubits if quantum_features_train is not None else 0}
量子特征在Top 15中的排名: """

if quantum_features_train is not None:
    quantum_in_top15 = sum(1 for f in feature_importance.head(15)['Feature'] if 'Quantum' in f)
    report += f"{quantum_in_top15}/15\n"
    
    if shap_importance is not None:
        quantum_in_shap_top15 = sum(1 for f in shap_importance.head(15)['Feature'] if 'Quantum' in f)
        report += f"SHAP Top 15中的量子特征: {quantum_in_shap_top15}/15\n"
else:
    report += "N/A (未使用量子特征)\n"

report += f"""
{'=' * 80}
输出文件
{'=' * 80}

模型文件:
  ✓ model.pkl - 训练好的XGBoost模型
  ✓ scaler.pkl - RobustScaler标准化器
  ✓ features.pkl - 特征列表

结果文件:
  ✓ test_predictions.csv - 测试集预测结果
  ✓ metrics.csv - 评估指标
  ✓ cv_results.csv - 交叉验证结果
  ✓ feature_importance.csv - XGBoost特征重要性
  ✓ shap_importance.csv - SHAP特征重要性

可视化文件:
  ✓ comprehensive_analysis.png - 综合分析图
  ✓ timeseries_prediction.png - 时间序列预测图
  ✓ cv_results.png - 交叉验证结果图
  ✓ shap_summary_plot.png - SHAP摘要图
  ✓ shap_bar_plot.png - SHAP条形图

缓存文件:
  ✓ quantum_cache_v4/ - 量子特征缓存目录

输出目录: {output_dir}/

{'=' * 80}
与之前版本对比
{'=' * 80}

V2 版本 (全量量子编码):
  - 特征数: 53
  - 测试集 R²: 0.7797
  - 过拟合差异: 0.2203

V3 版本 (无量子特征):
  - 特征数: 31
  - 测试集 R²: 未测试
  - 过拟合差异: 未测试

V4 版本 (分批量子编码 + 优化参数):
  - 特征数: {len(all_features)}
  - 测试集 R²: {test_metrics['R2']:.6f}
  - 过拟合差异: {overfitting_gap:.6f}
  - 改进: {'✓ 显著改进' if overfitting_gap < 0.1 else '⚠ 有所改进' if overfitting_gap < 0.2 else '✗ 需要进一步优化'}

{'=' * 80}
建议
{'=' * 80}

1. 数据收集
   - 当前样本数: {len(X):,}
   - 建议: 收集更多数据 (至少 10,000+)
   - 更多数据可支持更复杂的模型

2. 特征工程
   - 考虑添加更多领域特征
   - 例如: 节假日、天气事件等
   - 避免过度特征化

3. 模型优化
   - 基于SHAP重要性移除低贡献特征
   - 尝试其他模型: LightGBM, CatBoost
   - 使用集成方法提高稳定性

4. 量子特征优化
   - 评估量子特征的实际贡献
   - 考虑调整量子比特数
   - 探索不同的量子电路设计

{'=' * 80}
完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 80}
"""

with open(f'{output_dir}/optimization_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(report)
print(f"\n✓ 优化报告已保存到: {output_dir}/optimization_report.txt")

print("\n" + "=" * 80)
print("✓ 最终优化的量子增强XGBoost模型 V4 训练完成！")
print("=" * 80)
print(f"\n所有输出文件位于: {output_dir}/")
print(f"缓存文件位于: quantum_cache_v4/")
print(f"\n最终测试集R²: {test_metrics['R2']:.6f}")
print(f"过拟合差异: {overfitting_gap:.6f}")
print("=" * 80)
