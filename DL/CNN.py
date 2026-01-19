import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

print("="*80)
print("PyTorch CNN 优化版本 - 电力负荷预测")
print("="*80)
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# Step 1-4: 数据加载和特征工程
# ============================================================================
print("\nStep 1-4: 数据加载和特征工程")

chunk_size = 100000
chunks = []
for i, chunk in enumerate(pd.read_csv('data/df_merged_sorted_by_time.csv', chunksize=chunk_size)):
    chunks.append(chunk)
    if (i + 1) % 10 == 0:
        print(f"  已加载 {(i + 1) * chunk_size:,} 行...")

df = pd.concat(chunks, ignore_index=True)
transformer_counts = df['TRANSFORMER_ID'].value_counts()
selected_transformer = transformer_counts.index[0]
df = df[df['TRANSFORMER_ID'] == selected_transformer].copy()
df = df.sort_values('DATETIME').reset_index(drop=True)

df['DATETIME'] = pd.to_datetime(df['DATETIME'])

CORE_WEATHER_FEATURES = ['TEMP', 'MIN', 'MAX', 'DEWP', 'SLP', 'MXSPD', 'GUST', 'STP', 'WDSP', 'RH', 'PRCP']

df['Hour'] = df['DATETIME'].dt.hour
df['DayOfWeek'] = df['DATETIME'].dt.dayofweek
df['Month'] = df['DATETIME'].dt.month
df['Day'] = df['DATETIME'].dt.day
df['Quarter'] = df['DATETIME'].dt.quarter
df['DayOfYear'] = df['DATETIME'].dt.dayofyear
df['WeekOfYear'] = df['DATETIME'].dt.isocalendar().week

df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
df['IsPeakHour'] = df['Hour'].isin([8, 9, 10, 18, 19, 20]).astype(int)
df['IsHolidaySeason'] = df['Month'].isin([12, 1]).astype(int)
df['IsSummerPeak'] = df['Month'].isin([6, 7, 8]).astype(int)
df['IsWinterPeak'] = df['Month'].isin([12, 1, 2]).astype(int)

key_lags = [1, 2, 3, 4, 6, 12, 24, 48, 72, 168, 336, 504]
for lag in key_lags:
    df[f'LOAD_lag{lag}'] = df['LOAD'].shift(lag)

key_windows = [3, 6, 12, 24, 48, 168, 336]
for window in key_windows:
    df[f'LOAD_rolling_mean_{window}'] = df['LOAD'].rolling(window=window, min_periods=1).mean().shift(1)
    df[f'LOAD_rolling_std_{window}'] = df['LOAD'].rolling(window=window, min_periods=1).std().shift(1)
    df[f'LOAD_rolling_min_{window}'] = df['LOAD'].rolling(window=window, min_periods=1).min().shift(1)
    df[f'LOAD_rolling_max_{window}'] = df['LOAD'].rolling(window=window, min_periods=1).max().shift(1)
    df[f'LOAD_rolling_median_{window}'] = df['LOAD'].rolling(window=window, min_periods=1).median().shift(1)

df['TEMP_RH'] = df['TEMP'] * df['RH']
df['TEMP_MXSPD'] = df['TEMP'] * df['MXSPD']
df['TEMP_GUST'] = df['TEMP'] * df['GUST']
df['TEMP_range'] = df['MAX'] - df['MIN']
df['TEMP_squared'] = df['TEMP'] ** 2
df['TEMP_cubed'] = df['TEMP'] ** 3
df['RH_squared'] = df['RH'] ** 2
df['DEWP_TEMP'] = df['DEWP'] * df['TEMP']
df['SLP_TEMP'] = df['SLP'] * df['TEMP']
df['RH_MXSPD'] = df['RH'] * df['MXSPD']
df['TEMP_SLP'] = df['TEMP'] * df['SLP']

df['LOAD_diff1'] = df['LOAD'].diff(1)
df['LOAD_diff24'] = df['LOAD'].diff(24)
df['LOAD_diff168'] = df['LOAD'].diff(168)
df['LOAD_pct_change'] = df['LOAD'].pct_change()
df['TEMP_diff1'] = df['TEMP'].diff(1)
df['TEMP_diff24'] = df['TEMP'].diff(24)
df['LOAD_momentum_24'] = df['LOAD'].diff(24) - df['LOAD'].diff(48)
df['LOAD_momentum_168'] = df['LOAD'].diff(168) - df['LOAD'].diff(336)

df = df.dropna()

selected_features = (
    CORE_WEATHER_FEATURES +
    ['Hour', 'DayOfWeek', 'Month', 'Day', 'Quarter', 'DayOfYear', 'WeekOfYear'] +
    ['Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos', 'DayOfWeek_sin', 'DayOfWeek_cos'] +
    ['IsWeekend', 'IsPeakHour', 'IsHolidaySeason', 'IsSummerPeak', 'IsWinterPeak'] +
    [f'LOAD_lag{lag}' for lag in key_lags] +
    [f'LOAD_rolling_mean_{window}' for window in key_windows] +
    [f'LOAD_rolling_std_{window}' for window in key_windows] +
    [f'LOAD_rolling_min_{window}' for window in key_windows] +
    [f'LOAD_rolling_max_{window}' for window in key_windows] +
    [f'LOAD_rolling_median_{window}' for window in key_windows] +
    ['TEMP_RH', 'TEMP_MXSPD', 'TEMP_GUST', 'TEMP_range', 'TEMP_squared', 'TEMP_cubed',
     'RH_squared', 'DEWP_TEMP', 'SLP_TEMP', 'RH_MXSPD', 'TEMP_SLP'] +
    ['LOAD_diff1', 'LOAD_diff24', 'LOAD_diff168', 'LOAD_pct_change', 'TEMP_diff1', 'TEMP_diff24',
     'LOAD_momentum_24', 'LOAD_momentum_168']
)

selected_features = [f for f in selected_features if f in df.columns and df[f].dtype in ['int64', 'float64', 'int32', 'float32']]

X = df[selected_features].copy()
y = df['LOAD'].copy()

print(f"✓ 数据加载完成: {X.shape}")

# ============================================================================
# Step 5: 时间序列分割和标准化
# ============================================================================
print("\nStep 5: 时间序列分割和标准化")

train_size = int(0.75 * len(X))
val_size = int(0.10 * len(X))

X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:train_size + val_size]
y_val = y[train_size:train_size + val_size]
X_test = X[train_size + val_size:]
y_test = y[train_size + val_size:]

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

y_scaler = RobustScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
y_val_scaled = y_scaler.transform(y_val.values.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()

print(f"✓ 训练集: {X_train.shape[0]:,} 样本")
print(f"✓ 验证集: {X_val.shape[0]:,} 样本")
print(f"✓ 测试集: {X_test.shape[0]:,} 样本")

# ============================================================================
# Step 6: 创建时间窗口序列
# ============================================================================
print("\nStep 6: 创建时间窗口序列")

window_size = 24

def create_sequences(X, y, window_size):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size])
    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, window_size)
X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, window_size)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, window_size)

print(f"✓ 训练序列: {X_train_seq.shape}")
print(f"✓ 验证序列: {X_val_seq.shape}")
print(f"✓ 测试序列: {X_test_seq.shape}")

# ============================================================================
# Step 7: 转换为PyTorch张量
# ============================================================================
print("\nStep 7: 转换为PyTorch张量")

X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
y_train_tensor = torch.FloatTensor(y_train_seq).to(device)
X_val_tensor = torch.FloatTensor(X_val_seq).to(device)
y_val_tensor = torch.FloatTensor(y_val_seq).to(device)
X_test_tensor = torch.FloatTensor(X_test_seq).to(device)
y_test_tensor = torch.FloatTensor(y_test_seq).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("✓ 张量转换完成")


# ============================================================================
# Step 8: 定义优化的CNN模型 (Inception + 注意力机制)
# ============================================================================
print("\nStep 8: 定义优化的CNN模型")

class SpatialAttention(nn.Module):
    """空间注意力机制 - 关注重要特征"""
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv1d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (batch, channels, time)
        attention = self.sigmoid(self.conv(x))
        return x * attention

class TemporalAttention(nn.Module):
    """时间注意力机制 - 关注重要时间步"""
    def __init__(self, channels, time_steps):
        super(TemporalAttention, self).__init__()
        self.fc1 = nn.Linear(time_steps, time_steps // 4)
        self.fc2 = nn.Linear(time_steps // 4, time_steps)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (batch, channels, time)
        batch_size, channels, time_steps = x.size()
        
        # 全局平均池化
        gap = torch.mean(x, dim=1)  # (batch, time)
        
        # 注意力权重
        attention = self.fc2(torch.relu(self.fc1(gap)))
        attention = self.sigmoid(attention)  # (batch, time)
        
        # 应用注意力
        attention = attention.unsqueeze(1)  # (batch, 1, time)
        return x * attention

class InceptionBlock(nn.Module):
    """Inception块 - 多尺度特征提取"""
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        
        # 1x1卷积
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm1d(out_channels // 4),
            nn.GELU()
        )
        
        # 3x3卷积
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels // 4),
            nn.GELU()
        )
        
        # 5x5卷积
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels // 4),
            nn.GELU()
        )
        
        # 最大池化 + 1x1卷积
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm1d(out_channels // 4),
            nn.GELU()
        )
    
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)

class ResidualInceptionBlock(nn.Module):
    """残差Inception块"""
    def __init__(self, in_channels, out_channels):
        super(ResidualInceptionBlock, self).__init__()
        self.inception = InceptionBlock(in_channels, out_channels)
        
        # 残差连接投影
        self.use_projection = in_channels != out_channels
        if self.use_projection:
            self.projection = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        if self.use_projection:
            residual = self.projection(residual)
        
        out = self.inception(x)
        return out + residual

class OptimizedCNN(nn.Module):
    """优化的CNN模型 - Inception + 注意力机制"""
    def __init__(self, input_channels, time_steps):
        super(OptimizedCNN, self).__init__()
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.GELU()
        )
        
        # Inception块1
        self.inception1 = ResidualInceptionBlock(64, 128)
        self.spatial_attn1 = SpatialAttention(128)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Inception块2
        self.inception2 = ResidualInceptionBlock(128, 256)
        self.spatial_attn2 = SpatialAttention(256)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Inception块3
        self.inception3 = ResidualInceptionBlock(256, 256)
        self.spatial_attn3 = SpatialAttention(256)
        
        # 时间注意力
        self.temporal_attn = TemporalAttention(256, time_steps // 4)
        
        # 自适应池化
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # 全连接层
        self.fc1 = nn.Linear(256, 128)
        self.ln1 = nn.LayerNorm(128)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc_out = nn.Linear(64, 1)
        
        self.gelu = nn.GELU()
    
    def forward(self, x):
        # 输入投影
        x = self.input_proj(x)
        
        # Inception块1 + 注意力
        x = self.inception1(x)
        x = self.spatial_attn1(x)
        x = self.pool1(x)
        
        # Inception块2 + 注意力
        x = self.inception2(x)
        x = self.spatial_attn2(x)
        x = self.pool2(x)
        
        # Inception块3 + 注意力
        x = self.inception3(x)
        x = self.spatial_attn3(x)
        
        # 时间注意力
        x = self.temporal_attn(x)
        
        # 自适应池化
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = self.gelu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = self.gelu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = self.fc_out(x)
        return x

model = OptimizedCNN(X_train_seq.shape[1], X_train_seq.shape[2]).to(device)
print("✓ 优化的CNN模型构建完成")
print(f"  模型参数数: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# Step 9: 定义损失函数和优化器
# ============================================================================
print("\nStep 9: 定义损失函数和优化器")

criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-7)
scaler = GradScaler()

print("✓ 损失函数和优化器定义完成")
print("  优化器: AdamW (lr=0.0008, weight_decay=1e-4)")
print("  学习率调度: CosineAnnealingWarmRestarts")
print("  混合精度训练: 启用")

# ============================================================================
# Step 10: 训练模型
# ============================================================================
print("\nStep 10: 训练优化的CNN模型")

train_losses = []
val_losses = []
best_val_loss = float('inf')
patience = 35
patience_counter = 0

train_start = datetime.now()

for epoch in range(250):
    # 训练
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        
        # 混合精度训练
        with autocast():
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item() * X_batch.size(0)
    
    train_loss /= len(train_dataset)
    train_losses.append(train_loss)
    
    # 验证
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            with autocast():
                y_pred = model(X_batch).squeeze()
                loss = criterion(y_pred, y_batch)
            val_loss += loss.item() * X_batch.size(0)
    
    val_loss /= len(val_dataset)
    val_losses.append(val_loss)
    
    scheduler.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/250 - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_cnn_optimized_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

train_time = (datetime.now() - train_start).total_seconds()
print(f"\n✓ 训练完成（耗时: {train_time:.2f}秒）")

# 加载最佳模型
model.load_state_dict(torch.load('best_cnn_optimized_model.pt'))

# ============================================================================
# Step 11: 预测
# ============================================================================
print("\nStep 11: 预测")

model.eval()
with torch.no_grad():
    y_train_pred_scaled = model(X_train_tensor).cpu().numpy().flatten()
    y_val_pred_scaled = model(X_val_tensor).cpu().numpy().flatten()
    y_test_pred_scaled = model(X_test_tensor).cpu().numpy().flatten()

y_train_pred = y_scaler.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
y_val_pred = y_scaler.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).flatten()
y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()

# 对齐目标变量
y_train_aligned = y_train.values[window_size:]
y_val_aligned = y_val.values[window_size:]
y_test_aligned = y_test.values[window_size:]

print("✓ 预测完成")

# ============================================================================
# Step 12: 评估
# ============================================================================
print("\nStep 12: 评估")

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

train_metrics = calculate_metrics(y_train_aligned, y_train_pred, "训练集")
val_metrics = calculate_metrics(y_val_aligned, y_val_pred, "验证集")
test_metrics = calculate_metrics(y_test_aligned, y_test_pred, "测试集")

# ============================================================================
# Step 13: 保存模型和结果
# ============================================================================
print("\nStep 13: 保存模型和结果")

output_dir = 'model_output_cnn_pytorch_optimized'
os.makedirs(output_dir, exist_ok=True)

torch.save(model.state_dict(), f'{output_dir}/cnn_optimized_model.pt')
with open(f'{output_dir}/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open(f'{output_dir}/y_scaler.pkl', 'wb') as f:
    pickle.dump(y_scaler, f)
with open(f'{output_dir}/features.pkl', 'wb') as f:
    pickle.dump(selected_features, f)

results_df = pd.DataFrame({
    'Actual': y_test_aligned,
    'Predicted': y_test_pred,
    'Error': y_test_pred - y_test_aligned,
    'Abs_Error': np.abs(y_test_pred - y_test_aligned),
    'Percent_Error': np.abs(y_test_pred - y_test_aligned) / (np.abs(y_test_aligned) + 1e-10) * 100,
})
results_df.to_csv(f'{output_dir}/test_predictions.csv', index=False)

metrics_df = pd.DataFrame([
    {'Set': '训练集', **train_metrics},
    {'Set': '验证集', **val_metrics},
    {'Set': '测试集', **test_metrics},
])
metrics_df.to_csv(f'{output_dir}/metrics.csv', index=False)

print("✓ 模型和结果已保存")


# ============================================================================
# Step 14: 可视化
# ============================================================================
print("\nStep 14: 可视化")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(train_losses, label='训练损失', linewidth=2)
ax1.plot(val_losses, label='验证损失', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('损失', fontsize=11)
ax1.set_title('训练历史 - 损失', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_test_aligned, y_test_pred, alpha=0.3, s=1, color='coral')
min_val = min(y_test_aligned.min(), y_test_pred.min())
max_val = max(y_test_aligned.max(), y_test_pred.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测')
ax2.set_xlabel('实际值', fontsize=11)
ax2.set_ylabel('预测值', fontsize=11)
ax2.set_title(f'测试集 (R²={test_metrics["R2"]:.4f})', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[0, 2])
errors = y_test_pred - y_test_aligned
ax3.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax3.axvline(x=0, color='r', linestyle='--', lw=2)
ax3.set_xlabel('误差', fontsize=11)
ax3.set_ylabel('频数', fontsize=11)
ax3.set_title('误差分布', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(y_test_pred, errors, alpha=0.3, s=1, color='steelblue')
ax4.axhline(y=0, color='r', linestyle='--', lw=2)
ax4.set_xlabel('预测值', fontsize=11)
ax4.set_ylabel('残差', fontsize=11)
ax4.set_title('残差图', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

ax5 = fig.add_subplot(gs[1, 1])
sets = ['训练集', '验证集', '测试集']
r2_scores = [train_metrics['R2'], val_metrics['R2'], test_metrics['R2']]
colors = ['steelblue', 'lightgreen', 'coral']
bars = ax5.bar(sets, r2_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax5.set_ylabel('R² Score', fontsize=11)
ax5.set_title('R²对比', fontsize=12, fontweight='bold')
ax5.set_ylim([min(r2_scores) - 0.01, 1.0])
ax5.grid(True, alpha=0.3, axis='y')
for bar, score in zip(bars, r2_scores):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax6 = fig.add_subplot(gs[1, 2])
metrics_names = ['R²', 'RMSE', 'MAE']
train_vals = [train_metrics['R2'], train_metrics['RMSE']/1000, train_metrics['MAE']/1000]
test_vals = [test_metrics['R2'], test_metrics['RMSE']/1000, test_metrics['MAE']/1000]
x = np.arange(len(metrics_names))
width = 0.35
ax6.bar(x - width/2, train_vals, width, label='训练集', alpha=0.8)
ax6.bar(x + width/2, test_vals, width, label='测试集', alpha=0.8)
ax6.set_ylabel('值', fontsize=11)
ax6.set_title('指标对比', fontsize=12, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(metrics_names)
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

ax7 = fig.add_subplot(gs[2, :])
plot_size = min(1000, len(y_test_aligned))
x_axis = range(plot_size)
ax7.plot(x_axis, y_test_aligned[:plot_size], label='实际值', color='blue', alpha=0.7, linewidth=1)
ax7.plot(x_axis, y_test_pred[:plot_size], label='预测值', color='red', alpha=0.7, linewidth=1)
ax7.fill_between(x_axis, y_test_aligned[:plot_size], y_test_pred[:plot_size], alpha=0.2, color='gray')
ax7.set_xlabel('时间步', fontsize=12)
ax7.set_ylabel('负荷', fontsize=12)
ax7.set_title(f'时间序列预测对比（前{plot_size}个样本）', fontsize=14, fontweight='bold')
ax7.legend(fontsize=11)
ax7.grid(True, alpha=0.3)

plt.suptitle('PyTorch CNN 优化版本 - 综合分析', fontsize=16, fontweight='bold', y=0.995)
plt.savefig(f'{output_dir}/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
print("✓ 综合分析图已保存")
plt.close()

# ============================================================================
# Step 15: 生成报告
# ============================================================================
print("\nStep 15: 生成报告")

report = f"""
{'='*80}
PyTorch CNN 优化版本 - 电力负荷预测报告
{'='*80}

执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}
优化策略
{'='*80}

1. Inception块 (多尺度特征提取)
   - 1x1卷积: 捕捉点特征
   - 3x3卷积: 捕捉局部特征
   - 5x5卷积: 捕捉全局特征
   - 最大池化: 捕捉显著特征
   - 4个分支并行处理，提高特征多样性

2. 残差连接 (Residual Connections)
   - 解决深层网络梯度消失
   - 每个Inception块都有残差连接
   - 投影层处理维度不匹配

3. 空间注意力机制 (Spatial Attention)
   - 关注重要的特征维度
   - 每个Inception块后添加
   - 学习特征间的重要性权重

4. 时间注意力机制 (Temporal Attention)
   - 关注重要的时间步
   - 全局平均池化 + 全连接层
   - 学习时间步间的重要性权重

5. 自适应池化 (Adaptive Pooling)
   - 自动适应不同的输入大小
   - 保留最重要的信息
   - 更稳定的特征聚合

6. 混合精度训练 (Mixed Precision)
   - 使用autocast加速训练
   - 减少内存占用
   - 保持数值精度

{'='*80}
模型架构
{'='*80}

框架: PyTorch
设备: {device}
总参数数: {sum(p.numel() for p in model.parameters()):,}
时间窗口: {window_size}小时

层结构:
  输入投影: {X_train_seq.shape[1]} → 64 + GELU + BatchNorm
  
  Inception块1 (残差):
    - 4个分支 (1x1, 3x3, 5x5, MaxPool)
    - 输出: 128通道
    - 空间注意力
    - MaxPool (stride=2)
  
  Inception块2 (残差):
    - 4个分支 (1x1, 3x3, 5x5, MaxPool)
    - 输出: 256通道
    - 空间注意力
    - MaxPool (stride=2)
  
  Inception块3 (残差):
    - 4个分支 (1x1, 3x3, 5x5, MaxPool)
    - 输出: 256通道
    - 空间注意力
  
  时间注意力: 256通道
  
  自适应平均池化: 256 → 1
  
  全连接层:
    Linear(256, 128) + LayerNorm + Dropout(0.3)
    Linear(128, 64) + LayerNorm + Dropout(0.2)
    Linear(64, 1)

优化器: AdamW (lr=0.0008, weight_decay=1e-4)
损失函数: MSE
学习率调度: CosineAnnealingWarmRestarts (T_0=15, T_mult=2)
混合精度: 启用 (GradScaler)

{'='*80}
数据集
{'='*80}

原始样本数: {len(df):,}
训练序列: {X_train_seq.shape[0]:,}
验证序列: {X_val_seq.shape[0]:,}
测试序列: {X_test_seq.shape[0]:,}

特征数: {len(selected_features)}
时间窗口: {window_size}小时

{'='*80}
模型性能
{'='*80}

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
  训练-测试R²差异: {abs(train_metrics['R2'] - test_metrics['R2']):.6f}
  状态: {'✓ 良好' if abs(train_metrics['R2'] - test_metrics['R2']) < 0.05 else '⚠ 需要关注'}

{'='*80}
训练统计
{'='*80}

总训练时间: {train_time:.2f}秒
总Epoch数: {len(train_losses)}
最佳验证损失: {min(val_losses):.6f}
最终训练损失: {train_losses[-1]:.6f}

{'='*80}
输出文件
{'='*80}

模型文件:
  ✓ cnn_optimized_model.pt - 训练好的优化CNN模型
  ✓ scaler.pkl - 特征标准化器
  ✓ y_scaler.pkl - 目标变量标准化器
  ✓ features.pkl - 特征列表

结果文件:
  ✓ test_predictions.csv - 测试集预测结果
  ✓ metrics.csv - 评估指标汇总

可视化文件:
  ✓ comprehensive_analysis.png - 综合分析图

输出目录: {output_dir}/

{'='*80}
完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""

with open(f'{output_dir}/report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(report)

print("\n" + "="*80)
print("✓ PyTorch CNN优化版本训练完成！")
print("="*80)
print(f"\n最终测试集R²: {test_metrics['R2']:.6f}")
print(f"所有输出文件位于: {output_dir}/")
print("="*80)

# 清理临时文件
os.remove('best_cnn_optimized_model.pt')
