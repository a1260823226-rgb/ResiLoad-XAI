"""
SVR and RF Comparison Models - Time Series Prediction
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import os

warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

print("="*80)
print("SVR and RF Comparison Models - Time Series Prediction")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load data
print("\n" + "="*80)
print("Step 1: Load Data")
print("="*80)

chunk_size = 100000
chunks = []
print("Loading sorted data...")

try:
    for i, chunk in enumerate(pd.read_csv('data/df_merged_sorted_by_time.csv', chunksize=chunk_size)):
        chunks.append(chunk)
        if (i + 1) % 10 == 0:
            print(f"  Loaded {(i + 1) * chunk_size:,} rows...")
    
    df = pd.concat(chunks, ignore_index=True)
    print(f"OK - Data loaded: {df.shape}")
except FileNotFoundError:
    print("ERROR - Sorted file not found")
    exit(1)

# Select transformer
print("\n" + "="*80)
print("Step 2: Select Transformer")
print("="*80)

transformer_counts = df['TRANSFORMER_ID'].value_counts()
selected_transformer = transformer_counts.index[0]
selected_count = transformer_counts.iloc[0]

print(f"Selected transformer: {selected_transformer}")

df = df[df['TRANSFORMER_ID'] == selected_transformer].copy()
df = df.sort_values('DATETIME').reset_index(drop=True)

print(f"OK - Filtered data shape: {df.shape}")

# Feature engineering
print("\n" + "="*80)
print("Step 3: Feature Engineering")
print("="*80)

df['DATETIME'] = pd.to_datetime(df['DATETIME'])

print("Adding time features...")

df['Hour'] = df['DATETIME'].dt.hour
df['DayOfWeek'] = df['DATETIME'].dt.dayofweek
df['Month'] = df['DATETIME'].dt.month
df['Day'] = df['DATETIME'].dt.day
df['Quarter'] = df['DATETIME'].dt.quarter
df['DayOfYear'] = df['DATETIME'].dt.dayofyear

df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
df['IsPeakHour'] = df['Hour'].isin([8, 9, 10, 12, 13, 14, 18, 19, 20]).astype(int)
df['IsNightHour'] = df['Hour'].isin([0, 1, 2, 3, 4, 5]).astype(int)

print("Adding lag features...")

lags = [1, 2, 3, 6, 12, 24, 48, 72, 168, 336]
for lag in lags:
    df[f'LOAD_lag{lag}'] = df['LOAD'].shift(lag)

print("Adding rolling features...")

windows = [6, 12, 24, 48, 168]
for window in windows:
    df[f'LOAD_rolling_mean_{window}'] = df['LOAD'].rolling(window=window, min_periods=1).mean().shift(1)
    df[f'LOAD_rolling_std_{window}'] = df['LOAD'].rolling(window=window, min_periods=1).std().shift(1)

print("Adding difference features...")

df['LOAD_diff1'] = df['LOAD'].diff(1)
df['LOAD_diff24'] = df['LOAD'].diff(24)

print("Adding weather interaction features...")

df['TEMP_RH'] = df['TEMP'] * df['RH']
df['TEMP_MXSPD'] = df['TEMP'] * df['MXSPD']
df['TEMP_GUST'] = df['TEMP'] * df['GUST']
df['TEMP_squared'] = df['TEMP'] ** 2
df['TEMP_range'] = df['MAX'] - df['MIN']

df = df.dropna()

print(f"OK - Feature engineering complete")
print(f"  Data shape: {df.shape}")

# Feature selection
print("\n" + "="*80)
print("Step 4: Feature Selection")
print("="*80)

exclude_cols = ['DATETIME', 'TRANSFORMER_ID', 'LOAD']
selected_features = [col for col in df.columns if col not in exclude_cols]

numeric_features = []
for col in selected_features:
    if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
        numeric_features.append(col)

selected_features = numeric_features
print(f"Selected features: {len(selected_features)}")

# Prepare data
print("\n" + "="*80)
print("Step 5: Prepare Data")
print("="*80)

X = df[selected_features].copy()
y = df['LOAD'].copy()

print(f"Feature shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Time series split
print("\n" + "="*80)
print("Step 6: Time Series Split")
print("="*80)

train_size = int(0.7 * len(X))
val_size = int(0.15 * len(X))

X_train = X[:train_size]
y_train = y[:train_size]

X_val = X[train_size:train_size + val_size]
y_val = y[train_size:train_size + val_size]

X_test = X[train_size + val_size:]
y_test = y[train_size + val_size:]

print(f"Train set: {X_train.shape[0]:,} samples")
print(f"Val set: {X_val.shape[0]:,} samples")
print(f"Test set: {X_test.shape[0]:,} samples")

# Standardization
print("\n" + "="*80)
print("Step 7: Standardization")
print("="*80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("OK - Standardization complete")

# ============================================================================
# SVR Model
# ============================================================================

print("\n" + "="*80)
print("Step 8: SVR Model Training")
print("="*80)

print("Training SVR...")
train_start = datetime.now()

svr_model = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
svr_model.fit(X_train_scaled, y_train)

train_time = (datetime.now() - train_start).total_seconds()
print(f"OK - SVR training complete (Time: {train_time:.2f}s)")

# SVR Prediction
print("\nSVR Prediction...")

y_train_pred_svr = svr_model.predict(X_train_scaled)
y_val_pred_svr = svr_model.predict(X_val_scaled)
y_test_pred_svr = svr_model.predict(X_test_scaled)

def calculate_metrics(y_true, y_pred, set_name, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} - {set_name} Metrics:")
    print(f"  R2 Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  MAPE: {mape:.4f}%")
    
    return {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

svr_train_metrics = calculate_metrics(y_train, y_train_pred_svr, "Train", "SVR")
svr_val_metrics = calculate_metrics(y_val, y_val_pred_svr, "Val", "SVR")
svr_test_metrics = calculate_metrics(y_test, y_test_pred_svr, "Test", "SVR")

# ============================================================================
# Random Forest Model
# ============================================================================

print("\n" + "="*80)
print("Step 9: Random Forest Model Training")
print("="*80)

print("Training Random Forest...")
train_start = datetime.now()

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    verbose=1
)
rf_model.fit(X_train_scaled, y_train)

train_time = (datetime.now() - train_start).total_seconds()
print(f"OK - RF training complete (Time: {train_time:.2f}s)")

# RF Prediction
print("\nRF Prediction...")

y_train_pred_rf = rf_model.predict(X_train_scaled)
y_val_pred_rf = rf_model.predict(X_val_scaled)
y_test_pred_rf = rf_model.predict(X_test_scaled)

rf_train_metrics = calculate_metrics(y_train, y_train_pred_rf, "Train", "RF")
rf_val_metrics = calculate_metrics(y_val, y_val_pred_rf, "Val", "RF")
rf_test_metrics = calculate_metrics(y_test, y_test_pred_rf, "Test", "RF")

# ============================================================================
# Generate detailed results
# ============================================================================

print("\n" + "="*80)
print("Step 10: Generate Detailed Results")
print("="*80)

def create_detailed_results(y_true, y_pred, set_name):
    error = y_pred - y_true.values
    abs_error = np.abs(error)
    percent_error = np.abs(error) / (np.abs(y_true.values) + 1e-10) * 100
    
    results_df = pd.DataFrame({
        'Predicted': y_pred,
        'Actual': y_true.values,
        'Error': error,
        'Abs_Error': abs_error,
        'Percent_Error': percent_error,
    })
    
    stats = {
        'Set': set_name,
        'Samples': len(y_true),
        'Pred_Max': y_pred.max(),
        'Pred_Min': y_pred.min(),
        'Pred_Mean': y_pred.mean(),
        'Pred_Range': y_pred.max() - y_pred.min(),
        'Actual_Max': y_true.max(),
        'Actual_Min': y_true.min(),
        'Actual_Mean': y_true.mean(),
        'Actual_Range': y_true.max() - y_true.min(),
        'Error_Mean': error.mean(),
        'Error_Max': error.max(),
        'Error_Min': error.min(),
        'Abs_Error_Mean': abs_error.mean(),
        'Abs_Error_Max': abs_error.max(),
        'Percent_Error_Mean': percent_error.mean(),
        'R2_Score': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
    }
    
    return results_df, stats

# SVR results
svr_train_results_df, svr_train_stats = create_detailed_results(y_train, y_train_pred_svr, "Train")
svr_val_results_df, svr_val_stats = create_detailed_results(y_val, y_val_pred_svr, "Val")
svr_test_results_df, svr_test_stats = create_detailed_results(y_test, y_test_pred_svr, "Test")

# RF results
rf_train_results_df, rf_train_stats = create_detailed_results(y_train, y_train_pred_rf, "Train")
rf_val_results_df, rf_val_stats = create_detailed_results(y_val, y_val_pred_rf, "Val")
rf_test_results_df, rf_test_stats = create_detailed_results(y_test, y_test_pred_rf, "Test")

print("OK - Detailed results generated")

# Feature importance (RF only)
print("\n" + "="*80)
print("Step 11: Feature Importance (RF)")
print("="*80)

rf_feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 20 Features (RF):")
for idx, row in rf_feature_importance.head(20).iterrows():
    print(f"  {row['Feature']:40s}: {row['Importance']:.4f}")

# Save results
print("\n" + "="*80)
print("Step 12: Save Results")
print("="*80)

os.makedirs('model_output_comparison', exist_ok=True)

# Save SVR model
with open('model_output_comparison/svr_model.pkl', 'wb') as f:
    pickle.dump(svr_model, f)
print("OK - SVR model saved")

# Save RF model
with open('model_output_comparison/rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("OK - RF model saved")

# Save scaler
with open('model_output_comparison/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("OK - Scaler saved")

# Save features
with open('model_output_comparison/features.pkl', 'wb') as f:
    pickle.dump(selected_features, f)
print("OK - Features saved")

# Save SVR results
svr_train_results_df.to_csv('model_output_comparison/svr_train_predictions.csv', index=False)
svr_val_results_df.to_csv('model_output_comparison/svr_val_predictions.csv', index=False)
svr_test_results_df.to_csv('model_output_comparison/svr_test_predictions.csv', index=False)

svr_stats_summary = pd.DataFrame([svr_train_stats, svr_val_stats, svr_test_stats])
svr_stats_summary.to_csv('model_output_comparison/svr_statistics_summary.csv', index=False)

# Save RF results
rf_train_results_df.to_csv('model_output_comparison/rf_train_predictions.csv', index=False)
rf_val_results_df.to_csv('model_output_comparison/rf_val_predictions.csv', index=False)
rf_test_results_df.to_csv('model_output_comparison/rf_test_predictions.csv', index=False)

rf_stats_summary = pd.DataFrame([rf_train_stats, rf_val_stats, rf_test_stats])
rf_stats_summary.to_csv('model_output_comparison/rf_statistics_summary.csv', index=False)

# Save RF feature importance
rf_feature_importance.to_csv('model_output_comparison/rf_feature_importance.csv', index=False)

print("OK - All results saved to model_output_comparison/")

# Visualization
print("\n" + "="*80)
print("Step 13: Visualization")
print("="*80)

# SVR scatter plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for results_df, stats, ax, title in [(svr_train_results_df, svr_train_stats, axes[0], "Train"),
                                      (svr_val_results_df, svr_val_stats, axes[1], "Val"),
                                      (svr_test_results_df, svr_test_stats, axes[2], "Test")]:
    ax.scatter(results_df['Actual'], results_df['Predicted'], alpha=0.3, s=1, color='steelblue')
    min_val = min(results_df['Actual'].min(), results_df['Predicted'].min())
    max_val = max(results_df['Actual'].max(), results_df['Predicted'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
    ax.set_xlabel('Actual', fontsize=11)
    ax.set_ylabel('Predicted', fontsize=11)
    ax.set_title(f'{title}\n(R2={stats["R2_Score"]:.4f}, MAE={stats["MAE"]:.4f})', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_output_comparison/svr_scatter_plot.png', dpi=300, bbox_inches='tight')
print("OK - SVR scatter plot saved")
plt.close()

# RF scatter plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for results_df, stats, ax, title in [(rf_train_results_df, rf_train_stats, axes[0], "Train"),
                                      (rf_val_results_df, rf_val_stats, axes[1], "Val"),
                                      (rf_test_results_df, rf_test_stats, axes[2], "Test")]:
    ax.scatter(results_df['Actual'], results_df['Predicted'], alpha=0.3, s=1, color='steelblue')
    min_val = min(results_df['Actual'].min(), results_df['Predicted'].min())
    max_val = max(results_df['Actual'].max(), results_df['Predicted'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
    ax.set_xlabel('Actual', fontsize=11)
    ax.set_ylabel('Predicted', fontsize=11)
    ax.set_title(f'{title}\n(R2={stats["R2_Score"]:.4f}, MAE={stats["MAE"]:.4f})', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_output_comparison/rf_scatter_plot.png', dpi=300, bbox_inches='tight')
print("OK - RF scatter plot saved")
plt.close()

# RF feature importance plot
fig, ax = plt.subplots(figsize=(12, 10))
top_features = rf_feature_importance.head(20)
ax.barh(top_features['Feature'], top_features['Importance'], color='steelblue')
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('Top 20 Feature Importance (RF)', fontsize=14, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('model_output_comparison/rf_feature_importance.png', dpi=300, bbox_inches='tight')
print("OK - RF feature importance plot saved")
plt.close()

# Model comparison plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

models = ['SVR', 'RF']
test_r2_scores = [svr_test_metrics['R2'], rf_test_metrics['R2']]
test_rmse_scores = [svr_test_metrics['RMSE'], rf_test_metrics['RMSE']]
test_mae_scores = [svr_test_metrics['MAE'], rf_test_metrics['MAE']]
test_mape_scores = [svr_test_metrics['MAPE'], rf_test_metrics['MAPE']]

ax = axes[0, 0]
ax.bar(models, test_r2_scores, color=['coral', 'steelblue'], alpha=0.8)
ax.set_ylabel('R2 Score', fontsize=11)
ax.set_title('Test R2 Score Comparison', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

ax = axes[0, 1]
ax.bar(models, test_rmse_scores, color=['coral', 'steelblue'], alpha=0.8)
ax.set_ylabel('RMSE', fontsize=11)
ax.set_title('Test RMSE Comparison', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1, 0]
ax.bar(models, test_mae_scores, color=['coral', 'steelblue'], alpha=0.8)
ax.set_ylabel('MAE', fontsize=11)
ax.set_title('Test MAE Comparison', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1, 1]
ax.bar(models, test_mape_scores, color=['coral', 'steelblue'], alpha=0.8)
ax.set_ylabel('MAPE (%)', fontsize=11)
ax.set_title('Test MAPE Comparison', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('model_output_comparison/model_comparison.png', dpi=300, bbox_inches='tight')
print("OK - Model comparison plot saved")
plt.close()

# Summary
print("\n" + "="*80)
print("Training Complete - Model Comparison")
print("="*80)

print(f"\nSVR Test Performance:")
print(f"  R2: {svr_test_metrics['R2']:.4f}")
print(f"  RMSE: {svr_test_metrics['RMSE']:.4f}")
print(f"  MAE: {svr_test_metrics['MAE']:.4f}")
print(f"  MAPE: {svr_test_metrics['MAPE']:.4f}%")

print(f"\nRF Test Performance:")
print(f"  R2: {rf_test_metrics['R2']:.4f}")
print(f"  RMSE: {rf_test_metrics['RMSE']:.4f}")
print(f"  MAE: {rf_test_metrics['MAE']:.4f}")
print(f"  MAPE: {rf_test_metrics['MAPE']:.4f}%")

print(f"\nBetter Model: {'SVR' if svr_test_metrics['R2'] > rf_test_metrics['R2'] else 'RF'}")
print(f"R2 Difference: {abs(svr_test_metrics['R2'] - rf_test_metrics['R2']):.4f}")

print(f"\nOutput directory: model_output_comparison/")

print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
