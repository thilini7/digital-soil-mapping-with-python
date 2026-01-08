#!/usr/bin/env python3
"""
HYBRID APPROACH: TabPFN accuracy + XGBoost/RF speed

Strategy:
1. Use TabPFN to predict on 100,000 random pixels (pseudo-labels)
2. Train XGBoost (or Random Forest) to mimic TabPFN
3. Use XGBoost/RF for full map generation (fast!)

This gives you TabPFN-quality predictions with XGBoost speed.

Run:
  export HF_TOKEN="your_token"
  python3 generate_map_hybrid.py
"""

import os
import sys
import pickle
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
import warnings
warnings.filterwarnings('ignore')

# Force unbuffered output for real-time progress
sys.stdout.reconfigure(line_buffering=True)
os.environ['PYTHONUNBUFFERED'] = '1'

# ============================================================================
# GPU SETUP
# ============================================================================
print("=" * 80, flush=True)
print("HYBRID MAP GENERATION: TabPFN → XGBoost/RF", flush=True)
print("=" * 80, flush=True)

print("\n📍 GPU Setup...", flush=True)
import torch
if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.empty_cache()
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✓ GPU Available: {gpu_name}")
    print(f"  GPU Memory: {gpu_mem:.1f} GB")
else:
    device = 'cpu'
    print("⚠️  No GPU - using CPU")

# ============================================================================
# PATHS
# ============================================================================
print("\n📍 Setting up paths...")

if os.path.exists('/home/ec2-user'):
    BASE_DIR = Path('/home/ec2-user/digital-soil-mapping')
    print("✓ Detected: AWS EC2 environment")
else:
    BASE_DIR = Path(__file__).resolve().parent.parent
    print(f"✓ Detected: Local environment")

DATA_DIR = BASE_DIR / 'data'
TRAIN_DATA_DIR = DATA_DIR / 'soil_with_raster'
COVARIATE_DIR = DATA_DIR / 'soil_covariates_aligned'
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = BASE_DIR / 'predictions' / 'spatial_outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"  Base: {BASE_DIR}")
print(f"  Data: {DATA_DIR}")
print(f"  Output: {OUTPUT_DIR}")

# ============================================================================
# STEP 1: LOAD TRAINING DATA
# ============================================================================
print("\n📍 Step 1: Loading training data...")

csv_path = TRAIN_DATA_DIR / 'pH_with_covariates_new.csv'

df = pd.read_csv(csv_path)
print(f"✓ Loaded {len(df)} training samples")

# Define features (exclude target and non-feature columns)
target_col = 'X0.5cm'
depth_cols = ['X0.5cm', 'X5.15cm', 'X15.30cm', 'X30.60cm', 'X60.100cm', 'X100.200cm']
exclude_cols = depth_cols + ['id', 'Unnamed: 0', 'index', 'X', 'Y', 'Longitude', 'Latitude',
                              'longitude', 'latitude', 'lon', 'lat', 'FID',
                              'C_O_VegetationWater', 'C_R_TempRelief', 'R_P_WeatheringSlope',
                              'C_AridityIndex', 'R_WetnessDrainage']

feature_cols = [c for c in df.columns if c not in exclude_cols 
                and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

X_original = df[feature_cols].values
y_original = df[target_col].values

# Remove NaN targets
valid_mask = ~np.isnan(y_original)
X_original = X_original[valid_mask]
y_original = y_original[valid_mask]

# Filter data 3.5 to 9 (remove outliers for pH)
data_mask =  (y_original <= 9) & (y_original >= 3.5)
print(f"  Filtering Data 3.5-9: {data_mask.sum()}/{len(y_original)} samples retained")
X_original = X_original[data_mask]
y_original = y_original[data_mask]

print(f"  Features: {len(feature_cols)}")
print(f"  Valid samples: {len(y_original)}")
print(f"  Target range: [{y_original.min():.2f}, {y_original.max():.2f}]")

# VERIFICATION: Confirm target column is NOT in features
print(f"\n  🔍 VERIFICATION:")
print(f"     Target column: '{target_col}'")
print(f"     Target in features? {'❌ YES - DATA LEAKAGE!' if target_col in feature_cols else '✅ NO - Correctly excluded'}")
print(f"     Excluded columns ({len(exclude_cols)}): {exclude_cols[:5]}...")
print(f"     Feature columns: {feature_cols[:5]}...")

# ============================================================================
# STEP 2: FIT PREPROCESSING
# ============================================================================
print("\n📍 Step 2: Fitting preprocessing pipeline...")

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, StandardScaler

imputer = SimpleImputer(strategy='median')
power_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
scaler = StandardScaler()

X_imputed = imputer.fit_transform(X_original)
X_power = power_transformer.fit_transform(X_imputed)
X_train = scaler.fit_transform(X_power)

print(f"✓ Preprocessing fitted")
print(f"  X_train shape: {X_train.shape}")

# ============================================================================
# STEP 3: FIT TABPFN ON TRAINING DATA
# ============================================================================
print("\n📍 Step 3: Fitting TabPFN model...")

from tabpfn import TabPFNRegressor

tabpfn_model = TabPFNRegressor(
    device=device,
    n_estimators=4,  # Balance of speed and accuracy
    random_state=42,
    ignore_pretraining_limits=True,  # Allow CPU with large datasets
)

tabpfn_model.fit(X_train, y_original)
print(f"✓ TabPFN fitted on {len(y_original)} samples")

# Quick validation
val_pred = tabpfn_model.predict(X_train[:10])
print(f"  Validation predictions: {val_pred[:5]}")

# ============================================================================
# STEP 4: FIND AND MAP TIF FILES
# ============================================================================
print("\n📍 Step 4: Mapping features to TIF files...")

import rasterio

# Find TIF files
tif_dirs = [COVARIATE_DIR, DATA_DIR]
tif_files = []
for tdir in tif_dirs:
    if tdir.exists():
        tif_files.extend(list(tdir.glob("*.tif")))
        tif_files.extend(list(tdir.glob("*.tiff")))

print(f"  Found {len(tif_files)} TIF files")

# Map features to TIF files
feature_to_tif = {}
for feat in feature_cols:
    for tif in tif_files:
        tif_name = tif.stem
        if feat == tif_name or feat in tif_name or tif_name in feat:
            feature_to_tif[feat] = tif
            break

print(f"✓ Mapped {len(feature_to_tif)}/{len(feature_cols)} features to TIF files")

if len(feature_to_tif) < len(feature_cols):
    missing = set(feature_cols) - set(feature_to_tif.keys())
    print(f"  ⚠️  Missing TIFs for: {list(missing)[:5]}...")
    # Use only features with TIFs
    feature_cols = list(feature_to_tif.keys())
    print(f"  → Using {len(feature_cols)} features with matching TIFs")
    
    # Refit preprocessing with reduced features
    feature_indices = [i for i, c in enumerate(df.columns) if c in feature_cols]
    X_original = df[feature_cols].values[valid_mask]
    X_imputed = imputer.fit_transform(X_original)
    X_power = power_transformer.fit_transform(X_imputed)
    X_train = scaler.fit_transform(X_power)
    
    # Refit TabPFN
    print("  → Refitting TabPFN with matched features...")
    tabpfn_model = TabPFNRegressor(device=device, n_estimators=4, random_state=42, ignore_pretraining_limits=True)
    tabpfn_model.fit(X_train, y_original)

# Get raster dimensions
first_tif = list(feature_to_tif.values())[0]
with rasterio.open(first_tif) as src:
    height, width = src.height, src.width
    profile = src.profile.copy()
    nodata = src.nodata if src.nodata is not None else np.nan
    crs = src.crs
    transform = src.transform

print(f"✓ Raster dimensions: {height:,} × {width:,} = {height * width:,} pixels")
print(f"  CRS: {crs}")

# ============================================================================
# STEP 5: GENERATE PSEUDO-LABELS FROM RANDOM PIXELS
# ============================================================================
print("\n📍 Step 5: Sampling random pixels for pseudo-labeling...")

N_PSEUDO_SAMPLES = 100000  # 100k samples for pseudo-labeling

np.random.seed(42)
random_rows = np.random.randint(0, height, N_PSEUDO_SAMPLES)
random_cols = np.random.randint(0, width, N_PSEUDO_SAMPLES)

# Read values at random pixels
pseudo_X = np.zeros((N_PSEUDO_SAMPLES, len(feature_cols)), dtype=np.float32)

print(f"  Reading {len(feature_cols)} features for {N_PSEUDO_SAMPLES:,} pixels...")
for i, feat in enumerate(feature_cols):
    tif_path = feature_to_tif[feat]
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        pseudo_X[:, i] = data[random_rows, random_cols]
    if (i + 1) % 10 == 0:
        print(f"    Read {i + 1}/{len(feature_cols)} features...", end='\r')

print(f"\n✓ Read {len(feature_cols)} features")

# Remove invalid pixels (nodata)
valid_mask = ~np.any(np.isnan(pseudo_X), axis=1) & ~np.any(np.isinf(pseudo_X), axis=1)
if nodata is not None and not np.isnan(nodata):
    valid_mask &= ~np.any(pseudo_X == nodata, axis=1)

pseudo_X = pseudo_X[valid_mask]
print(f"✓ Valid pixels after filtering: {len(pseudo_X):,}")

# Preprocess
pseudo_X_imputed = imputer.transform(pseudo_X)
pseudo_X_power = power_transformer.transform(pseudo_X_imputed)
pseudo_X_scaled = scaler.transform(pseudo_X_power)

# ============================================================================
# STEP 6: PREDICT WITH TABPFN (PSEUDO-LABELS)
# ============================================================================
print("\n📍 Step 6: Generating pseudo-labels with TabPFN...")

BATCH_SIZE = 5000
n_samples = len(pseudo_X_scaled)
pseudo_y = np.zeros(n_samples, dtype=np.float32)

for start in range(0, n_samples, BATCH_SIZE):
    end = min(start + BATCH_SIZE, n_samples)
    batch_X = pseudo_X_scaled[start:end]
    
    with torch.no_grad():
        pseudo_y[start:end] = tabpfn_model.predict(batch_X)
    
    # Progress
    pct = (end / n_samples) * 100
    print(f"  TabPFN prediction: {pct:.1f}% ({end:,}/{n_samples:,})", end='\r')
    
    # Memory cleanup every 10 batches
    if (start // BATCH_SIZE) % 10 == 0:
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

print(f"\n✓ Generated {len(pseudo_y):,} pseudo-labels")
print(f"  Pseudo-label range: [{pseudo_y.min():.2f}, {pseudo_y.max():.2f}]")
print(f"  Pseudo-label mean: {pseudo_y.mean():.2f}")

# Free TabPFN memory
del tabpfn_model
if device == 'cuda':
    torch.cuda.empty_cache()
gc.collect()
print("✓ Released TabPFN memory")

# ============================================================================
# STEP 7: TRAIN XGBOOST ON PSEUDO-LABELS (More Accurate than RF)
# ============================================================================
print("\n📍 Step 7: Training model on pseudo-labels...")

# Try to import XGBoost, fall back to sklearn if not available
try:
    import xgboost as xgb
    USE_XGBOOST = True
    print("  ✓ Using XGBoost (more accurate)")
except ImportError:
    USE_XGBOOST = False
    print("  ⚠️ XGBoost not installed, using Random Forest")
    print("  → Install with: pip install xgboost")

# Combine original training data with pseudo-labels
X_combined = np.vstack([X_train, pseudo_X_scaled])
y_combined = np.concatenate([y_original, pseudo_y])

print(f"  Combined training set: {len(y_combined):,} samples")
print(f"    - Original: {len(y_original):,}")
print(f"    - Pseudo-labels: {len(pseudo_y):,}")

if USE_XGBOOST:
    # Check if GPU is available for XGBoost
    try:
        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method='hist',
            device='cuda',  # GPU
            random_state=42,
            verbosity=1,
            n_jobs=-1
        )
        # Quick test to check GPU works
        model.fit(X_train[:10], y_original[:10])
        print("  ✓ XGBoost GPU mode enabled")
    except Exception as e:
        print(f"  ⚠️ GPU failed: {e}")
        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            tree_method='hist',
            random_state=42,
            verbosity=1,
            n_jobs=-1
        )
        print("  ✓ XGBoost CPU mode")
else:
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

print("  Training model (this may take a few minutes)...")
model.fit(X_combined, y_combined)
print(f"✓ Model trained")

# Validate on original data
model_pred = model.predict(X_train)
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
r2 = r2_score(y_original, model_pred)
rmse = np.sqrt(mean_squared_error(y_original, model_pred))
mae = mean_absolute_error(y_original, model_pred)

print(f"\n  📊 Training Fit Metrics (overly optimistic - NOT for publication):")
print(f"     R²:   {r2:.4f}")
print(f"     RMSE: {rmse:.4f}")
print(f"     MAE:  {mae:.4f}")

# ============================================================================
# CROSS-VALIDATION (For Publication)
# ============================================================================
print(f"\n  📊 Cross-Validation Metrics (5-fold, USE FOR PUBLICATION):")
from sklearn.model_selection import cross_val_predict, KFold

# Use TabPFN for CV since that's the primary model
print("     Running 5-fold cross-validation on TabPFN...")

# Recreate TabPFN for CV
from tabpfn import TabPFNRegressor
tabpfn_cv = TabPFNRegressor(device=device, n_estimators=4, random_state=42, ignore_pretraining_limits=True)

# 5-fold CV predictions
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_predictions = np.zeros(len(y_original))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train = y_original[train_idx]
    
    tabpfn_cv.fit(X_fold_train, y_fold_train)
    cv_predictions[val_idx] = tabpfn_cv.predict(X_fold_val)
    print(f"       Fold {fold+1}/5 complete", end='\r')

# Calculate CV metrics
cv_r2 = r2_score(y_original, cv_predictions)
cv_rmse = np.sqrt(mean_squared_error(y_original, cv_predictions))
cv_mae = mean_absolute_error(y_original, cv_predictions)

# Concordance Correlation Coefficient (Lin's CCC)
def concordance_correlation(y_true, y_pred):
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred)**2)
    return ccc

cv_ccc = concordance_correlation(y_original, cv_predictions)

print(f"\n     ✅ 5-Fold Cross-Validation Results:")
print(f"        R²:   {cv_r2:.4f}")
print(f"        RMSE: {cv_rmse:.4f}")
print(f"        MAE:  {cv_mae:.4f}")
print(f"        CCC:  {cv_ccc:.4f} (Concordance Correlation)")

# Clean up CV model
del tabpfn_cv
gc.collect()

# FINAL VERIFICATION: Model uses correct number of features
print(f"\n  🔍 FINAL VERIFICATION:")
print(f"     Model input features: {model.n_features_in_ if hasattr(model, 'n_features_in_') else len(feature_cols)}")
print(f"     Expected features: {len(feature_cols)}")
print(f"     Target '{target_col}' in feature_cols? {'❌ LEAK!' if target_col in feature_cols else '✅ NO'}")

# Feature importance (if XGBoost)
if USE_XGBOOST:
    importance = model.feature_importances_
    top_features = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n  📊 Top 10 Important Features:")
    for feat, imp in top_features:
        print(f"     {feat}: {imp:.4f}")

# Save the model
model_name = 'xgboost' if USE_XGBOOST else 'random_forest'
model_path = MODELS_DIR / f'{model_name}_distilled_from_tabpfn.pkl'
with open(model_path, 'wb') as f:
    pickle.dump({
        'model': model,
        'imputer': imputer,
        'power_transformer': power_transformer,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'model_type': model_name,
        'metrics': {'r2': r2, 'rmse': rmse, 'mae': mae},
        'cv_metrics': {'r2': cv_r2, 'rmse': cv_rmse, 'mae': cv_mae, 'ccc': cv_ccc}
    }, f)
print(f"\n✓ Saved {model_name} model to: {model_path}")

# ============================================================================
# STEP 8: GENERATE FULL MAP WITH MODEL
# ============================================================================
print("\n📍 Step 8: Generating full spatial map...")

from tqdm import tqdm

# Output file
output_suffix = 'xgb' if USE_XGBOOST else 'rf'
output_path = OUTPUT_DIR / f'pH_NSW_ACT_0_5cm_hybrid_{output_suffix}.tif'

# Update profile for output
profile.update(
    dtype='float32',
    count=1,
    compress='lzw',
    nodata=np.nan,
    bigtiff='YES'
)

# Tile configuration - larger tiles for RF (it's fast!)
TILE_SIZE = 512
n_tiles_x = (width + TILE_SIZE - 1) // TILE_SIZE
n_tiles_y = (height + TILE_SIZE - 1) // TILE_SIZE
total_tiles = n_tiles_x * n_tiles_y

print(f"  Output: {output_path}")
print(f"  Tile size: {TILE_SIZE} × {TILE_SIZE}")
print(f"  Total tiles: {n_tiles_x} × {n_tiles_y} = {total_tiles:,}")

# Open all rasters
rasters = {}
for feat in feature_cols:
    rasters[feat] = rasterio.open(feature_to_tif[feat])

# Process tiles
with rasterio.open(output_path, 'w', **profile) as dst:
    with tqdm(total=total_tiles, desc="Generating map", unit="tile") as pbar:
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                # Tile bounds
                row_start = ty * TILE_SIZE
                row_end = min(row_start + TILE_SIZE, height)
                col_start = tx * TILE_SIZE
                col_end = min(col_start + TILE_SIZE, width)
                
                tile_height = row_end - row_start
                tile_width = col_end - col_start
                n_pixels = tile_height * tile_width
                
                # Read tile data
                tile_data = np.zeros((n_pixels, len(feature_cols)), dtype=np.float32)
                
                window = rasterio.windows.Window(col_start, row_start, tile_width, tile_height)
                
                for i, feat in enumerate(feature_cols):
                    data = rasters[feat].read(1, window=window)
                    tile_data[:, i] = data.flatten()
                
                # Find valid pixels
                valid_mask = ~np.any(np.isnan(tile_data), axis=1) & ~np.any(np.isinf(tile_data), axis=1)
                if nodata is not None and not np.isnan(nodata):
                    valid_mask &= ~np.any(tile_data == nodata, axis=1)
                
                # Initialize output tile
                output_tile = np.full(n_pixels, np.nan, dtype=np.float32)
                
                if valid_mask.sum() > 0:
                    # Preprocess valid pixels
                    valid_data = tile_data[valid_mask]
                    valid_imputed = imputer.transform(valid_data)
                    valid_power = power_transformer.transform(valid_imputed)
                    valid_scaled = scaler.transform(valid_power)
                    
                    # Predict with model (XGBoost or RF - FAST!)
                    predictions = model.predict(valid_scaled)
                    output_tile[valid_mask] = predictions
                
                # Reshape and write
                output_tile = output_tile.reshape(tile_height, tile_width)
                dst.write(output_tile, 1, window=window)
                
                pbar.update(1)

# Close rasters
for r in rasters.values():
    r.close()

# ============================================================================
# DONE
# ============================================================================
file_size = output_path.stat().st_size / (1024**3)
print(f"\n" + "=" * 80)
print(f"✅ HYBRID MAP GENERATED SUCCESSFULLY!")
print(f"=" * 80)
print(f"   Model: {model_name}")
print(f"   Output: {output_path}")
print(f"   Size: {file_size:.2f} GB")
print(f"\n   📊 Cross-Validation Metrics (for publication):")
print(f"      R²:   {cv_r2:.4f}")
print(f"      RMSE: {cv_rmse:.4f}")
print(f"      MAE:  {cv_mae:.4f}")
print(f"      CCC:  {cv_ccc:.4f}")
