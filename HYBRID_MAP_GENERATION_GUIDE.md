# Hybrid Map Generation Guide

## Understanding the `generate_map_hybrid.py` Script

**Purpose:** Generate high-resolution soil property maps using a hybrid approach that combines TabPFN accuracy with XGBoost speed.

---

## Overview: The Hybrid Approach

```
┌─────────────────────────────────────────────────────────────────┐
│                    HYBRID WORKFLOW                               │
├─────────────────────────────────────────────────────────────────┤
│  Training Data → TabPFN (accurate but slow)                     │
│       ↓                                                          │
│  Random Pixels → TabPFN predictions (pseudo-labels)             │
│       ↓                                                          │
│  XGBoost learns from pseudo-labels (knowledge distillation)     │
│       ↓                                                          │
│  XGBoost generates full map (fast!)                             │
└─────────────────────────────────────────────────────────────────┘
```

**Why Hybrid?**
- TabPFN: Very accurate but slow (~1000 predictions/second)
- XGBoost: Fast (~1 million predictions/second) but needs good training data
- Solution: Use TabPFN to create "pseudo-labels" for XGBoost training

---

## Step-by-Step Explanation

### Step 1: Load Training Data

**What happens:**
- Load CSV file with soil samples and covariate values
- Define target variable: `X0.5cm` (0-5cm depth Organic Carbon)
- Identify feature columns (environmental covariates)
- Filter outliers: Remove OC values > 10%

**Key code:**
```python
df = pd.read_csv('OC_with_covariates_new.csv')
target_col = 'X0.5cm'
feature_cols = [c for c in df.columns if c not in exclude_cols]
```

**Output:**
- ~1000 training samples
- ~30 environmental features (climate, terrain, radiometrics, vegetation)

---

### Step 2: Fit Preprocessing Pipeline

**What happens:**
Three-stage preprocessing to normalize data:

| Stage | Method | Purpose |
|-------|--------|---------|
| 1. Imputation | Median | Fill missing values |
| 2. Power Transform | Yeo-Johnson | Make distributions more Gaussian |
| 3. Scaling | StandardScaler | Zero mean, unit variance |

**Why preprocessing?**
- Machine learning models work better with normalized data
- Handles missing values consistently
- Same preprocessing applied to training data AND raster pixels

**Key code:**
```python
imputer = SimpleImputer(strategy='median')
power_transformer = PowerTransformer(method='yeo-johnson')
scaler = StandardScaler()

X_train = scaler.fit_transform(
    power_transformer.fit_transform(
        imputer.fit_transform(X_original)
    )
)
```

---

### Step 3: Fit TabPFN Model

**What happens:**
- Train TabPFN (Tabular Prior-Data Fitted Network) on preprocessed data
- TabPFN is a transformer-based model pretrained on synthetic tabular data
- Requires GPU for optimal performance

**Key parameters:**
```python
TabPFNRegressor(
    device='cuda',      # Use GPU
    n_estimators=4,     # Ensemble size (balance speed/accuracy)
    random_state=42     # Reproducibility
)
```

**Why TabPFN?**
- State-of-the-art accuracy on small tabular datasets
- No hyperparameter tuning needed
- Works well with <10,000 samples

---

### Step 4: Map Features to TIF Files

**What happens:**
- Find all GeoTIFF raster files in the covariate directory
- Match each feature column name to its corresponding raster file
- Extract raster metadata (dimensions, CRS, nodata value)

**Key matching logic:**
```python
for feat in feature_cols:
    for tif in tif_files:
        if feat == tif.stem or feat in tif.stem:
            feature_to_tif[feat] = tif
```

**Output:**
- Mapping: `{'Mean_Annual_RF_NSW_ACT_90m': '/path/to/rainfall.tif', ...}`
- Raster dimensions: e.g., 10,000 × 8,000 pixels = 80 million pixels

---

### Step 5: Sample Random Pixels for Pseudo-Labeling

**What happens:**
- Randomly select 100,000 pixel locations from the raster
- Read covariate values at those locations from all TIF files
- Filter out invalid pixels (nodata, NaN, Inf)

**Why random sampling?**
- 80 million pixels is too many for TabPFN
- 100,000 samples captures the covariate space well
- Represents the full range of environmental conditions

**Key code:**
```python
N_PSEUDO_SAMPLES = 100000
random_rows = np.random.randint(0, height, N_PSEUDO_SAMPLES)
random_cols = np.random.randint(0, width, N_PSEUDO_SAMPLES)

for feat in feature_cols:
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        pseudo_X[:, i] = data[random_rows, random_cols]
```

---

### Step 6: Generate Pseudo-Labels with TabPFN

**What happens:**
- Apply same preprocessing to random pixels
- Use TabPFN to predict OC values for all 100,000 pixels
- These predictions become "pseudo-labels" (synthetic training data)

**Batch processing:**
```python
BATCH_SIZE = 5000  # Predict in chunks to manage memory

for start in range(0, n_samples, BATCH_SIZE):
    pseudo_y[start:end] = tabpfn_model.predict(batch_X)
```

**Why pseudo-labels?**
- Creates a large training set for the fast model
- Transfers TabPFN's knowledge to XGBoost
- This is called "knowledge distillation"

---

### Step 7: Train XGBoost on Pseudo-Labels

**What happens:**
- Combine original training data with pseudo-labeled data
- Train XGBoost regressor on the combined dataset
- Run 5-fold cross-validation for publication metrics

**Combined training set:**
```
Original samples:     ~1,000 (with real OC measurements)
Pseudo-labeled:     ~100,000 (TabPFN predictions)
Total:              ~101,000 samples
```

**XGBoost parameters:**
```python
xgb.XGBRegressor(
    n_estimators=500,      # Number of trees
    max_depth=8,           # Tree depth
    learning_rate=0.05,    # Step size
    subsample=0.8,         # Sample fraction per tree
    colsample_bytree=0.8,  # Feature fraction per tree
    tree_method='hist',    # Fast histogram-based algorithm
    device='cuda'          # GPU acceleration
)
```

**Cross-Validation Metrics (for publication):**

| Metric | Description |
|--------|-------------|
| R² | Coefficient of determination (0-1, higher is better) |
| RMSE | Root Mean Square Error (lower is better) |
| MAE | Mean Absolute Error (lower is better) |
| CCC | Concordance Correlation Coefficient (agreement measure) |

---

### Step 8: Generate Full Spatial Map

**What happens:**
- Process the entire raster in tiles (512×512 pixels)
- For each tile: read covariates, preprocess, predict with XGBoost
- Write predictions to output GeoTIFF

**Tile-based processing:**
```
┌──────┬──────┬──────┬──────┐
│Tile 1│Tile 2│Tile 3│Tile 4│
├──────┼──────┼──────┼──────┤
│Tile 5│Tile 6│Tile 7│Tile 8│
├──────┼──────┼──────┼──────┤
│ ...  │ ...  │ ...  │ ...  │
└──────┴──────┴──────┴──────┘
```

**Why tiles?**
- Can't load entire raster into memory (80M pixels × 30 features = huge!)
- Tiles allow processing on machines with limited RAM
- Progress can be tracked and resumed

**Key code:**
```python
for ty in range(n_tiles_y):
    for tx in range(n_tiles_x):
        # Read tile data
        tile_data = read_from_rasters(window)
        
        # Preprocess
        tile_scaled = scaler.transform(
            power_transformer.transform(
                imputer.transform(tile_data)
            )
        )
        
        # Predict
        predictions = model.predict(tile_scaled)
        
        # Write to output
        dst.write(predictions, window)
```

---

## Output Files

| File | Description |
|------|-------------|
| `OC_NSW_ACT_0_5cm_hybrid_xgb.tif` | Final prediction map (GeoTIFF) |
| `xgboost_distilled_from_tabpfn.pkl` | Saved model + preprocessing pipeline |

---

## Data Flow Diagram

```
INPUT DATA
    │
    ├── Training CSV (soil samples + covariates)
    │        │
    │        ▼
    │   [Preprocessing: Impute → PowerTransform → Scale]
    │        │
    │        ▼
    │   [TabPFN Training]
    │        │
    │        ▼
    │   [Pseudo-Label Generation] ◄── Random Raster Pixels
    │        │
    │        ▼
    │   [XGBoost Training on Combined Data]
    │        │
    │        ▼
    └── Covariate TIF Files
             │
             ▼
        [Tile-by-Tile Prediction]
             │
             ▼
        OUTPUT MAP (GeoTIFF)
```

---

## Key Concepts

### 1. Knowledge Distillation
Training a simpler model (XGBoost) to mimic a complex model (TabPFN).

### 2. Pseudo-Labeling
Using model predictions as training labels for unlabeled data.

### 3. Feature Engineering
Environmental covariates used as predictors:
- **Climate:** Rainfall, Temperature, Evapotranspiration
- **Terrain:** DEM, Slope, TWI, Curvature, MRVBF
- **Radiometrics:** K, U, Th concentrations and ratios
- **Vegetation:** FPAR (photosynthetic activity)
- **Geology:** Weathering index, Gravity

### 4. Preventing Data Leakage
Target variable (`X0.5cm`) is explicitly excluded from features.

---

## Performance Comparison

| Model | Prediction Speed | Accuracy | Use Case |
|-------|-----------------|----------|----------|
| TabPFN | ~1,000/sec | Highest | Small samples, training |
| XGBoost | ~1,000,000/sec | High | Full map generation |
| Hybrid | Best of both | High | Production mapping |

---

## Requirements

```
Python packages:
- numpy, pandas
- scikit-learn
- tabpfn
- xgboost
- rasterio
- torch (with CUDA for GPU)
- tqdm

Hardware:
- GPU recommended (NVIDIA with CUDA)
- 16+ GB RAM
- Storage for output (~1-3 GB per map)
```

---

## Running the Script

```bash
# Activate environment
source .venv/bin/activate

# Run script
python scripts/generate_map_hybrid.py
```

**Expected runtime:**
- Step 1-6 (TabPFN): ~10-30 minutes
- Step 7 (XGBoost training): ~5-10 minutes
- Step 8 (Map generation): ~30-60 minutes

---

## Summary

The hybrid approach provides:
1. **TabPFN-quality predictions** - state-of-the-art accuracy
2. **XGBoost speed** - process millions of pixels efficiently
3. **Full spatial coverage** - complete NSW/ACT soil property maps
4. **Reproducible results** - saved models and preprocessing pipelines
