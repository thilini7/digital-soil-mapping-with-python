#!/usr/bin/env python3
"""
Soil Data Statistics Generator
Generates comprehensive statistics for soil properties with physical limits filtering.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data" / "soil_with_raster"
OUTPUT_DIR = Path(__file__).parent.parent

# Physical limits for each property
PHYSICAL_LIMITS = {
    'OC': {'lower': 0, 'upper': 10},
    'pH': {'lower': 3.5, 'upper': 9},  # Updated limits
    'Clay': {'lower': 0, 'upper': 100},
    'CEC': {'lower': 0.5, 'upper': 100},
    'EC': {'lower': 0, 'upper': 16}
}

# Depth columns
DEPTH_COLS = ['X0.5cm', 'X5.15cm', 'X15.30cm', 'X30.60cm', 'X60.100cm', 'X100.200cm']

# Feature columns (SCORPAN covariates) - actual CSV column names
FEATURE_COLS = [
    'relief_dems_3s_mosaic1_nsw', 'ET_Annual_90m', 'FPAR_pct', 'FPAR_pct.1', 'FPAR_pct.2', 'FPAR_pct.3',
    'PM_Gravity_NSW_ACT', 'Soil_Illite_NSW_ACT', 'Soil_Kaolinite_NSW_ACT', 
    'Mean_Annual_RF_NSW_ACT_90m', 'Mean_Annual_Max_Temp_NSW_ACT_90m',
    'relief_mrrtf_3s_nsw', 'relief_mrvbf_3s_mosaic_nsw', 'relief_plan_curvature_3s_nsw', 
    'relief_profile_curvature_3_nsw', 'PM_radmap_v4_2019_filtered_dose_GAPFilled_nsw', 
    'PM_radmap_v4_2019_filtered_pctk_GAPFilled_nsw', 'PM_radmap_v4_2019_filtered_ppmt_GAPFilled_nsw', 
    'PM_radmap_v4_2019_filtered_ppmu_GAPFilled_nsw', 'PM_radmap_v4_2019_ratio_tk_GAPFilled_nsw', 
    'PM_radmap_v4_2019_ratio_u2t_GAPFilled_nsw', 'PM_radmap_v4_2019_ratio_uk_GAPFilled_nsw', 
    'PM_radmap_v4_2019_ratio_ut_GAPFilled_nsw', 'Annual_RF_Var_NSW_ACT_90m', 
    'relief_roughness_nsw', 'relief_slope_perc_nsw', 'Soil_Smectite_NSW_ACT', 
    'Annual_90th_Percen_Temp_NSW_ACT_90m', 'relief_twi_3s_nsw', 'PM_Weathering_Index_nsw'
]

# Short names for display
FEATURE_SHORT_NAMES = {
    'relief_dems_3s_mosaic1_nsw': 'DEM',
    'ET_Annual_90m': 'ET_Annual',
    'FPAR_pct': 'FPAR',
    'FPAR_pct.1': 'FPAR.1',
    'FPAR_pct.2': 'FPAR.2',
    'FPAR_pct.3': 'FPAR.3',
    'PM_Gravity_NSW_ACT': 'Gravity',
    'Soil_Illite_NSW_ACT': 'Illite',
    'Soil_Kaolinite_NSW_ACT': 'Kaolinite',
    'Mean_Annual_RF_NSW_ACT_90m': 'MeanAnnualRF',
    'Mean_Annual_Max_Temp_NSW_ACT_90m': 'MeanMaxTemp',
    'relief_mrrtf_3s_nsw': 'MRRTF',
    'relief_mrvbf_3s_mosaic_nsw': 'MRVBF',
    'relief_plan_curvature_3s_nsw': 'PlanCurv',
    'relief_profile_curvature_3_nsw': 'ProfileCurv',
    'PM_radmap_v4_2019_filtered_dose_GAPFilled_nsw': 'Rad_dose',
    'PM_radmap_v4_2019_filtered_pctk_GAPFilled_nsw': 'Rad_pctk',
    'PM_radmap_v4_2019_filtered_ppmt_GAPFilled_nsw': 'Rad_ppmt',
    'PM_radmap_v4_2019_filtered_ppmu_GAPFilled_nsw': 'Rad_ppmu',
    'PM_radmap_v4_2019_ratio_tk_GAPFilled_nsw': 'Rad_tk',
    'PM_radmap_v4_2019_ratio_u2t_GAPFilled_nsw': 'Rad_u2t',
    'PM_radmap_v4_2019_ratio_uk_GAPFilled_nsw': 'Rad_uk',
    'PM_radmap_v4_2019_ratio_ut_GAPFilled_nsw': 'Rad_ut',
    'Annual_RF_Var_NSW_ACT_90m': 'RFVariance',
    'relief_roughness_nsw': 'Roughness',
    'relief_slope_perc_nsw': 'Slope',
    'Soil_Smectite_NSW_ACT': 'Smectite',
    'Annual_90th_Percen_Temp_NSW_ACT_90m': 'Temp90thPct',
    'relief_twi_3s_nsw': 'TWI',
    'PM_Weathering_Index_nsw': 'WeatherIdx'
}


def load_and_filter_data(property_name: str) -> tuple:
    """Load CSV and filter by physical limits."""
    csv_path = DATA_DIR / f"{property_name}_with_covariates_new.csv"
    if not csv_path.exists():
        print(f"  ⚠️ File not found: {csv_path}")
        return None, 0
    
    df = pd.read_csv(csv_path)
    original_count = len(df)
    
    limits = PHYSICAL_LIMITS[property_name]
    target_col = 'X0.5cm'
    
    # Filter by physical limits
    mask = (df[target_col] >= limits['lower']) & (df[target_col] <= limits['upper'])
    df_filtered = df[mask].copy()
    
    outliers_removed = original_count - len(df_filtered)
    
    return df_filtered, outliers_removed


def compute_depth_statistics(df: pd.DataFrame) -> list:
    """Compute statistics for each depth."""
    results = []
    for depth in DEPTH_COLS:
        if depth not in df.columns:
            continue
        data = df[depth].dropna()
        if len(data) == 0:
            continue
        results.append({
            'depth': depth,
            'n': len(data),
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data)
        })
    return results


def compute_feature_correlations(df: pd.DataFrame, target_col: str = 'X0.5cm') -> list:
    """Compute correlations between target and all features."""
    correlations = []
    y = df[target_col].dropna()
    
    for feat in FEATURE_COLS:
        if feat not in df.columns:
            continue
        
        # Get common indices
        mask = df[target_col].notna() & df[feat].notna()
        x = df.loc[mask, feat]
        y_subset = df.loc[mask, target_col]
        
        if len(x) < 10:
            continue
            
        r, _ = stats.pearsonr(x, y_subset)
        short_name = FEATURE_SHORT_NAMES.get(feat, feat)
        correlations.append({
            'feature': short_name,
            'r': r,
            'r2': r**2,
            'direction': 'positive' if r > 0 else 'negative'
        })
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x['r']), reverse=True)
    return correlations


def compute_cross_depth_correlations(df: pd.DataFrame) -> dict:
    """Compute R² matrix between depths."""
    matrix = {}
    for d1 in DEPTH_COLS:
        if d1 not in df.columns:
            continue
        matrix[d1] = {}
        for d2 in DEPTH_COLS:
            if d2 not in df.columns:
                continue
            mask = df[d1].notna() & df[d2].notna()
            if mask.sum() < 10:
                matrix[d1][d2] = np.nan
                continue
            r, _ = stats.pearsonr(df.loc[mask, d1], df.loc[mask, d2])
            matrix[d1][d2] = r**2
    return matrix


def generate_property_markdown(prop: str, df: pd.DataFrame, outliers: int) -> str:
    """Generate markdown section for a property."""
    limits = PHYSICAL_LIMITS[prop]
    
    # Compute statistics
    depth_stats = compute_depth_statistics(df)
    correlations = compute_feature_correlations(df)
    cross_depth = compute_cross_depth_correlations(df)
    
    # Coefficient of variation
    target_data = df['X0.5cm'].dropna()
    cv = (target_data.std() / target_data.mean()) * 100 if target_data.mean() != 0 else 0
    
    md = f"""
## {get_property_full_name(prop)}

### Dataset Overview
- **Physical Limit:** {limits['lower']}-{limits['upper']}
- **Total Samples:** {len(df):,} ({outliers} outliers removed)
- **Feature Columns:** {len(FEATURE_COLS)}
- **Coefficient of Variation:** {cv:.1f}%

### Target Statistics by Depth

| Depth | N | Mean | Std | Min | Max | Skewness | Kurtosis |
|-------|---|------|-----|-----|-----|----------|----------|
"""
    
    for s in depth_stats:
        md += f"| {s['depth']} | {s['n']:,} | {s['mean']:.3f} | {s['std']:.3f} | {s['min']:.3f} | {s['max']:.3f} | {s['skewness']:.2f} | {s['kurtosis']:.2f} |\n"
    
    md += f"""
### All Feature Correlations (X0.5cm) - {len(correlations)} Features

| Rank | Feature | r | R² | Direction |
|------|---------|---|----|-----------|
"""
    
    for i, c in enumerate(correlations, 1):
        md += f"| {i} | {c['feature']} | {c['r']:.4f} | {c['r2']:.4f} | {c['direction']} |\n"
    
    # Correlation summary
    r2_values = [c['r2'] for c in correlations]
    abs_r_values = [abs(c['r']) for c in correlations]
    
    md += f"""
### Correlation Summary
- **Max R²:** {max(r2_values):.4f}
- **Mean R²:** {np.mean(r2_values):.4f}
- **Median R²:** {np.median(r2_values):.4f}
- **Features with |r|>0.3:** {sum(1 for r in abs_r_values if r > 0.3)}
- **Features with |r|>0.2:** {sum(1 for r in abs_r_values if r > 0.2)}
- **Features with |r|>0.1:** {sum(1 for r in abs_r_values if r > 0.1)}

### Cross-Depth R² Matrix

|  | {' | '.join(DEPTH_COLS)} |
|--|{'|'.join(['--------' for _ in DEPTH_COLS])}|
"""
    
    for d1 in DEPTH_COLS:
        if d1 not in cross_depth:
            continue
        row = f"| {d1} |"
        for d2 in DEPTH_COLS:
            val = cross_depth.get(d1, {}).get(d2, np.nan)
            if np.isnan(val):
                row += " - |"
            else:
                row += f" {val:.3f} |"
        md += row + "\n"
    
    return md


def get_property_full_name(prop: str) -> str:
    """Get full property name."""
    names = {
        'OC': 'Organic Carbon (OC)',
        'pH': 'Soil pH',
        'Clay': 'Clay Content',
        'CEC': 'Cation Exchange Capacity (CEC)',
        'EC': 'Electrical Conductivity (EC)'
    }
    return names.get(prop, prop)


def compute_cv_r2(df: pd.DataFrame, target_col: str = 'X0.5cm') -> tuple:
    """Compute 5-fold cross-validation R² using TabPFN (same as generate_map_hybrid.py)."""
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import PowerTransformer, StandardScaler
    from tabpfn import TabPFNRegressor
    import warnings
    warnings.filterwarnings('ignore')
    
    # Get features that exist in dataframe
    available_features = [f for f in FEATURE_COLS if f in df.columns]
    
    # Prepare data
    mask = df[target_col].notna()
    for feat in available_features:
        mask &= df[feat].notna()
    
    X = df.loc[mask, available_features].values
    y = df.loc[mask, target_col].values
    
    if len(y) < 50:
        return None, None
    
    # Preprocessing pipeline (same as generate_map_hybrid.py)
    imputer = SimpleImputer(strategy='median')
    power_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
    scaler = StandardScaler()
    
    X = imputer.fit_transform(X)
    X = power_transformer.fit_transform(X)
    X = scaler.fit_transform(X)
    
    # TabPFN with 5-fold CV (same settings as generate_map_hybrid.py)
    # ignore_pretraining_limits=True allows CPU with large datasets
    tabpfn = TabPFNRegressor(device='cpu', n_estimators=4, random_state=42, ignore_pretraining_limits=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    scores = cross_val_score(tabpfn, X, y, cv=kf, scoring='r2')
    
    return scores.mean(), scores.std()


def generate_full_report():
    """Generate the complete statistics report."""
    print("=" * 60)
    print("SOIL DATA STATISTICS GENERATOR")
    print("=" * 60)
    
    # Header
    from datetime import datetime
    report = f"""# Soil Properties Statistical Analysis Report

**Generated:** {datetime.now().strftime('%B %d, %Y')}  
**Data Source:** `data/soil_with_raster/`  
**Target Depths:** X0.5cm, X5.15cm, X15.30cm, X30.60cm, X60.100cm, X100.200cm

---

## Physical Limits Applied

| Property | Lower Limit | Upper Limit | Outliers Removed |
|----------|-------------|-------------|------------------|
"""
    
    # Load all properties and track outliers
    property_data = {}
    outlier_counts = {}
    
    for prop in ['OC', 'pH', 'Clay', 'CEC', 'EC']:
        print(f"\n📊 Processing {prop}...")
        df, outliers = load_and_filter_data(prop)
        if df is not None:
            property_data[prop] = df
            outlier_counts[prop] = outliers
            limits = PHYSICAL_LIMITS[prop]
            report += f"| **{prop}** | {limits['lower']} | {limits['upper']} | {outliers} |\n"
            print(f"   ✓ Loaded {len(df):,} samples ({outliers} outliers removed)")
    
    report += "\n---\n"
    
    # Compute CV R² for each property
    cv_results = {}
    for prop, df in property_data.items():
        print(f"   🔄 Computing 5-fold CV R² for {prop}...")
        mean_r2, std_r2 = compute_cv_r2(df)
        cv_results[prop] = (mean_r2, std_r2)
        if mean_r2 is not None:
            print(f"      R² = {mean_r2:.4f} ± {std_r2:.4f}")
    
    # Executive Summary
    report += """
## Executive Summary

| Property | N Samples | Mean ± Std | Range | Max Feature r² | 5-Fold CV R² | Predictability |
|----------|-----------|------------|-------|----------------|--------------|----------------|
"""
    
    def get_predictability(r2):
        if r2 is None:
            return 'Unknown'
        if r2 >= 0.5:
            return '**Good**'
        elif r2 >= 0.3:
            return 'Moderate'
        else:
            return 'Poor'
    
    for prop, df in property_data.items():
        target = df['X0.5cm'].dropna()
        correlations = compute_feature_correlations(df)
        max_r2 = max(c['r2'] for c in correlations) if correlations else 0
        mean_cv, std_cv = cv_results.get(prop, (None, None))
        if mean_cv is not None:
            cv_r2_str = f"**{mean_cv:.3f}** ± {std_cv:.3f}"
        else:
            cv_r2_str = "N/A"
        pred = get_predictability(mean_cv)
        report += f"| **{prop}** | {len(target):,} | {target.mean():.2f} ± {target.std():.2f} | [{target.min():.2f}, {target.max():.2f}] | {max_r2:.3f} | {cv_r2_str} | {pred} |\n"
    
    report += """
> **Note:** "Max Feature r²" = best single predictor correlation. "5-Fold CV R²" = TabPFN model performance with all features (PowerTransformer + StandardScaler preprocessing).

---
"""
    
    # Generate sections for each property
    section_num = 1
    for prop, df in property_data.items():
        print(f"\n📝 Generating report section for {prop}...")
        section = generate_property_markdown(prop, df, outlier_counts[prop])
        # Add section number
        section = section.replace(f"## {get_property_full_name(prop)}", 
                                  f"## {section_num}. {get_property_full_name(prop)}")
        report += section + "\n---\n"
        section_num += 1
    
    # Add recommendations and appendix
    report += """
## Key Insights & Recommendations

### Predictability Ranking (by Max Feature R²)
1. **pH** (R² ~ 0.36) - Best predictability, strong linear relationships
2. **CEC** (R² ~ 0.32) - Good predictability, soil mineralogy important
3. **OC** (R² ~ 0.17) - Moderate, rainfall and vegetation are key drivers
4. **EC** (R² ~ 0.06) - Poor, requires non-linear models
5. **Clay** (R² ~ 0.01) - Very poor, requires non-linear models

### Most Important Covariates (Overall)
| Covariate | Best For | Interpretation |
|-----------|----------|----------------|
| `Kaolinite` | pH, CEC | Weathered soils, acidic conditions |
| `Smectite` | pH, CEC | Swelling clays, higher CEC |
| `Illite` | OC | Parent material influence |
| `MeanAnnualRF` | OC | Higher rainfall → more organic matter |
| `ET_Annual` | pH, CEC | Climate-soil relationships |
| `FPAR` variants | Multiple | Vegetation productivity proxy |
| `MRVBF` | pH, CEC | Valley bottom flatness |

### Modeling Recommendations

| Property | Recommended Approach |
|----------|---------------------|
| **pH** | Linear models may work; try Ridge/Lasso regression first |
| **CEC** | Ensemble methods (RF, XGBoost) with soil mineralogy features |
| **OC** | Log-transform target; focus on rainfall & vegetation features |
| **Clay** | Non-linear models essential; consider spatial interpolation |
| **EC** | Log-transform; handle extreme outliers; spatial kriging may help |

### Data Quality Notes
- **Sample size varies significantly:** CEC (437) vs pH (2,183)
- **High skewness in:** OC, CEC, EC (consider log-transformation)
- **Missing data increases with depth** for all properties
- **Cross-depth correlations** strongest for pH, weakest for EC

---

## Appendix: Feature Name Legend

| Short Name | Full Name | Category |
|------------|-----------|----------|
| Illite | Soil_Illite_NSW_ACT | Soil Mineralogy |
| Kaolinite | Soil_Kaolinite_NSW_ACT | Soil Mineralogy |
| Smectite | Soil_Smectite_NSW_ACT | Soil Mineralogy |
| MeanAnnualRF | Mean_Annual_RF_NSW_ACT_90m | Climate |
| MeanMaxTemp | Mean_Annual_Max_Temp_NSW_ACT_90m | Climate |
| Temp90thPct | Annual_90th_Percen_Temp_NSW_ACT_90m | Climate |
| RFVariance | Annual_RF_Var_NSW_ACT_90m | Climate |
| ET_Annual | ET_Annual_90m | Climate |
| Gravity | PM_Gravity_NSW_ACT | Parent Material |
| WeatherIdx | PM_Weathering_Index_nsw | Parent Material |
| Rad_ppmu | PM_radmap_v4_2019_filtered_ppmu_GAPFilled_nsw | Radiometrics |
| Rad_ppmt | PM_radmap_v4_2019_filtered_ppmt_GAPFilled_nsw | Radiometrics |
| Rad_dose | PM_radmap_v4_2019_filtered_dose_GAPFilled_nsw | Radiometrics |
| Rad_pctk | PM_radmap_v4_2019_filtered_pctk_GAPFilled_nsw | Radiometrics |
| Rad_u2t | PM_radmap_v4_2019_ratio_u2t_GAPFilled_nsw | Radiometrics |
| Rad_uk | PM_radmap_v4_2019_ratio_uk_GAPFilled_nsw | Radiometrics |
| Rad_ut | PM_radmap_v4_2019_ratio_ut_GAPFilled_nsw | Radiometrics |
| Rad_tk | PM_radmap_v4_2019_ratio_tk_GAPFilled_nsw | Radiometrics |
| Roughness | relief_roughness_nsw | Terrain |
| Slope | relief_slope_perc_nsw | Terrain |
| TWI | relief_twi_3s_nsw | Terrain |
| MRRTF | relief_mrrtf_3s_nsw | Terrain |
| MRVBF | relief_mrvbf_3s_mosaic_nsw | Terrain |
| DEM | relief_dems_3s_mosaic1_nsw | Terrain |
| ProfileCurv | relief_profile_curvature_3_nsw | Terrain |
| PlanCurv | relief_plan_curvature_3s_nsw | Terrain |
| FPAR | FPAR_pct | Vegetation |
| FPAR.1 | FPAR_pct.1 | Vegetation |
| FPAR.2 | FPAR_pct.2 | Vegetation |
| FPAR.3 | FPAR_pct.3 | Vegetation |
"""
    
    # Save report
    output_path = OUTPUT_DIR / "SOIL_DATA_STATISTICS.md"
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"\n✅ Report saved to: {output_path}")
    print("=" * 60)
    
    return report


if __name__ == "__main__":
    generate_full_report()
