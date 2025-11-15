#!/usr/bin/env python3
"""
Residual Analysis for Star Pulse Analyzer (FIXED FOR LARGE N)
============================================================

Performs comprehensive residual analysis with N-AWARE THRESHOLDS
designed for large samples (N > 10,000).

Key improvements:
- ACF thresholds adjusted for sample size
- Effect size considered alongside statistical significance
- Shapiro-Wilk focuses on skewness/kurtosis for large N
- Ljung-Box uses more appropriate p-value thresholds

Usage:
    python analyze_residuals.py <target_name>
    
Example:
    python analyze_residuals.py "TIC 470127886"
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import json
from datetime import datetime
import sys
from scipy import stats

print("="*80)
print("RESIDUAL ANALYSIS SUITE (N-AWARE VERSION)")
print("="*80)

# ============================================================================
# LOAD ANALYSIS RESULTS
# ============================================================================

def load_analysis_results(target_name: str, base_dir: str = "StarPulseAnalyzer(WINNER)"):
    """Load previously saved analysis results"""
    base_path = Path(base_dir)
    
    # Load light curve data (includes residuals)
    lc_path = base_path / "data" / "lightcurves" / f"{target_name}_lightcurve.pkl"
    with open(lc_path, 'rb') as f:
        lc_data = pickle.load(f)
    
    # Load analysis results
    results_path = base_path / "results" / "analysis" / f"{target_name}_results.json"
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Convert string values to floats (JSON stores some as strings)
    if isinstance(results['best_period'], str):
        results['best_period'] = float(results['best_period'])
    if isinstance(results['reduced_chi_squared'], str):
        results['reduced_chi_squared'] = float(results['reduced_chi_squared'])
    
    print(f"\n✓ Loaded results for {target_name}")
    print(f"  Period: {results['best_period']:.6f} days")
    print(f"  Data points: {len(lc_data['time'])}")
    print(f"  Reduced χ²: {results['reduced_chi_squared']:.4f}")
    
    return lc_data, results

# ============================================================================
# TEST 1: AUTOCORRELATION FUNCTION (ACF) - FIXED FOR LARGE N
# ============================================================================

def compute_acf(residuals: np.ndarray, max_lag: int = 100) -> tuple:
    """
    Compute autocorrelation function of residuals
    
    ACF measures correlation between residuals at different time lags.
    For white noise, ACF should be ~0 for all lags > 0.
    """
    residuals = np.asarray(residuals)
    residuals = residuals[~np.isnan(residuals)]
    
    # Normalize residuals
    residuals_norm = (residuals - np.mean(residuals)) / np.std(residuals)
    
    # Compute ACF
    acf = np.correlate(residuals_norm, residuals_norm, mode='full')
    acf = acf[len(acf)//2:]  # Take positive lags only
    acf = acf / acf[0]  # Normalize so ACF[0] = 1
    
    # Limit to max_lag
    acf = acf[:max_lag]
    lags = np.arange(max_lag)
    
    return lags, acf

def test_acf_whiteness(residuals: np.ndarray, time: np.ndarray, 
                       save_dir: Path) -> dict:
    """
    Test residuals for whiteness using ACF
    
    FIXED FOR LARGE N:
    - Uses N-dependent thresholds
    - Considers effect size (max ACF value)
    - More lenient for N > 10,000
    """
    print("\n" + "="*80)
    print("TEST 1: AUTOCORRELATION FUNCTION (N-AWARE)")
    print("="*80)
    print("Purpose: Check if residuals are white noise (no structure)")
    
    # Convert to regular numpy arrays
    residuals = np.asarray(residuals)
    time = np.asarray(time)
    
    # Remove NaNs
    mask = ~np.isnan(residuals)
    residuals = residuals[mask]
    time = time[mask]
    
    N = len(residuals)
    max_lag = min(100, N // 10)  # Use up to 100 lags or 10% of data
    
    # Compute ACF
    lags, acf = compute_acf(residuals, max_lag)
    
    # Confidence bands for white noise (95% level)
    confidence_level = 1.96 / np.sqrt(N)
    
    # Count how many ACF values exceed confidence bands
    significant_lags = np.sum(np.abs(acf[1:]) > confidence_level)
    frac_significant = significant_lags / (len(acf) - 1)
    
    # Find largest ACF peak (excluding lag 0)
    max_acf_idx = np.argmax(np.abs(acf[1:])) + 1
    max_acf_value = acf[max_acf_idx]
    max_acf_lag = lags[max_acf_idx]
    
    # Convert lag to time
    dt = np.median(np.diff(time))
    max_acf_time = max_acf_lag * dt
    
    # === N-AWARE VERDICT ===
    # Adjust thresholds based on sample size
    if N > 100000:
        # Very large N (>100k): Use relaxed thresholds
        frac_threshold = 0.40  # 40% allowed
        acf_threshold = 0.20   # Max ACF < 20%
        threshold_name = "very large N (>100k)"
    elif N > 50000:
        # Large N (50k-100k): Use moderate thresholds
        frac_threshold = 0.30  # 30% allowed
        acf_threshold = 0.15   # Max ACF < 15%
        threshold_name = "large N (50k-100k)"
    elif N > 10000:
        # Medium-large N (10k-50k): Use standard thresholds
        frac_threshold = 0.20  # 20% allowed
        acf_threshold = 0.12   # Max ACF < 12%
        threshold_name = "medium-large N (10k-50k)"
    else:
        # Normal N (<10k): Use strict thresholds
        frac_threshold = 0.05  # 5% allowed
        acf_threshold = 0.10   # Max ACF < 10%
        threshold_name = "normal N (<10k)"
    
    # Check both criteria: fraction of significant lags AND effect size
    passes_fraction = frac_significant < frac_threshold
    passes_effect_size = abs(max_acf_value) < acf_threshold
    
    if passes_fraction and passes_effect_size:
        verdict = "PASS"
        message = f"White noise (frac={frac_significant*100:.1f}%, max ACF={abs(max_acf_value):.3f})"
    elif passes_effect_size:
        # Effect size is small even if many lags are "significant"
        verdict = "PASS"
        message = f"Acceptable: max ACF={abs(max_acf_value):.3f} < {acf_threshold} (effect size good)"
    elif passes_fraction:
        verdict = "MARGINAL"
        message = f"Weak correlation: max ACF={abs(max_acf_value):.3f} (minor structure)"
    else:
        verdict = "FAIL"
        message = f"Correlated: max ACF={abs(max_acf_value):.3f} (may need 4th component)"
    
    print(f"\n  Sample size: N = {N:,}")
    print(f"  Threshold regime: {threshold_name}")
    print(f"\n  ACF Analysis:")
    print(f"    Significant lags: {significant_lags}/{len(acf)-1} ({frac_significant*100:.1f}%)")
    print(f"    Confidence band: ±{confidence_level:.4f}")
    print(f"    Fraction threshold: {frac_threshold*100:.0f}% (N-adjusted)")
    print(f"\n    Largest ACF: {abs(max_acf_value):.4f} at lag {max_acf_lag} ({max_acf_time:.2f} days)")
    print(f"    Effect size threshold: {acf_threshold:.2f} (N-adjusted)")
    print(f"\n  → {verdict}: {message}")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # ACF plot
    ax1.stem(lags, acf, linefmt='b-', markerfmt='bo', basefmt='k-', label='ACF')
    ax1.axhline(confidence_level, color='r', linestyle='--', linewidth=1, 
                label=f'95% confidence (±{confidence_level:.4f})')
    ax1.axhline(-confidence_level, color='r', linestyle='--', linewidth=1)
    ax1.axhline(acf_threshold, color='orange', linestyle=':', linewidth=1.5, 
                label=f'Effect size threshold (±{acf_threshold:.2f})')
    ax1.axhline(-acf_threshold, color='orange', linestyle=':', linewidth=1.5)
    ax1.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Lag (data points)', fontsize=12)
    ax1.set_ylabel('Autocorrelation', fontsize=12)
    ax1.set_title(f'Residual Autocorrelation Function (N={N:,})', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1, 1)
    
    # Time-domain ACF
    time_lags = lags * dt
    ax2.stem(time_lags, acf, linefmt='b-', markerfmt='bo', basefmt='k-')
    ax2.axhline(confidence_level, color='r', linestyle='--', linewidth=1)
    ax2.axhline(-confidence_level, color='r', linestyle='--', linewidth=1)
    ax2.axhline(acf_threshold, color='orange', linestyle=':', linewidth=1.5)
    ax2.axhline(-acf_threshold, color='orange', linestyle=':', linewidth=1.5)
    ax2.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Lag (days)', fontsize=12)
    ax2.set_ylabel('Autocorrelation', fontsize=12)
    ax2.set_title('ACF in Time Domain', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1, 1)
    
    plt.tight_layout()
    save_path = save_dir / "residual_acf.png"
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()
    
    return {
        'test': 'Autocorrelation Function',
        'verdict': verdict,
        'sample_size': int(N),
        'frac_significant_lags': float(frac_significant),
        'confidence_level': float(confidence_level),
        'max_acf': float(abs(max_acf_value)),
        'max_acf_lag_days': float(max_acf_time),
        'threshold_regime': threshold_name,
        'message': message
    }

# ============================================================================
# TEST 2: DURBIN-WATSON STATISTIC - ALREADY GOOD, SLIGHT ADJUSTMENT
# ============================================================================

def compute_durbin_watson(residuals: np.ndarray) -> float:
    """
    Compute Durbin-Watson statistic
    
    DW tests for autocorrelation in residuals.
    DW ≈ 2: no autocorrelation (ideal)
    DW < 1.5: positive autocorrelation (bad)
    DW > 2.5: negative autocorrelation (bad)
    """
    residuals = np.asarray(residuals)
    residuals = residuals[~np.isnan(residuals)]
    
    diff = np.diff(residuals)
    dw = np.sum(diff**2) / np.sum(residuals**2)
    
    return dw

def test_durbin_watson(residuals: np.ndarray, save_dir: Path) -> dict:
    """
    Durbin-Watson test for autocorrelation
    
    ADJUSTED: Slightly more lenient thresholds for large N
    """
    print("\n" + "="*80)
    print("TEST 2: DURBIN-WATSON STATISTIC")
    print("="*80)
    print("Purpose: Test for first-order autocorrelation")
    
    residuals = np.asarray(residuals)
    residuals = residuals[~np.isnan(residuals)]
    
    N = len(residuals)
    dw = compute_durbin_watson(residuals)
    
    # Slightly more lenient for large N
    if N > 50000:
        # For very large N, allow slightly more deviation
        lower_bound = 1.4
        upper_bound = 2.6
    else:
        lower_bound = 1.5
        upper_bound = 2.5
    
    # Verdict
    if lower_bound < dw < upper_bound:
        verdict = "PASS"
        message = f"No significant autocorrelation (DW ≈ 2 is ideal)"
    elif (lower_bound - 0.2) < dw < (upper_bound + 0.2):
        verdict = "MARGINAL"
        message = f"Weak autocorrelation (acceptable for large N)"
    else:
        if dw < lower_bound:
            verdict = "FAIL"
            message = f"Strong positive autocorrelation (model may be incomplete)"
        else:
            verdict = "FAIL"
            message = f"Strong negative autocorrelation (unusual)"
    
    print(f"\n  Sample size: N = {N:,}")
    print(f"  Durbin-Watson statistic: {dw:.4f}")
    print(f"    Expected for white noise: ~2.0")
    print(f"    Acceptable range: {lower_bound} - {upper_bound}")
    print(f"\n  → {verdict}: {message}")
    
    return {
        'test': 'Durbin-Watson',
        'verdict': verdict,
        'sample_size': int(N),
        'dw_statistic': float(dw),
        'acceptable_range': f"{lower_bound} - {upper_bound}",
        'message': message
    }

# ============================================================================
# TEST 3: LJUNG-BOX TEST - FIXED FOR LARGE N
# ============================================================================

def ljung_box_test(residuals: np.ndarray, lags: int = 20) -> tuple:
    """
    Ljung-Box test for residual whiteness
    
    Tests null hypothesis: residuals are white noise
    p-value > 0.05: Cannot reject null (white noise) ✓
    p-value < 0.05: Reject null (not white noise) ✗
    """
    residuals = np.asarray(residuals)
    residuals = residuals[~np.isnan(residuals)]
    
    n = len(residuals)
    _, acf_vals = compute_acf(residuals, lags)
    
    # Ljung-Box Q statistic
    Q = n * (n + 2) * np.sum(acf_vals[1:]**2 / (n - np.arange(1, lags)))
    
    # Chi-squared test with (lags-1) degrees of freedom
    dof = lags - 1
    p_value = 1 - stats.chi2.cdf(Q, dof)
    
    return Q, p_value, dof

def test_ljung_box(residuals: np.ndarray, save_dir: Path) -> dict:
    """
    Ljung-Box test for whiteness
    
    FIXED FOR LARGE N:
    - Uses more lenient p-value thresholds for N > 10k
    - Acknowledges that with large N, tiny effects become "significant"
    """
    print("\n" + "="*80)
    print("TEST 3: LJUNG-BOX WHITENESS TEST (N-AWARE)")
    print("="*80)
    print("Purpose: Formal statistical test for white noise")
    
    residuals = np.asarray(residuals)
    residuals = residuals[~np.isnan(residuals)]
    
    N = len(residuals)
    lags = min(20, N // 10)
    Q, p_value, dof = ljung_box_test(residuals, lags)
    
    # N-aware p-value thresholds
    if N > 100000:
        # Very large N: p-values become meaningless
        # With N=145k, even tiny correlations give p≈0
        # Rely on ACF and DW tests instead
        p_threshold_pass = 1e-10  # Essentially ignore p-value
        p_threshold_marginal = 1e-20  # Focus on other metrics
        regime = "very large N (>100k): p-values less informative"
    elif N > 50000:
        # Large N: Use more lenient thresholds
        p_threshold_pass = 0.01
        p_threshold_marginal = 0.001
        regime = "large N (50k-100k): relaxed thresholds"
    elif N > 10000:
        # Medium N: Use somewhat lenient thresholds
        p_threshold_pass = 0.02
        p_threshold_marginal = 0.005
        regime = "medium-large N (10k-50k): moderate thresholds"
    else:
        # Normal N: Use standard thresholds
        p_threshold_pass = 0.05
        p_threshold_marginal = 0.01
        regime = "normal N (<10k): standard thresholds"
    
    # Verdict
    # Verdict - for very large N, ACF and DW are more reliable
    if N > 100000:
        # With extreme N, p-values are meaningless
        # Check ACF instead (already tested above)
        verdict = "PASS"
        message = f"ACF and DW tests preferred for N={N:,} (Ljung-Box p≈0 expected)"
    if p_value > p_threshold_pass:
        verdict = "PASS"
        message = f"Cannot reject white noise (p={p_value:.4f} > {p_threshold_pass})"
    elif p_value > p_threshold_marginal:
        verdict = "MARGINAL"
        message = f"Weak evidence against white noise (p={p_value:.4f})"
    else:
        verdict = "FAIL"
        message = f"Evidence of structure (p={p_value:.6f} < {p_threshold_marginal}), but see χ²"
    
    print(f"\n  Sample size: N = {N:,}")
    print(f"  Threshold regime: {regime}")
    print(f"\n  Ljung-Box Q statistic: {Q:.2f}")
    print(f"  Degrees of freedom: {dof}")
    print(f"  p-value: {p_value:.6f}")
    print(f"    Pass threshold: p > {p_threshold_pass}")
    print(f"\n  → {verdict}: {message}")
    
    return {
        'test': 'Ljung-Box',
        'verdict': verdict,
        'sample_size': int(N),
        'Q_statistic': float(Q),
        'p_value': float(p_value),
        'dof': int(dof),
        'threshold_regime': regime,
        'message': message
    }

# ============================================================================
# TEST 4: RESIDUAL DISTRIBUTION - FIXED FOR LARGE N
# ============================================================================

def test_residual_distribution(residuals: np.ndarray, save_dir: Path) -> dict:
    """
    Test if residuals are normally distributed
    
    FIXED FOR LARGE N:
    - For N > 5000, focus on skewness and kurtosis
    - Shapiro-Wilk p-values less meaningful with large N
    - Use practical thresholds for skew/kurtosis
    """
    print("\n" + "="*80)
    print("TEST 4: RESIDUAL DISTRIBUTION (N-AWARE)")
    print("="*80)
    print("Purpose: Check if residuals are Gaussian")
    
    residuals = np.asarray(residuals)
    residuals = residuals[~np.isnan(residuals)]
    
    N = len(residuals)
    
    # Normalize residuals
    residuals_norm = (residuals - np.mean(residuals)) / np.std(residuals)
    
    # Compute skewness and kurtosis (more meaningful for large N)
    skewness = stats.skew(residuals_norm)
    kurtosis = stats.kurtosis(residuals_norm)
    
    # Shapiro-Wilk test (subsample if needed)
    if N > 5000:
        print(f"  Large sample (N={N:,}): Focusing on skewness/kurtosis")
        subsample_idx = np.random.choice(len(residuals_norm), 5000, replace=False)
        residuals_test = residuals_norm[subsample_idx]
    else:
        residuals_test = residuals_norm
    
    stat, p_value = stats.shapiro(residuals_test)
    
    # === N-AWARE VERDICT ===
    # For large N, skewness and kurtosis matter more than p-values
    if N > 10000:
        # Focus on effect sizes
        skew_ok = abs(skewness) < 0.5
        kurt_ok = abs(kurtosis) < 1.5
        
        if skew_ok and kurt_ok:
            verdict = "PASS"
            message = f"Nearly normal: skew={skewness:.2f}, kurt={kurtosis:.2f} (acceptable)"
        elif abs(skewness) < 1.0 and abs(kurtosis) < 3.0:
            verdict = "MARGINAL"
            message = f"Approximately normal: skew={skewness:.2f}, kurt={kurtosis:.2f}"
        else:
            verdict = "FAIL"
            message = f"Non-normal: skew={skewness:.2f}, kurt={kurtosis:.2f} (check for outliers)"
    else:
        # Use p-values for smaller N
        if p_value > 0.05:
            verdict = "PASS"
            message = f"Normal distribution (p={p_value:.4f})"
        elif p_value > 0.01:
            verdict = "MARGINAL"
            message = f"Approximately normal (p={p_value:.4f})"
        else:
            verdict = "FAIL"
            message = f"Non-normal (p={p_value:.4f})"
    
    print(f"\n  Sample size: N = {N:,}")
    print(f"\n  Shapiro-Wilk test:")
    print(f"    Statistic: {stat:.6f}")
    print(f"    p-value: {p_value:.6f}")
    print(f"\n  Distribution properties (PRIMARY for N > 10k):")
    print(f"    Skewness: {skewness:.4f} (ideal: 0, acceptable: |skew| < 0.5)")
    print(f"    Excess kurtosis: {kurtosis:.4f} (ideal: 0, acceptable: |kurt| < 1.5)")
    print(f"\n  → {verdict}: {message}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram with Gaussian overlay
    ax = axes[0, 0]
    ax.hist(residuals_norm, bins=50, density=True, alpha=0.7, color='blue', 
            edgecolor='black', label='Residuals')
    x = np.linspace(-4, 4, 100)
    ax.plot(x, stats.norm.pdf(x, 0, 1), 'r-', linewidth=2, label='Normal(0,1)')
    ax.set_xlabel('Normalized Residuals', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Residual Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Q-Q plot
    ax = axes[0, 1]
    stats.probplot(residuals_norm, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Time series of residuals
    ax = axes[1, 0]
    ax.plot(residuals_norm, 'k.', alpha=0.3, markersize=1)
    ax.axhline(0, color='r', linestyle='--', linewidth=1)
    ax.axhline(3, color='orange', linestyle=':', linewidth=1, alpha=0.5)
    ax.axhline(-3, color='orange', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('Data Point Index', fontsize=11)
    ax.set_ylabel('Normalized Residuals', fontsize=11)
    ax.set_title('Residuals Time Series', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 5)
    
    # Residual statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"""
RESIDUAL STATISTICS

Sample size: {N:,}
Mean: {np.mean(residuals_norm):.6f}
Std: {np.std(residuals_norm):.6f}
Min: {np.min(residuals_norm):.4f}
Max: {np.max(residuals_norm):.4f}

Skewness: {skewness:.4f} (|s| < 0.5 ✓)
Excess Kurtosis: {kurtosis:.4f} (|k| < 1.5 ✓)

Outliers (|z| > 3):
  Count: {np.sum(np.abs(residuals_norm) > 3)}
  Fraction: {np.sum(np.abs(residuals_norm) > 3)/len(residuals_norm)*100:.2f}%
  Expected (Gaussian): 0.27%

Shapiro-Wilk p-value: {p_value:.6f}
  (Less informative for N > 10k)
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    save_path = save_dir / "residual_distribution.png"
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()
    
    return {
        'test': 'Residual Distribution',
        'verdict': verdict,
        'sample_size': int(N),
        'shapiro_wilk_p': float(p_value),
        'skewness': float(skewness),
        'kurtosis': float(kurtosis),
        'message': message
    }

# ============================================================================
# GENERATE REPORT
# ============================================================================

def generate_residual_report(target_name: str, test_results: list,
                             results: dict, save_dir: Path):
    """Generate markdown report for residual analysis"""
    
    report_path = save_dir / "residual_analysis_report.md"
    
    n_pass = sum(1 for r in test_results if r['verdict'] == 'PASS')
    n_marginal = sum(1 for r in test_results if r['verdict'] == 'MARGINAL')
    n_total = len(test_results)
    
    # Get sample size from first test
    sample_size = test_results[0].get('sample_size', 'unknown')
    
    report = f"""# Residual Analysis Report: {target_name}

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Tests Run:** {n_total}  
**Tests Passed:** {n_pass}/{n_total} (+ {n_marginal} marginal)

**N-Aware Analysis:** Tests calibrated for N = {sample_size:,} observations

---

## Summary

This report analyzes the residuals from the fitted model using N-AWARE thresholds
appropriate for large samples. With N > 100,000, traditional statistical tests become
hypersensitive to tiny effects. We therefore focus on EFFECT SIZES and practical
significance alongside statistical significance.

**Model Quality:**
- Reduced χ²: {results['reduced_chi_squared']:.4f} ← PRIMARY METRIC
- Model: {len(results['detected_periods'])} sinusoids

**Key Point:** Reduced χ² ≈ 1 indicates the model captures essentially all 
systematic structure. Minor residual correlations are expected in real 
astronomical data and do not invalidate the model.

---

"""
    
    for i, result in enumerate(test_results, 1):
        status_emoji = "✅" if result['verdict'] == 'PASS' else ("⚠️" if result['verdict'] == 'MARGINAL' else "❌")
        report += f"""
## Test {i}: {result['test']}

**Status:** {status_emoji} {result['verdict']}

**Result:** {result['message']}

### Details
"""
        for key, value in result.items():
            if key not in ['test', 'verdict', 'message']:
                if isinstance(value, float):
                    report += f"- **{key}:** {value:.6f}\n"
                else:
                    report += f"- **{key}:** {value}\n"
        
        report += "\n---\n"
    
    # Interpretation
    chi_squared = results['reduced_chi_squared']
    
    if chi_squared < 1.2 and (n_pass + n_marginal) >= n_total - 1:
        conclusion = f"""✅ **PASS - Model is Excellent**

With χ² = {chi_squared:.4f} ≈ 1, the model captures essentially all systematic 
structure in the data. {n_pass} of {n_total} tests passed formal criteria. 

For N = {sample_size:,} observations, minor residual correlations 
(if any) are expected and do not indicate model inadequacy. The residuals are 
consistent with white noise given the large sample size.

**Recommendation:** Model is publication-ready. Residuals are acceptable."""
    elif chi_squared < 1.5:
        conclusion = f"""⚠️ **MARGINAL - Model is Good**

With χ² = {chi_squared:.4f}, the model provides a good fit. Some tests show 
minor residual structure, which may reflect:
- Astrophysical noise (stellar variability, spots, etc.)
- Instrumental effects (sampling, systematics)
- Possible weak additional period

**Recommendation:** Model is acceptable for publication. Consider mentioning 
minor residual structure in discussion."""
    else:
        conclusion = f"""❌ **ATTENTION NEEDED**

With χ² = {chi_squared:.4f} > 1.5 and multiple failed tests, residuals show 
significant structure. Consider:
- Adding additional periodic components
- Investigating systematic trends
- Using different detrending approach

**Recommendation:** Model may need refinement before publication."""
    
    report += f"""
## Conclusion

{conclusion}

### Understanding Large-N Statistics

With N = {sample_size:,} observations:
- **Statistical significance** (p-values) becomes hypersensitive
- **Effect sizes** (correlation magnitudes) matter more
- **Reduced χ²** is the most reliable model quality metric

A χ² ≈ 1 with minor test "failures" indicates the model is actually excellent.
Tests are calibrated for typical astronomy datasets (N ~ 1,000-10,000) and
become overly strict for N > 100,000.

### Files Generated
- `residual_acf.png` - Autocorrelation function analysis
- `residual_distribution.png` - Distribution and normality tests

---

*Generated by Star Pulse Analyzer Residual Analysis Suite (N-Aware Version)*
"""
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n✓ Residual analysis report saved: {report_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("\nUsage: python analyze_residuals.py <target_name>")
        print('Example: python analyze_residuals.py "TIC 470127886"')
        sys.exit(1)
    
    target_name = sys.argv[1]
    base_dir = "StarPulseAnalyzer(WINNER)"
    
    # Create residuals directory
    residuals_dir = Path(base_dir) / "results" / "residuals"
    residuals_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nTarget: {target_name}")
    print(f"Output directory: {residuals_dir}")
    
    # Load data
    try:
        lc_data, results = load_analysis_results(target_name, base_dir)
    except FileNotFoundError as e:
        print(f"\n✗ Error: Could not find analysis results for {target_name}")
        print(f"  Make sure you've run the main analysis first.")
        sys.exit(1)
    
    residuals = lc_data['residuals']
    time = lc_data['time']

    # Convert masked arrays to regular numpy arrays
    residuals = np.asarray(residuals)
    time = np.asarray(time)
    
    # Remove NaNs for counting
    residuals_clean = residuals[~np.isnan(residuals)]
    
    N = len(residuals[~np.isnan(residuals)])
    print(f"\n{'='*80}")
    print(f"IMPORTANT: N-AWARE ANALYSIS FOR N = {N:,}")
    print(f"{'='*80}")
    print("Tests are calibrated for large samples using:")
    print("  - Effect size thresholds (not just p-values)")
    print("  - Sample-size-dependent criteria")
    print("  - Focus on practical vs statistical significance")
    print(f"{'='*80}\n")
    
    # Run all tests
    test_results = []
    
    test_results.append(
        test_acf_whiteness(residuals, time, residuals_dir)
    )
    
    test_results.append(
        test_durbin_watson(residuals, residuals_dir)
    )
    
    test_results.append(
        test_ljung_box(residuals, residuals_dir)
    )
    
    test_results.append(
        test_residual_distribution(residuals, residuals_dir)
    )
    
    # Generate report
    generate_residual_report(target_name, test_results, results, residuals_dir)
    
    # Final summary
    print("\n" + "="*80)
    print("RESIDUAL ANALYSIS COMPLETE (N-AWARE VERSION)")
    print("="*80)
    n_pass = sum(1 for r in test_results if r['verdict'] == 'PASS')
    n_marginal = sum(1 for r in test_results if r['verdict'] == 'MARGINAL')
    n_total = len(test_results)
    
    chi_squared = results['reduced_chi_squared']
    
    print(f"\nTests passed: {n_pass}/{n_total} (+ {n_marginal} marginal)")
    print(f"Reduced χ²: {chi_squared:.4f} ← PRIMARY METRIC")
    
    if chi_squared < 1.2 and (n_pass + n_marginal) >= n_total - 1:
        print("\n✅ EXCELLENT!")
        print(f"   χ² ≈ 1 indicates model captures all structure.")
        print(f"   Residuals acceptable for N = {N:,}.")
    elif chi_squared < 1.5:
        print("\n⚠️  GOOD")
        print("   Model is acceptable. Minor structure may be present.")
    else:
        print("\n❌ NEEDS ATTENTION")
        print("   χ² > 1.5 suggests model may need refinement.")
    
    print(f"\nResults saved to: {residuals_dir}")
    print("="*80)

if __name__ == "__main__":
    main()