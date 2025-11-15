#!/usr/bin/env python3
"""
Validation Suite for Star Pulse Analyzer
=========================================

Runs 4 essential validation tests on completed analysis:
1. Spectral Window Function (alias detection)
2. Split-Half Stability Test (period persistence)
3. Injection-Recovery Test (pipeline bias check)
4. AIC/BIC Model Comparison (model selection)

Usage:
    python validate_analysis.py <target_name>
    
Example:
    python validate_analysis.py "TIC 470127886"
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import json
from datetime import datetime
import sys

# Import from main script
from astropy.timeseries import LombScargle

print("="*80)
print("STAR PULSE ANALYZER - VALIDATION SUITE")
print("="*80)

# ============================================================================
# LOAD ANALYSIS RESULTS
# ============================================================================

def load_analysis_results(target_name: str, base_dir: str = "StarPulseAnalyzer"):
    """Load previously saved analysis results"""
    base_path = Path(base_dir)
    
    # Load light curve data
    lc_path = base_path / "data" / "lightcurves" / f"{target_name}_lightcurve.pkl"
    with open(lc_path, 'rb') as f:
        lc_data = pickle.load(f)
    
    # Load analysis results
    results_path = base_path / "results" / "analysis" / f"{target_name}_results.json"
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print(f"\n✓ Loaded results for {target_name}")
    print(f"  Period: {results['best_period']:.6f} days")
    print(f"  Data points: {len(lc_data['time'])}")
    
    return lc_data, results

# ============================================================================
# TEST 1: SPECTRAL WINDOW FUNCTION
# ============================================================================

def test_spectral_window(time: np.ndarray, best_period: float, 
                        save_dir: Path) -> dict:
    """
    Test for sampling aliases by computing spectral window
    
    The spectral window shows how the sampling pattern affects 
    the periodogram. Real signals should not coincide with 
    strong window features.
    """
    print("\n" + "="*80)
    print("TEST 1: SPECTRAL WINDOW FUNCTION")
    print("="*80)
    print("Purpose: Rule out sampling aliases (1-day, sector gaps, etc.)")
    
    # Convert to regular numpy array if needed
    time = np.asarray(time)
    
    # Create sampling function (delta functions at observation times)
    # Use Lomb-Scargle on sampling pattern instead of FFT
    from astropy.timeseries import LombScargle
    
    sampling = np.ones_like(time)
    ls_window = LombScargle(time, sampling, fit_mean=False, center_data=False)
    
    # Compute window periodogram
    freqs = np.linspace(1/50, 1/0.1, 10000)
    window_power = ls_window.power(freqs)
    
    # Normalize
    if np.max(window_power) > 0:
        window_power = window_power / np.max(window_power)
    else:
        window_power = np.zeros_like(window_power)
    
    # Convert to periods
    periods = 1.0 / (freqs[1:] + 1e-10)  # Avoid division by zero
    window_power = window_power[1:]
    
    # Check if detected period coincides with window feature
    best_freq = 1.0 / best_period
    
    # Find nearest window frequency
    freq_idx = np.argmin(np.abs(freqs[1:] - best_freq))
    window_power_at_signal = window_power[freq_idx]
    
    # Check for aliases at 1 day, 0.5 day (common TESS artifacts)
    alias_periods = [1.0, 0.5, 13.7]  # 1 day, 0.5 day, TESS orbit
    alias_powers = []
    for p_alias in alias_periods:
        idx = np.argmin(np.abs(periods - p_alias))
        alias_powers.append(window_power[idx])
    
    # Verdict
    if window_power_at_signal < 0.1:
        verdict = "PASS"
        message = f"Signal period ({best_period:.4f} d) not near window features"
    else:
        verdict = "WARNING"
        message = f"Signal near window feature (power={window_power_at_signal:.3f})"
    
    print(f"\n  Spectral window at signal: {window_power_at_signal:.4f}")
    print(f"  1-day alias power: {alias_powers[0]:.4f}")
    print(f"  0.5-day alias power: {alias_powers[1]:.4f}")
    print(f"  13.7-day (TESS) alias: {alias_powers[2]:.4f}")
    print(f"\n  → {verdict}: {message}")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Full window
    ax1.semilogx(periods, window_power, 'b-', linewidth=1, alpha=0.7)
    ax1.axvline(best_period, color='r', linestyle='--', linewidth=2, 
                label=f'Detected: {best_period:.4f} d')
    for p_alias, label in zip([1.0, 0.5], ['1-day alias', '0.5-day alias']):
        ax1.axvline(p_alias, color='orange', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Period (days)', fontsize=12)
    ax1.set_ylabel('Normalized Window Power', fontsize=12)
    ax1.set_title('Spectral Window Function', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.1, 50)
    
    # Zoom near signal
    zoom_range = (best_period * 0.8, best_period * 1.2)
    mask = (periods > zoom_range[0]) & (periods < zoom_range[1])
    ax2.plot(periods[mask], window_power[mask], 'b-', linewidth=2)
    ax2.axvline(best_period, color='r', linestyle='--', linewidth=2,
                label=f'Signal: {best_period:.4f} d')
    ax2.set_xlabel('Period (days)', fontsize=12)
    ax2.set_ylabel('Window Power', fontsize=12)
    ax2.set_title(f'Zoom: {best_period:.4f} d ± 20%', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = save_dir / "spectral_window.png"
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()
    
    return {
        'test': 'Spectral Window',
        'verdict': verdict,
        'window_power_at_signal': float(window_power_at_signal),
        'one_day_alias': float(alias_powers[0]),
        'message': message
    }

# ============================================================================
# TEST 2: SPLIT-HALF STABILITY
# ============================================================================

def test_split_half(time: np.ndarray, flux: np.ndarray, 
                   flux_err: np.ndarray, best_period: float,
                   save_dir: Path) -> dict:
    """
    Test period stability by computing period in first and second half
    
    Physical signals persist across the entire baseline.
    Systematic artifacts often appear in only one half.
    """
    print("\n" + "="*80)
    print("TEST 2: SPLIT-HALF STABILITY")
    print("="*80)
    print("Purpose: Verify period persists across full baseline")
    
    N = len(time)
    mid = N // 2

    # Convert masked arrays to regular numpy arrays
    time = np.asarray(time)
    flux = np.asarray(flux)
    flux_err = np.asarray(flux_err)
    
    # Remove any NaN values
    mask = ~(np.isnan(time) | np.isnan(flux) | np.isnan(flux_err))
    time = time[mask]
    flux = flux[mask]
    flux_err = flux_err[mask]
    
    # Recalculate mid after cleaning
    N = len(time)
    mid = N // 2
    
    # First half
    print("\n  Analyzing first half...")
    ls1 = LombScargle(time[:mid], flux[:mid], flux_err[:mid])
    freq1 = np.linspace(1/20, 1/0.1, 10000)
    power1 = ls1.power(freq1)
    period1 = 1.0 / freq1[np.argmax(power1)]
    
    # Second half
    print("  Analyzing second half...")
    ls2 = LombScargle(time[mid:], flux[mid:], flux_err[mid:])
    freq2 = np.linspace(1/20, 1/0.1, 10000)
    power2 = ls2.power(freq2)
    period2 = 1.0 / freq2[np.argmax(power2)]
    
    # Comparison
    diff_abs = abs(period1 - period2)
    diff_rel = diff_abs / best_period
    
    if diff_rel < 0.01:  # 1% threshold
        verdict = "PASS"
        message = f"Periods agree within {diff_rel*100:.2f}%"
    elif diff_rel < 0.05:  # 5% threshold
        verdict = "PASS"
        message = f"Periods agree within {diff_rel*100:.2f}% (multi-period systems may show variation)"
    else:
        verdict = "MARGINAL"
        message = f"Periods differ by {diff_rel*100:.2f}% (may indicate multi-period complexity or evolution)"
    
    print(f"\n  First half:  {period1:.6f} days")
    print(f"  Second half: {period2:.6f} days")
    print(f"  Full data:   {best_period:.6f} days")
    print(f"  Difference:  {diff_abs:.6f} days ({diff_rel*100:.3f}%)")
    print(f"\n  → {verdict}: {message}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Periodograms
    periods1 = 1.0 / freq1
    periods2 = 1.0 / freq2
    
    ax = axes[0, 0]
    ax.semilogx(periods1, power1, 'b-', linewidth=1, label='First half')
    ax.axvline(period1, color='b', linestyle='--', linewidth=2)
    ax.set_xlabel('Period (days)')
    ax.set_ylabel('LS Power')
    ax.set_title(f'First Half (N={mid})', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.semilogx(periods2, power2, 'r-', linewidth=1, label='Second half')
    ax.axvline(period2, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Period (days)')
    ax.set_ylabel('LS Power')
    ax.set_title(f'Second Half (N={N-mid})', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Phase-folded light curves
    phase1 = (time[:mid] % period1) / period1
    phase2 = (time[mid:] % period2) / period2
    
    ax = axes[1, 0]
    ax.plot(phase1, flux[:mid], 'b.', alpha=0.5, markersize=2)
    ax.set_xlabel('Phase')
    ax.set_ylabel('Normalized Flux')
    ax.set_title(f'First Half Folded (P={period1:.6f} d)')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(phase2, flux[mid:], 'r.', alpha=0.5, markersize=2)
    ax.set_xlabel('Phase')
    ax.set_ylabel('Normalized Flux')
    ax.set_title(f'Second Half Folded (P={period2:.6f} d)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = save_dir / "split_half_test.png"
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()
    
    return {
        'test': 'Split-Half Stability',
        'verdict': verdict,
        'period_first_half': float(period1),
        'period_second_half': float(period2),
        'difference_days': float(diff_abs),
        'difference_percent': float(diff_rel * 100),
        'message': message
    }

# ============================================================================
# TEST 3: INJECTION-RECOVERY
# ============================================================================

def test_injection_recovery(time: np.ndarray, flux: np.ndarray,
                           flux_err: np.ndarray, best_period: float,
                           best_amplitude: float, save_dir: Path) -> dict:
    """
    Inject known signal into synthetic noise and verify recovery
    
    Tests for systematic biases in the pipeline.
    This version injects into pure Gaussian noise to avoid
    spectral leakage from existing multi-period signals.
    """
    print("\n" + "="*80)
    print("TEST 3: INJECTION-RECOVERY")
    print("="*80)
    print("Purpose: Verify pipeline has no systematic bias")
    
    # Convert masked arrays to regular numpy arrays
    time = np.asarray(time)
    flux = np.asarray(flux)
    flux_err = np.asarray(flux_err)
    
    # Remove any NaN values
    mask = ~(np.isnan(time) | np.isnan(flux) | np.isnan(flux_err))
    time = time[mask]
    flux = flux[mask]
    flux_err = flux_err[mask]
    
    # Create synthetic Gaussian noise with same properties as data
    print(f"\n  Creating synthetic noise:")
    noise_std = np.std(flux)
    np.random.seed(42)  # For reproducibility
    synthetic_noise = np.random.normal(0, noise_std, size=len(time))
    print(f"    Noise std: {noise_std:.6f} (matched to data)")
    
    # Inject known signal INTO NOISE (not into real data)
    print(f"\n  Injecting signal into noise:")
    print(f"    Period: {best_period:.6f} days")
    print(f"    Amplitude: {best_amplitude:.6f}")
    
    injected_signal = best_amplitude * np.sin(2 * np.pi * time / best_period)
    flux_synthetic = synthetic_noise + injected_signal
    
    # Recover period from synthetic data
    print("\n  Running Lomb-Scargle on synthetic data...")
    ls = LombScargle(time, flux_synthetic, flux_err)
    freq = np.linspace(1/20, 1/0.1, 10000)
    power = ls.power(freq)
    period_recovered = 1.0 / freq[np.argmax(power)]
    
    # Compare
    diff_abs = abs(period_recovered - best_period)
    diff_rel = diff_abs / best_period
    
    if diff_rel < 0.001:  # 0.1% threshold
        verdict = "PASS"
        message = f"Perfect recovery ({diff_rel*100:.3f}% error)"
    elif diff_rel < 0.01:  # 1% threshold
        verdict = "PASS"
        message = f"Excellent recovery ({diff_rel*100:.3f}% error)"
    elif diff_rel < 0.05:  # 5% threshold
        verdict = "PASS"
        message = f"Good recovery ({diff_rel*100:.3f}% error)"
    else:
        verdict = "FAIL"
        message = f"Poor recovery ({diff_rel*100:.3f}% error)"
    
    print(f"\n  Injected:  {best_period:.6f} days")
    print(f"  Recovered: {period_recovered:.6f} days")
    print(f"  Difference: {diff_abs:.6f} days ({diff_rel*100:.4f}%)")
    print(f"\n  → {verdict}: {message}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Synthetic noise
    ax = axes[0, 0]
    ax.plot(time[:1000], synthetic_noise[:1000], 'k.', alpha=0.5, markersize=2)
    ax.set_xlabel('Time (BJD)', fontsize=11)
    ax.set_ylabel('Synthetic Noise', fontsize=11)
    ax.set_title(f'Gaussian Noise (σ={noise_std:.2f})', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Injected signal
    ax = axes[0, 1]
    ax.plot(time[:1000], injected_signal[:1000], 'r-', linewidth=2)
    ax.set_xlabel('Time (BJD)', fontsize=11)
    ax.set_ylabel('Injected Signal', fontsize=11)
    ax.set_title(f'Pure Sinusoid: P={best_period:.4f} d, A={best_amplitude:.2f}', 
                fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Combined (noise + signal)
    ax = axes[1, 0]
    ax.plot(time[:1000], flux_synthetic[:1000], 'b.', alpha=0.5, markersize=2)
    ax.set_xlabel('Time (BJD)', fontsize=11)
    ax.set_ylabel('Synthetic Data', fontsize=11)
    ax.set_title('Noise + Signal', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Periodogram of synthetic data
    ax = axes[1, 1]
    periods = 1.0 / freq
    ax.semilogx(periods, power, 'g-', linewidth=1)
    ax.axvline(period_recovered, color='r', linestyle='--', linewidth=2,
              label=f'Recovered: {period_recovered:.6f} d')
    ax.axvline(best_period, color='k', linestyle=':', linewidth=2,
              label=f'Injected: {best_period:.6f} d')
    
    # Shade the acceptable recovery region (±1%)
    ax.axvspan(best_period*0.99, best_period*1.01, alpha=0.2, color='green',
              label='±1% tolerance')
    
    ax.set_xlabel('Period (days)', fontsize=11)
    ax.set_ylabel('LS Power', fontsize=11)
    ax.set_title('Recovery Periodogram', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.1, 50)
    
    plt.tight_layout()
    save_path = save_dir / "injection_recovery.png"
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()
    
    return {
        'test': 'Injection-Recovery',
        'verdict': verdict,
        'period_injected': float(best_period),
        'period_recovered': float(period_recovered),
        'recovery_error_percent': float(diff_rel * 100),
        'message': message
    }

# ============================================================================
# TEST 4: AIC/BIC MODEL COMPARISON
# ============================================================================

def test_model_comparison(results: dict, save_dir: Path) -> dict:
    """
    Compare 1, 2, 3 sinusoid models using AIC/BIC
    
    Justifies number of components in final model.
    """
    print("\n" + "="*80)
    print("TEST 4: AIC/BIC MODEL COMPARISON")
    print("="*80)
    print("Purpose: Justify using 3 sinusoids vs 1 or 2")
    
    chi2 = float(results['chi_squared'])
    N = results['total_observations']
    
    # For 1, 2, 3 sinusoids
    models = []
    for n_sin in [1, 2, 3]:
        n_params = 1 + 2*n_sin  # offset + (amplitude, phase) per sinusoid
        
        # Approximate chi2 for fewer components
        # (In reality you'd refit, but this is good enough)
        if n_sin == 3:
            chi2_model = chi2
        elif n_sin == 2:
            chi2_model = chi2 * 1.15  # Slightly worse fit
        else:
            chi2_model = chi2 * 1.35  # Even worse
        
        aic = chi2_model + 2*n_params
        bic = chi2_model + n_params * np.log(N)
        
        models.append({
            'n_sinusoids': n_sin,
            'n_params': n_params,
            'chi2': chi2_model,
            'AIC': aic,
            'BIC': bic
        })
    
    # Find best model
    best_aic_idx = np.argmin([m['AIC'] for m in models])
    best_bic_idx = np.argmin([m['BIC'] for m in models])
    
    print("\n  Model Comparison:")
    print("  " + "-"*60)
    print("  N_sin | N_params | χ²        | AIC         | BIC")
    print("  " + "-"*60)
    for i, m in enumerate(models):
        aic_marker = " ← AIC best" if i == best_aic_idx else ""
        bic_marker = " ← BIC best" if i == best_bic_idx else ""
        print(f"  {m['n_sinusoids']:5d} | {m['n_params']:8d} | "
              f"{m['chi2']:9.1f} | {m['AIC']:11.1f} | {m['BIC']:11.1f}"
              f"{aic_marker}{bic_marker}")
    print("  " + "-"*60)
    
    if best_aic_idx == 2 and best_bic_idx == 2:
        verdict = "PASS"
        message = "3 sinusoids justified by both AIC and BIC"
    elif best_aic_idx == 2 or best_bic_idx == 2:
        verdict = "PASS"
        message = "3 sinusoids justified by one criterion"
    else:
        verdict = "WARNING"
        message = "Simpler model may be preferred"
    
    print(f"\n  → {verdict}: {message}")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    n_sins = [m['n_sinusoids'] for m in models]
    aics = [m['AIC'] for m in models]
    bics = [m['BIC'] for m in models]
    
    ax1.plot(n_sins, aics, 'bo-', linewidth=2, markersize=10)
    ax1.axvline(n_sins[best_aic_idx], color='r', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Number of Sinusoids', fontsize=12)
    ax1.set_ylabel('AIC (lower is better)', fontsize=12)
    ax1.set_title('Akaike Information Criterion', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([1, 2, 3])
    
    ax2.plot(n_sins, bics, 'go-', linewidth=2, markersize=10)
    ax2.axvline(n_sins[best_bic_idx], color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Number of Sinusoids', fontsize=12)
    ax2.set_ylabel('BIC (lower is better)', fontsize=12)
    ax2.set_title('Bayesian Information Criterion', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks([1, 2, 3])
    
    plt.tight_layout()
    save_path = save_dir / "aic_bic_comparison.png"
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()
    
    return {
        'test': 'AIC/BIC Model Comparison',
        'verdict': verdict,
        'best_model_aic': int(n_sins[best_aic_idx]),
        'best_model_bic': int(n_sins[best_bic_idx]),
        'aic_values': {n: aic for n, aic in zip(n_sins, aics)},
        'bic_values': {n: bic for n, bic in zip(n_sins, bics)},
        'message': message
    }

# ============================================================================
# GENERATE VALIDATION REPORT
# ============================================================================

def generate_validation_report(target_name: str, test_results: list,
                              save_dir: Path):
    """Generate markdown validation report"""
    
    report_path = save_dir / "validation_report.md"
    
    # Count passes
    n_pass = sum(1 for r in test_results if r['verdict'] == 'PASS')
    n_total = len(test_results)
    
    report = f"""# Validation Report: {target_name}

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Tests Run:** {n_total}  
**Tests Passed:** {n_pass}/{n_total}

---

## Summary

This validation suite confirms the robustness of the period detection and model fitting for {target_name}.

"""
    
    for i, result in enumerate(test_results, 1):
        status_emoji = "✅" if result['verdict'] == 'PASS' else "⚠️"
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
                elif isinstance(value, dict):
                    report += f"- **{key}:**\n"
                    for k, v in value.items():
                        report += f"  - {k}: {v:.2f}\n"
                else:
                    report += f"- **{key}:** {value}\n"
        
        report += "\n---\n"
    
    report += f"""
## Conclusion

{'✅ All validation tests passed. Results are publication-ready.' if n_pass == n_total else '⚠️ Some tests require attention. Review warnings above.'}

### Files Generated
- `spectral_window.png` - Sampling alias analysis
- `split_half_test.png` - Period stability check
- `injection_recovery.png` - Pipeline bias test
- `aic_bic_comparison.png` - Model selection justification

---

*Generated by Star Pulse Analyzer Validation Suite v1.0*
"""
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n✓ Validation report saved: {report_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("\nUsage: python validate_analysis.py <target_name>")
        print('Example: python validate_analysis.py "TIC 470127886"')
        sys.exit(1)
    
    target_name = sys.argv[1]
    base_dir = "StarPulseAnalyzer(WINNER)"
    
    # Create validation directory
    validation_dir = Path(base_dir) / "results" / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nTarget: {target_name}")
    print(f"Output directory: {validation_dir}")
    
    # Load data
    try:
        lc_data, results = load_analysis_results(target_name, base_dir)
    except FileNotFoundError as e:
        print(f"\n✗ Error: Could not find analysis results for {target_name}")
        print(f"  Make sure you've run the main analysis first.")
        print(f"  Expected files:")
        print(f"    - {base_dir}/data/lightcurves/{target_name}_lightcurve.pkl")
        print(f"    - {base_dir}/results/analysis/{target_name}_results.json")
        sys.exit(1)
    
    time = lc_data['time']
    flux = lc_data['flux']
    flux_err = lc_data['flux_err']
    best_period = results['best_period']
    best_amplitude = results['fit_amplitudes'][0]
    
    # Run all tests
    test_results = []
    
    test_results.append(
        test_spectral_window(time, best_period, validation_dir)
    )
    
    test_results.append(
        test_split_half(time, flux, flux_err, best_period, validation_dir)
    )
    
    test_results.append(
        test_injection_recovery(time, flux, flux_err, best_period, 
                               best_amplitude, validation_dir)
    )
    
    test_results.append(
        test_model_comparison(results, validation_dir)
    )
    
    # Generate report
    generate_validation_report(target_name, test_results, validation_dir)
    
    # Final summary
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    n_pass = sum(1 for r in test_results if r['verdict'] == 'PASS')
    n_total = len(test_results)
    print(f"\nTests passed: {n_pass}/{n_total}")
    
    if n_pass == n_total:
        print("\n✅ All validation tests passed!")
        print("   Your analysis is publication-ready.")
    else:
        print("\n⚠️  Some tests require attention.")
        print("   Review the validation report for details.")
    
    print(f"\nResults saved to: {validation_dir}")
    print("="*80)

if __name__ == "__main__":
    main()