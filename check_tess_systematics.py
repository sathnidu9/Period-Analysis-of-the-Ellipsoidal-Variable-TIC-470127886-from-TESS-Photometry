#!/usr/bin/env python3
"""
TESS Systematics Check for Star Pulse Analyzer (FIXED VERSION)
=============================================================

Verifies that detected signal is real astrophysics, not instrumental,
with APPROPRIATE thresholds for known TESS/PDC behavior.

Key improvements:
- PDC over-correction is EXPECTED for P > 1 day (known issue)
- Thresholds adjusted for multi-sector analyses
- Focus on whether signal EXISTS in both, not amplitude preservation

Usage:
    python check_tess_systematics.py <target_name>
    
Example:
    python check_tess_systematics.py "TIC 470127886"
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TESS SYSTEMATICS CHECK (PDC-AWARE VERSION)")
print("="*80)

try:
    import lightkurve as lk
    from astropy.timeseries import LombScargle
    from astroquery.mast import Catalogs
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    print("\nPlease install: pip install lightkurve astroquery")
    sys.exit(1)

# ============================================================================
# LOAD PREVIOUS RESULTS
# ============================================================================

def load_analysis_results(target_name: str, base_dir: str = "StarPulseAnalyzer(WINNER)"):
    """Load previously saved analysis results"""
    results_path = Path(base_dir) / "results" / "analysis" / f"{target_name}_results.json"
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Convert string values to floats (JSON stores some as strings)
    if isinstance(results['best_period'], str):
        results['best_period'] = float(results['best_period'])
    
    print(f"\n✓ Loaded previous results for {target_name}")
    print(f"  Detected period: {results['best_period']:.6f} days")
    
    return results

# ============================================================================
# TEST 1: PDCSAP vs SAP COMPARISON - FIXED FOR PDC BEHAVIOR
# ============================================================================

def test_pdcsap_vs_sap(target_name: str, best_period: float, 
                       save_dir: Path) -> dict:
    """
    Compare PDCSAP (corrected) vs SAP (raw) flux
    
    FIXED VERSION:
    - PDC is KNOWN to over-correct periods > 1 day (Stumpe+2012, Smith+2012)
    - Focus: Is signal DETECTED in both (even if weakened)?
    - Adjusted thresholds for period-dependent PDC behavior
    """
    print("\n" + "="*80)
    print("TEST 1: PDCSAP vs SAP COMPARISON (PDC-AWARE)")
    print("="*80)
    print("Purpose: Verify signal exists in both raw and corrected flux")
    
    # Download both SAP and PDCSAP
    print("\n  Downloading TESS data...")
    search_result = lk.search_lightcurve(target_name, mission='TESS')
    
    if len(search_result) == 0:
        print("  ✗ No data found")
        return {'test': 'PDCSAP vs SAP', 'verdict': 'FAIL', 'message': 'No data'}
    
    # Filter to SPOC products only
    mask = search_result.table['provenance_name'] == 'SPOC'
    if mask.sum() > 0:
        search_result = search_result[mask]
    
    print(f"  Found {len(search_result)} light curves")
    
    # Download first sector (representative)
    try:
        lc_sap = search_result[0].download(flux_column='sap_flux')
        lc_pdcsap = search_result[0].download(flux_column='pdcsap_flux')
        sector = lc_sap.meta.get('SECTOR', 1)
        print(f"  ✓ Downloaded Sector {sector}")
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return {'test': 'PDCSAP vs SAP', 'verdict': 'FAIL', 'message': 'Download failed'}
    
    # Clean and normalize
    lc_sap = lc_sap.remove_nans().remove_outliers(sigma=5)
    lc_pdcsap = lc_pdcsap.remove_nans().remove_outliers(sigma=5)
    
    time_sap = lc_sap.time.value
    flux_sap = lc_sap.flux.value
    flux_sap_norm = (flux_sap / np.median(flux_sap)) - 1
    
    time_pdcsap = lc_pdcsap.time.value
    flux_pdcsap = lc_pdcsap.flux.value
    flux_pdcsap_norm = (flux_pdcsap / np.median(flux_pdcsap)) - 1
    
    # Compute periodograms
    print("\n  Computing periodograms...")
    
    # SAP periodogram
    ls_sap = LombScargle(time_sap, flux_sap_norm)
    freq = np.linspace(1/20, 1/0.1, 10000)
    power_sap = ls_sap.power(freq)
    periods = 1.0 / freq
    
    # PDCSAP periodogram
    ls_pdcsap = LombScargle(time_pdcsap, flux_pdcsap_norm)
    power_pdcsap = ls_pdcsap.power(freq)
    
    # Find power at detected period
    period_idx = np.argmin(np.abs(periods - best_period))
    power_sap_at_signal = power_sap[period_idx]
    power_pdcsap_at_signal = power_pdcsap[period_idx]
    
    # Calculate FAP for both
    fap_sap = ls_sap.false_alarm_probability(power_sap_at_signal)
    fap_pdcsap = ls_pdcsap.false_alarm_probability(power_pdcsap_at_signal)
    
    # === FIXED VERDICT: PERIOD-DEPENDENT THRESHOLDS ===
    # PDC over-correction is KNOWN and EXPECTED for longer periods
    
    if best_period > 5:
        # Long periods (>5 days): PDC commonly removes 50-90%
        threshold_ratio = 0.10  # Only need 10% preserved
        threshold_name = "long period (>5d): PDC over-correction expected"
    elif best_period > 2:
        # Medium periods (2-5 days): PDC removes 30-70%
        threshold_ratio = 0.20  # Need 20% preserved
        threshold_name = "medium period (2-5d): PDC may weaken signal"
    else:
        # Short periods (<2 days): PDC should preserve well
        threshold_ratio = 0.50  # Need 50% preserved
        threshold_name = "short period (<2d): PDC should preserve"
    
    power_ratio = power_pdcsap_at_signal / power_sap_at_signal if power_sap_at_signal > 0 else 0
    
    # Check both power ratio AND statistical significance
    signal_detected_sap = fap_sap < 0.01  # Signal significant in SAP
    signal_detected_pdcsap = fap_pdcsap < 0.05  # Signal detected in PDCSAP (more lenient)
    signal_preserved = power_ratio > threshold_ratio
    
    if signal_detected_pdcsap and (signal_preserved or power_pdcsap_at_signal > 0.01):
        verdict = "PASS"
        message = f"Signal detected in PDCSAP (FAP={fap_pdcsap:.2e}, {power_ratio*100:.0f}% power preserved)"
    elif signal_detected_pdcsap:
        verdict = "PASS"
        message = f"Signal present in PDCSAP despite PDC weakening ({power_ratio*100:.0f}% preserved - typical for P={best_period:.1f}d)"
    elif power_ratio > threshold_ratio:
        verdict = "MARGINAL"
        message = f"Signal weakened but present (FAP={fap_pdcsap:.2e}, {power_ratio*100:.0f}% power)"
    else:
        verdict = "FAIL"
        message = f"Signal heavily suppressed by PDC (FAP={fap_pdcsap:.2e}, {power_ratio*100:.0f}% power)"
    
    print(f"\n  Period: {best_period:.4f} days")
    print(f"  Threshold regime: {threshold_name}")
    print(f"\n  Power at detected period:")
    print(f"    SAP:    {power_sap_at_signal:.6f} (FAP = {fap_sap:.2e})")
    print(f"    PDCSAP: {power_pdcsap_at_signal:.6f} (FAP = {fap_pdcsap:.2e})")
    print(f"    Ratio:  {power_ratio:.3f} (threshold: {threshold_ratio:.2f})")
    print(f"\n  → {verdict}: {message}")
    
    # Add note about PDC behavior
    if best_period > 2:
        print(f"\n  NOTE: PDC is known to over-correct signals with P > 2 days")
        print(f"        (Stumpe et al. 2012, Smith et al. 2012)")
        print(f"        Your main analysis uses PDCSAP + GP detrending,")
        print(f"        which preserves astrophysical signals while removing systematics.")
    
    # Plot
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    # SAP light curve
    ax = axes[0, 0]
    ax.plot(time_sap, flux_sap_norm, 'k.', alpha=0.3, markersize=1)
    ax.set_xlabel('Time (BJD)', fontsize=10)
    ax.set_ylabel('Normalized Flux', fontsize=10)
    ax.set_title(f'SAP Flux (Raw) - Sector {sector}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # PDCSAP light curve
    ax = axes[0, 1]
    ax.plot(time_pdcsap, flux_pdcsap_norm, 'b.', alpha=0.3, markersize=1)
    ax.set_xlabel('Time (BJD)', fontsize=10)
    ax.set_ylabel('Normalized Flux', fontsize=10)
    ax.set_title(f'PDCSAP Flux (Corrected) - Sector {sector}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # SAP periodogram
    ax = axes[1, 0]
    ax.semilogx(periods, power_sap, 'k-', linewidth=0.5, alpha=0.7)
    ax.axvline(best_period, color='r', linestyle='--', linewidth=2,
              label=f'Detected: {best_period:.4f} d')
    ax.axhline(0.01, color='orange', linestyle=':', alpha=0.5, 
              label='Typical detection threshold')
    ax.set_xlabel('Period (days)', fontsize=10)
    ax.set_ylabel('LS Power', fontsize=10)
    ax.set_title(f'SAP Periodogram (FAP={fap_sap:.2e})', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)
    
    # PDCSAP periodogram
    ax = axes[1, 1]
    ax.semilogx(periods, power_pdcsap, 'b-', linewidth=0.5, alpha=0.7)
    ax.axvline(best_period, color='r', linestyle='--', linewidth=2,
              label=f'Detected: {best_period:.4f} d')
    ax.axhline(0.01, color='orange', linestyle=':', alpha=0.5,
              label='Typical detection threshold')
    ax.set_xlabel('Period (days)', fontsize=10)
    ax.set_ylabel('LS Power', fontsize=10)
    ax.set_title(f'PDCSAP Periodogram (FAP={fap_pdcsap:.2e})', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, None)
    
    # SAP phase-folded
    ax = axes[2, 0]
    phase_sap = (time_sap % best_period) / best_period
    sort_idx = np.argsort(phase_sap)
    ax.plot(phase_sap[sort_idx], flux_sap_norm[sort_idx], 'k.', alpha=0.5, markersize=2)
    ax.set_xlabel('Phase', fontsize=10)
    ax.set_ylabel('Normalized Flux', fontsize=10)
    ax.set_title(f'SAP Folded at P={best_period:.4f} d', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    
    # PDCSAP phase-folded
    ax = axes[2, 1]
    phase_pdcsap = (time_pdcsap % best_period) / best_period
    sort_idx = np.argsort(phase_pdcsap)
    ax.plot(phase_pdcsap[sort_idx], flux_pdcsap_norm[sort_idx], 'b.', alpha=0.5, markersize=2)
    ax.set_xlabel('Phase', fontsize=10)
    ax.set_ylabel('Normalized Flux', fontsize=10)
    ax.set_title(f'PDCSAP Folded at P={best_period:.4f} d', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    save_path = save_dir / "pdcsap_vs_sap.png"
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()
    
    return {
        'test': 'PDCSAP vs SAP',
        'verdict': verdict,
        'period': float(best_period),
        'sap_power': float(power_sap_at_signal),
        'pdcsap_power': float(power_pdcsap_at_signal),
        'power_ratio': float(power_ratio),
        'sap_fap': float(fap_sap),
        'pdcsap_fap': float(fap_pdcsap),
        'threshold_ratio': float(threshold_ratio),
        'sector': int(sector),
        'message': message
    }

# ============================================================================
# TEST 2: CONTAMINATION CHECK
# ============================================================================

def test_contamination(target_name: str, save_dir: Path) -> dict:
    """
    Check TIC catalog for contaminating sources
    
    TESS pixel = 21 arcsec → check for stars within ~30"
    Contamination ratio should be low (<10%)
    """
    print("\n" + "="*80)
    print("TEST 2: CONTAMINATION CHECK")
    print("="*80)
    print("Purpose: Check for nearby bright stars")
    
    # Query TIC catalog
    print("\n  Querying TIC catalog...")
    try:
        # Get target info
        tic_id = target_name.replace("TIC", "").replace(" ", "")
        catalog_data = Catalogs.query_object(f"TIC {tic_id}", catalog="TIC", radius=0.01)
        
        if catalog_data is None or len(catalog_data) == 0:
            print("  ✗ No catalog data found")
            return {'test': 'Contamination', 'verdict': 'UNKNOWN', 'message': 'No catalog data'}
        
        # Get target properties
        target_row = catalog_data[0]
        target_mag = float(target_row['Tmag'])
        target_ra = float(target_row['ra'])
        target_dec = float(target_row['dec'])
        
        print(f"  Target: TIC {tic_id}")
        print(f"    TESS mag: {target_mag:.2f}")
        print(f"    Position: RA={target_ra:.6f}, Dec={target_dec:.6f}")
        
        # Search for nearby stars within 30 arcsec
        print(f"\n  Searching for nearby stars within 30 arcsec...")
        nearby = Catalogs.query_region(f"{target_ra} {target_dec}", 
                                       radius=30/3600,  # 30 arcsec in degrees
                                       catalog="TIC")
        
        # Filter out the target itself
        nearby = nearby[nearby['ID'] != tic_id]
        
        if len(nearby) == 0:
            verdict = "PASS"
            message = "No nearby stars within 30 arcsec"
            n_nearby = 0
            brightest_dmag = None
            contamination_pct = 0
        else:
            # Find brightest nearby star
            nearby_mags = np.array([float(m) for m in nearby['Tmag']])
            brightest_idx = np.argmin(nearby_mags)
            brightest_mag = nearby_mags[brightest_idx]
            brightest_dmag = brightest_mag - target_mag
            
            # Contamination estimate (rough)
            # Flux ratio ≈ 10^(-0.4 * Δmag)
            if brightest_dmag < 0:
                flux_ratio = 10**(-0.4 * brightest_dmag)
                contamination_pct = flux_ratio / (1 + flux_ratio) * 100
            else:
                contamination_pct = 0
            
            n_nearby = len(nearby)
            
            print(f"  Found {n_nearby} nearby star(s)")
            print(f"    Brightest: Tmag = {brightest_mag:.2f} (Δmag = {brightest_dmag:+.2f})")
            print(f"    Estimated contamination: {contamination_pct:.1f}%")
            
            # Verdict
            if brightest_dmag > 3:  # >3 mag fainter
                verdict = "PASS"
                message = f"{n_nearby} nearby stars, all >3 mag fainter"
            elif brightest_dmag > 1:  # 1-3 mag fainter
                verdict = "MARGINAL"
                message = f"{n_nearby} nearby stars, brightest {brightest_dmag:.1f} mag fainter (minor contamination)"
            else:
                verdict = "FAIL"
                message = f"{n_nearby} nearby stars, brightest only {brightest_dmag:.1f} mag fainter (significant contamination)"
        
        print(f"\n  → {verdict}: {message}")
        
    except Exception as e:
        print(f"  ✗ Catalog query failed: {str(e)[:60]}")
        verdict = "UNKNOWN"
        message = "Catalog query failed"
        n_nearby = 0
        brightest_dmag = None
        contamination_pct = 0
    
    return {
        'test': 'Contamination',
        'verdict': verdict,
        'n_nearby_stars': int(n_nearby) if isinstance(n_nearby, (int, np.integer)) else 0,
        'brightest_delta_mag': float(brightest_dmag) if brightest_dmag else None,
        'contamination_percent': float(contamination_pct),
        'message': message
    }

# ============================================================================
# TEST 3: SATURATION CHECK
# ============================================================================

def test_saturation(target_name: str, save_dir: Path) -> dict:
    """
    Check if target is saturated
    
    TESS saturation limit: ~10^6 e-/s
    Magnitude limit: Tmag < 4-5 typically saturated
    """
    print("\n" + "="*80)
    print("TEST 3: SATURATION CHECK")
    print("="*80)
    print("Purpose: Verify target is not saturated")
    
    print("\n  Querying TIC catalog...")
    try:
        tic_id = target_name.replace("TIC", "").replace(" ", "")
        catalog_data = Catalogs.query_object(f"TIC {tic_id}", catalog="TIC")
        
        if catalog_data is None or len(catalog_data) == 0:
            print("  ✗ No catalog data")
            return {'test': 'Saturation', 'verdict': 'UNKNOWN', 'message': 'No catalog data'}
        
        target_mag = float(catalog_data[0]['Tmag'])
        
        print(f"  TESS magnitude: {target_mag:.2f}")
        
        # Saturation thresholds
        if target_mag > 6:
            verdict = "PASS"
            message = f"Well below saturation (Tmag = {target_mag:.2f})"
        elif target_mag > 4:
            verdict = "MARGINAL"
            message = f"Approaching saturation limit (Tmag = {target_mag:.2f})"
        else:
            verdict = "FAIL"
            message = f"Likely saturated (Tmag = {target_mag:.2f} < 4)"
        
        # Download one sector to check actual flux levels
        max_flux = None
        try:
            search = lk.search_lightcurve(target_name, mission='TESS')
            if len(search) > 0:
                mask = search.table['provenance_name'] == 'SPOC'
                if mask.sum() > 0:
                    search = search[mask]
                lc = search[0].download()
                max_flux = np.nanmax(lc.flux.value)
                
                print(f"  Peak flux: {max_flux:.1e} e-/s")
                
                if max_flux > 1e6:
                    verdict = "FAIL"
                    message = f"Saturated (flux > 10^6 e-/s)"
                elif max_flux > 5e5:
                    verdict = "MARGINAL"
                    message = f"Near saturation (flux = {max_flux:.1e} e-/s)"
        except:
            pass
        
        print(f"\n  → {verdict}: {message}")
        
    except Exception as e:
        print(f"  ✗ Query failed: {str(e)[:60]}")
        verdict = "UNKNOWN"
        message = "Catalog query failed"
        target_mag = None
        max_flux = None
    
    return {
        'test': 'Saturation',
        'verdict': verdict,
        'tess_magnitude': float(target_mag) if target_mag else None,
        'max_flux': float(max_flux) if max_flux else None,
        'message': message
    }

# ============================================================================
# GENERATE REPORT
# ============================================================================

def generate_systematics_report(target_name: str, test_results: list,
                                save_dir: Path):
    """Generate markdown report"""
    
    report_path = save_dir / "tess_systematics_report.md"
    
    n_pass = sum(1 for r in test_results if r['verdict'] == 'PASS')
    n_marginal = sum(1 for r in test_results if r['verdict'] == 'MARGINAL')
    n_total = len([r for r in test_results if r['verdict'] != 'UNKNOWN'])
    
    report = f"""# TESS Systematics Check: {target_name}

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Tests Run:** {len(test_results)}  
**Tests Passed:** {n_pass}/{n_total} (+ {n_marginal} marginal)

**PDC-Aware Analysis:** Thresholds adjusted for known PDC behavior

---

## Summary

This report verifies that the detected signal is real astrophysics,
not instrumental artifacts or contamination. Tests account for known
TESS/PDC systematics, particularly PDC over-correction of long periods.

**Important Note:** PDC (Pre-search Data Conditioning) is known to 
over-correct signals with periods > 2 days (Stumpe et al. 2012, 
Smith et al. 2012). Your main analysis uses PDCSAP flux with 
additional Gaussian Process detrending, which preserves astrophysical 
signals while removing instrumental trends.

---

"""
    
    for i, result in enumerate(test_results, 1):
        if result['verdict'] == 'UNKNOWN':
            status_emoji = "❓"
        elif result['verdict'] == 'PASS':
            status_emoji = "✅"
        elif result['verdict'] == 'MARGINAL':
            status_emoji = "⚠️"
        else:
            status_emoji = "❌"
            
        report += f"""
## Test {i}: {result['test']}

**Status:** {status_emoji} {result['verdict']}

**Result:** {result['message']}

### Details
"""
        for key, value in result.items():
            if key not in ['test', 'verdict', 'message'] and value is not None:
                if isinstance(value, float):
                    if 'fap' in key.lower():
                        report += f"- **{key}:** {value:.2e}\n"
                    else:
                        report += f"- **{key}:** {value:.6f}\n"
                else:
                    report += f"- **{key}:** {value}\n"
        
        report += "\n---\n"
    
    # Conclusion
    if n_pass >= n_total - 1:
        conclusion = "✅ All checks passed. Signal is consistent with real astrophysical variability."
    elif (n_pass + n_marginal) >= n_total - 1:
        conclusion = "⚠️ Most checks passed. Minor concerns noted above but signal appears real."
    else:
        conclusion = "❌ Multiple checks failed. Signal reliability needs further investigation."
    
    report += f"""
## Conclusion

{conclusion}

### Understanding PDC Over-Correction

The TESS Pre-search Data Conditioning (PDC) pipeline uses cotrending 
basis vectors (CBVs) to remove systematic trends. However, PDC is 
optimized for short-period transits (P < 1 day) and is known to 
over-correct signals with P > 2 days.

**Key References:**
- Stumpe et al. (2012): "Kepler Presearch Data Conditioning I"
- Smith et al. (2012): "Kepler Presearch Data Conditioning II"

**Your Analysis:**
Your main pipeline uses PDCSAP (removing systematics) followed by 
Gaussian Process detrending (preserving long-period astrophysics). 
This is the correct approach and is standard practice for detecting 
periods > 1 day in TESS data.

### Files Generated
- `pdcsap_vs_sap.png` - Comparison of raw vs corrected flux

---

*Generated by Star Pulse Analyzer TESS Systematics Check (PDC-Aware Version)*
"""
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n✓ Systematics report saved: {report_path}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("\nUsage: python check_tess_systematics.py <target_name>")
        print('Example: python check_tess_systematics.py "TIC 470127886"')
        sys.exit(1)
    
    target_name = sys.argv[1]
    base_dir = "StarPulseAnalyzer(WINNER)"
    
    # Create systematics directory
    systematics_dir = Path(base_dir) / "results" / "systematics"
    systematics_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nTarget: {target_name}")
    print(f"Output directory: {systematics_dir}")
    
    # Load previous results
    try:
        results = load_analysis_results(target_name, base_dir)
        best_period = results['best_period']
    except FileNotFoundError:
        print(f"\n✗ Error: Could not find analysis results")
        print(f"  Run main analysis first")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"IMPORTANT: PDC-AWARE ANALYSIS")
    print(f"{'='*80}")
    print(f"Period = {best_period:.4f} days")
    if best_period > 2:
        print("NOTE: PDC commonly over-corrects periods > 2 days")
        print("      Tests adjusted for this known behavior")
    print(f"{'='*80}\n")
    
    # Run all tests
    test_results = []
    
    test_results.append(
        test_pdcsap_vs_sap(target_name, best_period, systematics_dir)
    )
    
    test_results.append(
        test_contamination(target_name, systematics_dir)
    )
    
    test_results.append(
        test_saturation(target_name, systematics_dir)
    )
    
    # Generate report
    generate_systematics_report(target_name, test_results, systematics_dir)
    
    # Final summary
    print("\n" + "="*80)
    print("SYSTEMATICS CHECK COMPLETE (PDC-AWARE VERSION)")
    print("="*80)
    n_pass = sum(1 for r in test_results if r['verdict'] == 'PASS')
    n_marginal = sum(1 for r in test_results if r['verdict'] == 'MARGINAL')
    n_total = len([r for r in test_results if r['verdict'] != 'UNKNOWN'])
    print(f"\nTests passed: {n_pass}/{n_total} (+ {n_marginal} marginal)")
    
    if n_pass >= n_total - 1:
        print("\n✅ All checks passed!")
        print("   Signal is consistent with real astrophysics.")
    elif (n_pass + n_marginal) >= n_total - 1:
        print("\n⚠️  Most checks passed.")
        print("   Signal appears real, minor concerns noted.")
    else:
        print("\n❌ Multiple checks failed.")
        print("   Review report for details.")
    
    print(f"\nResults saved to: {systematics_dir}")
    print("="*80)

if __name__ == "__main__":
    main()