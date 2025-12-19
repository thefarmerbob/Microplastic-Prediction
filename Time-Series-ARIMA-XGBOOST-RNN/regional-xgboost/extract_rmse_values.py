#!/usr/bin/env python3
"""
Extract RMSE Values from XGBoost Models
========================================
This script extracts RMSE values from Optuna study databases and calculates
improvement percentages by comparing worst vs best trials.
"""

import optuna
from pathlib import Path
import json
import sqlite3

def extract_optuna_rmse(study_db_path, study_name):
    """Extract RMSE values from Optuna study database."""
    try:
        storage = f'sqlite:///{study_db_path}'
        study = optuna.load_study(study_name=study_name, storage=storage)
        
        if len(study.trials) == 0:
            return None
        
        # Get best (lowest) RMSE
        best_rmse = study.best_value
        best_params = study.best_params
        
        # Get worst (highest) RMSE from completed trials
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
        if len(completed_trials) == 0:
            return None
        
        worst_rmse = max([t.value for t in completed_trials])
        first_rmse = completed_trials[0].value if len(completed_trials) > 0 else None
        
        # Calculate improvement from worst to best
        if worst_rmse > 0:
            improvement_from_worst = ((worst_rmse - best_rmse) / worst_rmse) * 100
        else:
            improvement_from_worst = 0
        
        # Calculate improvement from first to best (simulating baseline)
        if first_rmse and first_rmse > 0:
            improvement_from_first = ((first_rmse - best_rmse) / first_rmse) * 100
        else:
            improvement_from_first = 0
        
        return {
            'best_rmse': best_rmse,
            'worst_rmse': worst_rmse,
            'first_rmse': first_rmse,
            'num_trials': len(completed_trials),
            'improvement_from_worst_pct': improvement_from_worst,
            'improvement_from_first_pct': improvement_from_first,
            'best_params': best_params
        }
    except Exception as e:
        print(f"Error loading study {study_name}: {e}")
        return None

def extract_from_db_direct(db_path):
    """Extract RMSE values directly from SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get study ID
        cursor.execute("SELECT study_id, study_name FROM studies LIMIT 1")
        study_row = cursor.fetchone()
        if not study_row:
            return None
        
        study_id = study_row[0]
        
        # Get all trial values
        cursor.execute("""
            SELECT value, state 
            FROM trials 
            WHERE study_id = ? AND state = 1
            ORDER BY trial_id
        """, (study_id,))
        
        values = [row[0] for row in cursor.fetchall() if row[0] is not None]
        
        if len(values) == 0:
            return None
        
        best_rmse = min(values)
        worst_rmse = max(values)
        first_rmse = values[0] if len(values) > 0 else None
        
        # Calculate improvements
        improvement_from_worst = ((worst_rmse - best_rmse) / worst_rmse) * 100 if worst_rmse > 0 else 0
        improvement_from_first = ((first_rmse - best_rmse) / first_rmse) * 100 if first_rmse and first_rmse > 0 else 0
        
        conn.close()
        
        return {
            'best_rmse': best_rmse,
            'worst_rmse': worst_rmse,
            'first_rmse': first_rmse,
            'num_trials': len(values),
            'improvement_from_worst_pct': improvement_from_worst,
            'improvement_from_first_pct': improvement_from_first
        }
    except Exception as e:
        print(f"Error reading database directly: {e}")
        return None

def main():
    """Main function to extract RMSE values from all regional models."""
    print("="*70)
    print("EXTRACTING RMSE VALUES FROM XGBOOST MODELS")
    print("="*70)
    
    regions = {
        'global': ('global_optuna_study.db', 'global_xgb_optimization'),
        'kyoto': ('kyoto_optuna_study.db', 'kyoto_xgb_optimization'),
        'osaka': ('osaka_optuna_study.db', 'osaka_xgb_optimization'),
        'tokyo': ('tokyo_optuna_study.db', 'tokyo_xgb_optimization'),
        'tsushima': ('tsushima_optuna_study.db', 'tsushima_xgb_optimization')
    }
    
    results = {}
    all_improvements = []
    
    for region, (db_file, study_name) in regions.items():
        print(f"\n{'='*70}")
        print(f"Processing {region.upper()} region...")
        print(f"{'='*70}")
        
        db_path = Path('results') / db_file
        
        if not db_path.exists():
            print(f"  ⚠️  Database not found: {db_path}")
            continue
        
        # Try Optuna method first
        result = extract_optuna_rmse(db_path, study_name)
        
        # If that fails, try direct database access
        if result is None:
            print(f"  Trying direct database access...")
            result = extract_from_db_direct(db_path)
        
        if result:
            results[region] = result
            print(f"  ✅ Best RMSE: {result['best_rmse']:.6f}")
            print(f"  ✅ Worst RMSE: {result['worst_rmse']:.6f}")
            print(f"  ✅ First RMSE: {result.get('first_rmse', 'N/A')}")
            print(f"  ✅ Number of trials: {result['num_trials']}")
            print(f"  ✅ Improvement (worst→best): {result['improvement_from_worst_pct']:.2f}%")
            if result.get('improvement_from_first_pct'):
                print(f"  ✅ Improvement (first→best): {result['improvement_from_first_pct']:.2f}%")
                all_improvements.append(result['improvement_from_first_pct'])
        else:
            print(f"  ❌ Could not extract RMSE values")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if all_improvements:
        avg_improvement = sum(all_improvements) / len(all_improvements)
        print(f"\nAverage improvement (first trial → best): {avg_improvement:.2f}%")
        print(f"Minimum improvement: {min(all_improvements):.2f}%")
        print(f"Maximum improvement: {max(all_improvements):.2f}%")
    
    # Calculate overall average best RMSE
    if results:
        avg_best_rmse = sum([r['best_rmse'] for r in results.values()]) / len(results)
        avg_worst_rmse = sum([r['worst_rmse'] for r in results.values()]) / len(results)
        overall_improvement = ((avg_worst_rmse - avg_best_rmse) / avg_worst_rmse) * 100
        
        print(f"\nOverall average best RMSE: {avg_best_rmse:.6f}")
        print(f"Overall average worst RMSE: {avg_worst_rmse:.6f}")
        print(f"Overall improvement (worst→best): {overall_improvement:.2f}%")
    
    # Save results
    output_file = Path('results/rmse_extraction_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()




