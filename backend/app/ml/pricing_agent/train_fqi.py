"""
Revalance — FQI Training Script (Improved)
=============================================
Offline training for the Pricing Agent (Model A).

🎓 IMPROVEMENTS IN THIS VERSION:
1. Proper train/test split (80/20)
2. R² score, MAE, and Q-value convergence tracking
3. Feature importance analysis
4. Demand-aware evaluation (high-demand vs low-demand pricing)
5. Normalized reward function for stable training
6. More data (500K rows) and tuned hyperparameters
"""

import os
import sys
import time
import json

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from app.ml.pricing_agent.fqi_model import FQIPricingModel, STATE_FEATURES, PRICE_ACTIONS
from app.data.preprocess import engineer_features, build_fqi_training_data


def prepare_training_arrays(sars_df: pd.DataFrame):
    """Convert SARS DataFrame into numpy arrays for training."""
    state_cols = [f"s_{f}" for f in STATE_FEATURES]
    next_state_cols = [f"ns_{f}" for f in STATE_FEATURES]
    
    states = sars_df[state_cols].values.astype(np.float64)
    actions = sars_df["action"].values.astype(np.float64)
    rewards = sars_df["reward"].values.astype(np.float64)
    next_states = sars_df[next_state_cols].values.astype(np.float64)
    
    return states, actions, rewards, next_states


def evaluate_model(model: FQIPricingModel, states: np.ndarray, 
                   actions: np.ndarray, rewards: np.ndarray,
                   next_states: np.ndarray, n_samples: int = 2000):
    """
    Comprehensive model evaluation with proper ML metrics.
    
    🎓 EVALUATION METRICS:
    1. R² Score: How much variance in Q-values does the model explain?
       - 1.0 = perfect, 0.0 = as good as predicting the mean, <0 = terrible
    2. MAE: Average prediction error in absolute terms
    3. Action Distribution: Does the model diversify its pricing?
    4. Demand-Aware Analysis: Does it charge MORE in high demand?
    5. Feature Importance: Which features matter most?
    """
    print(f"\n{'═'*60}")
    print(f"  COMPREHENSIVE MODEL EVALUATION")
    print(f"{'═'*60}")
    
    # Sample for evaluation
    n = min(n_samples, len(states))
    idx = np.random.choice(len(states), n, replace=False)
    eval_states = states[idx]
    eval_actions = actions[idx]
    eval_rewards = rewards[idx]
    eval_next_states = next_states[idx]
    
    # ── 1. R² Score on Q-value predictions ──
    features = np.hstack([eval_states, eval_actions.reshape(-1, 1)])
    predictions = model.model.predict(features)
    
    # Compute actual targets for the test samples
    targets = model._compute_targets(eval_rewards, eval_next_states)
    
    r2 = r2_score(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    
    print(f"\n  📐 Regression Metrics (on evaluation set):")
    print(f"     R² Score:           {r2:.4f}  {'✅ Good' if r2 > 0.7 else '⚠️ Needs improvement' if r2 > 0.4 else '❌ Poor'}")
    print(f"     MAE:                {mae:.4f}")
    print(f"     Mean Q Prediction:  {predictions.mean():.4f}")
    print(f"     Std Q Prediction:   {predictions.std():.4f}")
    
    # ── 2. Feature Importance ──
    importances = model.model.feature_importances_
    feature_names = STATE_FEATURES + ["action"]
    sorted_idx = np.argsort(importances)[::-1]
    
    print(f"\n  🔍 Feature Importance (what the model pays attention to):")
    for rank, i in enumerate(sorted_idx):
        bar = "█" * int(importances[i] * 50)
        print(f"     {rank+1}. {feature_names[i]:<25} {importances[i]:.4f} {bar}")
    
    # ── 3. Action Distribution ──
    ai_actions = []
    ai_q_values = []
    baseline_q_values = []
    
    for state in eval_states:
        best_action, best_q = model.predict_best_action(state)
        ai_actions.append(best_action)
        ai_q_values.append(best_q)
        all_q = model.get_q_values(state)
        baseline_q_values.append(all_q.get(1.0, 0.0))
    
    ai_actions = np.array(ai_actions)
    ai_q_values = np.array(ai_q_values)
    baseline_q_values = np.array(baseline_q_values)
    
    print(f"\n  📊 Action Distribution (AI pricing decisions):")
    for action in PRICE_ACTIONS:
        count = np.sum(ai_actions == action)
        pct = count / len(ai_actions) * 100
        bar = "█" * int(pct / 2)
        print(f"     {action:.1f}x: {count:>5} ({pct:>5.1f}%) {bar}")
    
    # ── 4. AI vs Baseline Comparison ──
    ai_mean = np.mean(ai_q_values)
    baseline_mean = np.mean(baseline_q_values)
    improvement = ((ai_mean - baseline_mean) / abs(baseline_mean)) * 100 if baseline_mean != 0 else 0
    
    # How often does AI beat baseline?
    ai_wins = np.sum(ai_q_values > baseline_q_values)
    win_rate = ai_wins / len(ai_q_values) * 100
    
    print(f"\n  💰 AI vs Baseline (1.0x fixed pricing):")
    print(f"     AI Mean Q-value:       {ai_mean:>10.4f}")
    print(f"     Baseline Mean Q-value: {baseline_mean:>10.4f}")
    print(f"     Improvement:           {improvement:>+9.1f}%")
    print(f"     AI win rate:           {win_rate:>9.1f}%")
    
    # ── 5. Demand-Aware Pricing Analysis ──
    print(f"\n  ⚡ Demand-Aware Pricing (does AI surge price correctly?):")
    for demand_level in [0, 1, 2, 3, 4, 5]:
        mask = eval_states[:, 4] == demand_level  # demand_level column
        if np.any(mask):
            avg_action = np.mean(ai_actions[mask])
            n_states = np.sum(mask)
            label = ["Very Low", "Low", "Medium", "High", "Very High", "Extreme"][demand_level]
            indicator = "📉" if avg_action <= 1.0 else "📈" if avg_action >= 1.3 else "➡️"
            print(f"     Demand {demand_level} ({label:>9}): avg price = {avg_action:.2f}x "
                  f"{indicator}  (n={n_states})")
    
    # ── 6. Time-of-Day Analysis ──
    print(f"\n  ⏰ Time-of-Day Pricing:")
    for hour in [7, 8, 9, 12, 14, 17, 18, 19, 22, 2]:
        hour_mask = eval_states[:, 0] == hour
        if np.any(hour_mask):
            hour_actions = ai_actions[hour_mask]
            avg_price = np.mean(hour_actions)
            label = "RUSH" if hour in [7, 8, 9, 17, 18, 19] else "OFF-PEAK"
            print(f"     Hour {hour:02d}:00 ({label:>8}): avg multiplier = {avg_price:.2f}x")
    
    results = {
        "r2_score": float(r2),
        "mae": float(mae),
        "ai_mean_q": float(ai_mean),
        "baseline_mean_q": float(baseline_mean),
        "improvement_pct": float(improvement),
        "win_rate_pct": float(win_rate),
        "action_distribution": {str(a): int(np.sum(ai_actions == a)) for a in PRICE_ACTIONS},
        "feature_importances": {feature_names[i]: float(importances[i]) for i in range(len(feature_names))},
    }
    
    return results


def train_fqi_model(
    data_path: str = None,
    output_path: str = None,
    n_rows: int = 500_000,
    n_iterations: int = 20,
    verbose: bool = True,
):
    """
    Complete FQI training pipeline with improved evaluation.
    """
    start_time = time.time()
    
    # ── Paths ──
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
    base_dir = os.path.normpath(base_dir)
    
    if data_path is None:
        data_path = os.path.join(base_dir, "data", "raw", "yellow_tripdata_2023-01.parquet")
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "model.joblib")
    
    print(f"\n{'═'*60}")
    print(f"  REVALANCE — FQI PRICING AGENT TRAINING (Improved)")
    print(f"{'═'*60}")
    print(f"  Data source: {data_path}")
    print(f"  Rows to use: {n_rows:,}")
    print(f"  FQI iterations: {n_iterations}")
    
    # ── Step 1: Load data ──
    print(f"\n📂 Step 1: Loading trip data...")
    df = pd.read_parquet(data_path)
    df = df.head(n_rows)
    print(f"  Loaded {len(df):,} rows")
    
    # Rename columns to match our schema
    df = df.rename(columns={
        "tpep_pickup_datetime": "pickup_datetime",
        "tpep_dropoff_datetime": "dropoff_datetime",
        "PULocationID": "pickup_zone_id",
        "DOLocationID": "dropoff_zone_id",
    })
    
    # Robust filtering
    df = df[df["fare_amount"] > 0]
    df = df[df["trip_distance"] > 0]
    df = df[df["trip_distance"] <= 100]
    df = df[(df["pickup_zone_id"] >= 1) & (df["pickup_zone_id"] <= 263)]
    df = df[(df["dropoff_zone_id"] >= 1) & (df["dropoff_zone_id"] <= 263)]
    df = df[df["fare_amount"] <= 500]
    print(f"  After filtering: {len(df):,} rows")
    
    # ── Step 2: Feature engineering ──
    print(f"\n🔧 Step 2: Engineering features...")
    featured_df = engineer_features(df)
    
    # ── Step 3: Build SARS tuples ──
    print(f"\n🎯 Step 3: Building SARS training tuples...")
    sars_df = build_fqi_training_data(featured_df)
    
    # ── Step 4: Train/Test Split ──
    print(f"\n📐 Step 4: Splitting into train/test sets...")
    n_total = len(sars_df)
    n_train = int(0.8 * n_total)
    
    # Shuffle before splitting
    sars_df = sars_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    train_df = sars_df.iloc[:n_train]
    test_df = sars_df.iloc[n_train:]
    
    train_states, train_actions, train_rewards, train_next_states = prepare_training_arrays(train_df)
    test_states, test_actions, test_rewards, test_next_states = prepare_training_arrays(test_df)
    
    print(f"  Train set: {len(train_df):,} samples")
    print(f"  Test set:  {len(test_df):,} samples")
    print(f"  Train reward: mean={train_rewards.mean():.2f}, std={train_rewards.std():.2f}")
    
    # ── Step 5: Train FQI model with tuned hyperparameters ──
    print(f"\n🧠 Step 5: Training FQI model...")
    model = FQIPricingModel(
        gamma=0.95,
        n_iterations=n_iterations,
        n_estimators=200,      # More trees → more stable
        max_depth=15,          # Deeper trees → capture more patterns
    )
    history = model.train(
        train_states, train_actions, train_rewards, train_next_states,
        verbose=verbose,
    )
    
    # ── Step 6: Evaluate on TEST set ──
    print(f"\n📈 Step 6: Evaluating model on TEST set...")
    eval_results = evaluate_model(
        model, test_states, test_actions, test_rewards, test_next_states
    )
    
    # ── Step 7: Save model and results ──
    print(f"\n💾 Step 7: Saving model and evaluation results...")
    model.save(output_path)
    
    # Save evaluation results as JSON
    results_path = os.path.join(os.path.dirname(output_path), "training_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "training_history": history,
            "evaluation": eval_results,
            "config": {
                "n_rows": n_rows,
                "n_iterations": n_iterations,
                "gamma": 0.95,
                "n_estimators": 200,
                "max_depth": 15,
                "train_samples": len(train_df),
                "test_samples": len(test_df),
            },
        }, f, indent=2)
    print(f"  📄 Results saved to {results_path}")
    
    # ── Summary ──
    elapsed = time.time() - start_time
    print(f"\n{'═'*60}")
    print(f"  ✅ TRAINING COMPLETE in {elapsed:.1f} seconds")
    print(f"     Model saved to: {output_path}")
    print(f"     R² Score:     {eval_results['r2_score']:.4f}")
    print(f"     Win Rate:     {eval_results['win_rate_pct']:.1f}%")
    print(f"     Improvement:  {eval_results['improvement_pct']:+.1f}%")
    print(f"{'═'*60}")
    
    return model, eval_results


# ─── CLI entry point ───
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train FQI Pricing Agent")
    parser.add_argument("--data", default=None, help="Path to Parquet data")
    parser.add_argument("--output", default=None, help="Path to save model")
    parser.add_argument("--rows", type=int, default=500_000, help="Number of rows")
    parser.add_argument("--iterations", type=int, default=20, help="FQI iterations")
    args = parser.parse_args()
    
    train_fqi_model(
        data_path=args.data,
        output_path=args.output,
        n_rows=args.rows,
        n_iterations=args.iterations,
    )
