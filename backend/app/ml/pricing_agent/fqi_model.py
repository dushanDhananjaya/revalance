"""
Revalance — FQI (Fitted Q-Iteration) Pricing Model
=====================================================
Model A: The Pricing Agent (Isuru's Model)

🎓 WHAT IS FITTED Q-ITERATION (FQI)?
═══════════════════════════════════════

Imagine you're running a taxi company, and every 15 minutes you need
to decide: "Should I set the price at 0.8x, 1.0x, 1.2x, 1.5x, or 2.0x
the base fare in each zone?"

You have historical data showing what happened when different prices
were charged. FQI learns from this data to answer:

    "In THIS situation (state), if I charge THIS price (action),
     how much total revenue will I earn now AND in the future?"

This total expectedreturn is called the Q-VALUE.

🎓 THE FQI ALGORITHM (Step by Step):
═════════════════════════════════════

1. START with training data: (State, Action, Reward, NextState) tuples
2. SET all Q-values to 0 initially
3. REPEAT for 20 iterations:
   a. For each sample, compute the TARGET:
      target = Reward + γ × max(Q(NextState, all_actions))
        ↑ immediate   ↑ discount   ↑ best future value
   b. TRAIN a Random Forest to predict: (State, Action) → target
   c. This new Random Forest IS the updated Q-function
4. SAVE the final Random Forest model

Why Random Forest instead of a table?
- With continuous states (hour, demand, price...), a lookup table
  would need infinite entries
- Random Forest can GENERALIZE: "I've never seen demand=47, but
  I've seen 45 and 50, so I can estimate 47"

Key Hyperparameters:
- γ (gamma) = 0.95  → How much we care about future rewards
  (0 = only care about NOW, 1 = care equally about future)
- n_iterations = 20 → How many times we refine the Q-function
- n_estimators = 100 → Number of trees in the Random Forest
"""

import os
import json
from typing import Optional

import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor


# The 5 possible price multipliers (discrete action space)
PRICE_ACTIONS = [0.8, 1.0, 1.2, 1.5, 2.0]

# State feature names (order matters — must match training data)
STATE_FEATURES = [
    "hour_of_day",       # 0-23
    "day_of_week",       # 0-6 (Mon-Sun)
    "is_weekend",        # 0 or 1
    "zone_id",           # 1-263
    "demand_level",      # 0-5 (bucketed)
    "supply_level",      # 0-5 (bucketed)
    "competitor_price_idx",  # ~0.7-1.5
]


class FQIPricingModel:
    """
    Fitted Q-Iteration model for dynamic pricing decisions.
    
    Usage:
        # Training
        model = FQIPricingModel()
        model.train(states, actions, rewards, next_states)
        model.save("model.joblib")
        
        # Inference
        model = FQIPricingModel.load("model.joblib")
        best_action = model.predict_best_action(state)
        q_values = model.get_q_values(state)
    """
    
    def __init__(
        self,
        gamma: float = 0.95,
        n_iterations: int = 20,
        n_estimators: int = 100,
        max_depth: int = 12,
        random_state: int = 42,
    ):
        """
        🎓 HYPERPARAMETERS EXPLAINED:
        
        gamma (γ): Discount factor (0-1)
            Controls how much we value future rewards vs immediate ones.
            0.95 means: "A dollar tomorrow is worth 95 cents today"
            
        n_iterations: Number of FQI refinement loops
            More iterations = better Q-function, but diminishing returns
            after ~20 iterations.
            
        n_estimators: Number of trees in Random Forest
            More trees = more stable predictions but slower inference.
            100 is a good balance.
            
        max_depth: Maximum tree depth
            Deeper trees = more complex patterns but risk overfitting.
            12 levels = can capture rich interactions between features.
        """
        self.gamma = gamma
        self.n_iterations = n_iterations
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        self.model: Optional[RandomForestRegressor] = None
        self.training_history: list[dict] = []
    
    def _build_features(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """
        Combine state and action into a single feature vector.
        
        🎓 WHY COMBINE STATE + ACTION?
        The Q-function maps (state, action) → value.
        To train a regressor, we need ONE input vector, so we
        concatenate: [hour, day, weekend, zone, demand, supply, price, ACTION]
        
        The model learns that the same state with different actions
        produces different values.
        """
        # Ensure 2D arrays
        if states.ndim == 1:
            states = states.reshape(1, -1)
        if actions.ndim == 0 or (actions.ndim == 1 and len(actions.shape) == 1):
            actions = actions.reshape(-1, 1)
        
        return np.hstack([states, actions])
    
    def _compute_targets(
        self,
        rewards: np.ndarray,
        next_states: np.ndarray,
    ) -> np.ndarray:
        """
        Compute regression targets for FQI update.
        
        🎓 THE BELLMAN EQUATION (heart of RL):
        
        Q(s, a) = R + γ × max_a' Q(s', a')
        
        In English:
        "The value of taking action a in state s equals:
         the immediate reward R   (money we earn NOW)
         PLUS a discounted estimate of the best we can do
         from the next state s'"
         
        If our current Q-model is None (first iteration),
        targets are just the raw rewards.
        
        🎓 VECTORIZATION TRICK:
        Instead of looping over each sample (SLOW), we compute
        Q(s', a) for ALL samples at once for EACH action.
        This turns N×5 individual predictions into just 5 batch
        predictions — 1000x faster!
        """
        if self.model is None:
            # First iteration: Q = 0, so target = reward
            return rewards
        
        # BATCH computation: predict Q(next_states, a) for each action at once
        n_samples = len(next_states)
        if next_states.ndim == 1:
            next_states = next_states.reshape(1, -1)
        
        # For each possible action, predict Q-values for ALL next_states at once
        q_all_actions = np.zeros((n_samples, len(PRICE_ACTIONS)))
        
        for j, action in enumerate(PRICE_ACTIONS):
            action_col = np.full((n_samples, 1), action)
            features = np.hstack([next_states, action_col])
            q_all_actions[:, j] = self.model.predict(features)
        
        # Take the max across actions for each sample
        best_future_values = q_all_actions.max(axis=1)
        
        # Bellman target: R + γ × max_a' Q(s', a')
        targets = rewards + self.gamma * best_future_values
        return targets
    
    def train(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        verbose: bool = True,
    ) -> list[dict]:
        """
        Train the FQI model.
        
        🎓 THE FQI TRAINING LOOP:
        
        Iteration 1: Target = Reward (we know nothing about the future)
            → Train RF to predict: given (state, action), what reward?
            
        Iteration 2: Target = Reward + γ × max Q(next_state)
            → Now we know SOMETHING about the future from iteration 1
            → RF learns better values
            
        Iteration 3: Even better future estimates...
        ...
        Iteration 20: Q-function has CONVERGED — predictions stabilize
        
        Parameters:
            states: (N, 7) array of state features
            actions: (N,) array of price actions taken
            rewards: (N,) array of rewards received
            next_states: (N, 7) array of next-state features
        """
        n_samples = len(states)
        
        if verbose:
            print(f"\n{'═'*60}")
            print(f"  FQI TRAINING")
            print(f"  Samples: {n_samples:,} | Iterations: {self.n_iterations}")
            print(f"  γ={self.gamma} | Trees={self.n_estimators} | Depth={self.max_depth}")
            print(f"{'═'*60}\n")
        
        self.training_history = []
        
        for iteration in range(self.n_iterations):
            # Step 1: Compute targets using current Q-function
            targets = self._compute_targets(rewards, next_states)
            
            # Step 2: Build feature matrix (state + action)
            features = self._build_features(states, actions)
            
            # Step 3: Train new Random Forest on (features → targets)
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,  # Use all CPU cores
            )
            self.model.fit(features, targets)
            
            # Step 4: Evaluate convergence
            predictions = self.model.predict(features)
            mse = np.mean((predictions - targets) ** 2)
            mean_q = np.mean(predictions)
            
            history_entry = {
                "iteration": iteration + 1,
                "mse": float(mse),
                "mean_q": float(mean_q),
                "mean_target": float(np.mean(targets)),
            }
            self.training_history.append(history_entry)
            
            if verbose:
                print(f"  Iteration {iteration + 1:>2}/{self.n_iterations} │ "
                      f"MSE: {mse:>10.2f} │ "
                      f"Mean Q: {mean_q:>8.2f} │ "
                      f"Mean Target: {np.mean(targets):>8.2f}")
        
        if verbose:
            print(f"\n  ✅ Training complete!")
            print(f"  Final MSE: {self.training_history[-1]['mse']:.2f}")
        
        return self.training_history
    
    def get_q_values(self, state: np.ndarray) -> dict[float, float]:
        """
        Get Q-value for each possible action in the given state.
        
        Returns:
            {action: q_value} for all 5 price actions
            
        Example:
            {0.8: 45.2, 1.0: 52.1, 1.2: 58.3, 1.5: 51.0, 2.0: 38.5}
            → Best action is 1.2x (highest Q-value of 58.3)
        """
        if self.model is None:
            # No model trained yet — return zeros
            return {a: 0.0 for a in PRICE_ACTIONS}
        
        state = np.array(state).flatten()
        q_values = {}
        
        for action in PRICE_ACTIONS:
            features = self._build_features(state, np.array([action]))
            q_values[action] = float(self.model.predict(features)[0])
        
        return q_values
    
    def predict_best_action(self, state: np.ndarray) -> tuple[float, float]:
        """
        Predict the best price multiplier for the given state.
        
        Returns:
            (best_action, q_value) — e.g., (1.2, 58.3)
        """
        q_values = self.get_q_values(state)
        best_action = max(q_values, key=q_values.get)
        return best_action, q_values[best_action]
    
    def get_top_actions(self, state: np.ndarray, n: int = 3) -> list[tuple[float, float]]:
        """
        Get the top-N actions by Q-value (used by MCTS for pruning).
        
        Returns:
            List of (action, q_value) sorted by value descending
        """
        q_values = self.get_q_values(state)
        sorted_actions = sorted(q_values.items(), key=lambda x: x[1], reverse=True)
        return sorted_actions[:n]
    
    def save(self, filepath: str):
        """Save the trained model to disk using joblib."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        save_data = {
            "model": self.model,
            "gamma": self.gamma,
            "n_iterations": self.n_iterations,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "training_history": self.training_history,
        }
        joblib.dump(save_data, filepath)
        print(f"  💾 Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "FQIPricingModel":
        """Load a trained model from disk."""
        data = joblib.load(filepath)
        instance = cls(
            gamma=data["gamma"],
            n_iterations=data["n_iterations"],
            n_estimators=data["n_estimators"],
            max_depth=data["max_depth"],
        )
        instance.model = data["model"]
        instance.training_history = data.get("training_history", [])
        print(f"  📂 Model loaded from {filepath}")
        return instance
