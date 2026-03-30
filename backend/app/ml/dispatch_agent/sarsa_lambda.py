"""
Revalance — SARSA(λ) Dispatch Agent
======================================
Model B: The Driver Rebalancing Agent

🎓 SARSA vs Q-LEARNING — Key Difference
═════════════════════════════════════════

Both learn Q(state, action) values, but:

SARSA (what we use): Updates Q using the action actually TAKEN
  Q(s,a) ← Q(s,a) + α × [R + γ × Q(s',a') - Q(s,a)]
                                        ↑ actual next action

Q-learning: Updates Q using the BEST possible next action
  Q(s,a) ← Q(s,a) + α × [R + γ × max Q(s',a') - Q(s,a)]
                                        ↑ optimal next action

SARSA is "safer" — it learns what will ACTUALLY happen given
the policy we're following, not what COULD happen optimally.
Good for real-world deployment where safety matters.

🎓 WHY λ (ELIGIBILITY TRACES)?
════════════════════════════════

Regular SARSA only updates the MOST RECENT state-action pair.
But a good dispatch decision affects outcomes for MANY future steps!

Example: Moving a driver North takes 15 minutes. The REWARD
comes 15 minutes later when a passenger is matched. Without
traces, the "move north" action gets NO credit.

Eligibility traces solve this by keeping a "memory" of recent
state-action pairs. When a reward arrives, ALL recent pairs get
partial credit (decaying by λ over time):

  Step 1: Move North → trace=1.0 → slight credit
  Step 2: Wait       → trace=λ   → less credit
  Step 3: Match! R=5 → trace=λ²  → least credit
  All three actions get credit for the Step 3 reward!

λ=0: Only update most recent (plain SARSA)
λ=1: Update all visited states equally (Monte Carlo)
λ=0.8: Good middle ground (our choice)
"""

import os
import json
from typing import Optional

import numpy as np
import joblib

from app.ml.dispatch_agent.tile_coding import (
    TileCoder,
    create_dispatch_tile_coder,
    DISPATCH_ACTIONS,
    N_ACTIONS,
    DISPATCH_STATE_DIMS,
)


class SARSALambdaAgent:
    """
    SARSA(λ) agent with tile coding for driver dispatch.
    
    The agent decides where to send idle drivers:
    - Stay: remain in current zone
    - Move_North/South/East/West: move to adjacent zone
    
    Goal: Minimize passenger wait time by proactively
    positioning drivers where demand is about to appear.
    """
    
    def __init__(
        self,
        alpha: float = 0.1 / 8,   # Learning rate (divided by n_tilings)
        gamma: float = 0.95,       # Discount factor
        lam: float = 0.8,          # Lambda (trace decay)
        epsilon: float = 0.1,      # Exploration rate
        n_tilings: int = 8,
        n_tiles_per_dim: int = 4,
        max_size: int = 4096,
    ):
        """
        🎓 HYPERPARAMETERS:
        
        alpha (α): Learning rate — how fast to update weights
            Divided by n_tilings because each state activates
            n_tilings tiles, and we want the TOTAL update to be α.
            
        gamma (γ): Discount factor — same as FQI, how much
            we care about future rewards.
            
        lam (λ): Trace decay rate
            Controls how far back credit propagates.
            0.8 means: a state 5 steps ago gets 0.8^5 = 33% credit
            
        epsilon (ε): Exploration probability
            10% of the time, pick a random action to discover
            new strategies. Rest of the time, pick the best.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.epsilon = epsilon
        
        # Create tile coder
        self.tile_coder = create_dispatch_tile_coder(
            n_tilings=n_tilings,
            n_tiles_per_dim=n_tiles_per_dim,
            max_size=max_size,
        )
        
        # Weight vector — one weight per tile per action
        self.total_weights = max_size * N_ACTIONS
        self.weights = np.zeros(self.total_weights)
        
        # Eligibility traces — same shape as weights
        self.traces = np.zeros(self.total_weights)
        
        # Training statistics
        self.episode_rewards: list[float] = []
        self.episode_steps: list[int] = []
        self.training_history: list[dict] = []
    
    def get_q_value(self, state, action: int) -> float:
        """
        Compute Q(state, action) using tile coding.
        
        🎓 HOW IT WORKS:
        Q(state, action) = sum of weights at active tiles
        
        If 8 tiles are active for this (state, action):
          Q = w[tile_0] + w[tile_1] + ... + w[tile_7]
        
        Each tile covers a different region of state space,
        so nearby states share some tiles (generalization!).
        """
        tiles = self.tile_coder.get_tiles_for_action(state, action)
        return sum(self.weights[t] for t in tiles)
    
    def get_all_q_values(self, state) -> dict[str, float]:
        """Get Q-values for all actions in the given state."""
        return {
            DISPATCH_ACTIONS[a]: self.get_q_value(state, a)
            for a in range(N_ACTIONS)
        }
    
    def select_action(self, state, training: bool = True) -> int:
        """
        Select an action using ε-greedy policy.
        
        🎓 EXPLORATION vs EXPLOITATION:
        - With probability ε (10%): pick a RANDOM action (explore)
        - With probability 1-ε (90%): pick the BEST action (exploit)
        
        During training, exploration discovers new strategies.
        During deployment, set training=False to always exploit.
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(N_ACTIONS)
        
        # Greedy: pick action with highest Q-value
        q_values = [self.get_q_value(state, a) for a in range(N_ACTIONS)]
        return int(np.argmax(q_values))
    
    def select_action_name(self, state, training: bool = False) -> str:
        """Select best action and return its name (for API use)."""
        action_idx = self.select_action(state, training=training)
        return DISPATCH_ACTIONS[action_idx]
    
    def update(
        self,
        state,
        action: int,
        reward: float,
        next_state,
        next_action: int,
        done: bool = False,
    ):
        """
        SARSA(λ) update step.
        
        🎓 THE SARSA(λ) UPDATE RULE:
        
        1. Compute TD Error (how surprised we are):
           δ = R + γ × Q(s', a') - Q(s, a)
               ↑ actual    ↑ expected    ↑ what we predicted
           
           If δ > 0: reward was BETTER than expected → increase Q
           If δ < 0: reward was WORSE than expected → decrease Q
        
        2. Update eligibility traces:
           - Mark current (s, a) tiles as recently visited (trace=1)
           - Decay all other traces by γ×λ
        
        3. Update ALL weights proportional to their trace:
           w[i] += α × δ × trace[i]
           
           High trace = recently visited = gets big update
           Low trace = visited long ago = gets tiny update
        """
        # Step 1: Compute TD error
        current_q = self.get_q_value(state, action)
        
        if done:
            next_q = 0.0
        else:
            next_q = self.get_q_value(next_state, next_action)
        
        td_error = reward + self.gamma * next_q - current_q
        
        # Step 2: Update eligibility traces
        # Decay all existing traces
        self.traces *= self.gamma * self.lam
        
        # Set traces for current state-action tiles to 1
        # (replacing traces — more stable than accumulating)
        current_tiles = self.tile_coder.get_tiles_for_action(state, action)
        for tile in current_tiles:
            self.traces[tile] = 1.0  # Replacing traces
        
        # Step 3: Update weights using traces
        self.weights += self.alpha * td_error * self.traces
        
        # If episode ended, clear traces
        if done:
            self.traces.fill(0)
    
    def reset_traces(self):
        """Clear eligibility traces (call at start of each episode)."""
        self.traces.fill(0)
    
    def train_episode(self, env, max_steps: int = 96):
        """
        Train for one episode (one simulated day = 96 steps of 15 min).
        
        Parameters:
            env: Environment that provides step(action) → (state, reward, done)
            max_steps: Maximum steps per episode
            
        Returns:
            (total_reward, n_steps)
        """
        self.reset_traces()
        state = env.reset()
        action = self.select_action(state, training=True)
        
        total_reward = 0.0
        
        for step in range(max_steps):
            next_state, reward, done = env.step(action)
            next_action = self.select_action(next_state, training=True)
            
            self.update(state, action, reward, next_state, next_action, done)
            
            total_reward += reward
            state = next_state
            action = next_action
            
            if done:
                break
        
        self.episode_rewards.append(total_reward)
        self.episode_steps.append(step + 1)
        
        return total_reward, step + 1
    
    def decay_epsilon(self, decay_rate: float = 0.995, min_epsilon: float = 0.01):
        """
        Gradually reduce exploration over training.
        
        🎓 EPSILON DECAY:
        Early training: explore a lot (ε=0.1 → 10% random)
        Late training: exploit more (ε=0.01 → 1% random)
        
        The agent needs to try different actions early on to
        discover what works. Later, it should mostly use its
        learned strategy with occasional exploration.
        """
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
    
    def save(self, filepath: str):
        """Save the trained agent to disk."""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        save_data = {
            "weights": self.weights,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "lam": self.lam,
            "epsilon": self.epsilon,
            "episode_rewards": self.episode_rewards,
            "episode_steps": self.episode_steps,
            "training_history": self.training_history,
            "tile_coder_config": {
                "n_tilings": self.tile_coder.n_tilings,
                "n_tiles_per_dim": self.tile_coder.n_tiles_per_dim,
                "max_size": self.tile_coder.max_size,
            },
        }
        joblib.dump(save_data, filepath)
        print(f"  💾 SARSA agent saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "SARSALambdaAgent":
        """Load a trained agent from disk."""
        data = joblib.load(filepath)
        config = data["tile_coder_config"]
        
        agent = cls(
            alpha=data["alpha"],
            gamma=data["gamma"],
            lam=data["lam"],
            epsilon=data["epsilon"],
            n_tilings=config["n_tilings"],
            n_tiles_per_dim=config["n_tiles_per_dim"],
            max_size=config["max_size"],
        )
        agent.weights = data["weights"]
        agent.episode_rewards = data.get("episode_rewards", [])
        agent.episode_steps = data.get("episode_steps", [])
        agent.training_history = data.get("training_history", [])
        
        print(f"  📂 SARSA agent loaded from {filepath}")
        return agent
