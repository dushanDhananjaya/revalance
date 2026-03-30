"""
Revalance — SARSA(λ) Training Script
=======================================
Online training for the Dispatch Agent (Model B).

🎓 ONLINE TRAINING (vs Offline FQI)
═════════════════════════════════════

FQI (Model A): "Here's 500K historical records. Learn offline."
  → Reads data once, no interaction needed.

SARSA (Model B): "Here's a simulation. Interact and learn."
  → Agent takes action → environment responds → agent learns → repeat
  → This is TRUE reinforcement learning!

We build a lightweight simulation environment that mimics
NYC taxi zone dynamics using the real data patterns we analyzed.

Training Loop:
  for episode in range(500):      # 500 simulated days
    for step in range(96):        # 96 × 15-min steps = 24 hours
      action = agent.select(state)
      next_state, reward = env.step(action)
      agent.update(state, action, reward, next_state)
"""

import os
import sys
import time
import json

import numpy as np
import pandas as pd

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from app.ml.dispatch_agent.sarsa_lambda import SARSALambdaAgent
from app.ml.dispatch_agent.tile_coding import DISPATCH_ACTIONS, N_ACTIONS


# ═══════════════════════════════════════════════════════════════
# DISPATCH SIMULATION ENVIRONMENT
# ═══════════════════════════════════════════════════════════════

class DispatchEnv:
    """
    Simulated environment for training the dispatch agent.
    
    🎓 WHY A SIMULATION?
    The FQI pricing agent learned from HISTORICAL data.
    But a dispatch agent needs to INTERACT with an environment
    — take actions and see what happens. We can't dispatch real
    taxis during training, so we simulate!
    
    State: [zone_id, demand_level, supply_level, hour_of_day]
    Actions: Stay, Move_North, Move_South, Move_East, Move_West
    Reward: +rides_matched - penalties, with supply competition
    """
    
    # Simplified NYC zone adjacency (representative high-traffic zones)
    ZONE_ADJACENCY = {
        161: {"N": 163, "S": 230, "E": 162, "W": 164},  # Midtown Center
        162: {"N": 163, "S": 230, "E": 170, "W": 161},  # Midtown East
        163: {"N": 236, "S": 161, "E": 234, "W": 164},  # Midtown North
        164: {"N": 107, "S": 158, "E": 161, "W": 246},  # Midtown West
        230: {"N": 161, "S": 114, "E": 144, "W": 158},  # Times Square
        170: {"N": 234, "S": 233, "E": 137, "W": 162},  # East Side
        234: {"N": 236, "S": 170, "E": 107, "W": 163},  # Upper East
        236: {"N": 236, "S": 234, "E": 237, "W": 143},  # Upper East 2 (self-loop N)
        114: {"N": 230, "S": 79, "E": 113, "W": 144},   # Greenwich
        107: {"N": 236, "S": 164, "E": 234, "W": 143},  # Central Park
        137: {"N": 170, "S": 233, "E": 137, "W": 231},  # Kips Bay (self-loop E)
        233: {"N": 170, "S": 231, "E": 137, "W": 144},  # East Village
        231: {"N": 233, "S": 231, "E": 114, "W": 79},   # SoHo (self-loop S)
        79:  {"N": 114, "S": 79, "E": 231, "W": 158},   # West Village (self-loop S)
        144: {"N": 230, "S": 231, "E": 233, "W": 79},   # Chelsea
        158: {"N": 164, "S": 79, "E": 230, "W": 246},   # Penn Station
        143: {"N": 236, "S": 107, "E": 163, "W": 246},  # Lincoln Center
        246: {"N": 143, "S": 158, "E": 164, "W": 246},  # Upper West (self-loop W)
        113: {"N": 114, "S": 231, "E": 137, "W": 79},   # Greenwich/SoHo border
        237: {"N": 236, "S": 234, "E": 237, "W": 163},  # Upper East border
    }
    
    HOURLY_DEMAND = {
        0: 0.3, 1: 0.2, 2: 0.15, 3: 0.1, 4: 0.1, 5: 0.15,
        6: 0.4, 7: 0.7, 8: 0.9, 9: 0.85, 10: 0.75, 11: 0.8,
        12: 0.85, 13: 0.8, 14: 0.85, 15: 0.8, 16: 0.75, 17: 0.9,
        18: 1.0, 19: 0.95, 20: 0.85, 21: 0.75, 22: 0.6, 23: 0.45,
    }
    
    # Zone demand multipliers — varied enough to create movement incentives
    ZONE_DEMAND_MULT = {
        161: 2.5, 162: 1.8, 163: 1.5, 164: 1.3, 230: 3.0,  # Midtown hotspots
        170: 1.2, 234: 1.0, 236: 0.8, 114: 1.8, 107: 0.5,   # Mixed
        137: 0.8, 233: 1.0, 231: 1.5, 79: 2.0, 144: 1.0,    # Lower Manhattan
        158: 1.8, 143: 0.6, 246: 0.4,                         # West Side
        113: 1.2, 237: 0.6,                                    # Border zones
    }
    
    VALID_ZONES = list(ZONE_ADJACENCY.keys())
    
    def __init__(self):
        self.step_count = 0
        self.hour = 0
        self.minute = 0
        self.state = None
        # Track how long the agent has been in the same zone
        self.consecutive_stays = 0
        self.current_zone = None
        # Simulated zone supply (resets each episode)
        self.zone_supply = {}
    
    def reset(self):
        """Reset environment — start at a RANDOM zone (not always hot ones)."""
        self.step_count = 0
        self.hour = np.random.randint(0, 24)
        self.minute = 0
        self.consecutive_stays = 0
        
        # Start in a random zone (including medium/low demand)
        driver_zone = np.random.choice(self.VALID_ZONES)
        self.current_zone = driver_zone
        
        # Initialize zone supply — each zone gets a random supply
        self.zone_supply = {}
        for z in self.VALID_ZONES:
            hot = self.ZONE_DEMAND_MULT.get(z, 1.0)
            # Hot zones attract more drivers → higher supply
            self.zone_supply[z] = int(np.clip(hot * 2 + np.random.poisson(1), 1, 8))
        
        demand = self._get_demand(driver_zone)
        supply = self._get_supply(driver_zone)
        
        self.state = [driver_zone, demand, supply, self.hour]
        return self.state
    
    def step(self, action: int):
        """Execute one 15-minute step."""
        old_zone = self.state[0]
        
        # Execute action
        new_zone = self._move(old_zone, action)
        
        # Track movement
        if new_zone == old_zone and action == 0:
            self.consecutive_stays += 1
        else:
            self.consecutive_stays = 0
        self.current_zone = new_zone
        
        # Advance time
        self.step_count += 1
        self.minute += 15
        if self.minute >= 60:
            self.minute = 0
            self.hour = (self.hour + 1) % 24
        
        # Evolve zone supply (natural drift)
        self._evolve_supply()
        
        # Compute state and reward
        demand = self._get_demand(new_zone)
        supply = self._get_supply(new_zone)
        moved = (new_zone != old_zone)
        reward = self._compute_reward(new_zone, demand, supply, moved)
        
        self.state = [new_zone, demand, supply, self.hour]
        done = self.step_count >= 96
        
        return self.state, reward, done
    
    def _move(self, zone: int, action: int) -> int:
        if action == 0:
            return zone
        direction_map = {1: "N", 2: "S", 3: "E", 4: "W"}
        direction = direction_map.get(action, "N")
        adjacency = self.ZONE_ADJACENCY.get(zone, None)
        if adjacency is None:
            return zone
        return adjacency.get(direction, zone)
    
    def _get_demand(self, zone: int) -> int:
        hourly = self.HOURLY_DEMAND.get(self.hour, 0.5)
        zone_mult = self.ZONE_DEMAND_MULT.get(zone, 1.0)
        base = hourly * zone_mult * 4
        actual = np.random.poisson(max(0.3, base))
        return min(5, actual)
    
    def _get_supply(self, zone: int) -> int:
        """Get supply level from tracked supply."""
        return min(5, max(0, self.zone_supply.get(zone, 2)))
    
    def _evolve_supply(self):
        """Supply drifts over time — drivers come and go."""
        for z in self.VALID_ZONES:
            hot = self.ZONE_DEMAND_MULT.get(z, 1.0)
            # Natural attraction to hot zones
            target_supply = int(hot * 2 + 1)
            current = self.zone_supply.get(z, 2)
            # Mean revert with noise
            delta = np.random.choice([-1, 0, 0, 1]) + (1 if current < target_supply else -1 if current > target_supply else 0)
            self.zone_supply[z] = int(np.clip(current + delta * 0.5, 0, 8))
    
    def _compute_reward(self, zone: int, demand: int, supply: int, moved: bool) -> float:
        """
        Improved reward function.
        
        🎓 KEY INSIGHT: The reward must make strategic movement
        CLEARLY better than staying in place. This means:
        - Moving to an undersupplied high-demand zone = BIG reward
        - Staying in an oversupplied zone = PENALTY
        - The magnitude difference must be large enough for SARSA to learn
        """
        reward = 0.0
        zone_hot = self.ZONE_DEMAND_MULT.get(zone, 1.0)
        
        # 1. Match probability (competitive) — supply saturates
        if demand > 0:
            # More supply = lower individual match probability
            match_prob = min(1.0, demand / max(supply + 1, 1))
            ride_revenue = match_prob * zone_hot * 2.0  # Revenue scales with zone value
            reward += ride_revenue
        
        # 2. Undersupplied bonus — being where drivers are NEEDED
        if demand > supply:
            gap = demand - supply
            reward += gap * 1.5  # Strong incentive for going where needed
        
        # 3. Oversupply penalty — too many drivers here, diminishing returns
        if supply > demand + 1:
            excess = supply - demand
            reward -= excess * 0.8
        
        # 4. Movement cost — small, so it doesn't discourage needed moves
        if moved:
            reward -= 0.1  # Very small cost
        
        # 5. Staleness penalty — staying too long in same zone
        #    Real taxi dynamics: good drivers reposition often
        if self.consecutive_stays > 3:
            reward -= 0.3 * (self.consecutive_stays - 3)
        
        # 6. Zone quality bonus — reward being in naturally busy zones
        if zone_hot >= 2.0:
            reward += 0.3
        elif zone_hot <= 0.5:
            reward -= 0.5  # Penalty for being in dead zones
        
        return reward


# ═══════════════════════════════════════════════════════════════
# TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════

def train_sarsa_agent(
    n_episodes: int = 500,
    max_steps: int = 96,
    output_path: str = None,
    verbose: bool = True,
):
    """
    Train the SARSA(λ) dispatch agent.
    
    🎓 ONLINE TRAINING LOOP:
    Unlike FQI which reads historical data, SARSA interacts
    with a simulation:
    
    for each episode (simulated day):
      reset environment
      for each step (15-min interval):
        observe state
        choose action (ε-greedy)
        take action, observe reward
        update weights using SARSA(λ)
      decay exploration rate
    """
    start_time = time.time()
    
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "sarsa_model.joblib")
    
    print(f"\n{'═'*60}")
    print(f"  REVALANCE — SARSA(λ) DISPATCH AGENT TRAINING")
    print(f"{'═'*60}")
    print(f"  Episodes: {n_episodes}")
    print(f"  Steps per episode: {max_steps}")
    print(f"  Algorithm: SARSA(λ) with Tile Coding")
    
    # Create agent and environment
    agent = SARSALambdaAgent(
        alpha=0.1 / 8,
        gamma=0.95,
        lam=0.8,
        epsilon=0.15,       # Start with 15% exploration
        n_tilings=8,
        n_tiles_per_dim=4,
        max_size=4096,
    )
    
    env = DispatchEnv()
    
    # Training loop
    print(f"\n  Training... (this takes about 30-60 seconds)\n")
    
    window_size = 50
    best_avg_reward = float('-inf')
    
    for episode in range(n_episodes):
        # Train one episode
        state = env.reset()
        action = agent.select_action(state, training=True)
        agent.reset_traces()
        
        total_reward = 0.0
        
        for step in range(max_steps):
            next_state, reward, done = env.step(action)
            next_action = agent.select_action(next_state, training=True)
            
            agent.update(state, action, reward, next_state, next_action, done)
            
            total_reward += reward
            state = next_state
            action = next_action
            
            if done:
                break
        
        agent.episode_rewards.append(total_reward)
        agent.episode_steps.append(step + 1)
        
        # Decay exploration
        agent.decay_epsilon(decay_rate=0.995, min_epsilon=0.01)
        
        # Print progress
        if verbose and (episode + 1) % 50 == 0:
            recent_rewards = agent.episode_rewards[-window_size:]
            avg_reward = np.mean(recent_rewards)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
            print(f"  Episode {episode+1:>4}/{n_episodes} │ "
                  f"Avg Reward (last {window_size}): {avg_reward:>7.2f} │ "
                  f"Epsilon: {agent.epsilon:.3f} │ "
                  f"Best: {best_avg_reward:.2f}")
    
    # Save training history
    agent.training_history = [
        {
            "episode": i + 1,
            "reward": float(agent.episode_rewards[i]),
            "steps": int(agent.episode_steps[i]),
        }
        for i in range(n_episodes)
    ]
    
    # ── Evaluation ──
    eval_results = evaluate_sarsa(agent, env, n_eval_episodes=50)
    
    # ── Save ──
    agent.save(output_path)
    
    # Save results
    results_path = os.path.join(os.path.dirname(output_path), "sarsa_training_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "training_summary": {
                "n_episodes": n_episodes,
                "final_avg_reward_50": float(np.mean(agent.episode_rewards[-50:])),
                "best_avg_reward_50": float(best_avg_reward),
                "final_epsilon": float(agent.epsilon),
            },
            "evaluation": eval_results,
        }, f, indent=2)
    print(f"  📄 Results saved to {results_path}")
    
    elapsed = time.time() - start_time
    print(f"\n{'═'*60}")
    print(f"  ✅ TRAINING COMPLETE in {elapsed:.1f} seconds")
    print(f"     Model saved to: {output_path}")
    print(f"     Agent reward: {np.mean(agent.episode_rewards[-50:]):.2f}")
    print(f"     Random baseline: {eval_results.get('random_avg_reward', 0):.2f}")
    print(f"     Improvement: {eval_results.get('improvement_pct', 0):+.1f}%")
    print(f"{'═'*60}")
    
    return agent, eval_results


def evaluate_sarsa(agent: SARSALambdaAgent, env: DispatchEnv, n_eval_episodes: int = 50):
    """
    Evaluate trained agent vs random baseline.
    """
    print(f"\n{'═'*60}")
    print(f"  SARSA DISPATCH EVALUATION")
    print(f"{'═'*60}")
    
    # ── AI Agent Evaluation ──
    ai_rewards = []
    ai_actions_taken = {a: 0 for a in DISPATCH_ACTIONS}
    ai_zone_visits = {}
    
    for _ in range(n_eval_episodes):
        state = env.reset()
        ep_reward = 0
        for step in range(96):
            action = agent.select_action(state, training=False)
            ai_actions_taken[DISPATCH_ACTIONS[action]] += 1
            
            zone = state[0]
            ai_zone_visits[zone] = ai_zone_visits.get(zone, 0) + 1
            
            state, reward, done = env.step(action)
            ep_reward += reward
            if done:
                break
        ai_rewards.append(ep_reward)
    
    # ── Random Baseline ──
    random_rewards = []
    for _ in range(n_eval_episodes):
        state = env.reset()
        ep_reward = 0
        for step in range(96):
            action = np.random.randint(N_ACTIONS)
            state, reward, done = env.step(action)
            ep_reward += reward
            if done:
                break
        random_rewards.append(ep_reward)
    
    # ── Static Baseline (always stay) ──
    stay_rewards = []
    for _ in range(n_eval_episodes):
        state = env.reset()
        ep_reward = 0
        for step in range(96):
            state, reward, done = env.step(0)  # Always stay
            ep_reward += reward
            if done:
                break
        stay_rewards.append(ep_reward)
    
    ai_avg = np.mean(ai_rewards)
    random_avg = np.mean(random_rewards)
    stay_avg = np.mean(stay_rewards)
    
    improvement_vs_random = ((ai_avg - random_avg) / abs(random_avg) * 100) if random_avg != 0 else 0
    improvement_vs_stay = ((ai_avg - stay_avg) / abs(stay_avg) * 100) if stay_avg != 0 else 0
    
    print(f"\n  📊 Performance Comparison ({n_eval_episodes} episodes):")
    print(f"     AI Agent (SARSA):   {ai_avg:>8.2f} avg reward")
    print(f"     Random Dispatch:    {random_avg:>8.2f} avg reward")
    print(f"     Static (Stay):      {stay_avg:>8.2f} avg reward")
    print(f"     vs Random:          {improvement_vs_random:>+7.1f}%")
    print(f"     vs Static:          {improvement_vs_stay:>+7.1f}%")
    
    # ── Action Distribution ──
    total_actions = sum(ai_actions_taken.values())
    print(f"\n  🚗 AI Action Distribution:")
    for action_name, count in ai_actions_taken.items():
        pct = count / max(total_actions, 1) * 100
        bar = "█" * int(pct / 2)
        print(f"     {action_name:<12}: {count:>5} ({pct:>5.1f}%) {bar}")
    
    # ── Top Zones ──
    top_zones = sorted(ai_zone_visits.items(), key=lambda x: x[1], reverse=True)[:8]
    print(f"\n  📍 Top Visited Zones (AI learned these are valuable):")
    for zone, count in top_zones:
        hotness = env.ZONE_DEMAND_MULT.get(zone, 1.0)
        print(f"     Zone {zone:>3}: {count:>4} visits (demand mult: {hotness:.1f}x)")
    
    results = {
        "ai_avg_reward": float(ai_avg),
        "random_avg_reward": float(random_avg),
        "stay_avg_reward": float(stay_avg),
        "improvement_pct": float(improvement_vs_random),
        "improvement_vs_stay_pct": float(improvement_vs_stay),
        "action_distribution": ai_actions_taken,
        "top_zones": {str(z): c for z, c in top_zones},
    }
    
    return results


# ─── CLI entry point ───
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train SARSA Dispatch Agent")
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes")
    parser.add_argument("--output", default=None, help="Path to save model")
    args = parser.parse_args()
    
    train_sarsa_agent(
        n_episodes=args.episodes,
        output_path=args.output,
    )
