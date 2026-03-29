"""
Revalance — MCTS (Monte Carlo Tree Search) Planner
=====================================================
The "lookahead brain" for Model A's pricing decisions.

🎓 MCTS — HOW IT WORKS (Chess Analogy)
═══════════════════════════════════════

Imagine a chess player thinking ahead:
"If I move my knight HERE, then my opponent might do THIS,
then I could do THAT..." — exploring a TREE of possibilities.

MCTS does exactly this for pricing:
"If I charge 1.5x NOW in this zone, demand might drop to X,
then in 15 minutes the situation looks like Y, and if I then
charge 1.2x, revenue would be Z..."

THE 4 PHASES OF MCTS (repeated thousands of times in 200ms):
═══════════════════════════════════════════════════════════════

1. SELECTION: Start at the root, walk down the tree picking
   the most promising child using UCB1 (exploration vs exploitation)

2. EXPANSION: Found a node with unexplored children? Pick one
   and add it to the tree.

3. EVALUATION: Instead of playing out the whole game (too slow),
   use our FQI model to ESTIMATE the value of this new node.
   This is a KEY OPTIMIZATION — full rollouts would take forever.

4. BACKPROPAGATION: Walk BACK UP the tree, updating every ancestor
   with the new value estimate.

After 200ms: Pick the root child with the MOST VISITS (not highest
value!) — more visits = more confident estimate.

🎓 UCB1 FORMULA (Upper Confidence Bound):
═══════════════════════════════════════════

UCB1 = Q_avg + C × √(ln(N_parent) / N_self)
         ↑           ↑
    Exploitation   Exploration
    (pick high-    (try under-
     value nodes)   explored nodes)

C = √2 ≈ 1.414 is the "exploration constant"
"""

import math
import time
import json
import os
from typing import Optional
from multiprocessing import Pool, cpu_count

import numpy as np

from app.ml.pricing_agent.fqi_model import FQIPricingModel, PRICE_ACTIONS, STATE_FEATURES


class MCTSNode:
    """
    A single node in the MCTS search tree.
    
    Each node represents a (state, action) pair — "I was in this
    state and chose this price multiplier."
    """
    
    def __init__(
        self,
        state: dict,
        action: Optional[float] = None,
        parent: Optional["MCTSNode"] = None,
    ):
        self.state = state
        self.action = action  # Price multiplier that LED to this state
        self.parent = parent
        self.children: dict[float, "MCTSNode"] = {}  # action → child node
        
        self.visit_count: int = 0
        self.total_value: float = 0.0
    
    @property
    def q_value(self) -> float:
        """Average value across all visits (exploitation term)."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count
    
    @property
    def ucb1_score(self) -> float:
        """
        UCB1 score for node selection.
        
        🎓 BALANCING EXPLORATION vs EXPLOITATION:
        
        - High q_value → Node has been good so far → EXPLOIT it
        - High exploration term → Node hasn't been tried much → EXPLORE it
        
        The √(ln(parent visits) / self visits) term grows when:
        - Parent has been visited a lot but this child hasn't
        → "Hey, you've checked the other options, try me too!"
        """
        C = math.sqrt(2)  # Exploration constant
        
        if self.visit_count == 0:
            return float('inf')  # Never visited → explore first!
        
        if self.parent is None:
            return self.q_value
        
        exploration = C * math.sqrt(
            math.log(self.parent.visit_count) / self.visit_count
        )
        return self.q_value + exploration
    
    @property
    def is_fully_expanded(self) -> bool:
        """Check if all possible actions have been tried from this node."""
        return len(self.children) >= len(PRICE_ACTIONS)
    
    def best_child(self) -> "MCTSNode":
        """Select the child with the highest UCB1 score."""
        return max(self.children.values(), key=lambda c: c.ucb1_score)
    
    def most_visited_child(self) -> tuple[float, "MCTSNode"]:
        """Select the child with the most visits (for final decision)."""
        best = max(self.children.items(), key=lambda x: x[1].visit_count)
        return best  # (action, node)


class PricingMCTS:
    """
    Monte Carlo Tree Search planner for pricing decisions.
    
    Usage:
        mcts = PricingMCTS(fqi_model, elasticity_coefficients)
        best_action, confidence = mcts.select_price(zone_state)
    """
    
    def __init__(
        self,
        fqi_model: FQIPricingModel,
        elasticity_coefficients: dict,
        time_budget_ms: int = 200,
        base_fare: float = 2.50,
    ):
        self.fqi_model = fqi_model
        self.elasticity = elasticity_coefficients
        self.time_budget_ms = time_budget_ms
        self.base_fare = base_fare
    
    def select_price(self, state: dict) -> tuple[float, float]:
        """
        Select the best price multiplier for the given zone state.
        
        🎓 THE MAIN MCTS LOOP:
        We repeat Selection→Expansion→Evaluation→Backpropagation
        as many times as we can within the 200ms time budget.
        
        Parameters:
            state: Zone state dict with keys from STATE_FEATURES
            
        Returns:
            (best_multiplier, confidence)
            where confidence = visit_count / total_iterations
        """
        start_time = time.monotonic()
        deadline = start_time + (self.time_budget_ms / 1000.0)
        
        # ── Pre-pruning: only consider top-3 actions from FQI ──
        # 🎓 ACTION PRUNING: Instead of exploring all 5 actions,
        # we ask FQI to score them first and only keep the top 3.
        # This focuses MCTS search on promising regions.
        state_array = self._state_to_array(state)
        top_actions = self.fqi_model.get_top_actions(state_array, n=3)
        candidate_actions = [a for a, _ in top_actions]
        
        # Create root node
        root = MCTSNode(state=state)
        
        # ── MCTS Loop ──
        iterations = 0
        while time.monotonic() < deadline:
            # 1. SELECTION: Walk down tree using UCB1
            node = self._select(root)
            
            # 2. EXPANSION: Add a new child node
            node = self._expand(node, candidate_actions)
            
            # 3. EVALUATION: Use FQI to estimate value
            value = self._evaluate(node)
            
            # 4. BACKPROPAGATION: Update ancestors
            self._backpropagate(node, value)
            
            iterations += 1
        
        # ── Final Decision: Pick most-visited action ──
        if not root.children:
            # No iterations completed — fall back to FQI
            best_action, _ = self.fqi_model.predict_best_action(state_array)
            return best_action, 0.0
        
        best_action, best_node = root.most_visited_child()
        confidence = best_node.visit_count / max(iterations, 1)
        
        elapsed_ms = (time.monotonic() - start_time) * 1000
        
        return float(best_action), float(confidence)
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        SELECTION phase: Walk down tree using UCB1 until we find
        a node that isn't fully expanded or is a leaf.
        """
        while node.is_fully_expanded and node.children:
            node = node.best_child()
        return node
    
    def _expand(self, node: MCTSNode, candidate_actions: list[float]) -> MCTSNode:
        """
        EXPANSION phase: Add ONE new child to the tree.
        Try an action we haven't explored yet from this node.
        """
        # Find untried actions
        tried = set(node.children.keys())
        untried = [a for a in candidate_actions if a not in tried]
        
        if not untried:
            # All actions tried — just return current node
            return node
        
        # Pick a random untried action
        action = untried[np.random.randint(len(untried))]
        
        # Simulate the transition to get the next state
        next_state, reward = self._transition(node.state, action)
        
        # Create child node
        child = MCTSNode(state=next_state, action=action, parent=node)
        node.children[action] = child
        
        return child
    
    def _evaluate(self, node: MCTSNode) -> float:
        """
        EVALUATION phase: Estimate the value of this node.
        
        🎓 KEY OPTIMIZATION:
        Traditional MCTS does a full "rollout" — randomly playing
        until the game ends. But our simulation has no clear "end",
        and rollouts are slow.
        
        Instead, we use our FQI model's Q-value estimate as a
        FAST APPROXIMATION. The FQI model already learned what
        states are valuable from historical data.
        
        This technique is called "neural network evaluation" in
        AlphaGo (but we use Random Forest instead).
        """
        state_array = self._state_to_array(node.state)
        _, best_q = self.fqi_model.predict_best_action(state_array)
        return best_q
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """
        BACKPROPAGATION phase: Update all ancestors with the value.
        Walk up from the evaluated node to the root, updating
        visit counts and accumulated values.
        """
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            node = node.parent
    
    def _transition(self, state: dict, action: float) -> tuple[dict, float]:
        """
        Forward model: Predict what happens if we take this action.
        
        🎓 DEMAND → PRICE RELATIONSHIP:
        If we raise price from 1.0x to 1.5x (50% increase):
        - In an INELASTIC hour (elasticity=0.3):
          demand change = -0.3 × 50% = -15% → barely any drop
        - In an ELASTIC hour (elasticity=0.8):
          demand change = -0.8 × 50% = -40% → big drop!
        
        We use Poisson sampling to add realistic randomness.
        """
        hour = int(state.get("hour_of_day", 12))
        elasticity = self.elasticity.get(hour, self.elasticity.get(str(hour), 0.5))
        
        # Base demand from state
        demand_level = state.get("demand_level", 2)
        base_demand = demand_level * 5 + 3  # Map level to approximate count
        
        # Price effect on demand
        price_multiplier = action
        expected_demand = base_demand * (1 - elasticity * (price_multiplier - 1.0))
        expected_demand = max(0.5, expected_demand)
        
        # Stochastic demand (Poisson distribution)
        actual_requests = np.random.poisson(expected_demand)
        
        # Available cars
        available_cars = state.get("supply_level", 2) * 4 + 2
        rides_accepted = min(actual_requests, available_cars)
        
        # Revenue
        revenue = price_multiplier * self.base_fare * rides_accepted
        
        # Penalty for unserved demand
        penalty = 50.0 if actual_requests > 0 and rides_accepted == 0 else 0.0
        reward = revenue - penalty
        
        # Build next state (advance by 15 minutes)
        next_hour = (hour + 1) % 24 if np.random.random() < 0.25 else hour
        next_supply = max(0, available_cars - rides_accepted)
        
        next_state = {
            "hour_of_day": next_hour,
            "day_of_week": state.get("day_of_week", 0),
            "is_weekend": state.get("is_weekend", 0),
            "zone_id": state.get("zone_id", 161),
            "demand_level": min(5, max(0, demand_level + np.random.choice([-1, 0, 1]))),
            "supply_level": min(5, max(0, int(next_supply / 4))),
            "competitor_price_idx": state.get("competitor_price_idx", 1.0),
        }
        
        return next_state, reward
    
    def _state_to_array(self, state: dict) -> np.ndarray:
        """Convert state dict to numpy array in the correct feature order."""
        return np.array([state.get(f, 0) for f in STATE_FEATURES], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════
# ROOT PARALLELIZATION (for faster MCTS)
# ═══════════════════════════════════════════════════════════════

def _worker_mcts_search(args):
    """
    Worker function for parallel MCTS searches.
    
    🎓 ROOT PARALLELIZATION:
    Instead of one tree searched for 200ms, we run 4 independent
    trees for 50ms each, then merge their statistics.
    
    This is like asking 4 people to each think about the problem
    independently, then combining their conclusions.
    """
    fqi_model_path, elasticity, state, time_budget_ms = args
    
    model = FQIPricingModel.load(fqi_model_path)
    mcts = PricingMCTS(model, elasticity, time_budget_ms=time_budget_ms)
    action, confidence = mcts.select_price(state)
    
    return action, confidence


def parallel_mcts_search(
    fqi_model_path: str,
    elasticity: dict,
    state: dict,
    n_workers: int = 4,
    total_budget_ms: int = 200,
) -> tuple[float, float]:
    """
    Run multiple MCTS searches in parallel and merge results.
    
    Each worker gets total_budget / n_workers milliseconds.
    The final action is the one chosen by the majority of workers.
    """
    per_worker_budget = total_budget_ms // n_workers
    
    args_list = [
        (fqi_model_path, elasticity, state, per_worker_budget)
        for _ in range(n_workers)
    ]
    
    # For simplicity in the demo, run sequentially if pool fails
    try:
        with Pool(processes=min(n_workers, cpu_count())) as pool:
            results = pool.map(_worker_mcts_search, args_list)
    except Exception:
        # Fallback to sequential
        results = [_worker_mcts_search(args) for args in args_list]
    
    # Merge: majority vote on action
    actions = [r[0] for r in results]
    from collections import Counter
    action_counts = Counter(actions)
    best_action = action_counts.most_common(1)[0][0]
    avg_confidence = sum(r[1] for r in results) / len(results)
    
    return best_action, avg_confidence
