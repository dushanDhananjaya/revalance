"""
Revalance — Tile Coding Feature Extractor
============================================
Converts continuous states into discrete features for SARSA(λ).

🎓 TILE CODING — HOW IT WORKS
═══════════════════════════════

Problem: The dispatch agent's state has CONTINUOUS values like:
  - Driver's zone: 1-263
  - Demand level: 0.0-5.0
  - Supply level: 0.0-5.0
  - Hour of day: 0-23

But SARSA(λ) uses a WEIGHT TABLE (like a dictionary). You can't
have an entry for every possible combination of continuous values
— there would be infinitely many!

Solution: TILE CODING divides the space into overlapping "tiles"
(think of laying multiple grids over the state space):

    Grid 1 (offset 0):      Grid 2 (offset 0.5):
    ┌───┬───┬───┐            ┌───┬───┬───┐
    │ 0 │ 1 │ 2 │              │ 0 │ 1 │ 2 │
    ├───┼───┼───┤            ├───┼───┼───┤
    │ 3 │ 4 │ 5 │              │ 3 │ 4 │ 5 │
    └───┴───┴───┘            └───┴───┴───┘

A state point near a border on Grid 1 will be in different tiles
on Grid 2, giving SMOOTHER generalization than a single grid.

🎓 WHY MULTIPLE TILINGS?
One grid → state near a boundary gets very different tile numbers
from a tiny change in value (jumpy!).
Multiple overlapping grids → smooth transitions. The combined
effect averages out boundary artifacts.

Parameters:
  - n_tilings: How many overlapping grids (8 is standard)
  - n_tiles_per_dim: How many tiles per dimension per tiling
  - Total features: n_tilings × n_tiles_per_dim^n_dimensions
    (but we use hashing to keep it bounded)
"""

import numpy as np
from typing import List, Tuple


class TileCoder:
    """
    Multi-dimensional tile coding with hashing.
    
    Usage:
        tc = TileCoder(
            n_tilings=8,
            n_tiles_per_dim=4,
            n_dimensions=4,
            value_ranges=[(0, 263), (0, 5), (0, 5), (0, 23)],
            max_size=4096,
        )
        
        # Get active tile indices for a state
        tiles = tc.get_tiles([161, 3, 2, 8])
        # Returns: [42, 1027, 2513, ...] (8 indices, one per tiling)
    """
    
    def __init__(
        self,
        n_tilings: int = 8,
        n_tiles_per_dim: int = 4,
        n_dimensions: int = 4,
        value_ranges: List[Tuple[float, float]] = None,
        max_size: int = 4096,
    ):
        """
        🎓 PARAMETERS:
        
        n_tilings: Number of overlapping grids (8 is standard in RL)
            More tilings = smoother function approximation but more memory
            
        n_tiles_per_dim: Resolution of each grid per dimension
            4 means each dimension is split into 4 bins
            
        n_dimensions: Number of state features
            We use 4: zone_id, demand_level, supply_level, hour_of_day
            
        value_ranges: (min, max) for each dimension
            Used to normalize values before tiling
            
        max_size: Maximum number of tile indices (uses hashing)
            🎓 HASHING TRICK: Instead of allocating n_tilings × tiles^dims
            entries (which could be millions), we hash indices into a
            fixed-size array. Some collisions occur, but that's OK —
            it acts as implicit regularization.
        """
        self.n_tilings = n_tilings
        self.n_tiles_per_dim = n_tiles_per_dim
        self.n_dimensions = n_dimensions
        self.max_size = max_size
        
        # Default ranges if not provided
        if value_ranges is None:
            value_ranges = [(0, 1)] * n_dimensions
        self.value_ranges = value_ranges
        
        # Pre-compute tiling offsets (each tiling is slightly shifted)
        # 🎓 The key insight: each tiling uses a different offset
        # so they don't all have tile boundaries at the same places
        self.offsets = np.array([
            [i / n_tilings for _ in range(n_dimensions)]
            for i in range(n_tilings)
        ])
    
    def _normalize(self, state: np.ndarray) -> np.ndarray:
        """
        Normalize state values to [0, n_tiles_per_dim] range.
        
        🎓 WHY NORMALIZE?
        Raw values (zone=161, hour=8, demand=3) have very different
        scales. Normalizing to the same range ensures each dimension
        contributes equally to the tile assignment.
        """
        state = np.array(state, dtype=np.float64)
        normalized = np.zeros(self.n_dimensions)
        
        for i in range(self.n_dimensions):
            low, high = self.value_ranges[i]
            range_size = max(high - low, 1e-8)
            # Scale to [0, n_tiles_per_dim)
            normalized[i] = (state[i] - low) / range_size * self.n_tiles_per_dim
            # Clip to valid range
            normalized[i] = np.clip(normalized[i], 0, self.n_tiles_per_dim - 0.01)
        
        return normalized
    
    def get_tiles(self, state) -> List[int]:
        """
        Get the active tile indices for a given state.
        
        Returns n_tilings indices — one index per tiling layer.
        These indices tell us WHICH tiles in the weight vector
        are "active" for this state.
        
        🎓 THE BIG IDEA:
        Instead of one index (one tile), we get 8 indices.
        The value of a state = sum of weights at all 8 indices:
          V(state) = w[tile_0] + w[tile_1] + ... + w[tile_7]
        
        This gives us SMOOTH generalization across nearby states.
        """
        normalized = self._normalize(state)
        tiles = []
        
        for tiling_idx in range(self.n_tilings):
            # Apply tiling offset
            shifted = normalized + self.offsets[tiling_idx]
            
            # Floor to get tile coordinates
            tile_coords = np.floor(shifted).astype(int)
            
            # Convert N-dimensional coordinates to a single index
            # using a hash function
            flat_index = self._hash(tiling_idx, tile_coords)
            tiles.append(flat_index)
        
        return tiles
    
    def get_tiles_for_action(self, state, action: int) -> List[int]:
        """
        Get tile indices for a (state, action) pair.
        
        Since we have a separate set of tiles for each action,
        we offset by action × max_size.
        
        🎓 STATE-ACTION TILES:
        To represent Q(state, action), we need different tiles
        for each action. We achieve this by adding an offset:
        - Action 0 (Stay):        tiles 0 to max_size-1
        - Action 1 (Move_North):  tiles max_size to 2×max_size-1
        - etc.
        """
        base_tiles = self.get_tiles(state)
        action_offset = action * self.max_size
        return [t + action_offset for t in base_tiles]
    
    def _hash(self, tiling_idx: int, coords: np.ndarray) -> int:
        """
        Hash N-dimensional tile coordinates to a single index.
        
        🎓 HASH FUNCTION:
        We need to convert (tiling_idx, x, y, z, ...) into one number
        within [0, max_size). We use a simple polynomial hash:
        index = (tiling × big_prime + coords × other_primes) % max_size
        
        Collisions are intentional — they provide regularization.
        """
        # Use prime numbers for mixing (reduces collision patterns)
        primes = [2039, 2053, 2063, 2069, 2081, 2083, 2087, 2089]
        
        hash_val = tiling_idx * 6151  # Base hash from tiling index
        for i, coord in enumerate(coords):
            hash_val += int(coord) * primes[i % len(primes)]
        
        return hash_val % self.max_size
    
    @property
    def total_size(self) -> int:
        """Total number of weights needed (tiles × actions)."""
        return self.max_size


# ═══════════════════════════════════════════════════════════════
# PRE-CONFIGURED TILE CODER FOR DISPATCH
# ═══════════════════════════════════════════════════════════════

# Dispatch agent state dimensions:
#   0: zone_id (1-263)
#   1: demand_level (0-5)
#   2: supply_level (0-5)
#   3: hour_of_day (0-23)

DISPATCH_STATE_DIMS = ["zone_id", "demand_level", "supply_level", "hour_of_day"]

DISPATCH_ACTIONS = ["Stay", "Move_North", "Move_South", "Move_East", "Move_West"]
N_ACTIONS = len(DISPATCH_ACTIONS)


def create_dispatch_tile_coder(
    n_tilings: int = 8,
    n_tiles_per_dim: int = 4,
    max_size: int = 4096,
) -> TileCoder:
    """Create a pre-configured TileCoder for the dispatch agent."""
    return TileCoder(
        n_tilings=n_tilings,
        n_tiles_per_dim=n_tiles_per_dim,
        n_dimensions=len(DISPATCH_STATE_DIMS),
        value_ranges=[
            (1, 263),    # zone_id
            (0, 5),      # demand_level
            (0, 5),      # supply_level
            (0, 23),     # hour_of_day
        ],
        max_size=max_size,
    )
