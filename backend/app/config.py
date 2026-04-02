"""
Revalance — Application Configuration
======================================
This module handles all configuration using Pydantic BaseSettings.

WHY CONFIGURATION MATTERS:
Instead of hardcoding values like database passwords or file paths
directly in our code (which is INSECURE and inflexible), we read them
from environment variables. This way:
1. Different environments (dev, test, production) can have different values
2. Secrets never get committed to GitHub
3. You can change behavior without modifying code

HOW IT WORKS:
- Pydantic BaseSettings reads values from environment variables
- It also reads from a .env file automatically
- Type validation ensures values are correct types
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Each field here maps to an environment variable.
    For example: DATABASE_URL in .env → settings.database_url in Python
    """

    # Database
    database_url: str = "postgresql://revalance:revalance@localhost:5432/revalance"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # ML Model paths
    model_a_path: str = "app/ml/pricing_agent/model.joblib"
    model_b_path: str = "app/ml/dispatch_agent/weights.npy"
    elasticity_path: str = "app/ml/pricing_agent/elasticity_coefficients.json"

    # Data
    nyc_data_path: str = "data/raw"
    base_fare: float = 2.50

    # Simulation defaults
    default_fleet_size: int = 100
    default_sim_speed: int = 5
    mcts_time_budget_ms: int = 200
    mcts_tree_depth: int = 3
    mcts_n_workers: int = 4

    class Config:
        env_file = ".env"
        case_sensitive = False


# Create a single instance to use throughout the app
# (Singleton pattern — only one Settings object exists)
settings = Settings()
