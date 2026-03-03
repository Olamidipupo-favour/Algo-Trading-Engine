"""
Genetic algorithm strategy evolution module.

This module evolves trading strategies through genetic algorithms with
strict overfitting safeguards and multi-regime validation.

CRITICAL: This module runs in RESEARCH MODE ONLY.
No live trading allowed with evolved strategies until validated.
"""

from .engine import GeneticEngine
from .chromosome import StrategyChromosome
from .operators import MutationOperator, CrossoverOperator
from .population import PopulationManager

__all__ = [
    "GeneticEngine",
    "StrategyChromosome",
    "MutationOperator",
    "CrossoverOperator",
    "PopulationManager",
]
