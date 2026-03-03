"""
Population management for genetic algorithm.

Manages strategy population with elitism, diversity maintenance,
and tournament selection.
"""

import random
from typing import List, Optional, Tuple
import numpy as np
from collections import defaultdict

from src.utils.logger import Logger
from .chromosome import StrategyChromosome, ChromosomeFactory

logger = Logger(__name__)


class PopulationManager:
    """
    Manage population of strategy chromosomes with elitism and diversity.
    """

    def __init__(
        self,
        population_size: int = 50,
        elitism_rate: float = 0.10,
        tournament_size: int = 5,
        max_generations: int = 100,
        diversity_threshold: float = 0.3,
    ):
        """
        Initialize population manager.

        Args:
            population_size: Number of strategies in population
            elitism_rate: Percentage of top performers to preserve
            tournament_size: Size of tournament for selection
            max_generations: Maximum generations before reset
            diversity_threshold: Minimum diversity score required
        """
        self.population_size = population_size
        self.elitism_count = max(1, int(population_size * elitism_rate))
        self.tournament_size = tournament_size
        self.max_generations = max_generations
        self.diversity_threshold = diversity_threshold

        self.population: List[StrategyChromosome] = []
        self.generation = 0
        self.hall_of_fame: List[StrategyChromosome] = []  # Top 5 all-time

        logger.info(
            f"PopulationManager initialized: size={population_size}, "
            f"elitism={self.elitism_count}, tournament_size={tournament_size}"
        )

    def initialize_population(self, seed_strategies: Optional[List[StrategyChromosome]] = None) -> None:
        """
        Initialize population with random or seeded strategies.

        Args:
            seed_strategies: Optional list of seed strategies
        """
        self.population = []

        if seed_strategies:
            self.population.extend(seed_strategies)
            logger.info(f"Seeded population with {len(seed_strategies)} strategies")

        # Fill remaining with random strategies
        while len(self.population) < self.population_size:
            strategy = ChromosomeFactory.create_random_chromosome(
                max_conditions=3, generation=0
            )
            if strategy.is_valid():
                self.population.append(strategy)

        logger.info(f"Initialized population with {len(self.population)} strategies")

    def select_parents(self, n_pairs: int) -> List[Tuple[StrategyChromosome, StrategyChromosome]]:
        """
        Select parent pairs using tournament selection.

        Args:
            n_pairs: Number of parent pairs to select

        Returns:
            List of parent pairs
        """
        pairs = []

        for _ in range(n_pairs):
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            pairs.append((parent1, parent2))

        logger.debug(f"Selected {n_pairs} parent pairs")
        return pairs

    def _tournament_selection(self) -> StrategyChromosome:
        """Select one parent using tournament selection."""
        tournament = random.sample(
            self.population, min(self.tournament_size, len(self.population))
        )
        winner = max(tournament, key=lambda x: x.fitness)
        return winner

    def evolve_generation(
        self, offspring: List[StrategyChromosome]
    ) -> List[StrategyChromosome]:
        """
        Create next generation with elitism and offspring.

        Args:
            offspring: List of newly created offspring

        Returns:
            New population
        """
        # Sort current population by fitness
        sorted_population = sorted(
            self.population, key=lambda x: x.fitness, reverse=True
        )

        # Elitism: Keep top performers
        elite = sorted_population[: self.elitism_count]

        logger.info(
            f"Generation {self.generation}: Elite fitness range "
            f"{elite[0].fitness:.3f} - {elite[-1].fitness:.3f}"
        )

        # Update hall of fame
        self._update_hall_of_fame(elite[0])

        # Combine elite with offspring
        new_population = elite + offspring

        # If too many, select best
        if len(new_population) > self.population_size:
            new_population = sorted(
                new_population, key=lambda x: x.fitness, reverse=True
            )[: self.population_size]

        # If too few, fill with mutations of elite
        while len(new_population) < self.population_size:
            from .operators import MutationOperator

            mutator = MutationOperator()
            mutant = mutator.mutate(random.choice(elite))
            new_population.append(mutant)

        self.population = new_population
        self.generation += 1

        # Check diversity
        diversity = self.calculate_diversity()
        logger.info(
            f"Generation {self.generation} complete. "
            f"Best fitness: {new_population[0].fitness:.3f}, "
            f"Diversity: {diversity:.2f}"
        )

        if diversity < self.diversity_threshold:
            logger.warning(
                f"Low diversity detected: {diversity:.2f} < {self.diversity_threshold}"
            )

        return new_population

    def _update_hall_of_fame(self, champion: StrategyChromosome) -> None:
        """Update hall of fame with champion if qualified."""
        # Add to hall of fame if better than worst or if not full
        if len(self.hall_of_fame) < 5:
            self.hall_of_fame.append(champion.clone())
            self.hall_of_fame.sort(key=lambda x: x.fitness, reverse=True)
            logger.info(
                f"New hall of fame member: {champion.chromosome_id} "
                f"(fitness: {champion.fitness:.3f})"
            )
        elif champion.fitness > min(self.hall_of_fame, key=lambda x: x.fitness).fitness:
            self.hall_of_fame.append(champion.clone())
            self.hall_of_fame.sort(key=lambda x: x.fitness, reverse=True)
            self.hall_of_fame = self.hall_of_fame[:5]
            logger.info(
                f"Hall of fame updated: {champion.chromosome_id} "
                f"(fitness: {champion.fitness:.3f})"
            )

    def calculate_diversity(self) -> float:
        """
        Calculate population diversity using complexity variance.

        Returns:
            Diversity score (0-1, higher is more diverse)
        """
        if not self.population:
            return 0.0

        # Diversity metrics
        complexities = [c.get_complexity() for c in self.population]
        fitness_values = [c.fitness for c in self.population if c.fitness > -999]

        # Complexity diversity
        complexity_std = np.std(complexities) if complexities else 0

        # Fitness diversity
        fitness_std = np.std(fitness_values) if fitness_values else 0

        # Indicator diversity (count unique indicator types)
        indicator_types = set()
        for c in self.population:
            for cond in c.long_entry_conditions + c.short_entry_conditions:
                indicator_types.add(cond.indicator_type)

        indicator_diversity = len(indicator_types) / len(list(IndicatorType))

        # Composite diversity score
        diversity = (
            0.3 * min(complexity_std / 3.0, 1.0)
            + 0.3 * min(fitness_std / 0.5, 1.0)
            + 0.4 * indicator_diversity
        )

        return diversity

    def get_best_strategy(self) -> Optional[StrategyChromosome]:
        """Get current best strategy."""
        if not self.population:
            return None
        return max(self.population, key=lambda x: x.fitness)

    def get_hall_of_fame(self) -> List[StrategyChromosome]:
        """Get top 5 historical strategies."""
        return self.hall_of_fame.copy()

    def should_reset(self) -> bool:
        """Determine if population should be reset due to stagnation."""
        if self.generation > self.max_generations:
            logger.warning(
                f"Max generations reached: {self.generation} > {self.max_generations}"
            )
            return True

        diversity = self.calculate_diversity()
        if diversity < self.diversity_threshold * 0.5:
            logger.warning(f"Severe diversity loss: {diversity:.2f}")
            return True

        return False

    def reset_with_elite(self) -> None:
        """Reset population while preserving hall of fame."""
        logger.info("Resetting population with hall of fame")
        self.initialize_population(seed_strategies=self.hall_of_fame[:3])
        self.generation = 0


# Import for type checking
try:
    from enum import Enum

    class IndicatorType(Enum):
        """Placeholder for diversity calculation."""

        pass

except ImportError:
    pass
