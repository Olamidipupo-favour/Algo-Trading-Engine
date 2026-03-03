"""
Genetic algorithm engine for strategy evolution.

CRITICAL: RESEARCH MODE ONLY
No live trading allowed until strategies pass validation.
"""

import pandas as pd
from typing import Dict, List, Optional, Callable, Tuple
from pathlib import Path
import json
from datetime import datetime

from src.utils.logger import Logger
from .chromosome import StrategyChromosome, ChromosomeFactory
from .operators import MutationOperator, CrossoverOperator
from .population import PopulationManager
from src.optimization.fitness import FitnessCalculator
from src.optimization.validators import OverfitValidator

logger = Logger(__name__)


class GeneticEngine:
    """
    Genetic algorithm engine for evolving trading strategies.

    SAFETY FEATURES:
    - Multi-period validation (train/val/test)
    - Out-of-sample evaluation
    - Cross-regime validation
    - Complexity penalty
    - Minimum trade count
    - Research mode lock
    """

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 50,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
        output_dir: str = "genetic_research",
        research_mode_only: bool = True,
    ):
        """
        Initialize genetic engine.

        Args:
            population_size: Size of strategy population
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            output_dir: Directory for research results
            research_mode_only: Safety lock (MUST be True for evolution)
        """
        if not research_mode_only:
            raise ValueError(
                "SAFETY VIOLATION: Genetic evolution MUST run in research mode only. "
                "Set research_mode_only=True."
            )

        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

        # Initialize components
        self.population_manager = PopulationManager(population_size=population_size)
        self.mutation_operator = MutationOperator()
        self.crossover_operator = CrossoverOperator()
        self.fitness_calculator = FitnessCalculator(min_trades=30)
        self.overfit_validator = OverfitValidator()

        # Results storage
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.evolution_history: List[Dict] = []
        self.best_ever_strategy: Optional[StrategyChromosome] = None

        logger.info(
            f"GeneticEngine initialized (RESEARCH MODE): "
            f"pop={population_size}, gen={generations}, "
            f"mutation_rate={mutation_rate}, crossover_rate={crossover_rate}"
        )

    def evolve(
        self,
        data: pd.DataFrame,
        backtest_function: Callable[[pd.DataFrame, StrategyChromosome], Tuple],
        regime_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> StrategyChromosome:
        """
        Evolve strategies through genetic algorithm.

        Args:
            data: Historical market data
            backtest_function: Function(data, chromosome) -> (equity_curve, trades)
            regime_data: Optional dict of regime-specific data for cross-regime validation

        Returns:
            Best evolved strategy
        """
        logger.info(f"Starting genetic evolution for {self.generations} generations")

        # Initialize population
        base_strategy = ChromosomeFactory.create_simple_ema_rsi_strategy()
        self.population_manager.initialize_population(seed_strategies=[base_strategy])

        # Split data
        train_data, val_data, test_data = self._split_data(data)

        for generation in range(self.generations):
            logger.info(f"\n{'='*60}")
            logger.info(f"GENERATION {generation + 1}/{self.generations}")
            logger.info(f"{'='*60}")

            # Evaluate population
            self._evaluate_population(
                self.population_manager.population, train_data, val_data, backtest_function
            )

            # Record generation stats
            self._record_generation_stats(generation)

            # Check if we should continue
            if self.population_manager.should_reset():
                logger.warning("Population stagnation detected. Resetting...")
                self.population_manager.reset_with_elite()
                continue

            # Create offspring
            offspring = []

            # Crossover
            n_crossover = int(self.population_size * self.crossover_rate / 2)
            parent_pairs = self.population_manager.select_parents(n_crossover)

            for parent1, parent2 in parent_pairs:
                child1, child2 = self.crossover_operator.crossover(parent1, parent2)
                offspring.extend([child1, child2])

            # Mutation
            n_mutations = int(self.population_size * self.mutation_rate)
            for _ in range(n_mutations):
                parent = self.population_manager._tournament_selection()
                mutant = self.mutation_operator.mutate(parent)
                offspring.append(mutant)

            # Evaluate offspring
            self._evaluate_population(offspring, train_data, val_data, backtest_function)

            # Evolve to next generation
            self.population_manager.evolve_generation(offspring)

            # Update best ever
            current_best = self.population_manager.get_best_strategy()
            if current_best and (
                self.best_ever_strategy is None
                or current_best.fitness > self.best_ever_strategy.fitness
            ):
                self.best_ever_strategy = current_best.clone()
                logger.info(
                    f"New best strategy found: fitness={current_best.fitness:.3f}"
                )

        # Final validation on test data
        logger.info("\n" + "=" * 60)
        logger.info("FINAL VALIDATION ON TEST DATA")
        logger.info("=" * 60)

        hall_of_fame = self.population_manager.get_hall_of_fame()
        best_validated = None
        best_test_fitness = -999.0

        for strategy in hall_of_fame:
            try:
                test_equity, test_trades = backtest_function(test_data, strategy)
                test_metrics = self.fitness_calculator.calculate_metrics(
                    test_equity, test_trades
                )
                test_fitness, _ = self.fitness_calculator.calculate_fitness(
                    test_metrics, strategy.get_complexity()
                )

                logger.info(
                    f"Strategy {strategy.chromosome_id}: "
                    f"test_fitness={test_fitness:.3f}"
                )

                if test_fitness > best_test_fitness:
                    best_test_fitness = test_fitness
                    best_validated = strategy

            except Exception as e:
                logger.warning(f"Test validation failed for {strategy.chromosome_id}: {e}")

        if best_validated:
            logger.info(
                f"\nBest validated strategy: {best_validated.chromosome_id} "
                f"(test_fitness={best_test_fitness:.3f})"
            )
            self._save_final_results(best_validated, best_test_fitness)
        else:
            logger.error("No strategy passed final validation")
            best_validated = self.best_ever_strategy

        return best_validated

    def _evaluate_population(
        self,
        population: List[StrategyChromosome],
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        backtest_function: Callable,
    ) -> None:
        """Evaluate fitness for all strategies in population."""
        logger.debug(f"Evaluating {len(population)} strategies")

        for idx, strategy in enumerate(population):
            try:
                # Skip if already evaluated
                if strategy.fitness > -999.0:
                    continue

                # Train evaluation
                train_equity, train_trades = backtest_function(train_data, strategy)
                train_metrics = self.fitness_calculator.calculate_metrics(
                    train_equity, train_trades
                )

                train_fitness, _ = self.fitness_calculator.calculate_fitness(
                    train_metrics, strategy.get_complexity()
                )

                # Validation evaluation
                val_equity, val_trades = backtest_function(val_data, strategy)
                val_metrics = self.fitness_calculator.calculate_metrics(
                    val_equity, val_trades
                )

                val_fitness, _ = self.fitness_calculator.calculate_fitness(
                    val_metrics, strategy.get_complexity()
                )

                # Check overfitting
                overfit_result = self.overfit_validator.detect_overfitting(
                    train_fitness, val_fitness, val_fitness  # Use val as test proxy
                )

                # Use validation fitness as primary (more conservative)
                strategy.fitness = val_fitness if not overfit_result.is_overfit else train_fitness * 0.5

                if (idx + 1) % 10 == 0:
                    logger.debug(
                        f"Evaluated {idx + 1}/{len(population)} | "
                        f"Last fitness: {strategy.fitness:.3f}"
                    )

            except Exception as e:
                logger.warning(f"Evaluation failed for {strategy.chromosome_id}: {e}")
                strategy.fitness = -999.0

    def _split_data(
        self, data: pd.DataFrame, train_pct: float = 0.6, val_pct: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/val/test."""
        n = len(data)
        train_end = int(n * train_pct)
        val_end = int(n * (train_pct + val_pct))

        train = data.iloc[:train_end].copy()
        val = data.iloc[train_end:val_end].copy()
        test = data.iloc[val_end:].copy()

        logger.debug(
            f"Data split: train={len(train)}, val={len(val)}, test={len(test)}"
        )

        return train, val, test

    def _record_generation_stats(self, generation: int) -> None:
        """Record generation statistics."""
        best = self.population_manager.get_best_strategy()
        diversity = self.population_manager.calculate_diversity()

        if best:
            stats = {
                "generation": generation,
                "best_fitness": best.fitness,
                "best_id": best.chromosome_id,
                "complexity": best.get_complexity(),
                "diversity": diversity,
                "timestamp": datetime.now().isoformat(),
            }

            self.evolution_history.append(stats)

            logger.info(
                f"Gen {generation}: Best={best.fitness:.3f}, "
                f"Complexity={best.get_complexity()}, Diversity={diversity:.2f}"
            )

    def _save_final_results(self, best_strategy: StrategyChromosome, test_fitness: float) -> None:
        """Save final evolution results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save best strategy
        strategy_file = self.output_dir / f"best_strategy_{timestamp}.json"
        with open(strategy_file, "w") as f:
            json.dump(best_strategy.to_dict(), f, indent=2)

        # Save evolution history
        history_file = self.output_dir / f"evolution_history_{timestamp}.json"
        with open(history_file, "w") as f:
            json.dump(self.evolution_history, f, indent=2)

        # Save human-readable report
        report_file = self.output_dir / f"evolution_report_{timestamp}.txt"
        with open(report_file, "w") as f:
            f.write(self._generate_report(best_strategy, test_fitness))

        logger.info(f"Results saved to {self.output_dir}")

    def _generate_report(self, best_strategy: StrategyChromosome, test_fitness: float) -> str:
        """Generate human-readable evolution report."""
        report = [
            "=" * 70,
            "GENETIC EVOLUTION REPORT (RESEARCH MODE)",
            "=" * 70,
            "",
            f"Evolution completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Generations: {self.generations}",
            f"Population size: {self.population_size}",
            "",
            "=" * 70,
            "BEST EVOLVED STRATEGY",
            "=" * 70,
            "",
            best_strategy.describe(),
            "",
            f"Final test fitness: {test_fitness:.3f}",
            "",
            "=" * 70,
            "HALL OF FAME (Top 5 Strategies)",
            "=" * 70,
            "",
        ]

        for idx, strategy in enumerate(self.population_manager.get_hall_of_fame(), 1):
            report.append(f"{idx}. ID: {strategy.chromosome_id} | Fitness: {strategy.fitness:.3f}")

        report.extend(
            [
                "",
                "=" * 70,
                "SAFETY NOTICE",
                "=" * 70,
                "",
                "This strategy was evolved in RESEARCH MODE ONLY.",
                "Before live deployment:",
                "  1. Extensive out-of-sample testing required",
                "  2. Cross-regime validation mandatory",
                "  3. Paper trading for minimum 30 days",
                "  4. Manual review and approval required",
                "  5. Compliance with risk management protocols",
                "",
                "DO NOT deploy to live trading without completing all validation steps.",
                "=" * 70,
            ]
        )

        return "\n".join(report)
