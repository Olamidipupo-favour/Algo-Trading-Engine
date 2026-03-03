"""
Genetic operators: mutation and crossover for strategy evolution.

These operators modify and combine strategies while maintaining
structural validity and preventing degenerate solutions.
"""

import random
import copy
from typing import List, Tuple
import numpy as np

from src.utils.logger import Logger
from .chromosome import (
    StrategyChromosome,
    ConditionGene,
    ExitLogicGene,
    IndicatorType,
    LogicalOperator,
    ChromosomeFactory,
)

logger = Logger(__name__)


class MutationOperator:
    """
    Mutation operations for strategy chromosomes.

    Supports multiple mutation types with configurable rates.
    """

    def __init__(
        self,
        parameter_mutation_rate: float = 0.3,
        indicator_replacement_rate: float = 0.1,
        condition_add_remove_rate: float = 0.05,
        operator_flip_rate: float = 0.1,
        max_conditions: int = 5,
    ):
        """
        Initialize mutation operator.

        Args:
            parameter_mutation_rate: Probability of mutating a parameter
            indicator_replacement_rate: Probability of replacing an indicator
            condition_add_remove_rate: Probability of adding/removing condition
            operator_flip_rate: Probability of flipping AND/OR operator
            max_conditions: Maximum conditions per entry
        """
        self.parameter_mutation_rate = parameter_mutation_rate
        self.indicator_replacement_rate = indicator_replacement_rate
        self.condition_add_remove_rate = condition_add_remove_rate
        self.operator_flip_rate = operator_flip_rate
        self.max_conditions = max_conditions

        logger.info(
            f"MutationOperator initialized: "
            f"param_rate={parameter_mutation_rate}, "
            f"indicator_rate={indicator_replacement_rate}"
        )

    def mutate(self, chromosome: StrategyChromosome) -> StrategyChromosome:
        """
        Apply mutations to chromosome.

        Args:
            chromosome: Chromosome to mutate

        Returns:
            Mutated chromosome (new instance)
        """
        mutated = chromosome.clone()

        # Parameter mutation
        if random.random() < self.parameter_mutation_rate:
            mutated = self._mutate_parameters(mutated)

        # Indicator replacement
        if random.random() < self.indicator_replacement_rate:
            mutated = self._replace_indicator(mutated)

        # Add/remove conditions
        if random.random() < self.condition_add_remove_rate:
            mutated = self._add_remove_condition(mutated)

        # Flip logical operators
        if random.random() < self.operator_flip_rate:
            mutated = self._flip_operators(mutated)

        # Exit logic mutation
        if random.random() < 0.2:
            mutated = self._mutate_exit_logic(mutated)

        # Risk parameter mutation
        if random.random() < 0.15:
            mutated = self._mutate_risk_parameters(mutated)

        # Update generation
        mutated.generation = chromosome.generation + 1
        mutated.fitness = -999.0  # Reset fitness
        mutated.parent_ids = [chromosome.chromosome_id]
        mutated.chromosome_id = mutated._generate_id()

        logger.debug(
            f"Mutated chromosome {chromosome.chromosome_id} -> "
            f"{mutated.chromosome_id}"
        )

        return mutated

    def _mutate_parameters(
        self, chromosome: StrategyChromosome
    ) -> StrategyChromosome:
        """Mutate indicator parameters."""
        # Choose random condition to mutate
        all_conditions = (
            chromosome.long_entry_conditions + chromosome.short_entry_conditions
        )

        if not all_conditions:
            return chromosome

        condition = random.choice(all_conditions)

        # Mutate based on indicator type
        if condition.indicator_type == IndicatorType.EMA_CROSSOVER:
            if "fast" in condition.parameters:
                condition.parameters["fast"] = self._perturb_integer(
                    condition.parameters["fast"], min_val=5, max_val=20
                )
            if "slow" in condition.parameters:
                condition.parameters["slow"] = self._perturb_integer(
                    condition.parameters["slow"], min_val=20, max_val=60
                )

        elif condition.indicator_type == IndicatorType.RSI_THRESHOLD:
            if "period" in condition.parameters:
                condition.parameters["period"] = self._perturb_integer(
                    condition.parameters["period"], min_val=8, max_val=20
                )
            if condition.threshold is not None:
                # Adjust threshold
                if condition.comparison_operator == ">":
                    condition.threshold = self._perturb_float(
                        condition.threshold, min_val=65, max_val=85
                    )
                else:
                    condition.threshold = self._perturb_float(
                        condition.threshold, min_val=15, max_val=35
                    )

        elif condition.indicator_type == IndicatorType.ADX_TREND:
            if condition.threshold is not None:
                condition.threshold = self._perturb_float(
                    condition.threshold, min_val=15, max_val=35
                )

        logger.debug(f"Parameter mutation applied to {condition.indicator_type.value}")
        return chromosome

    def _replace_indicator(
        self, chromosome: StrategyChromosome
    ) -> StrategyChromosome:
        """Replace a random indicator with a new one."""
        # Choose entry side
        if random.random() > 0.5 and chromosome.long_entry_conditions:
            conditions = chromosome.long_entry_conditions
        elif chromosome.short_entry_conditions:
            conditions = chromosome.short_entry_conditions
        else:
            return chromosome

        if not conditions:
            return chromosome

        # Replace random condition
        idx = random.randint(0, len(conditions) - 1)
        conditions[idx] = ChromosomeFactory._random_condition()

        logger.debug("Indicator replacement applied")
        return chromosome

    def _add_remove_condition(
        self, chromosome: StrategyChromosome
    ) -> StrategyChromosome:
        """Add or remove a condition."""
        # Choose entry side
        if random.random() > 0.5:
            conditions = chromosome.long_entry_conditions
            side = "long"
        else:
            conditions = chromosome.short_entry_conditions
            side = "short"

        # Decide add or remove
        if len(conditions) < self.max_conditions and (
            random.random() > 0.5 or len(conditions) == 0
        ):
            # Add condition
            new_condition = ChromosomeFactory._random_condition()
            conditions.append(new_condition)
            logger.debug(f"Added condition to {side} entry")
        elif len(conditions) > 1:
            # Remove condition
            conditions.pop(random.randint(0, len(conditions) - 1))
            logger.debug(f"Removed condition from {side} entry")

        return chromosome

    def _flip_operators(self, chromosome: StrategyChromosome) -> StrategyChromosome:
        """Flip logical operators (AND <-> OR)."""
        if random.random() > 0.5:
            chromosome.long_entry_operator = (
                LogicalOperator.OR
                if chromosome.long_entry_operator == LogicalOperator.AND
                else LogicalOperator.AND
            )
            logger.debug("Flipped long entry operator")
        else:
            chromosome.short_entry_operator = (
                LogicalOperator.OR
                if chromosome.short_entry_operator == LogicalOperator.AND
                else LogicalOperator.AND
            )
            logger.debug("Flipped short entry operator")

        return chromosome

    def _mutate_exit_logic(self, chromosome: StrategyChromosome) -> StrategyChromosome:
        """Mutate stop-loss and take-profit parameters."""
        # Choose side
        exit_logic = (
            chromosome.long_exit_logic
            if random.random() > 0.5
            else chromosome.short_exit_logic
        )

        if exit_logic is None:
            return chromosome

        # Mutate SL
        if random.random() > 0.5:
            exit_logic.stop_loss_value = self._perturb_float(
                exit_logic.stop_loss_value, min_val=0.5, max_val=4.0
            )

        # Mutate TP
        if random.random() > 0.5:
            exit_logic.take_profit_value = self._perturb_float(
                exit_logic.take_profit_value, min_val=1.0, max_val=5.0
            )

        # Toggle trailing
        if random.random() < 0.2:
            exit_logic.trailing_enabled = not exit_logic.trailing_enabled

        logger.debug("Exit logic mutation applied")
        return chromosome

    def _mutate_risk_parameters(
        self, chromosome: StrategyChromosome
    ) -> StrategyChromosome:
        """Mutate risk management parameters."""
        # Mutate risk per trade
        risk_choices = [0.005, 0.01, 0.015, 0.02]
        chromosome.risk_per_trade = random.choice(risk_choices)

        # Mutate max positions
        chromosome.max_positions = random.randint(1, 3)

        logger.debug(
            f"Risk parameters mutated: risk={chromosome.risk_per_trade:.1%}, "
            f"max_pos={chromosome.max_positions}"
        )
        return chromosome

    def _perturb_integer(self, value: int, min_val: int, max_val: int) -> int:
        """Perturb integer value within bounds."""
        perturbation = random.randint(-3, 3)
        return np.clip(value + perturbation, min_val, max_val)

    def _perturb_float(self, value: float, min_val: float, max_val: float) -> float:
        """Perturb float value within bounds."""
        perturbation = random.uniform(-0.5, 0.5)
        return np.clip(value + perturbation, min_val, max_val)


class CrossoverOperator:
    """
    Crossover operations for combining parent chromosomes.

    Creates offspring that inherit traits from both parents.
    """

    def __init__(self, condition_swap_rate: float = 0.5):
        """
        Initialize crossover operator.

        Args:
            condition_swap_rate: Probability of swapping each condition
        """
        self.condition_swap_rate = condition_swap_rate
        logger.info(f"CrossoverOperator initialized: swap_rate={condition_swap_rate}")

    def crossover(
        self, parent1: StrategyChromosome, parent2: StrategyChromosome
    ) -> Tuple[StrategyChromosome, StrategyChromosome]:
        """
        Perform crossover between two parent chromosomes.

        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome

        Returns:
            Tuple of two offspring chromosomes
        """
        offspring1 = parent1.clone()
        offspring2 = parent2.clone()

        # Crossover entry conditions
        offspring1, offspring2 = self._crossover_conditions(
            offspring1, offspring2, "long"
        )
        offspring1, offspring2 = self._crossover_conditions(
            offspring1, offspring2, "short"
        )

        # Crossover exit logic
        if random.random() > 0.5:
            offspring1.long_exit_logic, offspring2.long_exit_logic = (
                offspring2.long_exit_logic,
                offspring1.long_exit_logic,
            )

        if random.random() > 0.5:
            offspring1.short_exit_logic, offspring2.short_exit_logic = (
                offspring2.short_exit_logic,
                offspring1.short_exit_logic,
            )

        # Crossover operators
        if random.random() > 0.5:
            offspring1.long_entry_operator, offspring2.long_entry_operator = (
                offspring2.long_entry_operator,
                offspring1.long_entry_operator,
            )

        # Blend risk parameters
        offspring1.risk_per_trade = random.choice(
            [parent1.risk_per_trade, parent2.risk_per_trade]
        )
        offspring2.risk_per_trade = random.choice(
            [parent1.risk_per_trade, parent2.risk_per_trade]
        )

        # Update metadata
        for offspring in [offspring1, offspring2]:
            offspring.generation = max(parent1.generation, parent2.generation) + 1
            offspring.fitness = -999.0
            offspring.parent_ids = [parent1.chromosome_id, parent2.chromosome_id]
            offspring.chromosome_id = offspring._generate_id()

        logger.debug(
            f"Crossover: {parent1.chromosome_id} + {parent2.chromosome_id} -> "
            f"{offspring1.chromosome_id}, {offspring2.chromosome_id}"
        )

        return offspring1, offspring2

    def _crossover_conditions(
        self,
        offspring1: StrategyChromosome,
        offspring2: StrategyChromosome,
        side: str,
    ) -> Tuple[StrategyChromosome, StrategyChromosome]:
        """Crossover entry conditions for specified side."""
        if side == "long":
            cond1 = offspring1.long_entry_conditions
            cond2 = offspring2.long_entry_conditions
        else:
            cond1 = offspring1.short_entry_conditions
            cond2 = offspring2.short_entry_conditions

        # Single-point or uniform crossover
        if random.random() > 0.5 and cond1 and cond2:
            # Single-point crossover
            point = min(len(cond1), len(cond2)) // 2
            new_cond1 = cond1[:point] + cond2[point:]
            new_cond2 = cond2[:point] + cond1[point:]
        else:
            # Uniform crossover
            new_cond1 = []
            new_cond2 = []

            max_len = max(len(cond1), len(cond2))
            for i in range(max_len):
                if i < len(cond1) and i < len(cond2):
                    if random.random() > self.condition_swap_rate:
                        new_cond1.append(copy.deepcopy(cond1[i]))
                        new_cond2.append(copy.deepcopy(cond2[i]))
                    else:
                        new_cond1.append(copy.deepcopy(cond2[i]))
                        new_cond2.append(copy.deepcopy(cond1[i]))
                elif i < len(cond1):
                    new_cond1.append(copy.deepcopy(cond1[i]))
                elif i < len(cond2):
                    new_cond2.append(copy.deepcopy(cond2[i]))

        # Update conditions
        if side == "long":
            offspring1.long_entry_conditions = new_cond1
            offspring2.long_entry_conditions = new_cond2
        else:
            offspring1.short_entry_conditions = new_cond1
            offspring2.short_entry_conditions = new_cond2

        return offspring1, offspring2
