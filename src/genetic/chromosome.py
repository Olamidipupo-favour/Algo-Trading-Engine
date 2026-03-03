"""
Strategy chromosome representation for genetic evolution.

Encodes complete trading strategies as evolvable genetic structures
including entry/exit conditions, indicators, and logic operators.
"""

import copy
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib

from src.utils.logger import Logger

logger = Logger(__name__)


class IndicatorType(Enum):
    """Supported indicator types in gene pool."""

    EMA_CROSSOVER = "ema_crossover"
    RSI_THRESHOLD = "rsi_threshold"
    MACD_CROSSOVER = "macd_crossover"
    BOLLINGER_BANDS = "bollinger_bands"
    ATR_FILTER = "atr_filter"
    ADX_TREND = "adx_trend"
    STOCHASTIC = "stochastic"
    VOLUME_FILTER = "volume_filter"


class LogicalOperator(Enum):
    """Logical operators for combining conditions."""

    AND = "AND"
    OR = "OR"


@dataclass
class ConditionGene:
    """Individual condition gene (indicator-based rule)."""

    indicator_type: IndicatorType
    parameters: Dict[str, Any]
    comparison_operator: str  # ">", "<", ">=", "<=", "==", "cross_above", "cross_below"
    threshold: Optional[float] = None
    enabled: bool = True

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "indicator_type": self.indicator_type.value,
            "parameters": self.parameters,
            "comparison_operator": self.comparison_operator,
            "threshold": self.threshold,
            "enabled": self.enabled,
        }

    def describe(self) -> str:
        """Human-readable description of condition."""
        if self.indicator_type == IndicatorType.EMA_CROSSOVER:
            return (
                f"EMA({self.parameters['fast']}) cross "
                f"{self.comparison_operator} EMA({self.parameters['slow']})"
            )
        elif self.indicator_type == IndicatorType.RSI_THRESHOLD:
            return (
                f"RSI({self.parameters['period']}) "
                f"{self.comparison_operator} {self.threshold}"
            )
        elif self.indicator_type == IndicatorType.MACD_CROSSOVER:
            return "MACD cross " + self.comparison_operator.replace("_", " ")
        elif self.indicator_type == IndicatorType.ADX_TREND:
            return f"ADX({self.parameters['period']}) {self.comparison_operator} {self.threshold}"
        else:
            return f"{self.indicator_type.value} {self.comparison_operator}"


@dataclass
class ExitLogicGene:
    """Exit logic gene for stop-loss and take-profit."""

    stop_loss_type: str  # "fixed", "atr", "percentage", "trailing"
    stop_loss_value: float
    take_profit_type: str  # "fixed", "rrr", "percentage", "dynamic"
    take_profit_value: float
    trailing_enabled: bool = False
    trailing_distance: Optional[float] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    def describe(self) -> str:
        """Human-readable description."""
        sl_desc = f"SL: {self.stop_loss_value:.2f} {self.stop_loss_type}"
        tp_desc = f"TP: {self.take_profit_value:.2f} {self.take_profit_type}"
        trailing = " (trailing)" if self.trailing_enabled else ""
        return f"{sl_desc}, {tp_desc}{trailing}"


@dataclass
class StrategyChromosome:
    """
    Complete strategy chromosome encoding all strategic elements.

    This is the evolvable unit in the genetic algorithm.
    """

    # Entry conditions
    long_entry_conditions: List[ConditionGene] = field(default_factory=list)
    short_entry_conditions: List[ConditionGene] = field(default_factory=list)

    # Logical operators between conditions
    long_entry_operator: LogicalOperator = LogicalOperator.AND
    short_entry_operator: LogicalOperator = LogicalOperator.AND

    # Exit logic
    long_exit_logic: Optional[ExitLogicGene] = None
    short_exit_logic: Optional[ExitLogicGene] = None

    # Risk parameters
    risk_per_trade: float = 0.01  # 1% default
    max_positions: int = 1

    # Metadata
    generation: int = 0
    fitness: float = -999.0
    parent_ids: List[str] = field(default_factory=list)
    chromosome_id: str = field(default_factory=lambda: "")

    def __post_init__(self):
        """Generate unique ID if not provided."""
        if not self.chromosome_id:
            self.chromosome_id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate unique chromosome ID."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def to_dict(self) -> Dict:
        """Convert chromosome to dictionary for serialization."""
        return {
            "chromosome_id": self.chromosome_id,
            "generation": self.generation,
            "fitness": self.fitness,
            "long_entry_conditions": [c.to_dict() for c in self.long_entry_conditions],
            "short_entry_conditions": [
                c.to_dict() for c in self.short_entry_conditions
            ],
            "long_entry_operator": self.long_entry_operator.value,
            "short_entry_operator": self.short_entry_operator.value,
            "long_exit_logic": (
                self.long_exit_logic.to_dict() if self.long_exit_logic else None
            ),
            "short_exit_logic": (
                self.short_exit_logic.to_dict() if self.short_exit_logic else None
            ),
            "risk_per_trade": self.risk_per_trade,
            "max_positions": self.max_positions,
            "parent_ids": self.parent_ids,
        }

    def describe(self) -> str:
        """Generate human-readable strategy description."""
        desc = [
            f"═══════════════════════════════════════════════════════════",
            f"Strategy Chromosome ID: {self.chromosome_id}",
            f"Generation: {self.generation} | Fitness: {self.fitness:.3f}",
            f"═══════════════════════════════════════════════════════════",
            "",
            "LONG ENTRY CONDITIONS:",
        ]

        if self.long_entry_conditions:
            operator = f" {self.long_entry_operator.value} "
            conditions = operator.join(
                [
                    f"({c.describe()})"
                    for c in self.long_entry_conditions
                    if c.enabled
                ]
            )
            desc.append(f"  {conditions}")
        else:
            desc.append("  [None]")

        desc.append("")
        desc.append("SHORT ENTRY CONDITIONS:")

        if self.short_entry_conditions:
            operator = f" {self.short_entry_operator.value} "
            conditions = operator.join(
                [
                    f"({c.describe()})"
                    for c in self.short_entry_conditions
                    if c.enabled
                ]
            )
            desc.append(f"  {conditions}")
        else:
            desc.append("  [None]")

        desc.append("")
        desc.append("EXIT LOGIC:")
        if self.long_exit_logic:
            desc.append(f"  Long: {self.long_exit_logic.describe()}")
        if self.short_exit_logic:
            desc.append(f"  Short: {self.short_exit_logic.describe()}")

        desc.append("")
        desc.append(f"RISK: {self.risk_per_trade:.1%} per trade")
        desc.append(f"MAX POSITIONS: {self.max_positions}")
        desc.append(f"═══════════════════════════════════════════════════════════")

        return "\n".join(desc)

    def get_complexity(self) -> int:
        """Calculate strategy complexity (number of active conditions)."""
        complexity = 0
        complexity += sum(1 for c in self.long_entry_conditions if c.enabled)
        complexity += sum(1 for c in self.short_entry_conditions if c.enabled)
        return complexity

    def clone(self) -> "StrategyChromosome":
        """Create a deep copy of this chromosome."""
        return copy.deepcopy(self)

    def is_valid(self) -> bool:
        """Validate chromosome has minimum viable structure."""
        # Must have at least one entry condition
        has_long_entry = any(c.enabled for c in self.long_entry_conditions)
        has_short_entry = any(c.enabled for c in self.short_entry_conditions)

        if not (has_long_entry or has_short_entry):
            return False

        # Must have exit logic
        if has_long_entry and not self.long_exit_logic:
            return False
        if has_short_entry and not self.short_exit_logic:
            return False

        # Risk parameters must be reasonable
        if not (0.001 <= self.risk_per_trade <= 0.05):
            return False

        return True


class ChromosomeFactory:
    """Factory for creating initial chromosomes."""

    @staticmethod
    def create_random_chromosome(
        max_conditions: int = 3, generation: int = 0
    ) -> StrategyChromosome:
        """Create a random viable chromosome."""
        import random

        chromosome = StrategyChromosome(generation=generation)

        # Random long entry conditions
        num_long_conditions = random.randint(1, max_conditions)
        for _ in range(num_long_conditions):
            chromosome.long_entry_conditions.append(
                ChromosomeFactory._random_condition()
            )

        # Random short entry conditions
        num_short_conditions = random.randint(1, max_conditions)
        for _ in range(num_short_conditions):
            chromosome.short_entry_conditions.append(
                ChromosomeFactory._random_condition()
            )

        # Random operators
        chromosome.long_entry_operator = random.choice(list(LogicalOperator))
        chromosome.short_entry_operator = random.choice(list(LogicalOperator))

        # Random exit logic
        chromosome.long_exit_logic = ChromosomeFactory._random_exit_logic()
        chromosome.short_exit_logic = ChromosomeFactory._random_exit_logic()

        # Risk parameters
        chromosome.risk_per_trade = random.choice([0.005, 0.01, 0.015, 0.02])
        chromosome.max_positions = random.randint(1, 3)

        return chromosome

    @staticmethod
    def _random_condition() -> ConditionGene:
        """Create random condition gene."""
        import random

        indicator = random.choice(list(IndicatorType))

        if indicator == IndicatorType.EMA_CROSSOVER:
            return ConditionGene(
                indicator_type=indicator,
                parameters={
                    "fast": random.choice([8, 10, 12, 15]),
                    "slow": random.choice([26, 30, 34, 40]),
                },
                comparison_operator=random.choice(["cross_above", "cross_below"]),
            )
        elif indicator == IndicatorType.RSI_THRESHOLD:
            is_overbought = random.random() > 0.5
            return ConditionGene(
                indicator_type=indicator,
                parameters={"period": random.choice([10, 12, 14, 16])},
                comparison_operator=">" if is_overbought else "<",
                threshold=random.choice([70, 75, 80] if is_overbought else [20, 25, 30]),
            )
        elif indicator == IndicatorType.ADX_TREND:
            return ConditionGene(
                indicator_type=indicator,
                parameters={"period": 14},
                comparison_operator=">",
                threshold=random.choice([20, 25, 30]),
            )
        else:
            # Default generic condition
            return ConditionGene(
                indicator_type=indicator,
                parameters={},
                comparison_operator=">",
                threshold=0.0,
            )

    @staticmethod
    def _random_exit_logic() -> ExitLogicGene:
        """Create random exit logic gene."""
        import random

        return ExitLogicGene(
            stop_loss_type=random.choice(["atr", "percentage"]),
            stop_loss_value=random.choice([1.0, 1.5, 2.0, 2.5]),
            take_profit_type=random.choice(["rrr", "percentage"]),
            take_profit_value=random.choice([2.0, 2.5, 3.0, 4.0]),
            trailing_enabled=random.random() > 0.7,
            trailing_distance=random.choice([1.5, 2.0]) if random.random() > 0.7 else None,
        )

    @staticmethod
    def create_simple_ema_rsi_strategy() -> StrategyChromosome:
        """Create a simple EMA+RSI strategy as baseline."""
        chromosome = StrategyChromosome(generation=0)

        # Long: EMA fast > slow AND RSI < 30
        chromosome.long_entry_conditions = [
            ConditionGene(
                indicator_type=IndicatorType.EMA_CROSSOVER,
                parameters={"fast": 12, "slow": 26},
                comparison_operator="cross_above",
            ),
            ConditionGene(
                indicator_type=IndicatorType.RSI_THRESHOLD,
                parameters={"period": 14},
                comparison_operator="<",
                threshold=30,
            ),
        ]

        # Short: EMA fast < slow AND RSI > 70
        chromosome.short_entry_conditions = [
            ConditionGene(
                indicator_type=IndicatorType.EMA_CROSSOVER,
                parameters={"fast": 12, "slow": 26},
                comparison_operator="cross_below",
            ),
            ConditionGene(
                indicator_type=IndicatorType.RSI_THRESHOLD,
                parameters={"period": 14},
                comparison_operator=">",
                threshold=70,
            ),
        ]

        chromosome.long_entry_operator = LogicalOperator.AND
        chromosome.short_entry_operator = LogicalOperator.AND

        # Standard exit logic
        chromosome.long_exit_logic = ExitLogicGene(
            stop_loss_type="atr",
            stop_loss_value=2.0,
            take_profit_type="rrr",
            take_profit_value=3.0,
        )
        chromosome.short_exit_logic = ExitLogicGene(
            stop_loss_type="atr",
            stop_loss_value=2.0,
            take_profit_type="rrr",
            take_profit_value=3.0,
        )

        chromosome.risk_per_trade = 0.01
        chromosome.max_positions = 1

        return chromosome
