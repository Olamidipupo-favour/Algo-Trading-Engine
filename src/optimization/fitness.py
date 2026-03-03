"""
Composite fitness calculation with overfitting penalties.

Fitness function balances multiple performance metrics while penalizing
complexity and instability to produce robust, generalizable strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from src.utils.logger import Logger

logger = Logger(__name__)


@dataclass
class TradeMetrics:
    """Container for trade-level performance metrics."""

    returns: np.ndarray
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    total_trades: int
    avg_win: float
    avg_loss: float
    consecutive_losses: int
    recovery_factor: float


class FitnessCalculator:
    """
    Calculate composite fitness score for strategy evaluation.

    Fitness formula:
        fitness = (0.4 * Sharpe)
                + (0.3 * ProfitFactor)
                + (0.2 * WinRate)
                - (0.3 * MaxDrawdown)
                - (0.05 * ComplexityPenalty)

    Additional constraints:
        - Minimum trade count threshold
        - Maximum allowed drawdown
        - Stability requirements
    """

    def __init__(
        self,
        min_trades: int = 30,
        max_drawdown: float = 0.15,
        min_sharpe: float = 0.5,
        complexity_weight: float = 0.05,
    ):
        """
        Initialize fitness calculator with quality thresholds.

        Args:
            min_trades: Minimum number of trades required
            max_drawdown: Maximum acceptable drawdown (decimal)
            min_sharpe: Minimum acceptable Sharpe ratio
            complexity_weight: Weight for complexity penalty
        """
        self.min_trades = min_trades
        self.max_drawdown = max_drawdown
        self.min_sharpe = min_sharpe
        self.complexity_weight = complexity_weight

        logger.info(
            f"FitnessCalculator initialized: "
            f"min_trades={min_trades}, max_dd={max_drawdown}, "
            f"min_sharpe={min_sharpe}"
        )

    def calculate_metrics(
        self,
        equity_curve: pd.Series,
        trades: List[Dict],
        risk_free_rate: float = 0.02,
    ) -> TradeMetrics:
        """
        Calculate comprehensive trading metrics from equity curve and trades.

        Args:
            equity_curve: Time series of equity values
            trades: List of trade dictionaries with 'pnl', 'type', etc.
            risk_free_rate: Annual risk-free rate for Sharpe calculation

        Returns:
            TradeMetrics object with all calculated metrics
        """
        if len(trades) < self.min_trades:
            logger.warning(
                f"Insufficient trades: {len(trades)} < {self.min_trades}"
            )
            return self._create_failed_metrics(len(trades))

        # Extract returns
        pnl_values = np.array([t.get('pnl', 0.0) for t in trades])

        # Calculate win/loss statistics
        wins = pnl_values[pnl_values > 0]
        losses = pnl_values[pnl_values < 0]

        win_rate = len(wins) / len(pnl_values) if len(pnl_values) > 0 else 0.0
        avg_win = np.mean(wins) if len(wins) > 0 else 0.0
        avg_loss = np.mean(np.abs(losses)) if len(losses) > 0 else 1.0

        # Profit factor
        total_wins = np.sum(wins)
        total_losses = np.sum(np.abs(losses))
        profit_factor = (
            total_wins / total_losses if total_losses > 0 else 0.0
        )

        # Sharpe ratio
        returns = equity_curve.pct_change().dropna()
        if len(returns) > 0 and returns.std() > 0:
            sharpe_ratio = (
                (returns.mean() * 252 - risk_free_rate)
                / (returns.std() * np.sqrt(252))
            )
        else:
            sharpe_ratio = 0.0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = (
                (returns.mean() * 252 - risk_free_rate)
                / (downside_returns.std() * np.sqrt(252))
            )
        else:
            sortino_ratio = sharpe_ratio

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())

        # Consecutive losses
        consecutive_losses = self._calculate_consecutive_losses(pnl_values)

        # Recovery factor
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        recovery_factor = (
            total_return / max_drawdown if max_drawdown > 0 else 0.0
        )

        return TradeMetrics(
            returns=returns.values,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            total_trades=len(trades),
            avg_win=avg_win,
            avg_loss=avg_loss,
            consecutive_losses=consecutive_losses,
            recovery_factor=recovery_factor,
        )

    def calculate_fitness(
        self,
        metrics: TradeMetrics,
        num_conditions: int,
        stability_score: Optional[float] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate composite fitness score with penalties.

        Args:
            metrics: Calculated trade metrics
            num_conditions: Number of conditions in strategy (complexity)
            stability_score: Optional stability score from perturbation test

        Returns:
            Tuple of (fitness_score, component_scores_dict)
        """
        # Check disqualifying conditions
        if metrics.total_trades < self.min_trades:
            logger.warning(
                f"Disqualified: {metrics.total_trades} < {self.min_trades} trades"
            )
            return -999.0, {"reason": "insufficient_trades"}

        if metrics.max_drawdown > self.max_drawdown:
            logger.warning(
                f"Disqualified: DD {metrics.max_drawdown:.2%} > "
                f"{self.max_drawdown:.2%}"
            )
            return -999.0, {"reason": "excessive_drawdown"}

        if metrics.sharpe_ratio < self.min_sharpe:
            logger.warning(
                f"Disqualified: Sharpe {metrics.sharpe_ratio:.2f} < "
                f"{self.min_sharpe:.2f}"
            )
            return -999.0, {"reason": "insufficient_sharpe"}

        # Calculate component scores
        sharpe_component = 0.4 * min(metrics.sharpe_ratio, 3.0)

        # Cap profit factor contribution at 3.0
        pf_component = 0.3 * min(metrics.profit_factor, 3.0)

        # Win rate component
        wr_component = 0.2 * metrics.win_rate

        # Drawdown penalty (negative)
        dd_penalty = -0.3 * metrics.max_drawdown

        # Complexity penalty (scaled by number of conditions)
        complexity_penalty = -self.complexity_weight * (num_conditions / 10.0)

        # Base fitness
        fitness = (
            sharpe_component
            + pf_component
            + wr_component
            + dd_penalty
            + complexity_penalty
        )

        # Stability bonus/penalty
        if stability_score is not None:
            stability_adjustment = 0.1 * (stability_score - 0.5)
            fitness += stability_adjustment
        else:
            stability_adjustment = 0.0

        # Component breakdown for transparency
        components = {
            "sharpe_component": sharpe_component,
            "profit_factor_component": pf_component,
            "win_rate_component": wr_component,
            "drawdown_penalty": dd_penalty,
            "complexity_penalty": complexity_penalty,
            "stability_adjustment": stability_adjustment,
            "total_fitness": fitness,
        }

        logger.debug(
            f"Fitness: {fitness:.3f} | Sharpe: {metrics.sharpe_ratio:.2f} | "
            f"PF: {metrics.profit_factor:.2f} | WR: {metrics.win_rate:.1%} | "
            f"DD: {metrics.max_drawdown:.1%}"
        )

        return fitness, components

    def _calculate_consecutive_losses(self, pnl_values: np.ndarray) -> int:
        """Calculate maximum consecutive losses."""
        max_consecutive = 0
        current_consecutive = 0

        for pnl in pnl_values:
            if pnl < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def _create_failed_metrics(self, num_trades: int) -> TradeMetrics:
        """Create metrics object for failed evaluation."""
        return TradeMetrics(
            returns=np.array([]),
            win_rate=0.0,
            profit_factor=0.0,
            sharpe_ratio=-999.0,
            sortino_ratio=-999.0,
            max_drawdown=1.0,
            total_trades=num_trades,
            avg_win=0.0,
            avg_loss=0.0,
            consecutive_losses=999,
            recovery_factor=0.0,
        )

    def compare_strategies(
        self,
        current_fitness: float,
        new_fitness: float,
        improvement_threshold: float = 0.05,
    ) -> bool:
        """
        Determine if new strategy is significantly better than current.

        Args:
            current_fitness: Fitness of current strategy
            new_fitness: Fitness of candidate strategy
            improvement_threshold: Minimum improvement required

        Returns:
            True if new strategy should replace current
        """
        improvement = (new_fitness - current_fitness) / abs(current_fitness)

        if improvement >= improvement_threshold:
            logger.info(
                f"Strategy improvement: {improvement:.1%} "
                f"({current_fitness:.3f} -> {new_fitness:.3f})"
            )
            return True

        logger.debug(
            f"Insufficient improvement: {improvement:.1%} < "
            f"{improvement_threshold:.1%}"
        )
        return False
