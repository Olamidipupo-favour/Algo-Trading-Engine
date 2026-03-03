"""
Regime-aware strategy management.

Activates strategies only in regimes where they are historically profitable,
with automatic fallback to cash preservation when no strategy qualifies.
"""

import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

from src.utils.logger import Logger
from .classifier import RegimeClassifier, RegimeType, RegimeState

logger = Logger(__name__)


@dataclass
class StrategyPerformance:
    """Track strategy performance by regime."""

    strategy_id: str
    regime_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    overall_fitness: float = 0.0
    total_trades: int = 0
    is_active: bool = False


class RegimeManager:
    """
    Manage strategy selection based on market regime.

    Only activates strategies in regimes where they historically performed well.
    Falls back to cash when no qualified strategy exists.
    """

    def __init__(
        self,
        min_trades_per_regime: int = 10,
        min_regime_sharpe: float = 0.5,
        min_regime_profit_factor: float = 1.2,
        confidence_threshold: float = 0.6,
    ):
        """
        Initialize regime manager.

        Args:
            min_trades_per_regime: Minimum trades required for regime qualification
            min_regime_sharpe: Minimum Sharpe ratio in regime
            min_regime_profit_factor: Minimum profit factor in regime
            confidence_threshold: Minimum regime confidence to trade
        """
        self.min_trades_per_regime = min_trades_per_regime
        self.min_regime_sharpe = min_regime_sharpe
        self.min_regime_profit_factor = min_regime_profit_factor
        self.confidence_threshold = confidence_threshold

        self.classifier = RegimeClassifier()
        self.strategy_registry: Dict[str, StrategyPerformance] = {}
        self.active_strategy_id: Optional[str] = None
        self.cash_mode: bool = False

        logger.info(
            f"RegimeManager initialized: "
            f"min_trades={min_trades_per_regime}, "
            f"min_sharpe={min_regime_sharpe}, "
            f"min_pf={min_regime_profit_factor}"
        )

    def register_strategy(
        self, strategy_id: str, performance_by_regime: Dict[str, Dict[str, float]]
    ) -> None:
        """
        Register strategy with regime-specific performance.

        Args:
            strategy_id: Unique strategy identifier
            performance_by_regime: Dict mapping regime names to performance metrics
                Example: {
                    "trending_up": {"sharpe": 1.2, "pf": 1.8, "trades": 50},
                    "ranging": {"sharpe": 0.3, "pf": 0.9, "trades": 30}
                }
        """
        total_trades = sum(
            stats.get("trades", 0) for stats in performance_by_regime.values()
        )

        perf = StrategyPerformance(
            strategy_id=strategy_id,
            regime_stats=performance_by_regime,
            total_trades=total_trades,
        )

        self.strategy_registry[strategy_id] = perf

        logger.info(
            f"Registered strategy {strategy_id} with {total_trades} total trades "
            f"across {len(performance_by_regime)} regimes"
        )

    def select_strategy(self, data: pd.DataFrame) -> Optional[str]:
        """
        Select best strategy for current market regime.

        Args:
            data: Recent OHLCV data for regime classification

        Returns:
            Strategy ID to activate, or None if cash mode
        """
        # Classify current regime
        regime_state = self.classifier.classify(data)

        logger.info(
            f"Regime: {regime_state.primary_regime.value} | "
            f"Confidence: {regime_state.confidence:.2f} | "
            f"Trend: {regime_state.trend_strength:.2f}"
        )

        # Check if regime confidence is sufficient
        if regime_state.confidence < self.confidence_threshold:
            logger.warning(
                f"Regime confidence too low ({regime_state.confidence:.2f}), "
                f"entering cash mode"
            )
            self.cash_mode = True
            self.active_strategy_id = None
            return None

        # Find qualified strategies for this regime
        qualified_strategies = self._find_qualified_strategies(regime_state)

        if not qualified_strategies:
            logger.warning(
                f"No qualified strategies for regime {regime_state.primary_regime.value}, "
                f"entering cash mode"
            )
            self.cash_mode = True
            self.active_strategy_id = None
            return None

        # Select best qualified strategy
        best_strategy = max(
            qualified_strategies,
            key=lambda s: self._get_regime_performance(s, regime_state.primary_regime),
        )

        self.cash_mode = False
        self.active_strategy_id = best_strategy

        logger.info(
            f"Activated strategy: {best_strategy} for regime "
            f"{regime_state.primary_regime.value}"
        )

        return best_strategy

    def _find_qualified_strategies(
        self, regime_state: RegimeState
    ) -> List[str]:
        """Find strategies qualified for current regime."""
        qualified = []
        regime_key = regime_state.primary_regime.value

        for strategy_id, perf in self.strategy_registry.items():
            if regime_key not in perf.regime_stats:
                continue

            stats = perf.regime_stats[regime_key]

            # Check qualification criteria
            has_enough_trades = stats.get("trades", 0) >= self.min_trades_per_regime
            meets_sharpe = stats.get("sharpe", 0) >= self.min_regime_sharpe
            meets_pf = stats.get("pf", 0) >= self.min_regime_profit_factor

            if has_enough_trades and meets_sharpe and meets_pf:
                qualified.append(strategy_id)
                logger.debug(
                    f"Strategy {strategy_id} qualified: "
                    f"trades={stats.get('trades', 0)}, "
                    f"sharpe={stats.get('sharpe', 0):.2f}, "
                    f"pf={stats.get('pf', 0):.2f}"
                )

        return qualified

    def _get_regime_performance(
        self, strategy_id: str, regime: RegimeType
    ) -> float:
        """Get strategy performance score for regime."""
        perf = self.strategy_registry.get(strategy_id)
        if not perf:
            return 0.0

        regime_key = regime.value
        if regime_key not in perf.regime_stats:
            return 0.0

        stats = perf.regime_stats[regime_key]

        # Composite score
        sharpe_score = stats.get("sharpe", 0) * 0.6
        pf_score = stats.get("pf", 0) * 0.4

        return sharpe_score + pf_score

    def should_close_positions(self, data: pd.DataFrame) -> bool:
        """
        Determine if open positions should be closed due to regime change.

        Args:
            data: Recent OHLCV data

        Returns:
            True if positions should be closed
        """
        regime_state = self.classifier.classify(data)

        # Close if transitioning to unknown/transitional regime
        if regime_state.primary_regime in [
            RegimeType.TRANSITIONAL,
            RegimeType.UNKNOWN,
        ]:
            logger.warning("Transitional regime detected, closing positions")
            return True

        # Close if regime confidence drops
        if regime_state.confidence < self.confidence_threshold:
            logger.warning("Regime confidence dropped, closing positions")
            return True

        # Close if active strategy no longer qualified
        if self.active_strategy_id:
            qualified = self._find_qualified_strategies(regime_state)
            if self.active_strategy_id not in qualified:
                logger.warning(
                    f"Active strategy {self.active_strategy_id} no longer qualified, "
                    f"closing positions"
                )
                return True

        return False

    def get_status_report(self) -> Dict[str, Any]:
        """Generate status report."""
        regime_stats = self.classifier.get_regime_statistics()

        report = {
            "current_regime": (
                self.classifier.current_regime.primary_regime.value
                if self.classifier.current_regime
                else "unknown"
            ),
            "regime_confidence": (
                self.classifier.current_regime.confidence
                if self.classifier.current_regime
                else 0.0
            ),
            "active_strategy": self.active_strategy_id,
            "cash_mode": self.cash_mode,
            "registered_strategies": len(self.strategy_registry),
            "regime_distribution": regime_stats.get("regime_distribution", {}),
        }

        return report
