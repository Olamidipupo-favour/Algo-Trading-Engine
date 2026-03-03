"""
Probation manager for detecting and responding to strategy performance decay.

Implements flat period survival logic with automatic risk reduction
and re-optimization triggers.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import deque

from src.utils.logger import Logger

logger = Logger(__name__)


@dataclass
class PerformanceWindow:
    """Performance metrics over a time window."""

    sharpe_ratio: float
    returns: float
    num_trades: int
    win_rate: float
    max_drawdown: float
    start_time: datetime
    end_time: datetime


class ProbationManager:
    """
    Monitor strategy performance and trigger protective actions.

    Detects:
    - Equity stagnation
    - Performance decay
    - Consecutive losses

    Actions:
    - Risk reduction
    - Probation mode
    - Strategy suspension
    - Re-optimization trigger
    """

    def __init__(
        self,
        stagnation_threshold: float = 0.01,  # 1% growth required
        stagnation_lookback_trades: int = 30,
        decay_threshold: float = 0.30,  # 30% Sharpe degradation
        rolling_window_trades: int = 50,
        max_consecutive_losses: int = 5,
        probation_risk_multiplier: float = 0.5,
    ):
        """
        Initialize probation manager.

        Args:
            stagnation_threshold: Minimum equity growth required
            stagnation_lookback_trades: Trades to check for stagnation
            decay_threshold: Maximum allowed Sharpe degradation
            rolling_window_trades: Window for rolling Sharpe calculation
            max_consecutive_losses: Max consecutive losses before action
            probation_risk_multiplier: Risk multiplier during probation
        """
        self.stagnation_threshold = stagnation_threshold
        self.stagnation_lookback = stagnation_lookback_trades
        self.decay_threshold = decay_threshold
        self.rolling_window = rolling_window_trades
        self.max_consecutive_losses = max_consecutive_losses
        self.probation_risk_multiplier = probation_risk_multiplier

        # State tracking
        self.recent_trades: deque = deque(maxlen=max(rolling_window_trades, stagnation_lookback_trades))
        self.equity_curve: List[float] = []
        self.baseline_sharpe: Optional[float] = None
        self.probation_mode: bool = False
        self.suspended: bool = False
        self.consecutive_losses: int = 0

        logger.info(
            f"ProbationManager initialized: "
            f"stagnation_thresh={stagnation_threshold:.1%}, "
            f"decay_thresh={decay_threshold:.1%}"
        )

    def update(self, trade: Dict, current_equity: float) -> Dict[str, any]:
        """
        Update with new trade and equity.

        Args:
            trade: Trade dictionary with pnl, timestamp, etc.
            current_equity: Current account equity

        Returns:
            Dict with actions to take
        """
        self.recent_trades.append(trade)
        self.equity_curve.append(current_equity)

        # Track consecutive losses
        if trade.get("pnl", 0) < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        actions = {
            "reduce_risk": False,
            "enter_probation": False,
            "suspend_strategy": False,
            "trigger_reoptimization": False,
            "risk_multiplier": 1.0,
            "reason": "",
        }

        # Check for consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            logger.warning(
                f"Max consecutive losses reached: {self.consecutive_losses}"
            )
            actions["reduce_risk"] = True
            actions["enter_probation"] = True
            actions["risk_multiplier"] = self.probation_risk_multiplier
            actions["reason"] = "consecutive_losses"
            self.probation_mode = True

        # Check for equity stagnation
        if len(self.recent_trades) >= self.stagnation_lookback:
            stagnation_detected = self._detect_equity_stagnation()
            if stagnation_detected:
                logger.warning("Equity stagnation detected")
                actions["reduce_risk"] = True
                actions["risk_multiplier"] = 0.5
                actions["reason"] = "equity_stagnation"

        # Check for performance decay
        if len(self.recent_trades) >= self.rolling_window:
            decay_detected, decay_amount = self._detect_performance_decay()
            if decay_detected:
                logger.warning(
                    f"Performance decay detected: {decay_amount:.1%}"
                )
                actions["enter_probation"] = True
                actions["trigger_reoptimization"] = True
                actions["risk_multiplier"] = self.probation_risk_multiplier
                actions["reason"] = f"performance_decay_{decay_amount:.1%}"
                self.probation_mode = True

        # Check if probation should continue
        if self.probation_mode and not actions["enter_probation"]:
            # Check if performance recovered
            if self._check_recovery():
                logger.info("Strategy recovered, exiting probation")
                self.probation_mode = False
            else:
                logger.info("Probation continues")
                actions["risk_multiplier"] = self.probation_risk_multiplier

        # Check for suspension
        if self.probation_mode and len(self.recent_trades) >= 20:
            recent_sharpe = self._calculate_recent_sharpe()
            if recent_sharpe < 0:
                logger.error("Strategy performance negative during probation, suspending")
                actions["suspend_strategy"] = True
                actions["risk_multiplier"] = 0.0
                actions["reason"] = "negative_sharpe_in_probation"
                self.suspended = True

        return actions

    def _detect_equity_stagnation(self) -> bool:
        """Detect if equity has stagnated."""
        if len(self.equity_curve) < self.stagnation_lookback:
            return False

        lookback_equity = self.equity_curve[-self.stagnation_lookback]
        current_equity = self.equity_curve[-1]

        growth = (current_equity - lookback_equity) / lookback_equity

        logger.debug(
            f"Equity growth over {self.stagnation_lookback} trades: {growth:.2%}"
        )

        return growth < self.stagnation_threshold

    def _detect_performance_decay(self) -> Tuple[bool, float]:
        """Detect if performance has decayed significantly."""
        # Calculate rolling Sharpe
        rolling_sharpe = self._calculate_recent_sharpe()

        # Initialize baseline if needed
        if self.baseline_sharpe is None:
            if len(self.recent_trades) >= self.rolling_window * 2:
                self.baseline_sharpe = self._calculate_baseline_sharpe()
                logger.info(f"Baseline Sharpe established: {self.baseline_sharpe:.2f}")
            return False, 0.0

        # Compare rolling to baseline
        if self.baseline_sharpe > 0:
            decay_ratio = (self.baseline_sharpe - rolling_sharpe) / self.baseline_sharpe
        else:
            decay_ratio = 0.0

        logger.debug(
            f"Sharpe decay: baseline={self.baseline_sharpe:.2f}, "
            f"rolling={rolling_sharpe:.2f}, decay={decay_ratio:.1%}"
        )

        is_decayed = decay_ratio > self.decay_threshold

        return is_decayed, decay_ratio

    def _calculate_recent_sharpe(self) -> float:
        """Calculate Sharpe ratio over recent trades."""
        if len(self.recent_trades) < 10:
            return 0.0

        recent_pnl = [t.get("pnl", 0) for t in list(self.recent_trades)[-self.rolling_window:]]

        if not recent_pnl:
            return 0.0

        mean_return = np.mean(recent_pnl)
        std_return = np.std(recent_pnl)

        if std_return == 0:
            return 0.0

        sharpe = mean_return / std_return * np.sqrt(252)  # Annualized

        return sharpe

    def _calculate_baseline_sharpe(self) -> float:
        """Calculate baseline Sharpe from first half of available trades."""
        baseline_trades = list(self.recent_trades)[: len(self.recent_trades) // 2]

        if len(baseline_trades) < 10:
            return 1.0  # Default optimistic baseline

        pnl_values = [t.get("pnl", 0) for t in baseline_trades]
        mean_return = np.mean(pnl_values)
        std_return = np.std(pnl_values)

        if std_return == 0:
            return 1.0

        sharpe = mean_return / std_return * np.sqrt(252)

        return max(sharpe, 0.5)  # Minimum baseline

    def _check_recovery(self) -> bool:
        """Check if strategy has recovered from poor performance."""
        if len(self.recent_trades) < 20:
            return False

        # Check last 20 trades
        recent_20 = list(self.recent_trades)[-20:]
        recent_pnl = [t.get("pnl", 0) for t in recent_20]

        # Must have positive return
        total_return = sum(recent_pnl)
        if total_return <= 0:
            return False

        # Must have reasonable win rate
        wins = sum(1 for pnl in recent_pnl if pnl > 0)
        win_rate = wins / len(recent_pnl)
        if win_rate < 0.4:
            return False

        # Must have positive Sharpe
        recent_sharpe = self._calculate_recent_sharpe()
        if recent_sharpe < 0.5:
            return False

        logger.info(
            f"Recovery detected: return={total_return:.2f}, "
            f"wr={win_rate:.1%}, sharpe={recent_sharpe:.2f}"
        )

        return True

    def get_status(self) -> Dict:
        """Get current probation status."""
        recent_sharpe = self._calculate_recent_sharpe() if len(self.recent_trades) >= 10 else 0.0

        return {
            "probation_mode": self.probation_mode,
            "suspended": self.suspended,
            "consecutive_losses": self.consecutive_losses,
            "total_trades_monitored": len(self.recent_trades),
            "baseline_sharpe": self.baseline_sharpe,
            "recent_sharpe": recent_sharpe,
            "current_equity": self.equity_curve[-1] if self.equity_curve else 0.0,
        }

    def reset(self) -> None:
        """Reset probation state."""
        logger.info("Resetting probation manager")
        self.probation_mode = False
        self.suspended = False
        self.consecutive_losses = 0
        self.baseline_sharpe = None
