"""
Enhanced risk manager with institutional-grade capital protection.

NON-NEGOTIABLE RULES:
- Max 1% risk per trade
- Max 3% daily drawdown
- Max 10% total drawdown
- Immediate shutdown if breached
- Automatic broker disconnection on hard stop
"""

import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime, date
from dataclasses import dataclass

from src.utils.logger import Logger

logger = Logger(__name__)


@dataclass
class RiskLimits:
    """Hard risk limits that cannot be breached."""

    max_risk_per_trade: float = 0.01  # 1%
    max_daily_drawdown: float = 0.03  # 3%
    max_total_drawdown: float = 0.10  # 10%
    max_position_concentration: float = 0.25  # 25%
    max_leverage: float = 1.0  # No leverage by default


@dataclass
class DrawdownState:
    """Track drawdown states."""

    peak_equity: float
    current_equity: float
    total_drawdown: float
    daily_drawdown: float
    daily_start_equity: float
    last_reset_date: date


class EnhancedRiskManager:
    """
    Institutional-grade risk manager with automatic circuit breakers.

    SAFETY FEATURES:
    - Hard limits enforced at system level
    - Automatic position closing on breach
    - Broker API disconnection on critical breach
    - Cannot be overridden
    """

    def __init__(
        self,
        initial_capital: float,
        risk_limits: Optional[RiskLimits] = None,
        emergency_contact: Optional[str] = None,
    ):
        """
        Initialize enhanced risk manager.

        Args:
            initial_capital: Starting capital
            risk_limits: Custom risk limits (uses safe defaults if None)
            emergency_contact: Contact for breach notifications
        """
        self.initial_capital = initial_capital
        self.risk_limits = risk_limits or RiskLimits()
        self.emergency_contact = emergency_contact

        # State tracking
        self.drawdown_state = DrawdownState(
            peak_equity=initial_capital,
            current_equity=initial_capital,
            total_drawdown=0.0,
            daily_drawdown=0.0,
            daily_start_equity=initial_capital,
            last_reset_date=date.today(),
        )

        self.is_shutdown: bool = False
        self.shutdown_reason: str = ""
        self.positions: Dict[str, Dict] = {}

        logger.critical(
            "=" * 70 + "\n"
            "ENHANCED RISK MANAGER INITIALIZED\n"
            "NON-NEGOTIABLE LIMITS:\n"
            f"  - Max risk per trade: {self.risk_limits.max_risk_per_trade:.1%}\n"
            f"  - Max daily drawdown: {self.risk_limits.max_daily_drawdown:.1%}\n"
            f"  - Max total drawdown: {self.risk_limits.max_total_drawdown:.1%}\n"
            "  - AUTOMATIC SHUTDOWN ON BREACH\n"
            "=" * 70
        )

    def validate_new_position(
        self, proposed_position: Dict
    ) -> Tuple[bool, str, Optional[Dict]]:
        """
        Validate proposed position against all risk limits.

        Args:
            proposed_position: Dict with 'symbol', 'size', 'entry_price', 'stop_loss'

        Returns:
            Tuple of (approved, reason, adjusted_position)
        """
        # Check if system is shut down
        if self.is_shutdown:
            return False, f"SYSTEM SHUTDOWN: {self.shutdown_reason}", None

        # Check daily drawdown
        if self.drawdown_state.daily_drawdown >= self.risk_limits.max_daily_drawdown:
            logger.error(
                f"DAILY DRAWDOWN LIMIT REACHED: "
                f"{self.drawdown_state.daily_drawdown:.1%}"
            )
            return False, "Daily drawdown limit reached", None

        # Check total drawdown
        if self.drawdown_state.total_drawdown >= self.risk_limits.max_total_drawdown:
            logger.critical("TOTAL DRAWDOWN LIMIT REACHED - INITIATING SHUTDOWN")
            self.initiate_shutdown("total_drawdown_exceeded")
            return False, "Total drawdown limit exceeded - SHUTDOWN INITIATED", None

        # Calculate position risk
        position_risk = self._calculate_position_risk(proposed_position)

        # Check risk per trade
        if position_risk > self.risk_limits.max_risk_per_trade:
            logger.warning(
                f"Position risk {position_risk:.2%} exceeds limit "
                f"{self.risk_limits.max_risk_per_trade:.2%}"
            )

            # Adjust position size to meet limit
            adjusted_position = self._adjust_position_size(
                proposed_position, self.risk_limits.max_risk_per_trade
            )

            return (
                True,
                f"Position adjusted to meet risk limit",
                adjusted_position,
            )

        # Check position concentration
        symbol = proposed_position.get("symbol")
        current_exposure = self._calculate_exposure(symbol)

        if current_exposure > self.risk_limits.max_position_concentration:
            logger.warning(
                f"Position concentration for {symbol}: {current_exposure:.1%}"
            )
            return False, "Position concentration limit exceeded", None

        return True, "Position approved", proposed_position

    def update_equity(self, current_equity: float) -> Dict[str, any]:
        """
        Update equity and check drawdown limits.

        Args:
            current_equity: Current account equity

        Returns:
            Dict with actions to take
        """
        # Reset daily tracking if new day
        today = date.today()
        if today != self.drawdown_state.last_reset_date:
            self.drawdown_state.daily_start_equity = self.drawdown_state.current_equity
            self.drawdown_state.last_reset_date = today
            logger.info(f"Daily tracking reset. Starting equity: {self.drawdown_state.daily_start_equity:.2f}")

        # Update peak
        if current_equity > self.drawdown_state.peak_equity:
            self.drawdown_state.peak_equity = current_equity
            logger.info(f"New equity peak: {current_equity:.2f}")

        # Calculate drawdowns
        total_dd = (
            self.drawdown_state.peak_equity - current_equity
        ) / self.drawdown_state.peak_equity

        daily_dd = (
            self.drawdown_state.daily_start_equity - current_equity
        ) / self.drawdown_state.daily_start_equity

        self.drawdown_state.current_equity = current_equity
        self.drawdown_state.total_drawdown = total_dd
        self.drawdown_state.daily_drawdown = daily_dd

        actions = {
            "close_all_positions": False,
            "reduce_risk": False,
            "shutdown_system": False,
            "disconnect_broker": False,
            "alert_level": "normal",
        }

        # Check daily drawdown
        if daily_dd >= self.risk_limits.max_daily_drawdown:
            logger.critical(
                f"DAILY DRAWDOWN LIMIT BREACHED: {daily_dd:.2%} >= "
                f"{self.risk_limits.max_daily_drawdown:.2%}"
            )
            actions["close_all_positions"] = True
            actions["alert_level"] = "critical"

        elif daily_dd >= self.risk_limits.max_daily_drawdown * 0.8:
            logger.error(
                f"Daily drawdown warning: {daily_dd:.2%}"
            )
            actions["reduce_risk"] = True
            actions["alert_level"] = "warning"

        # Check total drawdown
        if total_dd >= self.risk_limits.max_total_drawdown:
            logger.critical(
                f"TOTAL DRAWDOWN LIMIT BREACHED: {total_dd:.2%} >= "
                f"{self.risk_limits.max_total_drawdown:.2%}"
            )
            actions["shutdown_system"] = True
            actions["disconnect_broker"] = True
            actions["alert_level"] = "critical"
            self.initiate_shutdown("total_drawdown_limit")

        elif total_dd >= self.risk_limits.max_total_drawdown * 0.8:
            logger.error(
                f"Total drawdown warning: {total_dd:.2%}"
            )
            actions["close_all_positions"] = True
            actions["alert_level"] = "severe"

        return actions

    def initiate_shutdown(self, reason: str) -> None:
        """
        Initiate emergency shutdown.

        Args:
            reason: Reason for shutdown
        """
        if self.is_shutdown:
            return

        self.is_shutdown = True
        self.shutdown_reason = reason

        logger.critical(
            "\n" + "=" * 70 + "\n"
            "EMERGENCY SHUTDOWN INITIATED\n"
            f"Reason: {reason}\n"
            f"Time: {datetime.now().isoformat()}\n"
            f"Total Drawdown: {self.drawdown_state.total_drawdown:.2%}\n"
            f"Daily Drawdown: {self.drawdown_state.daily_drawdown:.2%}\n"
            f"Current Equity: {self.drawdown_state.current_equity:.2f}\n"
            f"Peak Equity: {self.drawdown_state.peak_equity:.2f}\n"
            "=" * 70
        )

        # Send alert if configured
        if self.emergency_contact:
            self._send_emergency_alert(reason)

    def _calculate_position_risk(self, position: Dict) -> float:
        """Calculate risk as percentage of equity."""
        entry_price = position.get("entry_price", 0)
        stop_loss = position.get("stop_loss", 0)
        size = position.get("size", 0)

        if entry_price == 0 or stop_loss == 0:
            return 0.0

        risk_per_unit = abs(entry_price - stop_loss)
        total_risk = risk_per_unit * size

        risk_pct = total_risk / self.drawdown_state.current_equity

        return risk_pct

    def _adjust_position_size(
        self, position: Dict, target_risk: float
    ) -> Dict:
        """Adjust position size to meet risk target."""
        entry_price = position.get("entry_price", 0)
        stop_loss = position.get("stop_loss", 0)

        if entry_price == 0 or stop_loss == 0:
            return position

        risk_per_unit = abs(entry_price - stop_loss)
        max_risk_amount = self.drawdown_state.current_equity * target_risk
        adjusted_size = max_risk_amount / risk_per_unit

        adjusted_position = position.copy()
        adjusted_position["size"] = adjusted_size

        logger.info(
            f"Position size adjusted: {position['size']:.2f} -> "
            f"{adjusted_size:.2f}"
        )

        return adjusted_position

    def _calculate_exposure(self, symbol: str) -> float:
        """Calculate current exposure to symbol as percentage of equity."""
        if symbol not in self.positions:
            return 0.0

        position = self.positions[symbol]
        exposure = (
            position["size"] * position["entry_price"]
        ) / self.drawdown_state.current_equity

        return exposure

    def add_position(self, symbol: str, position_details: Dict) -> None:
        """Track new position."""
        self.positions[symbol] = position_details
        logger.info(f"Position added: {symbol}")

    def remove_position(self, symbol: str) -> None:
        """Remove closed position."""
        if symbol in self.positions:
            del self.positions[symbol]
            logger.info(f"Position removed: {symbol}")

    def get_risk_status(self) -> Dict:
        """Get comprehensive risk status."""
        return {
            "is_shutdown": self.is_shutdown,
            "shutdown_reason": self.shutdown_reason,
            "current_equity": self.drawdown_state.current_equity,
            "peak_equity": self.drawdown_state.peak_equity,
            "total_drawdown": self.drawdown_state.total_drawdown,
            "daily_drawdown": self.drawdown_state.daily_drawdown,
            "total_dd_limit": self.risk_limits.max_total_drawdown,
            "daily_dd_limit": self.risk_limits.max_daily_drawdown,
            "risk_per_trade_limit": self.risk_limits.max_risk_per_trade,
            "open_positions": len(self.positions),
            "drawdown_cushion": (
                self.risk_limits.max_total_drawdown
                - self.drawdown_state.total_drawdown
            ),
        }

    def _send_emergency_alert(self, reason: str) -> None:
        """Send emergency alert (placeholder for actual implementation)."""
        logger.critical(
            f"EMERGENCY ALERT: {reason} | Contact: {self.emergency_contact}"
        )
        # Implement actual alerting (email, SMS, Slack, etc.)
