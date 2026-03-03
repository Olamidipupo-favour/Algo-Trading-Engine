"""
Example Live Trading Workflow

Demonstrates production deployment with full safety systems.

CRITICAL SAFETY REQUIREMENTS:
- Parameters must be locked
- Risk limits enforced
- Regime monitoring active
- Probation system enabled
- Emergency shutdown configured
"""

import pandas as pd
import numpy as np
from datetime import datetime
import time

# Import components
from src.regime.manager import RegimeManager
from src.core.enhanced_risk_manager import EnhancedRiskManager, RiskLimits
from src.governance.lockdown import ParameterLockdown
from src.governance.probation import ProbationManager
from src.core.strategy_descriptor import StrategyDescriptor

from src.utils.logger import Logger

logger = Logger(__name__)


class LiveTradingSystem:
    """
    Production trading system with full safety controls.

    NON-NEGOTIABLE REQUIREMENTS:
    - Locked parameters (no live mutation)
    - Hard risk limits enforced
    - Regime-aware execution
    - Automatic circuit breakers
    - Comprehensive logging
    """

    def __init__(self, initial_capital: float, strategy_params: dict):
        """
        Initialize live trading system.

        Args:
            initial_capital: Starting capital
            strategy_params: Pre-validated, locked strategy parameters
        """
        logger.critical("=" * 70)
        logger.critical("INITIALIZING LIVE TRADING SYSTEM")
        logger.critical("=" * 70)

        # Verify parameters are locked
        self.lockdown = ParameterLockdown()
        if not self.lockdown.is_locked():
            logger.warning("Parameters not locked. Locking now...")
            self.lockdown.lock_parameters(
                parameters=strategy_params,
                reason="live_deployment",
                locked_by="live_system",
            )

        can_modify, reason = self.lockdown.can_modify_parameters()
        if can_modify:
            raise RuntimeError(
                "SAFETY VIOLATION: Parameters must be locked for live trading"
            )

        logger.info(f"✓ Parameters locked until {self.lockdown.current_lock.next_review_date}")

        # Initialize risk manager with hard limits
        self.risk_manager = EnhancedRiskManager(
            initial_capital=initial_capital,
            risk_limits=RiskLimits(
                max_risk_per_trade=0.01,
                max_daily_drawdown=0.03,
                max_total_drawdown=0.10,
            ),
            emergency_contact="your_email@example.com",
        )
        logger.info("✓ Risk manager initialized with hard limits")

        # Initialize regime manager
        self.regime_manager = RegimeManager()
        logger.info("✓ Regime manager initialized")

        # Initialize probation manager
        self.probation = ProbationManager()
        logger.info("✓ Probation system initialized")

        # Initialize descriptor
        self.descriptor = StrategyDescriptor()
        logger.info("✓ Strategy descriptor initialized")

        # Store parameters
        self.strategy_params = strategy_params
        self.current_equity = initial_capital
        self.is_running = True

        logger.critical("✓ LIVE SYSTEM READY")
        logger.critical("=" * 70)

    def pre_trade_checks(self, market_data: pd.DataFrame) -> bool:
        """
        Run pre-trade safety checks.

        Args:
            market_data: Current market data

        Returns:
            True if safe to trade
        """
        logger.debug("Running pre-trade safety checks...")

        # Check 1: System not shutdown
        if self.risk_manager.is_shutdown:
            logger.critical(
                f"SYSTEM SHUTDOWN: {self.risk_manager.shutdown_reason}"
            )
            return False

        # Check 2: Parameters still locked
        can_modify, _ = self.lockdown.can_modify_parameters()
        if can_modify:
            logger.error("Parameter lock compromised!")
            return False

        # Check 3: Regime confidence sufficient
        selected_strategy = self.regime_manager.select_strategy(market_data)
        if selected_strategy is None:
            logger.warning("No qualified strategy for current regime - staying in cash")
            return False

        # Check 4: Not in probation with suspension
        if self.probation.suspended:
            logger.error("Strategy suspended by probation system")
            return False

        # Check 5: Drawdown limits not breached
        risk_actions = self.risk_manager.update_equity(self.current_equity)
        if risk_actions["shutdown_system"]:
            logger.critical("SHUTDOWN TRIGGERED BY DRAWDOWN")
            self.emergency_shutdown("drawdown_limit_exceeded")
            return False

        logger.debug("✓ All pre-trade checks passed")
        return True

    def execute_trade(self, signal: dict, market_data: pd.DataFrame) -> bool:
        """
        Execute trade with full validation.

        Args:
            signal: Trading signal
            market_data: Current market data

        Returns:
            True if trade executed successfully
        """
        # Pre-trade checks
        if not self.pre_trade_checks(market_data):
            return False

        # Construct proposed position
        proposed_position = {
            "symbol": signal["symbol"],
            "size": signal["size"],
            "entry_price": signal["price"],
            "stop_loss": signal["stop_loss"],
        }

        # Validate with risk manager
        approved, reason, adjusted_position = self.risk_manager.validate_new_position(
            proposed_position
        )

        if not approved:
            logger.warning(f"Trade rejected: {reason}")
            return False

        # Use adjusted position if provided
        final_position = adjusted_position or proposed_position

        # Execute trade (placeholder - integrate with actual broker API)
        logger.info(
            f"TRADE EXECUTED: {final_position['symbol']} | "
            f"Size: {final_position['size']:.2f} | "
            f"Entry: {final_position['entry_price']:.5f} | "
            f"SL: {final_position['stop_loss']:.5f}"
        )

        # Register with risk manager
        self.risk_manager.add_position(
            final_position["symbol"], final_position
        )

        return True

    def update_position(self, symbol: str, pnl: float) -> None:
        """
        Update closed position and check probation.

        Args:
            symbol: Symbol that was closed
            pnl: Profit/loss of closed position
        """
        # Remove from risk manager
        self.risk_manager.remove_position(symbol)

        # Update equity
        self.current_equity += pnl

        # Update probation system
        trade = {
            "symbol": symbol,
            "pnl": pnl,
            "timestamp": datetime.now(),
        }

        actions = self.probation.update(trade, self.current_equity)

        # Handle probation actions
        if actions["enter_probation"]:
            logger.warning(
                f"ENTERING PROBATION MODE: {actions['reason']}"
            )

        if actions["reduce_risk"]:
            logger.warning(
                f"RISK REDUCTION: multiplier={actions['risk_multiplier']:.0%}"
            )

        if actions["trigger_reoptimization"]:
            logger.critical("RE-OPTIMIZATION RECOMMENDED")
            # Trigger notification to review system

        if actions["suspend_strategy"]:
            logger.critical("STRATEGY SUSPENDED")
            self.emergency_shutdown("probation_suspension")

        # Update risk manager
        risk_actions = self.risk_manager.update_equity(self.current_equity)

        if risk_actions["close_all_positions"]:
            logger.critical("CLOSING ALL POSITIONS")
            # Close all open positions

        if risk_actions["shutdown_system"]:
            logger.critical("EMERGENCY SHUTDOWN")
            self.emergency_shutdown("risk_limit_breach")

    def emergency_shutdown(self, reason: str) -> None:
        """
        Execute emergency shutdown protocol.

        Args:
            reason: Reason for shutdown
        """
        logger.critical("\n" + "=" * 70)
        logger.critical("EMERGENCY SHUTDOWN PROTOCOL INITIATED")
        logger.critical(f"Reason: {reason}")
        logger.critical(f"Time: {datetime.now().isoformat()}")
        logger.critical("=" * 70)

        # Stop trading
        self.is_running = False

        # Close all positions (integrate with actual broker API)
        logger.critical("Closing all open positions...")

        # Disconnect from broker
        logger.critical("Disconnecting from broker API...")

        # Send notifications
        logger.critical(
            f"Emergency notification sent to {self.risk_manager.emergency_contact}"
        )

        # Generate final report
        self.generate_shutdown_report(reason)

        logger.critical("SHUTDOWN COMPLETE")
        logger.critical("=" * 70)

    def generate_shutdown_report(self, reason: str) -> None:
        """Generate comprehensive shutdown report."""
        report_lines = [
            "=" * 70,
            "EMERGENCY SHUTDOWN REPORT",
            "=" * 70,
            "",
            f"Shutdown Time: {datetime.now().isoformat()}",
            f"Reason: {reason}",
            "",
            "RISK STATUS:",
        ]

        risk_status = self.risk_manager.get_risk_status()
        for key, value in risk_status.items():
            if isinstance(value, float):
                report_lines.append(f"  {key}: {value:.4f}")
            else:
                report_lines.append(f"  {key}: {value}")

        report_lines.extend(
            [
                "",
                "PROBATION STATUS:",
            ]
        )

        probation_status = self.probation.get_status()
        for key, value in probation_status.items():
            if isinstance(value, float):
                report_lines.append(f"  {key}: {value:.4f}")
            else:
                report_lines.append(f"  {key}: {value}")

        report_lines.append("\n" + "=" * 70)

        report = "\n".join(report_lines)
        print(report)

        # Save to file
        filename = f"shutdown_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, "w") as f:
            f.write(report)

        logger.critical(f"Shutdown report saved: {filename}")

    def generate_daily_report(self) -> str:
        """Generate daily performance and status report."""
        logger.info("Generating daily report...")

        # Get current statuses
        risk_status = self.risk_manager.get_risk_status()
        probation_status = self.probation.get_status()
        regime_status = self.regime_manager.get_status_report()
        lockdown_status = self.lockdown.get_status()

        # Generate description
        performance_metrics = {
            "current_equity": self.current_equity,
            "total_drawdown": risk_status["total_drawdown"],
            "daily_drawdown": risk_status["daily_drawdown"],
        }

        description = self.descriptor.describe_current_strategy(
            strategy_params=self.strategy_params,
            performance_metrics=performance_metrics,
            regime_state=regime_status,
            risk_config=risk_status,
            governance_status=lockdown_status,
        )

        return description


def example_live_deployment():
    """
    Example live trading deployment.

    This demonstrates the proper workflow for live trading.
    """
    logger.info("\n" + "=" * 70)
    logger.info("LIVE TRADING DEPLOYMENT EXAMPLE")
    logger.info("=" * 70)

    # Step 1: Define validated strategy parameters
    # (These should come from optimization/validation)
    strategy_params = {
        "ema_fast_period": 12,
        "ema_slow_period": 26,
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "stop_loss_multiplier": 2.0,
        "take_profit_multiplier": 3.0,
        "risk_per_trade": 0.01,
    }

    # Step 2: Initialize live system
    live_system = LiveTradingSystem(
        initial_capital=10000.0, strategy_params=strategy_params
    )

    # Step 3: Simulate trading loop
    logger.info("\n" + "=" * 70)
    logger.info("STARTING TRADING LOOP")
    logger.info("=" * 70)

    # Simulate market data
    dates = pd.date_range(start=datetime.now(), periods=100, freq="5min")
    market_data = pd.DataFrame(
        {
            "high": np.random.uniform(105, 110, len(dates)),
            "low": np.random.uniform(95, 100, len(dates)),
            "close": np.cumsum(np.random.randn(len(dates)) * 0.5) + 100,
            "volume": np.random.uniform(1000, 5000, len(dates)),
        },
        index=dates,
    )

    # Trading loop simulation
    for i in range(10):
        if not live_system.is_running:
            logger.critical("System stopped")
            break

        logger.info(f"\n--- Trading Cycle {i + 1} ---")

        # Simulate signal
        if np.random.random() > 0.7:  # 30% chance of signal
            signal = {
                "symbol": "EURUSD",
                "size": 0.1,
                "price": 1.1000 + np.random.uniform(-0.0050, 0.0050),
                "stop_loss": 1.0950,
            }

            success = live_system.execute_trade(signal, market_data)

            if success:
                # Simulate position close after some time
                pnl = np.random.uniform(-100, 200)
                live_system.update_position("EURUSD", pnl)

                logger.info(f"Position closed with P&L: {pnl:.2f}")

        # Sleep to simulate time passing
        time.sleep(0.1)

    # Step 4: Generate daily report
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING DAILY REPORT")
    logger.info("=" * 70)

    daily_report = live_system.generate_daily_report()
    print("\n" + daily_report)

    logger.info("\n" + "=" * 70)
    logger.info("LIVE TRADING EXAMPLE COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        example_live_deployment()
    except Exception as e:
        logger.critical(f"System error: {e}")
        import traceback

        traceback.print_exc()
