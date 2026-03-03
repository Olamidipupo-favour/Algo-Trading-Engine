"""
Strategy self-description capability.

Generates human-readable descriptions of current strategy state,
parameters, performance, and risk configuration.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import json

from src.utils.logger import Logger

logger = Logger(__name__)


class StrategyDescriptor:
    """
    Generate comprehensive strategy descriptions.

    Provides transparency into current strategy configuration and performance.
    """

    def __init__(self):
        """Initialize strategy descriptor."""
        self.current_strategy: Optional[Dict] = None
        self.performance_history: list = []

        logger.info("StrategyDescriptor initialized")

    def describe_current_strategy(
        self,
        strategy_params: Dict[str, Any],
        performance_metrics: Dict[str, float],
        regime_state: Optional[Dict] = None,
        risk_config: Optional[Dict] = None,
        governance_status: Optional[Dict] = None,
    ) -> str:
        """
        Generate comprehensive strategy description.

        Args:
            strategy_params: Current strategy parameters
            performance_metrics: Performance metrics
            regime_state: Current market regime information
            risk_config: Risk management configuration
            governance_status: Governance and lockdown status

        Returns:
            Human-readable strategy description
        """
        lines = [
            "╔" + "=" * 68 + "╗",
            "║" + " " * 20 + "STRATEGY DESCRIPTION" + " " * 28 + "║",
            "╠" + "=" * 68 + "╣",
            "",
        ]

        # Timestamp
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Entry Logic
        lines.append("═══ ENTRY LOGIC ═══")
        entry_desc = self._describe_entry_logic(strategy_params)
        lines.extend(entry_desc)
        lines.append("")

        # Exit Logic
        lines.append("═══ EXIT LOGIC ═══")
        exit_desc = self._describe_exit_logic(strategy_params)
        lines.extend(exit_desc)
        lines.append("")

        # Parameter Values
        lines.append("═══ PARAMETER VALUES ═══")
        param_desc = self._describe_parameters(strategy_params)
        lines.extend(param_desc)
        lines.append("")

        # Active Regime Filter
        if regime_state:
            lines.append("═══ MARKET REGIME ═══")
            regime_desc = self._describe_regime(regime_state)
            lines.extend(regime_desc)
            lines.append("")

        # Risk Model
        if risk_config:
            lines.append("═══ RISK MODEL ═══")
            risk_desc = self._describe_risk(risk_config)
            lines.extend(risk_desc)
            lines.append("")

        # Performance Summary
        lines.append("═══ HISTORICAL PERFORMANCE ═══")
        perf_desc = self._describe_performance(performance_metrics)
        lines.extend(perf_desc)
        lines.append("")

        # Governance Status
        if governance_status:
            lines.append("═══ GOVERNANCE STATUS ═══")
            gov_desc = self._describe_governance(governance_status)
            lines.extend(gov_desc)
            lines.append("")

        # Capital Allocation
        lines.append("═══ CAPITAL ALLOCATION ═══")
        capital_desc = self._describe_capital_allocation(
            strategy_params, performance_metrics
        )
        lines.extend(capital_desc)

        lines.append("")
        lines.append("╚" + "=" * 68 + "╝")

        description = "\n".join(lines)

        # Store current description
        self.current_strategy = {
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "parameters": strategy_params,
            "performance": performance_metrics,
        }

        return description

    def _describe_entry_logic(self, params: Dict) -> list:
        """Generate entry logic description."""
        lines = []

        # Check for common patterns
        if "ema_fast_period" in params and "ema_slow_period" in params:
            lines.append(
                f"  LONG: Enter when EMA({params['ema_fast_period']}) crosses above "
                f"EMA({params['ema_slow_period']})"
            )

        if "rsi_period" in params and "rsi_oversold" in params:
            lines.append(
                f"        AND RSI({params['rsi_period']}) < {params.get('rsi_oversold', 30)}"
            )

        if "adx_trend_threshold" in params:
            lines.append(
                f"        AND ADX > {params['adx_trend_threshold']} (trending market)"
            )

        lines.append("")

        if "ema_fast_period" in params and "ema_slow_period" in params:
            lines.append(
                f"  SHORT: Enter when EMA({params['ema_fast_period']}) crosses below "
                f"EMA({params['ema_slow_period']})"
            )

        if "rsi_period" in params and "rsi_overbought" in params:
            lines.append(
                f"         AND RSI({params['rsi_period']}) > {params.get('rsi_overbought', 70)}"
            )

        if not lines or len(lines) <= 1:
            lines = ["  Custom entry logic defined in strategy configuration"]

        return lines

    def _describe_exit_logic(self, params: Dict) -> list:
        """Generate exit logic description."""
        lines = []

        sl_mult = params.get("stop_loss_multiplier", 2.0)
        tp_mult = params.get("take_profit_multiplier", 3.0)

        lines.append(f"  Stop Loss: {sl_mult:.1f} × ATR (dynamic)")
        lines.append(f"  Take Profit: {tp_mult:.1f} × ATR (risk-reward ratio: 1:{tp_mult/sl_mult:.1f})")

        if params.get("trailing_stop_enabled", False):
            lines.append("  Trailing Stop: ENABLED")

        if params.get("regime_exit_enabled", True):
            lines.append("  Regime Change Exit: ENABLED (closes on unfavorable regime)")

        return lines

    def _describe_parameters(self, params: Dict) -> list:
        """Generate parameter description."""
        lines = []

        # Group parameters
        indicator_params = {}
        risk_params = {}
        other_params = {}

        for key, value in params.items():
            if any(
                x in key.lower()
                for x in ["ema", "rsi", "macd", "atr", "adx", "period", "threshold"]
            ):
                indicator_params[key] = value
            elif any(x in key.lower() for x in ["risk", "stop", "profit", "loss"]):
                risk_params[key] = value
            else:
                other_params[key] = value

        # Indicators
        if indicator_params:
            lines.append("  Indicators:")
            for key, value in sorted(indicator_params.items()):
                lines.append(f"    {key}: {value}")

        # Risk
        if risk_params:
            lines.append("  Risk Management:")
            for key, value in sorted(risk_params.items()):
                if isinstance(value, float):
                    lines.append(f"    {key}: {value:.2f}")
                else:
                    lines.append(f"    {key}: {value}")

        # Other
        if other_params:
            lines.append("  Other:")
            for key, value in sorted(other_params.items()):
                lines.append(f"    {key}: {value}")

        return lines

    def _describe_regime(self, regime_state: Dict) -> list:
        """Generate regime description."""
        lines = []

        regime = regime_state.get("current_regime", "unknown")
        confidence = regime_state.get("regime_confidence", 0.0)
        trend_strength = regime_state.get("trend_strength", 0.0)

        lines.append(f"  Current Regime: {regime.upper()}")
        lines.append(f"  Confidence: {confidence:.1%}")
        lines.append(f"  Trend Strength: {trend_strength:.2f}")

        if regime_state.get("active_strategy"):
            lines.append(f"  Active Strategy: {regime_state['active_strategy']}")
        elif regime_state.get("cash_mode", False):
            lines.append("  Mode: CASH (no qualified strategy for regime)")

        return lines

    def _describe_risk(self, risk_config: Dict) -> list:
        """Generate risk description."""
        lines = []

        lines.append(f"  Risk per Trade: {risk_config.get('risk_per_trade', 0.01):.1%}")
        lines.append(
            f"  Max Daily Drawdown: {risk_config.get('max_daily_drawdown', 0.03):.1%}"
        )
        lines.append(
            f"  Max Total Drawdown: {risk_config.get('max_total_drawdown', 0.10):.1%}"
        )
        lines.append(
            f"  Max Concurrent Positions: {risk_config.get('max_positions', 1)}"
        )

        if risk_config.get("is_shutdown", False):
            lines.append("")
            lines.append(f"  ⚠️  SYSTEM SHUTDOWN: {risk_config.get('shutdown_reason', 'unknown')}")

        return lines

    def _describe_performance(self, metrics: Dict) -> list:
        """Generate performance description."""
        lines = []

        lines.append(f"  Fitness Score: {metrics.get('fitness', 0.0):.3f}")
        lines.append(f"  Sharpe Ratio: {metrics.get('sharpe', 0.0):.2f}")
        lines.append(f"  Profit Factor: {metrics.get('profit_factor', 0.0):.2f}")
        lines.append(f"  Win Rate: {metrics.get('win_rate', 0.0):.1%}")
        lines.append(f"  Max Drawdown: {metrics.get('max_drawdown', 0.0):.1%}")
        lines.append(f"  Total Trades: {metrics.get('total_trades', 0)}")

        if "roi" in metrics:
            lines.append(f"  ROI: {metrics['roi']:.1%}")

        return lines

    def _describe_governance(self, gov_status: Dict) -> list:
        """Generate governance description."""
        lines = []

        # Lockdown status
        if gov_status.get("locked", False):
            lines.append("  Parameter Lock: ACTIVE")
            lines.append(
                f"    Next review: {gov_status.get('next_review_date', 'unknown')}"
            )
            lines.append(
                f"    Days until review: {gov_status.get('days_until_review', 0)}"
            )
        else:
            lines.append("  Parameter Lock: INACTIVE")

        # Probation status
        if gov_status.get("probation_mode", False):
            lines.append("  ⚠️  PROBATION MODE ACTIVE")
            lines.append(
                f"    Reason: {gov_status.get('probation_reason', 'performance_decay')}"
            )
            lines.append(f"    Risk multiplier: {gov_status.get('risk_multiplier', 0.5):.1f}")

        return lines

    def _describe_capital_allocation(
        self, params: Dict, metrics: Dict
    ) -> list:
        """Generate capital allocation description."""
        lines = []

        risk_per_trade = params.get("risk_per_trade", 0.01)
        max_positions = params.get("max_positions", 1)

        lines.append(f"  Risk per Trade: {risk_per_trade:.1%}")
        lines.append(f"  Max Simultaneous Positions: {max_positions}")
        lines.append(f"  Total Capital at Risk: {risk_per_trade * max_positions:.1%}")

        current_equity = metrics.get("current_equity", 0)
        if current_equity > 0:
            lines.append(f"  Current Equity: ${current_equity:,.2f}")

        return lines

    def export_description(self, filepath: str) -> None:
        """Export current description to file."""
        if not self.current_strategy:
            logger.warning("No current strategy to export")
            return

        try:
            with open(filepath, "w") as f:
                f.write(self.current_strategy["description"])
            logger.info(f"Strategy description exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export description: {e}")

    def export_json(self, filepath: str) -> None:
        """Export strategy data as JSON."""
        if not self.current_strategy:
            logger.warning("No current strategy to export")
            return

        try:
            with open(filepath, "w") as f:
                json.dump(self.current_strategy, f, indent=2, default=str)
            logger.info(f"Strategy data exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export JSON: {e}")
