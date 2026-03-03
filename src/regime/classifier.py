"""
Market regime classification module.

Classifies market conditions into distinct regimes to enable
regime-aware strategy selection and capital protection.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

from src.utils.logger import Logger

logger = Logger(__name__)


class RegimeType(Enum):
    """Market regime types."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRANSITIONAL = "transitional"
    UNKNOWN = "unknown"


@dataclass
class RegimeState:
    """Current market regime state."""

    primary_regime: RegimeType
    trend_strength: float  # 0-1
    volatility_percentile: float  # 0-100
    confidence: float  # 0-1
    regime_age: int  # bars since regime change
    previous_regime: Optional[RegimeType] = None


class RegimeClassifier:
    """
    Classify market regime using multiple technical indicators.

    Regime detection based on:
    - Trend: ADX or EMA slope strength
    - Volatility: ATR percentile ranking
    - Range: Low ADX + Bollinger width contraction
    """

    def __init__(
        self,
        adx_period: int = 14,
        adx_trend_threshold: float = 25.0,
        adx_range_threshold: float = 20.0,
        atr_period: int = 14,
        atr_lookback: int = 100,
        ema_fast: int = 20,
        ema_slow: int = 50,
        bb_period: int = 20,
        bb_std: float = 2.0,
    ):
        """
        Initialize regime classifier.

        Args:
            adx_period: ADX calculation period
            adx_trend_threshold: ADX threshold for trending market
            adx_range_threshold: ADX threshold below which is ranging
            atr_period: ATR calculation period
            atr_lookback: Lookback period for ATR percentile
            ema_fast: Fast EMA period
            ema_slow: Slow EMA period
            bb_period: Bollinger Bands period
            bb_std: Bollinger Bands standard deviation
        """
        self.adx_period = adx_period
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_range_threshold = adx_range_threshold
        self.atr_period = atr_period
        self.atr_lookback = atr_lookback
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.bb_period = bb_period
        self.bb_std = bb_std

        self.current_regime: Optional[RegimeState] = None
        self.regime_history: list = []

        logger.info(
            f"RegimeClassifier initialized: ADX_thresholds=({adx_range_threshold}, "
            f"{adx_trend_threshold}), ATR_lookback={atr_lookback}"
        )

    def classify(self, data: pd.DataFrame) -> RegimeState:
        """
        Classify current market regime.

        Args:
            data: OHLCV DataFrame with sufficient history

        Returns:
            RegimeState object
        """
        if len(data) < max(self.atr_lookback, self.bb_period, self.ema_slow):
            logger.warning(f"Insufficient data for regime classification: {len(data)}")
            return RegimeState(
                primary_regime=RegimeType.UNKNOWN,
                trend_strength=0.0,
                volatility_percentile=50.0,
                confidence=0.0,
                regime_age=0,
            )

        # Calculate indicators
        adx_value, dm_plus, dm_minus = self._calculate_adx(data)
        atr_percentile = self._calculate_atr_percentile(data)
        ema_slope = self._calculate_ema_slope(data)
        bb_width = self._calculate_bb_width(data)

        # Determine primary regime
        regime_type, confidence = self._determine_regime(
            adx_value, dm_plus, dm_minus, atr_percentile, ema_slope, bb_width
        )

        # Trend strength calculation
        trend_strength = min(adx_value / 50.0, 1.0)  # Normalize ADX to 0-1

        # Calculate regime age
        regime_age = self._calculate_regime_age(regime_type)

        regime_state = RegimeState(
            primary_regime=regime_type,
            trend_strength=trend_strength,
            volatility_percentile=atr_percentile,
            confidence=confidence,
            regime_age=regime_age,
            previous_regime=(
                self.current_regime.primary_regime if self.current_regime else None
            ),
        )

        # Update tracking
        self.current_regime = regime_state
        self.regime_history.append(
            {"regime": regime_type.value, "timestamp": data.index[-1]}
        )

        logger.debug(
            f"Regime: {regime_type.value} | Confidence: {confidence:.2f} | "
            f"Trend: {trend_strength:.2f} | Vol%: {atr_percentile:.1f}"
        )

        return regime_state

    def _calculate_adx(self, data: pd.DataFrame) -> Tuple[float, float, float]:
        """Calculate ADX and directional movement."""
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)

        # Directional movement
        dm_plus = np.maximum(high[1:] - high[:-1], 0)
        dm_minus = np.maximum(low[:-1] - low[1:], 0)

        # Smooth with Wilder's smoothing
        atr = self._wilder_smooth(tr, self.adx_period)
        smoothed_dm_plus = self._wilder_smooth(dm_plus, self.adx_period)
        smoothed_dm_minus = self._wilder_smooth(dm_minus, self.adx_period)

        # Directional indicators
        di_plus = 100 * smoothed_dm_plus / atr
        di_minus = 100 * smoothed_dm_minus / atr

        # ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
        adx = self._wilder_smooth(dx, self.adx_period)

        return adx[-1], di_plus[-1], di_minus[-1]

    def _wilder_smooth(self, data: np.ndarray, period: int) -> np.ndarray:
        """Wilder's smoothing method."""
        smoothed = np.zeros_like(data)
        smoothed[period - 1] = np.mean(data[: period])

        for i in range(period, len(data)):
            smoothed[i] = (smoothed[i - 1] * (period - 1) + data[i]) / period

        return smoothed

    def _calculate_atr_percentile(self, data: pd.DataFrame) -> float:
        """Calculate ATR percentile rank."""
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values

        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)

        # ATR
        atr = self._wilder_smooth(tr, self.atr_period)

        # Percentile rank over lookback period
        recent_atr = atr[-self.atr_lookback :]
        current_atr = atr[-1]

        percentile = (
            (recent_atr < current_atr).sum() / len(recent_atr) * 100
        )

        return percentile

    def _calculate_ema_slope(self, data: pd.DataFrame) -> float:
        """Calculate EMA slope strength."""
        close = data['close'].values

        # Calculate EMAs
        ema_fast = self._calculate_ema(close, self.ema_fast)
        ema_slow = self._calculate_ema(close, self.ema_slow)

        # Slope of fast EMA (normalized)
        slope = (ema_fast[-1] - ema_fast[-5]) / (ema_fast[-5] + 1e-10)

        # Separation between EMAs (trend strength)
        separation = (ema_fast[-1] - ema_slow[-1]) / (ema_slow[-1] + 1e-10)

        return slope + separation  # Combined metric

    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        multiplier = 2.0 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]

        for i in range(1, len(data)):
            ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1]

        return ema

    def _calculate_bb_width(self, data: pd.DataFrame) -> float:
        """Calculate Bollinger Band width (volatility measure)."""
        close = data['close'].values[-self.bb_period :]

        sma = np.mean(close)
        std = np.std(close)

        upper = sma + (self.bb_std * std)
        lower = sma - (self.bb_std * std)

        width = (upper - lower) / sma

        return width

    def _determine_regime(
        self,
        adx: float,
        dm_plus: float,
        dm_minus: float,
        atr_percentile: float,
        ema_slope: float,
        bb_width: float,
    ) -> Tuple[RegimeType, float]:
        """
        Determine regime type and confidence.

        Returns:
            Tuple of (RegimeType, confidence_score)
        """
        # High confidence thresholds
        strong_trend = adx > self.adx_trend_threshold
        ranging = adx < self.adx_range_threshold
        high_volatility = atr_percentile > 75
        low_volatility = atr_percentile < 25
        narrow_bb = bb_width < 0.02

        confidence = 0.5  # Base confidence

        # Trending up
        if strong_trend and dm_plus > dm_minus and ema_slope > 0:
            confidence = min(0.9, 0.5 + (adx - self.adx_trend_threshold) / 50.0)
            return RegimeType.TRENDING_UP, confidence

        # Trending down
        if strong_trend and dm_minus > dm_plus and ema_slope < 0:
            confidence = min(0.9, 0.5 + (adx - self.adx_trend_threshold) / 50.0)
            return RegimeType.TRENDING_DOWN, confidence

        # Ranging (low ADX + narrow BB)
        if ranging and narrow_bb:
            confidence = 0.7
            return RegimeType.RANGING, confidence

        # High volatility
        if high_volatility and not strong_trend:
            confidence = 0.6
            return RegimeType.HIGH_VOLATILITY, confidence

        # Low volatility
        if low_volatility:
            confidence = 0.6
            return RegimeType.LOW_VOLATILITY, confidence

        # Transitional (between regimes)
        confidence = 0.3
        return RegimeType.TRANSITIONAL, confidence

    def _calculate_regime_age(self, current_regime: RegimeType) -> int:
        """Calculate how long current regime has persisted."""
        if not self.current_regime:
            return 0

        if self.current_regime.primary_regime == current_regime:
            return self.current_regime.regime_age + 1
        else:
            return 0

    def get_regime_statistics(self) -> Dict:
        """Get statistics on regime history."""
        if not self.regime_history:
            return {}

        regime_counts = {}
        for entry in self.regime_history:
            regime = entry["regime"]
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        total = len(self.regime_history)
        regime_percentages = {k: v / total * 100 for k, v in regime_counts.items()}

        return {
            "total_classifications": total,
            "regime_distribution": regime_percentages,
            "current_regime": (
                self.current_regime.primary_regime.value
                if self.current_regime
                else "unknown"
            ),
            "current_age": (
                self.current_regime.regime_age if self.current_regime else 0
            ),
        }
