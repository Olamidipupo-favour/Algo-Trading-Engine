"""
Regime classification and strategy selection module.

Provides robust regime detection and regime-aware strategy activation
to prevent strategy decay during unfavorable market conditions.
"""

from .classifier import RegimeClassifier
from .manager import RegimeManager
from .analyzer import RegimePerformanceAnalyzer

__all__ = [
    "RegimeClassifier",
    "RegimeManager",
    "RegimePerformanceAnalyzer",
]
