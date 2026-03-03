"""
Rolling walk-forward parameter optimization engine.

Implements systematic parameter search with robust validation and
overfitting prevention through walk-forward analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
import itertools
from pathlib import Path

from src.utils.logger import Logger
from .fitness import FitnessCalculator, TradeMetrics
from .validators import StabilityValidator, OverfitValidator

logger = Logger(__name__)


@dataclass
class OptimizationResult:
    """Container for optimization results."""

    best_parameters: Dict[str, Any]
    fitness_score: float
    train_score: float
    validation_score: float
    test_score: float
    metrics: Dict[str, float]
    stability_result: Dict[str, Any]
    overfit_result: Dict[str, Any]
    timestamp: str
    optimization_method: str
    total_combinations_tested: int
    is_stable: bool
    is_overfit: bool


class ParameterOptimizer:
    """
    Rolling walk-forward parameter optimization engine.

    Optimizes strategy parameters using grid search or random search
    with strict validation on train/validation/test windows.
    """

    def __init__(
        self,
        train_days: int = 90,
        validation_days: int = 30,
        test_days: int = 14,
        reoptimization_interval_days: int = 30,
        min_trades: int = 30,
        max_drawdown: float = 0.15,
        output_dir: str = "optimization_results",
    ):
        """
        Initialize parameter optimizer.

        Args:
            train_days: Days in training window
            validation_days: Days in validation window
            test_days: Days in test window
            reoptimization_interval_days: Days between re-optimizations
            min_trades: Minimum trades required for valid backtest
            max_drawdown: Maximum allowed drawdown
            output_dir: Directory for saving results
        """
        self.train_days = train_days
        self.validation_days = validation_days
        self.test_days = test_days
        self.reoptimization_interval = reoptimization_interval_days

        # Initialize validators
        self.fitness_calculator = FitnessCalculator(
            min_trades=min_trades, max_drawdown=max_drawdown
        )
        self.stability_validator = StabilityValidator()
        self.overfit_validator = OverfitValidator()

        # Results storage
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.optimization_history: List[OptimizationResult] = []
        self.current_best_params: Optional[Dict[str, Any]] = None

        logger.info(
            f"ParameterOptimizer initialized: "
            f"train={train_days}d, val={validation_days}d, "
            f"test={test_days}d, reopt_interval={reoptimization_interval_days}d"
        )

    def define_parameter_space(self) -> Dict[str, List[Any]]:
        """
        Define the parameter space for optimization.

        Returns:
            Dictionary mapping parameter names to lists of candidate values
        """
        parameter_space = {
            "ema_fast_period": [8, 10, 12, 15, 20],
            "ema_slow_period": [26, 30, 34, 40, 50],
            "rsi_period": [8, 10, 12, 14, 16, 20],
            "rsi_overbought": [65, 70, 75, 80],
            "rsi_oversold": [20, 25, 30, 35],
            "stop_loss_multiplier": [1.0, 1.2, 1.5, 2.0, 2.5],
            "take_profit_multiplier": [1.5, 2.0, 2.5, 3.0, 4.0],
        }

        total_combinations = np.prod(
            [len(values) for values in parameter_space.values()]
        )
        logger.info(
            f"Parameter space defined: {len(parameter_space)} parameters, "
            f"{total_combinations:,} total combinations"
        )

        return parameter_space

    def grid_search(
        self,
        data: pd.DataFrame,
        backtest_function: Callable[[pd.DataFrame, Dict], Tuple],
        parameter_space: Optional[Dict[str, List[Any]]] = None,
        max_combinations: int = 1000,
    ) -> OptimizationResult:
        """
        Perform grid search optimization.

        Args:
            data: Historical price data
            backtest_function: Function(data, params) -> (equity_curve, trades)
            parameter_space: Custom parameter space (uses default if None)
            max_combinations: Maximum combinations to test (prevents explosion)

        Returns:
            OptimizationResult with best parameters and validation results
        """
        if parameter_space is None:
            parameter_space = self.define_parameter_space()

        # Generate all combinations
        param_names = list(parameter_space.keys())
        param_values = [parameter_space[name] for name in param_names]
        all_combinations = list(itertools.product(*param_values))

        # Limit search space if too large
        if len(all_combinations) > max_combinations:
            logger.warning(
                f"Grid space too large ({len(all_combinations)}), "
                f"sampling {max_combinations} combinations"
            )
            indices = np.random.choice(
                len(all_combinations), max_combinations, replace=False
            )
            all_combinations = [all_combinations[i] for i in indices]

        logger.info(f"Starting grid search over {len(all_combinations)} combinations")

        # Split data into train/val/test
        train_data, val_data, test_data = self._split_data(data)

        best_fitness = -999.0
        best_params = None
        best_results = None

        # Test each combination
        for idx, combination in enumerate(all_combinations):
            params = dict(zip(param_names, combination))

            # Validate parameter constraints
            if not self._validate_parameter_constraints(params):
                continue

            # Evaluate on train/val/test
            result = self._evaluate_parameter_set(
                params, train_data, val_data, test_data, backtest_function
            )

            if result and result.fitness_score > best_fitness:
                best_fitness = result.fitness_score
                best_params = params
                best_results = result

            if (idx + 1) % 100 == 0:
                logger.info(
                    f"Progress: {idx + 1}/{len(all_combinations)} | "
                    f"Best fitness: {best_fitness:.3f}"
                )

        if best_results is None:
            logger.error("No valid parameter sets found")
            raise ValueError("Optimization failed: no valid solutions")

        logger.info(
            f"Grid search complete. Best fitness: {best_fitness:.3f} | "
            f"Parameters: {best_params}"
        )

        # Save results
        self._save_results(best_results)
        self.current_best_params = best_params
        self.optimization_history.append(best_results)

        return best_results

    def random_search(
        self,
        data: pd.DataFrame,
        backtest_function: Callable[[pd.DataFrame, Dict], Tuple],
        parameter_space: Optional[Dict[str, List[Any]]] = None,
        n_iterations: int = 500,
        seed: Optional[int] = None,
    ) -> OptimizationResult:
        """
        Perform randomized parameter search.

        Args:
            data: Historical price data
            backtest_function: Function(data, params) -> (equity_curve, trades)
            parameter_space: Custom parameter space
            n_iterations: Number of random combinations to test
            seed: Random seed for reproducibility

        Returns:
            OptimizationResult with best parameters
        """
        if seed is not None:
            np.random.seed(seed)

        if parameter_space is None:
            parameter_space = self.define_parameter_space()

        logger.info(f"Starting random search with {n_iterations} iterations")

        # Split data
        train_data, val_data, test_data = self._split_data(data)

        best_fitness = -999.0
        best_params = None
        best_results = None

        for iteration in range(n_iterations):
            # Sample random parameters
            params = {
                name: np.random.choice(values)
                for name, values in parameter_space.items()
            }

            # Validate constraints
            if not self._validate_parameter_constraints(params):
                continue

            # Evaluate
            result = self._evaluate_parameter_set(
                params, train_data, val_data, test_data, backtest_function
            )

            if result and result.fitness_score > best_fitness:
                best_fitness = result.fitness_score
                best_params = params
                best_results = result

            if (iteration + 1) % 50 == 0:
                logger.info(
                    f"Progress: {iteration + 1}/{n_iterations} | "
                    f"Best fitness: {best_fitness:.3f}"
                )

        if best_results is None:
            logger.error("No valid parameter sets found")
            raise ValueError("Optimization failed: no valid solutions")

        logger.info(
            f"Random search complete. Best fitness: {best_fitness:.3f} | "
            f"Parameters: {best_params}"
        )

        # Save results
        self._save_results(best_results)
        self.current_best_params = best_params
        self.optimization_history.append(best_results)

        return best_results

    def _split_data(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/validation/test sets."""
        total_days = self.train_days + self.validation_days + self.test_days

        if len(data) < total_days:
            logger.warning(
                f"Insufficient data: {len(data)} < {total_days} required"
            )

        train_end = self.train_days
        val_end = train_end + self.validation_days

        train_data = data.iloc[:train_end].copy()
        val_data = data.iloc[train_end:val_end].copy()
        test_data = data.iloc[val_end : val_end + self.test_days].copy()

        logger.debug(
            f"Data split: train={len(train_data)}, "
            f"val={len(val_data)}, test={len(test_data)}"
        )

        return train_data, val_data, test_data

    def _evaluate_parameter_set(
        self,
        params: Dict[str, Any],
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        backtest_function: Callable,
    ) -> Optional[OptimizationResult]:
        """Evaluate a parameter set on train/val/test data."""
        try:
            # Train evaluation
            train_equity, train_trades = backtest_function(train_data, params)
            train_metrics = self.fitness_calculator.calculate_metrics(
                train_equity, train_trades
            )

            # Count conditions (complexity penalty)
            num_conditions = self._count_conditions(params)

            train_fitness, _ = self.fitness_calculator.calculate_fitness(
                train_metrics, num_conditions
            )

            # Validation evaluation
            val_equity, val_trades = backtest_function(val_data, params)
            val_metrics = self.fitness_calculator.calculate_metrics(
                val_equity, val_trades
            )
            val_fitness, _ = self.fitness_calculator.calculate_fitness(
                val_metrics, num_conditions
            )

            # Test evaluation
            test_equity, test_trades = backtest_function(test_data, params)
            test_metrics = self.fitness_calculator.calculate_metrics(
                test_equity, test_trades
            )
            test_fitness, _ = self.fitness_calculator.calculate_fitness(
                test_metrics, num_conditions
            )

            # Check for overfitting
            overfit_result = self.overfit_validator.detect_overfitting(
                train_fitness, val_fitness, test_fitness
            )

            if overfit_result.is_overfit:
                logger.debug(f"Rejected (overfit): {params}")
                return None

            # Stability test
            def eval_func(p):
                eq, tr = backtest_function(val_data, p)
                m = self.fitness_calculator.calculate_metrics(eq, tr)
                f, _ = self.fitness_calculator.calculate_fitness(m, num_conditions)
                return f

            stability_result = self.stability_validator.test_stability(
                params, eval_func, val_fitness
            )

            if not stability_result.is_stable:
                logger.debug(f"Rejected (unstable): {params}")
                return None

            # Use validation fitness as primary score
            return OptimizationResult(
                best_parameters=params,
                fitness_score=val_fitness,
                train_score=train_fitness,
                validation_score=val_fitness,
                test_score=test_fitness,
                metrics={
                    "sharpe": val_metrics.sharpe_ratio,
                    "profit_factor": val_metrics.profit_factor,
                    "win_rate": val_metrics.win_rate,
                    "max_drawdown": val_metrics.max_drawdown,
                    "total_trades": val_metrics.total_trades,
                },
                stability_result=asdict(stability_result),
                overfit_result=asdict(overfit_result),
                timestamp=datetime.now().isoformat(),
                optimization_method="walk_forward",
                total_combinations_tested=1,
                is_stable=stability_result.is_stable,
                is_overfit=overfit_result.is_overfit,
            )

        except Exception as e:
            logger.warning(f"Evaluation failed for {params}: {e}")
            return None

    def _validate_parameter_constraints(self, params: Dict[str, Any]) -> bool:
        """Validate parameter constraints."""
        # EMA fast must be less than slow
        if params.get("ema_fast_period", 0) >= params.get("ema_slow_period", 999):
            return False

        # RSI oversold must be less than overbought
        if params.get("rsi_oversold", 0) >= params.get("rsi_overbought", 100):
            return False

        # Take profit must be greater than stop loss
        if params.get("take_profit_multiplier", 0) <= params.get(
            "stop_loss_multiplier", 0
        ):
            return False

        return True

    def _count_conditions(self, params: Dict[str, Any]) -> int:
        """Count active conditions (complexity metric)."""
        # Simple heuristic: each parameter adds complexity
        return len(params)

    def _save_results(self, result: OptimizationResult) -> None:
        """Save optimization results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"optimization_{timestamp}.json"

        with open(filepath, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)

        logger.info(f"Results saved to {filepath}")

    def generate_summary(self, result: OptimizationResult) -> str:
        """Generate human-readable optimization summary."""
        params = result.best_parameters

        summary = f"""
╔══════════════════════════════════════════════════════════════════╗
║           PARAMETER OPTIMIZATION SUMMARY                          ║
╠══════════════════════════════════════════════════════════════════╣
║ Optimized Strategy Configuration:
║   - EMA: {params.get('ema_fast_period', 'N/A')}/{params.get('ema_slow_period', 'N/A')}
║   - RSI: Period={params.get('rsi_period', 'N/A')}, Levels={params.get('rsi_oversold', 'N/A')}/{params.get('rsi_overbought', 'N/A')}
║   - SL/TP: {params.get('stop_loss_multiplier', 'N/A')} ATR / {params.get('take_profit_multiplier', 'N/A')} ATR
║
║ Performance Metrics:
║   - Fitness Score:   {result.fitness_score:.3f}
║   - Sharpe Ratio:    {result.metrics['sharpe']:.2f}
║   - Profit Factor:   {result.metrics['profit_factor']:.2f}
║   - Win Rate:        {result.metrics['win_rate']:.1%}
║   - Max Drawdown:    {result.metrics['max_drawdown']:.1%}
║   - Total Trades:    {result.metrics['total_trades']}
║
║ Validation Results:
║   - Train Score:     {result.train_score:.3f}
║   - Val Score:       {result.validation_score:.3f}
║   - Test Score:      {result.test_score:.3f}
║   - Stability:       {'STABLE' if result.is_stable else 'UNSTABLE'}
║   - Overfitting:     {'YES' if result.is_overfit else 'NO'}
║
║ Timestamp: {result.timestamp}
╚══════════════════════════════════════════════════════════════════╝
        """

        return summary.strip()
