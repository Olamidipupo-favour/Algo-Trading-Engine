"""
Example Research Mode Workflow

Demonstrates parameter optimization and genetic strategy evolution
in research mode with full validation pipeline.

IMPORTANT: This is RESEARCH ONLY. No live trading.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import optimization components
from src.optimization.optimizer import ParameterOptimizer
from src.optimization.fitness import FitnessCalculator

# Import genetic components
from src.genetic.engine import GeneticEngine
from src.genetic.chromosome import ChromosomeFactory

# Import regime components
from src.regime.classifier import RegimeClassifier
from src.regime.manager import RegimeManager

# Import governance
from src.governance.lockdown import ParameterLockdown
from src.governance.probation import ProbationManager

# Import risk management
from src.core.enhanced_risk_manager import EnhancedRiskManager

# Import descriptor
from src.core.strategy_descriptor import StrategyDescriptor

from src.utils.logger import Logger

logger = Logger(__name__)


def example_backtest_function(data: pd.DataFrame, params: dict):
    """
    Placeholder backtest function.

    In real implementation, this would:
    1. Apply strategy with params to data
    2. Generate trades
    3. Calculate equity curve

    Args:
        data: OHLCV data
        params: Strategy parameters

    Returns:
        Tuple of (equity_curve, trades_list)
    """
    # Simulate equity curve
    equity = pd.Series(
        np.linspace(10000, 10000 * (1 + np.random.uniform(-0.1, 0.2)), len(data)),
        index=data.index,
    )

    # Simulate trades
    trades = [
        {
            "pnl": np.random.uniform(-100, 200),
            "entry_time": data.index[i],
            "exit_time": data.index[i + 10] if i + 10 < len(data) else data.index[-1],
        }
        for i in range(0, len(data), 20)
    ]

    return equity, trades


def research_workflow_optimization():
    """
    WORKFLOW 1: Parameter Optimization

    Steps:
    1. Load historical data
    2. Define parameter space
    3. Run walk-forward optimization
    4. Validate results
    5. Lock best parameters
    """
    logger.info("=" * 70)
    logger.info("RESEARCH WORKFLOW 1: PARAMETER OPTIMIZATION")
    logger.info("=" * 70)

    # Step 1: Load data (placeholder - use real data in production)
    logger.info("\n[1/5] Loading historical data...")
    dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="H")
    data = pd.DataFrame(
        {
            "open": np.random.uniform(100, 105, len(dates)),
            "high": np.random.uniform(105, 110, len(dates)),
            "low": np.random.uniform(95, 100, len(dates)),
            "close": np.random.uniform(100, 105, len(dates)),
            "volume": np.random.uniform(1000, 5000, len(dates)),
        },
        index=dates,
    )
    logger.info(f"Loaded {len(data)} bars of data")

    # Step 2: Initialize optimizer
    logger.info("\n[2/5] Initializing optimizer...")
    optimizer = ParameterOptimizer(
        train_days=90,
        validation_days=30,
        test_days=14,
        reoptimization_interval_days=30,
    )

    # Step 3: Run optimization
    logger.info("\n[3/5] Running parameter optimization...")
    logger.info("This may take several minutes...")

    result = optimizer.random_search(
        data=data,
        backtest_function=example_backtest_function,
        n_iterations=100,  # Reduced for example
        seed=42,
    )

    # Step 4: Display results
    logger.info("\n[4/5] Optimization complete!")
    summary = optimizer.generate_summary(result)
    print("\n" + summary)

    # Step 5: Lock parameters
    logger.info("\n[5/5] Locking optimized parameters...")
    lockdown = ParameterLockdown(lock_duration_days=30)
    lockdown.lock_parameters(
        parameters=result.best_parameters,
        reason="post_optimization_lock",
        locked_by="research_workflow",
    )

    logger.info("Parameters locked for 30 days")
    logger.info(f"Next review: {lockdown.current_lock.next_review_date}")

    return result


def research_workflow_genetic():
    """
    WORKFLOW 2: Genetic Strategy Evolution

    Steps:
    1. Initialize genetic engine (RESEARCH MODE ONLY)
    2. Create initial population
    3. Evolve strategies
    4. Validate top strategies
    5. Select best for further testing
    """
    logger.info("=" * 70)
    logger.info("RESEARCH WORKFLOW 2: GENETIC STRATEGY EVOLUTION")
    logger.info("=" * 70)

    # Step 1: Load data
    logger.info("\n[1/5] Loading historical data...")
    dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="H")
    data = pd.DataFrame(
        {
            "open": np.random.uniform(100, 105, len(dates)),
            "high": np.random.uniform(105, 110, len(dates)),
            "low": np.random.uniform(95, 100, len(dates)),
            "close": np.random.uniform(100, 105, len(dates)),
            "volume": np.random.uniform(1000, 5000, len(dates)),
        },
        index=dates,
    )

    # Step 2: Initialize genetic engine
    logger.info("\n[2/5] Initializing genetic engine (RESEARCH MODE)...")

    try:
        engine = GeneticEngine(
            population_size=20,  # Reduced for example
            generations=10,  # Reduced for example
            mutation_rate=0.3,
            crossover_rate=0.7,
            research_mode_only=True,  # MANDATORY
        )
    except ValueError as e:
        logger.error(f"Safety violation: {e}")
        return None

    # Step 3: Define backtest function for chromosomes
    def chromosome_backtest(data, chromosome):
        """Backtest a strategy chromosome."""
        # In real implementation, decode chromosome and run strategy
        params = {
            "risk_per_trade": chromosome.risk_per_trade,
            "complexity": chromosome.get_complexity(),
        }
        return example_backtest_function(data, params)

    # Step 4: Evolve strategies
    logger.info("\n[3/5] Evolving strategies...")
    logger.info("This will take several minutes...")

    best_strategy = engine.evolve(
        data=data, backtest_function=chromosome_backtest
    )

    # Step 5: Display results
    logger.info("\n[4/5] Evolution complete!")
    if best_strategy:
        print("\n" + best_strategy.describe())
    else:
        logger.error("No viable strategy evolved")

    # Step 6: Review hall of fame
    logger.info("\n[5/5] Hall of Fame (Top 5 Strategies):")
    hall_of_fame = engine.population_manager.get_hall_of_fame()
    for idx, strategy in enumerate(hall_of_fame, 1):
        logger.info(
            f"  {idx}. {strategy.chromosome_id} | Fitness: {strategy.fitness:.3f}"
        )

    logger.info(
        "\n⚠️  REMINDER: These strategies are RESEARCH MODE ONLY"
    )
    logger.info("Required before live deployment:")
    logger.info("  1. Extended out-of-sample testing")
    logger.info("  2. Cross-regime validation")
    logger.info("  3. Paper trading for 30+ days")
    logger.info("  4. Manual review and approval")

    return best_strategy


def research_workflow_regime_analysis():
    """
    WORKFLOW 3: Regime Analysis and Strategy Selection

    Steps:
    1. Classify market regimes
    2. Evaluate strategy performance by regime
    3. Register strategies with regime manager
    4. Test regime-aware selection
    """
    logger.info("=" * 70)
    logger.info("RESEARCH WORKFLOW 3: REGIME ANALYSIS")
    logger.info("=" * 70)

    # Step 1: Load data
    logger.info("\n[1/4] Loading historical data...")
    dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="H")
    data = pd.DataFrame(
        {
            "high": np.random.uniform(105, 110, len(dates)),
            "low": np.random.uniform(95, 100, len(dates)),
            "close": np.cumsum(np.random.randn(len(dates)) * 0.5) + 100,
            "volume": np.random.uniform(1000, 5000, len(dates)),
        },
        index=dates,
    )

    # Step 2: Classify regimes
    logger.info("\n[2/4] Classifying market regimes...")
    classifier = RegimeClassifier()

    regime_history = []
    for i in range(200, len(data), 100):
        regime_state = classifier.classify(data.iloc[: i + 1])
        regime_history.append(
            {"index": i, "regime": regime_state.primary_regime.value}
        )
        logger.info(
            f"  Bar {i}: {regime_state.primary_regime.value} "
            f"(confidence: {regime_state.confidence:.2f})"
        )

    # Step 3: Get regime statistics
    logger.info("\n[3/4] Regime distribution:")
    stats = classifier.get_regime_statistics()
    for regime, pct in stats.get("regime_distribution", {}).items():
        logger.info(f"  {regime}: {pct:.1f}%")

    # Step 4: Register strategies with regime manager
    logger.info("\n[4/4] Testing regime-aware strategy selection...")
    regime_manager = RegimeManager()

    # Register example strategies with regime-specific performance
    regime_manager.register_strategy(
        "trend_follower",
        {
            "trending_up": {"sharpe": 1.5, "pf": 2.0, "trades": 50},
            "trending_down": {"sharpe": 1.3, "pf": 1.8, "trades": 45},
            "ranging": {"sharpe": 0.2, "pf": 0.9, "trades": 30},
        },
    )

    regime_manager.register_strategy(
        "mean_reversion",
        {
            "ranging": {"sharpe": 1.4, "pf": 1.9, "trades": 60},
            "low_volatility": {"sharpe": 1.2, "pf": 1.6, "trades": 40},
            "trending_up": {"sharpe": -0.5, "pf": 0.7, "trades": 25},
        },
    )

    # Test selection
    selected = regime_manager.select_strategy(data)
    logger.info(f"\nSelected strategy for current regime: {selected}")

    report = regime_manager.get_status_report()
    logger.info(f"Current regime: {report['current_regime']}")
    logger.info(f"Regime confidence: {report['regime_confidence']:.2f}")
    logger.info(f"Cash mode: {report['cash_mode']}")

    return regime_manager


def research_workflow_complete_system():
    """
    WORKFLOW 4: Complete System Integration

    Demonstrates full integration of all components.
    """
    logger.info("=" * 70)
    logger.info("RESEARCH WORKFLOW 4: COMPLETE SYSTEM INTEGRATION")
    logger.info("=" * 70)

    # Initialize all components
    logger.info("\n[1/7] Initializing system components...")

    optimizer = ParameterOptimizer()
    fitness_calc = FitnessCalculator()
    regime_manager = RegimeManager()
    risk_manager = EnhancedRiskManager(initial_capital=10000.0)
    lockdown = ParameterLockdown()
    probation = ProbationManager()
    descriptor = StrategyDescriptor()

    logger.info("All components initialized")

    # Generate strategy description
    logger.info("\n[2/7] Generating strategy description...")

    strategy_params = {
        "ema_fast_period": 12,
        "ema_slow_period": 26,
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "stop_loss_multiplier": 2.0,
        "take_profit_multiplier": 3.0,
        "risk_per_trade": 0.01,
        "max_positions": 1,
    }

    performance_metrics = {
        "fitness": 1.25,
        "sharpe": 1.42,
        "profit_factor": 1.9,
        "win_rate": 0.58,
        "max_drawdown": 0.061,
        "total_trades": 145,
        "current_equity": 10850.0,
    }

    regime_state = {
        "current_regime": "trending_up",
        "regime_confidence": 0.85,
        "trend_strength": 0.72,
        "active_strategy": "trend_follower",
    }

    risk_config = {
        "risk_per_trade": 0.01,
        "max_daily_drawdown": 0.03,
        "max_total_drawdown": 0.10,
        "max_positions": 1,
        "is_shutdown": False,
    }

    description = descriptor.describe_current_strategy(
        strategy_params=strategy_params,
        performance_metrics=performance_metrics,
        regime_state=regime_state,
        risk_config=risk_config,
        governance_status=lockdown.get_status(),
    )

    print("\n" + description)

    # Export description
    logger.info("\n[3/7] Exporting strategy description...")
    descriptor.export_description("strategy_description.txt")
    descriptor.export_json("strategy_data.json")

    # Test risk validation
    logger.info("\n[4/7] Testing risk validation...")

    test_position = {
        "symbol": "EURUSD",
        "size": 1.0,
        "entry_price": 1.1000,
        "stop_loss": 1.0950,
    }

    approved, reason, adjusted = risk_manager.validate_new_position(test_position)
    logger.info(f"Position validation: {approved} | Reason: {reason}")

    # Simulate equity update
    logger.info("\n[5/7] Testing drawdown monitoring...")
    risk_manager.update_equity(10850.0)
    risk_manager.update_equity(10500.0)  # Small drawdown

    risk_status = risk_manager.get_risk_status()
    logger.info(f"Total drawdown: {risk_status['total_drawdown']:.2%}")
    logger.info(f"Daily drawdown: {risk_status['daily_drawdown']:.2%}")

    # Test probation
    logger.info("\n[6/7] Testing probation system...")

    # Simulate trades
    for i in range(10):
        trade = {"pnl": np.random.uniform(-100, 150)}
        current_equity = 10000 + sum(
            [t["pnl"] for t in [trade] * (i + 1)]
        )

        actions = probation.update(trade, current_equity)

        if actions["enter_probation"]:
            logger.warning("Probation mode activated!")
        if actions["reduce_risk"]:
            logger.warning(f"Risk reduced to {actions['risk_multiplier']:.0%}")

    probation_status = probation.get_status()
    logger.info(f"Probation mode: {probation_status['probation_mode']}")
    logger.info(f"Recent Sharpe: {probation_status['recent_sharpe']:.2f}")

    # Test lockdown
    logger.info("\n[7/7] Testing parameter lockdown...")

    can_modify, lock_reason = lockdown.can_modify_parameters()
    logger.info(f"Can modify parameters: {can_modify}")
    logger.info(f"Reason: {lock_reason}")

    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE SYSTEM INTEGRATION TEST FINISHED")
    logger.info("=" * 70)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RESEARCH MODE WORKFLOWS")
    print("=" * 70)
    print("\nAvailable workflows:")
    print("  1. Parameter Optimization")
    print("  2. Genetic Strategy Evolution")
    print("  3. Regime Analysis")
    print("  4. Complete System Integration")
    print("\n" + "=" * 70 + "\n")

    # Run workflows
    try:
        # Workflow 1: Optimization
        #result = research_workflow_optimization()

        # Workflow 2: Genetic (commented due to runtime)
        # best_strategy = research_workflow_genetic()

        # Workflow 3: Regime Analysis
        # regime_manager = research_workflow_regime_analysis()

        # Workflow 4: Complete System
        research_workflow_complete_system()

    except Exception as e:
        logger.error(f"Workflow error: {e}")
        import traceback

        traceback.print_exc()
