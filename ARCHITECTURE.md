# Professional Trading System Architecture

## Executive Summary

This document describes the architecture of a production-grade algorithmic trading platform with institutional-level risk management, parameter optimization, genetic strategy evolution, and comprehensive governance systems.

**Core Philosophy**: Robustness, statistical validity, regime awareness, and capital preservation over raw profit optimization.

---

## Architecture Tree

```
trading/
├── src/
│   ├── optimization/           # Parameter Optimization Module
│   │   ├── __init__.py
│   │   ├── fitness.py          # Composite fitness calculation
│   │   ├── validators.py       # Stability & overfit detection
│   │   └── optimizer.py        # Rolling walk-forward optimizer
│   │
│   ├── genetic/                # Genetic Algorithm Module (RESEARCH ONLY)
│   │   ├── __init__.py
│   │   ├── chromosome.py       # Strategy chromosome representation
│   │   ├── operators.py        # Mutation & crossover operators
│   │   ├── population.py       # Population management with elitism
│   │   └── engine.py           # Main genetic evolution engine
│   │
│   ├── regime/                 # Regime Classification Module
│   │   ├── __init__.py
│   │   ├── classifier.py       # Market regime detection (ADX, ATR, etc.)
│   │   ├── manager.py          # Regime-aware strategy selection
│   │   └── analyzer.py         # Performance analysis by regime
│   │
│   ├── governance/             # Anti-Interference Governance
│   │   ├── __init__.py
│   │   ├── lockdown.py         # Parameter lock system
│   │   ├── audit.py            # Audit logging
│   │   ├── reports.py          # Automated reporting
│   │   └── probation.py        # Flat period survival logic
│   │
│   ├── core/                   # Core Trading Engine
│   │   ├── data_provider.py    # Historical & live data
│   │   ├── indicator_engine.py # Technical indicators
│   │   ├── regime_detector.py  # [Existing] Regime detection
│   │   ├── trading_logic.py    # Signal generation
│   │   ├── trade_execution.py  # Order execution
│   │   ├── enhanced_risk_manager.py  # NEW: Institutional risk management
│   │   └── strategy_descriptor.py    # NEW: Self-description capability
│   │
│   ├── strategies/             # Strategy Implementations
│   │   ├── base_strategy.py
│   │   ├── trend_strategy.py
│   │   └── range_strategy.py
│   │
│   ├── backtesting/            # Backtesting Engine
│   │   ├── engine.py
│   │   ├── metrics.py
│   │   └── reports.py
│   │
│   └── utils/                  # Utilities
│       ├── config.py
│       ├── logger.py
│       └── alerts.py
│
├── examples/                   # Example Workflows
│   ├── research_workflow.py    # Research mode examples
│   └── live_workflow.py        # Live trading examples
│
├── config/                     # Configuration
│   └── main.yaml
│
├── optimization_results/       # Optimization outputs
├── genetic_research/           # Genetic evolution outputs
├── governance/                 # Governance configs & logs
├── backtest_results/          # Backtest results
└── logs/                       # System logs
```

---

## Module Descriptions

### 1. Optimization Module (`src/optimization/`)

**Purpose**: Rolling walk-forward parameter optimization with overfitting prevention.

#### Key Components:

**`fitness.py` - FitnessCalculator**
- Composite fitness function balancing multiple metrics
- Formula: `fitness = 0.4×Sharpe + 0.3×ProfitFactor + 0.2×WinRate - 0.3×MaxDD - 0.05×Complexity`
- Enforces minimum trade count, maximum drawdown, minimum Sharpe
- Provides detailed component breakdown for transparency

**`validators.py` - StabilityValidator & OverfitValidator**
- **StabilityValidator**: Perturbs parameters by ±10% to test robustness
- **OverfitValidator**: Compares train/val/test performance ratios
- Cross-regime validation support
- Rejects solutions with excessive degradation

**`optimizer.py` - ParameterOptimizer**
- Rolling walk-forward optimization (train/val/test windows)
- Supports grid search and randomized search
- Configurable re-optimization intervals
- Persists results with full validation history
- Generates human-readable summaries

**Parameters Optimized**:
- EMA fast/slow periods
- RSI period and thresholds
- Stop-loss and take-profit multipliers

**Safety Features**:
- Minimum trade count threshold
- Maximum allowed drawdown
- Parameter stability testing
- Overfit detection and rejection

---

### 2. Genetic Algorithm Module (`src/genetic/`)

**⚠️ CRITICAL: RESEARCH MODE ONLY - NO LIVE TRADING**

**Purpose**: Evolve trading strategies through genetic algorithms with strict safeguards.

#### Key Components:

**`chromosome.py` - StrategyChromosome**
- Complete strategy encoding:
  - Long/short entry conditions (multiple indicator rules)
  - Logical operators (AND/OR)
  - Exit logic (SL/TP parameters)
  - Risk parameters
- Human-readable descriptions
- Complexity calculations
- Validation checks

**`operators.py` - MutationOperator & CrossoverOperator**
- **Mutation Types**:
  - Parameter mutation (±10%)
  - Indicator replacement
  - Add/remove conditions
  - Logical operator flip
  - Exit logic modification
- **Crossover**:
  - Single-point and uniform crossover
  - Condition blending
  - Parameter averaging

**`population.py` - PopulationManager**
- Maintains population of 50+ strategies
- Elitism: Preserves top 10%
- Tournament selection
- Diversity monitoring and maintenance
- Hall of Fame tracking (top 5 all-time)

**`engine.py` - GeneticEngine**
- Multi-generation evolution (50-100 generations)
- Configurable mutation/crossover rates
- Multi-period validation (train/val/test)
- Out-of-sample evaluation mandatory
- Research mode enforcement (cannot be disabled)

**Safety Features**:
- **Research mode lock**: Cannot be used in live trading
- Multi-period validation
- Complexity penalty
- Cross-regime testing
- Minimum trade requirements
- Hall of fame for rollback capability

---

### 3. Regime Classification Module (`src/regime/`)

**Purpose**: Classify market regimes and activate strategies only in favorable conditions.

#### Key Components:

**`classifier.py` - RegimeClassifier**
- **Regime Types**:
  - Trending Up/Down (ADX > 25, directional movement)
  - Ranging (ADX < 20, narrow Bollinger Bands)
  - High/Low Volatility (ATR percentile)
  - Transitional (between regimes)
- **Indicators Used**:
  - ADX (14) for trend strength
  - ATR percentile for volatility
  - EMA slope for trend direction
  - Bollinger Band width for range identification
- Confidence scoring
- Regime age tracking

**`manager.py` - RegimeManager**
- Registers strategies with regime-specific performance
- Activates strategies only in historically profitable regimes
- Automatic fallback to cash when no strategy qualifies
- Minimum qualification criteria:
  - Minimum trades per regime (default: 10)
  - Minimum Sharpe ratio (default: 0.5)
  - Minimum profit factor (default: 1.2)
- Closes positions on regime transitions

**Regime-Aware Logic**:
```
IF current_regime == "trending_up":
    IF strategy.regime_stats["trending_up"].sharpe >= 0.5:
        ACTIVATE strategy
    ELSE:
        ENTER cash_mode
```

---

### 4. Governance Module (`src/governance/`)

**Purpose**: Prevent emotional tampering and enforce systematic discipline.

#### Key Components:

**`lockdown.py` - ParameterLockdown**
- Locks parameters for fixed review intervals (default: 30 days)
- Scheduled evolution windows (weekly)
- Override requests logged and require manual approval
- Configuration persisted to disk
- Prevents ad-hoc parameter changes

**`probation.py` - ProbationManager**
- **Equity Stagnation Detection**:
  - Monitors equity growth over rolling window
  - Triggers risk reduction if growth < 1%
- **Performance Decay Detection**:
  - Compares rolling Sharpe to baseline
  - Triggers probation if decay > 30%
- **Consecutive Loss Tracking**:
  - Monitors streak of losing trades
  - Reduces risk after 5 consecutive losses
- **Probation Actions**:
  - Risk reduction (50% multiplier)
  - Re-optimization trigger
  - Strategy suspension (if negative Sharpe in probation)

**`audit.py` & `reports.py`**
- Complete audit trail of all changes
- Automated weekly performance reports
- Override request logging

**Anti-Interference Principles**:
1. Parameters locked for scheduled intervals
2. Changes only during evolution windows
3. All modifications logged with timestamps
4. Manual override requires explicit approval
5. Emergency overrides trigger alerts

---

### 5. Enhanced Risk Management (`src/core/enhanced_risk_manager.py`)

**NON-NEGOTIABLE CAPITAL PROTECTION RULES**

#### Hard Limits (Cannot be Overridden):

```python
MAX_RISK_PER_TRADE = 1%      # Maximum capital risk per trade
MAX_DAILY_DRAWDOWN = 3%      # Maximum daily loss
MAX_TOTAL_DRAWDOWN = 10%     # Maximum total drawdown
```

#### Risk Manager Features:

**Position Validation**:
- Validates all proposed trades against risk limits
- Automatically adjusts position sizes if needed
- Rejects trades that violate concentration limits
- Checks leverage limits

**Drawdown Monitoring**:
- Tracks peak equity continuously
- Calculates real-time total and daily drawdown
- Resets daily tracking at market open
- Automatic actions on thresholds:
  - 80% of daily limit → Risk reduction
  - Daily limit → Close all positions
  - 80% of total limit → Close all positions
  - Total limit → **EMERGENCY SHUTDOWN**

**Emergency Shutdown Protocol**:
1. Stop all trading immediately
2. Close all open positions
3. Disconnect from broker API
4. Generate shutdown report
5. Send emergency notifications
6. Log comprehensive state

**Circuit Breakers**:
- Automatic position closure on daily drawdown breach
- System shutdown on total drawdown breach
- No manual override possible
- Broker disconnection on critical breach

---

### 6. Self-Description Capability (`src/core/strategy_descriptor.py`)

**Purpose**: Provide complete transparency into strategy state.

**`StrategyDescriptor.describe_current_strategy()`**

Generates human-readable report including:
- Entry logic (plain English)
- Exit logic with specific parameters
- All parameter values (organized by type)
- Current market regime and confidence
- Risk model and limits
- Historical performance summary
- Governance status (locked/probation)
- Current capital allocation

**Export Formats**:
- Plain text (human-readable)
- JSON (machine-readable)

Example output:
```
═══ ENTRY LOGIC ═══
  LONG: Enter when EMA(12) crosses above EMA(26)
        AND RSI(14) < 30
        AND ADX > 25 (trending market)

═══ EXIT LOGIC ═══
  Stop Loss: 2.0 × ATR (dynamic)
  Take Profit: 3.0 × ATR (risk-reward ratio: 1:1.5)

═══ PERFORMANCE ═══
  Fitness Score: 1.420
  Sharpe Ratio: 1.42
  Profit Factor: 1.9
  Win Rate: 58.0%
  Max Drawdown: 6.1%
```

---

## Data Flow Architecture

### Research Mode Flow:

```
Historical Data
    ↓
Parameter Optimizer / Genetic Engine
    ↓
Walk-Forward Validation (Train/Val/Test)
    ↓
Stability Testing (±10% perturbation)
    ↓
Overfit Detection (train/val/test ratios)
    ↓
Cross-Regime Validation
    ↓
Parameter Lockdown
    ↓
Extended Out-of-Sample Testing
    ↓
Manual Review & Approval
    ↓
Paper Trading (30+ days)
    ↓
Live Deployment (with locked parameters)
```

### Live Trading Flow:

```
Market Data Stream
    ↓
Regime Classification (ADX, ATR, BB)
    ↓
Regime-Aware Strategy Selection
    ↓
Strategy Qualification Check
    ↓
Signal Generation (if qualified)
    ↓
Pre-Trade Safety Checks:
  - System not shutdown?
  - Parameters locked?
  - Regime confidence sufficient?
  - Not in probation suspension?
  - Drawdown limits OK?
    ↓
Risk Manager Position Validation:
  - Risk per trade ≤ 1%?
  - Daily drawdown < 3%?
  - Total drawdown < 10%?
  - Position concentration OK?
    ↓
Position Size Adjustment (if needed)
    ↓
Trade Execution
    ↓
Position Monitoring:
  - SL/TP tracking
  - Regime change monitoring
  - Drawdown monitoring
    ↓
Position Closure
    ↓
Probation System Update:
  - Check equity stagnation
  - Check performance decay
  - Check consecutive losses
  - Trigger actions if needed
    ↓
Risk Manager Update:
  - Update equity
  - Check drawdown limits
  - Trigger circuit breakers if needed
```

---

## Key Algorithms

### 1. Composite Fitness Function

```python
def calculate_fitness(metrics, complexity):
    # Component calculations
    sharpe_component = 0.4 × min(sharpe_ratio, 3.0)
    pf_component = 0.3 × min(profit_factor, 3.0)
    wr_component = 0.2 × win_rate
    dd_penalty = -0.3 × max_drawdown
    complexity_penalty = -0.05 × (complexity / 10.0)

    # Composite fitness
    fitness = (sharpe_component + pf_component + wr_component
               + dd_penalty + complexity_penalty)

    # Disqualify if:
    if trades < min_trades: return -999
    if max_drawdown > limit: return -999
    if sharpe < min_sharpe: return -999

    return fitness
```

### 2. Stability Validation

```python
def test_stability(params, eval_func, original_score):
    perturbed_scores = []

    for param in params:
        for direction in [-1, 1]:
            for trial in range(5):
                # Perturb ±10%
                perturbed = param × (1 + 0.1 × direction)
                score = eval_func(perturbed)
                perturbed_scores.append(score)

    max_degradation = (original_score - min(perturbed_scores)) / original_score

    is_stable = (max_degradation ≤ 0.15)  # Max 15% degradation
    return is_stable
```

### 3. Overfit Detection

```python
def detect_overfitting(train_score, val_score, test_score):
    train_val_ratio = train_score / val_score
    val_test_ratio = val_score / test_score

    # Thresholds
    is_overfit = (train_val_ratio > 1.20 or val_test_ratio > 1.15)

    severity = "SEVERE" if ratios > 1.5 else "MODERATE" if ratios > 1.3 else "MILD"

    return is_overfit, severity
```

### 4. Regime Classification

```python
def classify_regime(data):
    # Calculate indicators
    adx = calculate_adx(data, period=14)
    atr_percentile = calculate_atr_percentile(data, lookback=100)
    ema_slope = calculate_ema_slope(data)
    bb_width = calculate_bb_width(data)

    # Regime determination
    if adx > 25 and dm_plus > dm_minus and ema_slope > 0:
        return TRENDING_UP, confidence
    elif adx > 25 and dm_minus > dm_plus and ema_slope < 0:
        return TRENDING_DOWN, confidence
    elif adx < 20 and bb_width < 0.02:
        return RANGING, confidence
    elif atr_percentile > 75:
        return HIGH_VOLATILITY, confidence
    else:
        return TRANSITIONAL, low_confidence
```

### 5. Risk Position Validation

```python
def validate_position(proposed_position):
    # Calculate position risk
    risk_per_unit = abs(entry_price - stop_loss)
    total_risk = risk_per_unit × size
    risk_pct = total_risk / current_equity

    # Check limits
    if risk_pct > MAX_RISK_PER_TRADE:
        # Adjust size
        adjusted_size = (equity × MAX_RISK) / risk_per_unit
        return APPROVED, "adjusted", adjusted_position

    # Check drawdown
    if daily_dd >= MAX_DAILY_DD:
        return REJECTED, "daily_drawdown_exceeded", None

    if total_dd >= MAX_TOTAL_DD:
        initiate_shutdown("total_drawdown")
        return REJECTED, "shutdown_initiated", None

    return APPROVED, "passed", proposed_position
```

---

## Type Hints & Testing

All modules use comprehensive type hints:

```python
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass

def optimize(
    data: pd.DataFrame,
    backtest_func: Callable[[pd.DataFrame, Dict], Tuple],
    parameter_space: Optional[Dict[str, List[Any]]] = None
) -> OptimizationResult:
    ...
```

**Unit Testing Requirements**:
- All fitness calculations
- Parameter validation logic
- Regime classification
- Risk limit enforcement
- Genetic operators (mutation/crossover)
- Circuit breakers

---

## Configuration Management

**Central Configuration**: `config/main.yaml`

Example structure:
```yaml
optimization:
  train_days: 90
  validation_days: 30
  test_days: 14
  reopt_interval: 30
  min_trades: 30
  max_drawdown: 0.15

genetic:
  population_size: 50
  generations: 100
  mutation_rate: 0.3
  crossover_rate: 0.7
  research_mode_only: true  # MANDATORY

risk_management:
  max_risk_per_trade: 0.01
  max_daily_drawdown: 0.03
  max_total_drawdown: 0.10
  emergency_contact: "your_email@example.com"

governance:
  lock_duration_days: 30
  evolution_windows: ["weekly"]

regime:
  min_trades_per_regime: 10
  min_regime_sharpe: 0.5
  confidence_threshold: 0.6
```

---

## Logging & Monitoring

**Comprehensive Logging**:
- Module-level loggers (`src.utils.logger`)
- Separate log files per component
- Critical events logged with CRITICAL level
- Audit trail for all parameter changes
- Trade execution logging
- Performance metrics logging

**Log Levels**:
- CRITICAL: Shutdowns, breaches, emergencies
- ERROR: Validation failures, rejections
- WARNING: Probation, risk reductions
- INFO: Normal operations, trade executions
- DEBUG: Detailed state information

---

## Deterministic Seeds

All randomized components support deterministic seeds:

```python
optimizer.random_search(seed=42)
engine.evolve(seed=42)
np.random.seed(42)
random.seed(42)
```

This ensures:
- Reproducible backtests
- Reproducible optimization results
- Reproducible genetic evolution
- Debugging capability

---

## Performance Considerations

**Optimization**:
- Grid search: O(n^k) where k = number of parameters
- Random search: O(n) iterations, more efficient
- Walk-forward: Data split overhead acceptable

**Genetic Evolution**:
- Population size: 50-100 strategies
- Generations: 50-100 iterations
- Evaluation time: Dominated by backtesting
- Parallelization opportunity: Population evaluation

**Live Trading**:
- Real-time regime classification: < 100ms
- Risk validation: < 10ms
- Position calculation: < 5ms
- Total latency: < 200ms (acceptable for H1/H4 strategies)

---

## Security & Privacy

- API keys stored in environment variables
- No credentials in code or logs
- Audit trail for all actions
- Emergency contact notifications
- Broker disconnection on critical events

---

## Extensibility

**Adding New Indicators**:
1. Implement in `indicator_engine.py`
2. Add to gene pool in `chromosome.py`
3. Update parameter space in `optimizer.py`

**Adding New Regimes**:
1. Add enum to `RegimeType`
2. Implement detection logic in `classifier.py`
3. Register strategies with new regime in `manager.py`

**Adding New Risk Rules**:
1. Add limits to `RiskLimits` dataclass
2. Implement validation in `enhanced_risk_manager.py`
3. Update shutdown protocol if needed

---

## Production Deployment Checklist

- [ ] Complete walk-forward optimization
- [ ] Validate on out-of-sample data
- [ ] Cross-regime testing passed
- [ ] Parameter stability verified
- [ ] Overfit detection passed
- [ ] Paper trading 30+ days
- [ ] Manual review completed
- [ ] Parameters locked
- [ ] Risk limits configured
- [ ] Emergency contacts configured
- [ ] Monitoring systems active
- [ ] Backup/failover tested
- [ ] Broker API connection tested
- [ ] Shutdown protocol tested

---

## Maintenance & Updates

**Scheduled Reviews**:
- Weekly: Automated performance reports
- Monthly: Parameter lock expires, re-optimization window
- Quarterly: Strategy review, regime distribution analysis
- Annually: Full system audit

**Re-Optimization Triggers**:
- Scheduled interval (30 days default)
- Probation system triggers
- Performance decay detected
- Regime distribution change
- Manual override (requires approval)

---

This architecture prioritizes **robustness over returns**, **validation over optimization**, and **capital preservation over profit maximization** — the hallmarks of institutional-grade systematic trading.
