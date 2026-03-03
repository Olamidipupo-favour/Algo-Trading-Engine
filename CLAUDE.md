# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an algorithmic trading engine implementing a dynamic regime-based trading strategy for Forex markets (primarily EUR/USD). The system automatically switches between trending and ranging market strategies based on real-time market regime detection using ADX and Bollinger Bandwidth indicators.

The engine supports both backtesting with historical data and live trading via MetaTrader 5 (MT5) through a FastAPI bridge running on Windows.

## Key Commands

### Running the System

**Backtest mode:**
```bash
python main.py
```
Ensure `live: false` in `config/main.yaml`

**Live trading mode:**
```bash
python main.py
```
Ensure `live: true` in `config/main.yaml` and MT5 FastAPI server is running

### Environment Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+

### MT5 Windows Server (for live trading)

The MT5 bridge must run on Windows with MetaTrader 5 installed:

```bash
cd mt5_windows_server
pip install -r windows_requirements.txt
python windows_server.py
```

Configure `trading_api.config.base_url` and `trading_api.config.access_key` in `config/main.yaml` to match the server settings.

### Testing

```bash
pytest
```

### Viewing Logs

Logs are written to `logs/` directory with separate files per module:

```bash
tail -f logs/src.core.trading_logic.log
```

## Architecture

### Core Data Flow

1. **Data Provider** (`src/core/data_provider.py`) - Fetches historical or live market data
   - Supports CSV files (`data_source: "csv"`), live streaming (`data_source: "live"`), or historical data from trading API (`data_source: "historical_live"`)
   - Yields synchronized multi-timeframe data snapshots as data stream

2. **Indicator Engine** (`src/core/indicator_engine.py`) - Calculates technical indicators
   - Supports: EMA, MACD, ATR, ADX, Stochastic, Bollinger Bandwidth, Support/Resistance, Pivot Points
   - Configured per timeframe in `config/main.yaml`

3. **Regime Detector** (`src/core/regime_detector.py`) - Identifies market regime
   - TRENDING: ADX > 25 + high Bollinger Bandwidth
   - RANGING: ADX < 20 + low Bollinger Bandwidth
   - NEUTRAL: ADX between 20-25

4. **Trading Logic Engine** (`src/core/trading_logic.py`) - Generates entry/exit signals
   - Different rule sets for each regime defined in `strategy.trading_logic` section
   - Uses condition parser to evaluate complex multi-indicator conditions

5. **Trade Executor** (`src/core/trade_execution.py`) - Executes trades and manages positions
   - Handles both backtesting and live execution
   - Manages stop-loss, take-profit, position sizing, risk management
   - Tracks performance metrics, drawdown limits, consecutive losses

6. **Trading APIs** (`src/core/trading_apis/`) - Interfaces to trading platforms
   - `mt5_fastapi.py`: FastAPI bridge to MetaTrader 5 (primary)
   - Stubs for Binance, Bybit, KuCoin (future expansion)

### Key Design Patterns

**Async/await throughout**: The entire system uses asyncio for non-blocking I/O operations.

**Data streaming model**: `DataProvider.stream_data()` yields data snapshots incrementally, allowing the main loop to process each tick/bar sequentially in both backtest and live modes.

**Regime-based strategy switching**: The system maintains a single unified engine but changes behavior dynamically based on detected market regime.

**Configuration-driven**: All strategy parameters, indicators, entry/exit conditions, and risk rules are defined in `config/main.yaml` - no code changes needed for strategy adjustments.

## Configuration (`config/main.yaml`)

The configuration file uses a structured YAML format with these main sections:

### Important Settings

**`live`**: `true` for live trading, `false` for backtesting

**`trading_api`**: Trading platform configuration
- `type`: API type (`mt5_fastapi`, etc.)
- `config.base_url`: API endpoint
- `config.access_key`: Authentication key

**`backtesting`**: Backtest parameters
- `data_source`: `"csv"`, `"live"`, or `"historical_live"`
- `start`/`end`: Date range in ISO format
- `initial_balance`: Starting capital
- `output_folder`: Where results are saved

**`strategy.indicators`**: Indicator definitions by timeframe
- Each indicator has: `name`, `type`, `params`
- The system automatically determines required data window size based on indicator periods

**`strategy.regime_logic.rules`**: Conditions for regime detection
- Uses condition syntax: `"H1;regime_adx;adx > 25"`

**`strategy.trading_logic`**: Entry/exit rules per regime
- Separate rules for TRENDING, RANGING, NEUTRAL regimes
- Entry conditions use multi-indicator logic
- Exit methods: ATR-based SL, risk-reward ratio TP, or fixed levels

**`risk_management`**: Risk controls
- `risk_per_trade`: Position sizing (% of equity)
- `minimum_rrr`: Minimum reward-to-risk ratio
- `max_concurrent_trades`: Position limits
- `drawdown_protection`: Daily/weekly/monthly loss limits
- `time_filters.sessions`: Trading hour restrictions

### Condition Syntax

The trading logic uses a specialized condition parser that supports:

- Indicator comparisons: `"H4;trend_macd;histogram > 0"`
- Cross-indicator comparisons: `"H1;trend_ema_fast;value > H1;trend_ema_slow;value"`
- Percentile thresholds: `"H1;volatility_bbw;width;value > H1;volatility_bbw;width;percentile_75"`
- Price conditions: `"price.close > price.open"`
- Default fallback: `"default"`

## Important Implementation Details

### Data Count Determination

The function `determine_counts_from_indicators()` in `main.py` analyzes indicator configurations to determine how many candles are needed for each timeframe. It uses a 1.25x safety multiplier on the largest indicator period to ensure sufficient historical data for calculations.

### Position Management

The `TradeExecutor` handles three types of position closures:
1. **Stop-loss/Take-profit hits**: Checked on every primary timeframe update
2. **Regime changes**: Optional (controlled by `close_with_regime_change` setting)
3. **Manual closure**: `close_all_positions()` at backtest end or keyboard interrupt

### Backtesting Output

When backtesting completes, results are saved to `backtest_results/`:
- `price_data/`: CSV files of price data per timeframe (if `save_datastore: true`)
- `trade_history.json`: All executed trades with entry/exit details (if `save_trade_history: true`)

Metrics logged include: ROI, max drawdown, win rate, total trades, profit factor.

### Multi-Symbol Support

Currently configured for single-symbol operation (EUR/USD), but architecture supports multiple symbols via `strategy.symbols` array. The data stream yields `(data_dict, current_symbol)` tuples to enable this.

### Primary Timeframe

The shortest configured timeframe is used as the "primary timeframe" for position updates and SL/TP checking. Higher timeframes are used for regime detection and signal generation only.

## Error Handling & Known Issues

- **MACD crossover detection**: Recent git commits mention errors relating to MACD crossover. Check `indicator_engine.py` for crossover calculation logic.
- **Data provision errors**: Ensure timeframes in indicator configuration match those in `strategy.timeframes`.
- **API disconnections**: The system properly handles disconnections in try/finally blocks, but MT5 server must be manually restarted if it crashes.

## Extending the System

### Adding New Indicators

1. Implement calculation logic in `IndicatorEngine._calculate_indicator()` (`src/core/indicator_engine.py`)
2. Add indicator configuration to `config/main.yaml` under appropriate timeframe
3. Reference in trading logic conditions using `timeframe;indicator_name;field` syntax

### Adding New Trading APIs

1. Create new file in `src/core/trading_apis/` (see `mt5_fastapi.py` as template)
2. Implement `connect()`, `disconnect()`, `get_account_info()`, `place_order()`, `close_position()`, `get_ohlcv()` methods
3. Add API type to `load_trading_api()` in `main.py`
4. Configure in `config/main.yaml` under `trading_api.type`

### Modifying Strategy Logic

For most strategy changes, edit `config/main.yaml` only:
- Adjust indicator parameters
- Change regime detection thresholds
- Modify entry/exit conditions
- Update risk management rules

Code changes only needed for:
- New condition operators in the parser (`src/core/trading_logic.py`)
- New stop-loss/take-profit methods (`src/core/trade_execution.py`)
- New regime types beyond TRENDING/RANGING/NEUTRAL

## Git Workflow

Recent commits show active development on:
- Windows server component integration
- MACD crossover detection refinement
- Data provision stability

When making changes:
- Test in backtest mode first before live trading
- Review logs in `logs/` to debug issues
- Update `config/main.yaml` comments if adding new features
