"""
XAUUSD Backtest — Moving Average Crossover Strategy
====================================================
Simulates trading Gold (XAU/USD) using a fast/slow MA crossover.
- BUY when fast MA crosses above slow MA
- SELL when fast MA crosses below slow MA

Uses free historical data from Yahoo Finance.
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
from datetime import datetime


# ── 1. Download XAUUSD historical data ──────────────────────────
print("Downloading XAUUSD historical data...")
gold = yf.download("GC=F", start="2023-01-01", end="2025-12-31", progress=False)

if gold.empty:
    print("GC=F failed, trying XAUUSD=X...")
    gold = yf.download("XAUUSD=X", start="2023-01-01", end="2025-12-31", progress=False)

# Flatten MultiIndex columns if present
if isinstance(gold.columns, pd.MultiIndex):
    gold.columns = gold.columns.get_level_values(0)

gold = gold.dropna()
print(f"Loaded {len(gold)} days of data: {gold.index[0].strftime('%Y-%m-%d')} → {gold.index[-1].strftime('%Y-%m-%d')}")
print(f"Price range: ${gold['Low'].min():.2f} – ${gold['High'].max():.2f}\n")


# ── 2. Define strategy ──────────────────────────────────────────
class GoldMACrossover(Strategy):
    fast_period = 10
    slow_period = 30

    def init(self):
        close = self.data.Close
        self.fast_ma = self.I(SMA, close, self.fast_period)
        self.slow_ma = self.I(SMA, close, self.slow_period)

    def next(self):
        if crossover(self.fast_ma, self.slow_ma):
            self.buy()
        elif crossover(self.slow_ma, self.fast_ma):
            self.sell()


# ── 3. Run backtest ─────────────────────────────────────────────
print("Running backtest (SMA 10/30 crossover)...")
bt = Backtest(
    gold,
    GoldMACrossover,
    cash=10_000,
    commission=0.0002,
    exclusive_orders=True,
)
stats = bt.run()

print("\n" + "=" * 55)
print("       BACKTEST RESULTS — XAUUSD MA CROSSOVER")
print("=" * 55)
key_stats = [
    ("Start", stats["Start"]),
    ("End", stats["End"]),
    ("Duration", stats["Duration"]),
    ("Initial Capital", "$10,000"),
    ("Final Equity", f"${stats['Equity Final [$]']:,.2f}"),
    ("Return", f"{stats['Return [%]']:.2f}%"),
    ("Buy & Hold Return", f"{stats['Buy & Hold Return [%]']:.2f}%"),
    ("Max Drawdown", f"{stats['Max. Drawdown [%]']:.2f}%"),
    ("Total Trades", int(stats["# Trades"])),
    ("Win Rate", f"{stats['Win Rate [%]']:.1f}%"),
    ("Sharpe Ratio", f"{stats['Sharpe Ratio']:.2f}" if pd.notna(stats['Sharpe Ratio']) else "N/A"),
    ("Profit Factor", f"{stats['Profit Factor']:.2f}" if pd.notna(stats['Profit Factor']) else "N/A"),
]
for label, val in key_stats:
    print(f"  {label:<22} {val}")
print("=" * 55)


# ── 4. Generate charts ──────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1, 1]})
fig.suptitle("XAUUSD Backtest — SMA 10/30 Crossover Strategy", fontsize=15, fontweight="bold")

# Chart 1: Price + MAs + signals
ax1 = axes[0]
ax1.plot(gold.index, gold["Close"], color="#888", linewidth=0.8, label="XAUUSD Close")
fast_ma = gold["Close"].rolling(10).mean()
slow_ma = gold["Close"].rolling(30).mean()
ax1.plot(gold.index, fast_ma, color="#2196F3", linewidth=1.2, label="SMA 10")
ax1.plot(gold.index, slow_ma, color="#FF5722", linewidth=1.2, label="SMA 30")

buy_signals = (fast_ma.shift(1) < slow_ma.shift(1)) & (fast_ma > slow_ma)
sell_signals = (fast_ma.shift(1) > slow_ma.shift(1)) & (fast_ma < slow_ma)
ax1.scatter(gold.index[buy_signals], gold["Close"][buy_signals], marker="^", color="#4CAF50", s=80, zorder=5, label="BUY")
ax1.scatter(gold.index[sell_signals], gold["Close"][sell_signals], marker="v", color="#F44336", s=80, zorder=5, label="SELL")

ax1.set_ylabel("Price (USD)")
ax1.legend(loc="upper left", fontsize=8)
ax1.grid(True, alpha=0.3)

# Chart 2: Equity curve
ax2 = axes[1]
equity = stats["_equity_curve"]
ax2.fill_between(equity.index, equity["Equity"], alpha=0.3, color="#4CAF50")
ax2.plot(equity.index, equity["Equity"], color="#4CAF50", linewidth=1)
ax2.set_ylabel("Equity ($)")
ax2.grid(True, alpha=0.3)
ax2.axhline(10_000, color="#999", linestyle="--", linewidth=0.8)

# Chart 3: Drawdown
ax3 = axes[2]
dd = equity["DrawdownPct"] * 100
ax3.fill_between(equity.index, dd, alpha=0.3, color="#F44336")
ax3.plot(equity.index, dd, color="#F44336", linewidth=1)
ax3.set_ylabel("Drawdown (%)")
ax3.set_xlabel("Date")
ax3.grid(True, alpha=0.3)

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

plt.tight_layout()
plt.savefig("/workspace/xauusd_backtest_results.png", dpi=150, bbox_inches="tight")
print("\nChart saved to /workspace/xauusd_backtest_results.png")


# ── 5. Trade log ────────────────────────────────────────────────
trades = stats["_trades"]
print(f"\n── Last 10 trades ──")
print(trades[["EntryTime", "ExitTime", "Size", "EntryPrice", "ExitPrice", "PnL", "ReturnPct"]].tail(10).to_string())
print(f"\nDone! Total trades: {len(trades)}")
