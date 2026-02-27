"""
XAUUSD Daily Trend Buyer — Long Only
=====================================
Actieve dagelijkse strategie: volgt de trend, koopt alleen (geen short).

Regels:
- BUY als EMA(8) > EMA(21) EN prijs sluit boven EMA(8) → uptrend bevestigd
- CLOSE als prijs sluit onder EMA(8) → trend verzwakt
- Alleen LONG posities, nooit short
- Trailing stop loss van 1.5% voor risicobeheer
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from backtesting import Backtest, Strategy
from backtesting.lib import crossover


def EMA(series, period):
    return pd.Series(series).ewm(span=period, adjust=False).mean()


class DailyTrendBuyer(Strategy):
    ema_fast = 8
    ema_slow = 21
    trail_pct = 1.5  # trailing stop %

    def init(self):
        close = pd.Series(self.data.Close)
        self.ema8 = self.I(EMA, close, self.ema_fast)
        self.ema21 = self.I(EMA, close, self.ema_slow)

    def next(self):
        price = self.data.Close[-1]
        ema_f = self.ema8[-1]
        ema_s = self.ema21[-1]

        if not self.position:
            # BUY: uptrend bevestigd
            if ema_f > ema_s and price > ema_f:
                self.buy(sl=price * (1 - self.trail_pct / 100))
        else:
            # Update trailing stop
            new_sl = price * (1 - self.trail_pct / 100)
            if self.trades[-1].sl and new_sl > self.trades[-1].sl:
                self.trades[-1].sl = new_sl

            # EXIT: trend verzwakt
            if price < ema_f:
                self.position.close()


# ── Download data ────────────────────────────────────────────────
print("Downloading XAUUSD data...")
gold = yf.download("GC=F", start="2023-01-01", end="2025-12-31", progress=False)
if isinstance(gold.columns, pd.MultiIndex):
    gold.columns = gold.columns.get_level_values(0)
gold = gold.dropna()
print(f"Data: {len(gold)} dagen ({gold.index[0].strftime('%Y-%m-%d')} → {gold.index[-1].strftime('%Y-%m-%d')})")
print(f"Prijsrange: ${gold['Low'].min():.2f} – ${gold['High'].max():.2f}\n")


# ── Run backtest ─────────────────────────────────────────────────
print("Running Daily Trend Buyer backtest...")
bt = Backtest(
    gold,
    DailyTrendBuyer,
    cash=10_000,
    commission=0.0002,
    exclusive_orders=True,
)
stats = bt.run()


# ── Optimalisatie: test verschillende EMA periodes ───────────────
print("Optimalisatie: beste EMA combinatie zoeken...")
opt_stats = bt.optimize(
    ema_fast=range(5, 15),
    ema_slow=range(15, 35, 2),
    trail_pct=[1.0, 1.5, 2.0, 2.5],
    maximize="Equity Final [$]",
    max_tries=200,
)


# ── Print resultaten ─────────────────────────────────────────────
def print_stats(label, s):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    rows = [
        ("Startkapitaal", "$10,000"),
        ("Eindkapitaal", f"${s['Equity Final [$]']:,.2f}"),
        ("Rendement", f"{s['Return [%]']:.2f}%"),
        ("Buy & Hold", f"{s['Buy & Hold Return [%]']:.2f}%"),
        ("Max Drawdown", f"{s['Max. Drawdown [%]']:.2f}%"),
        ("Trades", int(s["# Trades"])),
        ("Win Rate", f"{s['Win Rate [%]']:.1f}%"),
        ("Sharpe Ratio", f"{s['Sharpe Ratio']:.2f}" if pd.notna(s["Sharpe Ratio"]) else "N/A"),
        ("Profit Factor", f"{s['Profit Factor']:.2f}" if pd.notna(s["Profit Factor"]) else "N/A"),
        ("Gem. trade duur", f"{s['Avg. Trade Duration']}"),
    ]
    if hasattr(s, "_strategy"):
        strat = s._strategy
        rows.insert(0, ("EMA Fast/Slow", f"{strat.ema_fast}/{strat.ema_slow}"))
        rows.insert(1, ("Trailing Stop", f"{strat.trail_pct}%"))
    for label, val in rows:
        print(f"    {label:<22} {val}")
    print(f"{'=' * 60}")

print_stats("DAILY TREND BUYER — EMA 8/21 (standaard)", stats)
print_stats("DAILY TREND BUYER — GEOPTIMALISEERD", opt_stats)


# ── Charts ───────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(16, 14),
                         gridspec_kw={"height_ratios": [3, 1.2, 1, 1]})
fig.suptitle("XAUUSD Daily Trend Buyer — Long Only Strategy", fontsize=16, fontweight="bold", y=0.98)

# 1: Price + EMAs + buy/sell markers
ax = axes[0]
ax.plot(gold.index, gold["Close"], color="#888", lw=0.8, label="XAUUSD", alpha=0.7)
ema8 = gold["Close"].ewm(span=8, adjust=False).mean()
ema21 = gold["Close"].ewm(span=21, adjust=False).mean()
ax.plot(gold.index, ema8, color="#2196F3", lw=1.3, label="EMA 8")
ax.plot(gold.index, ema21, color="#FF9800", lw=1.3, label="EMA 21")

# Kleur achtergrond groen als EMA8 > EMA21 (uptrend)
trend_up = ema8 > ema21
for i in range(1, len(gold)):
    if trend_up.iloc[i]:
        ax.axvspan(gold.index[i - 1], gold.index[i], alpha=0.05, color="green")

trades = stats["_trades"]
for _, t in trades.iterrows():
    ax.scatter(t["EntryTime"], t["EntryPrice"], marker="^", color="#4CAF50", s=50, zorder=5)
    ax.scatter(t["ExitTime"], t["ExitPrice"], marker="x", color="#F44336", s=40, zorder=5)

ax.scatter([], [], marker="^", color="#4CAF50", s=50, label="BUY")
ax.scatter([], [], marker="x", color="#F44336", s=40, label="CLOSE")
ax.set_ylabel("Prijs (USD)")
ax.legend(loc="upper left", fontsize=8, ncol=2)
ax.grid(True, alpha=0.2)

# 2: Trade P&L per trade
ax2 = axes[1]
colors = ["#4CAF50" if pnl > 0 else "#F44336" for pnl in trades["PnL"]]
ax2.bar(range(len(trades)), trades["PnL"], color=colors, width=0.8)
ax2.axhline(0, color="#333", lw=0.8)
ax2.set_ylabel("P&L per trade ($)")
ax2.set_xlabel("Trade #")
ax2.grid(True, alpha=0.2)
avg_pnl = trades["PnL"].mean()
ax2.axhline(avg_pnl, color="#2196F3", ls="--", lw=1, label=f"Gem: ${avg_pnl:.0f}")
ax2.legend(fontsize=8)

# 3: Equity curve
ax3 = axes[2]
eq = stats["_equity_curve"]
ax3.fill_between(eq.index, eq["Equity"], alpha=0.3, color="#4CAF50")
ax3.plot(eq.index, eq["Equity"], color="#4CAF50", lw=1.2)
ax3.axhline(10_000, color="#999", ls="--", lw=0.8)
ax3.set_ylabel("Equity ($)")
ax3.grid(True, alpha=0.2)

# 4: Drawdown
ax4 = axes[3]
dd = eq["DrawdownPct"] * 100
ax4.fill_between(eq.index, dd, alpha=0.3, color="#F44336")
ax4.plot(eq.index, dd, color="#F44336", lw=1)
ax4.set_ylabel("Drawdown (%)")
ax4.grid(True, alpha=0.2)

for a in [axes[0], axes[2], axes[3]]:
    a.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    a.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

plt.tight_layout()
plt.savefig("/workspace/xauusd_daily_trend_results.png", dpi=150, bbox_inches="tight")
print("\nChart opgeslagen: /workspace/xauusd_daily_trend_results.png")

# ── Trade log ────────────────────────────────────────────────────
print(f"\n── Alle {len(trades)} trades ──")
trades_display = trades[["EntryTime", "ExitTime", "Size", "EntryPrice", "ExitPrice", "PnL", "ReturnPct"]].copy()
trades_display["PnL"] = trades_display["PnL"].apply(lambda x: f"${x:+,.2f}")
trades_display["ReturnPct"] = trades_display["ReturnPct"].apply(lambda x: f"{x:+.2%}")
trades_display["EntryPrice"] = trades_display["EntryPrice"].apply(lambda x: f"${x:,.2f}")
trades_display["ExitPrice"] = trades_display["ExitPrice"].apply(lambda x: f"${x:,.2f}")
print(trades_display.to_string(index=False))
print("\nKlaar!")
