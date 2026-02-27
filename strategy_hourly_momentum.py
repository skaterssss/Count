"""
Strategy: Hourly Momentum Scanner — "De Uurlijkse Scanner"
==========================================================
Elk uur kijken we naar de afgelopen 24 uur-candles.
Op basis van momentum-score beslissen we: kopen of rust laten.

Regels:
- Elke bar = 1 uur candle
- Bereken momentum-score over afgelopen 24 candles
- BUY als score positief + EMA bevestigt uptrend
- Max 5 open posities tegelijk (pyramid)
- Trailing stop per positie
- Sluit zwakste positie als momentum negatief wordt

Gebruikt 1:50 leverage (standaard voor goud CFDs).
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


def EMA(series, period):
    return pd.Series(series).ewm(span=period, adjust=False).mean()


def compute_momentum(close, period=24):
    """Rolling momentum score over afgelopen N bars."""
    out = pd.Series(np.zeros(len(close)), dtype=float)
    c = np.array(close, dtype=float)
    for i in range(period, len(c)):
        w = c[i - period:i + 1]
        rets = np.diff(w) / w[:-1]

        direction = (np.sum(rets > 0) - np.sum(rets < 0)) / period
        total_ret = (w[-1] - w[0]) / w[0]
        recent = (w[-1] - w[-min(6, len(w))]) / w[-min(6, len(w))]
        early = (w[min(5, len(w)-1)] - w[0]) / w[0]
        accel = recent - early
        ma_pos = (w[-1] - np.mean(w)) / np.mean(w)

        out.iloc[i] = direction * 30 + total_ret * 200 + accel * 150 + ma_pos * 100
    return out


class HourlyMomentumScanner(Strategy):
    lookback = 24
    buy_threshold = 2.0
    close_threshold = -2.0
    trail_pct = 1.2
    max_positions = 5

    def init(self):
        close = pd.Series(self.data.Close, dtype=float)
        self.momentum = self.I(compute_momentum, close, self.lookback)
        self.ema_fast = self.I(EMA, close, 8)
        self.ema_slow = self.I(EMA, close, 24)
        self.bar = 0

    def next(self):
        self.bar += 1
        price = self.data.Close[-1]
        score = self.momentum[-1]
        n_open = len(self.trades)

        # Update trailing stops
        for trade in self.trades:
            new_sl = price * (1 - self.trail_pct / 100)
            if trade.sl is None or new_sl > trade.sl:
                trade.sl = new_sl

        # BUY: momentum positief + EMA uptrend + ruimte
        if score > self.buy_threshold and n_open < self.max_positions:
            if self.ema_fast[-1] > self.ema_slow[-1]:
                sl = price * (1 - self.trail_pct / 100)
                frac = 0.90 / self.max_positions
                self.buy(size=frac, sl=sl)

        # CLOSE zwakste als momentum sterk negatief
        if score < self.close_threshold and n_open > 0:
            worst = min(self.trades, key=lambda t: t.pl)
            worst.close()


# ── Download data ────────────────────────────────────────────────
print("Downloading XAUUSD 1-hour data...")
gold_h1 = yf.download("GC=F", period="60d", interval="1h", progress=False)
if isinstance(gold_h1.columns, pd.MultiIndex):
    gold_h1.columns = gold_h1.columns.get_level_values(0)
gold_h1 = gold_h1.dropna()
if gold_h1.index.tzinfo is not None:
    gold_h1.index = gold_h1.index.tz_localize(None)

print(f"   {len(gold_h1)} uur-candles")
print(f"   {gold_h1.index[0].strftime('%Y-%m-%d %H:%M')} -> {gold_h1.index[-1].strftime('%Y-%m-%d %H:%M')}")
print(f"   Prijs: ${gold_h1['Low'].min():.2f} - ${gold_h1['High'].max():.2f}\n")


# ── Backtest met leverage (1:50, standaard voor goud) ────────────
print("Running Hourly Momentum Scanner (1:50 leverage)...\n")
bt = Backtest(
    gold_h1,
    HourlyMomentumScanner,
    cash=10_000,
    commission=0.0002,
    margin=1/50,           # 50x leverage = standaard goud CFD
    exclusive_orders=False,
    hedging=True,
)
stats = bt.run()


# ── Optimalisatie ────────────────────────────────────────────────
print("Optimalisatie loopt...")
opt_stats = bt.optimize(
    lookback=[12, 18, 24],
    buy_threshold=[1.0, 2.0, 3.0, 5.0],
    close_threshold=[-1.0, -2.0, -5.0],
    trail_pct=[0.6, 0.8, 1.0, 1.2],
    max_positions=[3, 5],
    maximize="Equity Final [$]",
    max_tries=400,
)


# ── Print ────────────────────────────────────────────────────────
def print_block(label, s):
    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print(f"{'=' * 65}")
    items = []
    if hasattr(s, "_strategy"):
        st = s._strategy
        items += [
            ("Lookback", f"{st.lookback} uur"),
            ("Koop drempel", f"{st.buy_threshold}"),
            ("Sluit drempel", f"{st.close_threshold}"),
            ("Trailing stop", f"{st.trail_pct}%"),
            ("Max posities", f"{st.max_positions}"),
        ]
    items += [
        ("Leverage", "1:50"),
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
    for lbl, val in items:
        print(f"    {lbl:<22} {val}")
    print(f"{'=' * 65}")

print_block("STANDAARD — EMA 8/24, lookback 24u", stats)
print_block("GEOPTIMALISEERD", opt_stats)


# ── Bepaal beste resultaat voor charts ───────────────────────────
best = opt_stats if opt_stats["# Trades"] > 0 and opt_stats["Equity Final [$]"] > stats["Equity Final [$]"] else stats
best_label = "geoptimaliseerd" if best is opt_stats else "standaard"
trades = best["_trades"]
eq = best["_equity_curve"]


# ── Charts ───────────────────────────────────────────────────────
fig, axes = plt.subplots(5, 1, figsize=(18, 18),
                         gridspec_kw={"height_ratios": [3, 1.2, 1.2, 1, 1]})
fig.suptitle(f'XAUUSD Hourly Momentum Scanner — {best_label} (1:50 leverage)',
             fontsize=16, fontweight="bold", y=0.98)

# 1: Price + EMAs + trades
ax = axes[0]
ax.plot(gold_h1.index, gold_h1["Close"], color="#888", lw=0.6, alpha=0.7, label="XAUUSD 1H")
ema8 = gold_h1["Close"].ewm(span=8, adjust=False).mean()
ema24 = gold_h1["Close"].ewm(span=24, adjust=False).mean()
ax.plot(gold_h1.index, ema8, color="#2196F3", lw=1, alpha=0.8, label="EMA 8")
ax.plot(gold_h1.index, ema24, color="#FF9800", lw=1, alpha=0.8, label="EMA 24")
if len(trades) > 0:
    for _, t in trades.iterrows():
        ax.scatter(t["EntryTime"], t["EntryPrice"], marker="^", color="#4CAF50", s=30, zorder=5, alpha=0.7)
        c = "#4CAF50" if t["PnL"] > 0 else "#F44336"
        ax.scatter(t["ExitTime"], t["ExitPrice"], marker="x", color=c, s=25, zorder=5, alpha=0.7)
ax.scatter([], [], marker="^", color="#4CAF50", s=40, label="BUY")
ax.scatter([], [], marker="x", color="#F44336", s=35, label="CLOSE")
ax.set_ylabel("Prijs (USD)")
ax.legend(loc="upper left", fontsize=7, ncol=3)
ax.grid(True, alpha=0.2)

# 2: Momentum score
ax2 = axes[1]
scores_plot = compute_momentum(gold_h1["Close"].values, 24).values
ax2.fill_between(gold_h1.index, scores_plot, where=scores_plot > 0, alpha=0.4, color="#4CAF50")
ax2.fill_between(gold_h1.index, scores_plot, where=scores_plot <= 0, alpha=0.4, color="#F44336")
ax2.plot(gold_h1.index, scores_plot, color="#333", lw=0.5)
ax2.axhline(2.0, color="#4CAF50", ls="--", lw=0.7, alpha=0.5, label="Koop drempel")
ax2.axhline(-2.0, color="#F44336", ls="--", lw=0.7, alpha=0.5, label="Sluit drempel")
ax2.axhline(0, color="#999", lw=0.5)
ax2.set_ylabel("Momentum Score")
ax2.legend(fontsize=7)
ax2.grid(True, alpha=0.2)

# 3: Trade P&L
ax3 = axes[2]
if len(trades) > 0:
    colors = ["#4CAF50" if p > 0 else "#F44336" for p in trades["PnL"]]
    ax3.bar(range(len(trades)), trades["PnL"], color=colors, width=0.8)
    avg = trades["PnL"].mean()
    ax3.axhline(avg, color="#2196F3", ls="--", lw=1, label=f"Gem: ${avg:.0f}")
    ax3.legend(fontsize=7)
ax3.axhline(0, color="#333", lw=0.8)
ax3.set_ylabel("P&L per trade ($)")
ax3.set_xlabel("Trade #")
ax3.grid(True, alpha=0.2)

# 4: Equity
ax4 = axes[3]
ax4.fill_between(eq.index, eq["Equity"], alpha=0.3, color="#4CAF50")
ax4.plot(eq.index, eq["Equity"], color="#4CAF50", lw=1)
ax4.axhline(10_000, color="#999", ls="--", lw=0.8)
ax4.set_ylabel("Equity ($)")
ax4.grid(True, alpha=0.2)

# 5: Drawdown
ax5 = axes[4]
dd = eq["DrawdownPct"] * 100
ax5.fill_between(eq.index, dd, alpha=0.3, color="#F44336")
ax5.plot(eq.index, dd, color="#F44336", lw=1)
ax5.set_ylabel("Drawdown (%)")
ax5.grid(True, alpha=0.2)

for a in [axes[0], axes[1], axes[3], axes[4]]:
    a.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    a.xaxis.set_major_locator(mdates.DayLocator(interval=5))

plt.tight_layout()
plt.savefig("/workspace/hourly_momentum_results.png", dpi=150, bbox_inches="tight")
print("\nChart opgeslagen: /workspace/hourly_momentum_results.png")


# ── Trade log ────────────────────────────────────────────────────
if len(trades) > 0:
    print(f"\n── Laatste 20 van {len(trades)} trades ──")
    td = trades[["EntryTime", "ExitTime", "Size", "EntryPrice", "ExitPrice", "PnL", "ReturnPct"]].tail(20).copy()
    td["PnL"] = td["PnL"].apply(lambda x: f"${x:+,.2f}")
    td["ReturnPct"] = td["ReturnPct"].apply(lambda x: f"{x:+.2%}")
    td["EntryPrice"] = td["EntryPrice"].apply(lambda x: f"${x:,.2f}")
    td["ExitPrice"] = td["ExitPrice"].apply(lambda x: f"${x:,.2f}")
    print(td.to_string(index=False))
else:
    print("\nGeen trades — controleer parameters.")

print(f"""
{'=' * 65}
  STRATEGIEEN OVERZICHT
{'=' * 65}
  1. strategy_sma_crossover.py      — "SMA Crossover"
  2. strategy_daily_trend_buyer.py   — "Daily Trend Buyer"
  3. strategy_hourly_momentum.py     — "De Uurlijkse Scanner"
{'=' * 65}
""")
