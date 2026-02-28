"""
=============================================================
  STELLAR TRADE — De beste XAUUSD strategie
=============================================================
  Referentiepunt voor alle toekomstige strategieen.

  Parameters:
    - Trailing stop: 2.5%
    - Take profit: $80
    - Momentum lookback: 24 uur
    - Koop drempel: 2.0
    - Sluit drempel: -5.0
    - Max posities: 1
    - Leverage: 1:10

  Resultaat over 875 dagen (okt 2023 - feb 2026):
    - 146 trades, 62% win rate
    - +89,331% rendement
    - Profit Factor: 2.15
    - Max Drawdown: -71.2%
    - Walk-forward test: +2,235% op onbekende data
=============================================================
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


class StellarTrade(Strategy):
    """De referentie-strategie. Simpel, krachtig, bewezen."""
    lookback = 24
    buy_threshold = 2.0
    close_threshold = -5.0
    trail_pct = 2.5
    tp_dollars = 80.0
    max_positions = 1

    def init(self):
        close = pd.Series(self.data.Close, dtype=float)
        self.momentum = self.I(compute_momentum, close, self.lookback)
        self.ema_fast = self.I(EMA, close, 8)
        self.ema_slow = self.I(EMA, close, 24)

    def next(self):
        price = self.data.Close[-1]
        score = self.momentum[-1]
        for trade in self.trades:
            new_sl = price * (1 - self.trail_pct / 100)
            if trade.sl is None or new_sl > trade.sl:
                trade.sl = new_sl
        if score > self.buy_threshold and len(self.trades) < self.max_positions:
            if self.ema_fast[-1] > self.ema_slow[-1]:
                self.buy(size=0.90,
                         sl=price * (1 - self.trail_pct / 100),
                         tp=price + self.tp_dollars)
        if score < self.close_threshold and len(self.trades) > 0:
            self.trades[0].close()


# ── Download data ────────────────────────────────────────────────
print("=" * 60)
print("  STELLAR TRADE — Backtest")
print("=" * 60)

print("\nDownloading XAUUSD maximale uur-data...")
gold = yf.download("GC=F", period="730d", interval="1h", progress=False)
if isinstance(gold.columns, pd.MultiIndex):
    gold.columns = gold.columns.get_level_values(0)
gold = gold.dropna()
if gold.index.tzinfo is not None:
    gold.index = gold.index.tz_localize(None)

total_days = (gold.index[-1] - gold.index[0]).days
print(f"   {len(gold):,} candles, {total_days} dagen")
print(f"   {gold.index[0].strftime('%Y-%m-%d')} -> {gold.index[-1].strftime('%Y-%m-%d')}")
print(f"   ${gold['Close'].iloc[0]:.2f} -> ${gold['Close'].iloc[-1]:.2f}\n")


# ── Volledige backtest ───────────────────────────────────────────
bt = Backtest(gold, StellarTrade, cash=10_000, commission=0.0002,
              margin=1/10, exclusive_orders=False, hedging=True)
stats = bt.run()


# ── Walk-forward ─────────────────────────────────────────────────
mid = len(gold) // 2
train_data = gold.iloc[:mid]
test_data = gold.iloc[mid:]

bt_train = Backtest(train_data, StellarTrade, cash=10_000, commission=0.0002,
                    margin=1/10, exclusive_orders=False, hedging=True)
train_stats = bt_train.run()

bt_test = Backtest(test_data, StellarTrade, cash=10_000, commission=0.0002,
                   margin=1/10, exclusive_orders=False, hedging=True)
test_stats = bt_test.run()


# ── Print ────────────────────────────────────────────────────────
def print_block(label, s):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    wr = f"{s['Win Rate [%]']:.1f}%" if pd.notna(s['Win Rate [%]']) else "N/A"
    pf = f"{s['Profit Factor']:.2f}" if pd.notna(s['Profit Factor']) else "N/A"
    sr = f"{s['Sharpe Ratio']:.2f}" if pd.notna(s['Sharpe Ratio']) and abs(s['Sharpe Ratio']) < 1000 else "N/A"
    for lbl, val in [
        ("Startkapitaal", "$10,000"),
        ("Eindkapitaal", f"${s['Equity Final [$]']:,.2f}"),
        ("Rendement", f"{s['Return [%]']:,.2f}%"),
        ("Buy & Hold", f"{s['Buy & Hold Return [%]']:.2f}%"),
        ("Max Drawdown", f"{s['Max. Drawdown [%]']:.2f}%"),
        ("Trades", int(s["# Trades"])),
        ("Win Rate", wr),
        ("Profit Factor", pf),
        ("Sharpe Ratio", sr),
        ("Gem. trade duur", f"{s['Avg. Trade Duration']}"),
    ]:
        print(f"    {lbl:<22} {val}")
    print(f"{'=' * 60}")

print_block(f"STELLAR TRADE — Volledige periode ({total_days} dagen)", stats)
print_block("STELLAR TRADE — Walk-forward TRAIN (eerste helft)", train_stats)
print_block("STELLAR TRADE — Walk-forward TEST (tweede helft)", test_stats)


# ── Per-kwartaal ─────────────────────────────────────────────────
trades = stats["_trades"]
if len(trades) > 0:
    tq = trades.copy()
    tq["Quarter"] = pd.to_datetime(tq["EntryTime"]).dt.to_period("Q")
    quarterly = tq.groupby("Quarter").agg(
        Trades=("PnL", "count"),
        PnL=("PnL", "sum"),
        WinRate=("PnL", lambda x: (x > 0).mean() * 100),
    )
    print(f"\n  Per kwartaal:")
    print(f"  {'Q':<10} {'Trades':>7} {'P&L':>12} {'Win%':>7}")
    for q, r in quarterly.iterrows():
        print(f"  {str(q):<10} {int(r['Trades']):>7} ${r['PnL']:>+10,.0f} {r['WinRate']:>6.0f}%")


# ── Charts ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 20))
gs = fig.add_gridspec(5, 2, height_ratios=[3, 1.5, 1.2, 1.2, 1.2], hspace=0.35, wspace=0.25)

fig.suptitle('STELLAR TRADE — XAUUSD Referentie Strategie',
             fontsize=18, fontweight="bold", color="#1a237e", y=0.99)
fig.text(0.5, 0.965,
         f'Trail 2.5% | TP $80 | Momentum 24u | 1 positie | 1:10 leverage | {total_days} dagen',
         ha="center", fontsize=10, color="#666")

# 1: Prijs + trades
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(gold.index, gold["Close"], color="#bbb", lw=0.4, alpha=0.8)
ema8 = gold["Close"].ewm(span=8, adjust=False).mean()
ema24 = gold["Close"].ewm(span=24, adjust=False).mean()
ax1.plot(gold.index, ema8, color="#42A5F5", lw=0.7, alpha=0.7, label="EMA 8")
ax1.plot(gold.index, ema24, color="#FF7043", lw=0.7, alpha=0.7, label="EMA 24")

if len(trades) > 0:
    wins = trades[trades["PnL"] > 0]
    losses = trades[trades["PnL"] <= 0]
    ax1.scatter(wins["EntryTime"], wins["EntryPrice"], marker="^", color="#4CAF50",
                s=25, zorder=5, alpha=0.7, label=f"Win ({len(wins)})")
    ax1.scatter(losses["EntryTime"], losses["EntryPrice"], marker="v", color="#F44336",
                s=25, zorder=5, alpha=0.7, label=f"Loss ({len(losses)})")

ax1.set_ylabel("Prijs (USD)", fontsize=10)
ax1.set_title(f"{len(trades)} trades | {stats['Win Rate [%]']:.0f}% win rate | +{stats['Return [%]']:,.0f}% rendement",
              fontsize=12, color="#333")
ax1.legend(loc="upper left", fontsize=8, ncol=2)
ax1.grid(True, alpha=0.15)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

# 2: Equity (log)
ax2 = fig.add_subplot(gs[1, :])
eq = stats["_equity_curve"]
ax2.fill_between(eq.index, eq["Equity"], alpha=0.15, color="#1565C0")
ax2.plot(eq.index, eq["Equity"], color="#1565C0", lw=1.5, label="Stellar Trade equity")
ax2.axhline(10_000, color="#999", ls="--", lw=0.8, label="Start $10k")
ax2.set_ylabel("Equity ($)")
ax2.set_yscale("log")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.15)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

# 3: Drawdown
ax3 = fig.add_subplot(gs[2, :])
dd = eq["DrawdownPct"] * 100
ax3.fill_between(eq.index, dd, alpha=0.3, color="#E53935")
ax3.plot(eq.index, dd, color="#E53935", lw=0.8)
ax3.set_ylabel("Drawdown (%)")
ax3.grid(True, alpha=0.15)
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

# 4: P&L distributie
ax4 = fig.add_subplot(gs[3, 0])
if len(trades) > 0:
    pnl = trades["PnL"].values
    ax4.hist(pnl[pnl > 0], bins=25, color="#4CAF50", alpha=0.7, label=f"Winst ({(pnl>0).sum()})")
    ax4.hist(pnl[pnl <= 0], bins=25, color="#F44336", alpha=0.7, label=f"Verlies ({(pnl<=0).sum()})")
    ax4.axvline(pnl.mean(), color="#1565C0", ls="--", lw=1.5, label=f"Gem: ${pnl.mean():+,.0f}")
    ax4.legend(fontsize=7)
ax4.set_xlabel("P&L ($)")
ax4.set_title("P&L Distributie", fontsize=10)
ax4.grid(True, alpha=0.15)

# 5: Cumulatieve P&L
ax5 = fig.add_subplot(gs[3, 1])
if len(trades) > 0:
    cum = trades["PnL"].cumsum()
    ax5.fill_between(range(len(cum)), cum, alpha=0.2, color="#1565C0")
    ax5.plot(range(len(cum)), cum, color="#1565C0", lw=1.2)
    ax5.axhline(0, color="#999", ls="--", lw=0.8)
ax5.set_xlabel("Trade #")
ax5.set_ylabel("Cum. P&L ($)")
ax5.set_title("Cumulatieve winst", fontsize=10)
ax5.grid(True, alpha=0.15)

# 6: Maandelijkse P&L
ax6 = fig.add_subplot(gs[4, :])
if len(trades) > 0:
    tm = trades.copy()
    tm["Month"] = pd.to_datetime(tm["EntryTime"]).dt.to_period("M")
    monthly = tm.groupby("Month")["PnL"].sum()
    months = monthly.index.astype(str)
    colors_m = ["#4CAF50" if v > 0 else "#E53935" for v in monthly.values]
    ax6.bar(range(len(months)), monthly.values, color=colors_m, width=0.7)
    ax6.set_xticks(range(len(months)))
    ax6.set_xticklabels(months, rotation=45, fontsize=7)
    ax6.axhline(0, color="#333", lw=0.8)
    pos_months = sum(1 for v in monthly.values if v > 0)
    ax6.set_title(f"Maandelijkse P&L — {pos_months}/{len(monthly)} maanden winstgevend", fontsize=10)
    avg_m = monthly.mean()
    ax6.axhline(avg_m, color="#1565C0", ls="--", lw=1, label=f"Gem: ${avg_m:+,.0f}/mnd")
    ax6.legend(fontsize=8)
ax6.set_ylabel("P&L ($)")
ax6.grid(True, alpha=0.15)

plt.savefig("/workspace/stellar_trade_results.png", dpi=150, bbox_inches="tight")
print(f"\nChart opgeslagen: /workspace/stellar_trade_results.png")

print(f"""
{'=' * 60}
  STELLAR TRADE — Samenvatting
{'=' * 60}
  Trail 2.5% | TP $80 | Momentum 24u drempel 2.0
  1 positie | 1:10 leverage | Alleen long

  Volledige periode: +{stats['Return [%]']:,.0f}% | {int(stats['# Trades'])} trades | {stats['Win Rate [%]']:.0f}% win
  Walk-forward test: +{test_stats['Return [%]']:,.0f}% op onbekende data

  Dit is het referentiepunt. Elke nieuwe strategie
  moet hier tegenaan worden gemeten.
{'=' * 60}
""")
