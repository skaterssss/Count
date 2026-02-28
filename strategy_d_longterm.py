"""
Strategy D Langetermijn — Robuustheid zoeken
=============================================
De 60-dagen resultaten waren overfitting. Nu zoeken we:
1. Parameters die over 875 dagen winstgevend zijn
2. Lagere leverage testen (1:10, 1:20, 1:50)
3. Walk-forward analyse: train op eerste helft, test op tweede
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


class HybridSLTP(Strategy):
    lookback = 24
    buy_threshold = 2.0
    close_threshold = -5.0
    trail_pct = 1.0
    tp_dollars = 50.0
    max_positions = 3

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
                self.buy(size=0.90 / self.max_positions,
                         sl=price * (1 - self.trail_pct / 100),
                         tp=price + self.tp_dollars)
        if score < self.close_threshold and len(self.trades) > 0:
            min(self.trades, key=lambda t: t.pl).close()


# ── Download data ────────────────────────────────────────────────
print("Downloading XAUUSD 730 dagen uur-data...")
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


# ── Test meerdere leverage niveaus ───────────────────────────────
print("=" * 75)
print("  STAP 1: Impact van leverage op methode D (originele params)")
print("=" * 75)

leverages = [5, 10, 20, 50]
lev_results = {}

for lev in leverages:
    bt = Backtest(gold, HybridSLTP, cash=10_000, commission=0.0002,
                  margin=1/lev, exclusive_orders=False, hedging=True)
    s = bt.run()
    lev_results[lev] = s
    wr = f"{s['Win Rate [%]']:.0f}%" if pd.notna(s['Win Rate [%]']) else "N/A"
    print(f"  1:{lev:<4} → {int(s['# Trades']):>4} trades, {wr:>5} win, {s['Return [%]']:>+10.1f}%, DD {s['Max. Drawdown [%]']:>6.1f}%, Eind ${s['Equity Final [$]']:>10,.2f}")


# ── Optimaliseer per leverage ────────────────────────────────────
print(f"\n{'=' * 75}")
print("  STAP 2: Optimalisatie per leverage niveau")
print(f"{'=' * 75}")

opt_results = {}
for lev in [10, 20]:
    print(f"\n  Optimaliseer 1:{lev}...")
    bt = Backtest(gold, HybridSLTP, cash=10_000, commission=0.0002,
                  margin=1/lev, exclusive_orders=False, hedging=True)
    s = bt.optimize(
        trail_pct=[0.8, 1.0, 1.5, 2.0, 2.5, 3.0],
        tp_dollars=[30, 50, 80, 100, 150],
        buy_threshold=[2.0, 3.0, 5.0, 8.0],
        close_threshold=[-5.0, -10.0, -15.0],
        max_positions=[1, 2, 3],
        maximize="Equity Final [$]",
        max_tries=800,
    )
    opt_results[lev] = s
    st = s._strategy
    wr = f"{s['Win Rate [%]']:.0f}%" if pd.notna(s['Win Rate [%]']) else "N/A"
    pf = f"{s['Profit Factor']:.2f}" if pd.notna(s['Profit Factor']) else "N/A"
    print(f"  1:{lev} best → trail {st.trail_pct}%, TP ${st.tp_dollars:.0f}, drempel {st.buy_threshold}, max {st.max_positions}")
    print(f"           {int(s['# Trades'])} trades, {wr} win, {s['Return [%]']:+.1f}%, DD {s['Max. Drawdown [%]']:.1f}%, PF {pf}")


# ── Walk-forward analyse ─────────────────────────────────────────
print(f"\n{'=' * 75}")
print("  STAP 3: Walk-forward (train eerste helft, test tweede helft)")
print(f"{'=' * 75}")

mid = len(gold) // 2
train_data = gold.iloc[:mid]
test_data = gold.iloc[mid:]
print(f"  Train: {train_data.index[0].strftime('%Y-%m-%d')} -> {train_data.index[-1].strftime('%Y-%m-%d')} ({len(train_data):,} candles)")
print(f"  Test:  {test_data.index[0].strftime('%Y-%m-%d')} -> {test_data.index[-1].strftime('%Y-%m-%d')} ({len(test_data):,} candles)")

# Train
bt_train = Backtest(train_data, HybridSLTP, cash=10_000, commission=0.0002,
                    margin=1/20, exclusive_orders=False, hedging=True)
train_opt = bt_train.optimize(
    trail_pct=[0.8, 1.0, 1.5, 2.0, 2.5, 3.0],
    tp_dollars=[30, 50, 80, 100, 150],
    buy_threshold=[2.0, 3.0, 5.0, 8.0],
    close_threshold=[-5.0, -10.0, -15.0],
    max_positions=[1, 2, 3],
    maximize="Equity Final [$]",
    max_tries=800,
)
train_st = train_opt._strategy
print(f"\n  Train-optimaal: trail {train_st.trail_pct}%, TP ${train_st.tp_dollars:.0f}, drempel {train_st.buy_threshold}, sluit {train_st.close_threshold}, max {train_st.max_positions}")
print(f"  Train resultaat: {train_opt['Return [%]']:+.1f}%, {int(train_opt['# Trades'])} trades")

# Test met getrainde params
class WalkForwardTest(HybridSLTP):
    trail_pct = train_st.trail_pct
    tp_dollars = train_st.tp_dollars
    buy_threshold = train_st.buy_threshold
    close_threshold = train_st.close_threshold
    max_positions = train_st.max_positions

bt_test = Backtest(test_data, WalkForwardTest, cash=10_000, commission=0.0002,
                   margin=1/20, exclusive_orders=False, hedging=True)
test_stats = bt_test.run()
wr_test = f"{test_stats['Win Rate [%]']:.0f}%" if pd.notna(test_stats['Win Rate [%]']) else "N/A"
pf_test = f"{test_stats['Profit Factor']:.2f}" if pd.notna(test_stats['Profit Factor']) else "N/A"
print(f"  Test resultaat:  {test_stats['Return [%]']:+.1f}%, {int(test_stats['# Trades'])} trades, {wr_test} win, PF {pf_test}, DD {test_stats['Max. Drawdown [%]']:.1f}%")


# ── Samenvatting ─────────────────────────────────────────────────
print(f"""
{'=' * 90}
  SAMENVATTING — LANGETERMIJN ROBUUSTHEID
{'=' * 90}

  {'Setup':<40} {'Leverage':>8} {'Trades':>7} {'Win%':>6} {'Return':>10} {'MaxDD':>8} {'PF':>6}
  {'-'*40} {'-'*8} {'-'*7} {'-'*6} {'-'*10} {'-'*8} {'-'*6}""")

# Original D at different leverages
for lev in leverages:
    s = lev_results[lev]
    wr = f"{s['Win Rate [%]']:.0f}%" if pd.notna(s['Win Rate [%]']) else "N/A"
    pf = f"{s['Profit Factor']:.2f}" if pd.notna(s['Profit Factor']) else "N/A"
    print(f"  {'Origineel D':<40} {'1:'+str(lev):>8} {int(s['# Trades']):>7} {wr:>6} {s['Return [%]']:>+9.1f}% {s['Max. Drawdown [%]']:>7.1f}% {pf:>6}")

print()
for lev in [10, 20]:
    s = opt_results[lev]
    wr = f"{s['Win Rate [%]']:.0f}%" if pd.notna(s['Win Rate [%]']) else "N/A"
    pf = f"{s['Profit Factor']:.2f}" if pd.notna(s['Profit Factor']) else "N/A"
    print(f"  {'Geoptimaliseerd':<40} {'1:'+str(lev):>8} {int(s['# Trades']):>7} {wr:>6} {s['Return [%]']:>+9.1f}% {s['Max. Drawdown [%]']:>7.1f}% {pf:>6}")

print()
print(f"  {'Walk-forward (train)':<40} {'1:20':>8} {int(train_opt['# Trades']):>7} {'':>6} {train_opt['Return [%]']:>+9.1f}%")
wr_wf = f"{test_stats['Win Rate [%]']:.0f}%" if pd.notna(test_stats['Win Rate [%]']) else "N/A"
pf_wf = f"{test_stats['Profit Factor']:.2f}" if pd.notna(test_stats['Profit Factor']) else "N/A"
print(f"  {'Walk-forward (TEST — onbekende data)':<40} {'1:20':>8} {int(test_stats['# Trades']):>7} {wr_wf:>6} {test_stats['Return [%]']:>+9.1f}% {test_stats['Max. Drawdown [%]']:>7.1f}% {pf_wf:>6}")

print(f"\n{'=' * 90}")


# ── Charts ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 20))
gs = fig.add_gridspec(5, 2, height_ratios=[2.5, 1.5, 1.2, 1.2, 1.2], hspace=0.35, wspace=0.25)
fig.suptitle(f'Methode D Langetermijn — {total_days} dagen robuustheid',
             fontsize=17, fontweight="bold", y=0.99)

# 1: Prijs + walk-forward split
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(gold.index, gold["Close"], color="#888", lw=0.4, alpha=0.7, label="XAUUSD 1H")
ax1.axvline(gold.index[mid], color="#9C27B0", ls="--", lw=2, label="Train|Test split")
ax1.fill_betweenx([gold["Close"].min(), gold["Close"].max()],
                    gold.index[0], gold.index[mid], alpha=0.03, color="#2196F3")
ax1.fill_betweenx([gold["Close"].min(), gold["Close"].max()],
                    gold.index[mid], gold.index[-1], alpha=0.03, color="#FF9800")
ax1.text(gold.index[mid // 2], gold["Close"].max() * 0.95, "TRAIN", fontsize=14, color="#2196F3", ha="center", fontweight="bold")
ax1.text(gold.index[mid + (len(gold) - mid) // 2], gold["Close"].max() * 0.95, "TEST", fontsize=14, color="#FF9800", ha="center", fontweight="bold")

# Plot test trades
test_trades = test_stats["_trades"]
if len(test_trades) > 0:
    wins = test_trades[test_trades["PnL"] > 0]
    losses = test_trades[test_trades["PnL"] <= 0]
    ax1.scatter(wins["EntryTime"], wins["EntryPrice"], marker="^", color="#4CAF50", s=15, zorder=5, alpha=0.6, label=f"Win ({len(wins)})")
    ax1.scatter(losses["EntryTime"], losses["EntryPrice"], marker="v", color="#F44336", s=15, zorder=5, alpha=0.6, label=f"Loss ({len(losses)})")
ax1.set_ylabel("Prijs (USD)")
ax1.legend(loc="upper left", fontsize=7, ncol=3)
ax1.grid(True, alpha=0.2)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

# 2: Equity per leverage
ax2 = fig.add_subplot(gs[1, :])
colors_lev = {5: "#4CAF50", 10: "#2196F3", 20: "#FF9800", 50: "#F44336"}
for lev in leverages:
    eq = lev_results[lev]["_equity_curve"]
    ax2.plot(eq.index, eq["Equity"], color=colors_lev[lev], lw=1, label=f"1:{lev} leverage")
# Geoptimaliseerde ook plotten
for lev in [10, 20]:
    eq = opt_results[lev]["_equity_curve"]
    ax2.plot(eq.index, eq["Equity"], color=colors_lev[lev], lw=2, ls="--", label=f"1:{lev} geoptim.")
ax2.axhline(10_000, color="#999", ls="--", lw=0.8)
ax2.set_ylabel("Equity ($)")
ax2.set_yscale("log")
ax2.set_title("Impact leverage — origineel vs geoptimaliseerd", fontsize=11)
ax2.legend(fontsize=7, ncol=3)
ax2.grid(True, alpha=0.2)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

# 3: Walk-forward equity
ax3 = fig.add_subplot(gs[2, :])
eq_train = train_opt["_equity_curve"]
eq_test = test_stats["_equity_curve"]
ax3.plot(eq_train.index, eq_train["Equity"], color="#2196F3", lw=1.2, label=f"Train: {train_opt['Return [%]']:+.0f}%")
ax3.plot(eq_test.index, eq_test["Equity"], color="#FF9800", lw=1.2, label=f"Test: {test_stats['Return [%]']:+.0f}%")
ax3.axhline(10_000, color="#999", ls="--", lw=0.8)
ax3.set_ylabel("Equity ($)")
ax3.set_title("Walk-Forward: getraind op eerste helft, getest op tweede", fontsize=11)
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.2)
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

# 4: Drawdowns per leverage
ax4 = fig.add_subplot(gs[3, :])
for lev in leverages:
    eq = lev_results[lev]["_equity_curve"]
    ax4.plot(eq.index, eq["DrawdownPct"] * 100, color=colors_lev[lev], lw=0.8, label=f"1:{lev}")
ax4.set_ylabel("Drawdown (%)")
ax4.set_title("Drawdown per leverage", fontsize=11)
ax4.legend(fontsize=7, ncol=4)
ax4.grid(True, alpha=0.2)
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

# 5: Rendement vs Leverage bar chart
ax5 = fig.add_subplot(gs[4, 0])
x = np.arange(len(leverages))
orig_rets = [lev_results[l]["Return [%]"] for l in leverages]
colors_bar = ["#4CAF50" if r > 0 else "#F44336" for r in orig_rets]
ax5.bar(x, orig_rets, color=colors_bar, width=0.5)
for i, (lev, ret) in enumerate(zip(leverages, orig_rets)):
    ax5.text(i, ret, f"{ret:+.0f}%", ha="center", va="bottom" if ret > 0 else "top", fontsize=9)
ax5.set_xticks(x)
ax5.set_xticklabels([f"1:{l}" for l in leverages])
ax5.axhline(0, color="#333", lw=0.8)
ax5.set_ylabel("Rendement (%)")
ax5.set_title("Rendement per leverage (origineel)", fontsize=10)
ax5.grid(True, alpha=0.2)

# 6: Walk-forward test trade P&L
ax6 = fig.add_subplot(gs[4, 1])
if len(test_trades) > 0:
    colors_t = ["#4CAF50" if p > 0 else "#F44336" for p in test_trades["PnL"]]
    ax6.bar(range(len(test_trades)), test_trades["PnL"], color=colors_t, width=0.8)
    avg = test_trades["PnL"].mean()
    ax6.axhline(avg, color="#2196F3", ls="--", lw=1, label=f"Gem: ${avg:+,.0f}")
    ax6.legend(fontsize=7)
ax6.axhline(0, color="#333", lw=0.8)
ax6.set_ylabel("P&L ($)")
ax6.set_title("Walk-forward TEST trades", fontsize=10)
ax6.grid(True, alpha=0.2)

plt.savefig("/workspace/strategy_d_longterm_results.png", dpi=150, bbox_inches="tight")
print("\nChart opgeslagen: /workspace/strategy_d_longterm_results.png")
print("\nKlaar!")
