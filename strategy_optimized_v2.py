"""
Strategy: Geoptimaliseerde Uurlijkse Scanner V2
================================================
Basis: trail 2.5%, TP $80, drempel 2.0, max 1 positie, 1:10 leverage
Verbeteringen:
  1. Volatiliteitsfilter — niet traden in te wilde markten
  2. Sessiefilter — alleen traden in actieve uren (London/NY overlap)
  3. Trendsterkte — hogere drempel in zwakke trends
  4. Dynamische TP/SL — aangepast aan huidige volatiliteit
  5. Cooldown — niet direct opnieuw kopen na een verlies
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


def ATR(high, low, close, period=14):
    """Average True Range — maat voor volatiliteit."""
    h = np.array(high, dtype=float)
    l = np.array(low, dtype=float)
    c = np.array(close, dtype=float)
    tr = np.zeros(len(c))
    for i in range(1, len(c)):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean()
    return atr


def RSI(close, period=14):
    """Relative Strength Index."""
    c = pd.Series(close, dtype=float)
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ── V1: Basis (de geoptimaliseerde van vorige run) ───────────────
class V1_Basis(Strategy):
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
                self.buy(size=0.90, sl=price * (1 - self.trail_pct / 100), tp=price + self.tp_dollars)
        if score < self.close_threshold and len(self.trades) > 0:
            self.trades[0].close()


# ── V2: + Volatiliteitsfilter + RSI ──────────────────────────────
class V2_VolFilter(Strategy):
    lookback = 24
    buy_threshold = 2.0
    close_threshold = -5.0
    trail_pct = 2.5
    tp_dollars = 80.0
    max_positions = 1
    atr_mult_max = 2.5    # skip als ATR > 2.5x gemiddelde (te volatiel)
    rsi_min = 40           # niet kopen als RSI te laag (oversold bounce risk)
    rsi_max = 75           # niet kopen als RSI te hoog (overbought)

    def init(self):
        close = pd.Series(self.data.Close, dtype=float)
        self.momentum = self.I(compute_momentum, close, self.lookback)
        self.ema_fast = self.I(EMA, close, 8)
        self.ema_slow = self.I(EMA, close, 24)
        self.atr = self.I(ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.atr_avg = self.I(EMA, self.I(ATR, self.data.High, self.data.Low, self.data.Close, 14), 100)
        self.rsi = self.I(RSI, self.data.Close, 14)

    def next(self):
        price = self.data.Close[-1]
        score = self.momentum[-1]
        for trade in self.trades:
            new_sl = price * (1 - self.trail_pct / 100)
            if trade.sl is None or new_sl > trade.sl:
                trade.sl = new_sl

        vol_ok = self.atr_avg[-1] > 0 and (self.atr[-1] / self.atr_avg[-1]) < self.atr_mult_max
        rsi_ok = self.rsi_min < self.rsi[-1] < self.rsi_max

        if score > self.buy_threshold and len(self.trades) < self.max_positions:
            if self.ema_fast[-1] > self.ema_slow[-1] and vol_ok and rsi_ok:
                self.buy(size=0.90, sl=price * (1 - self.trail_pct / 100), tp=price + self.tp_dollars)
        if score < self.close_threshold and len(self.trades) > 0:
            self.trades[0].close()


# ── V3: + Dynamische TP/SL op basis van ATR ──────────────────────
class V3_DynamicSLTP(Strategy):
    lookback = 24
    buy_threshold = 2.0
    close_threshold = -5.0
    trail_atr_mult = 2.0   # trailing stop = 2x ATR
    tp_atr_mult = 4.0      # take profit = 4x ATR
    max_positions = 1
    atr_mult_max = 2.5
    rsi_min = 40
    rsi_max = 75

    def init(self):
        close = pd.Series(self.data.Close, dtype=float)
        self.momentum = self.I(compute_momentum, close, self.lookback)
        self.ema_fast = self.I(EMA, close, 8)
        self.ema_slow = self.I(EMA, close, 24)
        self.atr = self.I(ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.atr_avg = self.I(EMA, self.I(ATR, self.data.High, self.data.Low, self.data.Close, 14), 100)
        self.rsi = self.I(RSI, self.data.Close, 14)

    def next(self):
        price = self.data.Close[-1]
        score = self.momentum[-1]
        current_atr = self.atr[-1]

        for trade in self.trades:
            new_sl = price - current_atr * self.trail_atr_mult
            if trade.sl is None or new_sl > trade.sl:
                trade.sl = new_sl

        vol_ok = self.atr_avg[-1] > 0 and (current_atr / self.atr_avg[-1]) < self.atr_mult_max
        rsi_ok = self.rsi_min < self.rsi[-1] < self.rsi_max

        if score > self.buy_threshold and len(self.trades) < self.max_positions:
            if self.ema_fast[-1] > self.ema_slow[-1] and vol_ok and rsi_ok:
                sl = price - current_atr * self.trail_atr_mult
                tp = price + current_atr * self.tp_atr_mult
                self.buy(size=0.90, sl=sl, tp=tp)
        if score < self.close_threshold and len(self.trades) > 0:
            self.trades[0].close()


# ── V4: + Cooldown na verlies + sessiefilter ─────────────────────
class V4_Full(Strategy):
    lookback = 24
    buy_threshold = 2.0
    close_threshold = -5.0
    trail_atr_mult = 2.0
    tp_atr_mult = 4.0
    max_positions = 1
    atr_mult_max = 2.5
    rsi_min = 40
    rsi_max = 75
    cooldown_bars = 6       # 6 uur wachten na verlies

    def init(self):
        close = pd.Series(self.data.Close, dtype=float)
        self.momentum = self.I(compute_momentum, close, self.lookback)
        self.ema_fast = self.I(EMA, close, 8)
        self.ema_slow = self.I(EMA, close, 24)
        self.ema_200 = self.I(EMA, close, 200)
        self.atr = self.I(ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.atr_avg = self.I(EMA, self.I(ATR, self.data.High, self.data.Low, self.data.Close, 14), 100)
        self.rsi = self.I(RSI, self.data.Close, 14)
        self.bars_since_loss = 999
        self.last_n_trades = 0

    def next(self):
        price = self.data.Close[-1]
        score = self.momentum[-1]
        current_atr = self.atr[-1]

        # Track cooldown
        if len(self.closed_trades) > self.last_n_trades:
            last_trade = self.closed_trades[-1]
            if last_trade.pl < 0:
                self.bars_since_loss = 0
            self.last_n_trades = len(self.closed_trades)
        self.bars_since_loss += 1

        # Update trailing stops
        for trade in self.trades:
            new_sl = price - current_atr * self.trail_atr_mult
            if trade.sl is None or new_sl > trade.sl:
                trade.sl = new_sl

        # Filters
        vol_ok = self.atr_avg[-1] > 0 and (current_atr / self.atr_avg[-1]) < self.atr_mult_max
        rsi_ok = self.rsi_min < self.rsi[-1] < self.rsi_max
        trend_ok = price > self.ema_200[-1]  # boven 200 EMA = lange termijn uptrend
        cooldown_ok = self.bars_since_loss > self.cooldown_bars
        session_ok = True
        if hasattr(self.data.index[-1], 'hour'):
            hour = self.data.index[-1].hour
            session_ok = 7 <= hour <= 20  # actieve markturen (UTC)

        if (score > self.buy_threshold
            and len(self.trades) < self.max_positions
            and self.ema_fast[-1] > self.ema_slow[-1]
            and vol_ok and rsi_ok and trend_ok and cooldown_ok and session_ok):
            sl = price - current_atr * self.trail_atr_mult
            tp = price + current_atr * self.tp_atr_mult
            self.buy(size=0.90, sl=sl, tp=tp)

        if score < self.close_threshold and len(self.trades) > 0:
            self.trades[0].close()


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
print(f"   {gold.index[0].strftime('%Y-%m-%d')} -> {gold.index[-1].strftime('%Y-%m-%d')}\n")

bt_kwargs = dict(cash=10_000, commission=0.0002, margin=1/10, exclusive_orders=False, hedging=True)


# ── Run alle versies ─────────────────────────────────────────────
versions = {
    "V1 Basis": (V1_Basis, {}),
    "V2 +VolFilter+RSI": (V2_VolFilter, {}),
    "V3 +Dynamische SL/TP": (V3_DynamicSLTP, {}),
    "V4 +Cooldown+Sessie+EMA200": (V4_Full, {}),
}

results = {}
for name, (cls, extra) in versions.items():
    print(f"Running {name}...")
    bt = Backtest(gold, cls, **bt_kwargs)
    results[name] = bt.run()

# ── Optimaliseer V3 en V4 ───────────────────────────────────────
print("\nOptimalisatie V3...")
bt_v3 = Backtest(gold, V3_DynamicSLTP, **bt_kwargs)
v3_opt = bt_v3.optimize(
    trail_atr_mult=[1.5, 2.0, 2.5, 3.0],
    tp_atr_mult=[3.0, 4.0, 5.0, 6.0, 8.0],
    buy_threshold=[1.0, 2.0, 3.0, 5.0],
    atr_mult_max=[2.0, 2.5, 3.0],
    rsi_max=[70, 75, 80],
    maximize="Equity Final [$]",
    max_tries=600,
)
results["V3 geoptimaliseerd"] = v3_opt

print("Optimalisatie V4...")
bt_v4 = Backtest(gold, V4_Full, **bt_kwargs)
v4_opt = bt_v4.optimize(
    trail_atr_mult=[1.5, 2.0, 2.5, 3.0],
    tp_atr_mult=[3.0, 4.0, 5.0, 6.0, 8.0],
    buy_threshold=[1.0, 2.0, 3.0, 5.0],
    cooldown_bars=[3, 6, 12],
    rsi_max=[70, 75, 80],
    maximize="Equity Final [$]",
    max_tries=600,
)
results["V4 geoptimaliseerd"] = v4_opt


# ── Walk-forward V4 ─────────────────────────────────────────────
print("\nWalk-forward V4...")
mid = len(gold) // 2
train_data = gold.iloc[:mid]
test_data = gold.iloc[mid:]

bt_wf = Backtest(train_data, V4_Full, **bt_kwargs)
wf_train = bt_wf.optimize(
    trail_atr_mult=[1.5, 2.0, 2.5, 3.0],
    tp_atr_mult=[3.0, 4.0, 5.0, 6.0, 8.0],
    buy_threshold=[1.0, 2.0, 3.0, 5.0],
    cooldown_bars=[3, 6, 12],
    rsi_max=[70, 75, 80],
    maximize="Equity Final [$]",
    max_tries=600,
)
wf_st = wf_train._strategy

class V4_WalkForward(V4_Full):
    trail_atr_mult = wf_st.trail_atr_mult
    tp_atr_mult = wf_st.tp_atr_mult
    buy_threshold = wf_st.buy_threshold
    cooldown_bars = wf_st.cooldown_bars
    rsi_max = wf_st.rsi_max

bt_wf_test = Backtest(test_data, V4_WalkForward, **bt_kwargs)
wf_test = bt_wf_test.run()
results["V4 walk-forward TRAIN"] = wf_train
results["V4 walk-forward TEST"] = wf_test


# ── Resultaten ───────────────────────────────────────────────────
print(f"\n{'=' * 105}")
print(f"  VERGELIJKING ALLE VERSIES — {total_days} dagen")
print(f"{'=' * 105}")
print(f"  {'Versie':<35} {'Trades':>7} {'Win%':>6} {'Return':>12} {'MaxDD':>8} {'PF':>6} {'Sharpe':>8} {'Gem.duur':>10}")
print(f"  {'-'*35} {'-'*7} {'-'*6} {'-'*12} {'-'*8} {'-'*6} {'-'*8} {'-'*10}")

for name, s in results.items():
    wr = f"{s['Win Rate [%]']:.0f}%" if pd.notna(s['Win Rate [%]']) else "N/A"
    pf = f"{s['Profit Factor']:.2f}" if pd.notna(s['Profit Factor']) else "N/A"
    sr = f"{s['Sharpe Ratio']:.2f}" if pd.notna(s['Sharpe Ratio']) and abs(s['Sharpe Ratio']) < 1000 else "N/A"
    dur = f"{s['Avg. Trade Duration']}" if pd.notna(s['# Trades']) and s['# Trades'] > 0 else "N/A"
    dur_short = str(dur).split(",")[0] if "," in str(dur) else str(dur)
    print(f"  {name:<35} {int(s['# Trades']):>7} {wr:>6} {s['Return [%]']:>+11.1f}% {s['Max. Drawdown [%]']:>7.1f}% {pf:>6} {sr:>8} {dur_short:>10}")

print(f"{'=' * 105}")

# Print optimale params
for name in ["V3 geoptimaliseerd", "V4 geoptimaliseerd"]:
    st = results[name]._strategy
    print(f"\n  {name} params:")
    for attr in ["trail_atr_mult", "tp_atr_mult", "buy_threshold", "close_threshold",
                 "atr_mult_max", "rsi_min", "rsi_max", "cooldown_bars", "max_positions"]:
        if hasattr(st, attr):
            print(f"    {attr}: {getattr(st, attr)}")

print(f"\n  V4 walk-forward params (getraind op eerste helft):")
for attr in ["trail_atr_mult", "tp_atr_mult", "buy_threshold", "cooldown_bars", "rsi_max"]:
    if hasattr(wf_st, attr):
        print(f"    {attr}: {getattr(wf_st, attr)}")


# ── Charts ───────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 24))
gs = fig.add_gridspec(6, 2, height_ratios=[2.5, 1.5, 1.2, 1.2, 1.2, 1.2], hspace=0.35, wspace=0.25)
fig.suptitle(f'XAUUSD Strategy Evolutie V1→V4 — {total_days} dagen',
             fontsize=17, fontweight="bold", y=0.99)

# 1: Prijs + V4 opt trades
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(gold.index, gold["Close"], color="#888", lw=0.4, alpha=0.7, label="XAUUSD 1H")
ema200 = gold["Close"].ewm(span=200, adjust=False).mean()
ax1.plot(gold.index, ema200, color="#9C27B0", lw=0.8, alpha=0.6, label="EMA 200")

best_name = max(
    [(n, s) for n, s in results.items() if "walk" not in n.lower()],
    key=lambda x: x[1]["Equity Final [$]"]
)[0]
best_trades = results[best_name]["_trades"]
if len(best_trades) > 0:
    wins = best_trades[best_trades["PnL"] > 0]
    losses = best_trades[best_trades["PnL"] <= 0]
    ax1.scatter(wins["EntryTime"], wins["EntryPrice"], marker="^", color="#4CAF50", s=15, zorder=5, alpha=0.5)
    ax1.scatter(losses["EntryTime"], losses["EntryPrice"], marker="v", color="#F44336", s=15, zorder=5, alpha=0.5)
ax1.set_title(f"Beste: {best_name} — {len(best_trades)} trades", fontsize=11)
ax1.set_ylabel("Prijs (USD)")
ax1.legend(loc="upper left", fontsize=7)
ax1.grid(True, alpha=0.2)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

# 2: Equity alle versies
ax2 = fig.add_subplot(gs[1, :])
colors_v = ["#F44336", "#FF9800", "#2196F3", "#4CAF50", "#9C27B0", "#00BCD4", "#795548", "#607D8B"]
for i, (name, s) in enumerate(results.items()):
    if "walk" not in name.lower() or "TEST" in name:
        eq = s["_equity_curve"]
        lw = 2 if "geopt" in name.lower() or "TEST" in name else 0.8
        ls = "--" if "TEST" in name else "-"
        ax2.plot(eq.index, eq["Equity"], color=colors_v[i % len(colors_v)], lw=lw, ls=ls, label=name)
ax2.axhline(10_000, color="#999", ls="--", lw=0.8)
ax2.set_ylabel("Equity ($)")
ax2.set_yscale("log")
ax2.set_title("Equity vergelijking alle versies", fontsize=11)
ax2.legend(fontsize=6, ncol=2)
ax2.grid(True, alpha=0.2)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

# 3: Drawdown vergelijking
ax3 = fig.add_subplot(gs[2, :])
for i, (name, s) in enumerate(results.items()):
    if "TRAIN" not in name:
        eq = s["_equity_curve"]
        ax3.plot(eq.index, eq["DrawdownPct"] * 100, color=colors_v[i % len(colors_v)], lw=0.8, label=name)
ax3.set_ylabel("Drawdown (%)")
ax3.legend(fontsize=6, ncol=2)
ax3.grid(True, alpha=0.2)
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

# 4: Win rate per versie
ax4 = fig.add_subplot(gs[3, 0])
names_bar = [n for n in results if "TRAIN" not in n]
winrates = [results[n]["Win Rate [%]"] if pd.notna(results[n]["Win Rate [%]"]) else 0 for n in names_bar]
colors_bar = ["#4CAF50" if w >= 50 else "#FF9800" if w >= 40 else "#F44336" for w in winrates]
bars = ax4.bar(range(len(names_bar)), winrates, color=colors_bar, width=0.6)
ax4.set_xticks(range(len(names_bar)))
ax4.set_xticklabels([n.replace(" ", "\n")[:20] for n in names_bar], fontsize=6)
ax4.axhline(50, color="#999", ls="--", lw=0.8)
for bar, val in zip(bars, winrates):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.0f}%", ha="center", fontsize=7)
ax4.set_ylabel("Win Rate (%)")
ax4.set_title("Win Rate per versie", fontsize=10)
ax4.grid(True, alpha=0.2)

# 5: Profit Factor per versie
ax5 = fig.add_subplot(gs[3, 1])
pfs = [results[n]["Profit Factor"] if pd.notna(results[n]["Profit Factor"]) else 0 for n in names_bar]
colors_pf = ["#4CAF50" if p > 1.5 else "#FF9800" if p > 1 else "#F44336" for p in pfs]
bars = ax5.bar(range(len(names_bar)), pfs, color=colors_pf, width=0.6)
ax5.set_xticks(range(len(names_bar)))
ax5.set_xticklabels([n.replace(" ", "\n")[:20] for n in names_bar], fontsize=6)
ax5.axhline(1.0, color="#F44336", ls="--", lw=0.8, label="Break-even")
ax5.axhline(1.5, color="#4CAF50", ls="--", lw=0.8, label="Goed")
for bar, val in zip(bars, pfs):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f"{val:.2f}", ha="center", fontsize=7)
ax5.set_ylabel("Profit Factor")
ax5.set_title("Profit Factor per versie", fontsize=10)
ax5.legend(fontsize=7)
ax5.grid(True, alpha=0.2)

# 6: P&L distributie beste versie
ax6 = fig.add_subplot(gs[4, 0])
if len(best_trades) > 0:
    pnl = best_trades["PnL"].values
    ax6.hist(pnl[pnl > 0], bins=30, color="#4CAF50", alpha=0.7, label=f"Winst ({(pnl>0).sum()})")
    ax6.hist(pnl[pnl <= 0], bins=30, color="#F44336", alpha=0.7, label=f"Verlies ({(pnl<=0).sum()})")
    ax6.axvline(pnl.mean(), color="#2196F3", ls="--", lw=1.5, label=f"Gem: ${pnl.mean():+,.0f}")
    ax6.legend(fontsize=7)
ax6.set_xlabel("P&L ($)")
ax6.set_ylabel("Aantal")
ax6.set_title(f"P&L distributie — {best_name}", fontsize=10)
ax6.grid(True, alpha=0.2)

# 7: Cumulatieve P&L
ax7 = fig.add_subplot(gs[4, 1])
for i, (name, s) in enumerate(results.items()):
    if "TRAIN" not in name and s["# Trades"] > 0:
        cum = s["_trades"]["PnL"].cumsum()
        ax7.plot(range(len(cum)), cum, color=colors_v[i % len(colors_v)], lw=1, label=name)
ax7.axhline(0, color="#999", ls="--", lw=0.8)
ax7.set_xlabel("Trade #")
ax7.set_ylabel("Cum. P&L ($)")
ax7.set_title("Cumulatieve P&L", fontsize=10)
ax7.legend(fontsize=5, ncol=2)
ax7.grid(True, alpha=0.2)

# 8: Maandelijkse P&L beste
ax8 = fig.add_subplot(gs[5, :])
if len(best_trades) > 0:
    t = best_trades.copy()
    t["Month"] = pd.to_datetime(t["EntryTime"]).dt.to_period("M")
    monthly = t.groupby("Month")["PnL"].sum()
    months = monthly.index.astype(str)
    colors_m = ["#4CAF50" if v > 0 else "#F44336" for v in monthly.values]
    ax8.bar(range(len(months)), monthly.values, color=colors_m, width=0.7)
    ax8.set_xticks(range(len(months)))
    ax8.set_xticklabels(months, rotation=45, fontsize=7)
    ax8.axhline(0, color="#333", lw=0.8)
    avg_m = monthly.mean()
    ax8.axhline(avg_m, color="#2196F3", ls="--", lw=1, label=f"Gem: ${avg_m:+,.0f}/mnd")
    ax8.legend(fontsize=8)
ax8.set_ylabel("P&L ($)")
ax8.set_title(f"Maandelijkse P&L — {best_name}", fontsize=10)
ax8.grid(True, alpha=0.2)

plt.savefig("/workspace/strategy_v2_evolution.png", dpi=150, bbox_inches="tight")
print(f"\nChart opgeslagen: /workspace/strategy_v2_evolution.png")
print("\nKlaar!")
