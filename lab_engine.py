#!/usr/bin/env python3
"""
Strategy lab: the strategy menu and the backtest engine (2026-11-08).

A strategy is a dict of choices from MENU. The same code produces backtests, shadow (forward)
records and the live champion's target weights, so a strategy trades exactly as it was tested.

Mechanics, per trading day: positions move with close-to-close returns; margin interest is charged
on borrowed cash and a borrow fee on shorts; trailing stops close positions at the close; every
rebalance_days the book is reset to target weights at the close, paying COST_BPS on what trades.
Signals use data up to that close only.
"""
import numpy as np
import pandas as pd

from lab_data import universe_mask

MENU = {
    # legacy24 (the old hand-picked list) is hindsight-biased: kept in lab_data for reference, not searched
    'universe':       ['sp500', 'sp500_tech', 'sp400', 'etf_rotation', 'leveraged_etfs'],
    'signal':         ['mom_12_1', 'mom_6m', 'mom_3m', 'mom_1m', 'reversal_1w', 'rsi_oversold',
                       'near_52w_high', 'low_vol', 'risk_adj_mom', 'mom_blend', 'trend_mom'],
    'top_k':          [3, 5, 10, 15, 20],
    'rebalance_days': [5, 10, 21],
    'weighting':      ['equal', 'inv_vol'],
    'leverage':       [1.0, 1.5, 2.0],        # gross exposure as a multiple of equity
    'shorts':         [0.0, 0.25, 0.5],       # share of gross exposure that is short the lowest-ranked
    'regime':         ['none', 'spy200_cash'],
    'trail_stop':     [None, 0.10, 0.15, 0.25],
}
SMALL_UNIVERSES = {'etf_rotation', 'leveraged_etfs', 'legacy24'}   # top_k capped at a third of these

COST_BPS        = 5       # per side, spread + slippage
MARGIN_RATE     = 0.065   # yearly, on borrowed cash (Alpaca live margin rate; paper doesn't charge it)
BORROW_FEE      = 0.005   # yearly, on short value (easy-to-borrow names)
MIN_PRICE       = 5.0
MIN_DOLLAR_VOL  = 5e6     # 20-day median, stock universes only
RISK_FREE       = 0.04
MAX_GROSS_LIVE  = 1.95    # Reg-T overnight limit is 2x


def config_id(cfg):
    return '|'.join(f'{k}={cfg[k]}' for k in MENU)


def normalise(cfg, u):
    """Clamp choices that don't fit the universe; returns a new dict."""
    cfg = dict(cfg)
    if cfg['universe'] in SMALL_UNIVERSES:
        cfg['top_k'] = min(cfg['top_k'], max(2, len(u.get(cfg['universe'], {})) // 3))
    return cfg


# ------------------------------------------------------------
# Signals: higher score = more attractive to own (lowest = short candidates)
# ------------------------------------------------------------
def compute_signal(name, close):
    ret = close.pct_change(fill_method=None)
    if name == 'mom_12_1':
        s = close.shift(21) / close.shift(252) - 1
    elif name == 'mom_6m':
        s = close / close.shift(126) - 1
    elif name == 'mom_3m':
        s = close / close.shift(63) - 1
    elif name == 'mom_1m':
        s = close / close.shift(21) - 1
    elif name == 'reversal_1w':
        s = -(close / close.shift(5) - 1)
    elif name == 'rsi_oversold':
        d = close.diff()
        up, down = d.clip(lower=0).rolling(14).mean(), (-d.clip(upper=0)).rolling(14).mean()
        s = -(100 - 100 / (1 + up / down))
    elif name == 'near_52w_high':
        s = close / close.rolling(252, min_periods=200).max()
    elif name == 'low_vol':
        s = -ret.rolling(60).std()
    elif name == 'risk_adj_mom':
        s = (close / close.shift(126) - 1) / ret.rolling(126).std()
    elif name == 'mom_blend':
        # (2026-11-08) average cross-sectional rank of 3m, 6m and 12-1 momentum: less dependent
        # on any one lookback window than a single momentum signal
        parts = [close / close.shift(63) - 1, close / close.shift(126) - 1,
                 close.shift(21) / close.shift(252) - 1]
        s = sum(p.rank(axis=1, pct=True) for p in parts) / len(parts)
    elif name == 'trend_mom':
        # (2026-11-08) 6-month momentum, but only for stocks above their own 200-day average;
        # the rest are left out (NaN), so the book can hold fewer names or go to cash in a selloff
        s = (close / close.shift(126) - 1).where(close > close.rolling(200, min_periods=180).mean())
    else:
        raise ValueError(name)
    return s.astype('float32')


class Lab:
    """Holds price data plus cached signals/eligibility so many strategies can be tested cheaply."""

    def __init__(self, data, universes):
        self.data, self.u = data, universes
        self.close = data['Close']
        self.ret = self.close.pct_change(fill_method=None).fillna(0).astype('float32')
        self.vol60 = self.ret.rolling(60).std().astype('float32')
        dollar_vol = (self.close * data['Volume']).rolling(20).median()
        self.liquid = (self.close >= MIN_PRICE) & (dollar_vol >= MIN_DOLLAR_VOL)
        spy = self.close['SPY']
        self.spy_bull = (spy > spy.rolling(200).mean())
        self._signals, self._masks = {}, {}

    def signal(self, name):
        if name not in self._signals:
            self._signals[name] = compute_signal(name, self.close)
        return self._signals[name]

    def eligible(self, universe):
        if universe not in self._masks:
            m = universe_mask(self.u, universe, self.close)
            if universe not in SMALL_UNIVERSES:
                m &= self.liquid
            self._masks[universe] = m
        return self._masks[universe]

    # --------------------------------------------------------
    def weights_at(self, cfg, i, score_row=None):
        """Target weights (numpy vector over all symbols) at day index i."""
        n = self.close.shape[1]
        w = np.zeros(n, dtype='float64')
        if cfg['regime'] == 'spy200_cash' and not bool(self.spy_bull.iloc[i]):
            return w
        if score_row is None:
            score_row = self.signal(cfg['signal']).values[i]
        ok = self.eligible(cfg['universe']).values[i] & np.isfinite(score_row)
        idx = np.flatnonzero(ok)
        k = cfg['top_k']
        if len(idx) < (2 * k if cfg['shorts'] else k):
            return w
        order = idx[np.argsort(score_row[idx])]          # ascending
        longs, shorts = order[-k:], order[:k]
        vol = self.vol60.values[i]

        def leg(sel, gross):
            if gross <= 0:
                return np.zeros(len(sel))
            if cfg['weighting'] == 'inv_vol':
                iv = 1 / np.where(np.isfinite(vol[sel]) & (vol[sel] > 0), vol[sel], np.nan)
                iv = np.nan_to_num(iv, nan=np.nanmean(iv) if np.isfinite(iv).any() else 1.0)
                share = iv / iv.sum()
            else:
                share = np.full(len(sel), 1 / len(sel))
            return share * gross

        lev = cfg['leverage']
        w[longs] += leg(longs, lev * (1 - cfg['shorts']))
        if cfg['shorts']:
            w[shorts] -= leg(shorts, lev * cfg['shorts'])
        return w

    # --------------------------------------------------------
    def backtest(self, cfg, start, end=None, random_seed=None, init=100_000.0):
        """Daily equity curve (pd.Series) for cfg between start and end."""
        dates = self.close.index
        i0 = int(dates.searchsorted(pd.Timestamp(start)))
        i1 = len(dates) if end is None else int(dates.searchsorted(pd.Timestamp(end), side='right'))
        if i1 - i0 < 2:
            return pd.Series(dtype=float)
        R, C = self.ret.values, self.close.values
        rng = np.random.default_rng(random_seed) if random_seed is not None else None
        n = C.shape[1]
        pos = np.zeros(n)          # signed $ value per symbol
        mark = np.zeros(n)         # peak (long) / trough (short) close since entry, for trailing stops
        cash, stop = init, cfg['trail_stop']
        out = np.empty(i1 - i0)
        for t in range(i0, i1):
            if t > i0:
                pos *= 1 + R[t]
                if cash < 0:
                    cash += cash * MARGIN_RATE / 252
                cash -= -pos[pos < 0].sum() * BORROW_FEE / 252
            c = C[t]
            if stop:
                held = pos != 0
                mark = np.where(held & (pos > 0), np.fmax(mark, c), mark)
                mark = np.where(held & (pos < 0), np.fmin(mark, c), mark)
                hit = held & (((pos > 0) & (c <= mark * (1 - stop))) | ((pos < 0) & (c >= mark * (1 + stop))))
                if hit.any():
                    cash += pos[hit].sum() - np.abs(pos[hit]).sum() * COST_BPS / 1e4
                    pos[hit] = 0
            if (t - i0) % cfg['rebalance_days'] == 0:
                equity = cash + pos.sum()
                scores = rng.random(n) if rng is not None else None
                target = self.weights_at(cfg, t, scores) * max(equity, 0)
                trade = target - pos
                cash -= trade.sum() + np.abs(trade).sum() * COST_BPS / 1e4
                new = (np.sign(target) != np.sign(pos)) & (target != 0)
                mark = np.where(new, c, mark)
                pos = target
            out[t - i0] = cash + pos.sum()
            if out[t - i0] <= 0:      # wiped out
                out[t - i0:] = 0
                break
        return pd.Series(out, index=dates[i0:i1])

    def spy_curve(self, start, end=None, init=100_000.0):
        s = self.close['SPY'].loc[pd.Timestamp(start):end]
        return s / s.iloc[0] * init


def metrics(curve, spy_curve):
    """Summary stats for an equity curve, with SPY over the same days."""
    if len(curve) < 2 or curve.iloc[0] <= 0:
        return {}
    years = (curve.index[-1] - curve.index[0]).days / 365.25
    total = curve.iloc[-1] / curve.iloc[0] - 1
    spy_total = spy_curve.iloc[-1] / spy_curve.iloc[0] - 1
    daily = curve.pct_change().dropna()
    vol = float(daily.std() * np.sqrt(252))
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else -1.0
    spy_cagr = (1 + spy_total) ** (1 / years) - 1 if years > 0 else 0.0
    dd = float((curve / curve.cummax() - 1).min())
    yearly = curve.resample('YE').last().pct_change().dropna()
    spy_yearly = spy_curve.resample('YE').last().pct_change().dropna()
    both = pd.concat([yearly, spy_yearly], axis=1).dropna()
    return {
        'start': f'{curve.index[0]:%Y-%m-%d}', 'end': f'{curve.index[-1]:%Y-%m-%d}',
        'total_return': round(float(total), 4), 'spy_total_return': round(float(spy_total), 4),
        'cagr': round(float(cagr), 4), 'spy_cagr': round(float(spy_cagr), 4),
        'excess_cagr': round(float(cagr - spy_cagr), 4),
        'vol': round(vol, 4),
        'sharpe': round(float((daily.mean() * 252 - RISK_FREE) / vol), 2) if vol > 0 else 0.0,
        'max_drawdown': round(dd, 4),
        'years_beating_spy': f'{int((both.iloc[:, 0] > both.iloc[:, 1]).sum())}/{len(both)}' if len(both) else '-',
    }
