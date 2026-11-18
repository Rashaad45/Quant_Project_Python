"""
PROJECT 1 - Ornstein-Uhlenbeck pairs trading with half-life position sizing
=========================================================================

Steps
 1. Pull two (hopefully) cointegrated price series.
 2. Engle-Granger step 1: OLS  log(Y) = c + beta * log(X) + S   -> hedge ratio beta.
 3. Engle-Granger step 2: ADF test on the spread S  (is it stationary?).
 4. Fit the OU process  dS = theta (mu - S) dt + sigma dW  with ONE OLS:
        dS_t = a + b * S_{t-1} + eps_t
    theta = -ln(1 + b) / dt,  mu = -a / b,  half-life = ln(2) / theta
 5. Trade the z-score of the spread; use the half-life for
      - the look-back window of the z-score,
      - the maximum holding period (time stop),
      - the position size (longer half-life -> smaller size).
 6. Walk-forward: parameters estimated on the first 60% of data,
    trading evaluated on the remaining 40% (out of sample).

Run:  python p1_ou_pairs.py            (default pair KO / PEP)
      python p1_ou_pairs.py GLD GDX    (any two tickers)
"""
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import get_prices, save_fig

warnings.filterwarnings("ignore", category=FutureWarning)

try:  # statsmodels is optional (some Windows policies block its DLLs)
    from statsmodels.tsa.stattools import adfuller
except Exception:
    adfuller = None

DT = 1.0  # time step = 1 trading day


# ----------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------
def synthetic_pair(n=2500, seed=0):
    """X is a random walk, Y = 1.2 X + OU noise  -> cointegrated by construction."""
    rng = np.random.default_rng(seed)
    logx = np.cumsum(rng.normal(0, 0.01, n)) + np.log(50)
    s = np.zeros(n)
    theta, sig = 0.05, 0.01                           # true half-life = 13.9 days
    for t in range(1, n):
        s[t] = s[t - 1] - theta * s[t - 1] + sig * rng.normal()
    logy = 0.3 + 1.2 * logx + s
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame({"Y": np.exp(logy), "X": np.exp(logx)}, index=idx)


# ----------------------------------------------------------------------------
# OLS from scratch:  b_hat = (X'X)^-1 X'y
# ----------------------------------------------------------------------------
def ols(y, X):
    """Returns coefficients, t-statistics and residuals.  X must include a constant column."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    XtX_inv = np.linalg.inv(X.T @ X)
    coef = XtX_inv @ X.T @ y
    resid = y - X @ coef
    s2 = resid @ resid / (len(y) - X.shape[1])      # residual variance
    se = np.sqrt(np.diag(s2 * XtX_inv))
    return coef, coef / se, resid


def add_const(x):
    x = np.asarray(x, float)
    return np.column_stack([np.ones(len(x)), x])


# ----------------------------------------------------------------------------
# 2-3. Engle-Granger cointegration
# ----------------------------------------------------------------------------
def hedge_ratio(logy, logx):
    """OLS: logy = c + beta*logx + e.  Returns (c, beta)."""
    coef, _, _ = ols(logy, add_const(logx))
    return coef[0], coef[1]


def adf_test(x, lags=1):
    """
    Augmented Dickey-Fuller regression (with constant):
        dx_t = a + g x_{t-1} + sum_j phi_j dx_{t-j} + e_t
    H0: g = 0 (unit root, NOT mean reverting).  Statistic = t-stat of g.
    Critical values: plain ADF (MacKinnon) for a series you observe directly,
    Engle-Granger values (more negative) when x is a residual from an
    estimated cointegrating regression with 2 variables.
    """
    x = np.asarray(x, float)
    dx = np.diff(x)
    rows = len(dx) - lags
    cols = [x[lags:-1]] + [dx[lags - j:len(dx) - j] for j in range(1, lags + 1)]
    coef, t, _ = ols(dx[lags:], add_const(np.column_stack(cols)))
    return t[1]


ADF_CRIT = {"1%": -3.43, "5%": -2.86, "10%": -2.57}
EG_CRIT = {"1%": -3.90, "5%": -3.34, "10%": -3.04}   # 2 variables, constant


# ----------------------------------------------------------------------------
# 4. OU fit by OLS on the discretised process (the "one-line OLS")
# ----------------------------------------------------------------------------
def fit_ou(spread, dt=DT):
    """
    Exact discretisation of the OU process:
        S_t = mu (1 - e^{-theta dt}) + e^{-theta dt} S_{t-1} + eta_t
    Writing it in differences:  dS_t = a + b S_{t-1} + eta_t
        b = e^{-theta dt} - 1      ->  theta = -ln(1+b)/dt
        a = mu (1 - e^{-theta dt}) ->  mu    = -a/b
        Var(eta) = sigma^2 (1 - e^{-2 theta dt}) / (2 theta)
    """
    s = np.asarray(spread)
    ds, s_lag = np.diff(s), s[:-1]
    (a, b), tstats, resid = ols(ds, add_const(s_lag))
    if b >= 0:
        raise ValueError("b >= 0: spread is not mean reverting on this sample")
    theta = -np.log(1 + b) / dt
    mu = -a / b
    sd_eta = np.std(resid, ddof=2)
    sigma = sd_eta * np.sqrt(2 * theta / (1 - np.exp(-2 * theta * dt)))
    half_life = np.log(2) / theta
    sigma_eq = sigma / np.sqrt(2 * theta)            # stationary std of S
    return dict(a=a, b=b, b_tstat=tstats[1], theta=theta, mu=mu,
                sigma=sigma, half_life=half_life, sigma_eq=sigma_eq)


# ----------------------------------------------------------------------------
# 5. Strategy
# ----------------------------------------------------------------------------
def backtest(logy, logx, beta, ou, entry=2.0, exit_=0.5, stop=4.0,
             risk_per_trade=0.01, cost_bps=5.0):
    """
    Z-score strategy.  Position is in 'spread units': +1 = long 1 unit of Y,
    short beta units of X (in log/return terms).

    Position size  (fraction of capital) =
          risk_per_trade / (sigma_eq * sqrt(1 - e^{-2 theta * H}))
    where H = holding horizon = half-life.  This is the std of the spread move
    you can expect while you hold the trade -> long half-life -> bigger
    expected adverse move -> smaller size.
    """
    hl = ou["half_life"]
    window = int(np.clip(round(hl), 5, 120))           # z-score look-back
    max_hold = int(np.ceil(3 * hl))                    # time stop
    spread = logy - beta * logx
    z = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()

    horizon_sd = ou["sigma_eq"] * np.sqrt(1 - np.exp(-2 * ou["theta"] * hl))
    size = min(risk_per_trade / horizon_sd, 3.0)       # cap leverage at 3x

    pos = np.zeros(len(z))
    held = 0
    for t in range(1, len(z)):
        p = pos[t - 1]
        zt = z.iloc[t]
        if np.isnan(zt):
            continue
        if p == 0:
            if zt > entry:
                p, held = -size, 0                     # spread too high -> short it
            elif zt < -entry:
                p, held = +size, 0
        else:
            held += 1
            if abs(zt) < exit_ or abs(zt) > stop or held >= max_hold:
                p = 0
        pos[t] = p

    pos = pd.Series(pos, index=z.index)
    spread_ret = spread.diff()                         # log-return of the spread
    gross = pos.shift(1) * spread_ret
    turnover = pos.diff().abs() * (1 + abs(beta))      # both legs trade
    net = gross - turnover * cost_bps / 1e4
    return pd.DataFrame({"spread": spread, "z": z, "pos": pos, "ret": net.fillna(0)}), window, max_hold, size


def perf(r):
    ann = 252
    sharpe = r.mean() / r.std() * np.sqrt(ann) if r.std() > 0 else 0.0
    eq = r.cumsum()
    mdd = (eq - eq.cummax()).min()
    return dict(total_logret=eq.iloc[-1], ann_ret=r.mean() * ann,
                ann_vol=r.std() * np.sqrt(ann), sharpe=sharpe, max_dd=mdd)


def main():
    tick = sys.argv[1:3] if len(sys.argv) >= 3 else ["KO", "PEP"]
    px = get_prices(tick, start="2015-01-01")
    if px is None:
        px = synthetic_pair()
        ny, nx = "Y", "X"
    else:
        ny, nx = tick[0], tick[1]
    logy, logx = np.log(px[ny]), np.log(px[nx])

    # ---- in-sample / out-of-sample split ------------------------------------
    cut = int(len(px) * 0.6)
    ly_is, lx_is = logy.iloc[:cut], logx.iloc[:cut]

    c, beta = hedge_ratio(ly_is, lx_is)
    spread_is = ly_is - beta * lx_is
    eg_stat = adf_test(spread_is, lags=1)
    ou = fit_ou(spread_is)

    print("\n=== In-sample estimation ===")
    print(f"pair: {ny} vs {nx}   train {ly_is.index[0].date()} -> {ly_is.index[-1].date()}")
    print(f"hedge ratio beta = {beta:.4f}   intercept c = {c:.4f}")
    print(f"Engle-Granger ADF stat on spread = {eg_stat:.3f}  (critical values {EG_CRIT})")
    if adfuller is not None:
        print(f"  statsmodels adfuller p-value (plain ADF table) = {adfuller(spread_is)[1]:.4f}")
    print(f"OLS dS = a + b S_lag:  a={ou['a']:.5f}  b={ou['b']:.5f} (t={ou['b_tstat']:.2f})")
    print(f"theta={ou['theta']:.4f}/day  mu={ou['mu']:.4f}  sigma={ou['sigma']:.4f}")
    print(f"HALF-LIFE = ln2/theta = {ou['half_life']:.1f} trading days")
    if eg_stat > EG_CRIT["5%"]:
        print("WARNING: cointegration not significant at 5% - treat results with care.")

    # ---- out-of-sample trading ---------------------------------------------
    bt, window, max_hold, size = backtest(logy, logx, beta, ou)
    oos = bt.iloc[cut:]
    print("\n=== Strategy (half-life driven) ===")
    print(f"z-score window = {window} d, time stop = {max_hold} d, size = {size:.2f}x capital")
    for name, part in [("in-sample", bt.iloc[:cut]), ("out-of-sample", oos)]:
        p = perf(part["ret"])
        print(f"{name:14s} ann_ret={p['ann_ret']:+.2%} vol={p['ann_vol']:.2%} "
              f"Sharpe={p['sharpe']:.2f} maxDD={p['max_dd']:.2%}")

    # ---- simulate the fitted OU to show what the model believes -------------
    rng = np.random.default_rng(1)
    n = 250
    sims = np.zeros((20, n))
    sims[:, 0] = ou["mu"] + 2 * ou["sigma_eq"]
    e = np.exp(-ou["theta"])
    sd = ou["sigma"] * np.sqrt((1 - e ** 2) / (2 * ou["theta"]))
    for t in range(1, n):
        sims[:, t] = ou["mu"] + (sims[:, t - 1] - ou["mu"]) * e + sd * rng.normal(size=20)

    # ---- plots --------------------------------------------------------------
    fig, ax = plt.subplots(4, 1, figsize=(11, 13))
    ax[0].plot(bt["spread"], lw=0.8)
    ax[0].axvline(bt.index[cut], color="k", ls="--", label="train | test")
    ax[0].axhline(ou["mu"], color="r", lw=1, label="OU mean mu")
    ax[0].set_title(f"Spread  log({ny}) - {beta:.2f} log({nx})"); ax[0].legend()

    ax[1].scatter(spread_is.shift(1).iloc[1:], spread_is.diff().iloc[1:], s=3, alpha=0.4)
    xs = np.linspace(spread_is.min(), spread_is.max(), 50)
    ax[1].plot(xs, ou["a"] + ou["b"] * xs, "r", label=f"OLS slope b={ou['b']:.4f}")
    ax[1].set_xlabel("S(t-1)"); ax[1].set_ylabel("dS(t)")
    ax[1].set_title(f"Estimating theta by OLS  ->  half-life = {ou['half_life']:.1f} days"); ax[1].legend()

    ax[2].plot(sims.T, lw=0.6, alpha=0.6)
    ax[2].plot(ou["mu"] + 2 * ou["sigma_eq"] * np.exp(-ou["theta"] * np.arange(n)), "k", lw=2,
               label="expected path  mu + (S0-mu) e^{-theta t}")
    ax[2].axvline(ou["half_life"], color="r", ls="--", label="half-life")
    ax[2].set_title("20 simulated paths of the fitted OU starting at +2 sigma"); ax[2].legend()

    ax[3].plot(bt["ret"].cumsum(), label="cumulative log-return (net of costs)")
    ax[3].axvline(bt.index[cut], color="k", ls="--")
    ax[3].set_title("Equity curve"); ax[3].legend()
    fig.tight_layout()
    save_fig(fig, "p1_ou_pairs.png")
    plt.show() if "--show" in sys.argv else plt.close(fig)


if __name__ == "__main__":
    main()
