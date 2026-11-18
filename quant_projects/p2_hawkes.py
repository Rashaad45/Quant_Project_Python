"""
PROJECT 2 - Fit a Hawkes process: do market orders trigger their own aftershocks?
================================================================================

Model (univariate Hawkes, exponential kernel):
    lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta (t - t_i))

    mu    : background ("news") intensity, events / second
    alpha : jump in intensity caused by each event
    beta  : decay speed of the excitement  (1/beta = memory timescale)
    n = alpha / beta : branching ratio = expected number of "child" trades
                       triggered by one trade = fraction of trades that are
                       endogenous (caused by other trades).  Need n < 1.

Log-likelihood on [0, T]:
    logL = sum_i log lambda(t_i)  -  integral_0^T lambda(s) ds
         = sum_i log(mu + alpha*A_i) - mu*T - (alpha/beta) sum_i (1 - e^{-beta (T - t_i)})
with the O(N) recursion (Ozaki 1979)
    A_1 = 0,   A_i = e^{-beta (t_i - t_{i-1})} (1 + A_{i-1})

Steps
 1. Get timestamps: real BTC-USDT trades from Binance's public REST API
    (no key needed), or a CSV you provide, or a simulated Hawkes path.
 2. Validate the estimator on simulated data (known true parameters).
 3. Fit real data by maximum likelihood.
 4. Goodness of fit with the time-rescaling theorem: compensator increments
    should be i.i.d. Exp(1).

Run:  python p2_hawkes.py                 (Binance BTCUSDT, fallback to simulation)
      python p2_hawkes.py trades.csv      (CSV with a 'timestamp' column in seconds)
"""
import json
import sys
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import minimize

from common import save_fig


# ----------------------------------------------------------------------------
# Simulation: Ogata's thinning algorithm
# ----------------------------------------------------------------------------
def simulate_hawkes(mu, alpha, beta, T, seed=0):
    """
    Between events the intensity only decays, so the intensity right now is an
    upper bound until the next event.  Propose a point from a Poisson process
    with that rate, accept it with probability lambda(t)/bound.
    """
    rng = np.random.default_rng(seed)
    t, events = 0.0, []
    excite = 0.0                                   # sum alpha e^{-beta(t-ti)}
    while True:
        lam_bar = mu + excite                      # valid upper bound
        w = rng.exponential(1 / lam_bar)
        t += w
        if t > T:
            break
        excite *= np.exp(-beta * w)                # decay to the candidate time
        if rng.uniform() * lam_bar <= mu + excite: # accept
            events.append(t)
            excite += alpha                        # self-excitation jump
    return np.array(events)


# ----------------------------------------------------------------------------
# Likelihood with the O(N) recursion
# ----------------------------------------------------------------------------
def neg_loglik(params, t, T):
    mu, alpha, beta = params
    if mu <= 0 or alpha < 0 or beta <= 0:
        return np.inf
    dt = np.diff(t)
    A = np.zeros(len(t))
    for i in range(1, len(t)):                     # O(N), not O(N^2)
        A[i] = np.exp(-beta * dt[i - 1]) * (1 + A[i - 1])
    lam = mu + alpha * A
    compensator = mu * T + (alpha / beta) * np.sum(1 - np.exp(-beta * (T - t)))
    return -(np.sum(np.log(lam)) - compensator)


def neg_loglik_bruteforce(params, t, T):
    """O(N^2) double loop - only to verify the recursion on small samples."""
    mu, alpha, beta = params
    ll = 0.0
    for i, ti in enumerate(t):
        ll += np.log(mu + alpha * np.sum(np.exp(-beta * (ti - t[:i]))))
    ll -= mu * T + (alpha / beta) * np.sum(1 - np.exp(-beta * (T - t)))
    return -ll


def fit_hawkes(t, T):
    """MLE in log-parameters (keeps mu, alpha, beta positive)."""
    t0 = t[0]
    t = np.asarray(t) - t0
    T = T - t0
    rate = len(t) / T
    best = None
    for b0 in [0.1, 1.0, 10.0]:                    # a few starts: likelihood can be flat
        x0 = np.log([0.5 * rate, 0.5 * b0, b0])
        f = lambda x: neg_loglik(np.exp(x), t, T)
        r = minimize(f, x0, method="Nelder-Mead",
                     options=dict(maxiter=4000, xatol=1e-6, fatol=1e-6))
        if best is None or r.fun < best.fun:
            best = r
    mu, alpha, beta = np.exp(best.x)
    # standard errors from the numerical Hessian of -logL in natural params
    se = numerical_se(lambda p: neg_loglik(p, t, T), np.array([mu, alpha, beta]))
    return dict(mu=mu, alpha=alpha, beta=beta, n=alpha / beta,
                loglik=-best.fun, se=se, t=t, T=T)


def numerical_se(f, p, h=1e-4):
    k = len(p)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            ei, ej = np.eye(k)[i] * h * p[i], np.eye(k)[j] * h * p[j]
            H[i, j] = (f(p + ei + ej) - f(p + ei - ej) - f(p - ei + ej) + f(p - ei - ej)) \
                      / (4 * ei[i] * ej[j])
    try:
        return np.sqrt(np.diag(np.linalg.inv(H)))
    except np.linalg.LinAlgError:
        return np.full(k, np.nan)


# ----------------------------------------------------------------------------
# Goodness of fit: time-rescaling theorem
# ----------------------------------------------------------------------------
def compensator_increments(t, mu, alpha, beta):
    """
    tau_i = Lambda(t_i) - Lambda(t_{i-1}) = integral of lambda between events.
    If the model is right, tau_i ~ i.i.d. Exp(1).
    """
    dt = np.diff(t)
    tau = np.zeros(len(dt))
    A = 0.0                                       # A just after event i-1, incl. its jump
    for i in range(len(dt)):
        tau[i] = mu * dt[i] + (alpha / beta) * (1 + A) * (1 - np.exp(-beta * dt[i]))
        A = (1 + A) * np.exp(-beta * dt[i])
    return tau


def intensity_path(t, mu, alpha, beta, grid):
    lam = np.full(len(grid), mu)
    for ti in t:
        m = grid > ti
        lam[m] += alpha * np.exp(-beta * (grid[m] - ti))
    return lam


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def binance_trades(symbol="BTCUSDT", pages=5):
    """Last ~pages*1000 aggregated trades (public endpoint, no API key)."""
    url = f"https://api.binance.com/api/v3/aggTrades?symbol={symbol}&limit=1000"
    trades = json.load(urllib.request.urlopen(url, timeout=10))
    for _ in range(pages - 1):                    # page backwards in time
        first_id = trades[0]["a"]
        u = f"{url}&fromId={max(first_id - 1000, 0)}"
        older = json.load(urllib.request.urlopen(u, timeout=10))
        trades = [x for x in older if x["a"] < first_id] + trades
    ts = np.array([x["T"] for x in trades], float) / 1000.0   # ms -> s
    return ts


def prepare(ts, seed=0):
    """Sort, and break ties: many trades share a millisecond -> spread them
    uniformly inside that millisecond (a standard, documented fix)."""
    rng = np.random.default_rng(seed)
    ts = np.sort(ts)
    ts = ts + rng.uniform(0, 1e-3, len(ts))
    ts = np.sort(ts)
    return ts - ts[0]


def main():
    # ---- 1. sanity check: recursion == double loop -------------------------
    small = simulate_hawkes(1.0, 0.5, 1.5, 100, seed=3)
    a = neg_loglik((1.0, 0.5, 1.5), small, 100)
    b = neg_loglik_bruteforce((1.0, 0.5, 1.5), small, 100)
    print(f"O(N) recursion = {a:.6f}   O(N^2) double loop = {b:.6f}   (must match)")

    # ---- 2. parameter recovery on simulated data ---------------------------
    true = dict(mu=0.5, alpha=0.8, beta=1.2)
    sim = simulate_hawkes(**true, T=5000, seed=1)
    fs = fit_hawkes(sim, 5000)
    print(f"\n=== Simulated check (N={len(sim)}) ===")
    print(f"true : mu={true['mu']:.3f} alpha={true['alpha']:.3f} beta={true['beta']:.3f} "
          f"n={true['alpha']/true['beta']:.3f}")
    print(f"fit  : mu={fs['mu']:.3f} alpha={fs['alpha']:.3f} beta={fs['beta']:.3f} "
          f"n={fs['n']:.3f}   se={np.round(fs['se'], 3)}")

    # ---- 3. real data --------------------------------------------------------
    label = "simulated"
    data = sim
    try:
        if len(sys.argv) > 1 and sys.argv[1].endswith(".csv"):
            import pandas as pd
            data = prepare(pd.read_csv(sys.argv[1])["timestamp"].to_numpy(float))
            label = sys.argv[1]
        else:
            data = prepare(binance_trades())
            label = "Binance BTCUSDT aggTrades"
    except Exception as exc:
        print(f"[data] real trades unavailable ({exc}); fitting simulated data instead")

    fr = fit_hawkes(data, data[-1])
    T = fr["T"]
    print(f"\n=== Fit on {label}: N={len(data)} events over {T:.0f} s ===")
    print(f"mu    = {fr['mu']:.4f} /s  (background rate, se {fr['se'][0]:.4f})")
    print(f"alpha = {fr['alpha']:.4f} /s  (jump per event, se {fr['se'][1]:.4f})")
    print(f"beta  = {fr['beta']:.4f} /s  (decay; memory 1/beta = {1000/fr['beta']:.3f} ms, "
          f"half-life ln2/beta = {1000*np.log(2)/fr['beta']:.3f} ms)")
    print(f"BRANCHING RATIO n = alpha/beta = {fr['n']:.3f}  -> ~{fr['n']:.0%} of trades are "
          f"triggered by earlier trades, {1-fr['n']:.0%} are exogenous")
    print(f"implied mean rate mu/(1-n) = {fr['mu']/(1-fr['n']):.3f}/s   observed = {len(data)/T:.3f}/s")

    # Poisson benchmark (alpha = 0) -> likelihood-ratio test
    ll_pois = len(data) * np.log(len(data) / T) - len(data)
    lr = 2 * (fr["loglik"] - ll_pois)
    print(f"LR test Hawkes vs Poisson: LR = {lr:.1f}  (chi2(2) 1% crit = 9.21)")

    tau = compensator_increments(fr["t"], fr["mu"], fr["alpha"], fr["beta"])
    ks = stats.kstest(tau, "expon")
    tau_p = np.diff(fr["t"]) * len(data) / T
    ks_p = stats.kstest(tau_p, "expon")
    print(f"KS test of rescaled times vs Exp(1): Hawkes stat={ks.statistic:.4f} | "
          f"Poisson stat={ks_p.statistic:.4f}   (smaller = better fit)")

    # ---- plots --------------------------------------------------------------
    fig, ax = plt.subplots(3, 1, figsize=(11, 11))
    w_end = min(T, fr["t"][min(300, len(data) - 1)])
    grid = np.linspace(0, w_end, 3000)
    tw = fr["t"][fr["t"] <= w_end]
    ax[0].plot(grid, intensity_path(tw, fr["mu"], fr["alpha"], fr["beta"], grid), lw=0.8)
    ax[0].plot(tw, np.zeros_like(tw), "|", color="r", ms=12, label="trades")
    ax[0].axhline(fr["mu"], color="k", ls="--", label="mu (background)")
    ax[0].set_title(f"Fitted intensity lambda(t) - {label} (first {len(tw)} trades)")
    ax[0].legend()

    q = np.sort(tau)
    theo = stats.expon.ppf((np.arange(1, len(q) + 1) - 0.5) / len(q))
    ax[1].plot(theo, q, ".", ms=2, label="Hawkes compensator")
    ax[1].plot(theo, np.sort(tau_p), ".", ms=2, label="Poisson (constant rate)")
    ax[1].plot([0, theo.max()], [0, theo.max()], "k--")
    ax[1].set_xlabel("Exp(1) quantiles"); ax[1].set_ylabel("rescaled inter-event times")
    ax[1].set_title("Time-rescaling QQ plot (on the line = model explains clustering)")
    ax[1].legend()

    cnt, _ = np.histogram(fr["t"], bins=np.arange(0, T, 1.0))
    acf = [np.corrcoef(cnt[:-k], cnt[k:])[0, 1] for k in range(1, 40)]
    ax[2].bar(range(1, 40), acf)
    ax[2].set_title("Autocorrelation of 1-second trade counts (Poisson would give ~0)")
    ax[2].set_xlabel("lag (s)")
    fig.tight_layout()
    save_fig(fig, "p2_hawkes.png")
    plt.show() if "--show" in sys.argv else plt.close(fig)


if __name__ == "__main__":
    main()
