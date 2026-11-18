"""
PROJECT 4 - Hill tail-index estimator vs Gaussian VaR
====================================================

Fat tails:  P(Loss > x) ~ C * x^(-alpha)  for large x   (power law / Pareto tail)
            smaller alpha  ->  fatter tail  ->  more extreme-event risk.
            Normal distribution: tail decays like exp(-x^2/2)  (alpha = infinity).

Hill estimator (Hill 1975), using the k largest losses X_(1) >= ... >= X_(k) >= X_(k+1):
    1/alpha_hat = (1/k) * sum_{i=1..k} ln( X_(i) / X_(k+1) )
    SE(alpha_hat) ~ alpha_hat / sqrt(k)
(the slide uses X_(k) as threshold - same idea; the i=k term is then 0.)

Tail quantile / VaR (Weissman 1978):
    VaR_p = X_(k+1) * ( k / (n * p) )^(1/alpha_hat)        p = 1 - confidence
Expected shortfall for a Pareto tail (alpha > 1):
    ES_p  = VaR_p * alpha / (alpha - 1)
Gaussian VaR:
    VaR_p = -(mu + sigma * z_p) = -mu + sigma * z_{1-p},   z_0.99 = 2.326

Also: Kupiec (1995) proportion-of-failures test of each VaR, rolling out-of-sample.

Run:  python p4_hill_var.py          (SPY)
      python p4_hill_var.py BTC-USD
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from common import get_prices, save_fig


def hill(losses, k):
    x = np.sort(losses[losses > 0])[::-1]          # descending
    thresh = x[k]                                  # X_(k+1)  (0-based index k)
    gamma = np.mean(np.log(x[:k] / thresh))        # = 1/alpha
    return 1.0 / gamma, thresh, len(losses)


def hill_var(losses, k, p):
    a, u, n = hill(losses, k)
    var = u * (k / (n * p)) ** (1 / a)
    es = var * a / (a - 1) if a > 1 else np.inf
    return var, es, a


def gauss_var(r, p):
    mu, sd = r.mean(), r.std(ddof=1)
    z = stats.norm.ppf(1 - p)
    var = -(mu - z * sd)
    es = -(mu - sd * stats.norm.pdf(z) / p)
    return var, es


def kupiec(n_obs, n_exc, p):
    """LR_pof = -2 ln[(1-p)^(n-x) p^x] + 2 ln[(1-x/n)^(n-x) (x/n)^x]  ~ chi2(1)"""
    x, n = n_exc, n_obs
    phat = x / n
    ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
    ll1 = (n - x) * np.log(1 - phat) + (x * np.log(phat) if x > 0 else 0)
    lr = -2 * (ll0 - ll1)
    return lr, 1 - stats.chi2.cdf(lr, 1)


def main():
    tick = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "SPY"
    px = get_prices(tick, start="2000-01-01")
    if px is None:
        rng = np.random.default_rng(0)
        nu = 3.0                                   # true tail index alpha = 3
        r = 0.01 * rng.standard_t(nu, 6000) / np.sqrt(nu / (nu - 2))
        label = "synthetic Student-t(3)  (true alpha = 3)"
        dates = None
    else:
        s = px.iloc[:, 0]
        r = np.log(s).diff().dropna().to_numpy()
        dates = s.index[1:]
        label = tick
    losses = -r
    n = len(r)

    # ---- Hill plot: alpha_hat as a function of k ----------------------------
    ks = np.arange(10, int(0.2 * n))
    alphas = np.array([hill(losses, k)[0] for k in ks])
    k_star = int(round(np.sqrt(n)))               # simple rule of thumb; check the plot is flat here
    a_star, u_star, _ = hill(losses, k_star)
    se = a_star / np.sqrt(k_star)
    print(f"\n=== {label}: n = {n} daily returns ===")
    print(f"daily mean = {r.mean():.5f}, std = {r.std():.4f}, skew = {stats.skew(r):.2f}, "
          f"excess kurtosis = {stats.kurtosis(r):.2f}  (normal = 0)")
    print(f"Hill: k = {k_star} (tail = top {k_star/n:.1%} of days), threshold loss = {u_star:.2%}")
    print(f"TAIL INDEX alpha_hat = {a_star:.2f}  (95% CI {a_star-1.96*se:.2f} - {a_star+1.96*se:.2f})")
    print("  -> moments of order >= alpha do not exist; alpha in 2-5 is typical for stocks")

    # ---- VaR comparison -----------------------------------------------------
    print("\n conf   Gaussian VaR   Hill VaR   empirical   | Gaussian ES   Hill ES   empirical ES")
    rows = []
    for conf in [0.95, 0.99, 0.995, 0.999]:
        p = 1 - conf
        gv, ge = gauss_var(r, p)
        hv, he, _ = hill_var(losses, k_star, p)
        ev = np.quantile(losses, conf)
        ees = losses[losses >= ev].mean()
        rows.append((conf, gv, hv, ev))
        print(f"{conf:6.1%} {gv:12.2%} {hv:10.2%} {ev:10.2%}   | {ge:10.2%} {he:10.2%} {ees:10.2%}")
    gv99, hv99 = rows[1][1], rows[1][2]
    print(f"\n99% VaR gap: Hill/Gaussian = {hv99/gv99:.2f}x  -> Gaussian hides "
          f"{hv99-gv99:.2%} of daily loss at the 99% level")

    # how often should a k-sigma day happen?
    sd = r.std()
    big = np.sum(losses > 5 * sd)
    print(f"days with loss > 5 sigma: observed {big}, normal predicts "
          f"{n*stats.norm.sf(5):.4f}")

    # ---- rolling out-of-sample backtest (1000-day ~ 4-year window) ----------
    W, p = 1000, 0.01
    exc_g = exc_h = tot = 0
    vg_path, vh_path = [], []
    for t in range(W, n):
        win = r[t - W:t]
        vg = gauss_var(win, p)[0]
        vh = hill_var(-win, int(np.sqrt(W)), p)[0]
        vg_path.append(vg); vh_path.append(vh)
        exc_g += losses[t] > vg
        exc_h += losses[t] > vh
        tot += 1
    print(f"\nrolling 99% VaR backtest ({W}-day window, {tot} days, expect {p*tot:.0f} breaches):")
    for name, e in [("Gaussian", exc_g), ("Hill", exc_h)]:
        lr, pv = kupiec(tot, e, p)
        print(f"  {name:9s} breaches = {e:4d} ({e/tot:.2%})  Kupiec LR = {lr:6.2f}  p = {pv:.4f}"
              f"  {'REJECT' if pv < 0.05 else 'ok'}")

    # ---- plots --------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    ax[0, 0].plot(ks, alphas)
    ax[0, 0].fill_between(ks, alphas - 1.96 * alphas / np.sqrt(ks), alphas + 1.96 * alphas / np.sqrt(ks),
                          alpha=0.2)
    ax[0, 0].axvline(k_star, color="r", ls="--", label=f"k = sqrt(n) = {k_star}")
    ax[0, 0].set_ylim(0, 8)
    ax[0, 0].set_title("Hill plot: alpha_hat vs k (pick k where it's flat)")
    ax[0, 0].set_xlabel("k (number of tail observations)"); ax[0, 0].legend()

    x = np.sort(losses[losses > 0])[::-1]
    surv = np.arange(1, len(x) + 1) / n
    ax[0, 1].loglog(x, surv, ".", ms=3, label="empirical P(L > x)")
    xs = np.linspace(x[k_star], x[0], 50)
    ax[0, 1].loglog(xs, (k_star / n) * (xs / x[k_star]) ** (-a_star), "r",
                    label=f"power law alpha = {a_star:.2f}")
    xx = np.linspace(x[-1] + 1e-6, x[0], 200)
    ax[0, 1].loglog(xx, stats.norm.sf(xx, -r.mean(), r.std()), "k--", label="normal")
    ax[0, 1].set_ylim(1 / n / 3, 1); ax[0, 1].set_xlabel("loss"); ax[0, 1].legend()
    ax[0, 1].set_title("Log-log tail: power law is a straight line")

    confs = np.linspace(0.95, 0.999, 60)
    ax[1, 0].plot(confs, [gauss_var(r, 1 - c)[0] for c in confs], "k--", label="Gaussian")
    ax[1, 0].plot(confs, [hill_var(losses, k_star, 1 - c)[0] for c in confs], "r", label="Hill / Weissman")
    ax[1, 0].plot(confs, np.quantile(losses, confs), ":", label="empirical")
    ax[1, 0].set_title("VaR vs confidence level - gap widens in the tail")
    ax[1, 0].set_xlabel("confidence"); ax[1, 0].legend()

    xt = dates[W:] if dates is not None else np.arange(W, n)
    ax[1, 1].plot(xt, losses[W:], lw=0.4, color="grey", label="daily loss")
    ax[1, 1].plot(xt, vg_path, "k", lw=0.8, label="Gaussian 99% VaR")
    ax[1, 1].plot(xt, vh_path, "r", lw=0.8, label="Hill 99% VaR")
    ax[1, 1].set_title("Rolling out-of-sample 99% VaR"); ax[1, 1].legend()
    fig.tight_layout()
    save_fig(fig, "p4_hill_var.png")
    plt.show() if "--show" in sys.argv else plt.close(fig)


if __name__ == "__main__":
    main()
