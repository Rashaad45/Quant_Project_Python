"""
PROJECT 3 - Limit order book simulator and your own market impact
================================================================

Part A: a price-time priority matching engine
    * bids / asks are dicts  price -> FIFO queue (deque) of resting orders
    * best price first (price priority), oldest order first (time priority)
    * order types: limit (may cross and execute), market, cancel

Part B: synthetic order flow (zero-intelligence model, Smith-Farmer et al. 2003,
         Cont-Stoikov-Talreja 2010)
    * limit orders arrive at rate  lam(d) = A * d^-gamma  at distance d ticks
      from the opposite best quote (more orders near the touch)
    * market orders arrive at rate  mkt_rate,  cancellations at rate  cancel_rate
      per resting order
    * calibrated so the simulated spread, depth and traded volume look like a
      liquid stock (tick $0.01, price ~$100, ~1-2 tick spread)

Part C: impact measurement
    * take many snapshots of the book, on a COPY of each send a market buy of
      Q shares, record VWAP; slippage = (VWAP - mid)/mid in bps
    * fit  slippage ~ c * Q^delta  and total cost ~ Q^(1+delta)

Run:  python p3_lob_simulator.py
"""
import copy
import sys
from collections import deque
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np

from common import save_fig

TICK = 0.01


@dataclass
class Order:
    oid: int
    side: str          # 'B' or 'S'
    price: int         # in ticks (integers avoid float rounding issues)
    qty: int


@dataclass
class OrderBook:
    bids: dict = field(default_factory=dict)   # price -> deque[Order]
    asks: dict = field(default_factory=dict)
    where: dict = field(default_factory=dict)  # oid -> (side, price) for O(1) cancel lookup
    trades: list = field(default_factory=list)
    next_id: int = 0

    # ---------- queries ----------
    def best_bid(self):
        return max(self.bids) if self.bids else None

    def best_ask(self):
        return min(self.asks) if self.asks else None

    def mid(self):
        b, a = self.best_bid(), self.best_ask()
        return None if b is None or a is None else 0.5 * (a + b)

    def depth(self, side, levels=5):
        book = self.bids if side == "B" else self.asks
        prices = sorted(book, reverse=(side == "B"))[:levels]
        return [(p, sum(o.qty for o in book[p])) for p in prices]

    # ---------- order entry ----------
    def limit(self, side, price, qty):
        """Limit order: match against the opposite side while it crosses,
        rest any remainder in the book (at the back of its price queue)."""
        qty = self._match(side, qty, limit_price=price)
        if qty > 0:
            self.next_id += 1
            o = Order(self.next_id, side, price, qty)
            book = self.bids if side == "B" else self.asks
            book.setdefault(price, deque()).append(o)
            self.where[o.oid] = (side, price)
            return o.oid
        return None

    def market(self, side, qty):
        """Market order: eat liquidity until filled. Returns list of (price, qty) fills."""
        n0 = len(self.trades)
        left = self._match(side, qty, limit_price=None)
        return self.trades[n0:], left

    def cancel(self, oid):
        if oid not in self.where:
            return False
        side, price = self.where.pop(oid)
        book = self.bids if side == "B" else self.asks
        q = book[price]
        for o in q:
            if o.oid == oid:
                q.remove(o)
                break
        if not q:
            del book[price]
        return True

    # ---------- matching engine ----------
    def _match(self, side, qty, limit_price):
        opp = self.asks if side == "B" else self.bids
        while qty > 0 and opp:
            best = min(opp) if side == "B" else max(opp)
            if limit_price is not None and (
                    (side == "B" and best > limit_price) or (side == "S" and best < limit_price)):
                break                                  # no longer crosses
            q = opp[best]
            while qty > 0 and q:
                resting = q[0]                         # time priority: oldest first
                fill = min(qty, resting.qty)
                self.trades.append((best, fill))
                resting.qty -= fill
                qty -= fill
                if resting.qty == 0:
                    q.popleft()
                    self.where.pop(resting.oid, None)
            if not q:
                del opp[best]
        return qty


# ----------------------------------------------------------------------------
# Part B: zero-intelligence order flow
# ----------------------------------------------------------------------------
@dataclass
class FlowParams:
    p0: int = 10_000            # starting price in ticks ($100.00)
    lim_rate: float = 12.0      # limit orders / sec (each side)
    mkt_rate: float = 2.0       # market orders / sec (each side)
    cancel_rate: float = 0.25   # cancel prob / sec per resting order
    gamma: float = 1.0          # placement power law exponent
    max_dist: int = 30          # ticks
    size_mean: float = 100      # shares, lognormal order size
    size_sigma: float = 0.7


def order_size(rng, P):
    s = rng.lognormal(np.log(P.size_mean) - 0.5 * P.size_sigma ** 2, P.size_sigma)
    return max(1, int(round(s)))


def place_dist(rng, P):
    d = np.arange(1, P.max_dist + 1)
    w = d ** (-P.gamma)
    return rng.choice(d, p=w / w.sum())


def simulate(P, seconds, rng, snapshot_every=None):
    """Event-driven (Gillespie) simulation. Returns final book and snapshots."""
    ob = OrderBook()
    # seed the book so it isn't empty at t=0
    for d in range(1, P.max_dist + 1):
        for _ in range(3):
            ob.limit("B", P.p0 - d, order_size(rng, P))
            ob.limit("S", P.p0 + d, order_size(rng, P))
    t, snaps, next_snap = 0.0, [], snapshot_every
    mids = []
    while t < seconds:
        n_rest = len(ob.where)
        rates = np.array([P.lim_rate, P.lim_rate, P.mkt_rate, P.mkt_rate,
                          P.cancel_rate * n_rest])
        R = rates.sum()
        t += rng.exponential(1 / R)
        ev = rng.choice(5, p=rates / R)
        bb, ba = ob.best_bid(), ob.best_ask()
        if bb is None:
            bb = (ba or P.p0) - 1
        if ba is None:
            ba = bb + 1
        if ev == 0:      # buy limit: placed d ticks below the best ask
            ob.limit("B", ba - place_dist(rng, P), order_size(rng, P))
        elif ev == 1:    # sell limit: d ticks above the best bid
            ob.limit("S", bb + place_dist(rng, P), order_size(rng, P))
        elif ev == 2:
            ob.market("B", order_size(rng, P))
        elif ev == 3:
            ob.market("S", order_size(rng, P))
        else:
            oid = rng.choice(list(ob.where))
            ob.cancel(oid)
        if ob.mid() is not None:
            mids.append((t, ob.mid()))
        if snapshot_every and t >= next_snap:
            snaps.append(copy.deepcopy(ob))
            next_snap += snapshot_every
    return ob, snaps, np.array(mids)


# ----------------------------------------------------------------------------
# Part C: market impact of a hypothetical order
# ----------------------------------------------------------------------------
def execution_cost(book, side, Q):
    b = copy.deepcopy(book)
    mid = b.mid()
    b.trades = []
    fills, left = b.market(side, Q)
    if left > 0:
        return np.nan, np.nan
    px = np.array([f[0] for f in fills], float)
    qty = np.array([f[1] for f in fills], float)
    vwap = (px * qty).sum() / qty.sum()
    sign = 1 if side == "B" else -1
    slip_bps = sign * (vwap - mid) / mid * 1e4
    return slip_bps, slip_bps * 1e-4 * mid * TICK * Q      # bps, dollars


def main():
    rng = np.random.default_rng(7)
    P = FlowParams()

    # ---- unit test of the matching engine -----------------------------------
    t = OrderBook()
    t.limit("S", 101, 50); t.limit("S", 101, 30); t.limit("S", 102, 100)
    fills, left = t.market("B", 60)
    assert fills == [(101, 50), (101, 10)] and left == 0, fills   # FIFO at 101
    assert t.depth("S") == [(101, 20), (102, 100)]
    print("matching engine unit test passed (price-time priority OK)")

    # ---- run the market ----------------------------------------------------
    print("simulating 1 hour of order flow ...")
    _, snaps, mids = simulate(P, seconds=3600, rng=rng, snapshot_every=20)
    snaps = snaps[5:]                           # discard warm-up
    spreads = np.array([s.best_ask() - s.best_bid() for s in snaps])
    depth5 = np.array([sum(q for _, q in s.depth("S", 5)) for s in snaps])
    print(f"snapshots: {len(snaps)}   mean spread = {spreads.mean():.2f} ticks "
          f"(${spreads.mean()*TICK:.3f})   mean depth top-5 ask levels = {depth5.mean():.0f} sh")
    ret = np.diff(mids[:, 1])
    print(f"mid moved {len(ret.nonzero()[0])} times; price range "
          f"${mids[:,1].min()*TICK:.2f} - ${mids[:,1].max()*TICK:.2f}")

    # ---- impact curve -------------------------------------------------------
    Qs = np.unique(np.round(np.logspace(1.3, 3.45, 14)).astype(int))
    slip = np.array([[execution_cost(s, "B", Q)[0] for s in snaps] for Q in Qs])
    m = np.nanmean(slip, axis=1)
    lo, hi = np.nanpercentile(slip, [10, 90], axis=1)
    print("\n   Q (sh)   avg slippage (bps)   total cost ($)")
    for Q, s in zip(Qs, m):
        print(f"{Q:9d} {s:14.2f} {s*1e-4*P.p0*TICK*Q:16.2f}")

    # fit slippage = c * Q^delta on orders larger than the touch (log-log OLS)
    big = Qs >= 300   # beyond the touch the book is walked level by level
    delta, logc = np.polyfit(np.log(Qs[big]), np.log(m[big]), 1)
    print(f"\nfit: slippage ~ Q^{delta:.2f}   ->   total cost ~ Q^{1+delta:.2f}")
    for Q1 in [500, 1000, 2000]:
        if Q1 * 2 <= Qs.max():
            r = np.interp(2 * Q1, Qs, m) / np.interp(Q1, Qs, m)
            print(f"doubling {Q1}->{2*Q1} shares multiplies slippage/share by {r:.2f} "
                  f"and total cost by {2*r:.2f}")

    # ---- plots --------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    s = snaps[len(snaps) // 2]
    bid = s.depth("B", 25); ask = s.depth("S", 25)
    ax[0, 0].bar([p * TICK for p, _ in bid], [q for _, q in bid], width=TICK * 0.9, color="g", label="bids")
    ax[0, 0].bar([p * TICK for p, _ in ask], [q for _, q in ask], width=TICK * 0.9, color="r", label="asks")
    ax[0, 0].set_title("Order book snapshot (depth per price level)"); ax[0, 0].legend()
    ax[0, 0].set_xlabel("price ($)")

    ax[0, 1].plot(mids[:, 0] / 60, mids[:, 1] * TICK, lw=0.6)
    ax[0, 1].set_title("Mid price produced by the order flow"); ax[0, 1].set_xlabel("minutes")

    ax[1, 0].plot(Qs, m, "o-", label="mean slippage")
    ax[1, 0].fill_between(Qs, lo, hi, alpha=0.25, label="10-90% across snapshots")
    ax[1, 0].plot(Qs, Qs * m[0] / Qs[0], "k--", lw=1, label="'linear' reference")
    ax[1, 0].set_xscale("log"); ax[1, 0].set_yscale("log")
    ax[1, 0].set_title(f"Slippage vs order size: slope delta = {delta:.2f}")
    ax[1, 0].set_xlabel("Q (shares)"); ax[1, 0].set_ylabel("bps vs mid"); ax[1, 0].legend()

    ax[1, 1].plot(Qs, m * 1e-4 * P.p0 * TICK * Qs, "o-")
    ax[1, 1].set_title("Total execution cost ($) - convex in Q")
    ax[1, 1].set_xlabel("Q (shares)")
    fig.tight_layout()
    save_fig(fig, "p3_lob_impact.png")
    plt.show() if "--show" in sys.argv else plt.close(fig)


if __name__ == "__main__":
    main()
