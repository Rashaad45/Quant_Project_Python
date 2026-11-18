"""
common.py - shared helpers for the four quant projects.

* get_prices(): downloads daily closes with yfinance; if there is no internet
  (or yfinance is not installed) it falls back to synthetic data so every
  script still runs.
* OUT_DIR: folder where every script saves its figures.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True)


def get_prices(tickers, start="2015-01-01", end=None):
    """Return a DataFrame of adjusted daily close prices, one column per ticker."""
    try:
        import yfinance as yf

        df = yf.download(tickers, start=start, end=end, auto_adjust=True,
                         progress=False)["Close"]
        if isinstance(df, pd.Series):
            df = df.to_frame(tickers if isinstance(tickers, str) else tickers[0])
        df = df.dropna()
        if len(df) < 250:
            raise ValueError("too little data returned")
        print(f"[data] downloaded {list(df.columns)}: {len(df)} rows "
              f"{df.index[0].date()} -> {df.index[-1].date()}")
        return df
    except Exception as exc:  # no internet / no yfinance -> synthetic fallback
        print(f"[data] download failed ({exc}); using SYNTHETIC data")
        return None


def save_fig(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"[plot] saved {path}")
    return path
