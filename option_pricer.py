#!/usr/bin/env python3
"""
European Option Pricing & Risk Toolkit
======================================
Black–Scholes–Merton (BSM) analytical pricing, Greeks, Monte Carlo simulation,
implied volatility (Brent), put–call parity checks, polished visualizations,
and a multi-page PDF report.

Run:
    python option_pricer.py

License: MIT
"""

from __future__ import annotations

import math
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless / CI use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from scipy.optimize import brentq
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Paths & styling
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
PLOTS_DIR = PROJECT_ROOT / "plots"
REPORT_PATH = PROJECT_ROOT / "Quant_Option_Pricing_Report.pdf"

# Professional color palette
COLORS = {
    "call": "#1f77b4",
    "put": "#d62728",
    "spot": "#2ca02c",
    "strike": "#7f7f7f",
    "bg": "#fafafa",
    "grid": "#e0e0e0",
    "accent": "#9467bd",
    "mc": "#ff7f0e",
}

OptionType = Literal["call", "put"]
ArrayLike = Union[float, np.ndarray]


# ===========================================================================
# Core pricing engine
# ===========================================================================


@dataclass
class EuropeanOption:
    """
    European equity (or equity-index) option under the Black–Scholes–Merton model.

    The continuous dividend yield ``q`` defaults to 0 (non-dividend stock).
    Setting ``q = r_f`` recovers the Garman–Kohlhagen FX framework (spot = S,
    domestic rate = r, foreign rate = q).

    Parameters
    ----------
    S : float
        Spot price of the underlying.
    K : float
        Strike price.
    T : float
        Time to expiry in years (e.g. 0.5 = 6 months).
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Annualized volatility of the underlying log-returns.
    q : float, optional
        Continuous dividend yield (default 0.0).
    option_type : {'call', 'put'}
        Option type (default 'call').

    Notes
    -----
    Closed-form BSM price::

        d1 = [ln(S/K) + (r - q + 0.5 σ²) T] / (σ √T)
        d2 = d1 - σ √T
        Call = S e^{-qT} N(d1) - K e^{-rT} N(d2)
        Put  = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)
    """

    S: float
    K: float
    T: float
    r: float
    sigma: float
    q: float = 0.0
    option_type: OptionType = "call"
    _validated: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Raise ``ValueError`` on economically invalid inputs."""
        if self.S <= 0:
            raise ValueError(f"Spot S must be > 0, got {self.S}")
        if self.K <= 0:
            raise ValueError(f"Strike K must be > 0, got {self.K}")
        if self.T < 0:
            raise ValueError(f"Time to expiry T must be >= 0, got {self.T}")
        if self.sigma < 0:
            raise ValueError(f"Volatility sigma must be >= 0, got {self.sigma}")
        if self.option_type not in ("call", "put"):
            raise ValueError(
                f"option_type must be 'call' or 'put', got {self.option_type!r}"
            )
        # Rates / yields can be negative in modern markets — only warn
        if abs(self.r) > 1.0:
            warnings.warn(
                f"Risk-free rate r={self.r} looks unusually large; "
                "ensure it is expressed as a decimal (e.g. 0.05 for 5%).",
                UserWarning,
                stacklevel=2,
            )
        self._validated = True

    def copy(self, **kwargs) -> "EuropeanOption":
        """Return a new option with selected fields overridden."""
        data = {
            "S": self.S,
            "K": self.K,
            "T": self.T,
            "r": self.r,
            "sigma": self.sigma,
            "q": self.q,
            "option_type": self.option_type,
        }
        data.update(kwargs)
        return EuropeanOption(**data)

    # ------------------------------------------------------------------
    # d1 / d2 (vectorized)
    # ------------------------------------------------------------------

    def d1(
        self,
        S: Optional[ArrayLike] = None,
        sigma: Optional[ArrayLike] = None,
        T: Optional[ArrayLike] = None,
    ) -> ArrayLike:
        """Black–Scholes d1 term (supports scalar or array inputs)."""
        S = self.S if S is None else S
        sigma = self.sigma if sigma is None else sigma
        T = self.T if T is None else T
        S = np.asarray(S, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        T = np.asarray(T, dtype=float)

        # Expiry / zero-vol edge cases handled carefully
        sqrt_T = np.sqrt(np.maximum(T, 0.0))
        denom = sigma * sqrt_T

        with np.errstate(divide="ignore", invalid="ignore"):
            d1 = (np.log(S / self.K) + (self.r - self.q + 0.5 * sigma**2) * T) / denom
            # Intrinsic limit as T→0 or σ→0
            intrinsic_sign = np.where(S * np.exp(-self.q * T) >= self.K * np.exp(-self.r * T), 1e10, -1e10)
            d1 = np.where((denom == 0) | ~np.isfinite(d1), intrinsic_sign, d1)
        return d1 if d1.ndim else float(d1)

    def d2(
        self,
        S: Optional[ArrayLike] = None,
        sigma: Optional[ArrayLike] = None,
        T: Optional[ArrayLike] = None,
    ) -> ArrayLike:
        """Black–Scholes d2 = d1 − σ√T."""
        S = self.S if S is None else S
        sigma = self.sigma if sigma is None else sigma
        T = self.T if T is None else T
        d1 = self.d1(S=S, sigma=sigma, T=T)
        sqrt_T = np.sqrt(np.maximum(np.asarray(T, dtype=float), 0.0))
        d2 = np.asarray(d1) - np.asarray(sigma) * sqrt_T
        return d2 if np.ndim(d2) else float(d2)

    # ------------------------------------------------------------------
    # Analytical price
    # ------------------------------------------------------------------

    def price(
        self,
        S: Optional[ArrayLike] = None,
        sigma: Optional[ArrayLike] = None,
        T: Optional[ArrayLike] = None,
    ) -> ArrayLike:
        """
        Black–Scholes–Merton analytical price (vectorized over S, σ, T).

        Returns
        -------
        float or np.ndarray
            Discounted risk-neutral expectation of the European payoff.
        """
        S = self.S if S is None else S
        sigma = self.sigma if sigma is None else sigma
        T = self.T if T is None else T
        S_arr = np.asarray(S, dtype=float)
        T_arr = np.asarray(T, dtype=float)

        d1 = np.asarray(self.d1(S=S, sigma=sigma, T=T))
        d2 = np.asarray(self.d2(S=S, sigma=sigma, T=T))

        df_r = np.exp(-self.r * T_arr)
        df_q = np.exp(-self.q * T_arr)

        if self.option_type == "call":
            px = S_arr * df_q * norm.cdf(d1) - self.K * df_r * norm.cdf(d2)
        else:
            px = self.K * df_r * norm.cdf(-d2) - S_arr * df_q * norm.cdf(-d1)

        # At expiry: pure intrinsic
        intrinsic = (
            np.maximum(S_arr - self.K, 0.0)
            if self.option_type == "call"
            else np.maximum(self.K - S_arr, 0.0)
        )
        px = np.where(T_arr <= 0, intrinsic, px)
        return float(px) if np.ndim(px) == 0 else px

    # ------------------------------------------------------------------
    # Greeks
    # ------------------------------------------------------------------

    def delta(self, S: Optional[ArrayLike] = None) -> ArrayLike:
        """
        Δ = ∂V/∂S.

        Call: e^{-qT} N(d1)   |   Put: e^{-qT} (N(d1) − 1)
        """
        S = self.S if S is None else S
        if self.T <= 0:
            S_arr = np.asarray(S, dtype=float)
            if self.option_type == "call":
                d = np.where(S_arr > self.K, 1.0, np.where(S_arr < self.K, 0.0, 0.5))
            else:
                d = np.where(S_arr < self.K, -1.0, np.where(S_arr > self.K, 0.0, -0.5))
            return float(d) if np.ndim(d) == 0 else d

        d1 = np.asarray(self.d1(S=S))
        disc_q = math.exp(-self.q * self.T)
        if self.option_type == "call":
            out = disc_q * norm.cdf(d1)
        else:
            out = disc_q * (norm.cdf(d1) - 1.0)
        return float(out) if np.ndim(out) == 0 else out

    def gamma(self, S: Optional[ArrayLike] = None) -> ArrayLike:
        """
        Γ = ∂²V/∂S² = e^{-qT} n(d1) / (S σ √T)

        Same for calls and puts (by put–call parity).
        """
        S = self.S if S is None else S
        S_arr = np.asarray(S, dtype=float)
        if self.T <= 0 or self.sigma <= 0:
            out = np.zeros_like(S_arr, dtype=float)
            return float(out) if np.ndim(out) == 0 else out

        d1 = np.asarray(self.d1(S=S))
        out = (
            math.exp(-self.q * self.T)
            * norm.pdf(d1)
            / (S_arr * self.sigma * math.sqrt(self.T))
        )
        return float(out) if np.ndim(out) == 0 else out

    def vega(self, S: Optional[ArrayLike] = None) -> ArrayLike:
        """
        ν = ∂V/∂σ = S e^{-qT} n(d1) √T

        Returned in absolute units (per 1.0 vol point, not per 1%).
        Divide by 100 for the conventional “per 1%” quoting convention.
        """
        S = self.S if S is None else S
        S_arr = np.asarray(S, dtype=float)
        if self.T <= 0:
            out = np.zeros_like(S_arr, dtype=float)
            return float(out) if np.ndim(out) == 0 else out

        d1 = np.asarray(self.d1(S=S))
        out = S_arr * math.exp(-self.q * self.T) * norm.pdf(d1) * math.sqrt(self.T)
        return float(out) if np.ndim(out) == 0 else out

    def theta(self, S: Optional[ArrayLike] = None) -> ArrayLike:
        """
        Θ = ∂V/∂t  (calendar-time; negative of ∂V/∂T in the usual PDE form).

        Returned as **per year**. Divide by 365 for per-calendar-day theta.
        """
        S = self.S if S is None else S
        S_arr = np.asarray(S, dtype=float)
        if self.T <= 0:
            out = np.zeros_like(S_arr, dtype=float)
            return float(out) if np.ndim(out) == 0 else out

        d1 = np.asarray(self.d1(S=S))
        d2 = np.asarray(self.d2(S=S))
        sqrt_T = math.sqrt(self.T)
        disc_q = math.exp(-self.q * self.T)
        disc_r = math.exp(-self.r * self.T)

        common = -(S_arr * disc_q * norm.pdf(d1) * self.sigma) / (2.0 * sqrt_T)

        if self.option_type == "call":
            out = (
                common
                - self.r * self.K * disc_r * norm.cdf(d2)
                + self.q * S_arr * disc_q * norm.cdf(d1)
            )
        else:
            out = (
                common
                + self.r * self.K * disc_r * norm.cdf(-d2)
                - self.q * S_arr * disc_q * norm.cdf(-d1)
            )
        return float(out) if np.ndim(out) == 0 else out

    def rho(self, S: Optional[ArrayLike] = None) -> ArrayLike:
        """
        ρ = ∂V/∂r.

        Call:  K T e^{-rT} N(d2)
        Put:  −K T e^{-rT} N(−d2)

        Absolute units (per 1.0 rate). Divide by 100 for per-1% convention.
        """
        S = self.S if S is None else S
        if self.T <= 0:
            S_arr = np.asarray(S, dtype=float)
            out = np.zeros_like(S_arr, dtype=float)
            return float(out) if np.ndim(out) == 0 else out

        d2 = np.asarray(self.d2(S=S))
        factor = self.K * self.T * math.exp(-self.r * self.T)
        if self.option_type == "call":
            out = factor * norm.cdf(d2)
        else:
            out = -factor * norm.cdf(-d2)
        return float(out) if np.ndim(out) == 0 else out

    def greeks(self) -> Dict[str, float]:
        """Return a dictionary of all first- and second-order Greeks."""
        return {
            "price": float(self.price()),
            "delta": float(self.delta()),
            "gamma": float(self.gamma()),
            "vega": float(self.vega()),
            "theta": float(self.theta()),
            "rho": float(self.rho()),
            # Market conventions (per 1% / per day)
            "vega_per_1pct": float(self.vega()) / 100.0,
            "theta_per_day": float(self.theta()) / 365.0,
            "rho_per_1pct": float(self.rho()) / 100.0,
        }

    # ------------------------------------------------------------------
    # Monte Carlo (risk-neutral GBM)
    # ------------------------------------------------------------------

    def monte_carlo_price(
        self,
        n_paths: int = 100_000,
        n_steps: int = 1,
        seed: Optional[int] = 42,
        antithetic: bool = True,
        return_paths: bool = False,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """
        Risk-neutral Monte Carlo price under geometric Brownian motion.

        Parameters
        ----------
        n_paths : int
            Number of simulated terminal prices (before antithetic doubling).
        n_steps : int
            Time steps per path (1 is exact for European under GBM).
        seed : int or None
            RNG seed for reproducibility.
        antithetic : bool
            Use antithetic variates (halves variance asymptotically).
        return_paths : bool
            If True, include the terminal price array in the result.

        Returns
        -------
        dict
            ``price``, ``stderr``, ``ci_95_low``, ``ci_95_high``, ``n_effective``,
            and optionally ``ST``.
        """
        if n_paths < 1:
            raise ValueError("n_paths must be >= 1")
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")

        rng = np.random.default_rng(seed)
        dt = self.T / n_steps
        drift = (self.r - self.q - 0.5 * self.sigma**2) * dt
        vol = self.sigma * math.sqrt(dt)

        # Simulate log-price increments
        Z = rng.standard_normal((n_paths, n_steps))
        if antithetic:
            Z = np.vstack([Z, -Z])

        log_ST = math.log(self.S) + np.sum(drift + vol * Z, axis=1)
        ST = np.exp(log_ST)

        if self.option_type == "call":
            payoffs = np.maximum(ST - self.K, 0.0)
        else:
            payoffs = np.maximum(self.K - ST, 0.0)

        disc = math.exp(-self.r * self.T)
        discounted = disc * payoffs
        mean = float(np.mean(discounted))
        # Sample standard error of the mean
        stderr = float(np.std(discounted, ddof=1) / math.sqrt(len(discounted)))

        result: Dict[str, Union[float, np.ndarray]] = {
            "price": mean,
            "stderr": stderr,
            "ci_95_low": mean - 1.96 * stderr,
            "ci_95_high": mean + 1.96 * stderr,
            "n_effective": float(len(discounted)),
        }
        if return_paths:
            result["ST"] = ST
            result["discounted_payoffs"] = discounted
        return result

    # ------------------------------------------------------------------
    # Implied volatility (Brent)
    # ------------------------------------------------------------------

    def implied_volatility(
        self,
        market_price: float,
        low: float = 1e-6,
        high: float = 5.0,
        tol: float = 1e-8,
        max_expand: int = 10,
    ) -> float:
        """
        Invert the BSM formula for σ using Brent's method (``scipy.optimize.brentq``).

        Parameters
        ----------
        market_price : float
            Observed / target option premium (must be arbitrage-free).
        low, high : float
            Initial bracket for volatility.
        tol : float
            Absolute tolerance on the price residual.
        max_expand : int
            How many times to double the upper bound if the root is not bracketed.

        Returns
        -------
        float
            Implied volatility (absolute, e.g. 0.25 = 25%).

        Raises
        ------
        ValueError
            If the price is outside no-arbitrage bounds or the root cannot be found.
        """
        if market_price < 0:
            raise ValueError("market_price must be non-negative")

        # No-arbitrage bounds
        disc_r = math.exp(-self.r * self.T)
        disc_q = math.exp(-self.q * self.T)
        if self.option_type == "call":
            lower = max(0.0, self.S * disc_q - self.K * disc_r)
            upper = self.S * disc_q
        else:
            lower = max(0.0, self.K * disc_r - self.S * disc_q)
            upper = self.K * disc_r

        # Numerical tolerance for bound checks
        eps = 1e-10
        if market_price < lower - eps:
            raise ValueError(
                f"market_price {market_price:.6f} below intrinsic lower bound {lower:.6f}"
            )
        if market_price > upper + eps:
            raise ValueError(
                f"market_price {market_price:.6f} above upper bound {upper:.6f}"
            )

        # Deep intrinsic / worthless → IV → 0
        if abs(market_price - lower) < 1e-12:
            return 0.0

        def objective(sig: float) -> float:
            return float(self.copy(sigma=sig).price()) - market_price

        # Expand bracket if needed
        a, b = low, high
        fa, fb = objective(a), objective(b)
        expand = 0
        while fa * fb > 0 and expand < max_expand:
            b *= 2.0
            fb = objective(b)
            expand += 1

        if fa * fb > 0:
            raise ValueError(
                f"Could not bracket IV root for market_price={market_price:.6f} "
                f"(tried high={b:.4f})"
            )

        iv = brentq(objective, a, b, xtol=tol, rtol=tol, maxiter=200)
        return float(iv)

    # ------------------------------------------------------------------
    # Put–call parity
    # ------------------------------------------------------------------

    def put_call_parity_check(self, tol: float = 1e-8) -> Dict[str, float]:
        """
        Validate C − P = S e^{-qT} − K e^{-rT}.

        Returns residual statistics; ``parity_holds`` is True if |residual| < tol.
        """
        call = self.copy(option_type="call").price()
        put = self.copy(option_type="put").price()
        forward_diff = self.S * math.exp(-self.q * self.T) - self.K * math.exp(
            -self.r * self.T
        )
        lhs = call - put
        residual = lhs - forward_diff
        return {
            "call_price": float(call),
            "put_price": float(put),
            "call_minus_put": float(lhs),
            "forward_diff": float(forward_diff),
            "residual": float(residual),
            "parity_holds": abs(residual) < tol,
            "tol": tol,
        }

    # ------------------------------------------------------------------
    # Intrinsic / time value helpers
    # ------------------------------------------------------------------

    def intrinsic_value(self, S: Optional[ArrayLike] = None) -> ArrayLike:
        """Immediate exercise value (European cannot exercise early, but useful for plots)."""
        S = self.S if S is None else S
        S_arr = np.asarray(S, dtype=float)
        if self.option_type == "call":
            out = np.maximum(S_arr - self.K, 0.0)
        else:
            out = np.maximum(self.K - S_arr, 0.0)
        return float(out) if np.ndim(out) == 0 else out

    def time_value(self) -> float:
        """Premium above intrinsic: price − intrinsic."""
        return float(self.price()) - float(self.intrinsic_value())

    def summary(self) -> pd.DataFrame:
        """Tabular summary of parameters, price, and Greeks."""
        g = self.greeks()
        rows = [
            ("Spot (S)", self.S),
            ("Strike (K)", self.K),
            ("Time to expiry (T, years)", self.T),
            ("Risk-free rate (r)", self.r),
            ("Volatility (σ)", self.sigma),
            ("Dividend yield (q)", self.q),
            ("Option type", self.option_type),
            ("BSM Price", g["price"]),
            ("Intrinsic", self.intrinsic_value()),
            ("Time value", self.time_value()),
            ("Delta", g["delta"]),
            ("Gamma", g["gamma"]),
            ("Vega (per 1.0 vol)", g["vega"]),
            ("Vega (per 1%)", g["vega_per_1pct"]),
            ("Theta (per year)", g["theta"]),
            ("Theta (per day)", g["theta_per_day"]),
            ("Rho (per 1.0 rate)", g["rho"]),
            ("Rho (per 1%)", g["rho_per_1pct"]),
        ]
        return pd.DataFrame(rows, columns=["Metric", "Value"])

    def __str__(self) -> str:
        g = self.greeks()
        return (
            f"European {self.option_type.upper()}  "
            f"S={self.S}, K={self.K}, T={self.T}, r={self.r}, σ={self.sigma}, q={self.q}\n"
            f"  Price = {g['price']:.6f}   Δ = {g['delta']:.6f}   "
            f"Γ = {g['gamma']:.6f}   ν = {g['vega']:.6f}   "
            f"Θ = {g['theta']:.6f}   ρ = {g['rho']:.6f}"
        )


# ===========================================================================
# Visualization
# ===========================================================================


def _style_matplotlib_ax(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_facecolor(COLORS["bg"])
    ax.grid(True, linestyle="--", alpha=0.5, color=COLORS["grid"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_payoff(
    option: EuropeanOption,
    path: Path,
    S_range: Optional[np.ndarray] = None,
) -> Path:
    """Payoff at expiry vs. current BSM value (long option)."""
    if S_range is None:
        S_range = np.linspace(0.5 * option.K, 1.5 * option.K, 300)

    payoff = option.intrinsic_value(S_range)
    # P&L at expiry if purchased at today's premium
    premium = float(option.price())
    # Forward value of premium for fair comparison
    fwd_premium = premium * math.exp(option.r * option.T)
    pnl = payoff - fwd_premium

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    ax.plot(S_range, payoff, color=COLORS["call"] if option.option_type == "call" else COLORS["put"],
            lw=2.2, label="Payoff at expiry")
    ax.plot(S_range, pnl, color=COLORS["accent"], lw=2.0, ls="--",
            label=f"P&L (premium fwd = {fwd_premium:.2f})")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(option.K, color=COLORS["strike"], ls=":", lw=1.5, label=f"Strike K={option.K}")
    ax.axvline(option.S, color=COLORS["spot"], ls="--", lw=1.2, label=f"Spot S={option.S}")
    _style_matplotlib_ax(
        ax,
        f"European {option.option_type.title()} Payoff Diagram",
        "Underlying price at expiry ($S_T$)",
        "Payoff / P&L",
    )
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_price_vs_spot(
    option: EuropeanOption,
    path: Path,
    vols: Tuple[float, ...] = (0.15, 0.25, 0.40),
) -> Path:
    """Option price vs. spot for several volatility levels."""
    S_range = np.linspace(0.6 * option.K, 1.4 * option.K, 250)
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)

    cmap = plt.cm.viridis(np.linspace(0.2, 0.85, len(vols)))
    for vol, color in zip(vols, cmap):
        prices = option.copy(sigma=vol).price(S=S_range)
        ax.plot(S_range, prices, lw=2.0, color=color, label=f"σ = {vol:.0%}")

    # Intrinsic envelope
    ax.plot(
        S_range,
        option.intrinsic_value(S_range),
        color="black",
        ls=":",
        lw=1.5,
        label="Intrinsic",
    )
    ax.axvline(option.K, color=COLORS["strike"], ls=":", lw=1.2, alpha=0.8)
    ax.axvline(option.S, color=COLORS["spot"], ls="--", lw=1.2, alpha=0.8)
    _style_matplotlib_ax(
        ax,
        f"BSM {option.option_type.title()} Price vs. Spot (varying σ)",
        "Spot price S",
        "Option price",
    )
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_greeks_profiles(option: EuropeanOption, path: Path) -> Path:
    """Delta, Gamma, Vega, Theta profiles vs. spot."""
    S_range = np.linspace(0.5 * option.K, 1.5 * option.K, 250)
    delta = option.delta(S_range)
    gamma = option.gamma(S_range)
    vega = option.vega(S_range) / 100.0  # per 1%
    theta = option.theta(S_range) / 365.0  # per day

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=150)
    specs = [
        (axes[0, 0], delta, "Delta (Δ)", COLORS["call"]),
        (axes[0, 1], gamma, "Gamma (Γ)", COLORS["accent"]),
        (axes[1, 0], vega, "Vega (per 1% vol)", COLORS["mc"]),
        (axes[1, 1], theta, "Theta (per day)", COLORS["put"]),
    ]
    for ax, series, title, color in specs:
        ax.plot(S_range, series, color=color, lw=2.0)
        ax.axvline(option.K, color=COLORS["strike"], ls=":", lw=1.0)
        ax.axvline(option.S, color=COLORS["spot"], ls="--", lw=1.0)
        ax.axhline(0, color="black", lw=0.6)
        _style_matplotlib_ax(ax, title, "Spot S", title.split()[0])
    fig.suptitle(
        f"Greeks Profiles — {option.option_type.title()}  "
        f"(K={option.K}, T={option.T}, σ={option.sigma:.0%})",
        fontsize=14,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_price_vs_vol(option: EuropeanOption, path: Path) -> Path:
    """Price and vega as functions of volatility."""
    sigmas = np.linspace(0.01, 0.80, 200)
    prices = np.array([option.copy(sigma=s).price() for s in sigmas])
    vegas = np.array([option.copy(sigma=s).vega() for s in sigmas]) / 100.0

    fig, ax1 = plt.subplots(figsize=(9, 5.5), dpi=150)
    ax1.plot(sigmas * 100, prices, color=COLORS["call"], lw=2.2, label="BSM price")
    ax1.set_xlabel("Volatility σ (%)", fontsize=11)
    ax1.set_ylabel("Option price", color=COLORS["call"], fontsize=11)
    ax1.tick_params(axis="y", labelcolor=COLORS["call"])
    ax1.axvline(option.sigma * 100, color=COLORS["spot"], ls="--", lw=1.2,
                label=f"σ = {option.sigma:.0%}")

    ax2 = ax1.twinx()
    ax2.plot(sigmas * 100, vegas, color=COLORS["mc"], lw=2.0, ls="--", label="Vega / 1%")
    ax2.set_ylabel("Vega (per 1%)", color=COLORS["mc"], fontsize=11)
    ax2.tick_params(axis="y", labelcolor=COLORS["mc"])

    ax1.set_facecolor(COLORS["bg"])
    ax1.grid(True, linestyle="--", alpha=0.5, color=COLORS["grid"])
    ax1.set_title("Price & Vega vs. Volatility", fontsize=13, fontweight="bold")
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_mc_convergence(option: EuropeanOption, path: Path, seed: int = 42) -> Path:
    """Monte Carlo running mean with analytical BSM reference."""
    mc = option.monte_carlo_price(
        n_paths=50_000, seed=seed, antithetic=True, return_paths=True
    )
    payoffs = np.asarray(mc["discounted_payoffs"])
    n = len(payoffs)
    running = np.cumsum(payoffs) / np.arange(1, n + 1)
    # Running stderr (approx)
    bsm = float(option.price())

    # Subsample for plotting speed
    idx = np.unique(np.logspace(1, np.log10(n), 400).astype(int) - 1)
    idx = idx[idx < n]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    ax.plot(idx + 1, running[idx], color=COLORS["mc"], lw=1.8, label="MC running mean")
    ax.axhline(bsm, color=COLORS["call"], lw=2.0, label=f"BSM = {bsm:.4f}")
    ax.fill_between(
        idx + 1,
        bsm - 1.96 * float(mc["stderr"]),
        bsm + 1.96 * float(mc["stderr"]),
        color=COLORS["call"],
        alpha=0.12,
        label="Final 95% CI half-width ref.",
    )
    _style_matplotlib_ax(
        ax,
        "Monte Carlo Convergence (antithetic GBM)",
        "Number of paths",
        "Estimated price",
    )
    ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_heatmap_price(option: EuropeanOption, path: Path) -> Path:
    """Price heatmap over (S, σ) grid."""
    S_grid = np.linspace(0.7 * option.K, 1.3 * option.K, 60)
    sig_grid = np.linspace(0.05, 0.60, 50)
    SS, VV = np.meshgrid(S_grid, sig_grid)
    # Vectorized over S for each sigma
    ZZ = np.zeros_like(SS)
    for i, sig in enumerate(sig_grid):
        ZZ[i, :] = option.copy(sigma=float(sig)).price(S=S_grid)

    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    im = ax.contourf(SS, VV * 100, ZZ, levels=25, cmap="RdYlBu_r")
    cs = ax.contour(SS, VV * 100, ZZ, levels=10, colors="k", linewidths=0.4, alpha=0.4)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")
    ax.plot(option.S, option.sigma * 100, "k*", markersize=14, label="Base case")
    ax.axvline(option.K, color="white", ls="--", lw=1.0, alpha=0.7)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Option price")
    _style_matplotlib_ax(
        ax,
        f"Price Heatmap — {option.option_type.title()}",
        "Spot S",
        "Volatility σ (%)",
    )
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plotly_price_surface(option: EuropeanOption, path: Path) -> Path:
    """Interactive 3D price surface (S × σ → price) saved as HTML."""
    S_grid = np.linspace(0.6 * option.K, 1.4 * option.K, 50)
    sig_grid = np.linspace(0.05, 0.60, 40)
    Z = np.zeros((len(sig_grid), len(S_grid)))
    for i, sig in enumerate(sig_grid):
        Z[i, :] = option.copy(sigma=float(sig)).price(S=S_grid)

    fig = go.Figure(
        data=[
            go.Surface(
                x=S_grid,
                y=sig_grid * 100,
                z=Z,
                colorscale="Viridis",
                colorbar=dict(title="Price"),
                hovertemplate="S=%{x:.2f}<br>σ=%{y:.1f}%<br>Price=%{z:.4f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title=dict(
            text=f"BSM {option.option_type.title()} Price Surface — "
            f"K={option.K}, T={option.T}, r={option.r:.1%}",
            x=0.5,
        ),
        scene=dict(
            xaxis_title="Spot S",
            yaxis_title="Volatility σ (%)",
            zaxis_title="Option Price",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        template="plotly_white",
        margin=dict(l=0, r=0, t=50, b=0),
        width=900,
        height=650,
    )
    # Mark base case
    fig.add_trace(
        go.Scatter3d(
            x=[option.S],
            y=[option.sigma * 100],
            z=[float(option.price())],
            mode="markers",
            marker=dict(size=6, color="red", symbol="diamond"),
            name="Base case",
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path


def plotly_sensitivity_dashboard(option: EuropeanOption, path: Path) -> Path:
    """Interactive multi-panel sensitivity dashboard (Plotly HTML)."""
    S_range = np.linspace(0.5 * option.K, 1.5 * option.K, 120)
    price = option.price(S=S_range)
    delta = option.delta(S_range)
    gamma = option.gamma(S_range)
    vega = option.vega(S_range) / 100.0

    T_range = np.linspace(1 / 365, max(option.T * 2, 0.1), 80)
    # Theta / price vs T at fixed S
    prices_T = np.array([option.copy(T=float(t)).price() for t in T_range])
    thetas_T = np.array([option.copy(T=float(t)).theta() for t in T_range]) / 365.0

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Price vs Spot",
            "Delta & Gamma vs Spot",
            "Vega vs Spot (per 1%)",
            "Price & Theta vs Time to Expiry",
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": True}],
            [{"secondary_y": False}, {"secondary_y": True}],
        ],
    )

    fig.add_trace(
        go.Scatter(x=S_range, y=price, name="Price", line=dict(color="#1f77b4", width=2)),
        row=1, col=1,
    )
    fig.add_vline(x=option.K, line_dash="dot", line_color="gray", row=1, col=1)

    fig.add_trace(
        go.Scatter(x=S_range, y=delta, name="Delta", line=dict(color="#2ca02c", width=2)),
        row=1, col=2, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=S_range, y=gamma, name="Gamma", line=dict(color="#9467bd", width=2)),
        row=1, col=2, secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(x=S_range, y=vega, name="Vega", line=dict(color="#ff7f0e", width=2)),
        row=2, col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=T_range,
            y=prices_T,
            name="Price vs T",
            line=dict(color="#1f77b4", width=2),
        ),
        row=2, col=2, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=T_range,
            y=thetas_T,
            name="Theta/day",
            line=dict(color="#d62728", width=2, dash="dash"),
        ),
        row=2, col=2, secondary_y=True,
    )

    fig.update_layout(
        title=dict(
            text=f"Sensitivity Dashboard — European {option.option_type.title()} "
            f"(S={option.S}, K={option.K}, σ={option.sigma:.0%})",
            x=0.5,
        ),
        template="plotly_white",
        height=750,
        width=1000,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.01),
        margin=dict(t=100),
    )
    fig.update_xaxes(title_text="Spot S", row=1, col=1)
    fig.update_xaxes(title_text="Spot S", row=1, col=2)
    fig.update_xaxes(title_text="Spot S", row=2, col=1)
    fig.update_xaxes(title_text="Time to expiry (years)", row=2, col=2)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Delta", row=1, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Gamma", row=1, col=2, secondary_y=True)
    fig.update_yaxes(title_text="Vega", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=2, col=2, secondary_y=False)
    fig.update_yaxes(title_text="Theta/day", row=2, col=2, secondary_y=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")
    return path


# ===========================================================================
# Unit / numerical tests (run as part of main)
# ===========================================================================


def run_tests(verbose: bool = True) -> bool:
    """
    Self-contained validation suite.

    Checks:
    1. Put–call parity residual ~ 0
    2. Known BSM benchmark (Haugh / standard textbook values)
    3. Monte Carlo within ~3 stderr of analytical
    4. Implied vol recovers input sigma
    5. Edge cases: deep ITM/OTM, T→0, σ→0
    6. Greeks sign conventions
    """
    results = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        results.append((name, cond, detail))
        if verbose:
            status = "PASS" if cond else "FAIL"
            print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

    if verbose:
        print("\n" + "=" * 60)
        print("RUNNING VALIDATION SUITE")
        print("=" * 60)

    # --- Base case ---
    opt = EuropeanOption(S=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.0, option_type="call")

    # 1. Put-call parity
    parity = opt.put_call_parity_check(tol=1e-10)
    check(
        "Put–call parity",
        parity["parity_holds"],
        f"residual={parity['residual']:.2e}",
    )

    # 2. Benchmark: ATM call S=K=100, T=1, r=5%, σ=20%, q=0
    # Well-known value ≈ 10.45058357
    bsm_call = float(opt.price())
    check(
        "ATM call price ≈ 10.4506",
        abs(bsm_call - 10.450583572) < 1e-6,
        f"price={bsm_call:.10f}",
    )

    # 3. Put price via parity
    put_px = float(opt.copy(option_type="put").price())
    expected_put = bsm_call - 100 + 100 * math.exp(-0.05)
    check(
        "Put via parity consistency",
        abs(put_px - expected_put) < 1e-10,
        f"put={put_px:.6f}",
    )

    # 4. Monte Carlo accuracy
    mc = opt.monte_carlo_price(n_paths=200_000, seed=7, antithetic=True)
    err = abs(mc["price"] - bsm_call)
    check(
        "MC within 3 stderr of BSM",
        err < 3 * mc["stderr"] + 1e-4,
        f"MC={mc['price']:.4f}, BSM={bsm_call:.4f}, stderr={mc['stderr']:.5f}, |err|={err:.5f}",
    )

    # 5. Implied volatility round-trip
    iv = opt.implied_volatility(bsm_call)
    check(
        "IV recovers σ",
        abs(iv - 0.2) < 1e-6,
        f"IV={iv:.8f}",
    )

    # 6. Greeks signs for long call
    g = opt.greeks()
    check("Call delta in (0,1)", 0 < g["delta"] < 1, f"Δ={g['delta']:.4f}")
    check("Call gamma > 0", g["gamma"] > 0, f"Γ={g['gamma']:.6f}")
    check("Call vega > 0", g["vega"] > 0, f"ν={g['vega']:.4f}")
    check("Call rho > 0", g["rho"] > 0, f"ρ={g['rho']:.4f}")

    # 7. Put delta in (-1, 0)
    put = opt.copy(option_type="put")
    check("Put delta in (-1,0)", -1 < float(put.delta()) < 0, f"Δ={put.delta():.4f}")

    # 8. T → 0: price → intrinsic
    expired_itm = EuropeanOption(S=110, K=100, T=0.0, r=0.05, sigma=0.2, option_type="call")
    check("T=0 ITM call = intrinsic", abs(expired_itm.price() - 10.0) < 1e-12)

    expired_otm = EuropeanOption(S=90, K=100, T=0.0, r=0.05, sigma=0.2, option_type="call")
    check("T=0 OTM call = 0", abs(expired_otm.price()) < 1e-12)

    # 9. Deep OTM put ≈ 0
    deep_otm_put = EuropeanOption(S=200, K=100, T=0.25, r=0.05, sigma=0.15, option_type="put")
    check("Deep OTM put small", deep_otm_put.price() < 0.01, f"P={deep_otm_put.price():.6f}")

    # 10. Input validation
    try:
        EuropeanOption(S=-1, K=100, T=1, r=0.05, sigma=0.2)
        check("Reject negative S", False)
    except ValueError:
        check("Reject negative S", True)

    try:
        EuropeanOption(S=100, K=100, T=1, r=0.05, sigma=0.2).implied_volatility(-1.0)
        check("Reject negative market price for IV", False)
    except ValueError:
        check("Reject negative market price for IV", True)

    # 11. Example parameters (user-specified)
    ex = EuropeanOption(S=150, K=155, T=0.5, r=0.05, sigma=0.25, option_type="call")
    check("Example call prices positive", ex.price() > 0, f"C={ex.price():.4f}")
    check(
        "Example parity",
        ex.put_call_parity_check()["parity_holds"],
        f"res={ex.put_call_parity_check()['residual']:.2e}",
    )

    # 12. Finite-difference Greek vs analytical (delta)
    h = 1e-4
    fd_delta = (ex.copy(S=ex.S + h).price() - ex.copy(S=ex.S - h).price()) / (2 * h)
    check(
        "Delta matches central FD",
        abs(fd_delta - ex.delta()) < 1e-5,
        f"anal={ex.delta():.6f}, FD={fd_delta:.6f}",
    )

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    if verbose:
        print("-" * 60)
        print(f"Results: {passed}/{total} passed")
        print("=" * 60 + "\n")
    return passed == total


# ===========================================================================
# PDF report
# ===========================================================================


def build_pdf_report(
    call: EuropeanOption,
    put: EuropeanOption,
    mc_call: Dict,
    mc_put: Dict,
    parity: Dict,
    iv_call: float,
    plot_paths: Dict[str, Path],
    output: Path = REPORT_PATH,
) -> Path:
    """Generate a multi-page professional PDF report with embedded charts."""
    output.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output),
        pagesize=letter,
        rightMargin=0.7 * inch,
        leftMargin=0.7 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="European Option Pricing & Risk Report",
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="CoverTitle",
            parent=styles["Title"],
            fontSize=22,
            spaceAfter=12,
            textColor=colors.HexColor("#1a365d"),
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverSub",
            parent=styles["Normal"],
            fontSize=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#4a5568"),
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHead",
            parent=styles["Heading1"],
            fontSize=14,
            textColor=colors.HexColor("#1a365d"),
            spaceBefore=14,
            spaceAfter=8,
            borderPadding=3,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyJust",
            parent=styles["Normal"],
            fontSize=9.5,
            leading=13,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.HexColor("#4a5568"),
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Formula",
            parent=styles["Normal"],
            fontSize=9,
            fontName="Courier",
            leading=12,
            leftIndent=10,
            spaceAfter=4,
            textColor=colors.HexColor("#2d3748"),
        )
    )

    story = []
    g_call = call.greeks()
    g_put = put.greeks()

    # ---- Cover ----
    story.append(Spacer(1, 1.2 * inch))
    story.append(Paragraph("European Option Pricing &amp; Risk Toolkit", styles["CoverTitle"]))
    story.append(Paragraph("Black–Scholes–Merton Analytical Engine", styles["CoverSub"]))
    story.append(Spacer(1, 0.25 * inch))
    story.append(
        Paragraph(
            f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            styles["CoverSub"],
        )
    )
    story.append(Spacer(1, 0.4 * inch))

    cover_info = [
        ["Parameter", "Value"],
        ["Spot (S)", f"{call.S:.4f}"],
        ["Strike (K)", f"{call.K:.4f}"],
        ["Time to expiry (T)", f"{call.T:.4f} years"],
        ["Risk-free rate (r)", f"{call.r:.4%}"],
        ["Volatility (σ)", f"{call.sigma:.4%}"],
        ["Dividend yield (q)", f"{call.q:.4%}"],
        ["Moneyness S/K", f"{call.S / call.K:.4f}"],
    ]
    t = Table(cover_info, colWidths=[2.8 * inch, 2.2 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a365d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("ALIGN", (1, 0), (1, -1), "CENTER"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#edf2f7")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.HexColor("#edf2f7"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e0")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 0.5 * inch))
    story.append(
        Paragraph(
            "This report documents closed-form BSM prices, full Greeks, Monte Carlo "
            "cross-validation with standard errors, implied-volatility inversion (Brent), "
            "and put–call parity diagnostics for a European equity option.",
            styles["BodyJust"],
        )
    )
    story.append(PageBreak())

    # ---- Theory recap ----
    story.append(Paragraph("1. Model &amp; Methodology", styles["SectionHead"]))
    story.append(
        Paragraph(
            "Under the Black–Scholes–Merton assumptions the underlying follows geometric "
            "Brownian motion with constant drift and volatility. In the risk-neutral measure:",
            styles["BodyJust"],
        )
    )
    story.append(
        Paragraph("dS_t = (r − q) S_t dt + σ S_t dW_t^Q", styles["Formula"])
    )
    story.append(
        Paragraph(
            "The unique no-arbitrage price of a European claim is the discounted risk-neutral "
            "expectation of its payoff. Closed forms:",
            styles["BodyJust"],
        )
    )
    story.append(
        Paragraph(
            "d1 = [ln(S/K) + (r − q + ½σ²)T] / (σ√T),   d2 = d1 − σ√T",
            styles["Formula"],
        )
    )
    story.append(
        Paragraph(
            "Call = S e^{−qT} N(d1) − K e^{−rT} N(d2)",
            styles["Formula"],
        )
    )
    story.append(
        Paragraph(
            "Put  = K e^{−rT} N(−d2) − S e^{−qT} N(−d1)",
            styles["Formula"],
        )
    )
    story.append(
        Paragraph(
            "Put–call parity:  C − P = S e^{−qT} − K e^{−rT}.  Monte Carlo uses exact GBM "
            "terminal sampling with antithetic variates. Implied volatility is solved with "
            "Brent’s root finder on the map σ ↦ BSM(σ) − market price.",
            styles["BodyJust"],
        )
    )

    # ---- Pricing results ----
    story.append(Paragraph("2. Pricing Results", styles["SectionHead"]))
    price_data = [
        ["Metric", "Call", "Put"],
        ["BSM analytical price", f"{g_call['price']:.6f}", f"{g_put['price']:.6f}"],
        ["Intrinsic value", f"{call.intrinsic_value():.6f}", f"{put.intrinsic_value():.6f}"],
        ["Time value", f"{call.time_value():.6f}", f"{put.time_value():.6f}"],
        [
            "Monte Carlo price",
            f"{mc_call['price']:.6f}",
            f"{mc_put['price']:.6f}",
        ],
        [
            "MC standard error",
            f"{mc_call['stderr']:.6f}",
            f"{mc_put['stderr']:.6f}",
        ],
        [
            "MC 95% CI",
            f"[{mc_call['ci_95_low']:.4f}, {mc_call['ci_95_high']:.4f}]",
            f"[{mc_put['ci_95_low']:.4f}, {mc_put['ci_95_high']:.4f}]",
        ],
        [
            "|MC − BSM|",
            f"{abs(mc_call['price'] - g_call['price']):.6f}",
            f"{abs(mc_put['price'] - g_put['price']):.6f}",
        ],
        ["Implied vol (from BSM price)", f"{iv_call:.6%}", "—"],
    ]
    t2 = Table(price_data, colWidths=[2.4 * inch, 2.1 * inch, 2.1 * inch])
    t2.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.HexColor("#ebf8ff"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#90cdf4")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(t2)
    story.append(Spacer(1, 0.15 * inch))
    story.append(
        Paragraph(
            f"Put–call parity residual = {parity['residual']:.2e} "
            f"({'HOLDS' if parity['parity_holds'] else 'FAILS'} at tol={parity['tol']}). "
            f"C−P = {parity['call_minus_put']:.6f},  "
            f"Se^{{-qT}}−Ke^{{-rT}} = {parity['forward_diff']:.6f}.",
            styles["BodyJust"],
        )
    )

    # ---- Greeks table ----
    story.append(Paragraph("3. Risk Sensitivities (Greeks)", styles["SectionHead"]))
    greeks_data = [
        ["Greek", "Call", "Put", "Interpretation"],
        [
            "Delta (Δ)",
            f"{g_call['delta']:.6f}",
            f"{g_put['delta']:.6f}",
            "∂V/∂S — hedge ratio",
        ],
        [
            "Gamma (Γ)",
            f"{g_call['gamma']:.6f}",
            f"{g_put['gamma']:.6f}",
            "∂²V/∂S² — convexity",
        ],
        [
            "Vega (ν) / 1%",
            f"{g_call['vega_per_1pct']:.6f}",
            f"{g_put['vega_per_1pct']:.6f}",
            "∂V/∂σ per vol point",
        ],
        [
            "Theta (Θ) / day",
            f"{g_call['theta_per_day']:.6f}",
            f"{g_put['theta_per_day']:.6f}",
            "Time decay per day",
        ],
        [
            "Rho (ρ) / 1%",
            f"{g_call['rho_per_1pct']:.6f}",
            f"{g_put['rho_per_1pct']:.6f}",
            "∂V/∂r per rate point",
        ],
        [
            "Vega (absolute)",
            f"{g_call['vega']:.6f}",
            f"{g_put['vega']:.6f}",
            "∂V/∂σ (σ in decimal)",
        ],
        [
            "Theta (per year)",
            f"{g_call['theta']:.6f}",
            f"{g_put['theta']:.6f}",
            "∂V/∂t annualized",
        ],
    ]
    t3 = Table(greeks_data, colWidths=[1.35 * inch, 1.15 * inch, 1.15 * inch, 2.9 * inch])
    t3.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#276749")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("ALIGN", (1, 0), (2, -1), "CENTER"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.HexColor("#f0fff4"), colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9ae6b4")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(t3)
    story.append(PageBreak())

    # ---- Charts ----
    story.append(Paragraph("4. Visual Analytics", styles["SectionHead"]))
    chart_order = [
        ("payoff", "Payoff &amp; P&amp;L at Expiry"),
        ("price_vs_spot", "Price vs. Spot for Varying Volatilities"),
        ("greeks", "Greeks Profiles vs. Spot"),
        ("price_vs_vol", "Price &amp; Vega vs. Volatility"),
        ("heatmap", "Price Heatmap (S × σ)"),
        ("mc_convergence", "Monte Carlo Convergence"),
    ]
    for key, title in chart_order:
        p = plot_paths.get(key)
        if p is None or not Path(p).exists():
            continue
        story.append(Paragraph(title, styles["Heading2"]))
        img = Image(str(p), width=6.4 * inch, height=3.9 * inch)
        img.hAlign = "CENTER"
        story.append(img)
        story.append(Spacer(1, 0.12 * inch))

    story.append(PageBreak())

    # ---- Summary ----
    story.append(Paragraph("5. Summary", styles["SectionHead"]))
    story.append(
        Paragraph(
            f"For the base case (S={call.S}, K={call.K}, T={call.T}, r={call.r}, "
            f"σ={call.sigma}) the European call is valued at "
            f"<b>{g_call['price']:.4f}</b> and the put at <b>{g_put['price']:.4f}</b>. "
            f"Monte Carlo (antithetic, n≈{int(mc_call['n_effective']):,}) recovers both "
            f"prices within sampling error. Implied volatility from the model price "
            f"round-trips to the input σ ({iv_call:.4%}). "
            f"Parity residual is {parity['residual']:.2e}.",
            styles["BodyJust"],
        )
    )
    story.append(Spacer(1, 0.3 * inch))
    story.append(
        Paragraph(
            "— End of Report —",
            styles["Small"],
        )
    )

    doc.build(story)
    return output


# ===========================================================================
# Main demonstration
# ===========================================================================


def main() -> int:
    """Run full demonstration: price, test, plot, report."""
    print("=" * 70)
    print("  EUROPEAN OPTION PRICING & RISK TOOLKIT")
    print("  Black–Scholes–Merton  |  Analytical + Monte Carlo + IV + Greeks")
    print("=" * 70)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Realistic equity-like example parameters ----
    S, K, T, r, sigma, q = 150.0, 155.0, 0.5, 0.05, 0.25, 0.0

    call = EuropeanOption(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type="call")
    put = EuropeanOption(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type="put")

    print("\n── Market Parameters ──")
    print(f"  Spot S          = {S}")
    print(f"  Strike K        = {K}")
    print(f"  Expiry T        = {T} years")
    print(f"  Rate r          = {r:.2%}")
    print(f"  Volatility σ    = {sigma:.2%}")
    print(f"  Dividend q      = {q:.2%}")
    print(f"  Moneyness S/K   = {S/K:.4f}")

    print("\n── Analytical Prices & Greeks ──")
    print(call)
    print(put)

    g_call = call.greeks()
    g_put = put.greeks()

    print("\n  Greeks table (market conventions):")
    df = pd.DataFrame(
        {
            "Call": [
                g_call["price"],
                g_call["delta"],
                g_call["gamma"],
                g_call["vega_per_1pct"],
                g_call["theta_per_day"],
                g_call["rho_per_1pct"],
            ],
            "Put": [
                g_put["price"],
                g_put["delta"],
                g_put["gamma"],
                g_put["vega_per_1pct"],
                g_put["theta_per_day"],
                g_put["rho_per_1pct"],
            ],
        },
        index=["Price", "Delta", "Gamma", "Vega/1%", "Theta/day", "Rho/1%"],
    )
    print(df.to_string(float_format=lambda x: f"{x:12.6f}"))

    # ---- Monte Carlo ----
    print("\n── Monte Carlo (antithetic GBM, 100k base paths) ──")
    mc_call = call.monte_carlo_price(n_paths=100_000, seed=42, antithetic=True)
    mc_put = put.monte_carlo_price(n_paths=100_000, seed=42, antithetic=True)
    print(
        f"  Call MC = {mc_call['price']:.6f}  ± {mc_call['stderr']:.6f}  "
        f"95% CI [{mc_call['ci_95_low']:.4f}, {mc_call['ci_95_high']:.4f}]  "
        f"|err|={abs(mc_call['price']-g_call['price']):.6f}"
    )
    print(
        f"  Put  MC = {mc_put['price']:.6f}  ± {mc_put['stderr']:.6f}  "
        f"95% CI [{mc_put['ci_95_low']:.4f}, {mc_put['ci_95_high']:.4f}]  "
        f"|err|={abs(mc_put['price']-g_put['price']):.6f}"
    )

    # ---- Implied vol ----
    print("\n── Implied Volatility (Brent) ──")
    iv_call = call.implied_volatility(g_call["price"])
    # Slightly perturbed market price demo
    mkt = g_call["price"] * 1.02
    iv_mkt = call.implied_volatility(mkt)
    print(f"  IV from model price        = {iv_call:.6%}  (input σ = {sigma:.2%})")
    print(f"  IV from +2% market premium = {iv_mkt:.6%}  (price = {mkt:.4f})")

    # ---- Parity ----
    print("\n── Put–Call Parity ──")
    parity = call.put_call_parity_check()
    print(f"  C − P              = {parity['call_minus_put']:.8f}")
    print(f"  S e^{{-qT}} − K e^{{-rT}} = {parity['forward_diff']:.8f}")
    print(f"  Residual           = {parity['residual']:.2e}")
    print(f"  Parity holds       = {parity['parity_holds']}")

    # ---- Tests ----
    ok = run_tests(verbose=True)
    if not ok:
        print("WARNING: Some validation checks failed.", file=sys.stderr)

    # ---- Plots ----
    print("── Generating visualizations ──")
    plot_paths: Dict[str, Path] = {}
    plot_paths["payoff"] = plot_payoff(call, PLOTS_DIR / "payoff_diagram.png")
    print(f"  ✓ {plot_paths['payoff'].name}")
    plot_paths["price_vs_spot"] = plot_price_vs_spot(
        call, PLOTS_DIR / "price_vs_spot.png"
    )
    print(f"  ✓ {plot_paths['price_vs_spot'].name}")
    plot_paths["greeks"] = plot_greeks_profiles(call, PLOTS_DIR / "greeks_profiles.png")
    print(f"  ✓ {plot_paths['greeks'].name}")
    plot_paths["price_vs_vol"] = plot_price_vs_vol(call, PLOTS_DIR / "price_vs_vol.png")
    print(f"  ✓ {plot_paths['price_vs_vol'].name}")
    plot_paths["heatmap"] = plot_heatmap_price(call, PLOTS_DIR / "price_heatmap.png")
    print(f"  ✓ {plot_paths['heatmap'].name}")
    plot_paths["mc_convergence"] = plot_mc_convergence(
        call, PLOTS_DIR / "mc_convergence.png"
    )
    print(f"  ✓ {plot_paths['mc_convergence'].name}")

    html_surface = plotly_price_surface(call, PLOTS_DIR / "price_surface.html")
    print(f"  ✓ {html_surface.name} (interactive)")
    html_dash = plotly_sensitivity_dashboard(
        call, PLOTS_DIR / "sensitivity_dashboard.html"
    )
    print(f"  ✓ {html_dash.name} (interactive)")

    # Also save put payoff for completeness
    plot_payoff(put, PLOTS_DIR / "payoff_diagram_put.png")
    print("  ✓ payoff_diagram_put.png")

    # ---- PDF ----
    print("\n── Building PDF report ──")
    report = build_pdf_report(
        call=call,
        put=put,
        mc_call=mc_call,
        mc_put=mc_put,
        parity=parity,
        iv_call=iv_call,
        plot_paths=plot_paths,
        output=REPORT_PATH,
    )
    print(f"  ✓ {report}")

    # ---- Export summary CSV ----
    summary_path = PROJECT_ROOT / "results_summary.csv"
    call.summary().to_csv(summary_path, index=False)
    print(f"  ✓ {summary_path.name}")

    print("\n" + "=" * 70)
    print("  DONE — project artifacts written to:")
    print(f"    {PROJECT_ROOT}")
    print("=" * 70)
    print(f"\n  Open the report : {REPORT_PATH}")
    print(f"  Interactive 3D  : {html_surface}")
    print(f"  Dashboard       : {html_dash}")
    print()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
