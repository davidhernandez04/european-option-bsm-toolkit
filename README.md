# European Option Pricing & Risk Toolkit

Black–Scholes–Merton (BSM) analytical pricing with full Greeks, Monte Carlo simulation, implied volatility (Brent), put–call parity checks, interactive Plotly charts, and a multi-page PDF risk report.

```
european_option_bsm_toolkit/
├── option_pricer.py                 # Main module (run this)
├── requirements.txt
├── README.md
├── results_summary.csv              # Generated summary table
├── Quant_Option_Pricing_Report.pdf  # Multi-page report
└── plots/
    ├── payoff_diagram.png
    ├── payoff_diagram_put.png
    ├── price_vs_spot.png
    ├── greeks_profiles.png
    ├── price_vs_vol.png
    ├── price_heatmap.png
    ├── mc_convergence.png
    ├── price_surface.html           # Interactive 3D surface
    └── sensitivity_dashboard.html   # Interactive multi-panel
```

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python option_pricer.py
```

| Artifact | Description |
|----------|-------------|
| `Quant_Option_Pricing_Report.pdf` | Parameters, prices, Greeks, charts |
| `plots/price_surface.html` | 3D price surface \(S \times \sigma\) |
| `plots/sensitivity_dashboard.html` | Interactive Greeks / sensitivity panels |

---

## Example parameters

| Parameter | Symbol | Value |
|-----------|--------|------:|
| Spot | \(S\) | 150 |
| Strike | \(K\) | 155 |
| Time to expiry | \(T\) | 0.5 years |
| Risk-free rate | \(r\) | 5% |
| Volatility | \(\sigma\) | 25% |
| Dividend yield | \(q\) | 0 |

---

## Features

### `EuropeanOption` class

| Method | Description |
|--------|-------------|
| `price()` | Closed-form BSM (vectorized over \(S\), \(\sigma\), \(T\)) |
| `delta()`, `gamma()`, `vega()`, `theta()`, `rho()` | First- and second-order Greeks |
| `greeks()` | Dict with absolute and market conventions (per 1%, per day) |
| `monte_carlo_price()` | Risk-neutral GBM with antithetic variates, SE, and 95% CI |
| `implied_volatility()` | Brent root find (`scipy.optimize.brentq`) |
| `put_call_parity_check()` | Residual of \(C - P = Se^{-qT} - Ke^{-rT}\) |
| `summary()` | Pandas table of parameters and risk metrics |
| `copy(**kwargs)` | Return a copy with selected fields overridden |

### Visualizations

- Payoff / P&L diagram at expiry
- Price vs spot for multiple volatilities
- Greeks profiles (Δ, Γ, ν, Θ)
- Price and vega vs volatility
- Heatmap over \((S, \sigma)\)
- Monte Carlo convergence vs analytical BSM
- Plotly 3D surface and sensitivity dashboard (HTML)

### Report

Multi-page PDF (`reportlab`): market parameters, methodology, pricing results, Greeks table, and embedded charts.

### Tests (run automatically)

- Put–call parity residual near machine epsilon
- ATM benchmark \(\approx 10.4506\) (\(S=K=100\), \(T=1\), \(r=5\%\), \(\sigma=20\%\))
- Monte Carlo within ~3 standard errors of BSM
- Implied vol round-trip recovers input \(\sigma\)
- Edge cases: \(T=0\), deep OTM, input validation
- Finite-difference Δ vs analytical Δ

---

## Theory

### Dynamics

Under the risk-neutral measure \(\mathbb{Q}\):

\[
dS_t = (r - q)\, S_t\, dt + \sigma\, S_t\, dW_t^{\mathbb{Q}}
\]

### Closed-form European prices

\[
d_1 = \frac{\ln(S/K) + (r - q + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \quad
d_2 = d_1 - \sigma\sqrt{T}
\]

\[
\begin{aligned}
C &= S e^{-qT} N(d_1) - K e^{-rT} N(d_2) \\
P &= K e^{-rT} N(-d_2) - S e^{-qT} N(-d_1)
\end{aligned}
\]

### Put–call parity

\[
C - P = S e^{-qT} - K e^{-rT}
\]

### Greeks

| Greek | Call intuition |
|-------|----------------|
| **Delta** | Hedge ratio; \(\in (0,1)\) for long calls |
| **Gamma** | Convexity; peaks near ATM, short-dated |
| **Vega** | Volatility sensitivity; peaks near ATM |
| **Theta** | Time decay; usually negative for long options |
| **Rho** | Rate sensitivity; positive for calls |

### Monte Carlo

Terminal sampling of GBM is exact for Europeans when `n_steps=1`. Antithetic variates pair \(Z\) and \(-Z\) to reduce variance. The sample standard error is:

\[
\mathrm{SE} = \frac{s}{\sqrt{N}}, \qquad
95\%\ \mathrm{CI} \approx \hat{V} \pm 1.96\,\mathrm{SE}
\]

### Implied volatility

Invert \(\sigma \mapsto \mathrm{BSM}(\sigma)\) with Brent’s method on a no-arbitrage bracket (intrinsic lower bound to discounted asset / strike upper bound).

---

## Usage

```python
from option_pricer import EuropeanOption

opt = EuropeanOption(S=150, K=155, T=0.5, r=0.05, sigma=0.25, option_type="call")

print(opt.price())
print(opt.greeks())
print(opt.monte_carlo_price(n_paths=100_000))
print(opt.implied_volatility(market_price=12.5))
print(opt.put_call_parity_check())

import numpy as np
S_grid = np.linspace(100, 200, 50)
prices = opt.price(S=S_grid)
```

### FX (Garman–Kohlhagen)

Set `q` to the foreign risk-free rate and `r` to the domestic rate:

```python
fx_call = EuropeanOption(
    S=1.10, K=1.12, T=0.25, r=0.04, sigma=0.10, q=0.02, option_type="call"
)
```

---

## Dependencies

- **numpy** — arrays and RNG
- **scipy** — `norm`, `brentq`
- **pandas** — summary tables
- **matplotlib** — static charts
- **plotly** — interactive HTML
- **reportlab** — PDF report

---

## License

MIT. Not investment advice; for educational purposes only.
