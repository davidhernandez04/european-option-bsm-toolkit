# European Option Pricing & Risk Toolkit

**Black–Scholes–Merton (BSM) analytical engine** with full Greeks, Monte Carlo simulation, implied volatility (Brent), put–call parity diagnostics, interactive Plotly charts, and a multi-page PDF risk report.

Built as a **portfolio / interview-ready** quant project: clean API, type hints, validation, vectorized NumPy, numerical tests, and professional deliverables.

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
# Optional: create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
python option_pricer.py
```

Open:

| Artifact | Description |
|----------|-------------|
| `Quant_Option_Pricing_Report.pdf` | Parameters, prices, Greeks, charts |
| `plots/price_surface.html` | 3D price surface \(S \times \sigma\) |
| `plots/sensitivity_dashboard.html` | Interactive Greeks / sensitivity panels |

---

## Example parameters (equity-like)

| Parameter | Symbol | Value |
|-----------|--------|------:|
| Spot | \(S\) | 150 |
| Strike | \(K\) | 155 |
| Time to expiry | \(T\) | 0.5 years |
| Risk-free rate | \(r\) | 5% |
| Volatility | \(\sigma\) | 25% |
| Dividend yield | \(q\) | 0 |

Slightly OTM call / ITM put — a realistic mid-vol equity single-name setup.

---

## Features

### `EuropeanOption` class

| Method | What it does |
|--------|----------------|
| `price()` | Closed-form BSM (vectorized over \(S\), \(\sigma\), \(T\)) |
| `delta()`, `gamma()`, `vega()`, `theta()`, `rho()` | Full first- and second-order Greeks |
| `greeks()` | Dict with absolute + market conventions (per 1%, per day) |
| `monte_carlo_price()` | Risk-neutral GBM + antithetic variates + SE & 95% CI |
| `implied_volatility()` | Brent root find (`scipy.optimize.brentq`) |
| `put_call_parity_check()` | Residual of \(C - P = Se^{-qT} - Ke^{-rT}\) |
| `summary()` | Pandas table of params + risk metrics |
| `copy(**kwargs)` | Immutable-style scenario overrides |

### Visualizations

- **Payoff / P&L** diagram at expiry  
- **Price vs spot** for multiple vols  
- **Greeks profiles** (Δ, Γ, ν, Θ)  
- **Price & vega vs volatility**  
- **Heatmap** over \((S, \sigma)\)  
- **MC convergence** vs analytical BSM  
- **Plotly** 3D surface + sensitivity dashboard (HTML)

### Report

`reportlab` multi-page PDF: cover parameters, methodology, pricing table, Greeks table, embedded charts, summary & extension roadmap.

### Tests (run automatically)

- Put–call parity residual ~ machine epsilon  
- ATM benchmark \(\approx 10.4506\) (classic \(S=K=100\), \(T=1\), \(r=5\%\), \(\sigma=20\%\))  
- MC within ~3 standard errors of BSM  
- IV round-trip recovers input \(\sigma\)  
- Edge cases: \(T=0\), deep OTM, input validation  
- Finite-difference Δ vs analytical Δ  

---

## Theory recap (interview-ready)

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

### Greeks (intuition)

| Greek | Call intuition |
|-------|----------------|
| **Delta** | Hedge ratio; \(\in (0,1)\) for long calls |
| **Gamma** | Convexity; peaks near ATM, short-dated |
| **Vega** | Vol sensitivity; peaks near ATM |
| **Theta** | Time decay; usually negative for long options |
| **Rho** | Rate sensitivity; positive for calls |

### Monte Carlo

Terminal sampling of GBM is **exact** for Europeans (no time-discretization bias when `n_steps=1`). Antithetic variates pair \(Z\) and \(-Z\) to reduce variance. Report the sample standard error:

\[
\mathrm{SE} = \frac{s}{\sqrt{N}}, \qquad
95\%\ \mathrm{CI} \approx \hat{V} \pm 1.96\,\mathrm{SE}
\]

### Implied volatility

Invert \(\sigma \mapsto \mathrm{BSM}(\sigma)\) with **Brent’s method** on a no-arbitrage bracket (intrinsic lower bound to discounted asset / strike upper bound).

---

## Usage snippets

```python
from option_pricer import EuropeanOption

opt = EuropeanOption(S=150, K=155, T=0.5, r=0.05, sigma=0.25, option_type="call")

print(opt.price())           # analytical premium
print(opt.greeks())          # full risk pack
print(opt.monte_carlo_price(n_paths=100_000))
print(opt.implied_volatility(market_price=12.5))
print(opt.put_call_parity_check())

# Vectorized price surface
import numpy as np
S_grid = np.linspace(100, 200, 50)
prices = opt.price(S=S_grid)
```

### FX / Garman–Kohlhagen

Set `q` equal to the **foreign** risk-free rate and interpret `r` as the domestic rate — same closed forms.

```python
fx_call = EuropeanOption(
    S=1.10, K=1.12, T=0.25, r=0.04, sigma=0.10, q=0.02, option_type="call"
)
```

---

## Extension ideas (next projects)

1. **American options** — CRR binomial tree or PSOR PDE solver; early-exercise premium  
2. **Local / stochastic vol** — Dupire, Heston (COS / MC)  
3. **Path-dependent** — Asian, barrier, lookback via MC  
4. **Smile calibration** — SABR / SVI fit to market quotes  
5. **Portfolio Greeks** — multi-option book, scenario P&L, stress grids  
6. **Live data** — pull chain from a free API, compute IV surface  
7. **Unit tests package** — `pytest` + coverage badges for GitHub  

The class boundary (`price`, `greeks`, `monte_carlo_price`) is designed so American / exotic engines can share the same reporting and plotting layer.

---

## How this demonstrates quant skills

| Skill | Where it shows up |
|-------|-------------------|
| Derivatives pricing theory | BSM derivation use, parity, RN expectation |
| Numerical methods | Brent IV, antithetic MC, FD Greek check |
| Software engineering | Typed API, validation, dataclasses, modular plots/report |
| Risk management | Full Greeks + market quoting conventions |
| Communication | PDF report + README + interactive charts |
| Interview talking points | Variance reduction, no-arb bounds for IV, edge cases at \(T=0\) |

**Suggested resume bullet:**

> Built a production-style European option pricing toolkit (Black–Scholes–Merton) with analytical Greeks, antithetic Monte Carlo (standard errors), Brent implied-vol solver, put–call parity diagnostics, Plotly surfaces, and automated PDF risk reports.

---

## Dependencies

- **numpy** — vectorized arrays / RNG  
- **scipy** — `norm`, `brentq`  
- **pandas** — summary tables  
- **matplotlib** — static publication charts  
- **plotly** — interactive HTML  
- **reportlab** — multi-page PDF  

---

## License

MIT — free to use in portfolios, interviews, and coursework. Not investment advice; for educational / demonstration purposes only.

---

## Author notes

Designed to be **read in a hiring loop**: open `option_pricer.py` at the `EuropeanOption` class, skim the validation suite, then open the PDF and the HTML surface. One command (`python option_pricer.py`) regenerates everything.
