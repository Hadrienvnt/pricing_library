# Python Options Pricing Library

A flexible, object-oriented Python library for pricing basic financial derivatives. This project implements various pricing models (Black-Scholes, Monte Carlo) and supports multiple asset classes including European-style, Asian-style, and Barrier options.

See notebooks for examples of use of this library.

## 🚀 Project Overview

This engine is designed with modularity in mind, separating **Assets** (contracts) from **Pricing Methods** (engines).

### Key Features Implemented:
* **Asset Classes:**
    * **European Options:** Standard Call/Put.
    * **Asian Options:** Arithmetic Average (Fixed Strike).
    * **Barrier Options:** Up-and-Out, Down-and-Out, Up-and-In, Down-and-In.
* **Pricing Engines:**
    * **Black-Scholes (Analytical):** Exact pricing for European options.
    * **Monte Carlo Simulation:** * Vectorized Numpy implementation for high performance.
        * Dynamic time-stepping (automatic adjustment based on maturity).
        * Support for European, Asian (Path-dependent), and Barrier (Continuous monitoring) options.
* **Market Data:**
    * **Yield Curve:** Discounting using linear interpolation of market rates (Zero-Coupon).
* **Validation:**
    * Automated tests for Put-Call Parity.
    * Convergence analysis for Monte Carlo simulations.
    * Parity checks for Barrier options (In + Out = European).

## ✅ Running Validation Tests

This project emphasizes reliability. We use `src/validation_tests.py` to check:
1. **Statistical Convergence:** Ensuring Monte Carlo error scales correctly.
2. **Arbitrage Conditions:** Put-Call Parity, Martingale property.
3. **Exotic Constraints:** Asian price bounds and Barrier In-Out parity.

See the `pricer_validation` notebook for validation results and graphs.


## 📂 Project Structure

```text
.
├── src/                        # Core library code
│   ├── asset_class.py          # Option definitions (European, Asian, Barrier)
│   ├── market_data.py          # Yield Curve & Market data handling
│   ├── monte_carlo.py          # Monte Carlo engines (Vectorized)
│   ├── black_scholes.py        # Analytical engines
│   └── validation_tests.py     # Validation & sanity checks
│
├── exotic_pricing.ipynb        # Example: Pricing Asian & Barrier options
├── pricer_validation.ipynb     # Example: Running parity tests & convergence analysis
├── vanilla_with_yc.ipynb       # Example: Yield Curve construction and use for pricing vanilla options
│
├── .gitignore                  # Git ignore rules
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation

```

## ⚙️ Setup 

1. Clone the repository
```bash
git clone https://github.com/Hadrienvnt/pricing_library.git
cd pricing_library
``` 

2. Add virtual environment
```bash
python -m venv .venv
```

- For Windows:
```bash
source .venv/Scripts/activate
```
- For Mac/Linux
```bash
source .venv/bin/activate
```


3. Download dependancies
```bash
pip install -r requirements.txt
```
