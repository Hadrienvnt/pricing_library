# Core Pricing Library (`src`)

This directory contains the implementation of the financial pricing engine. The architecture follows the **Strategy Design Pattern**, decoupling financial contracts (**Assets**) from the mathematical models used to price them (**PricingMethods**).

## 📂 Module Overview

| File | Description | Key Classes |
| :--- | :--- | :--- |
| **`asset_class.py`** | Defines financial instruments. Contains contract specifications (Strike, Expiry, Barrier levels). | `EuropeanOption`, `AsianOption`, `BarrierOption`, `Asset` (Abstract), `PricingMethod` (Abstract) |
| **`black_scholes.py`** | Analytical closed-form solutions for standard European options. | `BlackScholesPricer` |
| **`monte_carlo.py`** | Numerical integration using Geometric Brownian Motion (GBM). Vectorized for performance. | `EuropeanOptionMCPricer`, `AsianOptionMCPricer`, `BarrierOptionMCPricer` |
| **`market_data.py`** | Handles interest rate term structures and discounting. | `YieldCurve`, `YieldCurveMethod` (Linear, Nelson-Siegel, Svensson) |
| **`validation_tests.py`** | Test suite for checking model consistency and parity relationships. | `VanillaValidator`, `AsianValidator`, `BarrierValidator` |

---

## 🏗️ Architecture: The Strategy Pattern

The core philosophy of this engine is **Dependency Injection**. An `Asset` does not know how to price itself; it delegates the computation to a `PricingMethod`.

### Class Diagram Logic
```text
      +----------------+            +---------------------+
      |     Asset      | <--------> |    PricingMethod    |
      +----------------+            +---------------------+
      | - strike       |            | + calculate(asset)  |
      | - expiry       |            +----------^----------+
      | + price()      |                       |
      +-------^--------+            ___________|___________
              |                    |                       |
      +-------+--------+   +-------+--------+    +---------+----------+
      | EuropeanOption |   |   BSPricer     |    |   MCPricer         |
      +----------------+   +----------------+    +--------------------+
```

## ⏩ Usage Flow
The pipeline to pricing an asset is always the same:
1. Instantiate an Asset: Define what you want to price.
2. Instantiate an Pricer: Define how you want to price it (Market Data is injected here).
3. Link: Inject the Engine into the Asset.
4. Compute: Call asset.price(spot).

Example: 

```python
# Pseudo-code 
asset = AsianOption(K=100, T=1.0)                 # The Contract
engine = AsianOptionMCPricer(vol=0.2, r=0.05)     # The Algorithm
asset.set_pricing_method(engine)                  # The Link
price = asset.price(spot=100)                     # The Calculation
```
> NB: This is a pseudo-code, meaning that the instantiation of asset or pricer might use more arguments.


## 🧠 Implementation Details

Monte Carlo Pricers (monte_carlo.py)
* Vectorization: Uses numpy arrays to simulate $N$ paths simultaneously.
* Time Stepping: Standard dt derived from maturity and step count.
    * Asian Options: Accumulates the average price along the path.
    * Barrier Options: Checks barrier conditions at every time step (discrete monitoring approximation).
* Convergence: Standard Error is calculated for every pricing request to quantify uncertainty.

Market Data (market_data.py)
* Handles Zero-Coupon Curves.
* Supports construction from raw market points (Tenors vs Rates).
* Interpolation Methods:
    * LINEAR: Simple linear interpolation between points.
    * NELSON_SIEGEL: Parametric fitting (better for smoothing).
    * SVENSSON: Extension of Nelson-Siegel for more complex curves.

Validation (validation_tests.py)
* Model Agnostic: Validators accept any pricer implementing the PricingMethod interface.
* Key Tests:
    * Put-Call Parity: $C - P = S - Ke^{-rT}$
    * Barrier Parity: $In + Out = European$
    * Asian Bounds: $Asian <= European$ (for calls)