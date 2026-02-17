import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from enum import Enum
from typing import List, Dict, Optional

class MarketDataError(Exception):
    """Custom exception for market data related errors."""
    pass

class YieldCurveMethod(Enum):
    """
    Method used to construct the yield curve.
    """
    LINEAR = "linear"
    NELSON_SIEGEL = "nelson_siegel"
    SVENSSON = "svensson"

class YieldCurve:
    """
    Represents a term structure of interest rates.
    Can be constructed using interpolation or parametric fitting (Nelson-Siegel / Svensson).
    Includes visualization capabilities.
    """

    def __init__(self, tenors: List[float], rates: List[float], method: YieldCurveMethod = YieldCurveMethod.LINEAR):
        """
        Initializes and calibrates the yield curve based on the chosen method.
        """
        if len(tenors) != len(rates):
            raise MarketDataError("Tenors and rates must have the same length.")
        if any(t < 0 for t in tenors):
            raise MarketDataError("Tenors must be non-negative.")

        # Data preparation (sorting)
        self._raw_tenors = np.array(tenors)
        self._raw_rates = np.array(rates)
        sorted_indices = np.argsort(self._raw_tenors)
        self._tenors = self._raw_tenors[sorted_indices]
        self._rates = self._raw_rates[sorted_indices]
        
        self.method = method
        self._params = {} 

        # Calibration / Construction
        if self.method == YieldCurveMethod.LINEAR:
            self._build_linear()
        elif self.method == YieldCurveMethod.NELSON_SIEGEL:
            self._calibrate_nelson_siegel()
        elif self.method == YieldCurveMethod.SVENSSON:
            self._calibrate_svensson()
        else:
            raise NotImplementedError(f"Method {method} not implemented.")

    def get_rate(self, t: float) -> float:
        """
        Returns the zero rate for maturity t using the selected model.
        """
        if t < 0:
            raise MarketDataError(f"Cannot retrieve rate for negative time: {t}")
        
        if t == 0:
            return self._rates[0] if len(self._rates) > 0 else 0.0

        if self.method == YieldCurveMethod.LINEAR:
            return float(self._interpolator(t))
        
        elif self.method == YieldCurveMethod.NELSON_SIEGEL:
            return self._nelson_siegel_formula(t, **self._params)
        
        elif self.method == YieldCurveMethod.SVENSSON:
            return self._svensson_formula(t, **self._params)
        
        return 0.0

    def get_discount_factor(self, t: float) -> float:
        """Calculates D(t) = exp(-r(t) * t)."""
        r = self.get_rate(t)
        return np.exp(-r * t)

    def plot(self, title: Optional[str] = None):
        """
        Visualizes the fitted yield curve against the market data points.
        """
        # Define visualization range (add 20% extrapolation to see trend)
        t_max = self._tenors[-1] * 1.2
        t_grid = np.linspace(0, t_max, 200)
        r_grid = [self.get_rate(t) for t in t_grid]

        plt.figure(figsize=(10, 5))
        
        # Plot the fitted model
        plt.plot(t_grid, r_grid, label=f'Fitted Curve ({self.method.value})', 
                 color='blue', linewidth=2)
        
        # Plot the original market points
        plt.scatter(self._tenors, self._rates, color='red', marker='x', 
                    s=80, label='Market Data Inputs', zorder=5)
        
        # Styling
        final_title = title if title else f"Yield Curve Fitting ({self.method.name})"
        plt.title(final_title)
        plt.xlabel('Maturity (Years)')
        plt.ylabel('Zero Rate (Annualized)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.show()

    # --- INTERNAL BUILDERS ---

    def _build_linear(self):
        self._interpolator = interp1d(
            self._tenors, self._rates, kind='linear', fill_value="extrapolate"
        )

    # --- NELSON-SIEGEL ---
    
    def _nelson_siegel_formula(self, t: float, beta0: float, beta1: float, beta2: float, tau: float) -> float:
        ratio = t / tau
        term1 = (1 - np.exp(-ratio)) / ratio
        term2 = term1 - np.exp(-ratio)
        return beta0 + beta1 * term1 + beta2 * term2

    def _calibrate_nelson_siegel(self):
        def objective(params):
            b0, b1, b2, tau = params
            if tau <= 0: return 1e10
            model_rates = [self._nelson_siegel_formula(t, b0, b1, b2, tau) for t in self._tenors]
            return np.sum((model_rates - self._rates) ** 2)

        initial_guess = [self._rates[-1], self._rates[0] - self._rates[-1], 0.0, 1.0]
        result = minimize(objective, initial_guess, method='Nelder-Mead')
        self._params = {'beta0': result.x[0], 'beta1': result.x[1], 'beta2': result.x[2], 'tau': result.x[3]}

    # --- SVENSSON ---

    def _svensson_formula(self, t: float, beta0: float, beta1: float, beta2: float, beta3: float, tau1: float, tau2: float) -> float:
        ratio1 = t / tau1
        ratio2 = t / tau2
        term1 = (1 - np.exp(-ratio1)) / ratio1
        term2 = term1 - np.exp(-ratio1)
        term3 = ((1 - np.exp(-ratio2)) / ratio2) - np.exp(-ratio2)
        return beta0 + beta1 * term1 + beta2 * term2 + beta3 * term3

    def _calibrate_svensson(self):
        def objective(params):
            b0, b1, b2, b3, t1, t2 = params
            if t1 <= 0 or t2 <= 0: return 1e10
            model_rates = [self._svensson_formula(t, b0, b1, b2, b3, t1, t2) for t in self._tenors]
            return np.sum((model_rates - self._rates) ** 2)

        initial_guess = [self._rates[-1], self._rates[0] - self._rates[-1], 0.0, 0.0, 1.0, 3.0]
        result = minimize(objective, initial_guess, method='Nelder-Mead', options={'maxiter': 5000})
        self._params = {
            'beta0': result.x[0], 'beta1': result.x[1], 'beta2': result.x[2], 'beta3': result.x[3],
            'tau1': result.x[4], 'tau2': result.x[5]
        }