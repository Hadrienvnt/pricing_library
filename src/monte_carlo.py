import numpy as np
from typing import Dict, Union, Optional

from src.asset_class import PricingMethod, Asset, EuropeanOption, OptionType,AsianOption,BarrierOption,BarrierType
from src.market_data import YieldCurve

# --- EUROPEAN OPTIONS ---

class EuropeanOptionMCPricer(PricingMethod):
    """
    Specific Monte Carlo Engine for European Options.
    Supports YieldCurve for deterministic rates.
    """

    def __init__(self, volatility: float, risk_free_rate: Union[float, YieldCurve], 
                 n_simulations: int = 1e6, dividend_yield: float = 0.0):
        self.volatility = volatility
        self.rate_provider = risk_free_rate
        self.n_sims = int(n_simulations)
        self.dividend = dividend_yield

    def calculate(self, asset: Asset, spot_price: float) -> Dict[str, float]:
        if not isinstance(asset, EuropeanOption):
            raise TypeError("EuropeanMCEngine only supports EuropeanOption assets.")

        S = spot_price
        K = asset.strike
        T = asset.expiry
        q = self.dividend
        sigma = self.volatility
        N = self.n_sims

        # --- DYNAMIC RATE HANDLING ---
        if isinstance(self.rate_provider, YieldCurve):
            r = self.rate_provider.get_rate(T)
        else:
            r = self.rate_provider
        # -----------------------------

        if T <= 0:
             return {"value": asset.payoff(S), "std_error": 0.0}

        # Vectorized Simulation
        z = np.random.standard_normal(N)

        # Use the specific rate r for this maturity T
        drift = (r - q - 0.5 * sigma ** 2) * T
        diffusion = sigma * np.sqrt(T) * z
        
        S_T = S * np.exp(drift + diffusion)

        if asset.option_type == OptionType.CALL:
            payoffs = np.maximum(S_T - K, 0.0)
        else:
            payoffs = np.maximum(K - S_T, 0.0)

        # Discount using the same rate r
        discount_factor = np.exp(-r * T)
        discounted_payoffs = payoffs * discount_factor

        price_estimate = np.mean(discounted_payoffs)
        std_dev = np.std(discounted_payoffs)
        std_error = std_dev / np.sqrt(N)

        return {
            "value": price_estimate,
            "std_error": std_error,
            "simulations": N
        }
    
# --- HELPER FUNCTION FOR STEPS ---
def get_simulation_steps(expiry: float, user_steps: Optional[int] = None) -> int:
    """
    Determines the number of time steps (M).
    - If user_steps is provided, use it.
    - Otherwise, default to Daily Steps (T * 252).
    - Enforce a minimum of 50 steps for stability on short maturities.
    """
    if user_steps is not None:
        return int(user_steps)
    
    # Dynamic: 252 trading days per year
    daily_steps = int(np.round(expiry * 252))
    return max(daily_steps, 50)
    
# --- ASIAN OPTIONS ---

class AsianOptionMCPricer(PricingMethod):
    """
    Monte Carlo for Asian Options (Arithmetic Average).
    Uses Daily Steps by default (T * 252).
    """
    def __init__(self, volatility: float, risk_free_rate: Union[float, YieldCurve], 
                 n_simulations: int = 1e5, time_steps: Optional[int] = None, dividend_yield: float = 0.0):
        """
        Args:
            time_steps: If None, defaults to T * 252 (daily).
        """
        self.volatility = volatility
        self.rate_provider = risk_free_rate
        self.n_sims = int(n_simulations)
        self.steps = time_steps # Can be None
        self.dividend = dividend_yield

    def calculate(self, asset: Asset, spot_price: float) -> Dict[str, float]:
        if not isinstance(asset, AsianOption):
            raise TypeError("AsianOptionMCPricer only supports AsianOption assets.")

        S0 = spot_price
        T = asset.expiry
        sigma = self.volatility
        q = self.dividend
        N = self.n_sims
        
        # --- DYNAMIC STEPS CALCULATION ---
        M = get_simulation_steps(T, self.steps)
        dt = T / M
        # ---------------------------------

        if isinstance(self.rate_provider, YieldCurve):
            r = self.rate_provider.get_rate(T)
        else:
            r = self.rate_provider

        # Path Generation
        Z = np.random.standard_normal((N, M))
        
        # Scaling inputs for the time step dt
        drift_per_step = (r - q - 0.5 * sigma**2) * dt
        vol_per_step = sigma * np.sqrt(dt) # This effectively scales the annual vol
        
        increments = drift_per_step + vol_per_step * Z
        log_paths = np.cumsum(increments, axis=1)
        paths = S0 * np.exp(log_paths)

        # Average Price (Asian specific)
        average_prices = np.mean(paths, axis=1)

        if asset.option_type == OptionType.CALL:
            payoffs = np.maximum(average_prices - asset.strike, 0.0)
        else:
            payoffs = np.maximum(asset.strike - average_prices, 0.0)

        price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(N)

        return {"value": price, "std_error": std_error}


class BarrierOptionMCPricer(PricingMethod):
    """
    Monte Carlo for Barrier Options.
    Uses Daily Steps by default to accurately catch barrier crossing.
    """
    def __init__(self, volatility: float, risk_free_rate: Union[float, YieldCurve], 
                 n_simulations: int = 1e5, time_steps: Optional[int] = None, dividend_yield: float = 0.0):
        self.volatility = volatility
        self.rate_provider = risk_free_rate
        self.n_sims = int(n_simulations)
        self.steps = time_steps # Can be None
        self.dividend = dividend_yield

    def calculate(self, asset: Asset, spot_price: float) -> Dict[str, float]:
        if not isinstance(asset, BarrierOption):
            raise TypeError("BarrierOptionMCPricer only supports BarrierOption assets.")

        S0 = spot_price
        T = asset.expiry
        H = asset.barrier
        sigma = self.volatility
        q = self.dividend
        N = self.n_sims
        
        # --- DYNAMIC STEPS CALCULATION ---
        M = get_simulation_steps(T, self.steps)
        dt = T / M
        # ---------------------------------

        if isinstance(self.rate_provider, YieldCurve):
            r = self.rate_provider.get_rate(T)
        else:
            r = self.rate_provider

        # Path Generation
        Z = np.random.standard_normal((N, M))
        
        drift_per_step = (r - q - 0.5 * sigma**2) * dt
        vol_per_step = sigma * np.sqrt(dt)
        
        paths = S0 * np.exp(np.cumsum(drift_per_step + vol_per_step * Z, axis=1))

        # Barrier Check logic
        path_max = np.max(paths, axis=1)
        path_min = np.min(paths, axis=1)
        
        alive_mask = np.zeros(N, dtype=bool)
        b_type = asset.barrier_type

        if b_type == BarrierType.UP_AND_OUT:
            alive_mask = (path_max < H)
        elif b_type == BarrierType.DOWN_AND_OUT:
            alive_mask = (path_min > H)
        elif b_type == BarrierType.UP_AND_IN:
            alive_mask = (path_max >= H)
        elif b_type == BarrierType.DOWN_AND_IN:
            alive_mask = (path_min <= H)

        # Payoff at T (European style payoff, but conditional on barrier)
        S_T = paths[:, -1]
        
        if asset.option_type == OptionType.CALL:
            raw_payoffs = np.maximum(S_T - asset.strike, 0.0)
        else:
            raw_payoffs = np.maximum(asset.strike - S_T, 0.0)
            
        final_payoffs = raw_payoffs * alive_mask

        price = np.exp(-r * T) * np.mean(final_payoffs)
        std_error = np.exp(-r * T) * np.std(final_payoffs) / np.sqrt(N)

        return {"value": price, "std_error": std_error}