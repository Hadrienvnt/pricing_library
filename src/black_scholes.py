import numpy as np
from scipy.stats import norm
from typing import Dict, Union

from src.asset_class import PricingMethod, Asset, EuropeanOption, OptionType
from src.market_data import YieldCurve

class BlackScholesPricer(PricingMethod):
    """
    Analytic Black-Scholes pricing method.
    Robust handling of edge cases (K=0, S=0, Vol=0).
    """

    def __init__(self, volatility: float, risk_free_rate: Union[float, YieldCurve], dividend_yield: float = 0.0):
        self.volatility = volatility
        self.rate_provider = risk_free_rate
        self.dividend = dividend_yield

    def calculate(self, asset: Asset, spot_price: float) -> Dict[str, float]:
        if not isinstance(asset, EuropeanOption):
            raise TypeError("AnalyticalBS engine only supports EuropeanOption assets.")
        
        # 1. Extract Contract Data
        K = asset.strike
        T = asset.expiry
        is_call = (asset.option_type == OptionType.CALL)
        
        # 2. Extract Market Data
        S = spot_price
        sigma = self.volatility
        q = self.dividend
        
        # Dynamic rate retrieval
        if isinstance(self.rate_provider, YieldCurve):
            r = self.rate_provider.get_rate(T)
        else:
            r = self.rate_provider

        # Pre-calculate discount factors
        df_r = np.exp(-r * T)
        df_q = np.exp(-q * T)

        # --- 3. HANDLE EDGE CASES ---
        
        # Case A: Expiry passed or immediate
        if T <= 0:
            return {
                "value": asset.payoff(S), 
                "delta": 0.0, "gamma": 0.0, "vega": 0.0, "rho": 0.0
            }

        # Case B: Zero Strike (K -> 0)
        # For example: Used in Martingale Tests. A Call with K=0 is like holding the stock.
        if K < 1e-9:
            if is_call:
                # Call Price = S * exp(-qT)
                return {
                    "value": S * df_q,
                    "delta": df_q,
                    "gamma": 0.0, "vega": 0.0, "rho": 0.0
                }
            else:
                # Put Price = 0 (can't sell below 0)
                return {
                    "value": 0.0, "delta": 0.0, "gamma": 0.0, "vega": 0.0, "rho": 0.0
                }

        # Case C: Zero Spot (S -> 0)
        if S < 1e-9:
            if is_call:
                return {"value": 0.0, "delta": 0.0, "gamma": 0.0, "vega": 0.0, "rho": 0.0}
            else:
                # Put Price = PV(K)
                return {
                    "value": K * df_r, 
                    "delta": 0.0, # Technically undefined or 0 depending on convention
                    "gamma": 0.0, "vega": 0.0, 
                    "rho": -K * T * df_r
                }

        # Case D: Zero Volatility (Sigma -> 0)
        # Avoid division by zero in d1/d2. Return Intrinsic Value.
        if sigma < 1e-9:
            forward_price = S * np.exp((r - q) * T)
            intrinsic = 0.0
            if is_call:
                intrinsic = df_r * max(forward_price - K, 0.0)
            else:
                intrinsic = df_r * max(K - forward_price, 0.0)
            return {
                "value": intrinsic,
                "delta": 0.0, "gamma": 0.0, "vega": 0.0, "rho": 0.0 # Greeks are degenerate at 0 vol
            }

        # 4. Standard Black-Scholes Formulas
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        N_minus_d1 = norm.cdf(-d1)
        N_minus_d2 = norm.cdf(-d2)
        n_d1 = norm.pdf(d1)

        price = 0.0
        delta = 0.0
        rho = 0.0

        if is_call:
            price = S * df_q * N_d1 - K * df_r * N_d2
            delta = df_q * N_d1
            rho = K * T * df_r * N_d2
        else:
            price = K * df_r * N_minus_d2 - S * df_q * N_minus_d1
            delta = -df_q * N_minus_d1
            rho = -K * T * df_r * N_minus_d2

        gamma = (df_q * n_d1) / (S * sigma * sqrt_T)
        vega = S * df_q * n_d1 * sqrt_T

        return {
            "value": price,
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "rho": rho
        }
    
    def delta(self,asset: Asset, spot_price: float):
        res = self.calculate(asset=asset,spot_price=spot_price)
        return res.delta
    
    def gamma(self,asset: Asset, spot_price: float):
        res = self.calculate(asset=asset,spot_price=spot_price)
        return res.gamma
    
    def vega(self,asset: Asset, spot_price: float):
        res = self.calculate(asset=asset,spot_price=spot_price)
        return res.vega
    
    def rho(self,asset: Asset, spot_price: float):
        res = self.calculate(asset=asset,spot_price=spot_price)
        return res.rho

