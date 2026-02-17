from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict
import numpy as np
import matplotlib.pyplot as plt


# --- ABSTRACT BASE CLASSES ---

class Asset(ABC):
    """
    Abstract Base Class for all financial instruments.
    """
    
    def __init__(self):
        self._pricing_method: Optional['PricingMethod'] = None

    def set_pricing_method(self, method: 'PricingMethod'):
        """
        Assigns a pricing engine (method) to this specific asset.
        """
        self._pricing_method = method

    def price(self, spot_price: float) -> Dict[str, float]:
        """
        Delegates the calculation to the assigned PricingMethod.
        """
        if self._pricing_method is None:
            raise ValueError("No PricingMethod set for this Asset. Use set_pricing_method() first.")
        
        return self._pricing_method.calculate(self, spot_price)
    
    @abstractmethod
    def payoff(self, spot_price: float) -> float:
        pass


class PricingMethod(ABC):
    """
    Abstract Base Class for all pricing algorithms.
    """
    @abstractmethod
    def calculate(self, asset: Asset, spot_price: float) -> Dict[str, float]:
        pass


# --- EUROPEAN OPTION ---

class OptionType(Enum):
    """
    Defines the type of the option contract (Call or Put).
    """
    CALL = "call"
    PUT = "put"


@dataclass
class EuropeanOptionParams:
    strike: float
    expiry: float
    option_type: OptionType


class EuropeanOption(Asset):
    """
    Concrete implementation of a European Option with visualization capabilities.
    """
    
    def __init__(self, strike: float, expiry: float, option_type: OptionType):
        super().__init__()
        self._params = EuropeanOptionParams(strike, expiry, option_type)

    @property
    def strike(self) -> float:
        return self._params.strike

    @property
    def expiry(self) -> float:
        return self._params.expiry
    
    @property
    def option_type(self) -> OptionType:
        return self._params.option_type

    def payoff(self, spot: float) -> float:
        """
        Calculates max(S - K, 0) or max(K - S, 0).
        """
        if self._params.option_type == OptionType.CALL:
            return max(spot - self._params.strike, 0.0)
        else:
            return max(self._params.strike - spot, 0.0)

    def plot_analysis(self, spot_min: float = None, spot_max: float = None, points: int = 100):
        """
        Plots the Payoff vs The Model Price over a range of spot prices.
        Automatically adds annotations for Intrinsic Value.
        """
        if self._pricing_method is None:
            raise ValueError("Cannot plot analysis: No PricingMethod set.")

        K = self.strike
        
        # Default range: +/- 50% around strike if not provided
        if spot_min is None: spot_min = K * 0.5
        if spot_max is None: spot_max = K * 1.5

        spots = np.linspace(spot_min, spot_max, points)
        payoffs = [self.payoff(s) for s in spots]
        prices = []

        # Calculate prices using the attached engine
        for s in spots:
            res = self.price(s)
            prices.append(res['value'])

        # --- PLOTTING ---
        plt.figure(figsize=(10, 6))
        
        # Strike Line
        plt.axvline(x=K, color='gray', linestyle=':', alpha=0.5, label='Strike (K)')
        
        # Payoff Curve
        plt.plot(spots, payoffs, label='Payoff (Intrinsic)', color='black', linestyle='--', alpha=0.7)
        
        # Model Price Curve
        # Try to get volatility for the legend if the engine has it
        vol_info = ""
        if hasattr(self._pricing_method, 'volatility'):
            vol_info = r" ($\sigma$={self._pricing_method.volatility:.0%})"
            
        plt.plot(spots, prices, label=rf'Model Price{vol_info}', color='forestgreen', linewidth=2.5)

        # Styling
        plt.title(f'{self.option_type.value.capitalize()} Option Profile: K={K} @ T={self.expiry}Y')
        plt.xlabel('Spot Price ($S_0$)')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.show()


# --- ASIAN OPTION ---

@dataclass
class AsianOptionParams:
    strike: float
    expiry: float
    option_type: OptionType

class AsianOption(Asset):
    """
    Asian Option (Fixed Strike).
    Payoff depends on the AVERAGE price of the underlying over time.
    Payoff = Max(Average - K, 0) for Call.
    """
    def __init__(self, strike: float, expiry: float, option_type: OptionType):
        super().__init__()
        self._params = AsianOptionParams(strike, expiry, option_type)

    @property
    def strike(self) -> float:
        return self._params.strike
    
    @property
    def expiry(self) -> float:
        return self._params.expiry
    
    @property
    def option_type(self) -> OptionType:
        return self._params.option_type

    def payoff(self, average_price: float) -> float:
        """
        WARNING: For Asian Options, the input is the AVERAGE price, not the final spot.
        """
        if self._params.option_type == OptionType.CALL:
            return max(average_price - self._params.strike, 0.0)
        else:
            return max(self._params.strike - average_price, 0.0)
    

    def plot_analysis(self, spot_min: float = None, spot_max: float = None, points: int = 50):
        """
        Plots the Model Price over a range of spot prices.
        """
        if self._pricing_method is None:
            raise ValueError("Cannot plot analysis: No PricingMethod set.")

        K = self.strike
        
        # Default range: +/- 50% around strike if not provided
        if spot_min is None: spot_min = K * 0.5
        if spot_max is None: spot_max = K * 1.5

        spots = np.linspace(spot_min, spot_max, points)
        payoffs = [self.payoff(s) for s in spots]
        prices = []

        # Calculate prices using the attached engine
        for s in spots:
            res = self.price(s)
            prices.append(res['value'])

        # --- PLOTTING ---
        plt.figure(figsize=(10, 6))
        
        # Strike Line
        plt.axvline(x=K, color='gray', linestyle=':', alpha=0.5, label='Strike (K)')

        # European Call Payoff as a ref
        plt.plot(spots, payoffs, label='European Call Payoff', color='black', linestyle='--', alpha=0.7)
           
        # Model Price Curve
        # Try to get volatility for the legend if the engine has it
        vol_info = ""
        if hasattr(self._pricing_method, 'volatility'):
            vol_info = r" ($\sigma$" + f"={self._pricing_method.volatility:.0%})"
            
        plt.plot(spots, prices, label=rf'Model Price{vol_info}', color='forestgreen', linewidth=2.5)

        # Styling
        plt.title(f'Asian-Style {self.option_type.value.capitalize()} Option Profile: K={K} @ T={self.expiry}Y')
        plt.xlabel('Spot Price ($S_0$)')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()
        
    


# --- BARRIER OPTION ---

class BarrierType(Enum):
    """
    Defines the behavior of the barrier.
    """
    UP_AND_OUT = "Up & Out"     # Dies if S > H
    DOWN_AND_OUT = "Down & Out" # Dies if S < H
    UP_AND_IN = "Up & In"       # Born if S > H
    DOWN_AND_IN = "Down & In"   # Born if S < H


@dataclass
class BarrierOptionParams:
    strike: float
    expiry: float
    option_type: OptionType
    barrier: float
    barrier_type: BarrierType

class BarrierOption(Asset):
    """
    Barrier Option.
    Becomes active (In) or inactive (Out) if the price crosses the 'barrier' level.
    """
    def __init__(self, strike: float, expiry: float, option_type: OptionType, 
                 barrier: float, barrier_type: BarrierType):
        super().__init__()
        self._params = BarrierOptionParams(strike, expiry, option_type, barrier, barrier_type)

    @property
    def strike(self) -> float:
        return self._params.strike
    
    @property
    def expiry(self) -> float:
        return self._params.expiry
    
    @property
    def barrier(self) -> float:
        return self._params.barrier

    @property
    def barrier_type(self) -> BarrierType:
        return self._params.barrier_type
    
    @property
    def option_type(self) -> OptionType:
        return self._params.option_type

    def payoff(self, spot: float) -> float:
        """
        Standard European payoff.
        The Barrier logic (Alive/Dead) is handled by the PRICING ENGINE (Monte Carlo),
        not here. If the engine determines the option is dead, it won't call this method,
        or will multiply the result by 0.
        """
        if self._params.option_type == OptionType.CALL:
            return max(spot - self._params.strike, 0.0)
        else:
            return max(self._params.strike - spot, 0.0)
    
    def plot_analysis(self, spot_min: float = None, spot_max: float = None, points: int = 50):
        """
        Plots the Model Price over a range of spot prices.
        """
        if self._pricing_method is None:
            raise ValueError("Cannot plot analysis: No PricingMethod set.")

        K = self.strike
        H = self.barrier
        
        # Default range: +/- 50% around strike if not provided
        if spot_min is None: spot_min = K * 0.5
        if spot_max is None: spot_max = K * 1.5

        spots = np.linspace(spot_min, spot_max, points)
        payoffs = [self.payoff(s) for s in spots]
        prices = []

        # Calculate prices using the attached engine
        for s in spots:
            res = self.price(s)
            prices.append(res['value'])

        # --- PLOTTING ---
        plt.figure(figsize=(10, 6))
        
        # Strike Line
        plt.axvline(x=K, color='gray', linestyle=':', alpha=0.7, label='Strike (K)')

        # Barrier Line
        plt.axvline(x=H, color='gray', linestyle='--', alpha=0.5, label='Barrier (H)')

        # European Call Payoff as a ref
        plt.plot(spots, payoffs, label='European Call Payoff', color='black', linestyle='--', alpha=0.7)
           
        # Model Price Curve
        # Try to get volatility for the legend if the engine has it
        vol_info = ""
        if hasattr(self._pricing_method, 'volatility'):
            vol_info = r" ($\sigma$" + f"={self._pricing_method.volatility:.0%})"
            
        plt.plot(spots, prices, label=rf'Model Price{vol_info}', color='forestgreen', linewidth=2.5)

        # Styling
        plt.title(f'{self.barrier_type.value} {self.option_type.value.capitalize()} Barrier Option Profile: K={K}, H={H} @ T={self.expiry}Y')
        plt.xlabel('Spot Price ($S_0$)')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()