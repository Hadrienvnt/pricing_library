import numpy as np
from typing import Dict,List
from typing import Optional
import matplotlib.pyplot as plt
from src.asset_class import Asset, EuropeanOption, AsianOption, BarrierOption,OptionType, BarrierType, PricingMethod


# --- PRICER TESTS --- 

class VanillaValidator:
    """
    Universal validator for any PricingMethod (Analytical, Monte-Carlo, PDE, etc.).
    Checks financial consistency like Put-Call Parity and Martingale properties.
    """

    def __init__(self, engine: PricingMethod):
        """
        Args:
            engine (PricingMethod): An initialized pricing engine 
                                    (e.g., AnalyticalBS(vol=0.2, r=0.05)).
        """
        self.engine = engine

    def run_tests(self, spot: float, strike: float, expiry: float) -> Dict[str, bool]:
        """
        Runs a suite of financial sanity checks.
        """
        results = {}
        print(f"Running validation on {self.engine.__class__.__name__}...")

        # 1. Put-Call Parity
        pc_parity_passed = self.check_put_call_parity(spot, strike, expiry)
        results['Put-Call Parity'] = pc_parity_passed
        
        # 2. Martingale (Zero Strike Call)
        martingale_passed = self.check_martingale_property(spot, expiry)
        results['Martingale Property'] = martingale_passed

    def check_put_call_parity(self, S: float, K: float, T: float, tolerance: float = 1e-2) -> bool:
        """
        Verifies C - P = S - K * exp(-rT) = Fwd
        Note: We extract 'r' implicitly by pricing a Zero-Coupon Bond equivalent.
        """
        # Create temporary assets
        call = EuropeanOption(K, T, OptionType.CALL)
        put = EuropeanOption(K, T, OptionType.PUT)
        
        # Link our engine to them
        call.set_pricing_method(self.engine)
        put.set_pricing_method(self.engine)

        # Calculate prices
        c_price = call.price(S)['value']
        p_price = put.price(S)['value']
        
        # Calculate expected discount factor implied by the model
        # To make this truly generic, we need the discount factor used by the engine.
        # We can infer it by pricing a Zero Strike Call (value = S * exp(-qT)) 
        # or we assume we can access the rate if it's a property, 
        # BUT strictly speaking, let's use the engine's internal rate/curve if accessible:
        
        # Dynamic rate retrieval handling
        if hasattr(self.engine, 'rate_provider'):
            # Accessing the rate from the engine properties (BS or MC)
            rate_obj = self.engine.rate_provider
            if hasattr(rate_obj, 'get_rate'): # It's a YieldCurve
                r = rate_obj.get_rate(T)
            else: # It's a float
                r = rate_obj
        else:
            print("Warning: Could not extract rate for parity check. Assuming r=0 for test (risky).")
            r = 0.0

        # Assuming Dividend q=0 for this basic check, or we extract it similarly.
        # Let's assume the user of this validator knows the engine setup.
        dividend = getattr(self.engine, 'dividend', 0.0)

        # Theoretical Parity: Call - Put = S * e^-qT - K * e^-rT
        df_r = np.exp(-r * T)
        df_q = np.exp(-dividend * T)
        
        lhs = c_price - p_price
        rhs = S * df_q - K * df_r
        
        error = abs(lhs - rhs)
        passed = error < tolerance
        
        print(f"\n[Put-Call Parity]: Checks if C - P = S - K * exp(-rT)")
        print(f'   P-C = {lhs:.4f} vs S-K*exp(-rT) = {rhs:.4f}')
        print(f'   Err: {error:.6f} -> {'PASS' if passed else 'FAIL'}')
        return passed

    def check_martingale_property(self, S: float, T: float, tolerance: float = 0.1) -> bool:
        """
        Checks if Price(Call | K=0) == S * exp(-qT).
        This verifies that the underlying asset is correctly modeled (drift consistency).
        """
        zero_strike_call = EuropeanOption(0.0, T, OptionType.CALL)
        zero_strike_call.set_pricing_method(self.engine)
        
        price = zero_strike_call.price(S)['value']
        
        # Expected: S * exp(-qT)
        dividend = getattr(self.engine, 'dividend', 0.0)
        expected = S * np.exp(-dividend * T)
        
        error = abs(price - expected)
        passed = error < tolerance
        
        print(f"\n[Martingale Test]: Checks if Price(Call | K=0) == S * exp(-qT)")
        print(f'   Price(Call | K=0) = {price:.4f} vs. S*exp(-qT) = {expected:.4f}')
        print(f'   Err: {error:.6f} -> {'PASS' if passed else 'FAIL'}')
        return passed

class BarrierValidator:
    """
    Validator generic for ANY Barrier Pricing Method.
    Requires a reference engine (for European) to validate parity relationships.
    """

    def __init__(self, barrier_engine: PricingMethod, ref_european_engine: PricingMethod):
        """
        Args:
            barrier_engine: The engine configured to price Barrier Options (Candidate).
            ref_european_engine: The engine configured to price European Options (Reference).
                                 MUST have same Vol/Rate parameters for valid comparison.
        """
        self.engine = barrier_engine
        self.ref_engine = ref_european_engine
    
    def run_tests(self, spot: float, strike: float, expiry: float,barrier:float,tolerance:float=0.01) -> Dict[str, bool]:
        """
        Runs a suite of financial sanity checks.
        """
        self.check_in_out_parity(S=spot,K=strike,T=expiry,H=barrier,tolerance=tolerance)
        self.plot_ineffective_barrier_convergence(S=spot,K=strike,T=expiry)

    def check_in_out_parity(self, S: float, K: float, T: float, H: float, tolerance: float = 0.01) -> bool:
        """
        Verifies: Call(Up-In) + Call(Up-Out) == Call(European)
        """
        print(f"\n[Barrier Parity]: Testing In + Out = European (H={H})")
        
        # 1. Create Assets
        opt_in = BarrierOption(K, T, OptionType.CALL, H, BarrierType.UP_AND_IN)
        opt_out = BarrierOption(K, T, OptionType.CALL, H, BarrierType.UP_AND_OUT)
        opt_eur = EuropeanOption(K, T, OptionType.CALL)

        # 2. Link to injected engines
        opt_in.set_pricing_method(self.engine)
        opt_out.set_pricing_method(self.engine)
        opt_eur.set_pricing_method(self.ref_engine)

        # 3. Price
        try:
            p_in = opt_in.price(S)['value']
            p_out = opt_out.price(S)['value']
            p_eur = opt_eur.price(S)['value']
        except Exception as e:
            print(f"   Error during pricing: {e}")
            return False

        total = p_in + p_out
        error = abs(total - p_eur)
        
        # Relative tolerance helps if prices are very large or small
        passed = error < (p_eur * tolerance)

        print(f"   In ({p_in:.2f}) + Out ({p_out:.2f}) = {total:.2f}")
        print(f"   European Reference = {p_eur:.2f}")
        print(f"   Diff: {error:.4f} (Tol: {p_eur * tolerance:.4f}) -> {'PASS' if passed else 'FAIL'}")
        
        return passed

    def plot_ineffective_barrier_convergence(self, S: float, K: float, T: float, 
                                             barrier_range_mult: tuple = (1.1, 5.0), points: int = 20):
        """
        Plots Up-and-Out Price vs Barrier Level.
        As H -> Infinity, Price -> European Price.
        """
        print(f"\n[Barrier Plot]: Generating Convergence Plot...")
        
        # 1. Get Reference Price (Constant line)
        euro_opt = EuropeanOption(K, T, OptionType.CALL)
        euro_opt.set_pricing_method(self.ref_engine)
        euro_price = euro_opt.price(S)['value']

        # 2. Sweep Barrier Levels
        barriers = np.linspace(S * barrier_range_mult[0], S * barrier_range_mult[1], points)
        barrier_prices = []
        
        print(f"   Computing {points} points...")
        for H in barriers:
            b_opt = BarrierOption(K, T, OptionType.CALL, H, BarrierType.UP_AND_OUT)
            b_opt.set_pricing_method(self.engine)
            price = b_opt.price(S)['value']
            barrier_prices.append(price)

        # Plot
        plt.figure(figsize=(10, 6))

        plt.plot(barriers, barrier_prices, 'o-', label='Up-and-Out Call Price', color='blue', linewidth=2)
        
        # Reference Line
        plt.axhline(y=euro_price, color='red', linestyle='--', label=f'European Reference ({euro_price:.2f})')
        
        plt.title(f'Barrier Option Convergence Test\nStrike={K}, T={T}')
        plt.xlabel('Barrier Level (H)')
        plt.ylabel('Option Price')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

class AsianValidator:
    """
    Validator generic for ANY Asian Pricing Method.
    Requires a reference engine (European) to validate bounds.
    """

    def __init__(self, asian_engine: PricingMethod, ref_european_engine: PricingMethod):
        """
        Args:
            asian_engine: The engine configured to price Asian Options.
            ref_european_engine: The engine configured to price European Options.
        """
        self.engine = asian_engine
        self.ref_engine = ref_european_engine

    def check_bounds_vs_european(self, S: float, K: float, T: float) -> bool:
        """
        Verifies that Asian Call Price <= European Call Price.
        """
        print(f"\n[Asian Bounds]: Testing Asian <= European")
        
        # 1. Assets
        euro_opt = EuropeanOption(K, T, OptionType.CALL)
        asian_opt = AsianOption(K, T, OptionType.CALL)

        # 2. Link Engines
        euro_opt.set_pricing_method(self.ref_engine)
        asian_opt.set_pricing_method(self.engine)

        # 3. Price
        p_eur = euro_opt.price(S)['value']
        p_asian = asian_opt.price(S)['value']

        diff = p_eur - p_asian
        
        # We allow a tiny negative diff due to MC noise (e.g. -0.05), but ideally diff > 0
        passed = diff > -0.1 

        print(f"   European Ref: {p_eur:.4f}")
        print(f"   Asian Price:  {p_asian:.4f}")
        print(f"   Diff (Eur-As): {diff:.4f} -> {'PASS' if passed else 'FAIL'}")

    def check_deterministic_zero_vol(self, S: float, K: float, T: float, r: float, 
                                     zero_vol_engine: Optional[PricingMethod] = None):
        """
        Special Test: Requires an engine configured with Volatility ~ 0.
        Since we cannot force the injected 'self.engine' to change its vol,
        you must pass a specific 'zero_vol_engine' here.
        """
        print(f"\n[Asian Zero-Vol]: Testing Deterministic Path")
        
        if zero_vol_engine is None:
            print("   [SKIP] No zero_vol_engine provided. Skipping test.")
            return False

        # Theoretical Calculation
        if abs(r) < 1e-9:
            theoretical_avg = S
        else:
            theoretical_avg = (S / (r * T)) * (np.exp(r * T) - 1)
            
        theoretical_payoff = max(theoretical_avg - K, 0.0) * np.exp(-r * T)

        # Engine Calculation
        asian = AsianOption(K, T, OptionType.CALL)
        asian.set_pricing_method(zero_vol_engine)
        
        mc_price = asian.price(S)['value']
        
        error = abs(mc_price - theoretical_payoff)
        
        print(f"   Theoretical: {theoretical_payoff:.4f}")
        print(f"   Engine Price: {mc_price:.4f}")
        print(f"   Error: {error:.6f} -> {'PASS' if error < 0.01 else 'FAIL'}")
        

# --- CONVERGENCE ANALYSIS ---

class ConvergenceAnalyzer:
    """
    Helper class to analyze the convergence rate of a Monte Carlo engine.
    Does not price assets, but tests the engine's stability and plots convergence.
    """
    
    def __init__(self, mc_engine_class, volatility: float, risk_free_rate: float):
        """
        Args:
            mc_engine_class: The class type (e.g. EuropeanOptionMCPricer), NOT an instance.
            volatility, risk_free_rate: Parameters to instantiate the engine.
        """
        self.EngineType = mc_engine_class
        self.vol = volatility
        self.rate = risk_free_rate

    def run_convergence_test(self, asset: Asset, spot: float, 
                             simulation_steps: List[int] = [1000, 4000, 10000, 40000, 100000, 250000],
                             show_plot: bool = True):
        """
        Runs the pricing with increasing number of simulations.
        Prints the standard error reduction and plots the results.
        """
        print(f"\nRunning Convergence Test on {self.EngineType.__name__}...")
        print(f"{'Simulations':<15} | {'Price':<10} | {'Std Error':<10} | {'Ratio':<10}")
        print("-" * 55)

        prev_std_error = None
        
        # Lists to store data for plotting
        n_values = []
        error_values = []

        for n_sims in simulation_steps:
            # Instantiate a fresh engine with N simulations
            engine = self.EngineType(
                volatility=self.vol, 
                risk_free_rate=self.rate, 
                n_simulations=n_sims
            )
            
            # Helper to link manually for this test
            asset.set_pricing_method(engine)
            result = asset.price(spot)
            
            price = result['value']
            std_err = result['std_error']
            
            # Store data
            n_values.append(n_sims)
            error_values.append(std_err)
            
            # Calculate convergence ratio (should be approx sqrt(step_ratio))
            ratio_str = "N/A"
            if prev_std_error and std_err > 0:
                ratio = prev_std_error / std_err
                ratio_str = f"{ratio:.2f}"
            
            print(f"{n_sims:<15} | {price:.4f}     | {std_err:.6f}   | {ratio_str}")
            
            prev_std_error = std_err

        if show_plot:
            self._plot_results(n_values, error_values)

    def _plot_results(self, n_values: List[int], error_values: List[float]):
        """
        Generates a Log-Log plot of the standard error vs Number of simulations.
        """
        plt.figure(figsize=(10, 6))
        
        # 1. Plot Actual Monte Carlo Standard Error
        plt.loglog(n_values, error_values, 'o-', label='MC Standard Error', linewidth=2, markersize=8)
        
        # 2. Plot Theoretical 1/sqrt(N) Line
        # We scale the theoretical line to match the first point of the actual data
        # so they start at the same height, making it easier to compare slopes.
        # Theoretical model: Error ~ k / sqrt(N)
        # k = Error[0] * sqrt(N[0])
        if len(n_values) > 0 and len(error_values) > 0:
            scale_factor = error_values[0] * np.sqrt(n_values[0])
            theoretical_curve = [scale_factor / np.sqrt(n) for n in n_values]
            
            plt.loglog(n_values, theoretical_curve, '--', color='red', 
                       label=r'Theoretical $1/\sqrt{N}$', alpha=0.7)

        # Formatting
        plt.title('Monte Carlo Convergence Analysis')
        plt.xlabel('Number of Simulations (N)')
        plt.ylabel('Standard Error')
        plt.grid(True, which="both", ls="-", alpha=0.4)
        plt.legend()
        
        plt.show()