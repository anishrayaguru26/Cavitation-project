#!/usr/bin/env python3
"""
Solving the Rayleigh-Plesset equation for vapor bubble dynamics in liquid sodium
Based on Prosperetti and Plesset's paper on vapor bubble growth
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from tqdm import tqdm
import numba  # Added for JIT compilation

# Physical properties of liquid sodium
# Values are based on typical properties for liquid sodium
# Boiling point of sodium at 0.5 bar is approximately 1056K
# With 340K superheat, initial temperature is 1396K

# Constants
rho = 800  # Liquid density (kg/m^3)
sigma = 0.2  # Surface tension (N/m)
k = 60  # Thermal conductivity (W/m·K)
D = 6.5e-7  # Thermal diffusivity (m^2/s)
L = 4.3e6  # Latent heat of vaporization (J/kg)
p_inf = 0.5e5  # Ambient pressure (0.5 bar in Pa)
T_inf = 1396  # Initial liquid temperature (K)
T_b = 1056  # Boiling temperature at 0.5 bar (K)
R_initial = 1e-6  # Initial bubble radius (m)
dRdt_initial = 0  # Initial bubble growth rate (m/s)

# Antoine equation parameters for sodium vapor pressure (in bar)
# log10(P) = A - B/(T + C) where P is in bar and T is in K
A = 4.52
B = 5100
C = 0

@numba.jit(nopython=True)
def vapor_pressure(T):
    """Calculate vapor pressure of sodium using Antoine equation.
    T: temperature in K
    Returns: pressure in Pa
    """
    return 10**(A - B/(T + C)) * 1e5  # Convert bar to Pa

@numba.jit(nopython=True)
def dSdt(t, S):
    """Define the system of ODEs.
    
    S[0]: R - Bubble radius
    S[1]: dR/dt - Bubble growth rate
    S[2]: T_s - Bubble surface temperature
    
    Returns derivatives [dR/dt, d²R/dt², dT_s/dt]
    """
    R = max(S[0], 1e-10)  # Prevent division by zero
    dRdt = S[1]
    T_s = S[2]
    
    # Vapor pressure based on surface temperature
    p_v = vapor_pressure(T_s)
    
    # Rayleigh-Plesset equation
    d2Rdt2 = (1/rho) * (p_v - p_inf - 2*sigma/R) / R - 1.5 * (dRdt**2) / R
    
    # Surface temperature evolution
    # Simplified model for temperature evolution
    dTsdt = -L * rho * dRdt / k * (T_s - T_b) / R
    
    return np.array([dRdt, d2Rdt2, dTsdt])  # Return as numpy array for numba compatibility

class ProgressIVP:
    """Wrapper for solve_ivp that shows a progress bar during integration, including time."""
    
    def __init__(self, fun, t_span, y0, t_eval=None, **kwargs):
        self.fun = fun
        self.t_span = t_span
        self.y0 = y0
        self.t_eval = t_eval
        self.kwargs = kwargs
        self.n_steps = 0
        self.t_steps = []
        self.update_interval = 50  # Update progress every 10 steps
        
    def wrapper(self, t, y):
        """Wrapper function that counts calls to the ODE function."""
        self.n_steps += 1
        
        # Update progress bar less frequently to reduce overhead
        if self.n_steps % self.update_interval == 0:
            progress_bar.set_description(f"t = {t:.2e} s | steps = {self.n_steps}")
            progress_bar.update(self.update_interval)
        
        return self.fun(t, y)
    
    def solve(self, max_steps=10000):
        """Solve the IVP with progress bar."""
        self.n_steps = 0
        
        print(f"Starting simulation with estimated {max_steps} steps...")
        global progress_bar
        progress_bar = tqdm(total=max_steps, desc="Solving ODEs", dynamic_ncols=True)
        
        result = solve_ivp(
            self.wrapper, 
            self.t_span, 
            self.y0,
            t_eval=self.t_eval,
            **self.kwargs
        )
        
        # Update any remaining progress
        remaining = self.n_steps % self.update_interval
        if remaining > 0:
            progress_bar.update(remaining)
            
        progress_bar.close()
        print(f"Completed in {self.n_steps} steps.")
        
        return result


def solve_stage(t_start, t_end, S0, rtol, atol, t_points):
    """Solve a single stage with custom tolerances and points."""
    t_span = [t_start, t_end]
    t_eval = np.logspace(np.log10(t_start), np.log10(t_end), t_points)
    
    solver = ProgressIVP(
        dSdt,
        t_span,
        S0,
        t_eval=t_eval,
        method='LSODA',  # Changed from RK45 to LSODA which is often more efficient for stiff problems
        rtol=rtol,
        atol=atol,
        max_step=np.inf  # Allow solver to take larger steps when possible
    )
    
    solution = solver.solve(max_steps=3000)
    return solution

def solve_bubble_dynamics_two_stage():
    """Solve the bubble dynamics equations using two-stage fast approach."""
    print("Starting two-stage simulation...")
    
    # Stage 1: Early time (critical phase)
    print("Stage 1: Early time dynamics...")
    S0 = [R_initial, dRdt_initial, T_inf]
    sol1 = solve_stage(
        t_start=1e-9,
        t_end=1e-5,
        S0=S0,
        rtol=1e-6,
        atol=1e-9,
        t_points=300
    )
    
    # Stage 2: Later time (bubble already formed)
    print("Stage 2: Later time dynamics...")
    S0_stage2 = [sol1.y[0][-1], sol1.y[1][-1], sol1.y[2][-1]]  # Last values from Stage 1
    sol2 = solve_stage(
        t_start=1e-5,
        t_end=1e0,
        S0=S0_stage2,
        rtol=1e-5,
        atol=1e-8,
        t_points=300
    )
    
    # Combine results
    t_combined = np.concatenate((sol1.t, sol2.t))
    y_combined = np.hstack((sol1.y, sol2.y))
    
    class CombinedSolution:
        t = t_combined
        y = y_combined
        success = sol1.success and sol2.success
        
    return CombinedSolution()

def solve_bubble_dynamics_gpu():
    """GPU-accelerated version of the bubble dynamics solver (if CUDA is available)"""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        print(f"GPU acceleration: {'Available' if has_cuda else 'Not available'}")
        device = torch.device("cuda" if has_cuda else "cpu")
        
        if not has_cuda:
            print("Falling back to CPU mode")
            return solve_bubble_dynamics_two_stage()
            
        print("Using GPU acceleration with PyTorch...")
        
        # Convert main constants to torch tensors
        t_rho = torch.tensor(rho, device=device, dtype=torch.float64)
        t_sigma = torch.tensor(sigma, device=device, dtype=torch.float64)
        t_k = torch.tensor(k, device=device, dtype=torch.float64)
        t_L = torch.tensor(L, device=device, dtype=torch.float64)
        t_p_inf = torch.tensor(p_inf, device=device, dtype=torch.float64)
        t_T_b = torch.tensor(T_b, device=device, dtype=torch.float64)
        t_A = torch.tensor(A, device=device, dtype=torch.float64)
        t_B = torch.tensor(B, device=device, dtype=torch.float64)
        t_C = torch.tensor(C, device=device, dtype=torch.float64)
        
        def torch_vapor_pressure(T):
            """GPU version of vapor pressure calculation"""
            return torch.pow(10, t_A - t_B/(T + t_C)) * 1e5
            
        # Create time arrays for both stages
        t1 = torch.logspace(np.log10(1e-9), np.log10(1e-5), 500, device=device, dtype=torch.float64)
        t2 = torch.logspace(np.log10(1e-5), np.log10(1e0), 500, device=device, dtype=torch.float64)
        
        # Initial conditions
        R = torch.tensor(R_initial, device=device, dtype=torch.float64)
        dRdt = torch.tensor(dRdt_initial, device=device, dtype=torch.float64)
        T_s = torch.tensor(T_inf, device=device, dtype=torch.float64)
        
        # Arrays to store results
        R_values1 = torch.zeros_like(t1)
        dRdt_values1 = torch.zeros_like(t1)
        Ts_values1 = torch.zeros_like(t1)
        
        # Step size for explicit integration
        dt_factor = 0.05  # Safety factor
        
        print("Stage 1: Early time dynamics (GPU)...")
        with tqdm(total=len(t1)-1, desc="GPU solving stage 1") as pbar:
            # First stage explicit integration
            for i in range(len(t1)-1):
                dt = (t1[i+1] - t1[i]) * dt_factor
                sub_steps = int(1/dt_factor)
                
                # Sub-stepping for stability
                for _ in range(sub_steps):
                    # Calculate derivatives
                    p_v = torch_vapor_pressure(T_s)
                    d2Rdt2 = (1/t_rho) * (p_v - t_p_inf - 2*t_sigma/R) / R - 1.5 * (dRdt**2) / R
                    dTsdt = -t_L * t_rho * dRdt / t_k * (T_s - t_T_b) / R
                    
                    # Update variables
                    dRdt = dRdt + d2Rdt2 * dt
                    R = torch.clamp(R + dRdt * dt, min=1e-10)  # Prevent negative radius
                    T_s = T_s + dTsdt * dt
                
                # Store results
                R_values1[i] = R
                dRdt_values1[i] = dRdt
                Ts_values1[i] = T_s
                
                pbar.update(1)
                
        # Store final point
        R_values1[-1] = R
        dRdt_values1[-1] = dRdt
        Ts_values1[-1] = T_s
        
        # Arrays for stage 2
        R_values2 = torch.zeros_like(t2)
        dRdt_values2 = torch.zeros_like(t2)
        Ts_values2 = torch.zeros_like(t2)
        
        print("Stage 2: Later time dynamics (GPU)...")
        with tqdm(total=len(t2)-1, desc="GPU solving stage 2") as pbar:
            # Second stage integration
            for i in range(len(t2)-1):
                dt = (t2[i+1] - t2[i]) * dt_factor
                sub_steps = int(1/dt_factor)
                
                # Sub-stepping for stability
                for _ in range(sub_steps):
                    # Calculate derivatives
                    p_v = torch_vapor_pressure(T_s)
                    d2Rdt2 = (1/t_rho) * (p_v - t_p_inf - 2*t_sigma/R) / R - 1.5 * (dRdt**2) / R
                    dTsdt = -t_L * t_rho * dRdt / t_k * (T_s - t_T_b) / R
                    
                    # Update variables
                    dRdt = dRdt + d2Rdt2 * dt
                    R = torch.clamp(R + dRdt * dt, min=1e-10)
                    T_s = T_s + dTsdt * dt
                
                # Store results
                R_values2[i] = R
                dRdt_values2[i] = dRdt
                Ts_values2[i] = T_s
                
                pbar.update(1)
        
        # Store final point
        R_values2[-1] = R
        dRdt_values2[-1] = dRdt
        Ts_values2[-1] = T_s
        
        # Combine results and convert back to CPU/numpy
        t_combined = np.concatenate((t1.cpu().numpy(), t2.cpu().numpy()))
        R_combined = np.concatenate((R_values1.cpu().numpy(), R_values2.cpu().numpy()))
        dRdt_combined = np.concatenate((dRdt_values1.cpu().numpy(), dRdt_values2.cpu().numpy()))
        Ts_combined = np.concatenate((Ts_values1.cpu().numpy(), Ts_values2.cpu().numpy()))
        
        # Create solution object with the same format as solve_ivp
        class GPUSolution:
            t = t_combined
            y = np.array([R_combined, dRdt_combined, Ts_combined])
            success = True
            
        return GPUSolution()
        
    except ImportError:
        print("PyTorch not installed. Falling back to CPU mode.")
        return solve_bubble_dynamics_two_stage()

def plot_results(solution):
    """Create plots for R vs t, dR/dt vs t, and T_s vs t."""
    t = solution.t
    R = solution.y[0]
    dRdt = solution.y[1]
    T_s = solution.y[2]
    
    # Create plots
    fig, axs = plt.subplots(3, 1, figsize=(12, 14))
    
    # Plot 1: R vs t (log-log scale)
    axs[0].loglog(t, R * 100, 'b-', linewidth=2)  # Convert to cm
    axs[0].set_xlabel('Time (s)', fontsize=12)
    axs[0].set_ylabel('Bubble Radius (cm)', fontsize=12)
    axs[0].set_title('Bubble Radius vs Time', fontsize=14)
    axs[0].grid(True, which="both", ls="--")
    axs[0].text(0.05, 0.95, f'Initial temp: {T_inf} K\nSuperheat: {T_inf-T_b} K', 
                transform=axs[0].transAxes, fontsize=10, va='top')
    
    # Plot 2: dR/dt vs t (log-log scale)
    axs[1].loglog(t, abs(dRdt) * 100, 'r-', linewidth=2)  # Convert to cm/s
    axs[1].set_xlabel('Time (s)', fontsize=12)
    axs[1].set_ylabel('|dR/dt| (cm/s)', fontsize=12)
    axs[1].set_title('Bubble Growth Rate vs Time', fontsize=14)
    axs[1].grid(True, which="both", ls="--")
    
    # Plot 3: T_s vs t (semi-log scale)
    axs[2].semilogx(t, T_s, 'g-', linewidth=2)
    axs[2].set_xlabel('Time (s)', fontsize=12)
    axs[2].set_ylabel('Surface Temperature (K)', fontsize=12)
    axs[2].set_title('Bubble Surface Temperature vs Time', fontsize=14)
    axs[2].grid(True, which="both", ls="--")
    axs[2].axhline(y=T_b, color='k', linestyle='--', alpha=0.7)
    axs[2].text(t[-1]/100, T_b+5, f'Boiling point: {T_b} K', fontsize=10, va='bottom')
    
    plt.tight_layout()
    plt.savefig('bubble_dynamics_results.png', dpi=300)
    plt.show()

def main():
    """Main function to run the optimized simulation."""
    import time
    
    print("Solving Rayleigh-Plesset equation for vapor bubble in liquid sodium (OPTIMIZED MODE)...")
    print(f"Initial temperature: {T_inf} K")
    print(f"Superheat: {T_inf - T_b} K")
    print(f"Ambient pressure: {p_inf/1e5} bar")
    
    # Measure execution time
    start_time = time.time()
    
    # Try to use GPU first, falls back to CPU if needed
    solution = solve_bubble_dynamics_gpu()
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    if solution.success:
        print(f"Simulation completed successfully in {execution_time:.2f} seconds!")
        plot_results(solution)
        
        # Print some key results
        max_radius = np.max(solution.y[0]) * 1000  # mm
        max_velocity = np.max(solution.y[1]) * 100  # cm/s
        min_temperature = np.min(solution.y[2])
        
        print(f"\nResults summary:")
        print(f"Maximum bubble radius: {max_radius:.3f} mm")
        print(f"Maximum growth rate: {max_velocity:.3f} cm/s")
        print(f"Minimum surface temperature: {min_temperature:.1f} K")
    else:
        print("Simulation failed!")

if __name__ == "__main__":
    main()