#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm

# Physical properties
rho = 800      # kg/m^3
sigma = 0.2    # N/m
k = 60         # W/m·K
D = 6.5e-7     # m^2/s
L = 4.3e6      # J/kg
p_inf = 0.5e5  # Pa
T_inf = 1396   # K
T_b = 1056     # K
R_initial = 1e-6  # m
dRdt_initial = 0  # m/s

# Antoine equation parameters for vapor pressure (in bar)
A, B, C = 4.52, 5100, 0

def vapor_pressure(T):
    return 10**(A - B/(T + C)) * 1e5  # Pa

def dSdt(t, S):
    """ODE system with smarter handling of tiny R"""
    R = max(S[0], 1e-12)
    dRdt = S[1]
    T_s = S[2]
    
    if R < 1e-9:
        return [0.0, 0.0, 0.0]  # Freeze dynamics if too tiny

    p_v = vapor_pressure(T_s)
    d2Rdt2 = (1/rho) * (p_v - p_inf - 2*sigma/R) / R - 1.5 * (dRdt**2) / R
    dTsdt = -L * rho * dRdt / k * (T_s - T_b) / R
    
    return [dRdt, d2Rdt2, dTsdt]

def bubble_growth_event(t, S):
    """Stop integration when radius > 1mm"""
    return S[0] - 1e-3

bubble_growth_event.terminal = True
bubble_growth_event.direction = 1

class ProgressIVP:
    """Wrapper with progress bar"""
    
    def __init__(self, fun, t_span, y0, t_eval=None, rtol=1e-6, atol=1e-9, **kwargs):
        self.fun = fun
        self.t_span = t_span
        self.y0 = y0
        self.t_eval = t_eval
        self.rtol = rtol
        self.atol = atol
        self.kwargs = kwargs
        self.n_steps = 0

    def wrapper(self, t, y):
        self.n_steps += 1
        progress_bar.set_description(f"t = {t:.2e} s | steps = {self.n_steps}")
        progress_bar.update(1)
        return self.fun(t, y)

    def solve(self, max_steps=10000):
        global progress_bar
        progress_bar = tqdm(total=max_steps, dynamic_ncols=True)
        result = solve_ivp(
            self.wrapper,
            self.t_span,
            self.y0,
            t_eval=self.t_eval,
            method='LSODA',    # 🛑 Switch to LSODA
            rtol=self.rtol,
            atol=self.atol,
            events=[bubble_growth_event],  # 🛑 Auto-stop
            **self.kwargs
        )
        progress_bar.close()
        return result

def solve_stage(t_start, t_end, S0, t_points, dynamic_tol=False):
    """Solve one stage with optional dynamic tolerance relaxation"""
    t_span = [t_start, t_end]
    t_eval = np.logspace(np.log10(t_start), np.log10(t_end), t_points)

    # Set tighter tolerances initially
    rtol = 1e-6
    atol = 1e-9

    if dynamic_tol:
        rtol = 1e-4
        atol = 1e-7

    solver = ProgressIVP(
        dSdt,
        t_span,
        S0,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol
    )
    return solver.solve(max_steps=5000)

def solve_bubble_dynamics_two_stage():
    print("Starting two-stage simulation...")

    # Stage 1: Early time
    print("Stage 1: Early dynamics...")
    S0 = [R_initial, dRdt_initial, T_inf]
    sol1 = solve_stage(
        t_start=1e-9,
        t_end=1e-6,
        S0=S0,
        t_points=300,
        dynamic_tol=False
    )
    
    # Stage 2: Later time
    print("Stage 2: Later dynamics (looser tolerances)...")
    S0_stage2 = [sol1.y[0][-1], sol1.y[1][-1], sol1.y[2][-1]]
    sol2 = solve_stage(
        t_start=1e-6,
        t_end=1e0,
        S0=S0_stage2,
        t_points=300,
        dynamic_tol=True
    )

    t_combined = np.concatenate((sol1.t, sol2.t))
    y_combined = np.hstack((sol1.y, sol2.y))
    
    class CombinedSolution:
        t = t_combined
        y = y_combined
        success = sol1.success and sol2.success
        
    return CombinedSolution()

def plot_results(solution):
    t, R, dRdt, T_s = solution.t, solution.y[0], solution.y[1], solution.y[2]

    fig, axs = plt.subplots(3, 1, figsize=(12, 14))

    axs[0].loglog(t, R * 100, 'b-', linewidth=2)
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Radius (cm)')
    axs[0].set_title('Bubble Radius vs Time')
    axs[0].grid(True, which="both", ls="--")

    axs[1].loglog(t, np.abs(dRdt) * 100, 'r-', linewidth=2)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Growth Rate (cm/s)')
    axs[1].set_title('Growth Rate vs Time')
    axs[1].grid(True, which="both", ls="--")

    axs[2].semilogx(t, T_s, 'g-', linewidth=2)
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Surface Temperature (K)')
    axs[2].set_title('Surface Temperature vs Time')
    axs[2].grid(True, which="both", ls="--")
    axs[2].axhline(y=T_b, color='k', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig('bubble_dynamics_optimized.png', dpi=300)
    plt.show()

def main():
    print("Solving Rayleigh-Plesset for vapor bubble (FAST MODE)...")
    print(f"Initial temperature: {T_inf} K")
    print(f"Superheat: {T_inf - T_b} K")
    print(f"Ambient pressure: {p_inf/1e5} bar")

    solution = solve_bubble_dynamics_two_stage()

    if solution.success:
        print("\nSimulation completed successfully!")
        plot_results(solution)

        max_radius = np.max(solution.y[0]) * 1000  # mm
        max_velocity = np.max(solution.y[1]) * 100  # cm/s
        min_temp = np.min(solution.y[2])

        print(f"Max radius: {max_radius:.3f} mm")
        print(f"Max growth rate: {max_velocity:.2f} cm/s")
        print(f"Min surface temp: {min_temp:.2f} K")
    else:
        print("Simulation failed!")

if __name__ == "__main__":
    main()
