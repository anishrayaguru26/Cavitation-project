import numpy as np
from scipy.integrate import solve_ivp, quad
import matplotlib.pyplot as plt

# Physical constants for liquid sodium
rho = 850  # kg/m³
sigma = 0.18  # N/m
p_inf = 101325  # Pa
T_inf = 700  # K (about 427°C)
k = 71  # W/m·K
D = 2.5e-8  # m²/s
L = 8.22e5  # J/kg

# Vapor pressure function (Sodium) - Antoine like equation
def p_v(Ts):
    return np.exp(24.494 - 34100/Ts)  # Ts in K, p_v in Pa

# Initial conditions
R0 = 1e-5  # Initial bubble radius, 10 microns
dRdt0 = 0  # Initially stationary

# Time span and evaluation points
t_span = (1e-6, 1e-1)
t_eval = np.logspace(-6, -1, 1000)  # 1000 points logarithmically spaced

# Rayleigh-Plesset ODE system
def rayleigh_plesset(t, y):
    R, dRdt = y
    Pv = p_v(T_inf)  # Assuming T_s ≈ T_inf initially
    d2Rdt2 = (1/rho) * (Pv - p_inf - 2*sigma/R) - (3/2)*(dRdt**2)/R
    d2Rdt2 /= R
    return [dRdt, d2Rdt2]

# Function to compute temperature evolution for given initial radius
def compute_temperature(R0):
    # Solve with stricter tolerances and more steps
    sol = solve_ivp(rayleigh_plesset, t_span, [R0, dRdt0], 
                    t_eval=t_eval,
                    method='RK45',
                    rtol=1e-10,  # Stricter relative tolerance
                    atol=1e-12,  # Stricter absolute tolerance
                    max_step=1e-6)  # Smaller maximum step size
    
    if not sol.success:
        raise ValueError(f"Integration failed: {sol.message}")
    
    R = sol.y[0]
    time = sol.t
    
    # Ensure we have enough valid points
    valid_mask = np.isfinite(R)
    if np.sum(valid_mask) < 2:
        raise ValueError("Not enough valid points for interpolation")
        
    time = time[valid_mask]
    R = R[valid_mask]
    
    # Use PchipInterpolator with verified data
    from scipy.interpolate import PchipInterpolator
    R3_rhov_interp = PchipInterpolator(time, R**3 * rho)
    R4_interp = PchipInterpolator(time, R**4)
    
    Ts = []
    for t_now in time:
        def integrand1(x):
            if x >= t_now or x < time[0]:
                return 0.0
            
            try:
                dR3_rhov_dx = R3_rhov_interp.derivative()(x)
                inner_integral = R4_interp.antiderivative()(t_now) - R4_interp.antiderivative()(x)
                
                if inner_integral <= 0:
                    return 0.0
                    
                result = L * dR3_rhov_dx * inner_integral**(-0.5)
                return 0.0 if not np.isfinite(result) else result
            except:
                return 0.0

        try:
            outer_integral, _ = quad(integrand1, time[0], t_now, 
                                   epsabs=1e-6, epsrel=1e-6,
                                   limit=200)
        except:
            outer_integral = 0.0
        
        Ts_val = T_inf - (1/(3*k)) * (D/np.pi)**0.5 * outer_integral
        Ts_val = max(min(Ts_val, T_inf * 1.5), T_inf * 0.5)
        Ts.append(Ts_val)
    
    return time, np.array(Ts)

# Initial radii (in meters)
R0_values = [4e-6, 6e-6, 8e-6]  # 4, 6, and 8 microns

# Plotting with multiple curves
plt.figure(figsize=(10, 8))

for R0, label in zip(R0_values, ['4 μm', '6 μm', '8 μm']):
    time, Ts = compute_temperature(R0)
    plt.semilogx(time, Ts, '-', linewidth=1, label=label)
    plt.plot(time[::10], Ts[::10], 'o', markersize=3, fillstyle='none')

# Configure grid
plt.grid(True, which="major", ls="-", alpha=0.5)
plt.grid(True, which="minor", ls=":", alpha=0.3)

# Axis labels and ticks
plt.xlabel('t (s)')
plt.ylabel('$T_s$ (K)')

# Set specific limits
plt.xlim(1e-6, 1e-1)
plt.legend(frameon=False)

plt.tight_layout()
plt.show()
