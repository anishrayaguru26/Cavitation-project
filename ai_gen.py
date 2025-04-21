import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, cumulative_trapezoid

# Constants
k = 71.7        # W/mK
L = 2.6e6       # J/kg
rho_v = 0.5     # kg/m³
D = 6.4e-5      # m²/s
rho = 927       # kg/m³
p_inf = 1.0e5   # Pa
sigma = 0.2     # N/m

# Table 1: Tb, Delta_T, R0, p_v, Jakob
conditions = [
    (1083.6, 340.1, 2.5e-5, 1.253e5, 565.7),
    (1154.6, 278.9, 2.5e-5, 3.212e4, 227.7),
    (1154.6, 133.1, 1e-4,   9.899e4, 108.7),
    (1154.6, 22.1,  1e-3,   1.088e5, 18.04),
    (1235.2, 90.1,  1e-4,   2.923e5, 38.08),
    (1345.9, 14.7,  4e-4,   5.619e5, 2.979),
    (1390.2, 4.66,  1e-3,   2.936e4, 0.7331)
]

# Time array
t = np.logspace(-7, -1, 500)

# Subplots
fig, axs = plt.subplots(3, 3, figsize=(16, 12))
axs = axs.flatten()

for i, (Tb, Delta_T, R0, p_v, Jakob) in enumerate(conditions):
    ax = axs[i]

    # -- Thermal Growth --
    v_thermal = (3/np.pi)**0.5 * k / (L * rho_v) * Delta_T / np.sqrt(D * t)
    R_thermal = R0 + cumulative_trapezoid(v_thermal, t, initial=0)

    # -- Inertial Growth (only if p_v > p_inf) --
    if (p_v - p_inf) > 0:
        v_inertial = np.sqrt(2/3 * (p_v - p_inf) / rho)
        R_inertial = R0 + v_inertial * t
        ax.plot(t, R_inertial, 'b--', label='Inertial (asymptotic)')

    # -- Full RP + thermal model --
    def dRdt(ti, y):
        R, dR = y
        if R <= 0:
            return [0, 0]
        inertial_term = (p_v - p_inf - 2 * sigma / R) / rho
        d2R = (inertial_term - 1.5 * dR**2) / R
        return [dR, d2R]

    y0 = [R0, 1e-6]
    sol = solve_ivp(dRdt, [t[0], t[-1]], y0, rtol=1e-8, atol=1e-10, t_eval=t)
    R_rp = sol.y[0]
    ax.plot(sol.t, R_rp, 'k-', label='Full RP + thermal')

    # -- Plot thermal curve
    ax.plot(t, R_thermal, 'r--', label='Thermal (asymptotic)')

    # -- Axes and title
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(f'Case {i+1}: ΔT={Delta_T}K, R₀={R0*1e6:.1f}μm')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    if i % 3 == 0:
        ax.set_ylabel('Radius (m)')
    if i >= 6:
        ax.set_xlabel('Time (s)')

# -- Add legend and layout --
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3, fontsize='large')
fig.suptitle('Vapour Bubble Growth (Radius vs Time) for All 7 Cases', fontsize=16)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.show()
