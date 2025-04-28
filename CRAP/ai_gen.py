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

# Subplots: 2 rows per case (radius, velocity)
fig, axs = plt.subplots(7, 2, figsize=(14, 22))

for i, (Tb, Delta_T, R0, p_v, Jakob) in enumerate(conditions):
    ax_R = axs[i, 0]
    ax_v = axs[i, 1]

    # -- Thermal Growth --
    v_thermal = (3/np.pi)**0.5 * k / (L * rho_v) * Delta_T / np.sqrt(D * t)
    R_thermal = R0 + cumulative_trapezoid(v_thermal, t, initial=0)

    # -- Inertial Growth --
    inertial_valid = (p_v - p_inf) > 0
    if inertial_valid:
        v_inertial = np.sqrt(2/3 * (p_v - p_inf) / rho)
        R_inertial = R0 + v_inertial * t
        ax_R.plot(t, R_inertial, 'b--', label='Inertial (asymptotic)')
        ax_v.plot(t, v_inertial * np.ones_like(t), 'b--')

    # -- Full RP model --
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
    v_rp = np.gradient(R_rp, sol.t)  # numerical dR/dt

    ax_R.plot(sol.t, R_rp, 'k-', label='Full RP + thermal')
    ax_v.plot(sol.t, v_rp, 'k-')

    # -- Plot thermal model --
    ax_R.plot(t, R_thermal, 'r--', label='Thermal (asymptotic)')
    ax_v.plot(t, v_thermal, 'r--')

    # Radius plot styling
    ax_R.set_xscale('log')
    ax_R.set_yscale('log')
    ax_R.set_title(f'Case {i+1}: ΔT={Delta_T}K, R₀={R0*1e6:.1f}μm')
    ax_R.grid(True, which='both', linestyle='--', linewidth=0.5)
    if i == 6:
        ax_R.set_xlabel('Time (s)')
    if i == 0:
        ax_R.set_ylabel('Radius (m)')
        ax_R.legend(loc='upper left', fontsize='small')

    # Velocity plot styling
    ax_v.set_xscale('log')
    ax_v.set_yscale('log')
    ax_v.set_title(f'dR/dt vs Time — Case {i+1}')
    ax_v.grid(True, which='both', linestyle='--', linewidth=0.5)
    if i == 6:
        ax_v.set_xlabel('Time (s)')
    if i == 0:
        ax_v.set_ylabel('Velocity (m/s)')
        ax_v.legend(['Inertial', 'Full RP', 'Thermal'], fontsize='small', loc='upper right')

# Layout
plt.tight_layout()
plt.suptitle('Vapour Bubble Growth — Radius and Velocity for All 7 Cases', fontsize=18, y=1.02)
plt.subplots_adjust(hspace=0.5)
plt.show()
