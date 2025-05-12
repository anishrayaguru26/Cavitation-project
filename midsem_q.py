import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
R0 = 1e-3  # Initial radius
k = 1.4
C1 = (200/3) - 900 + 0.1
C2 = 900
C3 = 200 / 3

# Define the RHS of the ODE: dR/dt = sqrt(...) — only real solutions allowed
def bubble_rhs(t, R):
    if R[0] <= 0:
        return [0.0]  # stop if R goes nonphysical
    term = C1 * (R0 / R[0])**(3 * k) + C2 * (R0 / R[0])**3 - C3
    if term < 0:
        return [0.0]  # non-physical (imaginary speed)
    return [np.sqrt(term)]

# Time span (e.g. 0 to 5 milliseconds)
t_span = (0, 0.005)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Initial condition
R_init = [R0]

# Solve ODE
sol = solve_ivp(bubble_rhs, t_span, R_init, method='Radau', t_eval=t_eval, rtol=1e-6, atol=1e-9)

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(sol.t * 1000, sol.y[0], label="Bubble Radius $R(t)$")
plt.xlabel("Time [ms]")
plt.ylabel("Radius [units of $R_0$]")
plt.title("Bubble Growth from Stiff Equation")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()