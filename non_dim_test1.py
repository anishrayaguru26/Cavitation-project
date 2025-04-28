import numpy as np
import matplotlib.pyplot as plt

# Define the R~(t~) function
def R_tilde(t_tilde):
    term1 = (0.5 * np.pi**2 * t_tilde + 1)**(3/2)
    term2 = (0.5 * np.pi**2 * t_tilde)**(3/2)
    prefactor = (2/np.pi**2)**(2/3) * (2/3)**(1/2)
    return prefactor * (term1 - term2 - 1)

# Create a range of t~ values
t_tilde = np.logspace(-7, 0, 1000)  # 10^-7 to 1, logarithmic spacing

# Calculate R~ for each t~
R_tilde_values = R_tilde(t_tilde)

# Plotting
plt.figure(figsize=(8,6))
plt.loglog(t_tilde, R_tilde_values, 'k-', label=r'$\tilde{R}(\tilde{t})$')  # black solid line

plt.xlabel(r'Dimensionless Time $\tilde{t}$')
plt.ylabel(r'Dimensionless Radius $\tilde{R}$')
plt.title('Vapour-bubble Growth in Superheated Liquid')
plt.grid(True, which="both", ls='--')
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig('dimensionless_bubble_growth.png', dpi=300)
plt.show()
