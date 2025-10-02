# This code is intended to compare the results for the 2D power spectrum in real space.
# First, it will simply overlay the power spectrum results
# 
# Next, calculate chi squared using ____ method
#
# Finally, calculate chi squared using inverse covariance matrix

import numpy as np
import matplotlib.pyplot as plt

# read in powerspectrum files
# true power spectrum
pk_true = np.loadtxt("power_spectrum.txt", skiprows=1)
k_true = pk_true[:, 0]
P_k_true = pk_true[:, 1]

plt.loglog(k_true,P_k_true, c = "black", label = 'true')
plt.xlabel(r"$k$ [$h \mathrm{Mpc}^{-1}$]")
plt.ylabel(r"$P(k)$ [$h^{-3} \mathrm{Mpc}^3$]")
plt.legend()
plt.show()