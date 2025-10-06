# This code is intended to compare the results for the 2D power spectrum in real space.
# First, it will simply overlay the power spectrum results
# 
# Next, compute the residuals between the model and the calculated power spectra
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

# FFT power spectrum
pk_FFT = np.loadtxt(".txt", skiprows=1) # add in FFT powerspectrum text file
k_FFT = pk_FFT[:, 0]
P_k_FFT = pk_FFT[:, 1]

# 2pt correlation function power spectrum
pk_2PCP = np.loadtxt(".txt", skiprows=1) # add in 2PCP powerspectrum text file
k_2PCP = pk_2PCP[:, 0]
P_k_2PCP = pk_2PCP[:, 1]

# plot all of the power spectrums together
plt.loglog(k_true,P_k_true, c = "black", label = 'true', ls = 'solid')
plt.loglog(k_FFT,P_k_FFT, c = "black", label = 'dodgerblue', ls = 'dashdot')
plt.loglog(k_2PCP,P_k_2PCP, c = "black", label = 'mediumseagreen', ls = 'dotted')
plt.xlabel(r"$k$ [$h \mathrm{Mpc}^{-1}$]")
plt.ylabel(r"$P(k)$ [$h^{-3} \mathrm{Mpc}^3$]")
plt.legend()
plt.show()

# compute the residuals
res_FFT = []
res_2PCP = []

# use an interpolator to find values at the same values of k
# true power spectrum

# FFT power spectrum

# 2pt correlation power spectrum






