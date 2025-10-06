# This code is intended to compare the results for the 2D power spectrum in real space.
# First, it will simply overlay the power spectrum results
# 
# Next, compute the residuals between the model and the calculated power spectra
# 
# Next, calculate chi squared using just the chi squared formula 
#
# Finally, calculate chi squared using inverse covariance matrix

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# read in powerspectrum files
# true power spectrum
pk_true = np.loadtxt("power_spectrum.txt", skiprows=1)
k_true = pk_true[:, 0]
P_k_true = pk_true[:, 1]

# FFT power spectrum
pk_fft = np.loadtxt("power_spec_npfft.txt", skiprows=1)
k_fft = pk_fft[:, 0]
P_k_fft = pk_fft[:, 1]

# # 2pt correlation function power spectrum
pk_2PCP = np.loadtxt(".txt", skiprows=1) # add in 2PCP powerspectrum text file
k_2PCP = pk_2PCP[:, 0]
P_k_2PCP = pk_2PCP[:, 1]

# plot all of the power spectrums together
plt.loglog(k_true,P_k_true, c = "black", label = 'true', ls = 'solid')
plt.loglog(k_fft,P_k_fft, c = "dodgerblue", label = 'FFT', ls = 'dashdot')
plt.loglog(k_2PCP,P_k_2PCP, c = "mediumseagreen", label = '2PCP', ls = 'dotted')
plt.xlabel(r"$k$ [$h \mathrm{Mpc}^{-1}$]")
plt.ylabel(r"$P(k)$ [$h^{-3} \mathrm{Mpc}^3$]")
plt.legend()
plt.show()

# ----------------------------------------------------------------------------------------------
# use an interpolator to find values at the same values of k
# define the values of k 
k_values = np.linspace(0.01, 1, 1000)

# true power spectrum
interp_true = CubicSpline(k_true, P_k_true)
Pi_true = interp_true(k_values)

# FFT power spectrum
interp_fft = CubicSpline(k_fft, P_k_fft)
Pi_fft = interp_fft(k_values)

# 2pt correlation power spectrum
interp_2pcp = CubicSpline(k_fft, P_k_2PCP)
Pi_2pcp = interp_2pcp(k_values)

# compute the residuals
res_fft = []
res_2pcp = []

for x, y, z in zip(Pi_true, Pi_fft, Pi_2pcp):
    res_fft = np.append(res_fft, Pi_true - Pi_fft)
    res_2pcp = np.append(res_2pcp, Pi_true - Pi_2pcp)

# plot the residuals
plt.semilogx(k_values, res_fft, c = "dodgerblue", label = 'FFT', marker = 'o')
plt.semilogx(k_values, res_2pcp, c = "mediumseagreen", label = '2PCP', marker = 'o')
plt.xlabel(r"$k$ [$h \mathrm{Mpc}^{-1}$]")
plt.ylabel(r"residuals [$h^{-3} \mathrm{Mpc}^3$]")
plt.legend()
plt.show()

# ----------------------------------------------------------------------------------------------
# calculate 

