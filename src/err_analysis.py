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
from scipy.optimize import curve_fit
import glob

# read in powerspectrum files
# true power spectrum
pk_true = np.loadtxt("data/power_spectrum.txt", skiprows=1)
# print(pk_true.shape)
k_true = pk_true[:, 0]
P_k_true = pk_true[:, 1]

# 2pt correlation function power spectrum
pk_2PCP = np.loadtxt("data/pk_from_xi_1.txt", skiprows=1)
# print(pk_2PCP.shape)
k_2PCP = pk_2PCP[:, 0]
P_k_2PCP = pk_2PCP[:, 1]
err_2PCP = pk_2PCP[:,2]

# find all of the fft power spectra
ffts = glob.glob('data/power_spec_*.txt')
print(ffts)

k_ffts = []
Pk_ffts = []
for file in ffts:
    # FFT power spectrum
    pk_fft = np.loadtxt(file, skiprows=0)
    k_fft = pk_fft[0, :]
    # k_ftts = np.append(k_ffts, k_fft)
    P_k_fft = pk_fft[1, :]
    # Pk_ftts = np.append(Pk_ffts, P_k_fft)

    k_ffts.append(k_fft)
    Pk_ffts.append(P_k_fft)

cmap = plt.get_cmap("spring_r")
colors = [cmap(i / (len(ffts)-1)) for i in range(len(ffts))]

# plot all of the power spectrums together
plt.figure()
plt.loglog(k_true,P_k_true, c = "grey", label = 'true', ls = 'solid')
plt.loglog(k_2PCP,P_k_2PCP, c = "mediumseagreen", label = '2PCP', ls = 'dotted')
for i, (k, P) in enumerate(zip(k_ffts, Pk_ffts)):
    plt.loglog(k, P, c = colors[i], label = fr'FFT {i+1}', ls = 'dashdot')
plt.xlabel(r"$k$ [$h \mathrm{Mpc}^{-1}$]")
plt.ylabel(r"$P(k)$ [$h^{-3} \mathrm{Mpc}^3$]")
plt.xlim(1e-2, 1e0)
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
interp_2pcp = CubicSpline(k_2PCP, P_k_2PCP)
Pi_2pcp = interp_2pcp(k_values)

# compute the residuals
res_fft = []
res_2pcp = []

for x, y, z in zip(Pi_true, Pi_fft, Pi_2pcp):
    res_fft = np.append(res_fft, x - y) # true - measured
    res_2pcp = np.append(res_2pcp, x - z)

# plot the residuals
# plt.semilogx(k_values, res_fft, c = "dodgerblue", label = 'FFT', marker = '.')
# plt.semilogx(k_values, res_2pcp, c = "mediumseagreen", label = '2PCP', marker = '.')
# plt.hlines(0, np.min(k_values) - 10, 1, colors = "grey", linestyles = 'dashed')
# plt.xlabel(r"$k$ [$h \mathrm{Mpc}^{-1}$]")
# plt.ylabel(r"residuals [$h^{-3} \mathrm{Mpc}^3$]")
# plt.title("Residuals")
# plt.legend()
# plt.show()

# ----------------------------------------------------------------------------------------------
# define function for chi squared 
def chi_squared(calc, true):
    chi = []
    for c, t in zip(calc, true):
        num = (c - t)**2
        chi = np.append(chi, num/t)
    chi2 = np.sum(chi)
    return chi2

# chi^2 for FFT
chi2_fft = chi_squared(Pi_fft, Pi_true)
chi2_2pcp = chi_squared(Pi_2pcp, Pi_true)
dof = len(k_values - 1)
print("from basic equation")
print("----Reduced Chi-Squared Results----")
print("FFT:", chi2_fft/dof)
print("2PCP:", chi2_2pcp/dof)


# ----------------------------------------------------------------------------------------------
# calculate chi squared using covariance matrix
# pcov_fft = 
# pcov_2pcp = 
# def pspec(k, A, p):
#     return A*k**p

# params, pcov_fft = curve_fit(pspec, k_values, Pi_fft, [1, 2])



# # compute chi squared
# chi2_fft_cov = res_fft.T @ pcov_fft @ res_fft / dof
# # chi2_2pcp_cov = res_2pcp.T @ pcov_2pcp @ res_2pcp / dof
# print("from covariance matrix")
# print("----Reduced Chi-Squared Results----")
# print("FFT:", chi2_fft_cov/dof)
# # print("2PCP:", chi2_2pcp_cov/dof)
