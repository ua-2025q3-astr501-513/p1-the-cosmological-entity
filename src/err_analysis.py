# This code is intended to compare the results for the 2D power spectrum in real space.
# First, it will simply overlay the power spectrum results
# 
# Next, compute the residuals between the model and the calculated power spectra
# 
# Next, calculate chi squared using just the chi squared formula 
#
# Finally, calculate chi squared using inverse covariance matrix

import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import glob

# define sorting method in case files aren't read in order
def sort_key(filename):
    # find the version number
    match = re.search(r'v(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        # If there's no version (e.g. "power_spectrum.txt"), treat it as version 0
        return 0

# read in powerspectrum files
# true power spectra
truth = glob.glob('data/power_spectrum*.txt')
# sort the list
truth.sort(key=sort_key)
# print(truth)

k_true = []
Pk_true = []
for file in truth:
    pk_true = np.loadtxt(file, skiprows=1)
    k = pk_true[:, 0]
    Pk = pk_true[:, 1]

    k_true.append(k)
    Pk_true.append(Pk)

# fft power spectra
ffts = glob.glob('data/power_spec_*.txt')
# sort the list
ffts.sort(key=sort_key)
# print(ffts)

# create lists for each set of power spectra
k_ffts = []
Pk_ffts = []
for file in ffts:
    # FFT power spectrum
    pk_fft = np.loadtxt(file, skiprows=0)
    k_fft = pk_fft[0, :]
    P_k_fft = pk_fft[1, :]

    k_ffts.append(k_fft)
    Pk_ffts.append(P_k_fft)

# 2pt correlation function power spectrum
pk_2PCP = np.loadtxt("data/pk_from_xi_1.txt", skiprows=1)
k_2PCP = pk_2PCP[:, 0]
P_k_2PCP = pk_2PCP[:, 1]
err_2PCP = pk_2PCP[:,2]

# plot all of the power spectra together
# get colors
cmap = plt.get_cmap("cool")
colors = [cmap(i / (len(ffts)-1)) for i in range(len(ffts))]

# plot true power spectra for 2PCP and FFT 1, then plot rest of FFT
plt.figure()
plt.loglog(k_true[0],Pk_true[0], c = "grey", label = 'true', ls = 'solid')
plt.loglog(k_2PCP,P_k_2PCP, c = "mediumseagreen", label = '2PCP', ls = 'dotted')
for i, (k, P) in enumerate(zip(k_ffts, Pk_ffts)):
    plt.loglog(k, P, c = colors[i], label = fr'FFT {i+1}', ls = 'dashdot')
plt.xlabel(r"$k$ [$h \mathrm{Mpc}^{-1}$]")
plt.ylabel(r"$P(k)$ [$h^{-3} \mathrm{Mpc}^3$]")
plt.xlim(1e-2, 1e0)
plt.title("Power Spectra")
plt.legend()
plt.savefig("results/powerspectrum.pdf")
plt.show()

# ----------------------------------------------------------------------------------------------
# use an interpolator to find values at the same values of k
# define the values of k 
k_values = np.linspace(0.01, 0.15, 1000)

# true power spectrum
PI_true = []
for k, P in zip(k_true, Pk_true):
    interp = CubicSpline(k, P)
    P_true = interp(k_values)
    
    PI_true.append(P_true)

# FFT power spectrum
PI_fft = []
for k, P in zip(k_ffts, Pk_ffts):
    interp_fft = CubicSpline(k, P)
    P_fft = interp_fft(k_values)

    PI_fft.append(P_fft)

# 2pt correlation power spectrum
interp_2pcp = CubicSpline(k_2PCP, P_k_2PCP)
PI_2pcp = interp_2pcp(k_values)

# plot the interpolated power spectra
plt.figure()
plt.loglog(k_values,PI_true[0], c = "grey", label = 'true', ls = 'solid')
plt.loglog(k_values,PI_2pcp, c = "mediumseagreen", label = '2PCP', ls = 'dotted')
for i, P in enumerate(PI_fft):
    plt.loglog(k_values, P, c = colors[i], label = fr'FFT {i+1}', ls = 'dashdot')
plt.xlabel(r"$k$ [$h \mathrm{Mpc}^{-1}$]")
plt.ylabel(r"$P(k)$ [$h^{-3} \mathrm{Mpc}^3$]")
plt.xlim(1e-2, 0.16)
plt.title("Interpolated Power Spectra")
plt.legend()
plt.savefig("results/interpolated_powerspectrum.pdf")
plt.show()

# compute the residuals
res_fft = []
diff_fft = []
res_2pcp = []

# FFT
for true, fft in zip(PI_true, PI_fft):
    single_res = []
    single_diff = []
    for x, y in zip(true, fft):
        single_res.append(x - y) # true - measured
        single_diff.append(y - x) # measured - true
    
    res_fft.append(single_res)
    diff_fft.append(single_diff)

# 2PCP
res_2pcp = np.append(res_2pcp, PI_true[0] - PI_2pcp)

# plot the residuals
cmap = plt.get_cmap("cool")
colors = [cmap(i / (len(res_fft)-1)) for i in range(len(res_fft))]
# plt.figure()
plt.semilogx(k_values, res_2pcp, c = "mediumseagreen", label = '2PCP', marker = '.')
for i, res in enumerate(res_fft):
    plt.semilogx(k_values, res, c = colors[i], label = fr'FFT {i+1}', marker = ".")
plt.hlines(0, np.min(k_values) - 10, 1, colors = "grey", linestyles = 'dashed')
plt.xlabel(r"$k$ [$h \mathrm{Mpc}^{-1}$]")
plt.ylabel(r"residuals [$h^{-3} \mathrm{Mpc}^3$]")
plt.title("Residuals")
plt.xlim(1e-2, 0.16)
plt.legend()
plt.savefig("results/residuals.pdf")
plt.show()

# ----------------------------------------------------------------------------------------------
# define function for chi squared
def chi_squared(calc, true, err = None):
    chi = []
    for c, t in zip(calc, true):
        num = (c - t)**2
        if err is None:
            chi = np.append(chi, num/t)
        else:
            chi = np.append(chi, num/(err**2))
    chi2 = np.sum(chi)
    return chi2

# chi^2 for FFT
chi2_fft = []
for fft, true in zip(PI_fft, PI_true):
    chi2 = chi_squared(fft, true)

    chi2_fft.append(chi2)

# chi^2 for 2PCP
chi2_2pcp = chi_squared(PI_2pcp, PI_true)

# compare to non-interpolated data for 2PCP
chi2_2pcp_noi = chi_squared(P_k_2PCP, Pk_true[0])
dof_noi = len(P_k_2PCP) - 1

# calculate for a truncated data set
t_index = np.where(k_2PCP < 0.15)
P_2pcp_t = P_k_2PCP[:np.max(t_index)]
P_true_t = Pk_true[0][:np.max(t_index)]
P_err_t = err_2PCP[:np.max(t_index)]

chi2_2pcp_t = chi_squared(P_2pcp_t, P_true_t)
dof_t = np.max(t_index) - 1
# print(dof_t)

# calculate reduced chi^2
dof = len(k_values) - 1
rchi2_fft = [chi2 / dof for chi2 in chi2_fft]
# print(len(rchi2_fft))
rchi2_2pcp = chi2_2pcp/dof

print("from basic equation")
print("----Reduced Chi-Squared Results----")
print("2PCP Full w/  Interpolation:", chi2_2pcp/dof)
print("2PCP Full w/o Interpolation:", chi2_2pcp_noi/dof_noi)
print("2PCP Truncated w/o Interpolation:", chi2_2pcp_t/dof_t)
for i in range(len(rchi2_fft)):
    print(fr'FFT {i}:', chi2_fft[i]/dof)

# ----------------------------------------------------------------------------------------------
# calculate chi squared using covariance matrix
# find the covariance matrix of the FFT mocks
P_fft_cov = np.array(PI_fft)
meanpk0 = np.mean(P_fft_cov, axis=0)
deltaP = P_fft_cov - meanpk0 
cov = np.dot(deltaP.T, deltaP) / (P_fft_cov.shape[0] - 1)
# since we don't have enough mocks to use matrix multiplication to get 
# full covariance matrix, use the diagonals for the error
diag = np.diag(cov)
chi2_fft_cov = []
for fft, true in zip(PI_fft, PI_true):
    chi2 = chi_squared(fft, true, diag)

    chi2_fft_cov.append(chi2)

print("from covariance matrix")
print("----Reduced Chi-Squared Results----")
for i in range(len(chi2_fft_cov)):
    print(fr'FFT {i}:', chi2_fft_cov[i]/dof)

# save all chi squared results to text file
with open("results/chi2_results.txt", "w") as f:
    f.write("from basic equation\n")
    f.write("----Reduced Chi-Squared Results----\n")
    f.write(f"2PCP Full: {chi2_2pcp/dof}\n")
    f.write(f"2PCP Truncated: {chi2_2pcp_t/dof_t}\n")
    for i in range(len(rchi2_fft)):
        f.write(f"FFT {i}: {chi2_fft[i]/dof}\n")
    f.write("\n")
    f.write("from covariance matrix\n")
    f.write("----Reduced Chi-Squared Results----\n")
    for i in range(len(chi2_fft_cov)):
        f.write(f"FFT {i}: {chi2_fft_cov[i]/dof}\n")
    