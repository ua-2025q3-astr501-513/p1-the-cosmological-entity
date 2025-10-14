import numpy as np
import matplotlib.pyplot as plt
import time

import os
import ctypes
import sys

# Add GSL library path
# May need to include and edit path in commented lines below if Corrfunc had trouble finding the gsl library. 
#gsl_path = "/usr/lib/x86_64-linux-gnu/libgsl.so.27.0.0"

#ctypes.CDLL(gsl_path, mode=ctypes.RTLD_GLOBAL)

from Corrfunc.theory.DD import DD
from Corrfunc.utils import convert_3d_counts_to_cf


class CorrfuncCalculator:
    def __init__(self, datafile, sample_frac=0.01, boxsize=1380.0, nbins=20, rmin=0.1, rmax=300, nthreads=8, seed=42):
        """
        Compute the two-point correlation function using Corrfunc.

        Parameters
        ----------
        datafile : str
            Path to .npz mock catalog (expects 'pos' and 'weight' arrays)
        sample_frac : float
            Fraction of data points to randomly sample (0 < sample_frac <= 1)
        boxsize : float
            Box size in Mpc/h
        nbins : int
            Number of logarithmic separation bins
        rmin, rmax : float
            Minimum and maximum separations in Mpc/h
        nthreads : int
            Number of CPU threads for Corrfunc
        seed : int
            Random seed for reproducibility
        """
        self.datafile = datafile
        self.sample_frac = sample_frac
        self.boxsize = boxsize
        self.nbins = nbins
        self.rmin = rmin
        self.rmax = rmax
        self.nthreads = nthreads
        self.seed = seed

        # Load data and prepare coordinates
        self._load_data()

        # Define bins
        self.bins = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)

    def _load_data(self):
        np.random.seed(self.seed)
        data = np.load(self.datafile)
        pos = np.float64(data['pos'])
        N_total = len(pos)
        N_sample = int(N_total * self.sample_frac)
        print(f"Sampling {N_sample}/{N_total} points ({self.sample_frac*100:.1f}%)")

        idx = np.random.choice(N_total, size=N_sample, replace=False)
        self.x, self.y, self.z = pos[idx, 0], pos[idx, 1], pos[idx, 2]
        self.N = N_sample

        # Generate random catalog (3x number of data points)
        self.rand_N = 3 * self.N
        self.rand_X = np.random.uniform(0, self.boxsize, self.rand_N)
        self.rand_Y = np.random.uniform(0, self.boxsize, self.rand_N)
        self.rand_Z = np.random.uniform(0, self.boxsize, self.rand_N)

    def compute_xi(self):
        """Compute DD, DR, RR, and correlation function ξ(r)."""
        print("Computing DD, DR, RR with Corrfunc...")

        # DD
        start = time.time()
        DD_counts = DD(autocorr=1, nthreads=self.nthreads, binfile=self.bins,
                       X1=self.x, Y1=self.y, Z1=self.z, boxsize=self.boxsize)
        print(f"DD done in {time.time() - start:.2f} s")

        # DR
        start = time.time()
        DR_counts = DD(autocorr=0, nthreads=self.nthreads, binfile=self.bins,
                       X1=self.x, Y1=self.y, Z1=self.z,
                       X2=self.rand_X, Y2=self.rand_Y, Z2=self.rand_Z, boxsize=self.boxsize)
        print(f"DR done in {time.time() - start:.2f} s")

        # RR
        start = time.time()
        RR_counts = DD(autocorr=1, nthreads=self.nthreads, binfile=self.bins,
                       X1=self.rand_X, Y1=self.rand_Y, Z1=self.rand_Z, boxsize=self.boxsize)
        print(f"RR done in {time.time() - start:.2f} s")

        # Correlation function
        self.cf = convert_3d_counts_to_cf(self.N, self.N, self.rand_N, self.rand_N,
                                          DD_counts, DR_counts, DR_counts, RR_counts)
        self.r_centers = np.sqrt(self.bins[:-1] * self.bins[1:])

        dd = np.maximum(DD_counts['npairs'], 1.0)

        #simple Poisson estimate
        self.cf_err = (1.0 + self.cf) / np.sqrt(dd)

        #if any NaNs or infinities, replace with large finite numbers
        self.cf_err = np.nan_to_num(self.cf_err, nan=1e6, posinf=1e6, neginf=1e6)

        #return self.r_centers, self.cf, self.cf_err
        return self.r_centers, self.cf, self.cf_err

    def save_xi(self, filename=None):
        if filename is None:
            filename = f"data/xi_{self.sample_frac}.txt"
        np.savetxt(filename, np.column_stack([self.r_centers, self.cf, self.cf_err]),
                header="r [Mpc/h]   xi(r)     xi_err(r)")
        print(f"Saved results to {filename}")

    def plot_xi(self, outfile=None):
        if outfile is None:
            outfile = f"data/xi_plot_{self.sample_frac}.png"
        plt.figure(figsize=(6, 4))
        #plt.loglog(self.r_centers, self.cf, marker='o', linestyle='-', color='b')
        plt.errorbar(self.r_centers, self.cf, yerr=self.cf_err, fmt='o-', capsize=4, color='b', label='P(k)')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r"$r\,[{\rm Mpc}/h]$")
        plt.ylabel(r"$\xi(r)$")
        plt.xlim(self.rmin, self.rmax)
        plt.grid(True, which='both', ls='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(outfile, dpi=300)
        plt.show()
        print(f"Saved plot to {outfile}")


class HenkelTransform:
    def __init__(self, datafile, sample_frac=0.01, boxsize=1380.0, nbar=3e-3):
        """
        Compute the galaxy power spectrum using the Henkel Transformation

        Parameters
        ----------
        datafile : str
            Path to 2pt correlation file

        """
        self.datafile = datafile
        self.boxsize = boxsize
        self.nbar = nbar

        # Load data and prepare coordinates
        r, xi, xi_err = np.loadtxt(datafile, unpack=True)
        k = np.logspace(-2, 1, 200)

        self.r = r
        self.xi = xi
        self.k = k
        self.sample_frac = sample_frac

    def compute_pk(self):
        """Compute power spectrum"""
        print("Computing power spectrum from Henkel Transformation")

        Pk = np.zeros_like(self.k)

        W = np.exp(- (self.r / 300)**4) #300 Mpc/h is rmax
        xi_tapered = self.xi * W

        for i, kk in enumerate(self.k):
            integrand = self.r**2 * xi_tapered * np.sinc(kk * self.r / np.pi)
            Pk[i] = 4 * np.pi * np.trapz(integrand, self.r)

        self.Pk = Pk

        V = self.boxsize**3
        dk = np.gradient(self.k)
        Nm = V * 4 * np.pi * self.k**2 * dk / (2 * np.pi)**3
        sigma_Pk = np.sqrt(2 / Nm) * (Pk + 1.0 / self.nbar)

        self.sigma_Pk = sigma_Pk


        return self.k, self.Pk, self.sigma_Pk

    def save_pk(self, filename=None):
        if filename is None:
            filename = f"data/pk_from_xi_{self.sample_frac}.txt"
        np.savetxt(filename, np.column_stack([self.k, self.Pk, self.sigma_Pk]), header="k [h/Mpc]   P(k) [(Mpc/h)^3]   P(k) error [(Mpc/h)^3]")
        print(f"Saved results to {filename}")

    def plot_pk(self, outfile=None):
        if outfile is None:
            outfile = f"data/Pk_from_xi_{self.sample_frac}_errors.png"
        plt.figure(figsize=(6,4))
        plt.loglog(self.k, self.Pk, color = 'b', lw=1)
        plt.fill_between(self.k, self.Pk - self.sigma_Pk, self.Pk + self.sigma_Pk, color='b', alpha=0.3)
        #plt.errorbar(self.k, self.Pk, yerr=np.abs(self.sigma_Pk), capsize=5, markersize=3, color='b', label='P(k)')
        #plt.xscale('log')
        #plt.yscale('log')
        plt.xlabel(r"$k\,[h/{\rm Mpc}]$")
        plt.ylabel(r"$P(k)\,[(Mpc/h)^3]$")
        plt.grid(True, which='both', ls='--', alpha=0.5)
        plt.savefig(outfile, bbox_inches="tight")
        plt.show()  
        print(f"Saved plot to {outfile}")
        
