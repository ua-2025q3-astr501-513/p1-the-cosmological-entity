# Project Description / Methodology

This project computes the power spectrum of a mock galaxy catalog through two ways: Fast-Fourier Transform and Hankel Transform of the two-point correlation function. The workflow is as follows:

1. ## **Mock Catalog Generation** ##
The mock catalog is generated using the Python package [nbodykit](https://nbodykit.readthedocs.io/en/latest/) at redshift z=0.55 with a galaxy bias b=2 in a box of side length 1380 Mpc/h. The resulting catalog contains approximately 8 million galaxies. Log-normal mocks assume that the matter density field follows a log-normal distribution, ensuring positive densities while mimicking mildly non-linear galaxy clustering. Galaxies are then sampled from this distribution following an input power spectrum, which serves as the "true" power spectrum for comparison with the estimated P(k).
   - The mock catalog contains galaxy positions and optional weights in a `.npz` file within the `data` directory.  

Notes:
* The catalog uses a Cloud-in-Cell (CIC) interpolation for mesh assignment.
* Power spectrum P(k, μ) is computed for 5 μ bins; however, for this analysis, we effectively ignore the μ bins by combining them into a single μ bin.


### **Method 1: Hankel Transform** ###
1. **Two-Point Correlation Function ξ(r)**  
   - The `CorrfuncCalculator` class computes pair counts (DD, DR, RR) using the [Corrfunc](https://github.com/manodeep/Corrfunc) library.  
   - The correlation function is computed via the Landy-Szalay estimator $\xi(r) = \frac{DD - 2DR + RR}{RR}$
   - A random catalog is generated for normalization.  
   - ξ(r) is saved to a text file and plotted for visualization.

2. **Power Spectrum P(k) via Hankel Transform**  
   - The `HenkelTransform` class computes P(k) from ξ(r) using a numerical Hankel transform.  
   - A tapering window $W(r) = \exp[-(r/300)^4]$ is applied to reduce artifacts from the finite range of ξ(r).  
   - This method is an approximation because ξ(r) is discretely sampled and only available over a finite range.  
   - The resulting P(k) and associated errors are saved to a file and plotted.

3. **Outputs**  
   - `data/xi_<sample_frac>.txt` — two-point correlation function data  
   - `data/xi_plot_<sample_frac>.png` — plot of ξ(r)  
   - `data/pk_from_xi_<sample_frac>.txt` — power spectrum data  
   - `data/Pk_from_xi_<sample_frac>_errors.png` — plot of P(k) with errors

This method allows for power spectrum estimation on large-scales. It is computationally expensive and has large runtimes for large catalogs.

### **Method  2: Fourier Transform** ###
1. **Fourier Transform of Overdensity Field**
   - The mock catalog is assigned to a 3D grid using the CIC mass assignment scheme
   - The fractional overdensity is found at each grid point
   - We calculate the Fourier Transform of the fractional overdensity field $\tilde{\delta}(k)$ using `numpy.fft.fftn`

2. **Power Spectrum from Fourier Transform**
   - P(k) is given by the squared Fourier amplitudes 
   - P(k) is binned in spherical shells and the average power within each k shell is saved to a file
   - A log-log plot of the binned P(k) vs. k is plotted and saved

3. **Outputs**
   - `data/power_spec_npfft.txt` - power spectrum data
   - `data/pk_fft.png` - plot of P(k)

This method is much less computationally expensive and simple to implement. However, for a real survey with incomplete sky coverage, this method is less accurate than the Hankel transform.