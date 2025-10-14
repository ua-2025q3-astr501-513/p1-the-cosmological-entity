# Example Usage

This example demonstrates how to compute the power spectrum using the FFT- and Hankel- based methods from a mock catalog using the provided scripts and classes.

## Generating Mock Catalogs ##

To generate a mock catalog, the user can run `generate_mock.py`. This is a script that utilizes **nbodykit** to calculate the lognormal density field from an input power spectrum. The code then samples from this density field to obtain a mock catalog of positions, velocities, and weights. There are multiple parameters that can be changed in the script:

1. redshift
2. cosmological parameters used (cosmo)
3. linear galaxy bias parameter (b1)
4. galaxy number density (nbar)
5. mock catalog boxsize (boxsize)
6. number of grid cells per dimension used when constructing a 3D mesh of the density field in the box (Nmesh)
7. seed to randomly sample the density field to generate galaxy positions (seed)

To execute the script, run

```bash
python generate_mock.py
```

in a terminal within the `src` directory.

## Computing P(k) using the Hankel Transform ##

**A simple script to implemment this method can be found in `run_henkel_pk.py`** To execute this script, run

```bash
python run_henkel_pk.py
```

in a terminal within the `src` directory.

To compute the power spectrum using the Hankel Transform method, the user can run `run_henkel_pk.py`. This is a script that implemments the `CorrfuncCalculator` class to compute the two-point correlation function and the  `HenkelTransform` to perform the Hankel Transformation of the correlation function results to get P(k). 

### CorrfuncCalculator ###
`CorrfuncCalculator` is initialized with the path to the mock catalog (data/mock_catalog.npz) and a fraction of the points to sample (sample_frac=0.01). The user can change the sample fraction, which defines the amount of data used (ex: sample_frac=0.01 means 1% of the data is used). When sample_frac < 1, the code randomly samples the catalog for points to include in the correlation function calculation. There are multiple parameters that can be changed:

1. file path to mock catalog (datafile)
2. fraction of data used (sample_frac)
3. mock catalog boxsize (boxsize)
4. number of separation bins (nbins)
5. minimum separation [Mpc/h] (rmin)
6. maximum separation [Mpc/h] (rmax)
7. number of threads for parallel computing (nthreads)
8. random number seed for downsampling mock catalog to amount defined by sample_frac (seed)

**Note:** Using a smaller sample_frac reduces computation time but decreases statistical accuracy. To run the full data set (sample_frac=1) prepare for >2 hour computation time. Computation time depends on the capabilities of the machine users run the scripts on.

Running this class will create a plot of the two-point correlation function and save the information to a .txt file within the `data` folder.

### Hankel Transform ###
`HenkelTransform` is initialized with the ξ(r) file generated from the `CorrfuncCalculator` class previously, along with box size and galaxy number density (nbar) initially used to generate the mok catalogs. The `datafile` input should be the two-point correlation file output by running the `CorrfuncCalculator` class. The `sample_frac` parameter should match the value used in the ξ(r) calculation. The `boxsize` and `nbar` parameters should match the values used for the initial mock catalog generation. 

Running this class will create a plot the power spectrum and save the power spectrum values to a .txt file within the `data` folder.

## Computing P(k) using the Fourier Transform ##

## Comparing the Results ##

