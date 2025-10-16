[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/nqfiwWTG)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=20651122&assignment_repo_type=AssignmentRepo)
# ASTR 513 Mid-Term Project

## Running this project

To install the project, run
    ```bash
    pip install -e .
    ```

## Overview
This project generates mock catalogs and computes the matter power spectrum using two independent methods. The resulting power spectra are then compared to evaluate consistency between approaches. Mock catalogs are generated using **nbodykit**, an open-source Python package for large-scale structure simulations. An installation of nbodykit is required to create the galaxy mocks, with the necessary documentation at [nbodykit installation documentation](https://nbodykit.readthedocs.io/en/latest/getting-started/install.html). Multiple installation options are provided, but this project was developed using a conda environment in which `generate_mock.py` was executed. 

In addition to nbodykit, several other Python dependencies are required. These are listed in `pyproject.toml`. Since we produced our own mock catalogs, no external data downloads are required.

All following steps should be run within the main directory of this project, using 
    ```bash
    python src/<script_name>
    ```

## Workflow
1. Create the mock catalogs by running `generate_mock.py`. This script produces mock catalogs and saves them within the data directory. The ("truth") power spectrum used to produce the mock catalog can be calculated using `calc_pk.py` and is stored in the `data` directory. Each mock catalog should have a matching truth power spectrum saved.

2. Compute Power Spectrum via Hankel Method 
Execute `run_henkel_pk.py`. This method requires [Corrfunc](https://github.com/manodeep/Corrfunc/tree/master) to be installed (documentation found [here](https://app.readthedocs.org/projects/corrfunc/downloads/pdf/docs/)). **Corrfunc** is a high-performance Python/C library designed to efficiently compute pair counts and two-point correlation functions for large-scale structure analyses. The Corrfunc documentation lists several prerequisites for installation and provides multiple installation methods. This method loads the mock catalog and saves the power spectrum information in the `data` directory. **Note:** this method is computationally expensive and can take a significant amount of time. We ran this only once for the entire mock catalog.

3. Compute Power Spectrum via Fourier Transform
Run `fft_pk.py`. This loads the mock catalog and calculates the power spectrum using the squared Fourier modes of the galaxy overdensity field. The power spectrum values and a plot of P(k) vs. k is saved to the data directory. This script must be run for each mock catalog.

4. Compare Results
To compare the power spectra produced by each method, execute `err_analysis.py`. This will print results and save them to a text file. This script only needs to run once for all results. It will write numerical results to a text file and generate visualizations of
- Raw power spectra
- Interpolated power spectra
- Residuals
All plots are saved in the `results` directory in both PDF and PNG formats. Additionally, the $\chi_R^2$ results are saved to the `results` directory in a text file. 

