[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/nqfiwWTG)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=20651122&assignment_repo_type=AssignmentRepo)
# ASTR 513 Mid-Term Project

Welcome to the repository of the **Mid-Term Project** for ASTR 513.
This project is worth **20 points** and gives you the opportunity to
apply the numerical techniques we have covered in the course so far.

You are recommended to work in teams of 3 to 6 students.
To form team, come up with a unique team name and put it in this
[GitHub Classroom link](https://classroom.github.com/a/nqfiwWTG).

## Timeline & Deliverables

* Prsentation dates:
  October 13th or 15th
* Submission deadline:
  By 11:59pm (Arizona tgime) on the day of your presentation
* Submission platform: GitHub Classroom

Your final submission should include:

* Project code (inside the `src/` directory of this git repository)
* Documentation (inside the `doc/` directory)
* Presentation materials (slides or Jupyter notebook, also version
  controlled with this git repository)

Only **one submission per team** is needed.

**Late submissions may not be accepted. Please plan ahead.**

## Project Ideas

The file `doc/ideas.yaml` contains a compilation of topics from
homework set \#1.
Please use this list to help you look for other students with similar
interests and form teams.

Example from a past project:
[Exoplanet Statistics](https://github.com/ua-2024q3-astr513/ASTRSTATS513_final).

## Requirements

### 1. Code

* Submit well-documented, runnable source code.
* Include docstrings and inline comments.
* Update this `README.md` file to explain:
  * How to install and run your project
  * Any dependencies or data required

### 2. Presentation

* Deliver a ~ 15 minute presentation on your project.
* You may use either:
  * Slides (traditional format), or
  * A Jupyter notebook (similar to our class style).

Your presentation should:
* clearly explain the problem you tackled;
* show the numerical techniques you applied;
* present results with relevant plots, tables; or figures
* highlight your findings and insights.

Each team member should be prepared to discuss their contributions.

## Grading (20 points total)

Projects will be graded based on the following criteria:
1. Originality & clarity of the idea
2. Quality of the solution (numerical methods, implementation,
   correctness)
3. Thoroughness of documentation (code comments, docstrings, README)
4. Effectiveness of presentation (clarity, structure, visualizations,
   teamwork)

## Collaboration & GitHub Use

Projects are managed through GitHub Classroom.
* Multiple students can share a single repository.
  You can join by putting your unique team name in this
  [GitHub Classroom link](https://classroom.github.com/a/nqfiwWTG).
* Use GitHub to track progress, manage code, and collaborate.
* Only one final submission per team is needed.

## Final Note

This project is your chance to be creative, apply what you have
learned so far, and work collaboratively on a meaningful computational
astrophysics problem.
We look forward to your results!


## Running this project

To install the project, run
    ```bash
    pip install -e .
    ```

This project includes the generation of mock catalogs of galaxies to use for the calculation of the matter power spectrum. To create the mock catalogs, we use nbodykit, which is a free and openly accessible Python package which creates a mock catalog using a predetermined parameters for the matter powerspectrum. It is necessary to install nbodykit to properly create the galaxy mocks, with the necessary documentation at [nbodykit installation documentation](https://nbodykit.readthedocs.io/en/latest/getting-started/install.html). The documentation includes several methods to download, but we used Anaconda, creating an environment to run generate_mock.py. 

After installing nbodykit, there are a few other necessary dependencies required to run the rest of the project. Since we produced our own mock catalogs, we don't have any other data necessary to download or install. The other dependencies necessary to run the python scripts can be found in pyproject.toml. 

All following steps should be ran within the main directory of this project, using 
    ```bash
    python src/<script_name>
    ```

The first step of the project is to create the mock catalogs by running generate_mock.py. This will produce a mock catalog from which we find the power spectrum. Both the mock catalog and the power spectrum are saved within the data directory. The power spectrum used to produce the mock catalog can be calculated and saved using calc_pk.py. This saves the "truth" power spectra in the data directory. These should be ran individually for each mock catalog produced. 

To run the fourier transform method, run fft_pk.ipynb. This loads the mock catalog and performs the fourier transfrom to produce the power spectrum for each mock catalog. The power spectrum is saved in the data directory. This script must be run for each mock catalog. 

To run the hankel method, run run_henkel_pk.py. This method loads the mock catalog and saves the power spectrum information in the data directory. (This method is computationally expensive and takes a significant amount of time. We ran this only once for the entire mock catalog.) 

Finally, to compare the results of the different power spectra, run err_analysis.py. This will print results and save them to a text file. This script only needs to run once for all results. 

