# Differentiable multi-ridge regression for system identification

This github repository contains the Python code to reproduce the results of the paper Differentiable multi-ridge regression for system identification by Gabriele Maroni, Loris Cannelli and Dario Piga.

This paper has been accepted for publication for the *20th IFAC Symposium on System Identification (SYSID2024)*.

![License Badge](https://img.shields.io/badge/license-MIT-blue)

## Abstract
Regularization aims to shrink model parameters, reducing complexity and overfitting risk. Traditional methods like LASSO and Ridge regression, limited by a single regularization hyperparameter, can restrict bias-variance trade-off adaptability. This paper addresses sys-tem identification in a multi-ridge regression framework, where an l2-penalty on the model coefficients is introduced, and a different regularization hyperparameter is assigned to each
model parameter. To compute the optimal hyperparameters, a cross-validation-based criterion is optimized through gradient descent. Autoregressive and Output Error models are considered.
The former requires formulating a regularized least-squares problem. The identification of the latter class is more challenging and is addressed by adopting regularized instrumental variable methods to ensure a consistent parameter estimation.

## Software implementation
All the source code used to generate the results and figures in the paper are in the `src` and `notebooks` folders. Computations and figure generation are all run inside [Jupyter notebooks](http://jupyter.org/). Results generated by the code are saved in `results` folder.

## Getting the code
You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/gabribg88/Multiridge-SYSID.git

or [download a zip archive](https://github.com/gabribg88/Multiridge-SYSID/archive/refs/heads/master.zip).

## Requirements
You'll need a working Python environment to run the code.
The recommended way to set up your environment is through the
[Anaconda Python distribution](https://www.anaconda.com/download/) which
provides the `conda` package manager.
Anaconda can be installed in your user directory and does not interfere with
the system Python installation.
The required dependencies are specified in the file `requirements.txt`.

We recommend to use `conda` virtual environments to manage the project dependencies in
isolation.
Thus, you can install the dependencies without causing conflicts, with your
setup (even with different Python versions), with the pip package-management system.

Run the following command in the repository folder (where `requirements.txt`
is located) to create a separate environment and install all required
dependencies in it:

    conda create --name <env_name>
    source activate <env_name>
    pip install -r requirements.txt
