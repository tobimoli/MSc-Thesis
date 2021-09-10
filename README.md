# Deriving water quality indicators from high-resolution satellite data using spatio-temporal statistics
# Abstract (MSc Thesis TU Delft)
The derivation of water quality indicators is of importance, especially in coastal areas, as most of the economic activities are located here. However, the availability of high-spatial-resolution water quality information in coastal zones is limited. Nowadays, high-resolution satellite data is becoming available and can fill in this knowledge gap. This satellite data contains spectral reflectances, so a model needs to be designed to map these reflectances to water quality indicators. In this thesis, a Gaussian process regression (GPR) method will be introduced and analyzed extensively in terms of covariance functions, hyperparameters and computational costs. Remote sensing data is collected from the Sentinel-2 mission and the in-situ data is obtained from the ODYSSEA programme. The Matérn 3/2 kernel produces the best results and these are compared with the current models that rely on machine learning techniques. GPR shows promising results in terms of estimation accuracy and chlorophyll-a maps are made for different areas and depths. Various approximation methods are tested to speed up the computation time. Singular value decomposition shows promising results for doing predictions to reduce the computation time. Moreover, GPR can handle limited availability of in-situ data well and uncertainty quantification is induced by the Bayesian framework.

# Documentation scripts.
## CMEMS.py
From the CMEMS platform, a dataset has been downloaded and the CHL-a concentration is plotted with this script.

## EMODNet.py
The in-situ dataset obtained from the EMODNET platform is imported here for analysis of useability. 

## Function_file.py
Contains all function and classes used throughout this thesis.

## GPR.py
An example for GPR can be computed for a single input variable and output variable. Matrices and submatrices are computed manually. 

## GPR_complete.py
Using the package GPy, the posterior mean (estimate) and standard deviation (uncertainty) is computed for a specific area within the satellite data. It is possible to do this for large areas, using an iterative method. Also a manual (without GPy) method is included for the comparison with the approximation technique SVD.

## GPR_CV.py
Cross validation method is used to compute the MSE and other metrics for using GPR.

## GPR_EP.py
Expectation propagation method to speed up the inversion of the matrix. Done for a 1D toy-problem.

## GPR_kernel.py
Comparison for using different kernels within the GPy package. Kernels included are: squared-exponential, matérn kernels, neural network, linear.

## GPR_parameter.py
Analysis of hyperparameters within the GPR model. The following plots can be created: contour plots of the log marginal likelihood, the log marginal likelihood versus one hyperparameter, continuous estimations.

## GPR_sparse.py
The GPy.models.SparseGPRegression module has been used to compute sparse GPR estimations.

## GPR_sparse_example.py
Various techniques have been applied to reduce the computational complexity for 1D toy-problem. Cholesky decomposition, setting the covariance to zero when the covariance is under a certain threshold, SVD and the Nyström method.

## GPR_SVD.py
Singular value decomposition plots and analysis for the chlorophyll-a problem. 

## Interpolation.py
Analysis for various interpolation techniques for the chlorophyll-a problem as well as for toy-problems.

## Kernel.py
Plots of the kernel functions can be created here. Different plots have been created using the chlorophyll-a problem as well.

## LandDetection.py
This script shows the process of land detection. Using the NIR band, land-water can be classified and is compared with the classification maps from ESA.

## LinReg.py
Linear regression techniques have been applied to estimate the CHL-a concentration. Differences for shallow observations are shown.

## ODYSSEA.py
A first glance at the data from the ODYSSEA project. Scatterplots are made.

## ODYSSEA-depth.py
The importance of the depth variable is shown here.

## Olympiada.py
Overview of the Olympiada dataset.

## PANGAEA.py
Overview of the PANGAEA dataset.

## Sentinel-C2RCC.py
Analysis of the C2RCC algorithm which implemented in the SNAP software programme. Using SNAP the derivation of the 'iop_apig' is done and imported in this script. Using non-linear least squares, the parameters are estimated and estimations are plotted.
