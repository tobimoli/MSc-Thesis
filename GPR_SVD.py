# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 13:07:05 2021

@author: molenaar
"""

#%% load packages

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from Function_file import compute_cov_matrices, compute_gpr_parameters, marglike, marglike_svd, plot
from Function_file import marglike_nystrom, compute_gpr_parameters_svd, compute_gpr_parameters_nystrom
plt.rcParams.update({'font.size': 18})
log2pi = np.log(2*np.pi)
folder = 'C:/Users/molenaar/OneDrive - Stichting Deltares/Documents/Thesis - Deltares/Images/Animation_SVD'

#%% Initial values and matrices
np.random.seed(101)

#Define dimension.
d = 1
# Number of samples (training set).
n = 1000
X = np.linspace(start=0, stop=100, num=n).reshape(-1,1)
# Number of predictions
n_star = 2000
X_star = np.linspace(start=-20, stop=120, num=n_star).reshape(-1,1)

# Parameters
l = 4
sigma_f = 1
sigma_n = 1

K, K_star2, K_star = compute_cov_matrices(X, X_star, sigma_f=sigma_f, l=l)

Y = np.random.multivariate_normal(np.zeros(n), K+(sigma_n**2)*np.eye(n)).reshape(-1,1)

#%% Optimize parameters

b1= [10**-1, 10**2]
b2= [10**-3, 10**1]
b3= [10**-3, 10**1]
bnd = [b1,b2,b3] #bounds used for "minimize" function
start = [1,1,1] #initial hyperparameters values l, sigma_f, sigma_n

re = minimize(fun=marglike, x0=start, args=(X,Y), method="Nelder-Mead",
              options = {'disp':True},bounds=bnd)
l_opt, sigma_f_opt, sigma_n_opt = re.x
print(l_opt, sigma_f_opt, sigma_n_opt)

#l_opt, sigma_f_opt, sigma_n_opt = gpr.rbf.lengthscale[0], np.sqrt(gpr.rbf.variance[0]), np.sqrt(gpr.Gaussian_noise.variance[0])

#%% Predictions

K, K_star2, K_star = compute_cov_matrices(X, X_star, sigma_f=sigma_f_opt, l=l_opt)

# Compute posterior mean and covariance.
mean, var = compute_gpr_parameters(K, K_star2, K_star, sigma_n_opt, Y)
var += sigma_n_opt**2

for k in range(1,20):
    mean_svd, var_svd = compute_gpr_parameters_svd(K, K_star2, K_star, sigma_n_opt, k, Y)
    var_svd += sigma_n_opt**2
    plot(X_star, mean, var, mean_svd, var_svd, k, folder)
#%% GPy
import GPy

rbf = GPy.kern.RBF(input_dim = 1, variance=1, lengthscale=1, ARD = False)
gpr = GPy.models.GPRegression(X, Y, rbf, normalizer=True)
gpr.Gaussian_noise.variance = sigma_n**2

_ = gpr.optimize(messages = True)

print(gpr.rbf.lengthscale[0])
print(np.sqrt(gpr.rbf.variance[0]))
print(np.sqrt(gpr.Gaussian_noise.variance[0]))

prediction_gpy = gpr.predict(X_star,full_cov= False, include_likelihood= True)

#%% Figure of predictions Manually vs GPy

plt.figure(figsize = (12,8))
plt.plot(X_star, mean, 'g', label = 'Manually')
plt.plot(X_star, mean + 2* np.sqrt(var), 'g--')
plt.plot(X_star, mean - 2* np.sqrt(var), 'g--')
plt.grid(False)

plt.scatter(X, Y, color = 'k', marker = 'x', alpha = 0.5, label = 'Observations')
plt.xlabel('Input variable x')
plt.ylabel('Output variable y')
plt.plot(X_star, prediction_gpy[0], 'r', label = "GPy")
plt.plot(X_star, prediction_gpy[0] + 2* np.sqrt(prediction_gpy[1]), 'r--')
plt.plot(X_star, prediction_gpy[0] - 2* np.sqrt(prediction_gpy[1]), 'r--')
plt.legend()
plt.show()

#%% Figure of predictions Exact vs SVD

plt.figure(figsize = (12,8))
plt.plot(X_star, mean, 'g', label = 'Exact')
plt.plot(X_star, mean + 2* np.sqrt(var), 'g--')
plt.plot(X_star, mean - 2* np.sqrt(var), 'g--')
plt.grid(False)

plt.scatter(X, Y, color = 'k', marker = 'x', alpha = 0.5, label = 'Observations')
plt.xlabel('Input variable x')
plt.ylabel('Output variable y')
plt.plot(X_star, mean_svd, 'r', label = "SVD")
plt.plot(X_star, mean_svd + 2* np.sqrt(var_svd), 'r--')
plt.plot(X_star, mean_svd - 2* np.sqrt(var_svd), 'r--')
plt.legend()
plt.title(f'k={k}')
plt.show()

#%% Figure of true prediction versus SVD prediction (parameters are optimized exact)

fig, ax = plt.subplots(2,2,figsize=(8,8), sharex = True, sharey=True)
fig.add_subplot(111, frameon=False)

K, K_star2, K_star = compute_cov_matrices(X, X_star, sigma_f=sigma_f_opt, l=l_opt)
mean, var = compute_gpr_parameters(K, K_star2, K_star, sigma_n_opt, Y)
var += sigma_n_opt**2

k = 1
for row in range(2):
    for col in range(2):
        mean_svd, var_svd = compute_gpr_parameters_svd(K, K_star2, K_star, sigma_n_opt, k, Y)
        var_svd += sigma_n_opt**2
        ax[row,col].scatter(X, Y, color = 'k', marker = 'x', alpha = 0.1, label = 'Observations')
        ax[row,col].plot(X_star, mean_svd, 'r', label = "SVD")
        ax[row,col].plot(X_star, mean_svd + 2* np.sqrt(var_svd), 'r--')
        ax[row,col].plot(X_star, mean_svd - 2* np.sqrt(var_svd), 'r--')
        ax[row,col].plot(X_star, mean, 'g', label = 'Exact')
        ax[row,col].plot(X_star, mean + 2* np.sqrt(var), 'g--')
        ax[row,col].plot(X_star, mean - 2* np.sqrt(var), 'g--')
        ax[row,col].set_title(f'Rank = {k}')
        k += 3

#for axs in fig.get_axes():
#    axs.label_outer()
plt.tick_params(labelcolor = 'none', which = 'both', top = False, bottom = False, left = False, right = False)
plt.xlabel('Input variable x')
plt.ylabel('Output variable y')
plt.show()

#%% Figure of true prediction versus SVD prediction (parameters are optimized SVD)

fig, ax = plt.subplots(2,2,figsize=(8,8), sharex = True, sharey=True)
fig.add_subplot(111, frameon=False)

K, K_star2, K_star = compute_cov_matrices(X, X_star, sigma_f=sigma_f_opt, l=l_opt)
mean, var = compute_gpr_parameters(K, K_star2, K_star, sigma_n_opt, Y)
var += sigma_n_opt**2

k = 1
for row in range(2):
    for col in range(2):
        re_svd = minimize(fun=marglike_svd, x0=start, args=(X,Y,k), method="Nelder-Mead",
                      options = {'maxiter': 100}, bounds=bnd)
        print(re_svd.x)
        l_svd, sigma_f_svd, sigma_n_svd = re_svd.x
        K_svd, K_star2_svd, K_star_svd = compute_cov_matrices(X, X_star, sigma_f=sigma_f_svd, l=l_svd)
        mean_svd, var_svd = compute_gpr_parameters_svd(K_svd, K_star2_svd, K_star_svd, sigma_n_svd, k, Y)
        var_svd += sigma_n_svd**2
        
        ax[row,col].scatter(X, Y, color = 'k', marker = 'x', alpha = 0.1, label = 'Observations')
        ax[row,col].plot(X_star, mean_svd, 'r', label = "SVD")
        ax[row,col].plot(X_star, mean_svd + 2* np.sqrt(var_svd), 'r--')
        ax[row,col].plot(X_star, mean_svd - 2* np.sqrt(var_svd), 'r--')
        ax[row,col].plot(X_star, mean, 'g', label = 'Exact')
        ax[row,col].plot(X_star, mean + 2* np.sqrt(var), 'g--')
        ax[row,col].plot(X_star, mean - 2* np.sqrt(var), 'g--')
        ax[row,col].set_title(f'Rank = {k}')
        k += 3

#for axs in fig.get_axes():
#    axs.label_outer()
plt.tick_params(labelcolor = 'none', which = 'both', top = False, bottom = False, left = False, right = False)
plt.xlabel('Input variable x')
plt.ylabel('Output variable y')
plt.show()

#%% Figure of true prediction versus SVD prediction (parameters are optimized SVD)

K, K_star2, K_star = compute_cov_matrices(X, X_star, sigma_f=sigma_f_opt, l=l_opt)

# Compute posterior mean and covariance.
mean, var = compute_gpr_parameters(K, K_star2, K_star, sigma_n_opt, Y)
var += sigma_n_opt**2

for k in range(1,20):
    re_svd = minimize(fun=marglike_svd, x0=start, args=(X,Y,k), method="Nelder-Mead",
                      options = {'maxiter': 100}, bounds=bnd)
    print(re_svd.x)
    l_svd, sigma_f_svd, sigma_n_svd = re_svd.x
    K_svd, K_star2_svd, K_star_svd = compute_cov_matrices(X, X_star, sigma_f=sigma_f_svd, l=l_svd)
    mean_svd, var_svd = compute_gpr_parameters_svd(K_svd, K_star2_svd, K_star_svd, sigma_n_svd, k, Y)
    var_svd += sigma_n_svd**2
    plot(X_star, mean, var, mean_svd, var_svd, k, folder)

#%% Animation

import imageio

filenames = ['fig_'+str(i)+'.png' for i in range(1,20)]
images = []
for filename in filenames:
    images.append(imageio.imread(folder + '/' + filename))

imageio.mimsave(folder + '/movie.gif', images, duration = 0.5)
