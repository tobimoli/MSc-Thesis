# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 11:06:43 2021

@author: molenaar
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from Function_file import compute_cov_matrices, compute_gpr_parameters

sns.set_palette(palette='deep')

plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['figure.dpi'] = 100

#%%

# Define dimension.
d = 1
# Number of samples (training set). 
n = 2

x = np.linspace(start=-.5, stop=.5, num=n)

def f(x):
    f = np.sin((4*np.pi)*x) + np.sin((7*np.pi)*x)
    return(f)

f_x = f(x)

# fig, ax = plt.subplots()
# # Plot function f. 
# sns.lineplot(x=x, y=f_x, color='red', label = 'f(x)', ax=ax)
# # Plot function components.
# sns.lineplot(x=x, y=np.sin((4*np.pi)*x), color='orange', label='$\sin(4 \pi x)$', alpha=0.3, ax=ax)
# sns.lineplot(x=x, y=np.sin((7*np.pi)*x), color='purple', label='$\sin(7 \pi x)$', alpha=0.3, ax=ax)
# ax.legend(loc='upper right')
# ax.set_title(r'Graph of $f(x) = \sin(4\pi x) + \sin(7\pi x)$')

#%%

# Error standard deviation. 
sigma_n = 0
# Errors.
epsilon = np.random.normal(loc=0, scale=sigma_n, size=n)
# Observed target variable. 
y = f_x + epsilon

#%%

# fig, ax = plt.subplots()
# # Plot errors. 
# sns.distplot(epsilon, ax=ax)
# ax.set(title='Error Distribution')

#%%

# fig, ax = plt.subplots()
# # Plot training data.
# sns.scatterplot(x=x, y=y, label='training data', ax=ax);
# # Plot "true" linear fit.
# sns.lineplot(x=x, y=f_x, color='red', label='f(x)', ax=ax);

# ax.set(title='Sample Data')
# ax.legend(loc='upper right')

#%%

n_star = 100

x_star = np.linspace(start=-1, stop=1, num=n_star)
       
l = 0.1
sigma_f = 2

K, K_star2, K_star = compute_cov_matrices(x, x_star, sigma_f=sigma_f, l=l)

#%%

# fig, ax = plt.subplots(figsize=(10, 8))
# sns.heatmap(data=K, cmap='Blues', ax=ax)
# ax.set(title='Components of the Kernel Matrix K')

#%%

# fig, ax = plt.subplots(figsize=(10, 8))
# sns.heatmap(data=K_star2, cmap='Blues', ax=ax)
# ax.set(title='Components of the Kernel Matrix K_star2');

#%%

# fig, ax = plt.subplots(figsize=(10, 8))
# sns.heatmap(data=K_star, cmap='Blues', ax=ax)
# ax.set(title='Components of the Kernel Matrix K_star');

#%%

a = np.concatenate((K + (sigma_n**2)*np.eye(n), K_star), axis=0)
b = np.concatenate((K_star.T, K_star2), axis=0)
C = np.concatenate((a, b), axis=1)
np.all(C.T == C)

# fig, ax = plt.subplots(figsize=(10, 8))
# sns.heatmap(data=C, cmap='Blues', ax=ax)
# ax.set(title='Components of the Covariance Matrix C');

#%%

fig, ax = plt.subplots()

for i in range(0, 5):
    # Sample from prior distribution. 
    z_star = np.random.multivariate_normal(mean=np.zeros(n_star), cov=K_star2)
    # Plot function.
    ax.plot(x_star, z_star, color='blue')
    
# Plot "true" linear fit.
#sns.lineplot(x=x, y=f_x, color='red', label='f(x)', ax=ax)
ax.set(title='Samples of Prior Distribution')
#ax.legend(loc='lower right');

#%%

# Compute posterior mean and covariance. 
f_bar_star, cov_f_star = compute_gpr_parameters(K, K_star2, K_star, sigma_n, y)

#%%

# fig, ax = plt.subplots(figsize=(10, 8))
# sns.heatmap(data=cov_f_star, cmap='Blues', ax=ax)
# ax.set_title('Components of the Covariance Matrix cov_f_star');

#%%

fig, ax = plt.subplots()
ax.scatter(x=x, y=f_x, color='red', label = 'observations')

for i in range(0, 5):
    # Sample from posterior distribution. 
    z_star = np.random.multivariate_normal(mean=f_bar_star.squeeze(), cov=cov_f_star)
    # Plot function.
    ax.plot(x_star, z_star, color="blue");
    
# Plot "true" linear fit.
ax.scatter(x=x, y=f_x, color='red', label = 'observations')
ax.set(title='Samples of Posterior Distribution')
ax.legend(loc='upper right');

