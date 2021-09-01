# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 15:53:49 2021

@author: molenaar
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature

sns.set_style(
    style='darkgrid', 
    rc={'axes.facecolor': '.9', 'grid.color': '.8'}
)
sns.set_palette(palette='deep')

plt.rcParams['figure.figsize'] = [12, 6]
plt.rcParams['figure.dpi'] = 100

#--------------------------------------------

# Import in-situ data
FOLDER = 'P:/11202428-hisea-inter/Tobias_Molenaar/01-Data/In-Situ/'
NAME = 'ODYSSEA.csv'
data = pd.read_table(FOLDER+NAME, usecols = [2,4,5,6,8], sep = ';')
data.columns = ['time', 'lon', 'lat', 'dep', 'chl']
data['year'] = [int(n[6:10]) for n in data['time']]
data['date'] = [n[6:10]+n[2:5] + '-' + n[:2] for n in data['time']]
data['date'] = data['date'].astype("datetime64")

data = data[data['date'] == '2019-08-02']

NAME = 'ODYSSEA_unique.csv'
data = pd.read_table(FOLDER+NAME, sep = ',')

data['date'] = data['date'].astype("datetime64")

data = data[data['date'] == '2019-08-02']
#remove outliers
data = data[data['chl']>0]

n = 72
x = np.array(data[['lat','lon']][:n])
y = np.array(data['chl'][:n])

fig, ax = plt.subplots()
# Plot function f. 
sns.scatterplot(x=x[:,1], y=x[:,0], color='red', label = 'f(x)', ax=ax)
# Plot function components.
ax.set_title(r'Scatterplot of longitude versus chl-a concentration')

#--------------------------------------------

nx_star = 10
ny_star = 10

x_star = np.array(np.meshgrid(np.linspace(start = 25.06, stop = 25.23, num = nx_star), 
                              np.linspace(start = 40.32, stop = 40.365, num = ny_star))).T.reshape(-1,2)

def kernel_function(x, y, sigma_f=1, l=1):
    """Define squared exponential kernel function."""
    kernel = sigma_f * np.exp(- (np.linalg.norm(x - y)**2) / (2 * l**2))
    return kernel
        
l = 0.05
sigma_f = 1

import itertools

def compute_cov_matrices(x, x_star, sigma_f=1, l=1):
    """
    Compute components of the covariance matrix of the joint distribution.
    
    We follow the notation:
    
        - K = K(X, X) 
        - K_star = K(X_*, X)
        - K_star2 = K(X_*, X_*)
    """
    n = x.shape[0]
    n_star = x_star.shape[0]

    K = [kernel_function(i, j, sigma_f=sigma_f, l=l) for (i, j) in itertools.product(x, x)]

    K = np.array(K).reshape(n, n)
    
    K_star2 = [kernel_function(i, j, sigma_f=sigma_f, l=l) for (i, j) in itertools.product(x_star, x_star)]

    K_star2 = np.array(K_star2).reshape(n_star, n_star)
    
    K_star = [kernel_function(i, j, sigma_f=sigma_f, l=l) for (i, j) in itertools.product(x_star, x)]

    K_star = np.array(K_star).reshape(n_star, n)
    
    return (K, K_star2, K_star)

K, K_star2, K_star = compute_cov_matrices(x, x_star, sigma_f=sigma_f, l=l)

#--------------------------------------------

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data=K, cmap='Blues', ax=ax)
ax.set(title='Components of the Kernel Matrix K')

#--------------------------------------------

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data=K_star2, cmap='Blues', ax=ax)
ax.set(title='Components of the Kernel Matrix K_star2');

#--------------------------------------------

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data=K_star, cmap='Blues', ax=ax)
ax.set(title='Components of the Kernel Matrix K_star');

#--------------------------------------------
sigma_n = 1;

a = np.concatenate((K + (sigma_n**2)*np.eye(n), K_star), axis=0)
b = np.concatenate((K_star.T, K_star2), axis=0)
C = np.concatenate((a, b), axis=1)
np.all(C.T == C)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data=C, cmap='Blues', ax=ax)
ax.set(title='Components of the Covariance Matrix C');

#--------------------------------------------

fig, ax = plt.subplots()

for i in range(0, 100):
    # Sample from prior distribution. 
    z_star = np.random.multivariate_normal(mean=np.zeros(nx_star*ny_star), cov=K_star2)
    # Plot function.
    sns.lineplot(x=x_star[:,1], y=z_star, color='blue', alpha=0.2, ax=ax)
    
# Plot "true" linear fit.
sns.scatterplot(x=x[:,0], y=y, color='red', label = 'f(x)', ax=ax)
ax.set(title='Samples of Prior Distribution')
ax.legend(loc='lower right');

#--------------------------------------------
d= 1

def compute_gpr_parameters(K, K_star2, K_star, sigma_n):
    """Compute gaussian regression parameters."""
    n = K.shape[0]
    # Mean.
    f_bar_star = np.dot(K_star, np.dot(np.linalg.inv(K + (sigma_n**2)*np.eye(n)), y.reshape([n, d])))
    # Covariance.
    cov_f_star = K_star2 - np.dot(K_star, np.dot(np.linalg.inv(K + (sigma_n**2)*np.eye(n)), K_star.T))
    
    return (f_bar_star, cov_f_star)

# Compute posterior mean and covariance. 
f_bar_star, cov_f_star = compute_gpr_parameters(K, K_star2, K_star, sigma_n)

#--------------------------------------------

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(data=cov_f_star, cmap='Blues', ax=ax)
ax.set_title('Components of the Covariance Matrix cov_f_star');

#--------------------------------------------

fig, ax = plt.subplots()

for i in range(0, 100):
    # Sample from posterior distribution. 
    z_star = np.random.multivariate_normal(mean=f_bar_star.squeeze(), cov=cov_f_star)
    # Plot function.
    sns.lineplot(x=x_star[:,1], y=z_star, color="blue", alpha=0.2, ax=ax);
    
# Plot "true" linear fit.
sns.scatterplot(x=x[:,0], y=y, color='red', label = 'f(x)', ax=ax)
ax.set(title=f'Samples of Posterior Distribution, sigma_f = {sigma_f} and l = {l}')
ax.legend(loc='upper right');

#--------------------------------------------
matplotlib.rcParams['figure.figsize'] = (10,10)
# Initialize map
proj=ccrs.Mercator()
m = plt.axes(projection=proj)
plt.rcParams.update({'font.size': 18})
# Plot data
plt.pcolormesh(x_star[:,0].reshape(10,10), x_star[:,1].reshape(10,10), z_star.reshape(10,10), transform = ccrs.PlateCarree(), cmap = plt.get_cmap('viridis'))
plt.scatter(data['lon'], data['lat'], c = data['chl'], s = 150, marker = '+', transform = ccrs.PlateCarree())

m.coastlines(resolution="10m")
m.add_feature(cfeature.OCEAN)
land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m', facecolor=cfeature.COLORS['land'], zorder=0)
m.add_feature(land_10m)
gl=m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2,
                       color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.top_labels = False
gl.right_labels = False

# Add Colorbar
plt.clim(-1,1)
cbar = plt.colorbar(fraction=0.03, pad=0.04)
cbar.set_label('mg', fontsize = 18)

# Add Title
plt.title('Concentration Chlorophyll-a', fontsize = 18)
plt.show()

a=2; b=0.2; sigma=np.sqrt(2); d=1
def cov(h, tau):
    cov = sigma**2 * np.exp(-b**2 * np.linalg.norm(h)**2/(a**2 * tau**2 + 1)) / (a**2 * tau**2 + 1)**(d/2)
    return cov


COV = [cov(i, j) for (i, j) in itertools.product([2,6], [0.2,1.0])]
