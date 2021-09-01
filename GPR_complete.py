#R# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 13:09:48 2021

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
from netCDF4 import Dataset
import itertools

folder = 'P:/11202428-hisea-inter/Tobias_Molenaar/01-Data/sentinel2'
name = "S2B_MSIL1C_20190809T090559_N0208_R050_T35TLE_20190809T112133_resampled"
class Data:
    def __init__(self, subset = [0,-1,0,-1]):
        self.var = ''
        self.keys = []
        self.var_values = {}
        self.var_units = {}
        self.var_names = {}
        self.sub = subset
        self.df = pd.DataFrame()
        self.mask = []
    def import_data(self):
        file = folder + "/" + name + ".nc"
        dataset = Dataset(file, "r")
        self.keys = list(dataset.variables.keys())
        for key in self.keys:
            self.var_values[key] = dataset.variables[key][:]
            try:
                self.var_names[key] = dataset.variables[key].long_name
            except:
                self.var_names[key] = key
            try:
                self.var_units[key] = dataset.variables[key].units
            except:
                self.var_units[key] = '-'
        dataset.close()
        print('The data is imported')
    def hist(self, var, ymax = 0):
        self.var = var
        plt.figure(1, figsize = (10,5))
        plt.grid()
        plt.hist(self.var_values[self.var].flatten(), bins=100, label = self.var, alpha = 0.5)
        plt.legend()
        if ymax == 0:
            plt.ylim()
        else:
            plt.ylim([0,ymax])
        plt.plot()   
    def boxplot(self):
        fig, ax = plt.subplots(1, 4, figsize = (15,5))
        i=0
        for key in self.keys:
            if key != 'lon' and key != 'lat':
                ax[i].boxplot(self.var_values[key].flatten())
                ax[i].set_title(key)
                i+=1
        plt.show()
    def plot(self, var):
        self.var = var
        matplotlib.rcParams['figure.figsize'] = (10,10)
        # Initialize map
        proj=ccrs.Mercator()
        m = plt.axes(projection=proj)
        plt.rcParams.update({'font.size': 18})
        # Plot data
        plt.pcolormesh(self.var_values['lon'][self.sub[0]:self.sub[1],self.sub[2]:self.sub[3]],
                       self.var_values['lat'][self.sub[0]:self.sub[1],self.sub[2]:self.sub[3]],
                       self.var_values[self.var][self.sub[0]:self.sub[1],self.sub[2]:self.sub[3]], 
                       transform=ccrs.PlateCarree(), cmap = plt.get_cmap('viridis'))
        
        #shp = shapereader.Reader('C:/Users/molenaar/Downloads/gadm36_GRC_shp/gadm36_GRC_0')
        #for record, geometry in zip(shp.records(), shp.geometries()):
        #    m.add_geometries([geometry], ccrs.PlateCarree(), alpha = 0.5, facecolor='lightgray', edgecolor='black')

        gl=m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2,
                               color='gray', alpha=0.5, linestyle='--')
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.top_labels = False
        gl.right_labels = False
        
        # Add Colorbar
        cbar = plt.colorbar(fraction=0.03, pad=0.04)
        cbar.set_label(self.var_units[self.var], fontsize = 18)
        
        # Add Title
        plt.title(self.var_names[self.var], fontsize = 18)
        plt.show()
    def dataframe(self):
        for key in self.keys:
            self.df[key] = self.var_values[key].flatten()[~self.mask]
    def correlation(self):
        plt.figure(figsize = (8,8))
        corrMatrix = self.df.corr()
        sns.heatmap(corrMatrix, annot=True)
        plt.show()
        sns.pairplot(self.df.sample(n = 10000, random_state = 44))
    def masktotal(self):
        mask = np.zeros(len(self.var_values['lon'].flatten()))
        for key in self.keys:
            mask += ma.getmask(self.var_values[key].flatten())
        self.mask = np.array(mask != 0)
    def RGB(self):
        x,y = self.var_values['lon'][self.sub[0]:self.sub[1],self.sub[2]:self.sub[3]].shape
        retrack_original = np.zeros((x,y,3),dtype=int)
        MAX = [np.max(self.var_values['B4']), np.max(self.var_values['B3']), np.max(self.var_values['B2'])]
        for i in range(x):
            for j in range(y):
                retrack_original[i][j][0] = self.var_values['B4'].data[i+self.sub[0],j+self.sub[2]]/MAX[0]*255
                retrack_original[i][j][1] = self.var_values['B3'].data[i+self.sub[0],j+self.sub[2]]/MAX[1]*255
                retrack_original[i][j][2] = self.var_values['B2'].data[i+self.sub[0],j+self.sub[2]]/MAX[2]*255
        plt.imshow(retrack_original)
    def chl(self, alg):
        if alg == 0:
            self.var_values['chl'] = self.var_values['B2']/self.var_values['B3']
        elif alg == 1:
            self.var_values['chl'] = self.var_values['B8']/self.var_values['B4']
        self.var_units['chl'] = 'mg'
        self.var_names['chl'] = 'Chlorophyll-a Concentration ' + str(alg)
b = Data()
b.import_data()

Lo1, Lo2, La1, La2 = 25.10,25.12,40.33,40.34
#Lo1, Lo2, La1, La2 = 25.0,25.30,40.30,40.50 #small map

lo1, lo2 = np.sum(b.var_values['lon'][0,:]<Lo1), np.sum(b.var_values['lon'][0,:]<Lo2)
la1, la2 = np.sum(b.var_values['lat'][:,0]>La2), np.sum(b.var_values['lat'][:,0]>La1)

lon = b.var_values['lon'][la1:la2,lo1:lo2].flatten()
lat = b.var_values['lat'][la1:la2,lo1:lo2].flatten()
blue = b.var_values['B2'][la1:la2,lo1:lo2].flatten()
green = b.var_values['B3'][la1:la2,lo1:lo2].flatten()
red = b.var_values['B4'][la1:la2,lo1:lo2].flatten()
NIR = b.var_values['B8'][la1:la2,lo1:lo2].flatten()

x_sat = pd.DataFrame([lon,lat,blue,green,red,NIR]).T
x_sat.columns = ['lon','lat','blue','green','red','NIR']

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

x_situ = data[['lon', 'lat', 'dep', 'chl']]
X = pd.concat([x_sat, x_situ], ignore_index=True, sort=False)

#%%GPR for finding reflectances
def kernel_function(x, y, sigma_f=1, l=1):
    """Define squared exponential kernel function."""       
    kernel = sigma_f**2 * np.exp(- (1/2)* np.dot((x-y)**2, 1/l**2))
    return kernel

def compute_cov_matrices(x, x_star, sigma_f=1, l=1):
    """
    Compute components of the covariance matrix of the joint distribution.
    We follow the notation:
        - K = K(X, X) 
        - K_star = K(X_*, X)
        - K_star2 = K(X_*, X_*)
    """
    x = np.array(x); x_star = np.array(x_star)
    n = x.shape[0]
    n_star = x_star.shape[0]

    K = [kernel_function(i, j, sigma_f=sigma_f, l=l) for (i, j) in itertools.product(x, x)]
    K = np.array(K).reshape(n, n)
    K_star2 = [kernel_function(i, j, sigma_f=sigma_f, l=l) for (i, j) in itertools.product(x_star, x_star)]
    K_star2 = np.array(K_star2).reshape(n_star, n_star)
    K_star = [kernel_function(i, j, sigma_f=sigma_f, l=l) for (i, j) in itertools.product(x_star, x)]
    K_star = np.array(K_star).reshape(n_star, n)
    
    return (K, K_star2, K_star)

def compute_gpr_parameters(K, K_star2, K_star, y, sigma_n):
    """Compute gaussian regression parameters."""
    n = K.shape[0]
    Y = np.array(y).reshape([n, d])
    # Mean.
    f_bar_star = np.dot(K_star, np.dot(np.linalg.inv(K + (sigma_n**2)*np.eye(n)), Y))
    # Covariance.
    cov_f_star = K_star2 - np.dot(K_star, np.dot(np.linalg.inv(K + (sigma_n**2)*np.eye(n)), K_star.T))
    
    return (f_bar_star, cov_f_star)

def plotmatrix(matrix, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data=matrix, cmap='Blues', ax=ax)
    ax.set(title=title)
    
n = x_sat.shape[0]
n_star = x_situ.shape[0]
l = np.array([0.1, 0.1])
sigma_f = 2
sigma_n = 1
d = 4

K, K_star2, K_star = compute_cov_matrices(x_sat[['lon', 'lat']], x_situ[['lon','lat']], 
                                          sigma_f=sigma_f, l=l)

a = np.concatenate((K + (sigma_n**2)*np.eye(n), K_star), axis=0)
b = np.concatenate((K_star.T, K_star2), axis=0)
C = np.concatenate((a, b), axis=1)
plotmatrix(C, 'Components of the Covariance Matrix C')
plotmatrix(K_star2, 'Components of the Kernel Matrix K**')
#%%--------------------------------------------

fig, ax = plt.subplots()

for i in range(100):
    # Sample from prior distribution. 
    z_star = np.random.multivariate_normal(mean=np.zeros(n_star), cov=K_star2)
    # Plot function.
    sns.lineplot(x=x_situ['lon'], y=z_star, color='blue', alpha=0.2, ax=ax)
    
# Plot "true" linear fit.
#sns.lineplot(x=x, y=f_x, color='red', label='f(x)', ax=ax)
ax.set(title='Samples of Prior Distribution')
ax.legend(loc='lower right')

#%% Compute posterior mean and covariance. 
f_bar_star, cov_f_star = compute_gpr_parameters(K, K_star2, K_star, x_sat[['blue','green','red','NIR']], sigma_n)
plotmatrix(cov_f_star, 'Components of the Covariance Matrix cov_f_star')

fig, ax = plt.subplots()

for i in range(100):
    # Sample from posterior distribution. 
    z_star = np.random.multivariate_normal(mean=f_bar_star[:,1], cov=cov_f_star)
    # Plot function.
    sns.lineplot(x=x_situ['lon'], y=z_star, color="blue", alpha=0.2, ax=ax);
    
# Plot "true" linear fit.
#sns.lineplot(x=x, y=f_x, color='red', label = 'f(x)', ax=ax)
ax.set(title=f'Samples of Posterior Distribution, sigma_f = {sigma_f} and l = {l}')
ax.legend(loc='upper right');

#--------------------------------------------
#add reflectances and depth to the datasets
x_situ[['blue', 'green', 'red', 'NIR']] = f_bar_star
x_sat['dep'] = 5*np.ones(len(x_sat))

#%% GPR for finding chl-a concentration

n = x_situ.shape[0]
n_star = x_sat.shape[0]
l = np.array([0.1, 0.1, 1, 0.1, 0.1, 0.1, 0.1])
sigma_f = 2
sigma_n = 1
d = 1

K, K_star2, K_star = compute_cov_matrices(x_situ[['lon', 'lat', 'dep', 'blue', 'green', 'red', 'NIR']], 
                                          x_sat[['lon','lat', 'dep', 'blue', 'green', 'red', 'NIR']], 
                                          sigma_f=sigma_f, l=l)

a = np.concatenate((K + (sigma_n**2)*np.eye(n), K_star), axis=0)
b = np.concatenate((K_star.T, K_star2), axis=0)
C = np.concatenate((a, b), axis=1)
plotmatrix(C, 'Components of the Covariance Matrix C')

#%%--------------------------------------------

fig, ax = plt.subplots()

for i in range(100):
    # Sample from prior distribution. 
    z_star = np.random.multivariate_normal(mean=np.zeros(n_star), cov=K_star2)
    # Plot function.
    sns.lineplot(x=x_sat['lon'], y=z_star, color='blue', alpha=0.2, ax=ax)
    
# Plot "true" linear fit.
#sns.lineplot(x=x, y=f_x, color='red', label='f(x)', ax=ax)
ax.set(title='Samples of Prior Distribution')
ax.legend(loc='lower right')

#%%--------------------------------------------

# Compute posterior mean and covariance. 
f_bar_star, cov_f_star = compute_gpr_parameters(K, K_star2, K_star, x_situ[['chl']], sigma_n)
plotmatrix(cov_f_star, 'Components of the Covariance Matrix cov_f_star')

fig, ax = plt.subplots()

for i in range(100):
    # Sample from posterior distribution. 
    z_star = np.random.multivariate_normal(mean=f_bar_star[:,0], cov=cov_f_star)
    # Plot function.
    sns.lineplot(x=x_sat['lon'], y=z_star, color="blue", alpha=0.2, ax=ax);
    
# Plot "true" linear fit.
#sns.lineplot(x=x, y=f_x, color='red', label = 'f(x)', ax=ax)
ax.set(title=f'Samples of Posterior Distribution, sigma_f = {sigma_f} and l = {l}')
ax.legend(loc='upper right');

#%%--------------------------------------------

def plotgrid(y, title):
    matplotlib.rcParams['figure.figsize'] = (10,10)
    # Initialize map
    proj=ccrs.Mercator()
    m = plt.axes(projection=proj)
    plt.rcParams.update({'font.size': 18})
    # Plot data
    plt.pcolormesh(lon.reshape(la2-la1,lo1-lo2),
                   lat.reshape(la2-la1,lo1-lo2),
                   y.reshape(la2-la1,lo1-lo2), 
                   transform=ccrs.PlateCarree(), cmap = plt.get_cmap('viridis'))
    plt.scatter(x_situ['lon'][(x_situ['lon']>np.min(lon)) & (x_situ['lon']<np.max(lon))], 
             x_situ['lat'][(x_situ['lon']>np.min(lon)) & (x_situ['lon']<np.max(lon))], 
             c = x_situ['chl'][(x_situ['lon']>np.min(lon)) & (x_situ['lon']<np.max(lon))],
             edgecolor='black', linewidth=1.3, s = 150,
             transform=ccrs.PlateCarree())
    #m.set_extent([Lo1,Lo2,La1,La2])
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
    #plt.clim(0,0.3)
    cbar = plt.colorbar(fraction=0.03, pad=0.04)
    cbar.set_label('mg', fontsize = 18)
    
    # Add Title
    plt.title(title, fontsize = 18)
    plt.show()

plotgrid(f_bar_star, 'Concentration Chlorophyll-a using GPR')
plotgrid(np.sqrt(np.diag(cov_f_star)), 'Std. Dev. of Concentration Chlorophyll-a using GPR')
#%% 
from numpy.linalg import inv

def kernel(X1, X2, l=1.0, sigma_f=1.0):
    """
    Isotropic squared exponential kernel.
    
    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        (m x n) matrix.
    """
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

def posterior(X_s, X_train, Y_train, l=1.0, sigma_f=1.0, sigma_y=1e-8):
    """
    Computes the suffifient statistics of the posterior distribution 
    from m training data X_train and Y_train and n new inputs X_s.
    
    Args:
        X_s: New input locations (n x d).
        X_train: Training locations (m x d).
        Y_train: Training targets (m x 1).
        l: Kernel length parameter.
        sigma_f: Kernel vertical variation parameter.
        sigma_y: Noise parameter.
    
    Returns:
        Posterior mean vector (n x d) and covariance matrix (n x n).
    """
    K = kernel(X_train, X_train, l, sigma_f) + sigma_y**2 * np.eye(len(X_train))
    K_s = kernel(X_train, X_s, l, sigma_f)
    K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
    K_inv = inv(K)
    
    # Equation (7)
    mu_s = K_s.T.dot(K_inv).dot(Y_train)

    # Equation (8)
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
    
    return mu_s, cov_s
#%% GPy

X = np.array(x_sat[['lon', 'lat', 'dep', 'blue', 'green', 'red', 'NIR']])
X_train = np.array(x_situ[['lon', 'lat', 'dep', 'blue', 'green', 'red', 'NIR']])
Y_train = np.array(x_situ[['chl']])
sigma_n = 1
sigma_f = 1
l = np.array([1, 1, 1, 1, 1, 1, 1])
d = 1
import GPy

rbf = GPy.kern.RBF(input_dim=7, variance=sigma_f, lengthscale=l, ARD = True)
gpr = GPy.models.GPRegression(X_train, Y_train, rbf)

print(gpr)

# Fix the noise variance to known value 
gpr.Gaussian_noise.variance = sigma_n**2

# Run optimization
gpr.optimize();

print(gpr)

# Obtain optimized kernel parameters
l_opt = gpr.rbf.lengthscale.values
sigma_f_opt = np.sqrt(gpr.rbf.variance.values[0])
sigma_n_opt = np.sqrt(gpr.Gaussian_noise.variance[0])
print(l_opt)
print(sigma_f_opt)
print(sigma_n_opt)

#%% Compute Posterior
K, K_star2, K_star = compute_cov_matrices(X_train, X, sigma_f=sigma_f_opt, l=l_opt)

#a = np.concatenate((K + (sigma_n_opt**2)*np.eye(K.shape[0]), K_star), axis=0)
#b = np.concatenate((K_star.T, K_star2), axis=0)
#C = np.concatenate((a, b), axis=1)
#plotmatrix(C, 'Components of the Covariance Matrix C')

f_bar_star, cov_f_star = compute_gpr_parameters(K, K_star2, K_star, Y_train, sigma_n_opt)
plotmatrix(cov_f_star, 'Components of the Covariance Matrix cov_f_star')

#%% Predict using build-in function

prediction = gpr.predict(X, full_cov = False, include_likelihood = True)
plotgrid(prediction[0], 'Concentration Chlorophyll-a using GPR')
plotgrid(np.sqrt(prediction[1]), 'Std. Dev. of Concentration Chlorophyll-a using GPR')

#%% Plot Posterior

plotgrid(f_bar_star, 'Concentration Chlorophyll-a using GPR')
plotgrid(np.sqrt(np.diag(cov_f_star)), 'Std. Dev. of Concentration Chlorophyll-a using GPR')


