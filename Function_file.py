# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 12:08:04 2021

@author: molenaar
"""
#In this file, the functions for all other scripts are written
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
from netCDF4 import Dataset
from scipy.sparse.linalg import eigsh
log2pi = np.log(2*np.pi)

#functions:
class Data:
    def __init__(self, folder, name):
        self.var = ''
        self.keys = []
        self.var_values = {}
        self.var_units = {}
        self.var_names = {}
        self.df = pd.DataFrame()
        self.mask = []
        self.folder = folder
        self.name = name
    def import_data(self):
        file = self.folder + "/" + self.name + ".nc"
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
    def plot(self, var, subset = [0,-1,0,-1]):
        self.var = var
        matplotlib.rcParams['figure.figsize'] = (10,10)
        # Initialize map
        proj=ccrs.Mercator()
        m = plt.axes(projection=proj)
        plt.rcParams.update({'font.size': 18})
        # Plot data
        plt.pcolormesh(self.var_values['lon'][subset[0]:subset[1],subset[2]:subset[3]],
                       self.var_values['lat'][subset[0]:subset[1],subset[2]:subset[3]],
                       self.var_values[self.var][subset[0]:subset[1],subset[2]:subset[3]], 
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
    def RGB(self, subset = [0,-1,0,-1]):
        x,y = self.var_values['lon'][subset[0]:subset[1],subset[2]:subset[3]].shape
        retrack_original = np.zeros((x,y,3),dtype=int)
        MAX = [np.max(self.var_values['B4']), np.max(self.var_values['B3']), np.max(self.var_values['B2'])]
        for i in range(x):
            for j in range(y):
                retrack_original[i][j][0] = self.var_values['B4'].data[i+subset[0],j+subset[2]]/MAX[0]*255
                retrack_original[i][j][1] = self.var_values['B3'].data[i+subset[0],j+subset[2]]/MAX[1]*255
                retrack_original[i][j][2] = self.var_values['B2'].data[i+subset[0],j+subset[2]]/MAX[2]*255
        plt.imshow(retrack_original)
    def chl(self, alg):
        if alg == 0:
            self.var_values['chl'] = self.var_values['B2']/self.var_values['B3']
        elif alg == 1:
            self.var_values['chl'] = self.var_values['B8']/self.var_values['B4']
        self.var_units['chl'] = 'mg'
        self.var_names['chl'] = 'Chlorophyll-a Concentration ' + str(alg)

def plotmatrix(matrix, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data=matrix, cmap='Blues', ax=ax)
    ax.set(title=title)
def plotgrid(y, lon, lat, coord, x_situ = pd.DataFrame(), title = '', plot_insitu = False, var_insitu = 'chl', cmap = 'viridis'):
    matplotlib.rcParams['figure.figsize'] = (10,10)
    # Initialize map
    proj=ccrs.Mercator()
    m = plt.axes(projection=proj)
    plt.rcParams.update({'font.size': 18})
    # Plot data
    [lo1, lo2, la1, la2] = coord
    plt.pcolormesh(lon.reshape(la2-la1,lo1-lo2),
                   lat.reshape(la2-la1,lo1-lo2),
                   y.reshape(la2-la1,lo1-lo2), 
                   transform=ccrs.PlateCarree(), cmap = plt.get_cmap(cmap))
    # Add Colorbar
    #plt.clim(0.0,1.3)
    cbar = plt.colorbar(fraction=0.03, pad=0.04)
    cbar.set_label('mg m-3', fontsize = 18)
    if plot_insitu:
        plt.scatter(x_situ['lon'][(x_situ['lon']>np.min(lon)) & (x_situ['lon']<np.max(lon))], 
                  x_situ['lat'][(x_situ['lon']>np.min(lon)) & (x_situ['lon']<np.max(lon))], 
                  c = x_situ[var_insitu][(x_situ['lon']>np.min(lon)) & (x_situ['lon']<np.max(lon))],
                  edgecolor='black', linewidth=1.3, s = 150, vmin = np.nanmin(y), vmax = np.nanmax(y),
                  cmap = plt.get_cmap(cmap), transform=ccrs.PlateCarree())
    elif x_situ.empty:
        pass
    else:
        plt.scatter(x_situ['lon'][(x_situ['lon']>np.min(lon)) & (x_situ['lon']<np.max(lon)) & (x_situ['lat']>np.min(lat)) & (x_situ['lat']<np.max(lat))], 
                  x_situ['lat'][(x_situ['lon']>np.min(lon)) & (x_situ['lon']<np.max(lon)) & (x_situ['lat']>np.min(lat)) & (x_situ['lat']<np.max(lat))], 
                  c = 'r', transform=ccrs.PlateCarree())
    #m.set_extent([Lo1,Lo2,La1,La2])
    #m.coastlines(resolution="10m")
    m.add_feature(cfeature.OCEAN)
    land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m', facecolor=cfeature.COLORS['land'], zorder=0)
    m.add_feature(land_10m)
    gl=m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2,
                           color='gray', alpha=0.5, linestyle='--')
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.top_labels = False
    gl.right_labels = False
    
    # Add Title
    plt.title(title, fontsize = 18)
    plt.show()

def distance_matrix(x0, y0, x1, y1):
    obs = np.vstack((x0, y0)).T
    interp = np.vstack((x1, y1)).T

    # Make a distance matrix between pairwise observations
    # Note: from <http://stackoverflow.com/questions/1871536>
    # (Yay for ufuncs!)
    d0 = np.subtract.outer(obs[:,0], interp[:,0])
    d1 = np.subtract.outer(obs[:,1], interp[:,1])

    return np.hypot(d0, d1)
def simple_idw(x, y, z, xi, yi):
    dist = distance_matrix(x,y, xi,yi)

    # In IDW, weights are 1 / distance
    weights = 1.0 / dist**2

    # Make weights sum to one
    weights /= np.nansum(weights, axis=0)
    
    w1 = np.ma.array(weights, mask = np.isnan(weights))
    z1 = np.ma.array(z, mask = np.isnan(z))
    # Multiply the weights for each interpolated point by all observed Z-values
    zi = np.ma.dot(w1.T, z1)
    return zi

def compute_gpr_parameters(K, K_star2, K_star, sigma_n, Y):
    """Compute gaussian regression parameters."""
    n = K.shape[0]
    # Mean.
    f_bar_star = np.dot(K_star, np.dot(np.linalg.inv(K + (sigma_n**2)*np.eye(n)),Y.reshape([n, 1])))
    # Covariance.
    var_f_star = K_star2 - (np.dot(K_star, np.linalg.inv(K+(sigma_n**2)*np.eye(n)))*K_star).sum(-1)
    
    return (f_bar_star.flatten(), var_f_star.flatten())
def compute_gpr_parameters_svd(K, K_star2, K_star, sigma_n, k, Y):
    """Compute gaussian regression parameters."""
    n = K.shape[0]
    [S, U] = eigsh(K + (sigma_n**2)*np.eye(n), k = k)
    
    # Mean.
    f_bar_star = np.dot(K_star, np.dot(np.dot(U * 1/S, U.T), Y))
    # Covariance.
    var_f_star = K_star2 - (np.dot(K_star, np.dot(U * 1/S, U.T))* K_star).sum(-1)
    
    return (f_bar_star.flatten(), var_f_star.flatten())
def compute_gpr_parameters_nystrom(K, K_star2, K_star, sigma_n, k, Y):
    """Compute gaussian regression parameters."""
    n = K.shape[0]
    
    rows = np.random.randint(n, size = k)
    Kmm = K[rows, :][:,rows] #m x m
    Kmn = K[:, rows]; np.fill_diagonal(Kmn, Kmn.diagonal() + sigma_n**2)
    
    [U, S, _] = np.linalg.svd(Kmm + np.eye(k)*sigma_n**2, hermitian = True) #m x m
    print(S)
    S[S<sigma_n**2] = sigma_n**2
    Shat = n/k * S
    Uhat = np.sqrt(k/n)/S * np.dot(Kmn, U)
    
    # Mean.
    f_bar_star = np.dot(K_star, np.dot(np.dot(Uhat * 1/Shat, Uhat.T), Y.reshape([n, 1])))
    # Covariance.
    cov_f_star = K_star2 - np.dot(K_star, np.dot(np.dot(Uhat * 1/Shat, Uhat.T), K_star.T))
    
    var_f_star = np.diag(cov_f_star)
    return (f_bar_star.flatten(), var_f_star.flatten())

def kernel_function(x, y, sigma_f=1, l=1):
    """Define squared exponential kernel function."""
    dist = np.subtract.outer(x.flatten(),y.flatten())**2
    kernel = sigma_f**2 * np.exp(- dist/ (2 * l**2))
    return kernel
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
    
    K = kernel_function(x, x, sigma_f=sigma_f, l=l)
    K = np.array(K).reshape(n, n)
    
    K_star2 = kernel_function(x_star, x_star, sigma_f=sigma_f, l=l)
    K_star2 = np.array(K_star2).reshape(n_star, n_star)
    
    K_star = kernel_function(x_star, x, sigma_f=sigma_f, l=l)
    K_star = np.array(K_star).reshape(n_star, n)
    
    return (K, K_star2, K_star)
def marglike(par,x,y,Y): 
    l,std,sigma_n = par
    n = len(x)
    dist_x = (x - x.T)**2
    k = std**2 * np.exp(-(dist_x/(2*(l**2)))) + (sigma_n**2)*np.eye(n)
    print(par)
    inverse = np.linalg.inv(k)
    det = np.linalg.det(k)
    if det > 10**240:
        ml = (1/2)*np.dot(np.dot(Y.T,inverse),Y) + (1/2)*553 + (n/2)*log2pi
    elif det < 10**-240:
        ml = (1/2)*np.dot(np.dot(Y.T,inverse),Y) - (1/2)*553 + (n/2)*log2pi
    else:
        ml = (1/2)*np.dot(np.dot(Y.T,inverse),Y) + (1/2)*np.log(det) + (n/2)*log2pi
    return ml[0,0]
def marglike_svd(par,x,y,k,Y): 
    l,std,sigma_n = par
    n = len(x)
    dist_x = (x - x.T)**2
    kk = std**2 * np.exp(-(dist_x/(2*(l**2)))) + (sigma_n**2)*np.eye(n)
    [S, U] = eigsh(kk + (sigma_n**2)*np.eye(n), k = k)
    
    det = np.prod(S)
    #print(par)
    if det > 10**240:
        ml = (1/2)*np.dot(np.dot(Y.T,np.dot(U * 1/S, U.T)),Y) + (1/2)*553 + (n/2)*log2pi
    elif det < 10**-240:
        ml = (1/2)*np.dot(np.dot(Y.T,np.dot(U * 1/S, U.T)),Y) - (1/2)*553 + (n/2)*log2pi
    else:
        ml = (1/2)*np.dot(np.dot(Y.T,np.dot(U * 1/S, U.T)),Y) + (1/2)*np.log(det) + (n/2)*log2pi
    return ml[0,0]
def marglike_nystrom(par,x,y,k,Y,K): 
    l,std,sigma_n = par
    n = len(x)
    dist_x = (x - x.T)**2
    kk = std**2 * np.exp(-(dist_x/(2*(l**2)))) + (sigma_n**2)*np.eye(n)

    rows = np.random.randint(n, size = k)
    Kmm = kk[rows, :][:,rows] #m x m
    [U, S, _] = np.linalg.svd(Kmm, hermitian = True) #m x m
    Shat = n/k * S
    Uhat = np.sqrt(k/n)/S * np.dot(K[:,rows], U)
    
    det = np.prod(S)
    #print(par)
    if det > 10**240:
        ml = (1/2)*np.dot(np.dot(Y.T,np.dot(Uhat * 1/Shat, Uhat.T)),Y) + (1/2)*553 + (n/2)*log2pi
    elif det < 10**-240:
        ml = (1/2)*np.dot(np.dot(Y.T,np.dot(Uhat * 1/Shat, Uhat.T)),Y) - (1/2)*553 + (n/2)*log2pi
    else:
        ml = (1/2)*np.dot(np.dot(Y.T,np.dot(Uhat * 1/Shat, Uhat.T)),Y) + (1/2)*np.log(det) + (n/2)*log2pi
    return ml[0,0]

def plot(X_star, mean, var, mean_svd, var_svd, X, Y, k, folder, save = False):
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
    if save:
        plt.savefig(folder + '/fig_' + str(k) + '.png')
    plt.show()

def plot_errbar(Y_train, Prediction, xval, xlabel, n = 1000):
    plt.figure(figsize = (10, 8))
    plt.rcParams.update({'font.size': 18})
    plt.errorbar(xval[:n], Prediction[0,:n], yerr=2*np.sqrt(Prediction[1,:n]).flatten(), fmt='.k', label = 'Predicted value');
    plt.plot(xval[:n], Y_train[:n], 'r.', markersize = 8, label = 'True value')
    plt.grid(True)
    plt.ylim([0.0,0.5])
    plt.xlabel(xlabel)
    plt.ylabel('Chl-a concentration')
    plt.legend()
    plt.show()
def plot_errcont(xval, Y_train, Prediction, xfit, xlabel, title = ''):
    plt.figure(figsize = (10, 8))
    plt.rcParams.update({'font.size': 18})
    #plt.plot(xval, Y_train, 'r.', markersize = 8, label = 'True value')
    yfit = Prediction[0].flatten()
    dyfit = 2 * np.sqrt(Prediction[1].flatten())
    
    plt.plot(xfit, yfit, '-', color='gray', label = 'Posterior Mean')
    plt.fill_between(xfit, yfit - dyfit, yfit + dyfit, color='gray', alpha=0.2, label = "2 std range")
    plt.title(title)
    plt.grid(True)
    plt.ylim([0.0,0.5])
    plt.xlabel(xlabel)
    plt.ylabel('Chl-a concentration [mg.m-3]')
    plt.legend()
    plt.show()