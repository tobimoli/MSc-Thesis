#R# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 13:09:48 2021

@author: molenaar
"""
#%% load packages

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

#%% import data
folder = 'P:/11202428-hisea-inter/Tobias_Molenaar/01-Data/sentinel2'
name = "S2B_MSIL1C_20190809T090559_N0208_R050_T35TLE_20190809T112133_resampled"
name = "S2B_MSIL2A_20190809T090559_N0213_R050_T35TLE_20190809T121123_resampled"
#name = "subset_0_of_S2B_MSIL2A_20190809T090559_N0213_R050_T35TLE_20190809T121123_resampled"

class Data:
    def __init__(self):
        self.var = ''
        self.keys = []
        self.var_values = {}
        self.var_units = {}
        self.var_names = {}
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
b = Data()
b.import_data()

#%% select area

#Lo1, Lo2, La1, La2 = 23.0, 28.32, 35.23, 45.44
#Lo1, Lo2, La1, La2 = 25.0, 25.32, 40.23, 40.44
Lo1, Lo2, La1, La2 = 25.0,25.30,40.30,40.50 #small map
lo1, lo2 = np.sum(b.var_values['lon'][0,:]<Lo1), np.sum(b.var_values['lon'][0,:]<Lo2)
la1, la2 = np.sum(b.var_values['lat'][:,0]>La2), np.sum(b.var_values['lat'][:,0]>La1)

lon = b.var_values['lon'][la1:la2,lo1:lo2].flatten()
lat = b.var_values['lat'][la1:la2,lo1:lo2].flatten()
blue = b.var_values['B2'][la1:la2,lo1:lo2].flatten()
green = b.var_values['B3'][la1:la2,lo1:lo2].flatten()
red = b.var_values['B4'][la1:la2,lo1:lo2].flatten()
NIR = b.var_values['B8'][la1:la2,lo1:lo2].flatten()
Class = b.var_values['quality_scene_classification'][la1:la2,lo1:lo2].flatten()

x_sat = pd.DataFrame(np.array([lon,lat,blue,green,red,NIR]).T)
x_sat.columns = ['lon','lat','blue','green','red','NIR']

# Import in-situ data
FOLDER = 'P:/11202428-hisea-inter/Tobias_Molenaar/01-Data/In-Situ/'
NAME = 'ODYSSEA.csv'
#NAME = 'ODYSSEA_unique.csv'

#data = pd.read_table(FOLDER+NAME, sep = ',')

data = pd.read_table(FOLDER+NAME, usecols = [2,4,5,6,8], sep = ';')
data.columns = ['time', 'lon', 'lat', 'dep', 'chl']
data['year'] = [int(n[6:10]) for n in data['time']]
data['date'] = [n[6:10]+n[2:5] + '-' + n[:2] for n in data['time']]
data['date'] = data['date'].astype("datetime64")

data = data[data['date'] == '2019-08-02']
#remove outliers
data = data[data['chl']>0]

x_situ = data[['lon', 'lat', 'dep', 'chl']]
X = pd.concat([x_sat, x_situ], ignore_index=True, sort=False)
print(x_sat.shape)
#b.RGB(subset = [la1,la2,lo1,lo2])
#%%GPR for finding reflectances
x_sat.loc[Class != 6, :] = np.nan

def kernel_function(x, y, sigma_f, l):
    """Define squared exponential kernel function."""       
    kernel = sigma_f**2 * np.exp(- (1/2)* np.dot((x-y)**2, 1/l**2))
    return kernel
def compute_cov_matrices(x, x_star, sigma_f, l):
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
def plotgrid(y, title = '', plot_insitu = False, var_insitu = 'chl', cmap = 'viridis'):
    matplotlib.rcParams['figure.figsize'] = (10,10)
    # Initialize map
    proj=ccrs.Mercator()
    m = plt.axes(projection=proj)
    plt.rcParams.update({'font.size': 18})
    # Plot data
    plt.pcolormesh(lon.reshape(la2-la1,lo1-lo2),
                   lat.reshape(la2-la1,lo1-lo2),
                   y.reshape(la2-la1,lo1-lo2), 
                   transform=ccrs.PlateCarree(), cmap = plt.get_cmap(cmap))
    # Add Colorbar
    #plt.clim(0.09,0.14)
    cbar = plt.colorbar(fraction=0.03, pad=0.04)
    cbar.set_label('', fontsize = 18)
    if plot_insitu:
        plt.scatter(x_situ['lon'][(x_situ['lon']>np.min(lon)) & (x_situ['lon']<np.max(lon))], 
                  x_situ['lat'][(x_situ['lon']>np.min(lon)) & (x_situ['lon']<np.max(lon))], 
                  c = x_situ[var_insitu][(x_situ['lon']>np.min(lon)) & (x_situ['lon']<np.max(lon))],
                  edgecolor='black', linewidth=1.3, s = 150, vmin = np.nanmin(y), vmax = np.nanmax(y),
                  transform=ccrs.PlateCarree(), cmap = plt.get_cmap(cmap))
    else:
        plt.scatter(x_situ['lon'][(x_situ['lon']>np.min(lon)) & (x_situ['lon']<np.max(lon))], 
                  x_situ['lat'][(x_situ['lon']>np.min(lon)) & (x_situ['lon']<np.max(lon))], 
                  c = 'r', transform=ccrs.PlateCarree())
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
    
    # Add Title
    plt.title(title, fontsize = 18)
    plt.show()
def find_closest(lon, lat):
    return x_sat.iloc[((x_sat['lon']-lon)**2 + (x_sat['lat']-lat)**2).argsort()[:4]]
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
def idw(x,y,z,xi,yi):
    k = 4
    zi = np.zeros(z.shape)
    for i in range(len(xi)):
        print(i)
        arr = np.sqrt((x - xi[i])**2 + (y - xi[i])**2)
        pos = np.argpartition(arr, k)[:k]
        dist = arr[pos]
        weights = 1.0 / dist**2
        weights /= weights.sum()
        
        zi[i] = np.dot(weights, z[pos])
    return zi
P = simple_idw(np.array(x_sat['lon']),np.array(x_sat['lat']),np.array(x_sat[['blue', 'green', 'red', 'NIR']]), np.array(x_situ['lon']), np.array(x_situ['lat']))

#add reflectances and depth to the datasets
x_situ.loc[:, ['blue', 'green', 'red', 'NIR']] = P
x_sat.loc[:, 'dep'] = 1*np.ones(len(x_sat))

np.random.seed(101)
x_situ['chl'] = x_situ['chl'] + np.random.normal(0, 0.005, size = len(x_situ)).flatten()
#%% Correlations between ratio's and chlorophyll-a
import scipy.stats
from matplotlib.colors import ListedColormap
import matplotlib as mpl

#x_situ = x_situ[x_situ['dep']<5]
def correlation(x, y = np.log10(np.array(data[['chl']])).flatten()):
    #return scipy.stats.pearsonr(x, y)[0]
    return scipy.stats.spearmanr(x, y)[0]
    #return scipy.stats.kendalltau(x, y)[0]
def heatmap(x, y, title, **kwargs):
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = [1]*len(x)

    if 'palette' in kwargs:
        palette = kwargs['palette']
        n_colors = len(palette)
    else:
        n_colors = 256 # Use 256 colors for the diverging color palette
        palette = sns.color_palette("Blues", n_colors) 

    if 'color_range' in kwargs:
        color_min, color_max = kwargs['color_range']
    else:
        color_min, color_max = min(color), max(color) # Range of values that will be mapped to the palette, i.e. min and max possible correlation

    def value_to_color(val):
        if color_min == color_max:
            return palette[-1]
        else:
            val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            ind = int(val_position * (n_colors - 1)) # target index in the color palette
            return palette[ind]

    if 'size' in kwargs:
        size = kwargs['size']
    else:
        size = [1]*len(x)

    if 'size_range' in kwargs:
        size_min, size_max = kwargs['size_range'][0], kwargs['size_range'][1]
    else:
        size_min, size_max = min(size), max(size)

    size_scale = kwargs.get('size_scale', 500)

    def value_to_size(val):
        if size_min == size_max:
            return 1 * size_scale
        else:
            val_position = (val - size_min) * 0.99 / (size_max - size_min) + 0.01 # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            return val_position * size_scale
    if 'x_order' in kwargs: 
        x_names = [t for t in kwargs['x_order']]
    else:
        x_names = [t for t in sorted(set([v for v in x]))]
    x_to_num = {p[1]:p[0] for p in enumerate(x_names)}

    if 'y_order' in kwargs: 
        y_names = [t for t in kwargs['y_order']]
    else:
        y_names = [t for t in sorted(set([v for v in y]))]
    y_to_num = {p[1]:p[0] for p in enumerate(y_names)}

    plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1) # Setup a 1x10 grid
    ax = plt.subplot(plot_grid[:,:-1]) # Use the left 14/15ths of the grid for the main plot
    
    marker = kwargs.get('marker', 's')

    kwargs_pass_on = {k:v for k,v in kwargs.items() if k not in [
         'color', 'palette', 'color_range', 'size', 'size_range', 'size_scale', 'marker', 'x_order', 'y_order'
    ]}
    
    #plt.title("Pearson's r for depth > 5 meter")
    plt.title(title)
    #plt.title("Kendall's tau")
    ax.scatter(
        x=[x_to_num[v] for v in x],
        y=[y_to_num[v] for v in y],
        marker=marker,
        s=[value_to_size(v) for v in size], 
        c=[value_to_color(v) for v in color],
        **kwargs_pass_on
    )
    ax.set_xticks([v for k,v in x_to_num.items()])
    ax.set_xticklabels([k for k in x_to_num], rotation=45, horizontalalignment='right')
    ax.set_yticks([v for k,v in y_to_num.items()])
    ax.set_yticklabels([k for k in y_to_num])

    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax.set_facecolor('#F1F1F1')
    
    #add colorbar
    ax = plt.subplot(plot_grid[:,-1])
    my_cmap = ListedColormap(palette)
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=my_cmap), cax = ax, ticks=[-1, 0, 1])

def corrplot(data, title= "Spearman's rho", size_scale=1000, marker='s'):
    corr = pd.melt(data.reset_index(), id_vars='index')
    corr.columns = ['x', 'y', 'value']
    heatmap(
        corr['x'], corr['y'], title,
        color=corr['value'], color_range=[-1, 1],
        palette=sns.diverging_palette(20, 220, n=256),
        size=corr['value'].abs(), size_range=[0,1],
        marker=marker,
        x_order=data.columns,
        y_order=data.columns[::-1],
        size_scale=size_scale
    )

findepth = 100
corr_arr = np.zeros((len(list(range(2,findepth))),4))
for depth in range(2, findepth):
    #depth = 30
    data = x_situ[x_situ['dep']<depth]
    corr = data.corr(method = "spearman")
    corr_arr[depth-2] = np.array(corr['chl'])[-4:]
    #plt.figure(figsize=(8, 8))
    #plt.rcParams.update({'font.size': 18})
    #corrplot(corr, title = f"Spearman's rho for depth < {depth}m", size_scale = 1000)
colors = ['b', 'g', 'r','m']
labels = ['Blue', 'Green', 'Red', 'NIR']
plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 18})
for i in range(4):
    plt.plot(list(range(2,findepth)), corr_arr[:,i], color = colors[i], label = labels[i])
plt.xlabel('< Depth [m]'); plt.ylabel("Correlation"); plt.legend(); plt.show()
#%%
data = x_situ[x_situ["dep"]<5]
colors = ['blue', 'green', 'red', 'NIR']
Corr = pd.DataFrame(index=colors, columns=colors)
for color1 in colors:
    for color2 in colors:
        if color1 != color2:
            ratio = np.log10(np.array(data[[color1]])/np.array(data[[color2]])).flatten()
            print(f"Ratio {color1} / {color2} has correlation: {correlation(ratio, y=np.log10(np.array(data[['chl']])).flatten())}")
            Corr[color1][color2] = correlation(ratio, y = np.log10(np.array(data[['chl']])).flatten())
        else:
            Corr[color1][color2] = correlation(x = np.log10(np.array(data[color1])), y = np.log10(np.array(data[['chl']])).flatten())
            print(f"Color {color1} has correlation: {correlation(np.log10(np.array(data[color1])), y=np.log10(np.array(data[['chl']])).flatten())}")
plt.figure(figsize=(8, 8))
plt.rcParams.update({'font.size': 18})
corrplot(Corr, size_scale = 6000, title = "Spearman's rho (all)")
plt.show()

#plt.hist(x_situ['dep'], bins = 100)
#pd.plotting.scatter_matrix(data, alpha=0.2)
ratio = np.array(data[['blue']])/np.array(data[['NIR']])
plt.figure(figsize = (6,6))
plt.scatter(ratio.flatten(), np.array(data[['chl']]).flatten()); 
plt.xlabel('Blue-NIR ratio'); plt.ylabel('Chlorophyll-a concentration')
#%% groups

group = np.zeros(len(x_situ))
situ = np.array(x_situ)
for i in range(len(x_situ)):
    if situ[i,0]<25.07:
        group[i] = 1
    elif situ[i,0]<25.1:
        group[i] = 2
    elif situ[i,0]<25.13:
        group[i] = 3
    elif situ[i,0]<25.16:
        group[i] = 4
    elif situ[i,0]<25.19:
        group[i] = 5
    elif situ[i,1]< 40.35:
        group[i] = 6
    else:
        group[i] = 7
x_situ['cluster'] = group
x_situ['cluster'] = x_situ['cluster'].astype(int)
x_situ['cluster'] = x_situ['cluster'].astype(str)

sns.histplot(data = x_situ, x = "dep", hue = "cluster", multiple = 'stack', bins = 1200)
plt.xlabel('Depth [m]'); plt.ylabel("Number of observations")
#%% Linear Regression for finding chl-a

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

data = x_situ[x_situ['dep']<5]
MSE = pd.DataFrame(index=colors, columns=colors)
SE = pd.DataFrame(index=colors, columns=colors)
#ratio
colors1 = ['blue']
colors2 = ['NIR']

for color1 in colors1:
    for color2 in colors2:
        if color1 != color2:
            R_train = np.log10(np.array(data[[color1]])/np.array(data[[color2]]))
        else:
            R_train = np.log10(np.array(data[[color1]]))
        print(f"{color1}/{color2}")
        X_train = np.array([R_train, R_train**2, R_train**3, R_train**4]).reshape(len(R_train),4)
        Y_train = np.log10(np.array(data[['chl']]))
        
        K = 10; i = 0
        kf = KFold(n_splits=K, shuffle = True, random_state = 0)
        mse = np.zeros(K)
        y_hat = np.zeros((len(Y_train)))
        
        for train, test in kf.split(X_train):
            reg = LinearRegression().fit(X_train[train], Y_train[train])
            
            y_hat[test] = 10**reg.predict(X_train[test]).flatten()
            mse[i] = np.mean((10**Y_train[test]-y_hat[test])**2)
            i += 1
        MSE[color1][color2] = np.round(np.mean(mse),5)
        SE[color1][color2] = np.round(mse.std()/np.sqrt(K),5)
        Y_train = 10**Y_train
        print("The MSE (s.e.) = " + str(np.round(np.mean(mse),5)) + ' (' + str(np.round(mse.std()/np.sqrt(K),5)) + ')')
        print("R-squared = " + str(1 - np.sum((y_hat - Y_train)**2)/np.sum((Y_train - np.mean(Y_train))**2)))
        #print("AIC = " + str(len(Y_train)*np.log(np.mean(mse)) + 2*(1+2)))
        #print("BIC = " + str(len(Y_train)*np.log(np.mean(mse)) + np.log(len(Y_train))*(1+2)))


plt.figure(figsize = (6,6))
plt.plot([0,.5],[0,.5], '-'); plt.plot(y_hat, Y_train, '.', markersize = 10)
plt.xlabel("Prediction CHL-a conc."); plt.ylabel("Observed CHL-a conc."); plt.title("Blue-NIR (shallow)")
plt.xlim([0.09, 0.15])
plt.ylim([0.05, 0.2])
plt.show()

#%% Predict using build-in function
R = np.log10(np.array(x_sat[['green']])/np.array(x_sat[['red']]))
X = np.array([R, R**2, R**3, R**4]).reshape(len(R),4)
reg = LinearRegression().fit(X_train, Y_train)
print("R2 is: ", reg.score(X_train,Y_train))
print("coefficients are: ", reg.coef_)
print("intercept is ", reg.intercept_)
prediction = reg.predict(X)

#LandDetection
prediction[x_sat['NIR']>0.1,:] = np.nan

plotgrid(10**prediction, 'Concentration Chlorophyll-a using LinReg', plot_insitu=True, var_insitu = 'chl')

#%% Plot reflectances
plotgrid(np.array(x_sat['blue']), title = 'Interpolation of the blue wavelength using IDW' ,plot_insitu = True, var_insitu = 'blue', cmap = 'Blues')
plotgrid(np.array(x_sat['green']), title = 'Interpolation of the green wavelength using IDW' ,plot_insitu = True, var_insitu = 'green', cmap = 'Greens')
plotgrid(np.array(x_sat['red']), title = 'Interpolation of the red wavelength using IDW' ,plot_insitu = True, var_insitu = 'red', cmap = 'Reds')
plotgrid(np.array(x_sat['NIR']), title = 'Interpolation of the NIR wavelength using IDW' ,plot_insitu = True, var_insitu = 'NIR', cmap = 'RdPu')

plt.hist(np.array(x_situ[['blue', 'green', 'red', 'NIR']]), bins = 100, color = ['b', 'g', 'r', 'purple'], label = ['blue', 'green', 'red', 'NIR'])
plt.legend()
plt.grid(True)
plt.show()