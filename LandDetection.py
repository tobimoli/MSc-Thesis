# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 11:58:30 2021

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
import GPy
import time

#%% import data
folder = 'P:/11202428-hisea-inter/Tobias_Molenaar/01-Data/sentinel2'
name = "S2B_MSIL1C_20190809T090559_N0208_R050_T35TLE_20190809T112133_resampled"
name = "S2B_MSIL2A_20190809T090559_N0213_R050_T35TLE_20190809T121123_resampled"
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
Lo1, Lo2, La1, La2 = 25,25.40,39.80,40.00
lo1, lo2 = np.sum(b.var_values['lon'][0,:]<Lo1), np.sum(b.var_values['lon'][0,:]<Lo2)
la1, la2 = np.sum(b.var_values['lat'][:,0]>La2), np.sum(b.var_values['lat'][:,0]>La1)

lon = b.var_values['lon'][la1:la2,lo1:lo2].flatten()
lat = b.var_values['lat'][la1:la2,lo1:lo2].flatten()
blue = b.var_values['B2'][la1:la2,lo1:lo2].flatten()
green = b.var_values['B3'][la1:la2,lo1:lo2].flatten()
red = b.var_values['B4'][la1:la2,lo1:lo2].flatten()
NIR = b.var_values['B8'][la1:la2,lo1:lo2].flatten()

x_sat = pd.DataFrame(np.array([lon,lat,blue,green,red,NIR]).T)
x_sat.columns = ['lon','lat','blue','green','red','NIR']

# Import in-situ data
FOLDER = 'P:/11202428-hisea-inter/Tobias_Molenaar/01-Data/In-Situ/'
NAME = 'ODYSSEA_unique.csv'
data = pd.read_table(FOLDER+NAME, sep = ',')

data['date'] = data['date'].astype("datetime64")

data = data[data['date'] == '2019-08-02']
#remove outliers
data = data[data['chl']>0]

x_situ = data[['lon', 'lat', 'dep', 'chl']]
X = pd.concat([x_sat, x_situ], ignore_index=True, sort=False)
print(x_sat.shape)
b.RGB(subset = [la1,la2,lo1,lo2]); plt.title('RGB image of Limnos')
#%%GPR for finding reflectances
def plotmatrix(matrix, title):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data=matrix, cmap='Blues', ax=ax)
    ax.set(title=title)
def plotgrid(y, title = '', plot_insitu = False, var_insitu = 'chl', cmap = plt.get_cmap('viridis')):
    matplotlib.rcParams['figure.figsize'] = (10,10)
    # Initialize map
    proj=ccrs.Mercator()
    m = plt.axes(projection=proj)
    plt.rcParams.update({'font.size': 18})
    # Plot data
    plt.pcolormesh(lon.reshape(la2-la1,lo1-lo2),
                   lat.reshape(la2-la1,lo1-lo2),
                   y.reshape(la2-la1,lo1-lo2), 
                   transform=ccrs.PlateCarree(), cmap = cmap)
    # Add Colorbar
    #plt.clim(0,0.13)
    cbar = plt.colorbar(fraction=0.03, pad=0.04, ticks = [0, 0.67, 1.33, 2])
    cbar.ax.set_yticklabels(['0', '0.07', '0.13', '1'])
    cbar.set_label('Reflectance (ratio)', fontsize = 18)
    plt.scatter(z['lon'], z['lat'], c = 'orange', marker = '.', transform=ccrs.PlateCarree())
    if plot_insitu:
        plt.scatter(x_situ['lon'][(x_situ['lon']>np.min(lon)) & (x_situ['lon']<np.max(lon))], 
                  x_situ['lat'][(x_situ['lon']>np.min(lon)) & (x_situ['lon']<np.max(lon))], 
                  c = x_situ[var_insitu][(x_situ['lon']>np.min(lon)) & (x_situ['lon']<np.max(lon))],
                  edgecolor='black', linewidth=1.3, s = 150, vmin = np.nanmin(y), vmax = np.nanmax(y),
                  transform=ccrs.PlateCarree())
    else:
        plt.scatter(x_situ['lon'][(x_situ['lon']>np.min(lon)) & (x_situ['lon']<np.max(lon))], 
                  x_situ['lat'][(x_situ['lon']>np.min(lon)) & (x_situ['lon']<np.max(lon))], 
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
def find_closest(lon, lat, nr):
    return x_sat.iloc[((x_sat['lon']-lon)**2 + (x_sat['lat']-lat)**2).argsort()[:nr]]
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
    weights = 1.0 / dist

    # Make weights sum to one
    weights /= weights.sum(axis=0)

    # Multiply the weights for each interpolated point by all observed Z-values
    zi = np.dot(weights.T, z)
    return zi

#x_sat.loc[x_sat['NIR']>0.091,:] = np.nan

start = time.time()
for i in range(1):
    P = simple_idw(np.array(x_sat['lon']),np.array(x_sat['lat']),np.array(x_sat[['blue', 'green', 'red', 'NIR']]), np.array(x_situ['lon']), np.array(x_situ['lat']))
end = time.time()
print(end - start)

#add reflectances and depth to the datasets
x_situ.loc[:, ['blue', 'green', 'red', 'NIR']] = P
x_sat.loc[:, 'dep'] = 1*np.ones(len(x_sat))

#%% GPR for finding chl-a
X = np.array(x_sat[['lon', 'lat','dep', 'blue', 'green', 'red', 'NIR']])
X_train = np.array(x_situ[['lon', 'lat','dep', 'blue', 'green', 'red', 'NIR']])
Y_train = np.array(x_situ[['chl']])

sigma_n = 1
sigma_f = 1
l = np.array([10, 10, 100, .1, .1, 1, 1])

start = time.time()
for i in range(1):
    rbf = GPy.kern.RBF(input_dim=7, variance=sigma_f, lengthscale=l, ARD = True)
    gpr = GPy.models.GPRegression(X_train, Y_train, rbf)
    
    #print(gpr)
    
    # Fix the noise variance to known value 
    gpr.Gaussian_noise.variance = sigma_n**2
    
    # Run optimization
    gpr.optimize();

#print(gpr)
end = time.time()
print(end - start)

# Obtain optimized kernel parameters
l_opt = gpr.rbf.lengthscale.values
sigma_f_opt = np.sqrt(gpr.rbf.variance.values[0])
sigma_n_opt = np.sqrt(gpr.Gaussian_noise.variance[0])
np.set_printoptions(suppress=True)
print(np.around(l_opt,3))
np.set_printoptions(suppress=False)

# Include noise into predictions using Equation (9).
#cov_s = cov_s + noise**2 * np.eye(*cov_s.shape)

#%% Predict using build-in function

start = time.time()
for i in range(1):
    prediction = gpr.predict(X, full_cov= False, include_likelihood= False)
end = time.time()
print(end - start)

plotgrid(prediction[0], 'Concentration Chlorophyll-a using GPR', plot_insitu=True, var_insitu = 'chl')
plotgrid(np.sqrt(prediction[1]), 'Std. Dev. of Concentration Chlorophyll-a using GPR')

#%% Plot reflectances
x_situ = x_situ[(x_situ['lat']>np.min(lat)) & (x_situ['lat']<np.max(lat))]
z = x_sat[(x_sat['NIR']<0.13) & (x_sat['NIR']>0.07)]
x = x_sat[x_sat['NIR']<0.07]
y = x_sat[x_sat['NIR']>0.13]
from matplotlib import colors as c
C = np.array(x_sat['NIR'] > 0.07)
C = C.astype(int)
C += np.array(x_sat['NIR'] > 0.13)

cMap = c.ListedColormap(['blue','orange','green'])

plotgrid(np.array(x_sat['blue']), title = 'Limnos Island (Blue band)', plot_insitu = False, var_insitu = 'blue', cmap = plt.get_cmap('Blues'))
plotgrid(np.array(x_sat['green']), title = 'Limnos Island (Green band)', plot_insitu = False, var_insitu = 'green', cmap = plt.get_cmap('Greens'))
plotgrid(np.array(x_sat['red']), title = 'Limnos Island (Red band)', plot_insitu = False, var_insitu = 'red', cmap = plt.get_cmap('Reds'))
plotgrid(np.array(x_sat['NIR']), title = 'Limnos Island (NIR band)', plot_insitu = False, var_insitu = 'NIR', cmap = plt.get_cmap('RdPu'))
plotgrid(C, title = 'Land Detection of Limnos Island', plot_insitu = False, var_insitu = 'NIR', cmap = cMap)

plt.rcParams.update({'font.size': 32})
plt.figure(figsize = (18,10));plt.hist(x['NIR'], bins = 17, log = True, label = 'Water'); 
plt.title('Near-infrared Reflectance'); plt.grid(True); 
plt.ylabel('observations'); plt.xlabel('Reflectance (ratio)')
plt.hist(z['NIR'], log = True, bins = 20, label = 'Coast'); 
plt.hist(y['NIR'], log = True, bins = 163, label = 'Land'); plt.legend()

#%% classification map sentinel-2
import matplotlib.patches as mpatches

plt.figure(figsize = (10,8))

Class = b.var_values['quality_scene_classification'][la1:la2,lo1:lo2]
#Class = b.var_values['quality_scene_classification'][la1+100:la2-200,lo1+450:lo2]

values = np.unique(Class.ravel())
labels = ['No Data', 'Defective', 'Dark Area', 'Cloud Shadows', 'Vegetation', 'Not Vegetated', 'Water', 'Unclassified', 'Medium Cloud Prob.', 'High Cloud Prob.', 'Thin Cirrus', 'Snow']
Labels = [labels[i] for i in values]
cm = 'terrain'

im = plt.imshow(Class, cmap = cm)

cmap = matplotlib.cm.get_cmap(cm)
colors = [ im.cmap(im.norm(value)) for value in values]
patches = [ mpatches.Patch(color=colors[i], label=Labels[i] ) for i in range(len(values)) ]
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

plt.show()

#%% classification map clouds
plt.figure(figsize = (12,8))

Class = b.var_values['quality_scene_classification'][0:400,0:800]
values = np.unique(Class.ravel())
values = list(range(12))
labels = ['No Data', 'Defective', 'Dark Area', 'Cloud Shadows', 'Vegetation', 'Not Vegetated', 'Water', 'Unclassified', 'Medium Cloud Prob.', 'High Cloud Prob.', 'Thin Cirrus', 'Snow']
Labels = [labels[i] for i in values]
cm = 'terrain'
cmap = matplotlib.cm.get_cmap(cm)

im = plt.imshow(Class, cmap = cmap)

colors = [ im.cmap(im.norm(value)) for value in values]
patches = [ mpatches.Patch(color=colors[i], label=Labels[i] ) for i in range(len(values)) ]
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

plt.show()