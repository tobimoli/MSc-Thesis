# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 10:21:43 2021

@author: molenaar
"""

import pandas as pd
from pykrige.uk import UniversalKriging
from pykrige.ok import OrdinaryKriging
import numpy as np
from netCDF4 import Dataset
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import seaborn as sns
import numpy.ma as ma
import cartopy.feature as cfeature

#--------------------------------------------
FOLDER = 'P:/11202428-hisea-inter/Tobias_Molenaar/01-Data/In-Situ/'
NAME = 'ODYSSEA_unique.csv'
data = pd.read_table(FOLDER+NAME, sep = ',')

data['date'] = data['date'].astype("datetime64")

data = data[data['date'] == '2019-08-02']
#--------------------------------------------
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

Lo1, Lo2, La1, La2 = 25.0,25.25,40.30,40.38
lo1, lo2 = np.sum(b.var_values['lon'][0,:]<Lo1+0.1*(Lo2-Lo1)), np.sum(b.var_values['lon'][0,:]<Lo2-0.1*(Lo2-Lo1))
la1, la2 = np.sum(b.var_values['lat'][:,0]>La2-0.2*(La2-La1)), np.sum(b.var_values['lat'][:,0]>La1+0.1*(La2-La1))

lon = b.var_values['lon'][la1:la2,lo1:lo2].flatten()
lat = b.var_values['lat'][la1:la2,lo1:lo2].flatten()
blue = b.var_values['B2'][la1:la2,lo1:lo2].flatten()
green = b.var_values['B3'][la1:la2,lo1:lo2].flatten()
red = b.var_values['B4'][la1:la2,lo1:lo2].flatten()
NIR = b.var_values['B8'][la1:la2,lo1:lo2].flatten()
#--------------------------------------------

mydata = np.array([data['lon'], data['lat'], data['chl']]).T

#UK = UniversalKriging(mydata[:, 0],mydata[:, 1],mydata[:,2],variogram_model="linear",drift_terms=["regional_linear"])
OK = OrdinaryKriging(mydata[:, 0], mydata[:, 1], mydata[:, 2], variogram_model='linear', verbose=False, enable_plotting=False)
z, ss = OK.execute("points", lon, lat)

#--------------------------------------------
matplotlib.rcParams['figure.figsize'] = (10,10)
# Initialize map
proj=ccrs.Mercator()
m = plt.axes(projection=proj)
plt.rcParams.update({'font.size': 18})
# Plot data
plt.pcolormesh(b.var_values['lon'][la1:la2,lo1:lo2],
               b.var_values['lat'][la1:la2,lo1:lo2],
               z.reshape(la2-la1,lo1-lo2), 
               transform=ccrs.PlateCarree(), cmap = plt.get_cmap('viridis'))

m.set_extent([Lo1,Lo2,La1,La2])
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
plt.clim(0,0.3)
cbar = plt.colorbar(fraction=0.03, pad=0.04)
cbar.set_label('mg', fontsize = 18)

# Add Title
plt.title('Concentration Chlorophyll-a', fontsize = 18)
plt.show()

#--------------------------------------------
matplotlib.rcParams['figure.figsize'] = (10,10)
# Initialize map
proj=ccrs.Mercator()
m = plt.axes(projection=proj)
plt.rcParams.update({'font.size': 18})
# Plot data
plt.scatter(data['lon'], data['lat'], c = data['chl'], s = 150, marker = '+', transform = ccrs.PlateCarree())

m.set_extent([Lo1,Lo2,La1,La2])
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
plt.clim(0,0.3)
cbar = plt.colorbar(fraction=0.03, pad=0.04)
cbar.set_label('mg', fontsize = 18)

# Add Title
plt.title('Concentration Chlorophyll-a', fontsize = 18)
plt.show()

#--------------------------------------------
from sklearn.linear_model import LinearRegression

X = np.array([np.log(NIR/blue), np.log(NIR/blue)**2, np.log(NIR/blue)**3]).T
reg = LinearRegression().fit(X, z)
print(reg.score(X, z))
x = reg.coef_
I = reg.intercept_
plt.scatter(np.matmul(X,x)+I, z, alpha = 0.1)
plt.plot([0,1],[0,1], 'r-')
plt.show()

df = pd.DataFrame(data = np.array([z,NIR/blue,NIR/green,NIR/red,red/blue]).T, columns = ['chl', 'nir/blue', 'nir/green', 'nir/red', 'red/blue'])
# correlation matrix
sns.pairplot(df, kind="hist")
plt.show()
