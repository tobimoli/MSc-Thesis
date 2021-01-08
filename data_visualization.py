# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 10:22:27 2020

@author: molenaar
"""
import numpy as np
#import cv2
from netCDF4 import Dataset
import numpy.ma as ma
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import seaborn as sns
import cartopy.crs as ccrs
from cartopy.io import shapereader
#import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
warnings.filterwarnings("ignore", category=DeprecationWarning)

folder = 'C:/Users/molenaar/OneDrive - Stichting Deltares/Documents/Thesis - Deltares/Data'
name = "IMG_SW00_OPT_MS4_1C_20180612T092821_20180612T092823_TOU_1234_a2dc_R1C1"
#name = "IMG_SW00_OPT_MS4_1C_20180612T092819_20180612T092821_TOU_1234_ead8_R1C1"

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
            if key != 'lon' and key != 'lat':
                self.var_values[key] = ma.masked_array(ma.masked_values(dataset.variables[key][:],0), mask = dataset.variables['band_4'][:]>500)
            else:
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
    def hist(self, var):
        self.var = var
        plt.figure(1, figsize = (10,5))
        plt.grid()
        plt.hist(self.var_values[self.var].flatten(), bins=100, label = self.var, alpha = 0.5)
        plt.legend()
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
        
        shp = shapereader.Reader('C:/Users/molenaar/Downloads/gadm36_GRC_shp/gadm36_GRC_0')
        for record, geometry in zip(shp.records(), shp.geometries()):
            m.add_geometries([geometry], ccrs.PlateCarree(), alpha = 0.5, facecolor='lightgray', edgecolor='black')

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
        MAX = [np.max(self.var_values['band_1']), np.max(self.var_values['band_2']), np.max(self.var_values['band_3'])]
        for i in range(self.sub[1] - self.sub[0]):
            for j in range(self.sub[3] - self.sub[2]):
                retrack_original[i][j][0] = self.var_values['band_1'].data[i+self.sub[0],j+self.sub[2]]/MAX[0]*255
                retrack_original[i][j][1] = self.var_values['band_2'].data[i+self.sub[0],j+self.sub[2]]/MAX[1]*255
                retrack_original[i][j][2] = self.var_values['band_3'].data[i+self.sub[0],j+self.sub[2]]/MAX[2]*255
        plt.imshow(retrack_original)

b = Data([200,330,2450,2600])
#b = Data([200,600,2400,2800])
b.import_data()
b.masktotal()
b.dataframe()

#b.hist()
#b.boxplot()
#b.plot('band_1')

