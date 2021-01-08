# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 16:36:33 2021

@author: molenaar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import matplotlib.colors

FOLDER = 'C:/Users/molenaar/OneDrive - Stichting Deltares/Documents/Thesis - Deltares/Data/'
NAME = 'PANGAEA.txt'
CITY = 'worldcities.csv'

data = pd.read_table(FOLDER+NAME)
city = pd.read_table(FOLDER+CITY, sep = ',')

def findindex(lon,lat):
    distance = (city['lat']-lat)**2 + (city['lng']-lon)**2
    return np.argmin(distance)

countries = []
cities = []
for i in range(len(data)):
    if i%1000 == 0:
        print(i)
    lat = data['Latitude'][i]
    lon = data['Longitude'][i]
    countries.append(city['country'][findindex(lon,lat)])
    cities.append(city['city'][findindex(lon,lat)])
    
data['country'] = countries
data['city'] = cities
data['year'] = [int(n[:4]) for n in data['Date/Time']]

matplotlib.rcParams['figure.figsize'] = (20,10)
# Initialize map
proj=ccrs.Mercator()
m = plt.axes(projection=proj)
plt.rcParams.update({'font.size': 18})
# Plot data
cmap = plt.cm.rainbow
norm = matplotlib.colors.Normalize(vmin=1988, vmax=2012) 
plt.scatter(data['Longitude'], data['Latitude'], c = cmap(norm(data['year'].values)), s=3, transform = ccrs.PlateCarree())

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
plt.colorbar(sm)

m.coastlines(resolution="110m")
m.add_feature(cfeature.OCEAN)
land_10m = cfeature.NaturalEarthFeature('physical', 'land', '110m', facecolor=cfeature.COLORS['land'], zorder=0)
m.add_feature(land_10m)

gl=m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.top_labels = False
gl.right_labels = False

plt.show()