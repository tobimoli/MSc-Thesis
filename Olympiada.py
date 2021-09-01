# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 17:41:51 2021

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

FOLDER = 'P:/11202428-hisea-inter/Tobias_Molenaar/01-Data/In-Situ/'
NAME = 'Olympiada.csv'

data = pd.read_table(FOLDER+NAME, usecols = [0,1,3,6], sep = ',')
data.columns = ['time', 'oxy', 'temp', 'chl']
data['year'] = [int(n[0:4]) for n in data['time']]
data['month'] = [int(n[5:7]) for n in data['time']]
data['day'] = [int(n[8:10]) for n in data['time']]
data['date'] = [n[:10] for n in data['time']]
data['date'] = data['date'].astype("datetime64")
data['julian'] = list(range(len(data)))

#--------------------------------------------
matplotlib.rcParams['figure.figsize'] = (12,12)
# Initialize map
proj=ccrs.Mercator()
m = plt.axes(projection=proj)
plt.rcParams.update({'font.size': 18})
# Plot data
cmap = plt.cm.rainbow
norm = matplotlib.colors.Normalize()

plt.scatter(data['lon'], data['lat'], transform = ccrs.PlateCarree())

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#plt.colorbar(sm)

m.coastlines(resolution="10m")
m.add_feature(cfeature.OCEAN)
land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m', facecolor=cfeature.COLORS['land'], zorder=0)
m.add_feature(land_10m)

gl=m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.top_labels = False
gl.right_labels = False

plt.show()
#--------------------------------------------
plt.figure()
plt.scatter(data['oxy'], data['chl'], marker = '.', alpha = 0.1)
plt.show()
plt.figure()
plt.scatter(data['lat'], data['chl'])
plt.show()
#--------------------------------------------

plt.figure(figsize = (12,6))
plt.plot(data['date'], data['chl'])
plt.grid()
plt.ylabel('Chl-a concentration')
plt.show()