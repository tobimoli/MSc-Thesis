# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 15:32:04 2021

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
NAME = 'ODYSSEA.csv'
CITY = 'worldcities.csv'

data = pd.read_table(FOLDER+NAME, usecols = [2,4,5,6,8], sep = ';')
data.columns = ['time', 'lon', 'lat', 'dep', 'chl']
data['year'] = [int(n[6:10]) for n in data['time']]
data['date'] = [n[6:10]+n[2:5] + '-' + n[:2] for n in data['time']]
data['date'] = data['date'].astype("datetime64")

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
plt.scatter(data['lon'], data['chl'])
plt.show()
plt.figure()
plt.scatter(data['lat'], data['chl'])
plt.show()
plt.figure()
plt.scatter(data['dep'], data['chl'])
plt.show()
#--------------------------------------------

rows_to_drop = []
for i in range(len(data)):
    if i%1000 == 0:
        print(i)
    for j in range(i+1, min(len(data), i+100)):
        if data['lon'][i] == data['lon'][j] and data['lat'][i] == data['lat'][j] and data['date'][i] == data['date'][j]:
            if data['dep'][i] < data['dep'][j]:
                rows_to_drop.append(j)
            else:
                rows_to_drop.append(i)     
                
uudata = data.drop_duplicates(subset = ["lon", 'lat', 'date'])

uudata.to_csv(FOLDER+"ODYSSEA_unique.csv", index = False, header=True)
