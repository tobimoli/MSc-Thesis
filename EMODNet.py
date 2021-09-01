# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 09:21:50 2021

@author: molenaar
"""

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

FOLDER = 'P:/11202428-hisea-inter/Tobias_Molenaar/01-Data/In-Situ/'
NAME = 'EMODNet.csv'
#NAME = 'ODYSSEA.csv'
CITY = 'worldcities.csv'

if NAME == 'ODYSSEA.csv':
    data = pd.read_table(FOLDER+NAME, sep = ';')
else:
    data = pd.read_table(FOLDER+NAME, sep = ';')
#city = pd.read_table(FOLDER+CITY, sep = ',')

#data = data[data['Water body chlorophyll-a [mg/m^3]'].notnull()]
#data = data[data['time_ISO8601'].notnull()]
#data = data.fillna(method='ffill')
if NAME == 'ODYSSEA.csv':
    data['year'] = [int(n[6:10]) for n in data['PLD_REALTIMECLOCK']]
    data['date'] = [n[6:10]+n[2:5] + '-' + n[:2] for n in data['PLD_REALTIMECLOCK']]
else:
    data['year'] = [int(n[:4]) for n in data['time_ISO8601']]
    data['date'] = [n[:10] for n in data['time_ISO8601']]
data['date'] = data['date'].astype("datetime64")
#data = data[data['year']>2013]

def findindex(lon,lat):
    distance = (city['lat']-lat)**2 + (city['lng']-lon)**2
    return np.argmin(distance)

#countries = []
#cities = []
#for i in data['Latitude [degrees_north]'].index:
#    if i%10000 < 100:
#        print(i)
#    lat = data['Latitude [degrees_north]'][i]
#    lon = data['Longitude [degrees_east]'][i]
#    countries.append(city['country'][findindex(lon,lat)])
#    cities.append(city['city'][findindex(lon,lat)])
    

#data = data[(data['Longitude [degrees_east]']<29) & (data['Longitude [degrees_east]']>5) & (data['Latitude [degrees_north]']<45) & (data['Latitude [degrees_north]']>36)]
#data = data[(data['Longitude [degrees_east]']<20) & (data['Longitude [degrees_east]']>15) & (data['Latitude [degrees_north]']<43) & (data['Latitude [degrees_north]']>40)]
matplotlib.rcParams['figure.figsize'] = (12,12)
# Initialize map
proj=ccrs.Mercator()
m = plt.axes(projection=proj)
plt.rcParams.update({'font.size': 18})
# Plot data
cmap = plt.cm.rainbow
norm = matplotlib.colors.Normalize()
if NAME == "ODYSSEA.csv":
    plt.scatter(data['NAV_LONGITUDE'], data['NAV_LATITUDE'], c = cmap(norm(data['year'].values)), s=30, transform = ccrs.PlateCarree())
else:
    plt.scatter(data['Longitude [degrees_east]'], data['Latitude [degrees_north]'], c = 'red', s=30, transform = ccrs.PlateCarree())

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


#print(data['time_ISO8601'].value_counts()[:10].index.tolist())
#data.to_csv(FOLDER+"EMODNet.csv", index = False, header=True)
plt.figure(figsize = (10,10))
plt.hist(data['Longitude [degrees_east]'], bins = 100)
plt.xticks(rotation=90)
plt.show()