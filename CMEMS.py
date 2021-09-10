# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 12:02:32 2021

@author: tobias molenaar
"""
#%% load packages

import numpy as np
from Function_file import Data, plotgrid

#%% Import Data
folder = 'P:/11202428-hisea-inter/Tobias_Molenaar/01-Data/CMEMS'
#download this from CMEMS website
name = 'dataset-oc-eur-chl-multi-l3-chl_1km_daily-rt-v02_1630317736155'

b = Data(folder, name)
b.import_data()

#%% select area
q = 0.2; p = 4/3 * q
Lo1, Lo2, La1, La2 = 25, 25.40,39.75,40.05 #limnos
Lo1, Lo2, La1, La2 = Lo1-p, Lo2+p,La1-q,La2+q #limnos big
#Lo1, Lo2, La1, La2 = 25.0,25.30,40.30,40.50 #small map

lo1, lo2 = np.sum(b.var_values['lon']<Lo1), np.sum(b.var_values['lon']<Lo2)
la1, la2 = np.sum(b.var_values['lat']<La1), np.sum(b.var_values['lat']<La2)
coord = [lo1, lo2, la1, la2]

lon = np.tile(b.var_values['lon'][lo1:lo2], la2-la1)
lat = np.repeat(b.var_values['lat'][la1:la2], lo2-lo1)
chl = b.var_values['CHL'][0][la1:la2,lo1:lo2].flatten()

#%% Plot

plotgrid(chl, lon, lat, coord, title = 'Concentration Chlorophyll-a CMEMS')
