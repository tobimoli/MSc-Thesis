# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 11:58:30 2021

@author: molenaar
"""
#%% load packages

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import GPy
import time

from Function_file import Data, simple_idw, plotgrid
#%% import data
folder = 'P:/11202428-hisea-inter/Tobias_Molenaar/01-Data/sentinel2'
name = "S2B_MSIL1C_20190809T090559_N0208_R050_T35TLE_20190809T112133_resampled"
name = "S2B_MSIL2A_20190809T090559_N0213_R050_T35TLE_20190809T121123_resampled"

b = Data(folder, name)
b.import_data()

#%% select area
Lo1, Lo2, La1, La2 = 25,25.40,39.80,40.00
lo1, lo2 = np.sum(b.var_values['lon'][0,:]<Lo1), np.sum(b.var_values['lon'][0,:]<Lo2)
la1, la2 = np.sum(b.var_values['lat'][:,0]>La2), np.sum(b.var_values['lat'][:,0]>La1)
coord = [lo1, lo2, la1, la2]

lon = b.var_values['lon'][la1:la2,lo1:lo2].flatten()
lat = b.var_values['lat'][la1:la2,lo1:lo2].flatten()
blue = b.var_values['B2'][la1:la2,lo1:lo2].flatten()
green = b.var_values['B3'][la1:la2,lo1:lo2].flatten()
red = b.var_values['B4'][la1:la2,lo1:lo2].flatten()
NIR = b.var_values['B8'][la1:la2,lo1:lo2].flatten()

x_sat = pd.DataFrame(np.array([lon,lat,blue,green,red,NIR]).T)
x_sat.columns = ['lon','lat','blue','green','red','NIR']

#%% Import in-situ data
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
#x_sat.loc[x_sat['NIR']>0.091,:] = np.nan

start = time.time()
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
    gpr.optimize()

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

plotgrid(prediction[0], lon, lat, coord, x_situ, 'Concentration Chlorophyll-a using GPR', plot_insitu=True, var_insitu = 'chl')
plotgrid(np.sqrt(prediction[1]), lon, lat, coord, x_situ, 'Std. Dev. of Concentration Chlorophyll-a using GPR')

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

plotgrid(np.array(x_sat['blue']), lon, lat, coord, x_situ, title = 'Limnos Island (Blue band)', plot_insitu = False, var_insitu = 'blue', cmap = plt.get_cmap('Blues'))
plotgrid(np.array(x_sat['green']), lon, lat, coord, x_situ, title = 'Limnos Island (Green band)', plot_insitu = False, var_insitu = 'green', cmap = plt.get_cmap('Greens'))
plotgrid(np.array(x_sat['red']), lon, lat, coord, x_situ, title = 'Limnos Island (Red band)', plot_insitu = False, var_insitu = 'red', cmap = plt.get_cmap('Reds'))
plotgrid(np.array(x_sat['NIR']), lon, lat, coord, x_situ, title = 'Limnos Island (NIR band)', plot_insitu = False, var_insitu = 'NIR', cmap = plt.get_cmap('RdPu'))
plotgrid(C, lon, lat, coord, x_situ, title = 'Land Detection of Limnos Island', plot_insitu = False, var_insitu = 'NIR', cmap = cMap)

plt.rcParams.update({'font.size': 32})
plt.figure(figsize = (18,10));plt.hist(x['NIR'], bins = 17, log = True, label = 'Water')
plt.title('Near-infrared Reflectance'); plt.grid(True)
plt.ylabel('observations'); plt.xlabel('Reflectance (ratio)')
plt.hist(z['NIR'], log = True, bins = 20, label = 'Coast')
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