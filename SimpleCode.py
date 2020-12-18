from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
warnings.filterwarnings("ignore", category=DeprecationWarning)

folder = 'C:/Users/molenaar/OneDrive - Stichting Deltares/Documents/Thesis - Deltares/Data'
name = "IMG_SW00_OPT_MS4_1C_20180612T092821_20180612T092823_TOU_1234_a2dc_R1C1"
#name = "IMG_SW00_OPT_MS4_1C_20180612T092819_20180612T092821_TOU_1234_ead8_R1C1"

file = folder + "/" + name + ".nc"
dataset = Dataset(file, "r")
lst = list(dataset.variables.keys())

var_values = {}
var_units = {}
var_names = {}
for key in lst:
    if key != 'lon' and key != 'lat':
        var_values[key] = dataset.variables[key][:]#ma.masked_array(ma.masked_values(dataset.variables[key][:],0), mask = dataset.variables['band_4'][:]>500)
    else:
        var_values[key] = dataset.variables[key][:]
    try:
        var_names[key] = dataset.variables[key].long_name
    except:
        var_names[key] = key
    try:
        var_units[key] = dataset.variables[key].units
    except:
        var_units[key] = 'nm'
    
plt.figure(1, figsize = (16,8))
plt.grid()
for key in lst:
    if key != 'lon' and key != 'lat':
        plt.hist(var_values[key].flatten(), bins=100, label = key, alpha = 0.5)
        plt.legend()
        plt.yscale('log', nonposy='clip')
        plt.plot()
        
#Blue: 450-520; Green: 520-590; Red: 630-690; NIR: 770-890
#Nanometers

for key in lst:
    #if key != 'lon' and key != 'lat':
    plt.figure()
    plt.matshow(var_values[key])
    plt.title(key)
    plt.colorbar()
    plt.plot()

fig, ax = plt.subplots(1, 4, figsize = (15,5))
i=0
for key in lst:
    if key != 'lon' and key != 'lat':
        ax[i].boxplot(var_values[key].flatten())
        ax[i].set_title(key)
        i+=1
plt.show()

df = pd.DataFrame()
for key in lst:
    df[key] = var_values[key].flatten()
df.corr()

import seaborn as sns
sns.heatmap(df.corr(), annot=True)




plt.figure()
plt.hist(df['band_2'], bins=10000)
plt.ylim([0,100000])
plt.show()


matplotlib.rcParams['figure.figsize'] = (10,10)
# Initialize map
proj=ccrs.Mercator()
m = plt.axes(projection=proj)
plt.rcParams.update({'font.size': 18})
# Plot data
plt.pcolormesh(var_values['lon'],var_values['lat'],var_values['band_4'], transform=ccrs.PlateCarree(), cmap = plt.get_cmap('viridis'))

#m.coastlines(resolution="10m")
#m.add_feature(cfeature.BORDERS)
#m.add_feature(cfeature.LAKES, edgecolor='black')
#m.add_feature(cfeature.RIVERS, edgecolor='blue')
#m.add_feature(cfeature.OCEAN)
#land_10m = cfeature.NaturalEarthFeature('physical', 'land', '10m',
#                                        facecolor=cfeature.COLORS['land'], zorder=0)
#m.add_feature(land_10m)
gl=m.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2,
                       color='gray', alpha=0.5, linestyle='--')
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
gl.top_labels = False
gl.right_labels = False

# Add Colorbar
cbar = plt.colorbar(fraction=0.03, pad=0.04)
cbar.set_label(var_units['band_4'], fontsize = 18)

# Add Title
plt.title(var_names['band_4'], fontsize = 18)
plt.show()