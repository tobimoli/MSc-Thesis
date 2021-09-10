# -*- coding: utf-8 -*-
"""
Created on Fri May  7 10:09:22 2021

@author: molenaar
"""
#%% load packages

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from Function_file import Data, plotgrid
#%% import data
folder = 'P:/11202428-hisea-inter/Tobias_Molenaar/01-Data/sentinel2'
name = "S2B_MSIL1C_20190809T090559_N0208_R050_T35TLE_20190809T112133_resampled"
name = "S2B_MSIL2A_20190809T090559_N0213_R050_T35TLE_20190809T121123_resampled"

b = Data(folder, name)
b.import_data()

#%% select area
Lo1, Lo2, La1, La2 = 25.10, 25.12, 40.33, 40.34
#Lo1, Lo2, La1, La2 = 25.10, 25.14, 40.33, 40.36
lo1, lo2 = np.sum(b.var_values['lon'][0,:]<Lo1), np.sum(b.var_values['lon'][0,:]<Lo2)
la1, la2 = np.sum(b.var_values['lat'][:,0]>La2), np.sum(b.var_values['lat'][:,0]>La1)
coord = [lo1,lo2,la1,la2]

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
b.RGB(subset = [la1,la2,lo1,lo2])
#%%Interpolation

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
def nni(x, y, z, xi, yi):
    dist = distance_matrix(x,y, xi,yi)
    index = np.argmin(dist, axis = 0)
    zi = z[index,:]
    return zi

from sklearn.model_selection import KFold
from scipy.interpolate import NearestNDInterpolator
from scipy import interpolate
from scipy.interpolate import Rbf
from scipy.interpolate import griddata

X = np.array(x_sat)
K = 2
kf = KFold(n_splits=K, shuffle = True, random_state = 0)
epsilon = 1e-5 
mse = np.zeros((5,4))
i = 0; p = 3
for train, test in kf.split(X):
    print(i)
    x, y, z = X[train,0], X[train,1], X[train,2:6]
    x_new, y_new = X[test,0], X[test,1]
    
    start = time.time()
    #nearest neighbour
    interp = NearestNDInterpolator(list(zip(x, y)), z)
    Z = interp(x_new, y_new)
    mse[0,:] += np.mean((Z-X[test,2:6])**2, axis = 0)
    
    end = time.time()
    print(end - start)
    start = time.time()
    
    #IDW  
    dist = distance_matrix(x, y, x_new, y_new)
    weights = 1/(dist**p)
    weights /= weights.sum(axis=0)
    idw = np.dot(weights.T, z)    
    mse[1,:] += np.mean((idw-X[test,2:6])**2, axis = 0)
    
    end = time.time()
    print(end - start)
    start = time.time()
    
    #Cubic Spline
    Z = griddata(np.array([x,y]).T, z, (x_new, y_new), method='linear')
    mse[2,:] += np.nanmean((Z-X[test,2:6])**2, axis = 0)
    
    Z = griddata(np.array([x,y]).T, z, (x_new, y_new), method='cubic')
    mse[3,:] += np.nanmean((Z-X[test,2:6])**2, axis = 0)

    #RBF
    end = time.time()
    print(end - start)
    start = time.time()
    for j in range(4):
        rbf = Rbf(x, y, z[:,j], function = 'linear', epsilon = epsilon)
        Z = rbf(x_new, y_new)
        mse[4,j] += np.mean((Z-X[test,2+j])**2, axis = 0)
    end = time.time()
    print(end - start)
    i+=1 
mse = mse/K
print(mse)

# plt.figure(figsize = (8,8))
# plt.rcParams.update({'font.size': 26})
# plt.title('Gaussian')
# plt.loglog(EPS, mse[:,0], 'bo-', label = 'blue') 
# plt.loglog(EPS, mse[:,1], 'go-', label = 'green')
# plt.loglog(EPS, mse[:,2], 'ro-', label = 'red') 
# plt.loglog(EPS, mse[:,3], 'mo-', label = 'NIR')
# plt.legend()
# plt.grid(True)
# plt.ylabel(r'$MSE_{(10)}(\epsilon)$')
# plt.xlabel(r'$\epsilon$')
# plt.show()
# rows = ['NNI', 'IDW', 'Lin Spline', 'Cub Spline', 'RBF']

df=pd.DataFrame(mse,columns=["blue","green","red","NIR"], index = ['NNI', 'IDW', 'Lin Spline', 'Spline', 'RBF'])
df = df.drop('Lin Spline')
df.plot(kind="bar",figsize=(9,8), color = ['b', 'g', 'r', 'm'], rot = 0, alpha = 0.7)
plt.grid(True)
plt.show()

Q = np.zeros(len(X))
Q[test] = np.ones(len(test))
plotgrid(Q, lon, lat, coord, x_situ)
# start = time.time()
# end = time.time()
# print(end - start)


#%% Plot reflectances
x_situ = x_situ[(x_situ['lat']>np.min(lat)) & (x_situ['lat']<np.max(lat))]

plotgrid(np.array(x_sat['blue']), lon, lat, coord, x_situ, title = 'Interpolation of the blue wavelength using IDW' ,plot_insitu = True, var_insitu = 'blue', cmap = 'Blues')
plotgrid(np.array(x_sat['green']), lon, lat, coord, x_situ, title = 'Interpolation of the green wavelength using IDW' ,plot_insitu = False, var_insitu = 'green', cmap = 'Greens')
plotgrid(np.array(x_sat['red']), lon, lat, coord, x_situ, title = 'Interpolation of the red wavelength using IDW' ,plot_insitu = False, var_insitu = 'red', cmap = 'Reds')
plotgrid(np.array(x_sat['NIR']), lon, lat, coord, x_situ, title = 'Interpolation of the NIR wavelength using IDW' ,plot_insitu = False, var_insitu = 'NIR', cmap = 'RdPu')

#%% Nearest Neighbour Interpolation
from scipy.interpolate import NearestNDInterpolator

np.random.seed(0)
x = np.linspace(0, 1, 5)
y = np.linspace(0, 1, 5)
x, y = np.meshgrid(x, y)
x, y = x.flatten(), y.flatten()
z = np.random.random(25)
X = np.linspace(0, 1, 101)
Y = np.linspace(0, 1, 101)
X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation

interp = NearestNDInterpolator(list(zip(x, y)), z)
Z = interp(X, Y)
plt.figure(figsize = (12,10))
plt.pcolormesh(X, Y, Z, shading='auto')
plt.rcParams.update({'font.size': 18})
plt.plot(x, y, "ok")
plt.colorbar()
plt.clim(0,1)
plt.show()

#%% Inverse Distance Weighting
p = 4

np.random.seed(0)
x = np.linspace(0, 1, 5)
y = np.linspace(0, 1, 5)
x, y = np.meshgrid(x, y)
x, y = x.flatten(), y.flatten()
z = np.random.random(25)
X = np.linspace(0, 1, 101)
Y = np.linspace(0, 1, 101)
X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation

dist = distance_matrix(x, y, X.flatten(), Y.flatten())
weights = 1/(dist**p + .1**(10))
weights /= weights.sum(axis=0)
Z = np.dot(weights.T, z)

plt.figure(figsize = (12,10))
plt.pcolormesh(X, Y, Z.reshape(101,101), shading='auto')
plt.rcParams.update({'font.size': 18})
plt.plot(x, y, "ok")
plt.colorbar()
plt.clim(0,1)
plt.show()

#%% Bilinear Interpolation

def bilinear_interpolation(x, y, points):
    '''Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.
    '''
    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= x <= x2 or not y1 <= y <= y2:
        raise ValueError('(x, y) not within the rectangle')

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1))

def bilinear(x, y, z, X, Y):
    Z = np.zeros(len(X)*len(Y))
    k = 0
    for j in Y:
        for i in X:
            in_x = np.sum(x < i)
            in_y = np.sum(y < j)
            points = [(x[in_x], y[in_y], z[5*in_y + in_x]),
                      (x[in_x-1], y[in_y], z[5*in_y + in_x-1]),
                      (x[in_x], y[in_y-1], z[5*in_y-5 + in_x]),
                      (x[in_x-1], y[in_y-1], z[5*in_y-5 + in_x-1])]
            Z[k] = bilinear_interpolation(i, j, points)
            k+=1
    return Z

np.random.seed(0)
x = np.linspace(0, 1, 5)
y = np.linspace(0, 1, 5)
z = np.random.random(25)
X = np.linspace(0, 1, 101)
Y = np.linspace(0, 1, 101)

Z = bilinear(x,y,z, X,Y)
x, y = np.meshgrid(x, y)
x, y = x.flatten(), y.flatten()

plt.figure(figsize = (12,10))
plt.pcolormesh(X, Y, Z.reshape(101,101), shading='auto')
plt.rcParams.update({'font.size': 18})
plt.plot(x, y, "ok")
plt.colorbar()
plt.clim(0,1)
plt.show()

#imshow manier
plt.figure(figsize = (12,10))
plt.plot(x, y, "ok")
plt.imshow(z.reshape(5,5), interpolation='bilinear', cmap='viridis', extent = [-0.125, 1.125, 1.125, -0.125])
plt.xlim([0,1])
plt.ylim([0,1])
plt.colorbar()
plt.clim(0,1)
plt.show()

#%% Bicubic Interpolation
np.random.seed(0)
x = np.linspace(0, 1, 5)
y = np.linspace(0, 1, 5)
x, y = np.meshgrid(x, y)
x, y = x.flatten(), y.flatten()
z = np.random.random(25)
X = np.linspace(0, 1, 101)
Y = np.linspace(0, 1, 101)

plt.figure(figsize = (12,10))
plt.plot(x, y, "ok")
plt.imshow(z.reshape(5,5), interpolation='bicubic', cmap='viridis', extent = [-0.125, 1.125, 1.125, -0.125])
plt.xlim([0,1])
plt.ylim([0,1])
plt.colorbar()
plt.clim(0,1)
plt.show()

#%% Bilinear Spline Interpolation
from scipy.interpolate import griddata

np.random.seed(0)
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
x, y = np.meshgrid(x, y)
x, y = x.flatten(), y.flatten()
z = np.random.random(10000)
X = np.linspace(0, 1, 11)
Y = np.linspace(0, 1, 11)

grid_x, grid_y = np.meshgrid(X, Y)
grid_x, grid_y = grid_x.flatten(), grid_y.flatten()
points = np.array([x,y]).T
values = z

grid_z2 = griddata(points, values, (grid_x, grid_y), method='linear')

plt.figure(figsize = (12,10))
plt.pcolormesh(X, Y, grid_z2.reshape(11,11), shading='auto')
plt.rcParams.update({'font.size': 18})
#plt.plot(x, y, "ok")
plt.colorbar()
plt.clim(0,1)
plt.show()
#%% Bicubic Spline Interpolation

np.random.seed(0)
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
x, y = np.meshgrid(x, y)
x, y = x.flatten(), y.flatten()
z = np.random.random(10000)
X = np.linspace(0, 1, 11)
Y = np.linspace(0, 1, 11)

grid_x, grid_y = np.meshgrid(X, Y)
grid_x, grid_y = grid_x.flatten(), grid_y.flatten()
points = np.array([x,y]).T
values = z

grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')

plt.figure(figsize = (12,10))
plt.pcolormesh(X, Y, grid_z2.reshape(11,11), shading='auto')
plt.rcParams.update({'font.size': 18})
plt.plot(x, y, "ok")
plt.colorbar()
plt.clim(0,1)
plt.show()

#%% Cubic Spline Interpolation 
from scipy import interpolate

np.random.seed(0)
x = np.linspace(0, 1, 5)
y = np.linspace(0, 1, 5)
x, y = np.meshgrid(x, y)
x, y = x.flatten(), y.flatten()
z = np.random.random(25)
X = np.linspace(0, 1, 101)
Y = np.linspace(0, 1, 101)
grid_x, grid_y = np.meshgrid(X, Y)
grid_x, grid_y = grid_x.flatten(), grid_y.flatten()

tck = interpolate.bisplrep(x, y, z, s=0, kx = 3, ky = 3)
znew = interpolate.bisplev(X, Y, tck)

f = interpolate.interp2d(x, y, z, kind='cubic')
znew = f(X, Y).T

Q = np.array([f(xi, yi)[0] for xi, yi in zip(grid_x, grid_y)])

plt.figure(figsize = (12,10))
plt.pcolormesh(X, Y, Q.reshape(101,101), shading='auto')
plt.rcParams.update({'font.size': 18})
plt.plot(x, y, "ok")
plt.colorbar()
plt.clim(0,1)
plt.show()

#%% RBF interpolation
from scipy.interpolate import Rbf

np.random.seed(0)
x = np.linspace(0, 1, 5)
y = np.linspace(0, 1, 5)
x, y = np.meshgrid(x, y)
x, y = x.flatten(), y.flatten()
z = np.random.random(25)
X = np.linspace(0, 1, 101)
Y = np.linspace(0, 1, 101)
grid_x, grid_y = np.meshgrid(X, Y)

start = time.time()
def fun(self, r):
    return (r+0.01)**(2)
rbf = Rbf(x, y, z, function = fun, epsilon = 1)

end = time.time()
print(end - start)
start = time.time()

Z = rbf(grid_x, grid_y)

end = time.time()
print(end - start)

plt.figure(figsize = (12,10))
plt.pcolormesh(X, Y, Z.reshape(101,101), shading='auto')
plt.rcParams.update({'font.size': 18})
plt.plot(x, y, "ok")
plt.colorbar()
plt.clim(0,1)
plt.show()