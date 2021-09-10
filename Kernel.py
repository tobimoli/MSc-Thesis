# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 10:17:42 2021

@author: molenaar
"""
#%% load packages
import GPy
import numpy as np
import matplotlib.pyplot as plt

from Function_file import plot_errcont, plotmatrix

def generate_noisy_points(n=10, noise_variance=1e-6):
    np.random.seed(4372859)
    X = np.random.uniform(-3., 3., (n, 1))
    y = np.sin(3*X) + np.random.randn(n, 1) * noise_variance**0.5
    return X, y

#%% plot Matérn

sigma_f = 1
d = np.linspace(0,3,101)
l = 1

GPR = sigma_f**2 * np.exp(- d**2/(2*l**2))
Mat32 = sigma_f**2 * (1 + np.sqrt(3)*np.abs(d)/l) * np.exp(-np.sqrt(3)*abs(d)/l)
Mat52 = sigma_f**2 * (1 + np.sqrt(5)*np.abs(d)/l + 5/3*(d/l)**2) * np.exp(-np.sqrt(5)*abs(d)/l)
Mat12 = sigma_f**2 * np.exp(- abs(d)/l)

plt.figure(figsize = (7,7))
plt.rcParams.update({'font.size': 18})
plt.plot(d, GPR, d, Mat12, '--', d, Mat32, '-.', d, Mat52, ':', linewidth = 3)
plt.grid(True)
plt.legend(['SE', 'Mat12', "Mat32", "Mat52"])
plt.xlabel("distance " +  r"$|x-x'|$")
plt.ylabel("covariance " + r"$k(x,x')$")
plt.show()

#%% GPy example
sigma_n, sigma_f, l = .05, 0.01, 1.5
X, y = generate_noisy_points(30,noise_variance=.1)

D = np.linspace(-35,35, 100)

kern1 = GPy.kern.RBF(1, sigma_f, l)
kern2 = GPy.kern.OU(1, sigma_f, l)
kern3 = GPy.kern.sde_Matern32(1, sigma_f, l)
kern4 = GPy.kern.sde_Matern52(1, sigma_f, l)
kern5 = GPy.kern.Linear(1, variances=sigma_f)
kern6 = GPy.kern.MLP(1, variance=sigma_f, weight_variance = 1, bias_variance = 1)

kernel = [kern1, kern2, kern3, kern4, kern5, kern6]

model = GPy.models.GPRegression(X,y,kernel[5]) 
model.Gaussian_noise.variance = sigma_n**2
pr = model.predict(D.reshape(1,100).T)
(a, b) = pr
mean, var = np.exp(a + 0.5*b), (np.exp(b)-1)*np.exp(2*a+b)
plot_errcont(X,y, pr, D, 'Input Variable')

#%% GPy optimize
model.optimize()
pr = model.predict(D.reshape(1,100).T)

plot_errcont(X,y, pr, D, 'Input Variable')

#%% Sample from the Gaussian process distribution
kern1 = GPy.kern.RBF(1, sigma_f, l)
kern2 = GPy.kern.OU(1, sigma_f, l)
kern3 = GPy.kern.sde_Matern32(1, sigma_f, l)
kern4 = GPy.kern.sde_Matern52(1, sigma_f, l)
kern5 = GPy.kern.Linear(1, variances=sigma_f)
kern6 = GPy.kern.MLP(1, variance=sigma_f, weight_variance = 1, bias_variance = 1)

kernel = [kern1, kern2, kern3, kern4, kern5, kern6]
nb_of_samples = 200  # Number of points in each function

X = np.expand_dims(np.linspace(-5, 5, nb_of_samples), 1)
#np.random.seed(1)

plt.figure(figsize=(7, 7))
plt.grid(True)
labels = ['SE', 'Mat12', "Mat32", "Mat52", 'Lin', 'MLP']
stls = ['-', '--', '-.', ':', '-', '--']
for i in range(5,len(kernel)):
    ys = np.random.multivariate_normal(mean=np.zeros(nb_of_samples), cov=kernel[i].K(X,X)) 
    plt.plot(X, ys, linestyle=stls[i], linewidth = 3, label = labels[i])
plt.xlabel('Input variable')
plt.ylabel('Output variable')
plt.rcParams.update({'font.size': 18})
plt.xlim([-3.5, 3.5])
plt.legend()
plt.show()

#%% Sample GPR using MLP
sigma_f = 1

kern1 = GPy.kern.MLP(1, variance=sigma_f, weight_variance = 1, bias_variance = 1)
kern2 = GPy.kern.MLP(1, variance=sigma_f, weight_variance = 9, bias_variance = 1)
kern3 = GPy.kern.MLP(1, variance=sigma_f, weight_variance = 100, bias_variance = 1)

kernel = [kern1, kern2, kern3]
q = 10
nb_of_samples = 200  # Number of points in each function

X = np.expand_dims(np.linspace(-3.5, 3.5, nb_of_samples), 1)
np.random.seed(10000)

plt.figure(figsize=(7, 7))
plt.grid(True)
labels = [r'$\sigma_v = 1$', r'$\sigma_v = 3$', r'$\sigma_v = 10$']
stls = ['-', '--', '-.']
for i in range(len(kernel)):
    ys = np.random.multivariate_normal(mean=np.zeros(nb_of_samples), cov=kernel[i].K(X,X)) 
    plt.plot(X, ys, linestyle=stls[i], linewidth = 3, label = labels[i])
plt.xlabel('Input variable')
plt.ylabel('Output variable')
plt.rcParams.update({'font.size': 18})
plt.xlim([X.min(), X.max()])
plt.legend()
plt.show()

#%% MLP covariance function
X = np.expand_dims(np.linspace(-3.5, 3.5, nb_of_samples//q), 1)
plotmatrix(kern1.K(X,X),'')
plt.xlabel("Input variable x")
plt.ylabel("Input variable x'")
plt.xticks([14/q, 43/q, 71/q, 100/q, 129/q, 157/q, 186/q], [-3, -2, -1, 0, 1, 2, 3], rotation = 0)
plt.yticks([14/q, 43/q, 71/q, 100/q, 129/q, 157/q, 186/q], [-3, -2, -1, 0, 1, 2, 3], rotation = 0)

#%% Mat32 for different noise, variance and lengthscale

sigma_f = 1
d = np.linspace(0,3,101)
l = 1

Mat1 = sigma_f**2 * (1 + np.sqrt(3)*np.abs(d)/l) * np.exp(-np.sqrt(3)*abs(d)/l)

l = 3
Mat2 = sigma_f**2 * (1 + np.sqrt(3)*np.abs(d)/l) * np.exp(-np.sqrt(3)*abs(d)/l)

l = 0.3
Mat3 = sigma_f**2 * (1 + np.sqrt(3)*np.abs(d)/l) * np.exp(-np.sqrt(3)*abs(d)/l)


plt.figure(figsize = (7,7))
plt.rcParams.update({'font.size': 18})
plt.plot(d, Mat1, d, Mat2, '--', d, Mat3, '-.', linewidth = 3)
plt.grid(True)
plt.legend(['$l = 1$', '$l = 5$', "$l = 0.05$"])
plt.xlabel("distance " +  r"$|x-x'|$")
plt.ylabel("covariance " + r"$k(x,x')$")
plt.show()


#%% Sample GPR using Mat32
sigma_f = 1

kern1 = GPy.kern.sde_Matern32(1, variance=sigma_f, lengthscale = 1)
kern2 = GPy.kern.sde_Matern32(1, variance=sigma_f, lengthscale = 3)
kern3 = GPy.kern.sde_Matern32(1, variance=sigma_f, lengthscale = 0.3)

kernel = [kern1, kern2, kern3]
q = 10
nb_of_samples = 200  # Number of points in each function

X = np.expand_dims(np.linspace(-3.5, 3.5, nb_of_samples), 1)
np.random.seed(10000)

plt.figure(figsize=(7, 7))
plt.grid(True)
labels = [r'$l = 1$', r'$l = 3$', r'$l = 0.3$']
stls = ['-', '--', '-.']
for i in range(len(kernel)):
    ys = np.random.multivariate_normal(mean=np.zeros(nb_of_samples), cov=kernel[i].K(X,X)) 
    plt.plot(X, ys, linestyle=stls[i], linewidth = 3, label = labels[i])
plt.xlabel('Input variable')
plt.ylabel('Output variable')
plt.rcParams.update({'font.size': 18})
plt.xlim([X.min(), X.max()])
plt.legend()
plt.show()

#%% GPy example 2
sigma_n, sigma_f, l = 1, 1, 2
X, y = generate_noisy_points(10,noise_variance=.1)
plt.rcParams.update({'font.size': 30})

D = np.linspace(-3,3, 100)

kern1 = GPy.kern.RBF(1, sigma_f, l)
kern2 = GPy.kern.OU(1, sigma_f, l)
kern3 = GPy.kern.sde_Matern32(1, sigma_f, l)
kern4 = GPy.kern.sde_Matern52(1, sigma_f, l)
kern5 = GPy.kern.Linear(1, variances=sigma_f)
kern6 = GPy.kern.MLP(1, variance=sigma_f, weight_variance = 1, bias_variance = 1)

kernel = [kern1, kern2, kern3, kern4, kern5, kern6]

gpr = GPy.models.GPRegression(X,y,kernel[2]) 
gpr.Gaussian_noise.variance = sigma_n**2

gpr.kern.lengthscale.fix()
gpr.optimize(messages = True)

pr = gpr.predict(D.reshape(1,100).T)
(a, b) = pr
mean, var = np.exp(a + 0.5*b), (np.exp(b)-1)*np.exp(2*a+b)
plot_errcont(X,y, pr, D, 'Input Variable', title = f"lengthscale $l = {l}$")

l_opt = gpr.Mat32.lengthscale.values
sigma_f_opt = np.sqrt(gpr.Mat32.variance.values[0])
sigma_n_opt = np.sqrt(gpr.Gaussian_noise.variance[0])
print(np.around(l_opt,3))
print(sigma_f_opt, sigma_n_opt)
