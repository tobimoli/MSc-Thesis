# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 14:40:51 2021

@author: molenaar
"""

import numpy as np
import math as m
from numpy.linalg import cholesky, inv
import GPy
import matplotlib.pyplot as plt
import time

sqrt_2_pi = np.sqrt(2*np.pi)

def Norm(x):
    return np.exp(-x**2/2)/sqrt_2_pi
def Phi(x):
    return .5*(1+ m.erf(x/np.sqrt(2)))
def converged(x,y,eps):
    if np.abs(np.sum(x-y)) < eps:
        return True
    else:
        return False
#%% input
N = 201
kern = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=8, ARD=False)
X = np.linspace(0,100,N)[:,None]
K = kern.K(X)
y = np.random.multivariate_normal(np.zeros(N),kern.K(X)+np.eye(N)*np.sqrt(.1)).reshape(-1,1)

X_star = np.linspace(1,100,1001)[:,None]

#%% GPR
sigma_f = 0.1
l = 1
sigma_n = 1
start = time.time()
rbf = GPy.kern.RBF(input_dim=1, variance=sigma_f, lengthscale=l, ARD = False)
m_full = GPy.models.GPRegression(X,y, kernel = kern)
m_full.Gaussian_noise.variance = sigma_n**2
m_full.optimize(max_iters = 100, messages = True)
end = time.time()
GP1 = end - start
start = time.time()

prediction = m_full.predict(X_star, full_cov= False, include_likelihood= True)

end = time.time()
GP2 = end - start
plt.figure(figsize = (12,8))
m_full.plot()
plt.show()
#%% initialization
variance = .1

nu_tilde = np.zeros((N,1))
tau_tilde = np.zeros((N,1))
Sigma = K.copy()
mu = np.zeros((N,1))

tau_min = np.zeros((N,1))
nu_min = np.zeros((N,1))
z = np.zeros((N,1))
mu_hat = np.zeros((N,1))
sigma_hat = np.zeros((N,1))
Sigma_old = np.zeros((N,N))

#%% repeat until convergence
step = 0
eta = 1
eps = 1e-3
delta = 1
printvalues = False
convergence = False

start = time.time()
while not convergence and step<100:
    step += 1
    print(step)
    print(np.abs(np.sum(Sigma_old-Sigma)))
    Sigma_old = Sigma.copy()
    
    update_order = np.random.permutation(N)
    for i in update_order:
        #3.56 cavity update i
        #Cavity distribution parameters
        tau_min[i] = 1/Sigma[i,i] - eta * tau_tilde[i]
        nu_min[i] = mu[i]/Sigma[i,i] - eta * nu_tilde[i]
        
        #Marginal moments Gaussian
        #sigma_hat[i] = 1./(1./variance + tau_min[i])
        #mu_hat[i] = sigma_hat[i]*(y[i]/variance + nu_min[i])
        #sum_var = variance + 1/tau_min[i]
        #Z_hat = 1./(sqrt_2_pi*np.sqrt(sum_var))*np.exp(-.5*(y[i]-nu_min[i]/tau_min[i])**2./sum_var)
        
        #3.58
        z[i] = y[i]* nu_min[i] / np.sqrt(tau_min[i]* (tau_min[i] + 1))
        mu_hat[i] = nu_min[i]/tau_min[i] + y[i]* Norm(z[i])/(Phi(z[i])*np.sqrt(tau_min[i]*(tau_min[i]+1)))
        sigma_hat[i] = 1/tau_min[i] - Norm(z[i])*(z[i] + Norm(z[i])/Phi(z[i]))/(tau_min[i]*(tau_min[i]+1)*Phi(z[i]))
        
        #3.59
        #delta_tau = delta/eta * (1/sigma_hat[i] - 1/Sigma[i,i])
        #delta_nu = delta/eta * (mu_hat[i]/sigma_hat[i] - mu[i]/Sigma[i,i])
        #tau_tilde_prev = tau_tilde[i]
        #tau_tilde[i] += delta_tau
        #if tau_tilde[i]< eps: #make sure tau_tilde>0
        #    tau_tilde[i] = eps
        #    delta_tau = tau_tilde[i] - tau_tilde_prev
        #    print('tau_tilde smaller than eps')
        #nu_tilde[i] += delta_nu
        
        delta_tau = 1/sigma_hat[i] - tau_min[i] - tau_tilde[i]
        tau_tilde[i] += delta_tau
        nu_tilde[i] = mu_hat[i]/sigma_hat[i] - nu_min[i]
        if printvalues:
            print("tau- = " + str(tau_min[i]))
            print("nu- = " + str(nu_min[i]))
            print("Sigma = " + str(Sigma[i,i]))
            print("z = " + str(z[i]))
            print("mu^ = " + str(mu_hat[i]))
            print("sigma^ = " + str(sigma_hat[i]))
            print(f"d_tau~ = {delta_tau}")
            print("tau~ = " + str(tau_tilde[i]))
            print("nu~ = " + str(nu_tilde[i]))
            print("-------------"+str(i)+"---------------")
        #3.53 & 3.70
        Sigma -= np.array([Sigma[:,i]]) * np.array([Sigma[:,i]]).T/(1/delta_tau + Sigma[i,i])
        mu = np.dot(Sigma, nu_tilde)
    S_tilde12 = np.diag(np.sqrt(tau_tilde.flatten()))
    L = cholesky(np.eye(N) + np.dot(np.dot(S_tilde12, K),S_tilde12))
    Linv = inv(L)
    V = np.dot(Linv, np.dot(S_tilde12, K))
    Sigma = K - np.dot(V.T, V)
    mu = np.dot(Sigma, nu_tilde)
    convergence = converged(Sigma_old, Sigma, 10**(-8))
end = time.time()
EP1 = end - start
#%% Prediction

start = time.time()
V = np.dot(Linv, S_tilde12)
SBS = np.dot(V.T, V)
Q = np.dot(np.eye(N) - np.dot(SBS, K), nu_tilde)
PR = np.zeros(len(X_star))
VR = np.zeros(len(X_star))
K_star = kern.K(X,X_star)
PR = np.dot(K_star.T, Q)
VR = np.array([kern.Kdiag(X_star) - np.sum(np.multiply(np.dot(K_star.T, SBS).T, K_star), axis = 0)]).T

end = time.time()
EP2 = end - start

plt.figure(figsize = (12,8))
plt.plot(X_star, prediction[0], 'g', label = 'Exact')
plt.plot(X_star, prediction[0] + 2* np.sqrt(prediction[1]), 'g--')
plt.plot(X_star, prediction[0] - 2* np.sqrt(prediction[1]), 'g--')
plt.grid(False)
plt.rcParams.update({'font.size': 18})

plt.scatter(X, y, color = 'k', marker = 'x', alpha = 0.5, label = 'Observations')
plt.xlabel('Input variable x')
plt.ylabel('Output variable y')
plt.plot(X_star, PR, 'r', label = "EP")
plt.plot(X_star, PR + 2* np.sqrt(VR), 'r--')
plt.plot(X_star, PR - 2* np.sqrt(VR), 'r--')
plt.legend()
plt.show()

print(f"GP cost {GP1} seconds + {GP2} seconds.")
print(f"EP cost {EP1} seconds + {EP2} seconds.")