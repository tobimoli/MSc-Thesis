import numpy as np
import GPy
import time
import matplotlib.pyplot as plt
from numpy.linalg import inv, cholesky, norm
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh, solve

#%% set initial values
np.random.seed(101)

N = 1001
M_range = list(range(10,100,10))

bericht = True

#%% create data
results = np.zeros((3,len(M_range)+1))
noise_var = 0.1
X = np.linspace(0,100,N)[:,None]
X_star = np.linspace(-20,120,2001)[:,None]

# Sample response variables from a Gaussian process with exponentiated quadratic covariance.
kern = GPy.kern.RBF(1, lengthscale = 10)
y = np.random.multivariate_normal(np.zeros(N),kern.K(X)+np.eye(N)*noise_var).reshape(-1,1)

#%% plot prediction with GPR (GPy)
s_full = time.time()

kern = GPy.kern.RBF(1, variance = 1,  lengthscale = 1)
m_full = GPy.models.GPRegression(X,y, kernel = kern)
m_full.Gaussian_noise.variance = 1
_ = m_full.optimize(messages=bericht) # Optimize parameters of covariance function

e_full = time.time()

prediction = m_full.predict(X_star, full_cov= False, include_likelihood= True)
m_full.plot()
plt.title("n=1001")

results[:,-1] = N, m_full.log_likelihood(), e_full-s_full
#%% Plot prediction with Sparse GPR (GPy)
i = 0
for M in M_range:
    print(M)
    s = time.time()
    
    Z = np.random.rand(M,1)*100
    m = GPy.models.SparseGPRegression(X,y,kernel=kern, Z = Z)
    m.noise_var = 1
    _ = m.optimize(messages=bericht)
    
    e = time.time()
    if M == 10:
        PR = m.predict(X_star, full_cov= False, include_likelihood= True)
    m.plot()
    plt.title(f'm = {M}')
    
    results[:,i] = M, m.log_likelihood()[0][0], e-s
    i += 1

#plt.plot(results[0],results[1], '.'); plt.xlabel("M"); plt.ylabel("Log Likelihood")
#plt.plot(results[0],results[2], '.'); plt.xlabel("M"); plt.ylabel("time [s]")
#plt.plot(results[1],results[2], '.'); plt.xlabel("Log Likelihood"); plt.ylabel("time [s]")

#%% make a plot with different y-axis using second axis object
x = list(range(len(M_range)+1))

fig,ax = plt.subplots(figsize=(8,6))
plt.rcParams.update({'font.size': 18})
ax.plot(x, results[1], color="red", marker="o")
ax.set_xlabel("Number of inducing points m")
ax.set_ylabel("Log Likelihood",color="red")
ax2=ax.twinx()
ax2.bar(x, results[2] ,color="blue", align = 'center', alpha = 0.3)
ax2.set_ylabel("Time [s]",color="blue")
plt.title(f"n = {N}")
plt.xticks(x, results[0].astype(int))
plt.show()

plt.figure(figsize = (12,8))
plt.plot(X_star, prediction[0], 'g', label = 'Exact')
plt.plot(X_star, prediction[0] + 2* np.sqrt(prediction[1]), 'g--')
plt.plot(X_star, prediction[0] - 2* np.sqrt(prediction[1]), 'g--')
plt.grid(False)
plt.rcParams.update({'font.size': 18})

plt.scatter(X, y, color = 'k', marker = 'x', alpha = 0.5, label = 'Observations')
plt.xlabel('Input variable x')
plt.ylabel('Output variable y')
plt.plot(X_star, PR[0], 'r', label = "20 inducing points")
plt.plot(X_star, PR[0] + 2* np.sqrt(PR[1]), 'r--')
plt.plot(X_star, PR[0] - 2* np.sqrt(PR[1]), 'r--')
plt.legend()
plt.show()

#%% Compute inverse using Gauss elimination and Cholesky
I = 10

R = np.zeros((3,I))
for i in range(I):
    print(i)
    N = 10 * 2**i
    X = np.linspace(0,100000,N)[:,None]
    K = kern.K(X)
    J = max(int(10000000/N**2), 1)
    print(J)
    start = time.time()
    for j in range(J):
        X_inv = inv(K)
    end = time.time()
    
    s = time.time()
    for j in range(J):
        X_inv = cholesky(K)
    e = time.time()
    
    R[:,i] = N, (end-start)/J, (e-s)/J

#%% Gauss elimination versus Cholesky
x_new = np.linspace(40,R[0,-1],100)
z = np.polyfit(R[0], R[1], 3)
f = np.poly1d(np.array([z[0],0,0,0]))
y_new = 1e-10 * (x_new**3)

plt.figure(figsize = (7,7))
plt.rcParams.update({'font.size': 18})
plt.plot(x_new, y_new, label = "1e-10 * n^3", color = 'limegreen', linestyle = '-.')
plt.plot(R[0], R[1], marker = 'o', linestyle = '--', label = 'LU decomposition'); plt.xlabel('n (size of matrix is nxn)'); plt.ylabel("Time [s]"); plt.grid(True)
plt.plot(R[0], R[2], marker = 'o', color = 'red', label = "Cholesky decomposition"); plt.legend()
plt.yscale('log'); 
plt.show()

print(R[1,1:]/R[1,:-1])
print(R[2,1:]/R[2,:-1])

#%% Cov(X,X) = 0 if d(X,X) < r

N = 3000
M = 600

N_range = list(range(0,N+1,M))[1:]
R = np.zeros((len(N_range),4))
i = 0
for n in N_range:
    X = np.linspace(0,1000000,int(n))[:,None]
    K = kern.K(X)
    start = time.time()
    Kinv = inv(K)
    end = time.time()
    
    K[K<0.00001] = 0
    
    s = time.time()
    Kinv2 = inv(K)
    e = time.time()
    
    E = Kinv - Kinv2
    
    R[i,:] = n, np.sum(abs(E))/1000**2, end-start, e-s
    i += 1
    
#%% plot
x = list(range(len(N_range)))
fig,ax = plt.subplots()
ax.plot(x, R[:,1], color="red", marker="o")
ax.set_xlabel("N",fontsize=14)
ax.set_ylabel("Error",color="red",fontsize=14)
ax2=ax.twinx()
ax2.bar(x, R[:,2] ,color="blue", align = 'center', alpha = 0.3)
ax2.bar(x, R[:,3] ,color="green", align = 'center', alpha = 0.3)
ax2.set_ylabel("Time [s]",color="blue",fontsize=14)
plt.title(f"N = {N}")
plt.xticks(x, R[:,0].astype(int))
plt.show()

#%% SVD

kern = GPy.kern.RBF(1, lengthscale = 10)
n = 1000
X = np.linspace(0,100,n)[:,None]
K = kern.K(X) + np.eye(n)
normK = norm(K)
TIMES = np.zeros((9, 1000))
start = time.time()
I = 10
for i in range(I):
    startt = time.time()
    Kinv = inv(K)
    endd = time.time()
    TIMES[-1,i] = endd- startt
end = time.time()
print((end-start)/I)

klst = list(range(1, 9))
#klst = [1, 5, 10, 50, 100]
results = np.zeros((len(klst)+1, 5))
i=0
J = 10
for k in klst:
    s = time.time()
    for j in range(J):
        ss = time.time()
        [S, U] = eigsh(K, k = k)
        inverse = np.dot(U * 1/S, U.T)
        ee = time.time()
        TIMES[i,j] = ee - ss
    e = time.time()
    
    Khat = np.dot(U*S, U.T)
    results[i] = norm(K-Khat)/normK, k, (n+1)*k, n**2/((n+1)*k), (e-s)/J
    i += 1
results[-1] = 0, n, n**2, 1, (end-start)/I
#%% Plot table of results

print("Error   |  Rank |    Storage |   Factor |     Time")
for c1, c2, c3, c4, c5 in results:  
    print("%.5f | %5.0f | %10.0f | %8.1f | %6.2e " % (c1, c2, c3, c4, c5))

#%% plot inverse matrices next to each other
k = 8
val = 1

[S, U] = eigsh(K, k = k)
inverse = np.dot(U * 1/S, U.T)
Khat = np.dot(U*S, U.T)

plt.subplot(1, 2, 1)
plt.imshow(Khat, vmin=-val, vmax=val, cmap='jet', aspect='auto')
plt.subplot(1, 2, 2)
plt.imshow(K, vmin=-val, vmax=val, cmap='jet', aspect='auto')
plt.colorbar()

#%% plot singular values
plt.figure(figsize = (8,8))
x = list(range(1,n+1)); y = np.flip(eigh(np.dot(K.T,K), eigvals_only = True))
plt.plot(x, y, '.')
plt.yscale('log')
plt.grid(True, which = 'both')
#plt.ylim([-1000, 10000])
plt.title("Singular values")
plt.xlabel('i')
plt.ylabel('$\sigma_i$')
plt.show()

#%% plot subplots matrices B for different ranks
kern = GPy.kern.RBF(1, lengthscale = 10)
n = 1000
X = np.linspace(0,100,n)[:,None]
K = kern.K(X) + np.eye(n)

fig, ax = plt.subplots(3,3, figsize = (8,8))
rank = 1
for row in range(3):
    for col in range(3):
        if row == 2 & col == 2:
            r = ax[row,col].imshow(K, vmin = 0, vmax = 2)
            ax[row,col].set_title(f'Rank = {n}')
        else:
            [S, U] = eigsh(K, rank)
            ax[row,col].imshow(np.dot(U*S,U.T), vmin = 0, vmax = 2)
            ax[row,col].set_title(f'Rank = {rank}')
        rank += 1
    
for axs in fig.get_axes():
    axs.label_outer()
fig.colorbar(r, ax = ax, fraction = 0.04)  
plt.show()

#%% Compute speed for eigsh

kern = GPy.kern.RBF(1, lengthscale = 10)
N = 8
k = 5
J = 12

result = np.zeros((J,2))
for j in range(J):
    n = N*2**j
    print(n)
    X = np.linspace(0,100,n)[:,None]
    K = kern.K(X)
    
    s = time.time()

    for i in range(4096*10//n):
        [S, U] = eigsh(K, k)
    e = time.time()
    result[j,:] = n, (e-s)/(4096*10//n)

print(result[1:,1]/result[:-1,1])

#%% plot subplots inverse matrices B^-1 for different ranks
kern = GPy.kern.RBF(1, lengthscale = 10)
n = 1000
X = np.linspace(0,100,n)[:,None]
K = kern.K(X) + np.eye(n)

fig, ax = plt.subplots(2,3, figsize = (10,8))
ranks = [1, 5, 10, 50, 100]
val = 0.1
mval = 0.05
for row in range(2):
    for col in range(3):
        if row == 1 and col == 2:
            r = ax[row,col].imshow(inv(K), vmin = -mval, vmax = val)
            ax[row,col].set_title(f'Rank = {n}')
        else:
            rank = ranks[row*3 + col]
            [S, U] = eigsh(K, rank)
            ax[row,col].imshow(np.dot(U * 1/S, U.T), vmin = -mval, vmax = val)
            ax[row,col].set_title(f'Rank = {rank}')
        
    
for axs in fig.get_axes():
    axs.label_outer()
fig.colorbar(r, ax = ax, fraction = 0.03)  
plt.show()

#%% Compute errors for inverses
kern = GPy.kern.RBF(1, lengthscale = 10)
n = 1000
X = np.linspace(0,100,n)[:,None]
K = kern.K(X) + np.eye(n)
TIMES = np.zeros((9, 1000))
start = time.time()
I = 10
for i in range(I):
    startt = time.time()
    Kinv = inv(K)
    endd = time.time()
    TIMES[-1,i] = endd- startt
end = time.time()
print((end-start)/I)

normK = norm(Kinv)

klst = list(range(1, 9))
klst = [1, 5, 10, 50, 100, 200, 500]
results = np.zeros((len(klst)+1, 5))
i=0
J = 1
for k in klst:
    s = time.time()
    for j in range(J):
        ss = time.time()
        [S, U] = eigsh(K, k = k, tol = 10**-5)
        inverse = np.dot(U * 1/S, U.T)
        ee = time.time()
        TIMES[i,j] = ee - ss
    e = time.time()
    
    results[i] = norm(Kinv-inverse)/normK, k, (n+1)*k, n**2/((n+1)*k), (e-s)/J
    i += 1
results[-1] = 0, n, n**2, 1, (end-start)/I
#%% Plot table of results

print("Error   |  Rank |    Storage |   Factor |     Time")
for c1, c2, c3, c4, c5 in results:  
    print("%.5f | %5.0f | %10.0f | %8.1f | %6.2e " % (c1, c2, c3, c4, c5))

#%% Cholesky inverse

L = solve(K + 1*np.eye(K.shape[0]), )
Kinv = np.dot(L,L.T)

#%% Nystrom
np.random.seed(101)

m = 30

rows = np.random.randint(len(K), size = m)
#rowtf = np.isin(np.arange(len(K)), rows)

Kmm = K[rows, :][:,rows] #m x m
#Knm = K[~rowtf,:][:,rows] #(m-n) x m
#Kmn = K[rows,:][:,~rowtf] #m x (m-n)

[U, S, _] = np.linalg.svd(Kmm, hermitian = True) #m x m

#Util = np.dot(Knm, U*1/S) #(m-n) x m
#Vtil = np.dot(1/S * U.T, Kmn) #m x (m-n)

#Uhat = np.concatenate((U, Util), axis = 0) #n x m
#Vhat = np.concatenate((U.T, Vtil), axis = 1) #m x n

Shat = len(K)/m * S
uhat = np.sqrt(m/len(K))/S * np.dot(K[:,rows], U)

Khat = np.dot(uhat * Shat, uhat.T)

#%% Nystrom time/norm comparison

np.random.seed(101)
kern = GPy.kern.RBF(1, lengthscale = 10)
n = 1000
X = np.linspace(0,100,n)[:,None]
K = kern.K(X) + np.eye(n)
normK = norm(K)
start = time.time()
I = 10
for i in range(I):
    Kinv = inv(K)
end = time.time()
print((end-start)/I)

klst = list(range(1, 50))
#klst = [1, 5, 10, 50, 100]
results = np.zeros((len(klst)+1, 5))
i=0
J = 10
for k in klst:
    s = time.time()
    for j in range(J):
        rows = np.random.randint(len(K), size = k)
        Kmm = K[rows, :][:,rows] #m x m
        [U, S, _] = np.linalg.svd(Kmm, hermitian = True) #m x m
        Shat = len(K)/k * S
        Uhat = np.sqrt(k/len(K))/S * np.dot(K[:,rows], U)
        inverse = np.dot(Uhat * 1/Shat, Uhat.T)
    e = time.time()
    
    Khat = np.dot(Uhat*Shat, Uhat.T)
    results[i] = norm(K-Khat)/normK, k, (n+1)*k, n**2/((n+1)*k), (e-s)/J
    i += 1
results[-1] = 0, n, n**2, 1, (end-start)/I
#%% Plot table of results

print("Error   |  Rank |    Storage |   Factor |     Time")
for c1, c2, c3, c4, c5 in results:  
    print("%.5f | %5.0f | %10.0f | %8.1f | %6.2e " % (c1, c2, c3, c4, c5))

#%% Nystrom time/norm comparison for inverse

np.random.seed(101)
kern = GPy.kern.RBF(1, lengthscale = 10)
n = 1000
X = np.linspace(0,100,n)[:,None]
K = kern.K(X) + np.eye(n)
normK = norm(K)
start = time.time()
I = 1
for i in range(I):
    Kinv = inv(K)
end = time.time()
print((end-start)/I)

klst = list(range(1, 20))
#klst = [1, 5, 10, 50, 100]
results = np.zeros((len(klst)+1, 5))
i=0
J = 10
for k in klst:
    s = time.time()
    for j in range(J):
        rows = np.random.randint(len(K), size = k)
        Kmm = K[rows, :][:,rows] #m x m
        [U, S, _] = np.linalg.svd(Kmm, hermitian = True) #m x m
        Shat = len(K)/k * S
        Uhat = np.sqrt(k/len(K))/S * np.dot(K[:,rows], U)
        inverse = np.dot(Uhat * 1/Shat, Uhat.T)
    e = time.time()
    
    results[i] = norm(Kinv-inverse)/normK, k, (n+1)*k, n**2/((n+1)*k), (e-s)/J
    i += 1
results[-1] = 0, n, n**2, 1, (end-start)/I
#%% Plot table of results

print("Error   |  Rank |    Storage |   Factor |     Time")
for c1, c2, c3, c4, c5 in results:  
    print("%.5f | %5.0f | %10.0f | %8.1f | %6.2e " % (c1, c2, c3, c4, c5))
