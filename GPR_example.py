# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#
# License: BSD 3 clause

import numpy as np

from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel)

plt.rcParams.update({'font.size': 18})

kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
           

# Specify Gaussian Process
gp = GaussianProcessRegressor(kernel=kernel)

# Plot prior
plt.figure(figsize=(15, 8))
plt.subplot(1, 2, 1)
X_ = np.linspace(-1, 1, 100)
y_mean, y_std = gp.predict(X_[:, np.newaxis], return_std=True)
plt.plot(X_, y_mean, 'k', lw=3, zorder=9)
plt.fill_between(X_, y_mean - 2*y_std, y_mean + 2*y_std,
                 alpha=0.2, color='k')
y_samples = gp.sample_y(X_[:, np.newaxis], 10)
plt.plot(X_, y_samples, lw=2, alpha = 0.5, color = 'blue')
plt.xlim(-1, 1)
plt.ylim(-2.5, 2.5)
plt.ylabel("Output")
plt.xlabel("Input")
plt.title("Prior")

# Generate data and fit GP
rng = np.random.RandomState(4)
X = rng.uniform(-1, 1, 2)[:, np.newaxis]
y = np.sin((X[:, 0] - 2.5) ** 2)
gp.fit(X, y)

X_ = np.linspace(-1, 1, 100)

# Plot posterior
plt.subplot(1, 2, 2)
y_mean, y_std = gp.predict(X_[:, np.newaxis], return_std=True)
plt.plot(X_, y_mean, 'k', lw=3, zorder=9)
plt.fill_between(X_, y_mean - 2*y_std, y_mean + 2*y_std,
                 alpha=0.2, color='k')

y_samples = gp.sample_y(X_[:, np.newaxis], 10)
plt.plot(X_, y_samples, lw=2, alpha = 0.5, color = 'blue')
plt.scatter(X[:, 0], y, c='r', s=200, zorder=10, edgecolors=(0, 0, 0))
plt.xlim(-1, 1)
plt.ylim(-2.5, 2.5)
plt.xlabel("Input")
plt.title("Posterior")
plt.tight_layout()

plt.show()