from matplotlib.pylab import *
from pypr.clustering.kmeans import *
from pypr.clustering.gmm import *

# Make three clusters:
mc = [0.4, 0.4, 0.2] # Mixing coefficients
centroids = [ array([1,1]), array([3,3]), array([3,0]) ]
ccov = [ array([[1,0],[0,1]]), array([[1,0],[0,1]]), \
          array([[0.2, 0],[0, 0.2]]) ]
X = sample_gaussian_mixture(centroids, ccov, mc, samples=1000)

figure(figsize=(10,5))
subplot(121)
title('Original unclustered data')
plot(X[:,0], X[:,1], '.')
xlabel('$x_1$'); ylabel('$x_2$')

subplot(122)
title('Clustered data')
m, cc = kmeans.kmeans(X, 3)
plot(X[m==0, 0], X[m==0, 1], 'r.')
plot(X[m==1, 0], X[m==1, 1], 'b.')
plot(X[m==2, 0], X[m==2, 1], 'g.')
xlabel('$x_1$'); ylabel('$x_2$')
