import numpy as np
import matplotlib.pyplot as plt

class KMeans(object):

    def __init__(self): #No need to implement
        pass
    
    def pairwise_dist(self, x, y):
        # dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))
        # euc2 = np.linalg.norm(x[:,None,:] - y[None, :, :], axis=1)

        # Expand both x and y to shape [N x M x D] using np.expand_dims + np.repeat,
        # subtract expanded y from expanded x, and apply np.sum to the square of the result on the right axis.

        # x = np.expand_dims(x) + np.repeat(x)
        # y = np.expand_dims(y) + np.repeat(x)

        eucl = x[:, :, None] - y[:, :, None].T
        eucl = np.sqrt((eucl * eucl).sum(axis=1))
        return eucl

    def _init_centers(self, points, K, **kwargs):

        #Take average of each grouping of points

        lowk = np.minimum(K, (np.unique(points)).size)

        maxim = np.max(points, axis=0)

        n, d = points.shape

        #select random points (data points to be) cluster centers for _init_centers

        result = maxim * np.random.rand(lowk, d)

        return result

    def _update_assignment(self, centers, points):

        cluster_index = np.argmin(self.pairwise_dist(points, centers), axis=1)

        newArr = np.transpose(cluster_index)

        return newArr

    def _update_centers(self, old_centers, cluster_idx, points):

        newCenters = np.empty(old_centers.shape)

        K = old_centers.shape[0]

        for i in range(K):
            # Taking average of each grouping of points
            newCenters[i] = np.mean(points[cluster_idx == i], axis=0)
        return newCenters

    def _get_loss(self, centers, cluster_idx, points):

        #Square the pairwise distance for all points and their respective cluster centers
        # and then add them together. The loss is a single value that is the sum of the squared
        # L2 distance from each point to its corresponding center

        #loss = np.sum(np.sqrt(self.pairwise_dist(centers, points)))

        #totsum = 0

        #for i in range(len(centers)):
         #   temploss = np.sum(np.sqrt(self.pairwise_dist(centers[i], points[cluster_idx == i])))
          #  totsum = totsum + temploss

        loss = 0
        i = 0

        while i < len(centers):
            cluster_loss = np.sum((points[cluster_idx == i] - centers[i])**2)

            loss = loss + cluster_loss
            i += 1

        return loss
        
    def __call__(self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, verbose=False, **kwargs):

        centers = self._init_centers(points, K, **kwargs)
        #prepend __call__ with self beacause  __call__ is a=
        # part of the class KMeans and I can use data from optimal_num_clusters for points in __call__
        for i in range(max_iters):
            cluster_index = self._update_assignment(centers, points)
            centers = self._update_centers(centers, cluster_index, points)
            loss = self._get_loss(centers, cluster_index, points)
            K = centers.shape[0]
            if i:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol: break
            prev_loss = loss
            if verbose:
                print('iter %d, loss: %.4f' % (i, loss))
        return cluster_index, centers, loss

    def find_optimal_num_clusters(self, data, max_K=15):

        #Run the Kmeans from 1 cluster to 15 clusters and
        # return a list which includes the loss value for each case.
        #return the list in find optimal num clusters
        # and plot the loss values for different number of clusters.

        #The maximum possible number of clusters will be equal to the number of observations(points) in the dataset.

        losses = []

        for i in range(1, max_K):
            tupl = self.__call__(data, i)
            loss = tupl[2]
            losses.append(loss)

        return losses

