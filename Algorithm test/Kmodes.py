import inline
import matplotlib
import pandas as pd
import numpy as np
import math
from kmodes.kmodes import KModes


# Data viz lib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib.pyplot import xticks


from sklearn.datasets import make_blobs
from sklearn.metrics import jaccard_score
from yellowbrick.cluster import KElbowVisualizer
from subprocess import call
from sklearn import preprocessing






data = pd.read_csv(r'C:\Users\ibrad\PycharmProjects\cleaned_COVID_Data.csv')





# Generate dataset with 8 random clusters
#X, y = make_blobs(n_samples=100, n_features=8, centers=670000, random_state=42)

# Instantiate the clustering model and visualizer
#model = KModes()
#visualizer = KElbowVisualizer(
   # model, k=99, metric='calinski_harabasz', timings=False, locate_elbow=False
#)


#visualizer.fit(X)        # Fit the data to the visualizer
#visualizer.show()        # Finalize and render the figure





cost = []
for num_clusters in list(range(1,5)):
    kmode = KModes(n_clusters=num_clusters, init = "Cao", n_init = 1, verbose=1)
    kmode.fit_predict(data)
    cost.append(kmode.cost_)

y = np.array([i for i in range(1,5,1)])
plt.title('Kmode Elbow Method Modelisation')
plt.xlabel('k')
plt.ylabel('Cost')
plt.plot(y,cost)
plt.show()


#Similar to the elbow method from kmeans, we looked at the centroids from each number of clusters and examine the broadness of the
#labels found in those clusters to determine the optimal amount of clusters since kmodes does not currently have any general
#method of finding it as kmeans does.

km = KModes(n_clusters=10, init='Huang', n_init=11, verbose=1)
# fit the clusters to the skills dataframe
clusters = km.fit_predict(data)
# get an array of cluster modes
kmodes = km.cluster_centroids_
shape = kmodes.shape
# For each cluster mode where our team one-hot encoded (a vector of "1" and "0")
# we will print the column headings where "1" appears.
# If no "1" appears, we assign it to a "no category" cluster.
for i in range(shape[0]):
    if sum(kmodes[i,:]) == 0:
        print("\ncluster " + str(i) + ": ")
        print("no category cluster")
    else:
        print("\ncluster " + str(i) + ": ")
        cent = kmodes[i,:]
        for j in data.columns[np.nonzero(cent)]:
            print(j)
