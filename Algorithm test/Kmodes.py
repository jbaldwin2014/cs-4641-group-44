import pandas as pd
import numpy as np
import math
from kmodes.kmodes import KModes
from sklearn.datasets import make_blobs
from sklearn.metrics import jaccard_score


from yellowbrick.cluster import KElbowVisualizer


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


def jacScorePlot():

    ecArr = []
    jcArr = []
    jcXArr = []
    for i in range(2, 10):
        km = KModes(n_clusters=i)
        y = km.fit_predict(X)     #X is not one-hot encoded data, it is normal categorical data
        tempArrjc = []
        tempArrec = []
        tempArrjcX = []
        for j in range(i):
            # print(sum(y==j))
            # print(XX[y==j].mode())
            jcscore = []
            ecscore = []
            jcXscore = []
            for k in data[y == j].T:
                try:
                    # jcscore.append(jaccard_similarity_score(XX.loc[k],XX[y==j].mode().T[0]))

                    ecscore.append(np.linalg.norm(np.array(data.loc[k]) - np.array(data[y == j].mode().T[0])))

                    jcXscore.append(jaccard_score(list(X.loc[k]), list(X[y == j].mode().T[0])))
                except:
                    # print(XX.loc[k].T)
                    # print(XX[y==j].mode())
                    print(k)
                    break
                    # print(np.mean(jcscore))
                    # tempArrjc.append(np.mean(jcscore))
                    # tempArrec.append(np.mean(ecscore))
                tempArrjcX.append(np.mean(jcXscore))

                print("n_cluster =", i, ":", np.mean(tempArrjcX))
                # jcArr.append(np.mean(tempArrjc))
                # ecArr.append(np.mean(tempArrec))
                jcXArr.append(np.mean(tempArrjcX))






def makeClusters(data,numClusters):
    km = KModes(n_clusters=numClusters, init="Cao", n_init=1, verbose=1)
    #subsetDf = data.loc[data['categoryWant'] == categorLabel].drop(['ICU', 'feature1', 'feature2', 'feature3', 'feature4'], axis=1)
    #subsetData = subsetDf.values
    fitClusters = km.fit_predict(data.values)
    clusterCentroidsDf = pd.DataFrame(km.cluster_centroids_)
    clusterCentroidsDf.columns = data.columns

    return fitClusters, clusterCentroidsDf

#Similar to the elbow method, we looked at the centroids from each number of clusters adn examine the broadness of the
#labels found in those clusters to determine the optimal amount of clusters since kmodes does not currently have any general
#method of finding it as kmeans does.

clusterData1, newData = makeClusters(data, 5)

newData.to_csv(r'C:\Users\ibrad\PycharmProjects\centroid5_COVID_Data.csv')

clusterData2 = makeClusters(data, 10)

newData.to_csv(r'C:\Users\ibrad\PycharmProjects\centroid10_COVID_Data.csv')


clusterData3 = makeClusters(data, 25)

newData.to_csv(r'C:\Users\ibrad\PycharmProjects\centroid25_COVID_Data.csv')

clusterData4 = makeClusters(data, 50)

newData.to_csv(r'C:\Users\ibrad\PycharmProjects\centroid50_COVID_Data.csv')














# define the k-modes model
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


