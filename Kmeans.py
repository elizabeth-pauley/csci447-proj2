import numpy as np
import pandas as pd

class Kmeans:
    def __init__(self, k, data):
        self.k = k
        self.data = data
        self.centroids = np.zeros(k)
        self.clusters = {} # dict to store each centroid + associated points
        self.findClusters()
    
    def findClusters(self):
        # run kmeans algorithm
        # randomly determine k cluster centroids within the space
        for c in range(self.k):
            # randomly generate values between -1 and 1 for initial centroid;
            # num values = num examples in dataset
            centroid = 2*(2*np.random.random((self.data.shape[1],))-1) # array of numbers
            self.centroids[c] = centroid

            points = [] # points assigned to given cluster
            cluster = {
                'centroid': centroid,
                'points' : []
            }

            self.clusters[c] = cluster

        oldCentroids = self.centroids
        # initial point assignment
        self.pointsToClusters()
        self.recalculateCentroids()

        # recalculate centroids + reassign points until the centroids are no longer changing
        while(oldCentroids != self.centroids):
            # oldCentroids will track the previously calculated centroids
            oldCentroids = self.centroids
            self.pointsToClusters()
            self.recalculateCentroids()

        # create array of final cluster predictions for each datapoint
        self.centroids = pd.DataFrame(self.finalPredictCluster())

    # HELPER METHODS FOR K-MEANS ALGORITHM
    def __euclideanDistance(p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))

    def __pointsToClusters(self):
        numRows = self.data.shape[0]
        for row in range(numRows):
            # store distances between point and centroids
            distances = []

            currEx = self.data[row]

            # calculate distance between given point and each current centroid
            for i in range(self.k):
                dist = self.euclideanDistance(currEx, self.centroids[i])
                distances.append(dist)
                # choose cluster based on closest distance
                assignedCluster = np.argmin(distances)
                # store point in associated cluster's array of points
                self.clusters[assignedCluster]['points'].append(currEx)
            return self.clusters

    def __recalculateCentroids(self):
        for i in range(self.k):
            # local instance of points for given centroid
            points = np.array(self.clusters[i]['points'])

            if points.shape[0] > 0:
                # newCentroid = mean of current cluster
                newCentroid = points.mean(axis = 0)
                self.clusters[i]['center'] = newCentroid
                self.centroids[i] = newCentroid

                # empty out 'points' array so that they can be re-assessed with the new centroids
                self.clusters[i]['points'] = []
        return self.clusters

    def __finalPredictCluster(self):
        predictions = []
        # for each row in the dataset...
        for i in range(self.data.shape[0]):
            distances = []
            # calculate distance between given point and each cluster centroid
            for j in range(self.k):
                distances.append(self.euclideanDistance(self.data[i], self.centroids(self.k)))

            # append predicted cluster index based on which centroid is closest to the given point
            predictions.append(np.argmin(distances))
        return predictions

    # END HELPER METHODS

    def getCentroids(self):
        return self.centroids
    
    def displayClusters(self):
        # show graph of clusters 
        pass
    