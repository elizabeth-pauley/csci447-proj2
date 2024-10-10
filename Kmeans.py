import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Kmeans:
    def __init__(self, k, data):
        self.k = k
        self.data = data
        self.centroids = []
        self.clusters = {}  # dict to store each cluster value + associated centroid and points
        print("RUN K-MEANS ALGORITHM...")
        self.findClusters()
        self.displayClusters()

    def findClusters(self):
        print("INITIALIZE K RANDOM CENTROIDS...")
        # Randomly determine k cluster centroids within the data
        for c in range(self.k):
            # Take k random samples from the data for initial centroids
            centroid = self.data.sample()
            self.centroids.append(centroid)

            # Store each cluster's data in a dictionary
            cluster = {
                'centroid': centroid,
                'points': []
            }
            self.clusters[c] = cluster

        # booolean to control print statements
        displayProcess = True
        # Initial point assignment to clusters
        print("ASSIGN POINTS TO INITIAL CLUSTERS...")
        print("THE FOLLOWING IS AN EXAMPLE OF THE FIRST DATA POINT BEING ASSIGNED A CLUSTER.")
        self.__pointsToClusters(displayProcess)
        displayProcess = False

        # Recalculate centroids based on the means of current clusters
        print("RECALCULATE CENTROIDS AS MEANS OF CURRENT CLUSTERS...")
        self.__recalculateCentroids()

        # Iterate the assignment of points to clusters and recalculation of clusters until
        # the previous centroids have only a 5% overall difference from the most recently calculated centroids
        centroids_changed = True
        movement_threshold = 0.90  # 5% threshold

        print("REPEAT THIS PROCESS UNTIL CENTROIDS EXPERIENCE <= 5% OVERALL CHANGE FROM THE LAST CENTROIDS...")
        while centroids_changed:
            oldCentroids = np.array([c.values.flatten() for c in self.centroids])  # Store old centroids before updating clusters
            self.__pointsToClusters(displayProcess)
            self.__recalculateCentroids()
            newCentroids = np.array([c.values.flatten() for c in self.centroids])

            # Check if centroids have changed based on the threshold
            differences = np.abs(newCentroids - oldCentroids) / (np.abs(oldCentroids) + 1e-10)
            centroids_changed = np.any(differences > movement_threshold)

        # Determine final cluster predictions for each value
        print("DETERMINE FINAL CLUSTER PREDICTIONS FOR EACH DATA POINT...")
        self.predictions = self.__finalPredictCluster()
        print("K-MEANS COMPLETE")

    def __euclideanDistance(self, p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))

    def __pointsToClusters(self, displayProcess):
        for i in range(self.k):
            self.clusters[i]['points'] = []  # Clear previous points

        # For each example in the dataset...
        for i in range(self.data.shape[0]):
            row = self.data.iloc[i]
            if i == 0 and displayProcess:
                print("First example being assigned a cluster: ")
                print(row)
            distances = []

            # Calculate distance between given example and each current centroid
            if i == 0 and displayProcess:
                print(f"Calculating distances between first example and centroids 0 through {self.k}...")
            for j in range(self.k):
                dist = self.__euclideanDistance(row.values, self.centroids[j])
                distances.append(dist)
            # Choose predicted cluster based on closest distance
            assignedCluster = np.argmin(distances)
            if i == 0 and displayProcess:
                print(f"The closest centroid is centroid {assignedCluster}. Adding first example to the associated cluster...")

            # Add example to cluster associated with closest centroid
            self.clusters[assignedCluster]['points'].append(row)

    def __recalculateCentroids(self):
        for i in range(self.k):
            points = np.array(self.clusters[i]['points'])
            # If a given cluster has been assigned data points...
            if points.shape[0] > 0:
                # Recalculate the centroid as the mean of all data points within
                newCentroid = points.mean(axis=0)
                self.clusters[i]['centroid'] = newCentroid
                self.centroids[i] = pd.DataFrame(newCentroid).T  # Store back as DataFrame for future use (to track old and new centroids)

    def __finalPredictCluster(self):
        # Store final cluster assignments here
        predictions = []

        # Same as pointsToClusters(), except we are just adding each prediction to an array (to be returned later)
        for i in range(self.data.shape[0]):
            row = self.data.iloc[i]
            distances = []

            for j in range(self.k):
                dist = self.__euclideanDistance(row.values, self.centroids[j])
                distances.append(dist)
            minDistIndex = np.argmin(distances)
            predictions.append(minDistIndex)
        return predictions

    def getCentroids(self):
        return pd.DataFrame(self.centroids)

    def getPredictions(self):
        return self.predictions

    def displayClusters(self):
        # Implement PCA to reduce to 2 dimensions
        # graph data w/ colors being clusters
        # red set aside for centroids
        print("DATA: ")
        print(self.data.isna().count())
        #dataPCA = PCA(n_components=2)
        #dataTransformed = dataPCA.fit_transform(self.data)

        #plt.scatter(dataTransformed[:, 0], dataTransformed[:, 1], color=self.predictions)
        #plt.title('2D Projection of PCA Transformed Dataset, Colored by Cluster')
        #plt.xlabel("First PCA")
        #plt.ylabel("Second PCA")
