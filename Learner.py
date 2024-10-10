from enum import Enum
import math
import pandas as pd
import numpy as np
import ClassificationInfo
import Kmeans

class Accuracy(Enum):
    TP = 1
    TN = 2
    FP = 3
    FN = 4
    
class Learner:
    
    def __init__(self, data, classificationType, targetPlace):
        self.classificationType = classificationType
        self.size = data.shape[0]
        self.targetPlace = targetPlace
        self.threshold = None
        self.k = None
        self.kernel = None
        if(self.classificationType == "regression"):
            self.setThreshold(data)
        self.data = data
        self.euclidean = np.zeros((self.size, self.size))
        self.pointIndex = {}
        self.createDistances()
        self.tuningData = self.getTuneData(data)
        self.data = data.drop(self.tuningData.index)
        self.folds = self.crossValidation(self.data, self.targetPlace, False)
        self.tuneData()
        self.edited = pd.DataFrame()

    def createDistances(self):
        n = self.data.shape[0]
        print("Creating distances...")
        #create distances for euclidean distance
        for i in range(n):
            print("on index: ", i, " out of ", n)
            for j in range(i + 1, n):
                distance = self.euclideanDistance(self.data.iloc[i], self.data.iloc[j])
                self.pointIndex[str(self.data.iloc[i])] = i
                self.pointIndex[str(self.data.iloc[j])] = j
                self.euclidean[i, j] = distance
                self.euclidean[j, i] = distance  # The distance is symmetric

    def setThreshold(self, data):
        colAverage = data[self.targetPlace].mean()
        self.threshold = colAverage * 0.1
    
    def getTuneData(self, data):
        print("Splitting data for tuning...")
        # split data for tuning
        tune_data = data.groupby(self.targetPlace, group_keys=False).apply(lambda x: x.sample(frac=0.1))
        return tune_data

    def tuneData(self):
        # Tune the data
        print("Tuning data...")
        centerK = round(math.sqrt(self.size))
        start = max(0, centerK - 2 * 2)
        end = min(self.size, centerK + 2 * 2)

        # Generate 5 values in the range
        possibleK = list(range(start, end + 1, 2))

        possibleKernel = [0.1, 0.5, 2, 5, 10]
        count = 0
        classifications = {}
        if self.classificationType == "classification":
            for k in possibleK:
                train = self.data.drop(self.folds[count % 10].index)
                accuracy = self.tuneClassification(k, train)
                classifications[accuracy] = k
                count += 1
            
            best = max(classifications.keys())
            self.k = classifications[best]
            print("best k = ", self.k)
        else:
            for k in possibleK:
                for kernel in possibleKernel:
                    print("test k = ", k)
                    print("test kernel = ", kernel)
                    train = self.data.drop(self.folds[count % 10].index)
                    accuracy = self.tuneRegression(k, kernel, train)
                    classifications[accuracy] = [k, kernel]
                    count += 1

            best = max(classifications.keys())
            self.k = classifications[best][0]
            self.kernel = classifications[best][1]
            print("best k = ", self.k)
            print("best kernel = ", self.kernel)

    def crossValidation(self, cleanDataset, classColumn, printSteps):
        print("Running cross validation...")
    # 10-fold cross validation with stratification of classes
        if printSteps == True:
            print("Running cross calidation with stratification...")
        dataChunks = [None] * 10
        classes = np.unique(cleanDataset[classColumn])
        dataByClass = dict()

        for uniqueVal in classes:
            # Subset data based on unique class values
            classSubset = cleanDataset[cleanDataset[classColumn] == uniqueVal]
            if printSteps == True:
                print("Creating a subset of data for class " + str(uniqueVal) + " with size of " + str(classSubset.size))
            dataByClass[uniqueVal] = classSubset

            numRows = math.floor(classSubset.shape[0] / 10) # of class instances per fold

            for i in range(9):
                classChunk = classSubset.sample(n=numRows)
                if printSteps:
                    print("Number of values for class " + str(uniqueVal), " in fold " + str(i+1) + " is: " + str(classChunk.shape[0]))
                if dataChunks[i] is None:
                    dataChunks[i] = classChunk
                else:
                    dataChunks[i] = pd.concat([dataChunks[i], classChunk])

                classSubset = classSubset.drop(classChunk.index)

            # the last chunk might be slightly different size if dataset size is not divisible by 10
            if printSteps == True:
                print("Number of values for class " + str(uniqueVal), " in fold " + str(10) + " is: " + str(classSubset.shape[0]))
            dataChunks[9] = pd.concat([dataChunks[9], classSubset])

        if printSteps == True:
            for i in range(len(dataChunks)):
                print("Size of fold " + str(i+1) + " is " + str(dataChunks[i].shape[0]))

        return dataChunks

    def editData(self):
        #run classification and regression on edited data
        self.kmeanClusters = self.edited.shape[0]
        copy = self.data
        copyFolds = self.folds 
        self.data = self.edited
        self.folds = self.crossValidation(self.data, self.targetPlace, False)
        if self.classificationType == "classification":
            output =  self.classification()
        else:
            output =  self.regression()
        self.data = copy
        self.folds = copyFolds
        return output
    
    def tuneClassification(self, k, train):
        print("k = ", k)
        sum = 0
        total = 0
        for i in range(self.tuningData.shape[0]):
            neighbors = self.findNeighbors(self.tuningData.iloc[i], train, k)
            correctClass = self.tuningData.iloc[i][self.targetPlace]
            assignedClasses = {}
            for neighbor in neighbors:
                neighborClass = train.iloc[neighbor[1]][self.targetPlace]
                if neighborClass in assignedClasses:
                    assignedClasses[neighborClass] += 1
                else:
                    assignedClasses[neighborClass] = 1
            assignedClass = max(assignedClasses, key=assignedClasses.get)
            classAccuracy = self.classificationAccuracy(correctClass, assignedClass)
            if(classAccuracy == Accuracy.TP or classAccuracy == Accuracy.TN):
                sum += 1
            total += 1

        return sum/total
    
    def tuneRegression(self, k, kernel , train):
        sum = 0
        total = 0

        for i in range(self.tuningData.shape[0]):
            neighbors = self.findNeighbors(self.tuningData.iloc[i], train ,k)
            print("neighbors: ", neighbors)
            correctValue = self.tuningData.iloc[i][self.targetPlace]
            numerator = 0
            denominator = 0
            for neighbor in neighbors:
                weight = self.kernelWeight(self.tuningData.iloc[i], train.iloc[neighbor[1]], kernel)
                value = train.iloc[neighbor[1]][self.targetPlace]
                numerator += (weight * value)
                denominator += weight
            assignedValue = numerator/denominator
            if self.regressionAccuracy(correctValue, assignedValue) == Accuracy.TP or self.regressionAccuracy(correctValue, assignedValue) == Accuracy.TN:
                sum += 1
            total += 1
        return sum / total
    
    def classification(self):
        classification = ClassificationInfo.ClassificationInfo()
        # return the classification info for each dataset
        for fold in self.folds: 
            train = self.data.drop(fold.index)
            total = 0
            sum = 0
            for i in range(fold.shape[0]):
                neighbors = self.findNeighbors(fold.iloc[i], train)
                correctClass = fold.iloc[i][self.targetPlace]
                assignedClasses = {}
                for neighbor in neighbors:
                    neighborClass = train.iloc[neighbor[1]][self.targetPlace]
                    if neighborClass in assignedClasses:
                        assignedClasses[neighborClass] += 1
                    else:
                        assignedClasses[neighborClass] = 1
                assignedClass = max(assignedClasses, key=assignedClasses.get)
                classAccuracy = self.classificationAccuracy(correctClass, assignedClass)
                if(classAccuracy == Accuracy.TP or classAccuracy == Accuracy.TN):
                    data_df = pd.DataFrame([fold.iloc[i]])
                    self.edited = pd.concat([self.edited, data_df])
                    sum += 1
                total += 1
                classification.addTrueClass([correctClass, assignedClass])
                classification.addConfusion(classAccuracy)
            print("fold accuracy: ", sum/total)
            print() 

        return classification
    
    def kernelWeight(self, point1, point2, kernel):
        #return weight
        squaredDist = abs(self.euclideanDistance(point1, point2)) ** 2
        return math.exp(-squaredDist / (2 * kernel ** 2))
        

    def regression(self):
        classification = ClassificationInfo.ClassificationInfo()
        # return the regression classification info 
        # return the classification info for each dataset
        for fold in self.folds: 
            train = self.data.drop(fold)
            for i in range(fold.shape[0]):
                sum = 0
                total = 0
                neighbors = self.findNeighbors(fold.iloc[i], train)
                correctValue = fold.iloc[i][self.targetPlace]
                numerator = 0
                denominator = 0
                for neighbor in neighbors:
                    weight = self.kernelWeight(fold.iloc[i], train.iloc[neighbor[1]], self.kernel)
                    value = train.iloc[neighbor[1]][self.targetPlace]
                    numerator += (weight * value)
                    denominator += weight
                assignedValue = numerator/denominator

                classAccuracy = self.regressionAccuracy(correctValue, assignedValue)
                if(classAccuracy == Accuracy.TP or classAccuracy == Accuracy.TN):
                    data_df = pd.DataFrame([fold.iloc[i]])
                    self.edited = pd.concat([self.edited, data_df])
                    sum += 1
                total += 1
                classification.addTrueClass([correctValue, assignedValue])
                classification.addConfusion(classAccuracy)
                print("fold accuracy: ", sum/total)

        return classification

    def euclideanDistance(self, point1, point2):
        #return distance 
        featuresSum = 0
        for i in range(self.data.shape[1]):
            featuresSum += (point2.iloc[i] - point1.iloc[i])**2

        return math.sqrt(featuresSum)

    def classificationAccuracy(self, trueClass, assignedClass):
        classNames = list(self.data[self.targetPlace].unique())
        #decide where classification falls on confusion matrix
        if trueClass == assignedClass:
            if trueClass == classNames[0]:
                return Accuracy.TP
            else:
                return Accuracy.TN
        else:
            if assignedClass == classNames[0]:
                return Accuracy.FP
            else:
                return Accuracy.FN
            
    def regressionAccuracy(self, trueValue, assignedValue):
        #decide where regression falls on confusion matrix
        if trueValue > assignedValue:
            if trueValue - assignedValue < self.threshold:
                return Accuracy.TP
            else:
                return Accuracy.FP
        else:
            if assignedValue - trueValue < self.threshold:
                return Accuracy.TN
            else:
                return Accuracy.FN
    
    def findNeighbors(self, point, train, k=-1):
        #find the k nearest neighbors
        distances = []

        if( k == -1):
            for i in range(train.shape[0]):
                k = self.pointIndex[str(train.iloc[i])]
                j = self.pointIndex[str(point)]
                distance = self.euclidean[k][j]
                distances.append([distance, i])
            distances.sort()
            return distances[:self.k]
        else:
            for i in range(train.shape[0]):
                k = self.pointIndex[str(train.iloc[i])]
                j = self.pointIndex[str(point)]
                distance = self.euclidean[k][j]
                distances.append([distance, i])
            distances.sort()
            return distances[:k]
    def kmeans(self):
        #run kmeans algorithm
        kmeans = Kmeans.Kmeans(self.kmeanClusters, self.data)
        centroids = kmeans.getCentroids()
        copy = self.data
        copyFolds = self.folds 
        self.data = centroids
        print(self.data.head(20))
        self.folds = self.crossValidation(self.data, self.targetPlace, False)
        if self.classificationType == "classification":
            output =  self.classification()
        else:
            output =  self.regression()
        self.data = copy
        self.folds = copyFolds
        return output
        