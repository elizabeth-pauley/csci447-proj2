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
        if self.classificationType == "classification":
            self.folds = self.crossValidation(self.data, self.targetPlace, True)
        else:
            self.folds = self.crossValidationRegression(self.data.copy(), self.targetPlace, False)
        self.tuneData()
        self.edited = pd.DataFrame()

    def createDistances(self):
        print("Creating distances...")
        n = self.data.shape[0]
        #create distances for euclidean distance
        for i in range(n):
            print("on index ", i, " of ", n)
            for j in range(i + 1, n):
                distance = self.euclideanDistance(self.data.iloc[i], self.data.iloc[j])
                self.pointIndex[str(self.data.iloc[i])] = i
                self.pointIndex[str(self.data.iloc[j])] = j
                self.euclidean[i, j] = distance
                self.euclidean[j, i] = distance 

    def setThreshold(self, data):
        colAverage = data[self.targetPlace].mean()
        self.threshold = colAverage * 0.1
    
    def getTuneData(self, data):
        print("Splitting data for tuning...")
        # split data for tuning
        tune_data = data.groupby(self.targetPlace, group_keys=False).apply(lambda x: x.sample(frac=0.1))
        return tune_data

    def tuneData(self):
        print("Tuning data...")
        # Tune the data
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
                train = self.data
                accuracy = self.tuneClassification(k, train)
                classifications[accuracy] = k
                count += 1
            
            best = max(classifications.keys())
            self.k = classifications[best]
            print("best k = ", self.k)
        else:
            for k in possibleK:
                for kernel in possibleKernel:
                    train = self.data
                    accuracy = self.tuneRegression(k, kernel, train)
                    classifications[accuracy] = [k, kernel]
                    count += 1

            best = max(classifications.keys())
            self.k = classifications[best][0]
            self.kernel = classifications[best][1]
            print("best k = ", self.k)
            print("best kernel = ", self.kernel)

    def crossValidationRegression(self, cleanDataset, classColumn, printSteps):
        print("Running cross validation with stratification...")
        dataChunks = []
        for i in range(10):
            dataChunks.append(cleanDataset.sample(frac=0.1))
            cleanDataset = cleanDataset.drop(dataChunks[i].index)
        return dataChunks

    def crossValidation(self, cleanDataset, classColumn, printSteps):
        print("Running cross validation with stratification...")
        # 10-fold cross validation with stratification of classes
        if printSteps == True:
            print("Running cross validation with stratification...")
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
        if self.classificationType == "classification":
            self.folds = self.crossValidation(self.data, self.targetPlace, False)
            output =  self.classification()
        else:
            self.folds = self.crossValidationRegression(self.data.copy(), self.targetPlace, False)
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
    
    def classification(self, printSteps=False):
        count = 0
        classification = ClassificationInfo.ClassificationInfo()
        # return the classification info for each dataset
        for fold in self.folds:          
            train = self.data.drop(fold.index)
            total = 0
            sum = 0
            for i in range(fold.shape[0]):
                neighbors = self.findNeighbors(fold.iloc[i], train)
                if(printSteps and count == 0):
                    print("Classifying point ")
                    print(fold.iloc[i])
                    print()
                    print("Neighbors: ")
                    for neighbor in neighbors:
                        print(train.iloc[neighbor[1]])
                correctClass = fold.iloc[i][self.targetPlace]
                assignedClasses = {}
                for neighbor in neighbors:
                    neighborClass = train.iloc[neighbor[1]][self.targetPlace]
                    if neighborClass in assignedClasses:
                        assignedClasses[neighborClass] += 1
                    else:
                        assignedClasses[neighborClass] = 1
                assignedClass = max(assignedClasses, key=assignedClasses.get)
                if(printSteps and count == 0):
                    print("Assigned class: ", assignedClass)
                classAccuracy = self.classificationAccuracy(correctClass, assignedClass)
                if(classAccuracy == Accuracy.TP or classAccuracy == Accuracy.TN):
                    data_df = pd.DataFrame([fold.iloc[i]])
                    self.edited = pd.concat([self.edited, data_df])
                    sum += 1
                total += 1
                classification.addTrueClass([correctClass, assignedClass])
                classification.addConfusion(classAccuracy)
                count += 1

        return classification
    
    def kernelWeight(self, point1, point2, kernel, printSteps=False):
        #return weight
        squaredDist = abs(self.euclideanDistance(point1, point2)) ** 2
        if printSteps:
            print("calculating euclidean distance between two points and taking absolute value squared...")
            print("squared distance: ", squaredDist)
            print("Taking the exponential of the squared distance over 2*kernel^2...")
        return math.exp(-squaredDist / (2 * kernel ** 2))
        

    def regression(self, printSteps=False):
        classification = ClassificationInfo.ClassificationInfo()
        count = 0
        # return the regression classification info 
        # return the classification info for each dataset
        for fold in self.folds: 
            train = self.data.drop(fold.index)
            total = 0
            sum = 0
            for i in range(fold.shape[0]):
                neighbors = self.findNeighbors(fold.iloc[i], train)
                if(printSteps and count == 0):
                    print("Regressing point ")
                    print(fold.iloc[i])
                    print()
                    print("Neighbors: ")
                    for neighbor in neighbors:
                        print(train.iloc[neighbor[1]])
                correctValue = fold.iloc[i][self.targetPlace]
                numerator = 0
                denominator = 0
                for neighbor in neighbors:
                    weight = self.kernelWeight(fold.iloc[i], train.iloc[neighbor[1]], self.kernel)
                    value = train.iloc[neighbor[1]][self.targetPlace]
                    numerator += (weight * value)
                    denominator += weight
                if denominator == 0:
                    assignedValue = 0
                else:
                    assignedValue = numerator/denominator
                if(printSteps and count == 0):
                    print("Assigned value: ", assignedValue)

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

    def euclideanDistance(self, point1, point2, printSteps=False):
        #return distance 
        if printSteps:
            print("Calculating Euclidean distance between two points...")
        featuresSum = 0
        for i in range(self.data.shape[1]):
            if(printSteps):
                print("adding ", (point2.iloc[i] - point1.iloc[i])**2 , "to feature sum")
            featuresSum += (point2.iloc[i] - point1.iloc[i])**2

        if printSteps:
            print("Taking the square root of the feature sum...")
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
                f = self.pointIndex[str(train.iloc[i])]
                j = self.pointIndex[str(point)]
                distance = self.euclidean[f][j]
                distances.append([distance, i])
            distances.sort()
            return distances[:self.k]
        else:
            for i in range(train.shape[0]):
                f = self.pointIndex[str(train.iloc[i])]
                j = self.pointIndex[str(point)]
                distance = self.euclidean[f][j]
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
        if self.classificationType == "classification":
            self.folds = self.crossValidationRegression(self.data.copy(), self.targetPlace, False)
            output =  self.classification()
        else:
            self.folds = self.crossValidationRegression(self.data.copy(), self.targetPlace, False)
            output =  self.regression()
        self.data = copy
        self.folds = copyFolds
        return output
        