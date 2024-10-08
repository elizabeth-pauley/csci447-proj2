from enum import Enum
import math
import pandas as pd
import numpy as np
import ClassificationInfo

class Accuracy(Enum):
    TP = 1
    TN = 2
    FP = 3
    FN = 4
    
class Learner:
    
    def __init__(self, data, classificationType, size, targetPlace):
        self.classificationType = classificationType
        self.size = size
        self.targetPlace = targetPlace
        self.threshold = None
        if(self.classificationType == "regression"):
            self.setThreshold(data)
        self.tuningData = self.getTuneData(data)
        self.data = data.drop(self.tuningData.index)
        self.k = self.tuneData()
        self.folds = self.crossValidation(self.data, self.targetPlace, False)
        self.edited = pd.DataFrame()

    def setThreshold(self, data):
        colAverage = data[self.targetPlace].mean()
        self.threshold = colAverage * 0.1
    
    def getTuneData(self, data):
        print("Splitting data for tuning...")
        # split data for tuning
        tune_data = data.sample(frac=0.1)
        return tune_data

    def tuneData(self):
        # Tune the data
        print("Tuning data...")
        center = round(math.sqrt(self.size))
        possibleK = list(range(center - 5, center + 5))
        percentCorrect = 0
        bestK = -1
        folds = self.crossValidation(self.tuningData, self.targetPlace, False)
        
        if self.classificationType == "classification":
            for k in possibleK:
                classification = self.classification(k, folds)
                if ((classification.getTP() + classification.getTN())/ (classification.getFP() + classification.getTN() + classification.getTP() + classification.getFN())) > percentCorrect:
                    percentCorrect = (classification.getTP() + classification.getTN())/ (classification.getFP() + classification.getTN() + classification.getTP() + classification.getFN())
                    bestK = k
        else:
            for k in possibleK:
                regression = self.regression(k, folds)
                if ((regression.getTP() + regression.getTN())/ (regression.getFP() + regression.getTN() + regression.getTP() + regression.getFN())) > percentCorrect:
                    percentCorrect = (regression.getTP() + regression.getTN())/ (regression.getFP() + regression.getTN() + regression.getTP() + regression.getFN())
                    bestK = k

        print("Best K value is: " + str(bestK))
        return bestK

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

    def classification(self, k=-1, folds=None):
        classification = ClassificationInfo.ClassificationInfo()
        # return the classification info for each dataset
        if(folds != None):
            for fold in folds: 
                total = 0
                sum = 0
                train = self.tuningData.drop(fold.index)
                for i in range(fold.shape[0]):
                    neighbors = self.findNeighbors(fold.iloc[i],train, k)
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
                        sum += 1
                        total += 1
                    classification.addTrueClass([correctClass, assignedClass])
                    classification.addConfusion(classAccuracy)
                print("fold accuracy: ", sum/total)
                print() 

        else:
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

    def regression(self, k=-1 , folds = None):
        classification = ClassificationInfo.ClassificationInfo()
        if(folds == None):
        # return the regression classification info 
        # return the classification info for each dataset
            for fold in self.folds: 
                train = self.data.drop(fold)
                for i in range(fold.shape[0]):
                    neighbors = self.findNeighbors(fold.iloc[i], train)
                    correctValue = fold.iloc[i][self.targetPlace]
                    total = 0
                    sum = 0
                    for neighbor in neighbors:
                        neighborValue = train.iloc[neighbor[1]][self.targetPlace]
                        sum += neighborValue
                        total += 1
                    assignedValue = sum/total

                    classAccuracy = self.regressionAccuracy(correctValue, assignedValue)
                    if(classAccuracy == Accuracy.TP or classAccuracy == Accuracy.TN):
                        data_df = pd.DataFrame([fold.iloc[i]])
                        self.edited = pd.concat([self.edited, data_df])
                    classification.addTrueClass([correctValue, assignedValue])
                    classification.addConfusion(classAccuracy)
        else:
            for fold in folds: 
                train = self.tuningData.drop(fold)
                for i in range(fold.shape[0]):
                    neighbors = self.findNeighbors(fold.iloc[i], train ,k)
                    correctValue = fold.iloc[i][self.targetPlace]
                    total = 0
                    sum = 0
                    for neighbor in neighbors:
                        neighborValue = train.iloc[neighbor[1]][self.targetPlace]
                        sum += neighborValue
                        total += 1
                    assignedValue = sum/total

                    classAccuracy = self.regressionAccuracy(correctValue, assignedValue)
                    classification.addTrueClass([correctValue, assignedValue])
                    classification.addConfusion(classAccuracy)

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
                distances.append((self.euclideanDistance(point, train.iloc[i]), i))
            distances.sort()
            return distances[:self.k]
        else:
            for i in range(train.shape[0]):
                distances.append((self.euclideanDistance(point, train.iloc[i]), i))
            distances.sort()
            return distances[:k]