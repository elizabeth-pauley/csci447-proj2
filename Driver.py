import pandas as pd
import Learner
import Cleaner
import AlgorithmAccuracy
from ucimlrepo import fetch_ucirepo

import ssl
import urllib.request


def main():
    ssl._create_default_https_context = ssl._create_unverified_context

    # INITIALIZE DATASETS
    print("FETCH AND CLEAN DATASETS...")
    breastCancerData =  fetch_ucirepo(id=15)
    breastCancerDataFrame = pd.DataFrame(breastCancerData.data.original)
    breastCancerClean = Cleaner.cleaner(breastCancerData).cleaner(breastCancerDataFrame, ['Sample_code_number'], 'Class')

    glassData =  fetch_ucirepo(id=42)
    glassDataFrame = pd.DataFrame(glassData.data.original)
    glassClean = Cleaner.cleaner(glassData).cleaner(glassDataFrame, ['Id_number'], 'Type_of_glass')

    soybeanData =  fetch_ucirepo(id=91)
    soybeanDataFrame = pd.DataFrame(soybeanData.data.original)
    soybeanClean = Cleaner.cleaner(soybeanData).cleaner(soybeanDataFrame, [], 'class')

    abaloneData = fetch_ucirepo(id=1)
    abaloneDataFrame = pd.DataFrame(abaloneData.data.original)
    abaloneClean = Cleaner.cleaner(abaloneData).cleaner(abaloneDataFrame, [], 'Rings')

    computerHardwareData =  fetch_ucirepo(id=29)
    computerHardwareDataFrame = pd.DataFrame(computerHardwareData.data.original)
    computerClean = Cleaner.cleaner(computerHardwareData).cleaner(computerHardwareDataFrame, [], 'ERP')

    forestFiresData =  fetch_ucirepo(id=162)
    forestFiresDataFrame = pd.DataFrame(forestFiresData.data.original)
    forestClean = Cleaner.cleaner(forestFiresData).cleaner(forestFiresDataFrame, [], 'area')

    test = Learner.Learner(glassClean, "classification", "Type_of_glass")
    
    classification = test.classification(True)
    glassAccuracy = AlgorithmAccuracy.AlgorithmAccuracy(classification, glassDataFrame.shape[1], "Glass")

    print()
    print("edited")
    edited = test.editData()
    editedGlassAccuracy = AlgorithmAccuracy.AlgorithmAccuracy(edited, glassDataFrame.shape[1], "Glass")

    print()
    print("kmeans")
    kmeans = test.kmeans()
    kmeansGlassAccuracy = AlgorithmAccuracy.AlgorithmAccuracy(kmeans, glassDataFrame.shape[1], "Glass")

    print("regular")
    print("F1: " + str(glassAccuracy.getF1()))
    print("Loss: " + str(glassAccuracy.getLoss()))
    print()

    print("edited")
    print("F1: " + str(editedGlassAccuracy.getF1()))
    print("Loss: " + str(editedGlassAccuracy.getLoss()))
    print()

    print("kmeans")
    print("F1: " + str(kmeansGlassAccuracy.getF1()))
    print("Loss: " + str(kmeansGlassAccuracy.getLoss()))
    print()

    print("CLASSIFICATION, REGRESSION, KMEANS...")
    # CLASSIFICATION + REGRESSION FOR DATASETS
    # BREAST CANCER
    print("BREAST CANCER CODE...")


    print("BREAST CANCER CODE")
    plot("Breast Cancer", breastCancerDataFrame, breastCancerClean, "classification", "Class")
    print("GLASS CODE")
    plot("Glass Identification", glassDataFrame, glassClean, "classification", "Type_of_glass")
    print("SOYBEAN CODE")
    plot("Soybean", soybeanDataFrame, soybeanClean, "classification", "class")
    print("ABALONE CODE")
    plot("Abalone", abaloneDataFrame, abaloneClean, "regression", "Rings")
    print("COMPUTER HARDWARE CODE")
    plot("Computer Hardware", computerHardwareDataFrame, computerClean, "regression", "ERP")
    print("FOREST FIRES CODE")
    plot("Forest Fires", forestFiresDataFrame, forestClean, "regression", "area")

    # END CLASSIFICATION + REGRESSION FOR DATASETS

    testRegression = Learner.Learner(computerClean, "regression", "ERP")

    print("regular")
    computer = testRegression.regression(True)
    computerAccuracy = AlgorithmAccuracy.AlgorithmAccuracy(computer, computerHardwareDataFrame.shape[1], "Computer Hardware")

    print()
    print("edited")
    computerEdited = testRegression.editData()
    computerEditedAccuracy = AlgorithmAccuracy.AlgorithmAccuracy(computerEdited, computerHardwareDataFrame.shape[1], "Computer Hardware")

    print("kmeans")
    computerKmeans = testRegression.kmeans()
    computerKmeansAccuracy = AlgorithmAccuracy.AlgorithmAccuracy(computerKmeans, computerHardwareDataFrame.shape[1], "Computer Hardware")

    print("regular")
    print("F1: " + str(computerAccuracy.getF1()))
    print("Loss: " + str(computerAccuracy.getLoss()))
    print()

    print("edited")
    print("F1: " + str(computerEditedAccuracy.getF1()))
    print("Loss: " + str(computerEditedAccuracy.getLoss()))
    print()

    print("kmeans")
    print("F1: " + str(computerKmeansAccuracy.getF1()))
    print("Loss: " + str(computerKmeansAccuracy.getLoss()))
    print()

    #print("regular")

    #test = Learner.Learner(breastCancerDataFrame, "classification", "Class")
    

    #classification = test.classification()
    #print()
    #print("edited")
    #edited = test.editData()
    #print()
    #print("kmeans")
    #kmeans = test.kmeans()

main()


def plot(title, dataFrame, dataClean, typeAlg, classColName):
    dataClassLearner = Learner.Learner(dataFrame, typeAlg, classColName)
    dataClassification = dataClassLearner.classification()
    dataClassificationAccuracy = AlgorithmAccuracy.AlgorithmAccuracy(dataClassification, len(dataClean[0].columns),f"{title} Initial K-NN")
    dataEdited = dataClassLearner.editData()
    dadtaEditedAccuracy = AlgorithmAccuracy.AlgorithmAccuracy(dataEdited, len(dataClean[0].columns),f"{title} Edited K-NN")
    dataKmeans = dataClassLearner.kmeans()
    dataKmeansAccuracy = AlgorithmAccuracy.AlgorithmAccuracy(dataKmeans, len(dataClean[0].columns),f"{title} K-Means K-NN")