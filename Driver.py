import pandas as pd
import Learner
import AlgorithmAccuracy
from ucimlrepo import fetch_ucirepo

import ssl
import urllib.request


def main():
    ssl._create_default_https_context = ssl._create_unverified_context

    breastCancerData =  fetch_ucirepo(id=15)
    breastCancerDataFrame = pd.DataFrame(breastCancerData.data.original)
    breastCancerDataFrame = breastCancerDataFrame.drop('Sample_code_number', axis=1)

    glassData =  fetch_ucirepo(id=42)
    glassDataFrame = pd.DataFrame(glassData.data.original)

    test = Learner.Learner(glassDataFrame, "classification", "Type_of_glass")
    
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

    soybeanData =  fetch_ucirepo(id=91)
    soybeanDataFrame = pd.DataFrame(soybeanData.data.original)


    abalone = fetch_ucirepo(id=1) 
    abaloneDataFrame = pd.DataFrame(abalone.data.original)

    computerHardwareData =  fetch_ucirepo(id=29)
    computerHardwareDataFrame = pd.DataFrame(computerHardwareData.data.original)

    forestFiresData =  fetch_ucirepo(id=162)
    forestFiresDataFrame = pd.DataFrame(forestFiresData.data.original)

    computerHardwareDataFrame = computerHardwareDataFrame.drop('VendorName', axis=1)
    computerHardwareDataFrame = computerHardwareDataFrame.drop('ModelName', axis=1)

    testRegression = Learner.Learner(computerHardwareDataFrame, "regression", "ERP")

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