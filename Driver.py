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
    print("BREAST CANCER")
    breastCancerData =  fetch_ucirepo(id=15)
    breastCancerDataFrame = pd.DataFrame(breastCancerData.data.original)
    breastCancerClean = Cleaner.cleaner(breastCancerData).cleaner(breastCancerDataFrame, ['Sample_code_number'], 'Class')

    print("GLASS")
    glassData =  fetch_ucirepo(id=42)
    glassDataFrame = pd.DataFrame(glassData.data.original)
    glassClean = Cleaner.cleaner(glassData).cleaner(glassDataFrame, ['Id_number'], 'Type_of_glass')

    print("SOYBEAN")
    soybeanData =  fetch_ucirepo(id=91)
    soybeanDataFrame = pd.DataFrame(soybeanData.data.original)
    soybeanClean = Cleaner.cleaner(soybeanData).cleaner(soybeanDataFrame, [], 'class')

    print("ABALONE")
    abaloneData = fetch_ucirepo(id=1)
    abaloneDataFrame = pd.DataFrame(abaloneData.data.original)
    abaloneClean = Cleaner.cleaner(abaloneData).cleaner(abaloneDataFrame, [], 'Rings')

    print("COMPUTER HARDWARE")
    computerHardwareData =  fetch_ucirepo(id=29)
    computerHardwareDataFrame = pd.DataFrame(computerHardwareData.data.original)
    computerClean = Cleaner.cleaner(computerHardwareData).cleaner(computerHardwareDataFrame, [], 'ERP')

    print("FOREST FIRES")
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

    # # CLASSIFICATION + REGRESSION FOR DATASETS
    # # BREAST CANCER
    # print("BREAST CANCER CODE...")
    # cancerClassLearner = Learner.Learner(computerHardwareDataFrame, "classification", "Class")
    # print("\nbreast cancer regular")
    # cancerClassification = cancerClassLearner.classification()
    # print("\nbreast cancer edited")
    # cancerEdited = cancerClassLearner.editData()
    # print("\nbreast cancer kmeans")
    # cancerKmeans = cancerClassLearner.kmeans()
    #
    # # GLASS
    # print("GLASS CODE...")
    # glassClassLearner = Learner.Learner(glassDataFrame, "classification", "Type_of_glass")
    # print("\nglass regular")
    # glassClassification = glassClassLearner.classification()
    # print("\nglass edited")
    # glassEdited = glassClassLearner.editData()
    # print("\nglass kmeans")
    # glassKmeans = glassClassLearner.kmeans()
    #
    # # SOYBEAN
    # print("SOYBEAN CODE...")
    # soybeanClassLearner = Learner.Learner(soybeanDataFrame, "classification", "class")
    # print("\nsoybean regular")
    # soybeanClassification = soybeanClassLearner.classification()
    # print("\nsoybean edited")
    # soybeanEdited = soybeanClassLearner.editData()
    # print("\nsoybean kmeans")
    # soybeanKmeans = soybeanClassLearner.kmeans()
    #
    # # ABALONE
    # print("ABALONE CODE...")
    # abaloneRegLearner = Learner.Learner(abaloneDataFrame, "regression", "Rings")
    # print("\nabalone regular")
    # abaloneRegression = abaloneRegLearner.regression()
    # print("\nabalone edited")
    # abaloneEdited = abaloneRegLearner.editData()
    # print("\nabalone kmeans")
    # abaloneKmeans = abaloneRegLearner.kmeans()
    #
    # # COMPUTER HARDWARE
    # print("COMPUTER HARDWARE CODE...")
    # compRegLearner = Learner.Learner(computerHardwareDataFrame, "regression", "ERP")
    # print("\ncomputer hardware regular")
    # compRegression = compRegLearner.regression()
    # print("\ncomputer hardware edited")
    # compEdited = compRegLearner.editData()
    # print("\ncomputer hardware kmeans")
    # compKmeans = compRegLearner.kmeans()
    #
    # # FOREST FIRE
    # print("FOREST FIRE CODE...")
    # forestRegLearner = Learner.Learner(forestFiresDataFrame, "regression", "ERP")
    # print("\nforest fire regular")
    # forestRegression = forestRegLearner.regression()
    # print("\nedited")
    # forestEdited = forestRegLearner.editData()
    # print("\ncomputer hardware kmeans")
    # forestKmeans = forestRegLearner.kmeans()

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