import pandas as pd
import Learner
from ucimlrepo import fetch_ucirepo

import ssl
import urllib.request


def main():
    ssl._create_default_https_context = ssl._create_unverified_context

    breastCancerData =  fetch_ucirepo(id=15)
    breastCancerDataFrame = pd.DataFrame(breastCancerData.data.original)

    glassData =  fetch_ucirepo(id=42)
    glassrDataFrame = pd.DataFrame(glassData.data.original)

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

    print(computerHardwareDataFrame.head())

    testRegression = Learner.Learner(computerHardwareDataFrame, "regression", "ERP")

    print()
    print("regular")
    testRegression.regression()

    print()
    print("edited")
    testRegression.editData()

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