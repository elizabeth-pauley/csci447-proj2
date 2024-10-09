import pandas as pd
import Learner
from ucimlrepo import fetch_ucirepo

import ssl
import urllib.request


def main():
    ssl._create_default_https_context = ssl._create_unverified_context

    breastCancerData =  fetch_ucirepo(id=15)
    breastCancerDataFrame = pd.DataFrame(breastCancerData.data.original)

    abalone = fetch_ucirepo(id=1) 
    abaloneDataFrame = pd.DataFrame(abalone.data.original)
    abaloneDataFrame = abaloneDataFrame.drop('Sex', axis=1)

    print(abaloneDataFrame.head())

    testRegression = Learner.Learner(abaloneDataFrame, "regression", "Rings")

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