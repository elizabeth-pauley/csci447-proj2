import pandas as pd
import Learner
from ucimlrepo import fetch_ucirepo

import ssl
import urllib.request


def main():
    ssl._create_default_https_context = ssl._create_unverified_context

    breastCancerData =  fetch_ucirepo(id=15)
    breastCancerDataFrame = pd.DataFrame(breastCancerData.data.original)

    test = Learner.Learner(breastCancerDataFrame, "classification", breastCancerDataFrame.shape[0], "Class")
    classification = test.classification()
    print()
    edited = test.editData()
    

main()