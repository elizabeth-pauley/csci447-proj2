class Learner:
    
    def __init__(self, data, classificationType):
        self.classificationType = classificationType
        self.tuningData = self.getTuneData()
        self.data = data.drop(self.tuningData)
        self.k = self.tuneData()
        self.folds = self.crossValidation()
    
    def getTuneData(self):
        # split data for tuning
        pass

    def tuneData(self):
        # Tune the data
        pass

    def crossValidation(self):
        # return 10 equal sized folds 
        pass

    def editData(self):
        # return the edited data
        pass

    def classification(k):
        # return the classification info for each dataset
        pass

    def regression(k):
        # return the regression classification info 
        pass

    def euclideanDistance(point1, point2):
        #return distance 
        pass
    