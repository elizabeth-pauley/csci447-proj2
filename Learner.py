class Learner:
    
    def __init__(self, data, classificationType):
        self.classificationType = classificationType
        self.tuningData = self.getTuneData()
        self.data = data.drop(self.tuningData)
    
    def getTuneData(self):
        # split data for tuning
        pass

    def tuneData(self):
        # Tune the data
        pass