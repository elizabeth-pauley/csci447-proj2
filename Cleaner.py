import math
import numpy as np
import pandas as pd

class cleaner:
    def __init__(self, dataset):
        self.dataset = dataset

    def cleaner(self, dataFrame, categoricalColumns):

        classColumnName = self.dataset.variables.loc[self.dataset.variables['role'] == 'Target', 'name'].values[0]
        columnTypes = dict(zip(self.dataset.variables['name'], self.dataset.variables['type']))

        # ADDRESS NULL VALUES WHERE COLUMNS/ROWS NEED TO BE REMOVED
        # If true class is unknown, drop the row
        dataFrame = dataFrame.dropna(subset=[classColumnName])
        # Drop any rows where all values are null
        dataFrame = dataFrame.dropna(how = 'all')
        # Columns must have 70% of their values for rows to remain in dataset
        dataFrame = dataFrame.dropna(axis=1, thresh = math.floor(0.70*dataFrame.shape[0]))

        # BINNING OF CONTINUOUS/CATEGORICAL COLUMNS
        for columnName in dataFrame.columns:
            columnRole = self.dataset.variables.loc[self.dataset.variables['name'] == columnName, 'role'].values[0]

            # Ignore class column (target)
            if columnRole != 'Target':
                # if continuous, make categorical (5 categories total)
                if columnTypes[columnName] == 'Continuous':
                    # split dataset into 5 equal-width bins
                    binVals = np.linspace(dataFrame[columnName].min(), dataFrame[columnName].max(), 6)
                    binLabels = ['A', 'B', 'C', 'D', 'E']

                    # assign column vals to a bin
                    dataFrame[columnName] = pd.cut(dataFrame[columnName], bins = binVals, labels = binLabels, include_lowest = True)
                    # set 'type' to Categorical
                    columnTypes[columnName] = 'Categorical'

                if columnTypes[columnName] == 'Categorical':
                    dataFrame = dataFrame.ffill()
                else:
                    # fill na's with the rounded mean of the column (whole numbers will work w/ ints and floats)
                    dataFrame = dataFrame.fillna(round(dataFrame[columnName].mean()))

        print('test')

        # One Hot Encoding
        encodedDataFrame = pd.get_dummies(dataFrame, columns=categoricalColumns)
        
        return encodedDataFrame