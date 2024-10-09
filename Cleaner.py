import math
import numpy as np
import pandas as pd

class cleaner:
    def __init__(self, dataset):
        self.dataset = dataset

    def cleaner(self, dataFrame, dropColumns):

        columnTypes = dict(zip(self.dataset.variables['name'], self.dataset.variables['type']))

        # ADDRESS NULL VALUES WHERE COLUMNS/ROWS NEED TO BE REMOVED
        # If true class is unknown, drop the row
        for column in dropColumns:
            dataFrame = dataFrame.dropna(subset=[column])
        # Drop any rows where all values are null
        dataFrame = dataFrame.dropna(how = 'all')
        # Columns must have 70% of their values for rows to remain in dataset
        dataFrame = dataFrame.dropna(axis=1, thresh = math.floor(0.70*dataFrame.shape[0]))

        # Remove class/id columns
        dataFrame = dataFrame.drop(columns=dropColumns, axis = 1)

        # Fill remaining empty values with mean of column
        # for column in dataFrame.columns:
        #     dataFrame[column] = dataFrame[column].fillna(dataFrame[column].mean())

        # One Hot Encoding
        categoricalColumns = []
        for columnName in dataFrame.columns:
            columnRole = self.dataset.variables.loc[self.dataset.variables['name'] == columnName, 'role'].values[0]

            #ignore class column
            if columnRole != 'Target':
                if (columnTypes[columnName] == 'Categorical'):
                    # add to list to one-hot encode
                    categoricalColumns.append(columnName)

        encodedDataFrame = pd.get_dummies(dataFrame, columns=categoricalColumns)
        
        return encodedDataFrame