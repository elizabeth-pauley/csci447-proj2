import math
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class cleaner:
    def __init__(self, dataset):
        self.dataset = dataset

    def cleaner(self, dataFrame, dropColumns, classCol):
        # ADDRESS NULL VALUES WHERE COLUMNS/ROWS NEED TO BE REMOVED
        # If true class is unknown, drop the row
        cleanedData = dataFrame.dropna(subset=[classCol])
        # Drop any rows where all values are null
        cleanedData = cleanedData.dropna(how = 'all')
        # Columns must have 70% of their values for rows to remain in dataset
        cleanedData = cleanedData.dropna(axis=1, thresh = math.floor(0.70*cleanedData.shape[0]))

        # Remove unnecessary columns (i.e., ID columns)
        if len(dropColumns) > 0:
            cleanedData = cleanedData.drop(columns=dropColumns, axis = 1)

        # Get list of categorical column names
        classColDataFrame = cleanedData[classCol]
        cleanedData = cleanedData.drop(columns=classCol, axis = 1)
        categoricalColumns = cleanedData.select_dtypes(exclude=['int', 'float']).columns.tolist()

        # One Hot Encoding
        cleanedData = pd.get_dummies(cleanedData, columns=categoricalColumns, dtype=int)

        # Iterate through columns and fill NaN values for string/object columns
        for col in cleanedData.select_dtypes(include='object'):
            modeVal = cleanedData[col].mode()[0]  # Get the mode (most frequent value)
            cleanedData[col] = cleanedData[col].fillna(modeVal)  # Fill NaN with the mode

        # Fill all other na values with the mean of each column
        cleanedData = cleanedData.fillna(cleanedData.mean())
        cleanedData[classCol] = classColDataFrame

        return cleanedData