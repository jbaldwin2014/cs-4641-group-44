import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib


# read the data
df = pd.read_csv(r'C:\Users\ibrad\PycharmProjects\COVID_predProj\datasets\COVID_Data.csv')

# shape and data types of the data
print(df.shape)
print(df.dtypes)

# select columns
featuresArr = df.columns.values
print('featureArr:' + featuresArr)

# deleting unneeded features, which would be the first 3 columns (dates)
cols_to_drop = featuresArr[:3]
df = df.drop(cols_to_drop, axis=1)
print(df.columns.values)


print(df.shape)
print(df.dtypes)



