import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib


import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib

# read the data
df = pd.read_csv(r'C:\Users\ibrad\PycharmProjects\Datasets\COVID_Data.csv')

# shape and data types of the data
print(df.shape)
print(df.dtypes)

# select columns
featuresArr = df.columns.values
print(featuresArr)


# Drop rows with any empty cells
df.dropna(
    axis=0,
    how='any',
    thresh=None,
    subset=None,
    inplace=True
)


# drop rows with missing or unknown values.
# drop rows with missing or unknown values.
df = df[df.icu_yn != 'Missing']
df = df[df.icu_yn != 'Unknown']
df = df[df.icu_yn != 'NA']

# drop rows with missing or unknown values.
for i in range(len(featuresArr)):
    ind_missing = df[df[featuresArr[i]] == 'Missing'].index
    df.drop(ind_missing, axis=0)
    ind_missing2 = df[df[featuresArr[i]] == 'Unknown'].index
    df.drop(ind_missing2, axis=0)
    ind_missing3 = df[df[featuresArr[i]] == 'NA'].index
    df.drop(ind_missing3, axis=0)
    print(featuresArr[i])



# deleting unneeded features, which would be the first 3 columns
cols_to_drop = featuresArr[:3]
df = df.drop(cols_to_drop, axis=1)
print(df.columns.values)
print(df.shape)

df.to_csv(r'C:\Users\ibrad\PycharmProjects\cleaned_COVID_Data.csv')




