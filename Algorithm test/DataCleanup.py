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


# drop rows with missing or unknown values.
# drop rows with missing or unknown values.
df = df[df.age_group != 'Missing']
df = df[df.age_group != 'Unknown']
df = df[df.age_group != 'NA']
df = df[df.death_yn != 'Missing']
df = df[df.death_yn != 'Unknown']
df = df[df.death_yn != 'NA']
df = df[df.medcond_yn != 'Missing']
df = df[df.medcond_yn != 'Unknown']
df = df[df.medcond_yn != 'NA']





# deleting unneeded features, which would be the first 3 columns
cols_to_drop = featuresArr[:3]
df = df.drop(cols_to_drop, axis=1)
print(df.columns.values)
print(df.shape)

# Drop rows with any empty cells
df.dropna(
    axis=0,
    how='any',
    thresh=None,
    subset=None,
    inplace=True
)

df.to_csv(r'C:\Users\ibrad\PycharmProjects\cleaned_COVID_Data.csv')




