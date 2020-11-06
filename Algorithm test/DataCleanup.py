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

# rename race column to better format
df = df.rename(columns={'Race and ethnicity (combined)': 'race_ethnicity_combo'})

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
df = df[df.current_status != 'Probable Case']
df = df[df.hosp_yn != 'Missing']
df = df[df.hosp_yn != 'Unknown']
df = df[df.race_ethnicity_combo != 'Unknown']
df = df[df.race_ethnicity_combo != 'NA']





# deleting unneeded features, which would be the first 3 columns
cols_to_drop = featuresArr[:3]
df = df.drop(cols_to_drop, axis=1)
print(df.columns.values)
print(df.shape)

# Drop rows with any empty cells in them
df.dropna(
    axis=0,
    how='any',
    thresh=None,
    subset=None,
    inplace=True
)

# integer encoding
encode_medcond = {"medcond_yn": {"Yes": 1, "No": 0}}
print(encode_medcond)
df = df.replace(encode_medcond)

encode_death = {"death_yn": {"Yes": 1, "No": 0}}
print(encode_death)
df = df.replace(encode_death)

encode_sex = {"sex": {"Male": 1, "Female": 0}}
print(encode_sex)
df = df.replace(encode_sex)

encode_age_group = {"age_group": {"0 - 9 Years": 0, "10 - 19 Years": 1, "20 - 29 Years": 2, "30 - 39 Years": 3, "40 - 49 Years": 4, "50 - 59 Years": 5, "60 - 69 Years": 6, "70 - 79 Years": 7, "80+ Years": 8}} # dont know how to 1HE this yet
print(encode_age_group)
df = df.replace(encode_age_group)
print(df["age_group"].value_counts())

encode_race = {"race_ethnicity_combo": {"White, Non-Hispanic": 0, "Hispanic/Latino": 1, "Black, Non-Hispanic": 2, "Multiple/Other, Non-Hispanic": 3, "Asian, Non-Hispanic": 4, "American Indian/Alaska Native, Non-Hispanic": 5, "Native Hawaiian/Other Pacific Islander, Non-Hispanic": 6}} # dont know how to 1HE this yet
print(encode_race)
df = df.replace(encode_race)


df.to_csv(r'C:\Users\ibrad\PycharmProjects\cleaned_encoded_COVID_Data.csv')


