from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv(r'C:\Users\shubh\PycharmProjects\cs4641\cleaned_encoded_COVID_Data.csv')

x_train ,x_test = train_test_split(df,test_size=0.5)

pd.DataFrame(x_train).to_csv(r'C:\Users\shubh\PycharmProjects\cs4641\training_data.csv')
pd.DataFrame(x_test).to_csv(r'C:\Users\shubh\PycharmProjects\cs4641\test_data.csv')
