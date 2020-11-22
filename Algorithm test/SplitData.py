from sklearn.model_selection import train_test_split
import random
import numpy
import pandas as pd

with open(r'C:\Users\shubh\PycharmProjects\cs4641\cleaned_encoded_COVID_Data.csv', 'rb') as f:
   data = f.read().decode().split('\n')
   random.shuffle(data)
   data = numpy.array(data)  #convert array to numpy type array

   x_train ,x_test = train_test_split(data,test_size=0.5)

   pd.DataFrame(x_train).to_csv(r'C:\Users\shubh\PycharmProjects\cs4641\training_data.csv')
   pd.DataFrame(x_test).to_csv(r'C:\Users\shubh\PycharmProjects\cs4641\test_data.csv')
