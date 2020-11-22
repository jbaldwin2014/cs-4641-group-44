from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import pandas as pd

from yellowbrick.datasets import load_concrete
from yellowbrick.regressor import prediction_error

from sklearn.linear_model import LassoCV
from yellowbrick.regressor.alphas import alphas

from yellowbrick.datasets import load_energy

# Load dataset
X, y = load_energy() # make our dataset read as x and y axis values somehow and replace this dataset with ours
# X = []                         # makes a list
# y = []                         # makes a list
# data = pd.read_csv(r'C:\Users\Jujin\Desktop\cs-4641-group-44\cleaned_encoded_COVID_Data_Copy.csv')
# for row in data:
#     X.append(row[1])     # selects data from the ith row
#     y.append(row[2])     # selects data from the ith row

# Use the quick method and immediately show the figure
alphas(LassoCV(random_state=0), X, y)

# Load a regression dataset
X, y = load_concrete()  # same as above

#X = pd.read_csv(r'C:\Users\Jujin\Desktop\cs-4641-group-44\training_data.csv')
#y = pd.read_csv(r'C:\Users\Jujin\Desktop\cs-4641-group-44\test_data.csv')

# X_train = []                         # makes a list
# y_train = []                         # makes a list
# d = pd.read_csv(r'C:\Users\Jujin\Desktop\cs-4641-group-44\training_data.csv')
# for row in d:
#     X_train.append(row[1])     # selects data from the ith row
#     y_train.append(row[2])     # selects data from the ith row
#
# X_test = []                         # makes a list
# y_test = []                         # makes a list
# da = pd.read_csv(r'C:\Users\Jujin\Desktop\cs-4641-group-44\test_data.csv')
# for row in da:
#     X_test.append(row[1])     # selects data from the ith row
#     y_test.append(row[2])     # selects data from the ith row

# Create the train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate the linear model and visualizer
model = Lasso()
visualizer = prediction_error(model, X_train, y_train, X_test, y_test)
