# Step 1: Import packages, functions, and classes
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib

from yellowbrick.regressor import prediction_error

from sklearn.linear_model import LassoCV
from yellowbrick.regressor.alphas import alphas


# Step 2: Get data
#df = pd.read_csv(r'C:\Users\shubh\PycharmProjects\cs4641\training_data.csv')  # read the data
df = pd.read_csv(r'C:\Users\Jujin\Desktop\cs-4641-group-44\training_data.csv')
npdata = df.to_numpy()
x = npdata[:,3:]  # x = covid data mtx
x = np.delete(x, 5, 1)  # delete death col
ytemp = npdata[:,8]  # y = death label data

y = np.zeros(len(ytemp))
for i in range(len(ytemp)): # to fix mixed datatype problem
    if ytemp[i] == 1:
        y[i] = 1

dftest = pd.read_csv(r'C:\Users\Jujin\Desktop\cs-4641-group-44\test_data.csv')  # read the data
npdatatest = dftest.to_numpy()
xtest = npdatatest[:,3:]  # x = covid data mtx
xtest = np.delete(xtest, 5, 1)  # delete death col
ytemptest = npdatatest[:,8]  # y = death label data

ytest = np.zeros(len(ytemptest))
for i in range(len(ytemptest)): # to fix mixed datatype problem
    if ytemptest[i] == 1:
        ytest[i] = 1


# Step 3: Create a model and train it
model = LogisticRegression(solver='liblinear', random_state=0)
model.fit(x, y)

# Step 4: Evaluate the model
p_pred = model.predict_proba(x)
y_pred = model.predict(x)
score_ = model.score(x, y)
conf_m = confusion_matrix(y, y_pred)
report = classification_report(y, y_pred)

# Print stuff
print('p_pred:', p_pred, sep='\n', end='\n\n')
print('y_pred:', y_pred, end='\n\n')
print('score_:', score_, end='\n\n')
print('conf_m:', conf_m, sep='\n', end='\n\n')
print('report:', report, sep='\n')
print(model.coef_)
print(model.intercept_)

#plt.plot(x[:,0], y)
#plt.show()

print(model.score(xtest,ytest))

alphas(LassoCV(random_state=0), x, y)
model2 = Lasso()
visualizer = prediction_error(model2, x, y, xtest, ytest)
