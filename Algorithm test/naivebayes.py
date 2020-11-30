from sklearn.naive_bayes import GaussianNB
import pandas as pd
#import numpy as np

# create data frame containing your data, each column can be accessed # by df['column   name']
df = pd.read_csv(r'C:\Users\Jujin\Desktop\cs-4641-group-44\training_data.csv')

training_length = len(df)*0.67

training_data = df.iloc[:int(training_length),:]
test_data = df.iloc[int(training_length)+1:,:]

'''
Different Classifiers can be used. As all the classifiers have been imported.
Also Accuracy for this Dataset can be calculated for all the different classifiers.
'''


classifier = GaussianNB()
classifier.fit(training_data.iloc[:,:-2], training_data.iloc[:,-1])

predicted_labels = classifier.predict(test_data.iloc[:,:-2])

expected_labels = test_data.iloc[:,-1]

print("expected: ", expected_labels, "\n\n", "predicted: ", predicted_labels)

accuracy = classifier.score(test_data.iloc[:,:-2], expected_labels)

print("accuracy: ", accuracy)

# target_names = np.array(['Positives','Negatives'])
#
# # add columns to your data frame
# df['is_train'] = np.random.uniform(0, 1, len(df)) <= 0.75
# df['Type'] = pd.Factor(targets, target_names)
# df['Targets'] = targets
#
# # define training and test sets
# train = df[df['is_train']==True]
# test = df[df['is_train']==False]
#
# trainTargets = np.array(train['Targets']).astype(int)
# testTargets = np.array(test['Targets']).astype(int)
#
# # columns you want to model
# features = df.columns[0:7]
#
# # call Gaussian Naive Bayesian class with default parameters
# gnb = GaussianNB()
#
# # train model
# y_gnb = gnb.fit(train[features], trainTargets).predict(train[features])
