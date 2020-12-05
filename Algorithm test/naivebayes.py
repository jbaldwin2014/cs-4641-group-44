from sklearn.naive_bayes import GaussianNB
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import numpy as np
import sklearn.metrics as f1
# import seaborn as sns


# create data frame containing your data, each column can be accessed # by df['column   name']
df = pd.read_csv(r'C:\Users\Jujin\Desktop\cs-4641-group-44\training_data.csv')
df2 = pd.read_csv(r'C:\Users\Jujin\Desktop\cs-4641-group-44\test_data.csv')

training_length = (len(df)+len(df2))*0.2

training_data_plot = df.iloc[:int(training_length),:]
test_data_plot = df2.iloc[int(training_length)+1:,:]

training_data = df
test_data = df2

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

print("f1 score: ", f1.f1_score(expected_labels, predicted_labels))

# classifier2 = GaussianNB()
# classifier2.fit(training_data_plot.iloc[:,:-2], training_data_plot.iloc[:,-1])
#
# predicted_labels_plot = classifier2.predict(test_data_plot.iloc[:,:-2])
#
# expected_labels_plot = test_data_plot.iloc[:,-1]

# plt.plot(predicted_labels_plot, expected_labels_plot)
# plt.show()

# numbers = pd.Series(df.columns)
# df[numbers].hist(figsize = (14, 14))
# plt.show()

# data = pd.DataFrame([predicted_labels_plot, expected_labels_plot], columns=['A', 'B'])
#
# sns.pairplot(data[['A', 'B']], diag_kind="kde")
# plt.show()



def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[1].set_title(title)

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")

    return plt


fig, axes = plt.subplots(1, 2, figsize=(10, 15))

X = df
y = None

title = "Scalability of the model (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = GaussianNB()
estimator.fit(training_data.iloc[:,:-2], training_data.iloc[:,-1])
plot_learning_curve(estimator, title, X, y, axes=None, ylim=(0.7, 1.01),
                    cv=cv, n_jobs=2)

title = r"Scalability of the model (SVM, RBF kernel, $\gamma=0.001$)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
estimator = SVC(gamma=0.001)
plot_learning_curve(estimator, title, X, y, axes=None, ylim=(0.7, 1.01),
                    cv=cv, n_jobs=2)

plt.show()
