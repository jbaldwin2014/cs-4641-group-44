from sklearn.naive_bayes import GaussianNB
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import numpy as np
from matplotlib.ticker import PercentFormatter
import sklearn.metrics as f1
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, plot_confusion_matrix, plot_precision_recall_curve
# import seaborn as sns


# create data frame containing your data, each column can be accessed # by df['column   name']
df = pd.read_csv(r'C:\Users\Jujin\Desktop\cs-4641-group-44\training_data.csv')
df2 = pd.read_csv(r'C:\Users\Jujin\Desktop\cs-4641-group-44\test_data.csv')
df3 = pd.read_csv(r'C:\Users\Jujin\Desktop\cs-4641-group-44\cleaned_encoded_COVID_Data.csv')


# For baseline/guesswork performance
# df = pd.read_csv(r'C:\Users\Jujin\Desktop\cs-4641-group-44\training_data_guess.csv')

x = df3[['sex', 'age_group', 'race_ethnicity_combo', 'hosp_yn', 'icu_yn', 'medcond_yn']]
y = df3[['death_yn']]

# Splitting data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.2)

# For baseline/guesswork performance
# x_train = df[['sex', 'age_group', 'race_ethnicity_combo', 'hosp_yn', 'icu_yn', 'medcond_yn']]
# x_test = df2[['sex', 'age_group', 'race_ethnicity_combo', 'hosp_yn', 'icu_yn', 'medcond_yn']]
# y_train = df[['death_yn']]
# y_test = df2[['death_yn']]

# training_data = df
# test_data = df2

# classifier = GaussianNB()
# classifier.fit(training_data.iloc[:,:-2], training_data.iloc[:,-1])
#
# predicted_labels = classifier.predict(test_data.iloc[:,:-2])
#
# expected_labels = test_data.iloc[:,-1]
#
# print("expected: ", expected_labels, "\n\n", "predicted: ", predicted_labels)
#
# accuracy = classifier.score(test_data.iloc[:,:-2], expected_labels)
#
# print("accuracy: ", accuracy)
#
# print("f1 score: ", f1.f1_score(expected_labels, predicted_labels))

# Second Method: Better Accuracy:

scores = []
classifier = GaussianNB()
classifier.fit(x_train, y_train)
predict = classifier.predict(x_test)
score = accuracy_score(y_test, predict)
scores.append(score)
print("Accuracy: ", round(100 * score, 2))
p = precision_score(y_test, predict)
r = recall_score(y_test, predict)
f = f1.f1_score(y_test, predict)
print("Precision Score: ", round(100 * p, 2))
print("Recall Score: ", round(100 * r, 2))
print("F1 Score: ", round(100 * f, 2))
scores.append(p)
scores.append(r)
scores.append(f)

plot_confusion_matrix(classifier, x_test, y_test)
plt.show()

plot_precision_recall_curve(classifier, x_test, y_test)
plt.show()

plt.bar(range(len(scores)), scores)
plt.xlabel('Accuracy, Precision, Recall, F1-Score')
plt.ylabel('Value')
plt.title('Naive Bayes Accuracy and Score Values')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show()

# def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
#                         n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
#
#     if axes is None:
#         _, axes = plt.subplots(1, 3, figsize=(20, 5))
#
#     axes[1].set_title(title)
#
#     train_sizes, train_scores, test_scores, fit_times, _ = \
#         learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
#                        train_sizes=train_sizes,
#                        return_times=True)
#     fit_times_mean = np.mean(fit_times, axis=1)
#     fit_times_std = np.std(fit_times, axis=1)
#     # Plot n_samples vs fit_times
#     axes[1].grid()
#     axes[1].plot(train_sizes, fit_times_mean, 'o-')
#     axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
#                          fit_times_mean + fit_times_std, alpha=0.1)
#     axes[1].set_xlabel("Training examples")
#     axes[1].set_ylabel("fit_times")
#
#     return plt
#
#
# fig, axes = plt.subplots(1, 2, figsize=(10, 15))
#
# X = df
# y = None
#
# title = "Scalability of the model (Naive Bayes)"
# # Cross validation with 100 iterations to get smoother mean test and train
# # score curves, each time with 20% data randomly selected as a validation set.
# cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
#
# estimator = GaussianNB()
# estimator.fit(training_data.iloc[:,:-2], training_data.iloc[:,-1])
# plot_learning_curve(estimator, title, X, y, axes=None, ylim=(0.7, 1.01),
#                     cv=cv, n_jobs=2)
#
# title = r"Scalability of the model (SVM, RBF kernel, $\gamma=0.001$)"
# # SVC is more expensive so we do a lower number of CV iterations:
# cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
# estimator = SVC(gamma=0.001)
# plot_learning_curve(estimator, title, X, y, axes=None, ylim=(0.7, 1.01),
#                     cv=cv, n_jobs=2)
#
# plt.show()
