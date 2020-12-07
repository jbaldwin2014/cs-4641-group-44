---
layout: default
---

### Summary Figure

![Branching](img/figure.JPG)

### Introduction

#### Background
The death toll of the 2019 coronavirus disease (CoViD-19) has reached above 200,000 just in the U.S. CoViD-19 has been shown to affect various demographics of population differently. Certain factors like age and pre-existing health conditions have an impact on mortality outcome in CoViD-19 patients. People are not aware of their likelihood of death upon contraction of CoViD-19 and don’t have much information on which risk factors and predispositions negatively impact outcomes. The goal is to create a ML model which would predict the chance/risk of someone dying from CoViD-19 given biological factors and pre-existing diseases in order to increase awareness and responsibility for one's well-being in this crisis.

#### Dataset
To complete our project, we will primarily use datasets from Kaggle and the CDC containing information about reported CoViD-19 deaths. Ideally, our dataset will consist of data points (rows) that represent a state, where the columns are the features (such as age, gender, other health risk factors) with a label that represents the cause of death for that person with CoViD-19. With the dataset also having labels that describe the amount of patients that survived CoViD with these factors, we could use this data as our ground truth when examining our unsupervised algorithm.  Though there are some data points with empty columns (unknown race, unknown pre-existing disease), with 6.5k rows of data points, we believe that we have more than enough data for our algorithm. After a dataset cleanup with data that is provided on a weekly basis, whereas there are no empty datasets, the accuracy in the prediction of death chance determined by the factors will significantly increase. 


### Methods

Given our chosen data sets, we will identify the most relevant features contributing to each individual CoViD-19 death, and how these deaths correlate to said features. This data will then be used to make a prediction model using unsupervised clustering and supervised neural network techniques in order to predict how at-risk a certain individual or population may be, given data about said individual or population. To process our data sets, we have a few basic methods in mind:

#### Unsupervised Learning: Clustering
In order to get a better picture of which pre-existing health factors contribute most to the probability of a CoViD-19 fatality, we plan on implementing a clustering method such as KMeans. When run given our chosen data sets, the output of this KMeans algorithm will ideally group CoViD-19 deaths by the co-occurrence of similar factors into n-dimensional groups, identifying pre-existing conditions that, when combined, increase the risk of death from CoViD-19.

#### Supervised Learning: Neural Network
To get a more specific picture of the exact risk of death from CoViD-19 to a given individual, we will implement a form of neural network trained from our data sets. We will provide ground-truth data regarding whether CoVid-19 infected patients died as a result of their infection or another cause. This data will also encompass features regarding health data of each patient. The goal of this training is to obtain a network that will accept a variety of health information about a patient, and return a probability of said individual succumbing to a CoViD-19 infection, as well as conditions that may significantly increase their risk of a CoViD-19 fatality.

Both of these algorithms will have the flexibility to be able to be run over multiple different datasets containing different sets of factors and information, in order to better refine the models that can be used to predict the risk of death from CoViD-19.

### Results

Pre-existing data shows early trends in how certain conditions may increase the risk of death from CoVid-19. However, there is limited concrete understanding of exactly which individual conditions contribute most to this risk, and even less information about how having a multitude of conditions will affect risk. By taking into account a wide range of data that encompasses not just health conditions, but also environmental and genetic factors, and using it to predict the likelihood of future deaths, we hope to reveal more detail about risk factors and further enlighten the population on what groups are at risk using data they can see and understand. We expect at least 95% accuracy to predict mortality probability for patients with CoViD-19. To evaluate how effective our algorithms perform, we will use recall and precision metrics and expect scores of > 0.9. 
 
### Discussion

It is easy to ignore risk when risk is not well understood. Our goal is to present the general population with data that is easy to understand, digest, and interpret that provides a clear picture of how at-risk they may be from a CoVid-19 infection. When faced with the issue of deciding when to re-open our economies and the risk of “second wave” increases in infections, we want to do our best to ensure everyone is fully informed on how an infection may affect them personally. This is done in order to identify the most at-risk groups and to reiterate the importance of social distancing and mask wearing in order to prevent the spread of CoViD-19 to these groups.

### References

*   https://www.cdc.gov/nchs/nvss/vsrr/covid19/health_disparities.htm

*   https://towardsdatascience.com/predicting-mortality-in-the-icu-2e4832cc94d2

*   https://data.cdc.gov/NCHS/Deaths-involving-coronavirus-disease-2019-CoViD-19/ks3g-spdg

*   https://www.cdc.gov/coronavirus/2019-ncov/CoViD-data/investigations-discovery/assessing-risk-factors.html

*   https://www.news-medical.net/news/20200927/Risk-factors-for-mortality-among-hospitalized-patients-with-CoViD-19.aspx

*   https://ourworldindata.org/mortality-risk-CoViD


## ---

### Midterm Report

![Branching](img/figure2.png)

### Progress Highlights

#### Dataset
For our project, we used a public dataset from a CDC database containing CoViD-19 case surveillance information. Each data point represents a deidentified patient and includes their age, sex, presence of comorbidity, hospitalization status, etc. as well as binary class data on death outcome, which can be used as ground-truth label for some algorithms. Though there are some data points with empty columns (unknown race, unknown pre-existing disease), the raw dataset has over five million points and 11 features. Over the course of our project thus far, we have taken steps to clean, standardize, and run unsupervised learning algorithms on our data in order to better understand how these risk factors contribute to the likelihood that a patient who has contracted CoViD-19 will die. Our dataset did not seem to contain any outliers.

#### Data Cleaning and Standardization
With our data set only containing 11 features to begin with, data cleaning was fairly straightforward given visual analysis and modification through Python code. We determined that features involving dates, such as CDC report dates and symptom onset dates, were discarded as it is easy to understand that the day an individual contracts CoViD-19 does not contribute to their risk of death. Based on this assumption, we decided to cut the number of features down to six by removing these unwanted features.

Similarly, because we only wanted to look at data on confirmed cases of CoViD-19, we removed data points marked with the label ‘probable case’, as we only wanted to look at data marked ‘lab-confirmed case’. Finally, rows containing values of ‘missing’, ‘NA’, or ‘unknown’ in regard to any feature were also completely eliminated from the set. 

After the stages of cutting out data, we were left with a sizable 596807 data points to run our algorithms on. Before running these algorithms, however, it was imperative to standardize our data. Many of our remaining features were entirely categorical, making it impossible to run many unsupervised learning algorithms. To solve this issue, we wrote Python code to encode these categorical features as integers indicating ‘0’ as false and ‘1’ as true in some cases, or by using a range of integers based on the number of possible labels for a given feature.

![Branching](img/encodeagegroup.png)

_Above: Example Code Used to Perform Integer Encoding of Categorical Features_

![Branching](img/encodeddata.png)

_Above: Example of Cleaned and Encoded Data_

#### Identifying Important Features
After cleaning and standardizing our data set, the next crucial step is identifying which features contribute to the majority of the variance in the data, and which can be discarded as irrelevant. This can be done in one of two ways. First, in datasets with high dimensionality, Principal Component Analysis, or PCA, can be used to empirically determine which features are most important. In cases of low dimensionality, such as our dataset that contains only six features after data cleaning, a more qualitative method can be employed. In our case, we were able to determine that, based on the goal of our project, all but one of the remaining features were crucial in order to get a complete understanding of how each contributes to the likelihood of a CoViD-19 fatality. Because we had removed data points labeled ‘probable case,’ this only left one type of data remaining for this feature. This means that this feature would not at all contribute to variance in our data and, for this reason, we decided to continue by running our unsupervised learning algorithms on the remaining five features.

#### Unsupervised Learning: Kmode Clustering
In order to understand which of the categorical pre-existing health factors contribute most to the probability of a CoViD-19 fatality, we decided to implement the K-Modes clustering algorithm as our unsupervised learning technique. When run on our chosen data sets, the output of this K-Modes algorithm is expected to, ideally, group CoViD-19 deaths data points by the co-occurrence of similar factors into various n-dimensional groups, identifying pre-existing risk factors that, when combined, contribute to an increase the risk of death from CoViD-19.

The first step in this process is to identify the optimal number of clusters to use in our K-Modes cluster predictions. To do this, we decided to implement the elbow method in order to determine this optimal number of clusters using a range of values for k between 1 and 25. Then, for each value of k, we computed an average score for all clusters, which in our cases represents the distortion cost. By default, the distortion cost is computed as the sum of square distances from each point to its assigned cluster center.

![Branching](img/numclusters.png)

_Above: Code Used to Determine Clustering Cost_

A plot of our elbow method will then be created in order to have a better visual understanding of what the optimal number of clusters should be. This number is chosen based on a quick analysis to time a value of k after which cost reduction does not decrease dramatically, located at the “elbow” of the plot.

![Branching](img/elbowmethod.png)

_Above: Code Used to Create Our Elbow Plot_

We then computed the cost given an increasing number of cluster centers and predicted the cluster index for each sample by minimizing a dissimilarity measure, whereas k modes count the number of “features” that are not the same. 

![Branching](img/Figure_3.png)

_Above: (25 Clusters)_

![Branching](img/Figure_2.png)

_Above: (15 clusters)_

![Branching](img/runs.png)

_Above: Example of Terminal Output_

![Branching](img/predictcluster.png)

_Above: Example of Starting Data With Predicted Cluster from K-Modes_

The elbow-method graphed results above show that the optimal number of clusters for our dataset appears at about 5 clusters, exemplified by the sharp plateau at k=5 in our 15 and 25 cluster runs. While costs do decrease beyond this point, such diseases are trivial compared to the significant and sharp increase in the amount of time it takes to run our implementation of the K-Modes algorithm. For each iteration, the amount of moves needed decreases, though the cost stays mostly stagnant, changing by very little if at all, with the approximate range being about 1,000,000 - 2,000,000.

Our results clearly show not only that we can group our data into clusters based on these features, but also that the optimal number of clusters for grouping is k=5. With this in mind, we are able to run the K-Modes algorithm and visualize our results:

![Branching](img/clustercode.png)

_Above: Code Used to Run K-Modes with k=5_

![Branching](img/barchart2.png)

_Above: Bar Chart on Age-Group with an Optimal 5 Clusters_

![Branching](img/barchart.png)

_Above: Example, Another Bar Chart on Age-Group with only 2 Clusters_

We can view our data based on certain features and their data counts (shown by the bar graph above). For our visualizations, we used age-range as our clustering basis, comparing the count of data points in each cluster for each age range. Beyond just showing that ages 40-59 have the highest count of cases, our clusters also indicate grouping by other categorical features that could contribute to risk of CoViD-19 death. In both bar charts, cluster 0 represents data points that were clustered by the co-occurrence of ICU hospitalization and the existence of pre-existing health conditions. Because of this, our K-Modes algorithm indicates higher counts of CoViD-19 death in every age range for this cluster. Qualitatively, this means that these factors contribute significantly to the probability that an individual will succumb to a case of CoViD-19.

Empirically, our K-Modes defines clusters using the amount of matching categories between data points. For example, take clusters 2 and 3 in the above example run with 5 total clusters. For the mode of each cluster, if a data point has a given categorical feature, the mode vector K consists of x categorical values for which each is the mode of a feature. For our data, the mode of an attribute can be “1” or “0” for category features that can be defined as true or false, and from 0-8 for features like age group and race/ethnicity. The mode becomes whichever number is most common in the cluster. The total number of labels that appeared on each predicted index cluster can be seen through the visualization above for the age group feature. From observation, it can be seen that the cluster at index 0 seemed to have the largest number of datasets, making it seem to have a unique constraint to its index, and thus indicating that tease features contribute most to the likelihood of a CoViD-19 death.

### Current Challenges and Next Steps
For this report, the current challenges our group has encountered so far are as follows: in cleaning and standardizing our data, we had to remove some data points that had too many missing or unknowns, which left only a small chunk of the original data left. However, because we started with so many data points to begin with, around 4.5 million, the loss is mostly negligible, leaving us with about 500k points with encoding and cleaning. Our K-Modes implementation runs without error, and our current challenge primarily lies in creating proper visualizations of the data in graph form by category. We tried different implementations and have settled on K-Modes, as so far this has provided us the best visual representations of our clustering, and works well with categorical data. We also have started on a supervised technique, binary logistic regression, because we have these many categorical independent variables and only two possible outcomes.


### References:

*   https://data.cdc.gov/Case-Surveillance/COVID-19-Case-Surveillance-Public-Use-Data/vbim-akqf 

*   https://medium.com/@davidmasse8/unsupervised-learning-for-categorical-data-dd7e497033ae 

*   https://pypi.org/project/kmodes/

*   https://www.researchgate.net/post/K_modes_clustering_how_to_choose_the_number_of_clusters

## ---

### Final Report

![Branching](img/figure3.png)

### Supervised Learning + Performance Metrics

#### Random Forest

Our goal for this project is to create a classification model that can determine if a Covid-19 infection will be fatal or non-fatal to an individual given various information about said individual. To best accomplish this, a decision tree can be used in order to “split” the data along planes of greatest information gain in order to classify the outcome of a given data point. While this can be a powerful method, even better is the Random Forest model, which overall is more robust to overfitting and benefits from the utilization of multiple weak single decision trees in order to create a more accurate overall prediction. The primary advantages of using random forest with our data set is that, as an ensemble learning technique, random forest can ultimately predict the class of a data point better than a single decision tree ever could. To build this random forest, we used 3/4 of our data set as training data, and the remaining ¼ of data points were used to validate and test the performance of the random forest. The individual decision tree estimators for the random forest were trained to classify data points by the feature “death_yn” in our data set, as ultimately we wanted to predict the binary class indicating if a given individual would die from a Covid-19 infection.

![Branching](img/randomforest1.png)

_Above: Code Snippet Showing the Selection of Split that Provides Greatest Info Gain_

In order to run random forest, the programmer must also choose various hyper-parameters, such as the number of estimators, or decision trees, to be trained in the forest, the maximum depth of each tree, and the maximum percentage of features to be randomly selected for training from the total number of features, such that it can be certain that there is some variation in the training data that the trees receive. Ultimately, these values were set to 8, 4, and 0.9 respectively after an iterative tuning process that ensured a high level of accuracy while ensuring not too much data was lost during the training process. The factors also played into the bootstrapping phase of the algorithm, where subsets of our larger data set are randomly selected for training. The number of features selected, 0.9, ensures that 90% of our dataset’s features will be used for training, which are randomly selected, all but making certain two training sets will never be identical.

After running the random forest algorithm, our model is able to predict the death of a Covid 19 patient with a high accuracy of 98.7%. Accuracy, however, is not the only measure by which to judge the performance of this algorithm. In our case, achieving such a high accuracy, while desired, may be indicative of a problem or bias within our data set. Because it is important to be both accurate as well as minimize the number of false negative predictions, the recall metric can also be used to measure the performance of our model. For this metric, the random forest performed much worse, achieving only 61.8% recall. This is most likely due to the fact that with so few cases of death in our data set, a maximum depth of four was not able to accurately determine which splits in the data would be certain to lead to an outcome of death from Covid-19.

![Branching](img/randomforest2.png)

_Above: When the tree reaches max depth, it predicts the mode of the remaining y-values, which, due to data imbalances, will usually be cases not resulting in death_

![Branching](img/randomforest3.png)

_Above: Two of the Eight Decision Trees that Make Up Our Random Forest shown with Information Gain for each split_

#### Naive Bayes

![Branching](img/nb_scale.png)

_Above: Scalability Graph_

Another algorithm we chose to implement was Naive Bayes mainly due to the results from Naive Bayes can be used to visualize the scalability of our prediction model (see figure above). Naive Bayes serves as one of our probabilistic prediction models for supervised learning in order to predict, based on current data, what future labels might look like for each feature. Additionally, results from Naive Bayes can be used to visualize the scalability of our prediction model (see figure above). Based on the above figure, as the training examples increase, so do the fit times, though they do seem to slowly reach an equilibrium as time goes on and the curve flattens slightly. Naive Bayes generally has a lower accuracy and f1 score  when run using our data set compared to the previous Random Forest method, but still remains above 50% accuracy consistently.

![Branching](img/nb_matrix.png)

_Above: Confusion Matrix_

We were able to improve the accuracy of Naive Bayes with a different method up to 89.06%, but the f1 score didn’t change much at around 62%. With this new method, we could also calculate the precision and recall, which were at about 49% and 86% respectively as well as plot the confusion matrix. From our confusion matrix, we have the predicted and expected labels and the data to calculate our precision, recall and f1-score. From the confusion matrix, we can interpret our predicted and expected labels to have 38864 true negatives (TN), 4582 false positives (FP), 720 false negatives (FN), and 4319 true positives (TP), which means we have 43183 true values and 5302 false values total. We calculate recall by R = TP/(TP + FN) = ~0.86 = 86%, precision by P = TP/(TP + FP) = ~0.49 = 49%, and f1-score by F = (2*R*P)/(R+P) = ~0.62 = 62%. In general, since Naive Bayes uses the feature independence assumption, when features are rarely truly independent, class probabilities output using naive Bayes can be fairly inaccurate. In other words, the Naive Bayes classifier assumes that the effect of the value of a predictor (x) on a given class (c) is independent of the values of other predictors. P(c|x) is the posterior probability of class (target) given predictor (attribute), and P(c) is the prior probability of class (Sayad). Naive Bayes has little to no hyperparameter tuning necessary, which lends to it generalizing well compared to other algorithms.

![Branching](img/naivebayescode.png)

_Above: Naive Bayes Code_

The above code uses the GaussianNB library from sklearn to fit our training set and predict based on our test data, calculating the accuracy, precision, recall and f1 score from the y test set and results of predict on the x test set, where x is the above listed columns (sex, age_group, etc) and y and whether a death occurred.

![Branching](img/nb_curve.png)

_Above: Precision vs Recall Curve_

From the figure you can see above, as the recall is increasing, the precision of the prediction from the Naive Bayes algorithm is decreasing in our case. The calculated recall and precision for our data occurs just before a rather steep decrease on the graph, where recall is 0.89 and precision is 0.49.

![Branching](img/nb_bargraph.png)

_Above: Bar Graph Comparison of Accuracy, Precision, Recall and F1-Score_

The above bar graph simply displays the values of the accuracy (at 0.0 on the x-axis), precision (1.0), recall (2.0) and f1-score (3.0) in that order in graphical form to more easily compare the values.

**Baseline Performance of Naive Bayes:**
Setting the death_yn column to all ones in our training dataset and running Naive Bayes, we get the following statistics for our guesswork/baseline performance:
Accuracy:  10.21
Precision Score:  10.21
Recall Score:  100.0
F1 Score:  18.53
This is expectedly low compared to running our supervised learning model on our original dataset.

#### Neural Networks

To accomplish our project’s goal, one of the other algorithms we’ve implemented was a neural network, as our project necessitates the use of an activation function to turn an unbounded input into an output with a predictable form. We used NN due to its capability of modeling and processing nonlinear relationships between the inputs(COVID death risk factors) and outputs in parallel. To do this, our implementation makes use of the sigmoid function, which always outputs values between 0 and 1. We had to One-hot encode our data due to the sigmoid function being unable to take in any categorical data.

![Branching](img/nn1.png)

_Above: One-hot encoded data_

Because we did not have any optimal weights to assign to each of our features when creating the model since each input can have a different influence on the subsequent calculations in the inner layers and on the output layer of the entire network, we needed a way for our implementation to find the optimal weight for each feature algorithmically. Neural Networks can accomplish this through backpropagation, where the appropriate weight for a given feature is updated based on the accuracy of the given output when compared to the target output.  

Since we did not know the optimal amounts of hidden layers needed to get the algorithm with the best performance, we decided to test out the accuracies of each run of the algorithm from 0,5,10,...40 numbers of the hyperparameter, hidden layers, with increments of 5 when tuning this hyperparameter. We set the learning rate(alpha) hyperparameter to 0.1 to ensure faster speed on training the NN. For time efficiency, we set the max Iteration hyperparameter to be 10 due to the timely cost of running the algorithm with over 400,000 data points. 

On the feedforward of the neural network, we had a set of input features including  age, race, medcon, icu, and sex with random inital weights. 

![Branching](img/nn2.png)

_Above: Fast forward method implementation_

Similar to training and test sets used in random forests, we separated 70% of the data into training data and the remaining 30% into test data, but in this algorithm, we did this to ensure a higher accuracy deterministic rather than focusing on recall since recall was already analyzed in random forests.

In this case, we optimized our feature weights using back propagation, where we had our model calculate the error in a given prediction, and then updated our weight values and reran our code. After each iteration, we found our error to be decreasing and the accuracy of our algorithm to be increasing. If we had input values of  (0,0, 0,0….), the sum of the products of the input nodes and weights is always going to be zero, causing the weights of each feature input unable to be updated. In this case, the output will always be zero, no matter how much we train our model, due to the sigma activation function that we’ve used. To resolve this issue and make reliable predictions, we used the bias term to aid in making a robust neural network.

![Branching](img/nn3.png)

_Above: Backpropagation implementation_

Since the number of training samples were large (~350,000), using gradient descent would take too long because in every iteration where we would be updating the values of the parameters, we would be running through the complete training set. Due to this, we decided to use the Stochastic Gradient Descent since it is faster because of us only using one training sample and it starts improving itself right away from the first sample. 

![Branching](img/nn4.png)

_Above: Stochastic gradient implementation in backpropagation_

We then began to test the algorithm on the testing data to view the accuracy for each amount of the hidden layers. We ran the algorithm 5 times for each hidden layer amount and then calculated the average accuracy out of those 5 runs for better a deterministic of optimal values, as seen in the figures below. The highest average accuracy was 87%, which was received from having 35 hidden layers. 

![Branching](img/nn5.png)

_Above: Accuracy analysis per run of the NN with 5 hidden layers_

As seen in the plot below, there is little to no change in accuracy between some of the hidden layer amounts, such as the accuracies with 0 hidden layers and 5 hidden layers having no change. After a guess performance diagnostics test where we changed the value of all our  testing data to 1, we had an accuracy of 17% from NN, which is close to the same percentage of the class distribution where 80% (0) and 15% (1). Since our performance is very close to the guesswork, this shows that NN did not learn from the data. This is primarily due to the dataset being severely unbalanced, where the majority of death classifications were 0 and a very small amount were 1, causing overfitting in our algorithm, making NN unable to learn effectively as seen below. 

![Branching](img/nn6.png)

_Above: Average accuracy of each hidden layer amount_

#### Linear Regression

Since our data consisted of categorical features and binary class labels, logistic regression was chosen as another supervised learning classification algorithm. Data was split in 80:20 ratio to training and test datasets. After the model was fit on the training set, it was run on the test data and returned a score of 0.938. However, the score seemed too high and after looking further into the results, the high score was due to the large imbalance of the dataset labels with a ratio of around 92:8 for labels of 0 and 1, respectively. The labels of 1 indicate a case of death. The model correctly classified most of the 0 labels and poorly classified the 1 labels. Recall was calculated to be 0.45. The model parameters for class weight in the code were changed to compensate for the imbalance and returned a score of 0.889 with an improved recall of 0.93.

![Branching](img/regression1.png)

![Branching](img/regression2.png)

_Above: Confusion Matrix: count of true negatives = C<0,0>; false negatives = C<1,0>; true positives = C<1,1>; false positives = C<0,1>_

Recall was our chosen performance metric because we wanted to minimize the number of false negatives since we placed a higher cost on giving a false negative death prediction than a false positive one. The regularization parameter was chosen to be C = 1.0 since it resulted in the highest score on the test data.

![Branching](img/regression3.png)

The weakness of logistic regression came from the large imbalance of data labels so that the model would be unable to strongly predict the minority of the 0 and 1 label data. The performance of this model was also measured using a receiver operating curve(ROC) which plots true positive rate vs. false positive rate at different classification thresholds. The area under the curve was high at 0.9576 which suggests good performance at classification prediction from the model.

![Branching](img/regression4.png)

### References:

*   https://www.kaggle.com/rude009/heart-failure-model-prediction-comparisons-95 

*   https://www.saedsayad.com/naive_bayesian.htm

