---
layout: default
---

### Summary Figure

![Branching](img/figure.JPG)

### Introduction

#### Background
The death toll of the 2019 coronavirus disease (CoViD-19) has reached above 200,000 just in the U.S. CoViD-19 has been shown to affect various demographics of population differently. Certain factors like age and pre-existing health conditions have an impact on mortality outcome in CoViD-19 patients. People are not aware of their likelihood of death upon contraction of CoViD-19 and don’t have much information on which risk factors and predispositions negatively impact outcomes. The goal is to create a ML model which would predict the chance/risk of someone dying from CoViD-19 given biological factors and pre-existing diseases in order to increase awareness and responsibility for one's well-being in this crisis.

#### Dataset
To complete our project, we will primarily use datasets from Kaggle and the CDC containing information about reported CoViD-19 deaths. Ideally, our dataset will consist of data points (rows) that represent a state, where the columns are the features (such as age, gender, other health risk factors) with a label that represents the cause of death for that person with CoViD-19. With the dataset also having labels that describe the amount of patients that survived CoViD with these factors, we could use this data as our ground truth when examining our unsupervised algorithm.  Though there are some data points with empty columns(unknown race, unknown pre-existing disease), with 6.5k rows of data points, we believe that we have more than enough data for our algorithm. After a dataset cleanup with data that is provided on a weekly basis, whereas there are no empty datasets, the accuracy in the prediction of death chance determined by the factors will significantly increase. 


### Methods

Given our chosen data sets, we will identify the most relevant features contributing to each individual CoViD-19 death, and how these deaths correlate to said features. This data will then be used to make a prediction model using unsupervised clustering and supervised neural network techniques in order to predict how at-risk a certain individual or population may be, given data about said individual or population. To process our data sets, we have a few basic methods in mind:

Unsupervised Learning: Clustering
In order to get a better picture of which pre-existing health factors contribute most to the probability of a CoViD-19 fatality, we plan on implementing a clustering method such as KMeans. When run given our chosen data sets, the output of this KMeans algorithm will ideally group CoViD-19 deaths by the co-occurrence of similar factors into n-dimensional groups, identifying pre-existing conditions that, when combined, increase the risk of death from CoViD-19.

Supervised Learning: Neural Network
To get a more specific picture of the exact risk of death from CoViD-19 to a given individual, we will implement a form of neural network trained from our data sets. We will provide ground-truth data regarding whether CoVid-19 infected patients died as a result of their infection or another cause. This data will also encompass features regarding health data of each patient. The goal of this training is to obtain a network that will accept a variety of health information about a patient, and return a probability of said individual succumbing to a CoViD-19 infection, as well as conditions that may significantly increase their risk of a CoViD-19 fatality.

Both of these algorithms will have the flexibility to be able to be run over multiple different datasets containing different sets of factors and information, in order to better refine the models that can be used to predict the risk of death from CoViD-19.




### Results

Pre-existing data shows early trends in how certain conditions may increase the risk of death from CoVid-19. However, there is limited concrete understanding of exactly which individual conditions contribute most to this risk, and even less information about how having a multitude of conditions will affect risk. By taking into account a wide range of data that encompasses not just health conditions, but also environmental and genetic factors, and using it to predict the likelihood of future deaths, we hope to reveal more detail about risk factors and further enlighten the population on what groups are at risk using data they can see and understand. We expect at least 95% accuracy to predict mortality probability for patients with CoViD-19. To evaluate how effective our algorithms perform, we will use recall and precision metrics and expect scores of >0.9. 
 
### Discussion

It is easy to ignore risk when risk is not well understood. Our goal is to present the general population with data that is easy to understand, digest, and interpret that provides a clear picture of how at-risk they may be from a CoVid-19 infection. When faced with the issue of deciding when to re-open our economies and the risk of “second wave” increases in infections, we want to do our best to ensure everyone is fully informed on how an infection may affect them personally. This is done in order to identify the most at-risk groups and to reiterate the importance of social distancing and mask wearing in order to prevent the spread of CoViD-19 to these groups.

### References

*   https://www.cdc.gov/nchs/nvss/vsrr/covid19/health_disparities.htm
https://towardsdatascience.com/predicting-mortality-in-the-icu-2e4832cc94d2

https://data.cdc.gov/NCHS/Deaths-involving-coronavirus-disease-2019-CoViD-19/ks3g-spdg

https://www.cdc.gov/coronavirus/2019-ncov/CoViD-data/investigations-discovery/assessing-risk-factors.html

https://www.news-medical.net/news/20200927/Risk-factors-for-mortality-among-hospitalized-patients-with-CoViD-19.aspx

https://ourworldindata.org/mortality-risk-CoViD

