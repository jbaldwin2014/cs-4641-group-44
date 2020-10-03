---
layout: default
---

### Summary Figure

![Branching](img/figure.JPG)

### Introduction

#### Background
The death toll of the 2019 coronavirus disease (CoViD-19) has reached above 200,000 just in the U.S. CoViD-19 has been shown to affect various demographics of population differently. Certain factors like age and pre-existing health conditions have an impact on mortality outcome in CoViD-19 patients. People are not aware of their likelihood of death upon contraction of CoViD-19 and don’t have much information on which risk factors and predispositions negatively impact outcomes. The goal is to create a ML model which would predict the chance/risk of someone dying from CoViD-19 given biological factors and pre-existing diseases in order to increase awareness and responsibility for one's well-being in this crisis.

#### Dataset
Ideally, our dataset will consist of data points (rows) that represent one patient, where the columns are the features (such as age, gender, other health risk factors) with a label that represents the risk of that person dying from COVID.


### Methods

Given our chosen data sets, we will identify the most relevant features contributing to each individual CoViD-19 death, and how these deaths correlate to said features. This data will then be used to make a prediction model using unsupervised clustering and supervised neural network techniques in order to predict how at-risk a certain individual or population may be, given data about said individual or population. To process our data sets, we have a few basic methods in mind:

Unsupervised Learning: Clustering
In order to get a better picture of which pre-existing health factors contribute most to the probability of a CoViD-19 fatality, we plan on implementing a clustering method such as KMeans. When run given our chosen data sets, the output of this KMeans algorithm will ideally group CoViD-19 deaths by the co-occurrence of similar factors into n-dimensional groups, identifying pre-existing conditions that, when combined, increase the risk of death from CoViD-19.

Supervised Learning: Neural Network
To get a more specific picture of the exact risk of death from CoViD-19 to a given individual, we will implement a form of neural network trained from our data sets. We will provide ground-truth data regarding whether CoVid-19 infected patients died as a result of their infection or another cause. This data will also encompass features regarding health data of each patient. The goal of this training is to obtain a network that will accept a variety of health information about a patient, and return a probability of said individual succumbing to a CoViD-19 infection, as well as conditions that may significantly increase their risk of a CoViD-19 fatality.

Both of these algorithms will have the flexibility to be able to be run over multiple different datasets containing different sets of factors and information, in order to better refine the models that can be used to predict the risk of death from CoViD-19.




### Results

Pre-existing data shows some trends in ages and pre-existing conditions, but gender and race are not commonly factors taken into consideration, and furthermore, some of this information is ignored by populations, and the source of this information is not always clear. By taking into account this extra data, and using it to predict future deaths, we hope to reveal more detail about risk factors and further enlighten the population on what groups are at risk using data they can see and understand. We expect atleast %95 accuracy to predict death probability for patients with COVID.  

### Discussion

The motivations and goals for this project are that we want to know what factors primarily contribute to deaths related to COVID such as age, race, gender, location, pre-existing conditions, and any other relevant data that may give us some indication about the groups most affected by COVID-19. The best outcome would be to have a prediction model using the most accurate ML algorithm that uses pre-existing data on these factors to predict and understand which factors are the cause of the most deaths when in combination with COVID. If none of the observed ML algorithm was found to predict the probability accurately enough, then we will move on to testing dimension reduction accuracy on the probability predicion. This is all done due to the importance for identifying the most at-risk groups and reiterating the importance of social distancing and masks in order to prevent the spread of COVID to these groups.

### References

*   https://www.cdc.gov/nchs/nvss/vsrr/covid19/health_disparities.htm
