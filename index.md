---
layout: default
---

### Summary Figure

![Branching](img/figure.JPG)

### Introduction

There are currently thousands of cases where patients in hospitals either die from a heightened severity due to a pre-existing condition they’ve had before contracting COVID or having the underlying cause of death be from COVID itself. There’s an ongoing issue in the U.S in which people are not aware of their death risk if they were to contract COVID based on the diseases they have at the concurrent time and their biological factors. The goal is to create a ML algorithm from a clustering technique that would predict the chance/risk of someone dying from COVID if they were to have a pre-existing disease in order to increase awareness and responsibility for one's well-being in this crisis. Our current potential unsupervised methods we want to test out include fuzzy clustering, SVM, and decision trees. The algorithm that shows the most accuarte predictablility in deaths will be used throughout the project towards the intended desirability of our results in calculating percentage chance of dying with COVID. 

Dataset:
Ideally, our dataset will consist of data points (rows) that represent one patient, where the columns are the features (such as age, gender, other health risk factors) with a label that represents the risk of that person dying from COVID.


### Methods

To complete our project, we will primarily use datasets from Kaggle and the CDC containing information about reported Covid-19 deaths. Data sets that we choose to focus on will be those containing the most relevant and detailed information on each individual Covid-19 death, and how these deaths correlated to said information. This data will then be used to make a prediction model using clustering and linear regression techniques in order to predict how at-risk a certain individual or population may be, given data about said individual or population. Our current potential unsupervised methods we want to test out include fuzzy clustering, SVM, and decision trees. We are leaning more towards fuzzy clustering since the patients can be in more than 1 group. For example, our model will be able to determine if more people die in a specific age range, from a certain gender, with certain pre-existing health conditions, etc. Additionally, our algorithm will have the flexibility to be able to be run over multiple different datasets containing different sets of factors and information, in order to create multiple different models that can be used to predict the risk of death from Covid-19.



### Results

Pre-existing data shows some trends in ages and pre-existing conditions, but gender and race are not commonly factors taken into consideration, and furthermore, some of this information is ignored by populations, and the source of this information is not always clear. By taking into account this extra data, and using it to predict future deaths, we hope to reveal more detail about risk factors and further enlighten the population on what groups are at risk using data they can see and understand. We expect atleats %95 accuracy to predict death probability for patients with COVID.  

### Discussion

The motivations and goals for this project are that we want to know what factors primarily contribute to deaths related to COVID such as age, race, gender, location, pre-existing conditions, and any other relevant data that may give us some indication about the groups most affected by COVID-19. The best outcome would be to have a prediction model using the most accurate ML algorithm that uses pre-existing data on these factors to predict and understand which factors are the cause of the most deaths when in combination with COVID. If none of the observed ML algorithm was found to predict the probability accurately enough, then we will move on to testing dimension reduction accuracy on the probability predicion. This is all done due to the importance for identifying the most at-risk groups and reiterating the importance of social distancing and masks in order to prevent the spread of COVID to these groups.

### References

*   https://www.cdc.gov/nchs/nvss/vsrr/covid19/health_disparities.htm
