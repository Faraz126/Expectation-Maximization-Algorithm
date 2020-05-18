# Implementation of Expectation Maximization Algorithm
**By [Faraz Ahmed Khan](https://www.linkedin.com/in/faraz03983/)

This algorithm is a generalization of the k-means algorithm, instead rather than assigning each observation to a cluster, it provides us a probability of an observation belonging to each of the clusters.

It is generally when there are latent (hidden) variables in our data, and we want to determine those variables.

My implementation deals with a simple case, where we know the distribution of the data is normal. 

## Data Generation
I generate n data points from a 3 different normal distributions, each having different mean and variance. These n data-points are mixed together, and their distribution is hidden.

Following is a density plot of 1000 different data-points generated from 3 different normal distributions with means 5, 10, 12 and variances 1, 2 and 4 respectively.

![Data Generation](/data_points.png)

## Expectation Maximization
The data is then fed into the expectation maximization algorithm. The aim of the algorithm is to identify the latent variables i.e. the mean and variances of the underlying distributtions. The algorithm starts with random guess for the latent variables and then iteratively improves the estimates.

Following figure shows the estimations of the underlying distriubtions at different stages of the algorithm.
![Iterations](/Iterations.png)

I ran the algorithm for 100 iterations, and extracted the estimations at that point.

Following figure shows the estimations of the latent variables as compared to the actual values of the latent variables.
![Comparison](/differnce.png)



