# Implementation of Expectation Maximization Algorithm
**By [Faraz Ahmed Khan](https://www.linkedin.com/in/faraz03983/)

This algorithm is a generalization of the k-means algorithm, instead rather than assigning each observation to a cluster, it provides us a probability of an observation belonging to each of the clusters.

It is generally when there are latent (hidden) variables in our data, and we want to determine those variables.

My implementation deals with a simple case, where we know the distribution of the data. 

## Data Generation
I generate n data points from a 3 different normal distributions, each having different mean and variance. I mix all the data points, and hide their distributions.


