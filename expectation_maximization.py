#http://ai.stanford.edu/~chuongdo/papers/em_tutorial.pdf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import math
from typing import List, Tuple



def data_generation(n: int, clusters, means, vars) -> List[float]:
    """
    The function to generate a total of n data points randomly from two
        different gaussian distributions.

        n = number of data points to generate
        clusters = number of distributions
        means = mean of each distribution
        vars = variance of each distribution

    Returns a list of n floats
    """

    assert n > 0

    data_points = []

    for i in range(n):
        toss = np.random.randint(low = 0, high = clusters) #chosing the distribution to use
        data_points.append(np.random.normal(loc = means[toss], scale= math.sqrt(vars[toss]))) #taking root because std. devivation is square root of variance

    return data_points


def plot_histogram(data: List, means = None) -> None:
    """
    Plots histogram of data points on x-y plane using matplotlib.
        If mean1 and mean2 are not None, it plots vertical lines at x =
        mean1 and x = mean2 for better visualization
    """
    sns.kdeplot(data, shade= True, label = "Density of Data")

    if means is not None:
        for i in means:
            plt.axvline(x = i, c = "orange", label = "Mean")


    plt.legend()
    plt.grid()
    plt.rcParams["figure.figsize"] = (16,9)
    plt.savefig(f"data_points.png", bbox_inches='tight', dpi=100)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def expectation_maximization(data, n_clust, with_plot = False, max_iter = 50) -> List:
    """
    Extracts means and variances of hidden distributions
        data = data points with hidden distriubtions
        n_clust = number of distributions to extract from the data
    """
    #defining parameters
    means = [np.random.sample() * max(data) for i in range(n_clust)]
    vars = [np.random.randint(low=1, high=20, size=1)[0] for i in range(n_clust)]

    pi = [np.random.sample() for i in range(n_clust)]
    pi = softmax(pi)

    probabilites = [[pi[j] for i in data] for j in range(n_clust)]
    iter_num = 0
    if with_plot:
        progress = 0
        plot_num = 0
        fig, axs = plt.subplots(2, 3)
    while iter_num <= max_iter:

        for n in range(n_clust):
            current_pi = pi[n]
            for i in range(len(data)):
                xi = data[i]
                numerator = stats.norm.pdf(x = xi, loc = means[n], scale = math.sqrt(vars[n])) * current_pi
                denominator = sum([stats.norm.pdf(x = xi, loc = means[k], scale = math.sqrt(vars[k])) * (pi[k]) for k in range(n_clust)])
                #denominator = numerator + (stats.norm.pdf(x = xi, loc = means[1], scale = math.sqrt(vars[1])) * (1 - pi))
                prob = numerator/denominator
                probabilites[n][i] = prob
                #probabilites[1][i] = 1 - prob

        for n in range(len(means)):
            means[n] = sum([probabilites[n][i] * data[i] for i in range(len(data))])/sum(probabilites[n])
            vars[n] = sum([probabilites[n][i] * ((data[i] - means[n])**2) for i in range(len(data))])/sum(probabilites[n])

        if with_plot:
            if np.isclose(iter_num/max_iter, progress):
                y_s = []
                for i in range(n_clust):
                    y = np.random.normal(loc = means[i], scale= math.sqrt(vars[i]), size = 5000)
                    sns.distplot(y, hist=False, kde_kws={"shade": True}, ax=axs[plot_num // 3, plot_num % 3]).set_title(f"iteration {iter_num}")
                    axs[plot_num // 3, plot_num % 3].axvline(x = means[i])
                plot_num += 1
                progress += 0.2

        iter_num += 1


    fig.set_size_inches(16, 9)
    plt.savefig(f"Iterations.png", bbox_inches='tight', dpi=100)

    return means, vars


def plot_means_variances(names, true_means, estimated_means, true_vars, estimated_vars):
    fig, axs = plt.subplots(2, 1)
    width = 0.20
    axs[0].bar([i for i in range(len(names))], true_means, width = width, label = "True Means")
    axs[0].bar([i + width for i in range(len(names))], estimated_means, width= width, label = "Estimated Means")
    axs[0].set_xticklabels(names)
    axs[0].legend()

    axs[1].bar([i for i in range(len(names))], true_vars, width=width, label="True Variances")
    axs[1].bar([i + width for i in range(len(names))], estimated_vars, width=width, label="Estimated  Variances")
    axs[1].set_xticklabels(names)
    axs[1].legend()

    fig.set_size_inches(16, 9)
    plt.savefig(f"differnce.png", bbox_inches='tight', dpi=100)



























if __name__ == "__main__":
    plt.style.use('seaborn')
    n = 1000 #number of points to generate
    true_means = [5, 10, 15]
    true_vars = [1, 2, 4]
    clusters = 3
    names = ["Cluster " + str(i + 1) for i in range(clusters)]

    data = data_generation(n = n, clusters= clusters, means= true_means, vars=true_vars)
    plot_histogram(data, true_means)


    means, vars = expectation_maximization(data, clusters, with_plot = True, max_iter=100)



    true_vars = [x for _, x in sorted(zip(true_means, true_vars))]
    true_means.sort()


    vars = [x for _, x in sorted(zip(means, vars))]
    means.sort()


    plot_means_variances(names, true_means, means, true_vars, vars)
