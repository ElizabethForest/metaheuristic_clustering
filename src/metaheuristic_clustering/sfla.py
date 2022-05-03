# based on:
# Amiri, B., Fathian, M., & Maroosi, A. (2009). Application of shuffled frog-leaping algorithm on clustering. 
#   The International Journal of Advanced Manufacturing Technology, 45(1), 199-209.

# https://github.com/theDIG95/Shuffled-frog-leaping-algorithm/
# https://github.com/LubnaAlhenaki/Genetic-Frog-Leaping-Algorithm-for-Text-Document-Clustering/


import numpy as np
import itertools
from pyclustering.cluster.center_initializer import random_center_initializer as rand_init
from .util import fitness, get_labels

from sklearn.base import ClusterMixin, BaseEstimator


def generate_frog(data, num_of_clusters, metric):
    centroids = rand_init(data, num_of_clusters).initialize()  # can also use k_init for kmeans++ initalisation
    fit = fitness(data, centroids, metric)
    # Todo: give it a uuid for determine if same frog is best multiple times

    return {"centroids": centroids,
            "fit": fit}


def generate_population(data, num_of_frogs, num_of_clusters, metric):
    return [generate_frog(data, num_of_clusters, metric) for i in range(num_of_frogs)]


def rank_frogs(frogs):
    sorted_acs = sorted(frogs, key=lambda k: k["fit"])
    return sorted_acs


def create_memeplexes(frogs, num_of_memeplexes):
    ranked_frogs = rank_frogs(frogs)

    return ranked_frogs[0], [ranked_frogs[i::num_of_memeplexes] for i in range(num_of_memeplexes)]


def evolve(worst_frog, best_frog, data, metric):
    worst_centroids = worst_frog["centroids"]
    best_centroids = best_frog["centroids"]
    new_centroids = worst_centroids + (np.random.rand() * np.subtract(best_centroids, worst_centroids))
    fit = fitness(data, new_centroids, metric)
    return {"centroids": new_centroids, "fit": fit}


def sfla(data, num_of_frogs=30, num_of_clusters=3, num_of_memeplexes=5, memeplex_iterations=10, max_iterations=50,
         metric='euclidean'):
    """

    :param data:
    :param num_of_frogs: Total number of frogs (F)
    :param num_of_clusters: How many clusters are desired (k)
    :param num_of_memeplexes: Number of memeplexes (m)
    :param memeplex_iterations: Number of memeplex iterations (iN)
    :param max_iterations: Maximum number of iterations before the algorithm will terminate
    :param metric: possible values: euclidean
    :return: returns the best frog/best set of centroids found
    """
    # todo: early exit if same frog remains best

    all_frogs = generate_population(data, num_of_frogs, num_of_clusters, metric)

    for i in range(max_iterations):
        # pg is global best frog
        pg, memeplexes = create_memeplexes(all_frogs, num_of_memeplexes)

        new_memeplexes = []
        for im in memeplexes:
            for iN in range(memeplex_iterations):
                im = rank_frogs(im)  # re-rank frogs
                pb = im[0]  # local best frog
                pw = im[-1]  # local worst frog

                # evolve the frog
                new_frog = evolve(pw, pb, data, metric)

                # if fitness doesn't improve try with global best frog
                if not new_frog["fit"] < pw["fit"]:
                    new_frog = evolve(pw, pg, data, metric)

                    # if still doesn't improve, generate a new frog
                    if not new_frog["fit"] < pw["fit"]:
                        new_frog = generate_frog(data, num_of_clusters, metric)

                im[-1] = new_frog

            pb = im[0]
            if pb["fit"] < pg["fit"]:
                pg = pb
            new_memeplexes.append(im)

        all_frogs = list(itertools.chain(*new_memeplexes))

    pg = rank_frogs(all_frogs)[0]
    return pg


# todo: add comments and docstrings
class SFLAClustering(BaseEstimator, ClusterMixin):
    """
    Creates clusters based on the Shuffled Frog Leaping Algorithm

    Based on the paper:
    M. Fathian, B. Amiri, and A. Maroosi, "Application of honey-bee mating optimization algorithm on clustering,"
    Applied Mathematics and Computation, vol. 190, no. 2, pp. 1502-1513, 2007 2007.

    And referenced implementations:
    https://github.com/theDIG95/Shuffled-frog-leaping-algorithm/
    https://github.com/LubnaAlhenaki/Genetic-Frog-Leaping-Algorithm-for-Text-Document-Clustering/
    """

    def __init__(self, num_of_frogs=30, num_of_clusters=3, num_of_memeplexes=5, memeplex_iterations=10,
                 max_iterations=50, metric='euclidean'):
        """
        :param num_of_frogs: Total number of frogs (F)
        :param num_of_clusters: How many clusters are desired (k)
        :param num_of_memeplexes: Number of memeplexes (m)
        :param memeplex_iterations: Number of memeplex iterations (iN)
        :param max_iterations: Maximum number of iterations before the algorithm will terminate
            Note: each iteration contains a full set of memeplexe iterations. So it memeplex_iterations=10 and
            max_iterations=10 then a frog will be updated a total number of 100 times
        :param metric: possible values: euclidean

        """
        self.num_of_frogs = num_of_frogs
        self.num_of_clusters = num_of_clusters
        self.num_of_memeplexes = num_of_memeplexes
        self.memeplex_iterations = memeplex_iterations
        self.max_iterations = max_iterations
        self.metric = metric

    def fit(self, X):
        """
        :param X: the data to be clustered
        :return: the updated object
        """
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        best_frog = sfla(X,
                         num_of_frogs=self.num_of_frogs,
                         num_of_clusters=self.num_of_clusters,
                         num_of_memeplexes=self.num_of_memeplexes,
                         memeplex_iterations=self.memeplex_iterations,
                         max_iterations=self.max_iterations,
                         metric=self.metric)
        self.labels_ = get_labels(X, best_frog)
        return self
