# based on:
# M. Fathian, B. Amiri, and A. Maroosi, "Application of honey-bee mating optimization algorithm on clustering,"
#   Applied Mathematics and Computation, vol. 190, no. 2, pp. 1502-1513, 2007 2007.

# https://github.com/theDIG95/Shuffled-frog-leaping-algorithm/
# https://github.com/LubnaAlhenaki/Genetic-Frog-Leaping-Algorithm-for-Text-Document-Clustering/


import numpy as np
import itertools
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer as k_init
from pyclustering.cluster.center_initializer import random_center_initializer as rand_init
from src.util import fitness, get_labels

from sklearn.base import ClusterMixin, BaseEstimator


def generate_frog(data, num_of_clusters, metric):
    centroids = rand_init(data, num_of_clusters).initialize()  # can also use k_init for kmeans++ initalisation
    fit = fitness(data, centroids, metric)
    # Todo: give it a uuid for determine if same frog is best multiple times

    return {"centroids": centroids,
            "fit": fit}


def generate_population(data, num_of_frogs, num_of_clusters, metric):
    return [generate_frog(data, num_of_clusters, metric) for i in range(num_of_frogs)]


def rank_frogs(frogs, metric):
    sorted_acs = sorted(frogs, key=lambda k: k["fit"])
    if metric == 'ecludian' or metric == 'davies_bouldin':
        return sorted_acs
    else:
        return sorted_acs[::-1]


def create_memeplexes(frogs, num_of_memeplexes, metric):
    ranked_frogs = rank_frogs(frogs, metric)

    return ranked_frogs[0], [ranked_frogs[i::num_of_memeplexes] for i in range(num_of_memeplexes)]


def evolve(worst_frog, best_frog, data, metric):
    worst_centroids = worst_frog["centroids"]
    best_centroids = best_frog["centroids"]
    new_centroids = worst_centroids + (np.random.rand() * np.subtract(best_centroids, worst_centroids))
    fit = fitness(data, new_centroids, metric)
    return {"centroids": new_centroids, "fit": fit}


def is_more_fit(frog1, frog2, metric):
    if metric == 'ecludian' or metric == 'davies_bouldin':
        return frog1["fit"] < frog2["fit"]
    else:
        return frog1["fit"] > frog2["fit"]


def sfla(data, num_of_frogs=30, num_of_clusters=3, num_of_memeplexes=5, memeplex_iterations=10, max_iterations=50,
         metric='ecludian'):
    """

    :param data:
    :param num_of_frogs: Total number of frogs (F)
    :param num_of_clusters: How many clusters are desired (k)
    :param num_of_memeplexes: Number of memeplexes (m)
    :param memeplex_iterations: Number of memeplex iterations (iN)
    :param max_iterations: Maximum number of iterations before the algorithm will terminate
    :param metric: possible values: ecludian
    :return:
    """
    # todo: early exit if same frog remains best

    all_frogs = generate_population(data, num_of_frogs, num_of_clusters, metric)

    for i in range(max_iterations):
        # pg is global best frog
        pg, memeplexes = create_memeplexes(all_frogs, num_of_memeplexes, metric)

        new_memeplexes = []
        for im in memeplexes:
            for iN in range(memeplex_iterations):
                im = rank_frogs(im, metric)  # re-rank frogs
                pb = im[0]  # local best frog
                pw = im[-1]  # local worst frog

                # evolve the frog
                new_frog = evolve(pw, pb, data, metric)

                # if fitness doesn't improve try with global best frog
                if not is_more_fit(new_frog, pw, metric):
                    new_frog = evolve(pw, pg, data, metric)

                # if still doesn't improve, generate a new frog
                if not is_more_fit(new_frog, pw, metric):
                    new_frog = generate_frog(data, num_of_clusters, metric)

                im[-1] = new_frog

            pb = im[0]
            if is_more_fit(pb, pg, metric):
                pg = pb
            new_memeplexes.append(im)

        all_frogs = list(itertools.chain(*new_memeplexes))

    pg = rank_frogs(all_frogs, metric)[0]
    return pg


# todo: add comments and docstrings
class SFLAClustering(BaseEstimator, ClusterMixin):

    def __init__(self, num_of_frogs=30, num_of_clusters=3, num_of_memeplexes=5, memeplex_iterations=10,
                 max_iterations=50, metric='ecludian'):
        self.num_of_frogs = num_of_frogs
        self.num_of_clusters = num_of_clusters
        self.num_of_memeplexes = num_of_memeplexes
        self.memeplex_iterations = memeplex_iterations
        self.max_iterations = max_iterations
        self.metric = metric

    def fit(self, X):
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
