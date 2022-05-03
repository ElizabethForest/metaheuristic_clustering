# Based on:
# Karaboga and C. Ozturk, "A novel clustering approach: Artificial Bee Colony (ABC) algorithm," Applied soft computing
# https://github.com/ntocampos/artificial-bee-colony

from pyclustering.cluster.center_initializer import random_center_initializer as rand_init
from sklearn.base import ClusterMixin, BaseEstimator
import numpy as np
import random

from .util import fitness, get_labels


def new_bee(centroids, fit):
    new_fitness = 1 / (1 + fit)
    return {"centroids": centroids,
            "fit": fit,
            "fitness": new_fitness,
            "discards": 0}


def generate_bee(data, num_of_clusters, metric):
    centroids = rand_init(data, num_of_clusters).initialize()  # can also use k_init for kmeans++ initalisation
    fit = fitness(data, centroids, metric)
    return new_bee(centroids, fit)


def generate_population(data, num_of_frogs, num_of_clusters, metric):
    return [generate_bee(data, num_of_clusters, metric) for i in range(num_of_frogs)]


def generate_new_centroids(original, updater, dimensions):
    phi = np.random.uniform(-1, 1, dimensions)
    return original["centroids"] + (phi * np.subtract(original["centroids"], updater["centroids"]))


def update_bee(bee, k, dimensions, data, metric):
    new_centroids = generate_new_centroids(bee, k, dimensions)
    new_fit = fitness(data, new_centroids, metric)

    if new_fit < bee["fit"]:
        bee = new_bee(new_centroids, new_fit)
    else:
        bee["discards"] += 1

    return bee


def abc(data, num_bees=30, num_of_clusters=3, max_iterations=50, metric='euclidean', discard_limit=20):
    """
    :param data: the data to be clustered
    :param num_bees: the number of bees  - this is the number of potential solutions to be considered at one time
    :param num_of_clusters: the number of clusters (k)
    :param max_iterations: the max number of iterations
    :param metric: can only be Euclidean distance atm
    :param discard_limit: the max number of times for a bee to not improve before being re-created
    :return: The best solution after the max number of iterations
    """

    dimensions = data.shape[1]

    all_bees = generate_population(data, num_bees, num_of_clusters, metric)

    best_solution = None

    for i in range(max_iterations):
        new_bees = []
        print("iteration:", i)

        for worker in all_bees:
            random_bee = random.choice(all_bees)  # get a random choice
            bee = update_bee(worker, random_bee, dimensions, data, metric)
            new_bees.append(bee)

        all_bees = new_bees
        new_bees = []

        fitness_sum = sum(x["fitness"] for x in all_bees)

        for bee in all_bees:
            bee["prob"] = bee["fitness"] / fitness_sum
            new_bees.append(bee)

        all_bees = new_bees
        new_bees = []

        for onlooker in all_bees:
            sorted_bees = sorted(all_bees, key=lambda x: x["prob"])
            choice = random.randint(0, (num_bees // 2))
            # Todo: should randomly choose a bee thats > threshold
            random_bee = sorted_bees[choice]

            bee = update_bee(onlooker, random_bee, dimensions, data, metric)
            new_bees.append(bee)

        all_bees = new_bees
        new_bees = []

        for scout in all_bees:
            if scout["discards"] > discard_limit:
                bee = generate_bee(data, num_of_clusters, metric)
                new_bees.append(bee)
            else:
                new_bees.append(scout)

        all_bees = new_bees
        current_best = sorted(all_bees, key=lambda k: k["fit"])[0]

        if best_solution is None or current_best["fit"] < best_solution["fit"]:
            best_solution = current_best

    return best_solution


# TODO add comments and docstrings
class ABCClustering(BaseEstimator, ClusterMixin):
    """
    Creates clusters based on the Artificial Bee Colony optimisation algorithm

    Karaboga and C. Ozturk, "A novel clustering approach: Artificial Bee Colony (ABC) algorithm," Applied soft computing

    Also based on an implementation by: https://github.com/ntocampos/artificial-bee-colony

    """

    def __init__(self, num_bees=30, num_of_clusters=3, max_iterations=50, metric='euclidean', discard_limit=20):
        """
        :param num_bees: the number of bees  - this is the number of potential solutions to be considered at one time
        :param num_of_clusters: the number of clusters to be created (k)
        :param max_iterations: the max number of iterations
        :param metric: can only be Euclidean distance atm
        :param discard_limit: the max number of times for a bee to not improve before being re-created

        """
        self.num_bees = num_bees
        self.num_of_clusters = num_of_clusters
        self.max_iterations = max_iterations
        self.metric = metric
        self.discard_limit = discard_limit

    def fit(self, X):
        """
        :param X: the data to be clustered
        :return: the updated object
        """
        if not isinstance(X, np.ndarray):
            X = X.to_numpy()
        best_bee = abc(X,
                       num_bees=self.num_bees,
                       num_of_clusters=self.num_of_clusters,
                       max_iterations=self.max_iterations,
                       metric=self.metric,
                       discard_limit=self.discard_limit)
        self.labels_ = get_labels(X, best_bee)
        return self
