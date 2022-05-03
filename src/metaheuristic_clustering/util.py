import numpy as np


def get_euclidean_distance(row, centroids):
    """ Calculates the euclidean distance between a row and each centroid """
    return [np.linalg.norm(row - centroid) for centroid in centroids]


def get_centroid_labels(data, centroids):
    """ Assigns each data point to its closest centroid based on euclidean distance """
    labels = []
    for row in data:
        distances = get_euclidean_distance(row, centroids)
        min_distance = min(distances)
        index = distances.index(min_distance)
        labels.append(index)

    return labels


def get_labels(data, element):
    return get_centroid_labels(data, element["centroids"])


def euclidean_fitness(data, centroids):
    """ Calculates the average distance between each data point and it's closest cluster center """
    total_distance = 0
    dimensions = data.shape[1]
    for row in data:
        # calculate the euclidean distance between each row and each centroid
        distances = get_euclidean_distance(row, centroids)
        # use the centroid the row is closest to
        total_distance += min(distances)

    # get the average distance
    avg_distance = total_distance / dimensions
    return avg_distance


# Todo: support other fitness measures
def fitness(data, centroids, metric):
    """
    Calculates how well a set of centroids fit a data set based on a metric

    Currently only supports Euclidean distance - the average distance between each data point and it's closest cluster center
    """
    fit = None
    if metric == 'euclidean':
        fit = euclidean_fitness(data, centroids)

    return fit
