import numpy as np
from typing import Optional, Union, Tuple


def euclidean_distance(x: np.array, y: np.array) -> float:
    """
    Calculate euclidean distance between points x and y
    Args:
        x, y: two points in Euclidean n-space
    Returns:
        Length of the line segment connecting given points
    """
    return np.sum((x - y) ** 2) ** 0.5


def euclidean_similarity(x: np.array, y: np.array) -> float:
    """
    Calculate euclidean similarity between points x and y
    Args:
        x, y: two points in Euclidean n-space
    Returns:
        Similarity between points x and y
    """
    return 1 / (1 + euclidean_distance(x, y))


def pearson_similarity(x: np.array, y: np.array) -> float:
    """
    Calculate a Pearson correlation coefficient given 1-D data arrays x and y
    Args:
        x, y: two points in n-space
    Returns:
        Pearson correlation between x and y
    """
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denumerator = (np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2)) ** 0.5
    if denumerator == 0:
	return 0
    return numerator / denumerator

	
def apk(actual: np.array, predicted: np.array, k: int = 10) -> float:
    """
    Compute the average precision at k
    Args:
        actual: a list of elements that are to be predicted (order doesn't matter)
        predicted: a list of predicted elements (order does matter)
        k: the maximum number of predicted elements
    Returns:
        The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if len(actual) == 0:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual: np.array, predicted: np.array, k: int = 10) -> float:
    """
    Compute the mean average precision at k
    Args:
        actual: a list of lists of elements that are to be predicted
        predicted: a list of lists of predicted elements
        k: the maximum number of predicted elements
    Returns:
        The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])
