from typing import List, Set

import numpy as np
import scipy as sp
import scipy.spatial
import matplotlib.pyplot as plt

from src.util import linear_regression


def draw_mat(a: np.ndarray, name: str = None):
    plt.title(name)
    plt.matshow(a)
    if name is not None:
        plt.savefig(f"{name}.png")
    plt.show()
    plt.clf()


def draw_size(size_list: List[int], bins: int = 10, name: str= None, log: bool=False):
    hist, edges = np.histogram(np.array(size_list), bins=bins)
    centers = []
    for i in range(len(hist)):
        centers.append((edges[i] + edges[i+1])/2)
    if not log:
        plt.title(name)
        plt.xlabel("cluster size")
        plt.ylabel("occurrences")
        plt.bar(centers, hist, width=(centers[1] - centers[0]))
    else:
        log_centers = [] # log
        log_hist = [] # log
        for i in range(len(hist)):
            if hist[i] > 0:
                log_centers.append(np.log(centers[i]))
                log_hist.append(np.log(hist[i]))
        plt.plot(log_centers, log_hist)
        # linear regression
        X = np.array(log_centers).reshape((len(log_centers), 1))
        y = np.array(log_hist).reshape((len(log_hist), 1))
        coef, intercept = linear_regression(X, y)
        coef = float(coef)
        intercept = float(intercept)
        minX = min(log_centers)
        maxX = max(log_centers)
        minY = coef * minX + intercept
        maxY = coef * maxX + intercept
        plt.plot([minX, maxX], [minY, maxY])
        name = f"{name}: log(y) = {coef}*log(x) + {intercept}"
        plt.title(name)
        plt.xlabel("log(cluster size)")
        plt.ylabel("log(occurrences)")


    if name is not None:
        plt.savefig(f"{name}.png")
    plt.show()
    plt.clf()


def draw_data(data: np.ndarray, cluster_list: List[Set[int]] = [], name: str= None):
    plt.title(name)
    plt.scatter(data[:, 0], data[:, 1], s=2)

    for cluster in cluster_list:
        cluster_list = data[list(cluster), :]
        if len(cluster) < 3:
            plt.plot(cluster_list[:, 0], cluster_list[:, 1], linewidth=2)
        else:
            hull = sp.spatial.ConvexHull(cluster_list)
            vertices = list(hull.vertices) + [hull.vertices[0]]
            plt.plot(cluster_list[vertices, 0], cluster_list[vertices, 1], linewidth=2)

    if name is not None:
        plt.savefig(f"{name}.png")
    plt.show()
    plt.clf()

