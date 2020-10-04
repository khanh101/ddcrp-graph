from typing import List, Set

import numpy as np
import scipy as sp
import scipy.spatial
import matplotlib.pyplot as plt

def draw_mat(a: np.ndarray):
    plt.matshow(a)
    plt.show()


def draw_size(size_list: List[int]):
    hist, edges = np.histogram(np.array(size_list), bins=10)
    centers = []
    for i in range(len(hist)):
        centers.append((edges[i] + edges[i+1])/2)
    plt.scatter(centers, hist)
    plt.show()


def draw_data(data: np.ndarray, cluster_list: List[Set[int]] = []):
    plt.scatter(data[:, 0], data[:, 1], s=2)

    for cluster in cluster_list:
        cluster_list = data[list(cluster), :]
        if len(cluster) < 3:
            plt.plot(cluster_list[:, 0], cluster_list[:, 1], linewidth=2)
        else:
            hull = sp.spatial.ConvexHull(cluster_list)
            vertices = list(hull.vertices) + [hull.vertices[0]]
            plt.plot(cluster_list[vertices, 0], cluster_list[vertices, 1], linewidth=2)

    plt.show()

