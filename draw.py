from typing import List, Set

import numpy as np
import matplotlib.pyplot as plt

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

