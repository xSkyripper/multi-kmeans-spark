import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm


def main():
    pass


def plot_clusters(data_items, centroids, clusters, title):
    k = len(centroids)

    final_centroids = [[] for _ in range(k)]
    final_clusters = [[] for _ in range(k)]

    for centroid_index, centroid in centroids:
        final_centroids[centroid_index].extend(centroid)

    for centroid_index, points_indexes in clusters:
        for point_idx in points_indexes:
            assert data_items[point_idx][0] == point_idx, 'Should not happen'
            final_clusters[centroid_index].append(data_items[point_idx][1])

    plt.figure("K-Means: {}".format(title))
    lspace = np.linspace(0.0, 1.0, k * 2)
    colors = cm.rainbow(lspace)

    for cluster_index in range(k):
        cluster_color = colors[cluster_index]
        cluster_matrix = np.asmatrix(final_clusters[cluster_index])
        centroids_matrix = np.asmatrix(final_centroids[cluster_index])

        plt.scatter(
            x=np.ravel(cluster_matrix[:, 0]), y=np.ravel(cluster_matrix[:, 1]),
            marker='.', s=100, c=cluster_color
        )

        plt.scatter(
            x=np.ravel(centroids_matrix[:, 0]), y=np.ravel(centroids_matrix[:, 1]),
            marker='*', s=200, c=cluster_color,
            edgecolors="black"
        )

    plt.show()


if __name__ == '__main__':
    main()
