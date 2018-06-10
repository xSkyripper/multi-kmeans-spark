import time
from pprint import pprint

import click
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from pyspark.sql import SparkSession
from scipy.spatial.distance import mahalanobis as sp_mahalanobis


def parse_vector(line):
    return np.array([float(x) for x in line.split(" ")])


def dist_mahalanobis(x1, x2, cov_mat):
    return sp_mahalanobis(x1, x2, cov_mat.T)


def dist_euclidean(x1, x2):
    return np.linalg.norm(x1 - x2)


def compute_cov_mat(data_items):
    # print("Computing cov mat for ")
    # pprint(data_items)
    if len(data_items) < 2:
        return None
    vstack = np.vstack(data_items)
    return np.cov(vstack.T)


def preliminary_step(k, data_items):
    # 1. Calculate K preliminary centroids
    centroids = data_items.takeSample(False, k)
    # print('\nPS: Centroids')
    # pprint(centroids)

    # 2. Assign points to centroids
    # assign points to centroids
    closest = data_items.map(lambda point: (euclidean_closest_point(point, centroids, k), (point, 1)))
    # print('\nPS: closest - assign points to centroids')
    # pprint(closest.collect())

    # 3. Compute the centroid Ck belonging
    # sum points dimensions for each centroid and add 1 count for each point
    point_stats = closest.reduceByKey(
        lambda p1_c1, p2_c2: (
            p1_c1[0] + p2_c2[0],
            p1_c1[1] + p2_c2[1],
        )
    )
    # print('\nPS: point stats')
    # pprint(point_stats.collect())

    # recalculate centroids
    centroids_new = point_stats.map(lambda st: (st[0], st[1][0] / st[1][1])).collect()
    # print('\nPS: new centroids')
    # pprint(centroids_new)

    # 2. Assign points to centroids
    closest = data_items.map(lambda point: (euclidean_closest_point(point, centroids, k), point))
    # print('\nPS: closest 2')
    # pprint(closest.collect())

    # 3. Compute the Sk
    grouped_by_cluster = closest.groupByKey().map(lambda x: (x[0], list(x[1])))
    # print('\nPS: grouped by cluster')
    # pprint(grouped_by_cluster.collect())

    sk = grouped_by_cluster.map(lambda r: (r[0], compute_cov_mat(r[1])))
    # print('\nPS: sk')
    # pprint(sk.collect())

    for centroid_idx, centroid in centroids_new:
        centroids[centroid_idx] = centroid

    return centroids, sk.collect()


def euclidean_closest_point(point, centroids, k):
    best_index = 0
    smallest_dist = np.inf

    for idx in range(k):
        temp_dist = dist_euclidean(point, centroids[idx])
        if temp_dist < smallest_dist:
            smallest_dist = temp_dist
            best_index = idx

    return best_index


def closest_point(point, centroids, cov_mat):
    best_index = 0
    min_distance = np.inf

    for i in range(len(centroids)):
        centroid_sk = None
        for j in cov_mat:
            if j[0] == i:
                centroid_sk = j[1]
                break

        if centroid_sk is not None:
            # print("Trying to mahalanobis btw")
            # pprint(point)
            # pprint(centroids[i])
            # pprint(centroid_sk)
            distance = dist_mahalanobis(point, centroids[i], centroid_sk)
        else:
            distance = dist_euclidean(point, centroids[i])

        if distance < min_distance:
            min_distance, best_index = distance, i

    return best_index


def mahalanobis(input_file, no_clusters, convergence_dist, max_iterations, plot):
    start_time = time.time()
    spark = SparkSession.builder.appName('KMeans - Mahalanobis Distance').getOrCreate()
    lines = spark.read.text(input_file).rdd.map(lambda r: r[0])
    data_items = lines.map(parse_vector).cache()

    iterations = 0
    centroids_delta_dist = 1.0
    centroids, sk = preliminary_step(no_clusters, data_items)
    max_iterations = max_iterations or np.inf

    while centroids_delta_dist > convergence_dist and iterations < max_iterations:
        # Compute new clusters with Mahalanobis and assign

        closest = data_items.map(lambda p: (closest_point(p, centroids, sk), (p, 1)))
        point_stats = closest.reduceByKey(lambda p1_c1, p2_c2: (p1_c1[0] + p2_c2[0], p1_c1[1] + p2_c2[1]))
        # Compute new centroids
        new_points = point_stats.map(lambda st: (st[0], st[1][0] / st[1][1])).collect()

        # Compute new covariance per cluster
        # closest = data.map(lambda point: (euclidean_closest_point(point, centroids), (point[0], point[1])))
        closest = data_items.map(lambda p: (closest_point(p, centroids, sk), p))
        grouped_by_cluster = closest.groupByKey().map(lambda x: (x[0], list(x[1])))
        print("Grouped by clusters")
        # pprint(grouped_by_cluster.collect())

        sk = grouped_by_cluster.map(lambda r: (r[0], compute_cov_mat(r[1]))).collect()

        centroids_delta_dist = sum(np.sum((centroids[iK] - p) ** 2) for (iK, p) in new_points)

        for (iK, p) in new_points:
            centroids[iK] = p

        iterations += 1
        print(centroids_delta_dist)
        print("Iteration {} done".format(iterations))
        print("Iteration Time: {}".format(time.time() - start_time))
        start_time = time.time()

    print("Final centroids in {} iterations:".format(iterations))
    pprint(centroids)

    def plot_mahalanobis(data_items, centroids, clusters, k):
        data_items_indexed = data_items \
            .zipWithIndex() \
            .map(lambda x: (x[1], x[0])) \
            .collect()

        centroids_indexed = [(index, point) for index, point in enumerate(centroids)]
        plot_clusters(data_items_indexed, centroids_indexed, clusters.collect(), k)

        pass

    if plot:
        print("Data Items")
        pprint(data_items.collect())

        print("Centroids")
        pprint(centroids)

        print("Clusters")
        pprint(grouped_by_cluster.collect())

        plot_mahalanobis(data_items, centroids, grouped_by_cluster, no_clusters)

    spark.stop()


def plot_clusters(data_items, centroids, clusters, title):
    k = len(centroids)

    final_centroids = [[] for _ in range(k)]
    final_clusters = [[] for _ in range(k)]

    for centroid_index, centroid in centroids:
        final_centroids[centroid_index].extend(centroid)

    for centroid_index, points in clusters:
        for point in points:
            # assert data_items[point_idx][0] == point_idx, 'Should not happen'
            final_clusters[centroid_index].append(point)

    plt.figure("K-Means: {}".format(title))
    lspace = np.linspace(0.0, 1.0, k * 2)
    pprint(lspace)
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


@click.command()
@click.option('-f', '--file', required=True)
@click.option('-k', '--no-clusters', required=True, type=click.INT)
@click.option('-c', '--convergence-dist', required=True, type=click.FLOAT)
@click.option('-i', '--max-iterations', type=click.INT)
@click.option('--plot', is_flag=True)
def main(file, no_clusters, convergence_dist, max_iterations, plot):
    mahalanobis(file, no_clusters, convergence_dist, max_iterations, plot)


if __name__ == '__main__':
    main()
