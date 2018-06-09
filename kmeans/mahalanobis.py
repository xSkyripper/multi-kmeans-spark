import click
import numpy as np
from pprint import pprint

import time
from pyspark.sql import SparkSession
from scipy.spatial.distance import mahalanobis as sp_mahalanobis
from kmeans.utils import plot_clusters


def parse_vector(line, delimiter):
    return np.array([float(x) for x in line.split(delimiter)])


def dist_mahalanobis(x1, x2, cov_mat):
    return sp_mahalanobis(x1, x2, cov_mat.T)


def dist_euclidean(x1, x2):
    return np.linalg.norm(x1 - x2)


def compute_cov_mat(data_items):
    if len(data_items) < 2:
        return None
    vstack = np.vstack(data_items)
    return np.cov(vstack.T)


def preliminary_step(k, data_items):
    # 1. Calculate K preliminary centroids
    centroids = data_items.takeSample(False, k)

    # 2. Assign points to centroids
    closest = data_items.map(lambda point: (euclidean_closest_point(point, centroids, k), (point, 1)))

    # 3. Compute the centroid Ck belonging (sum points dimensions for each centroid and add 1 count for each point)
    point_stats = closest.reduceByKey(
        lambda p1_c1, p2_c2: (
            p1_c1[0] + p2_c2[0],
            p1_c1[1] + p2_c2[1],
        )
    )

    # recalculate centroids
    centroids_new = point_stats.map(lambda st: (st[0], st[1][0] / st[1][1])).collect()

    # 2. Assign points to centroids
    closest = data_items.map(lambda point: (euclidean_closest_point(point, centroids, k), point))

    # 3. Compute the Sk
    grouped_by_cluster = closest.groupByKey().map(lambda x: (x[0], list(x[1])))

    sk = grouped_by_cluster.map(lambda r: (r[0], compute_cov_mat(r[1])))

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
            distance = dist_mahalanobis(point, centroids[i], centroid_sk)
        else:
            distance = dist_euclidean(point, centroids[i])

        if distance < min_distance:
            min_distance, best_index = distance, i

    return best_index


NUM_PARTITIONS = 4


def mahalanobis(input_file, delimiter, no_clusters, convergence_dist, max_iterations, plot):
    start_time = time.time()
    spark = SparkSession.builder.appName('KMeans - Mahalanobis Distance').getOrCreate()
    lines = spark.read.text(input_file).rdd.map(lambda r: r[0])
    data_items = lines.map(lambda l: parse_vector(l, delimiter)).cache()
    max_iterations = max_iterations or np.inf

    iterations = 0
    centroids_delta_dist = 1.0
    centroids, sk = preliminary_step(no_clusters, data_items)

    while centroids_delta_dist > convergence_dist and iterations < max_iterations:
        # Compute new clusters with Mahalanobis and assign
        closest = data_items.map(lambda p: (closest_point(p, centroids, sk), (p, 1)))
        point_stats = closest \
            .reduceByKey(lambda p1_c1, p2_c2: (p1_c1[0] + p2_c2[0], p1_c1[1] + p2_c2[1]), numPartitions=NUM_PARTITIONS)
        # Compute new centroids
        new_points = point_stats.map(lambda st: (st[0], st[1][0] / st[1][1])).collect()

        # Compute new covariance per cluster
        # closest = data.map(lambda point: (euclidean_closest_point(point, centroids), (point[0], point[1])))
        closest = data_items.map(lambda p: (closest_point(p, centroids, sk), p))
        grouped_by_cluster = closest \
            .groupByKey(numPartitions=NUM_PARTITIONS) \
            .map(lambda x: (x[0], list(x[1])))
        sk = grouped_by_cluster.map(lambda r: (r[0], compute_cov_mat(r[1]))).collect()

        centroids_delta_dist = sum(np.sum((centroids[iK] - p) ** 2) for (iK, p) in new_points)

        for (iK, p) in new_points:
            centroids[iK] = p

        iterations += 1
        print("Iteration {} done".format(iterations))
        print("Iteration Time {}".format(time.time() - start_time))
        start_time = time.time()

    print("Final centroids in {} iterations:".format(iterations))
    pprint(centroids)

    def plot_mahalanobis(data_items, centroids, clusters, k):
        # preparing data for plotting
        print('Data items indexed')
        data_items_indexed = data_items \
            .zipWithIndex() \
            .map(lambda x: (x[1], x[0])) \
            .collect()
        pprint(data_items_indexed)

        print('Centroids indexed')
        centroids_indexed = list(zip([i for i in range(no_clusters)], centroids))
        pprint(centroids_indexed)

        print('Clusters indexed')
        clusters_indexed = closest \
            .zipWithIndex() \
            .map(lambda x: (x[0][0], x[1])) \
            .groupByKey() \
            .map(lambda x: (x[0], list(x[1]))) \
            .collect()
        pprint(clusters_indexed)

        plot_clusters(data_items_indexed, centroids_indexed, clusters_indexed,
                      'Mahalanobis')

    if plot:
        plot_mahalanobis(data_items, centroids, closest, no_clusters)

    spark.stop()


@click.command()
@click.option('-f', '--input-file', required=True)
@click.option('-d', '--delimiter', default=' ')
@click.option('-k', '--no-clusters', required=True, type=click.INT)
@click.option('-c', '--convergence-dist', required=True, type=click.FLOAT)
@click.option('-i', '--max-iterations', type=click.INT)
@click.option('--plot', is_flag=True)
def main(input_file, delimiter, no_clusters, convergence_dist, max_iterations, plot):
    mahalanobis(input_file, delimiter, no_clusters, convergence_dist, max_iterations, plot)


if __name__ == '__main__':
    main()
