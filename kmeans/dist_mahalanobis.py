import click
import numpy as np
from pprint import pprint
from pyspark.sql import SparkSession
from scipy.spatial.distance import mahalanobis
from kmeans.default import closest_point as euclidean_closest_point


def parse_vector(line):
    return np.array([float(x) for x in line.split(" ")])


def dist_mahalanobis(point1, point2, sk):
    return mahalanobis(point1, point2, sk.T)


def dist_euclidean(point1, point2):
    return np.sum((point1 - point2) ** 2)


def closest_point(point, centroids, sk):
    best_index = 0
    min_distance = float("+inf")

    for i in range(len(centroids)):
        centroid_sk = None

        for j in sk:
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


def compute_cov_mat(points):
    if len(points) < 2:
        return None
    stack = np.stack(points)
    return np.cov(stack.T)


def preliminary_step(k, points):
    # 1. Calculate K preliminary centroids
    # centroids = points.takeSample(False, k, 1)
    centroids = np.array([(1.90, 0.97), (3.17, 4.96)])
    print("initial centroids")
    pprint(centroids)

    # 2. Assign points to centroids
    # assign points to centroids
    closest = points.map(lambda point: (euclidean_closest_point(point, centroids), (point, 1)))

    # 3. Compute the centroid Ck belonging
    # sum points dimensions for each centroid and add 1 count for each point
    point_stats = closest.reduceByKey(
        lambda p1_c1, p2_c2: (
            p1_c1[0] + p2_c2[0],
            p1_c1[1] + p2_c2[1],
        )
    )

    # recalculate centroids
    centroids_new = point_stats.map(lambda st: (st[0], st[1][0] / st[1][1])).collect()

    # 2. Assign points to centroids
    # assign points to centroids
    closest = points.map(lambda point: (euclidean_closest_point(point, centroids), (point[0], point[1])))

    # 3. Compute the Sk
    grouped_by_cluster = closest.groupByKey().map(lambda x: (x[0], list(x[1])))
    sk = grouped_by_cluster.map(lambda r: (r[0], compute_cov_mat(r[1])))

    for centroid_idx, centroid in centroids_new:
        centroids[centroid_idx] = centroid

    return centroids, sk.collect()


@click.command()
@click.option('-f', '--file', required=True)
@click.option('-k', '--no-clusters', required=True)
@click.option('-c', '--converge-dist', required=True)
def main(file, no_clusters, converge_dist):
    spark = SparkSession.builder.appName('KMeans - Mahalanobis Distance').getOrCreate()
    lines = spark.read.text(file).rdd.map(lambda r: r[0])
    data = lines.map(parse_vector).cache()
    k = int(no_clusters)
    converge_dist = float(converge_dist)

    # k_points = data.take_sample(False, k, 1)
    # centroids = np.array([(1.90, 0.97), (3.17, 4.96)])
    centroids_delta_dist = 1.0

    centroids, sk = preliminary_step(k, data)
    pprint(centroids)
    # pprint(sk.collect())

    while centroids_delta_dist > converge_dist:
        closest = data.map(lambda p: (closest_point(p, centroids, sk), (p, 1)))
        pprint(centroids)
        pprint(closest.collect())
        point_stats = closest.reduceByKey(lambda p1_c1, p2_c2: (p1_c1[0] + p2_c2[0], p1_c1[1] + p2_c2[1]))
        new_points = point_stats.map(lambda st: (st[0], st[1][0] / st[1][1])).collect()

        closest = data.map(lambda point: (euclidean_closest_point(point, centroids), (point[0], point[1])))
        grouped_by_cluster = closest.groupByKey().map(lambda x: (x[0], list(x[1])))
        sk = grouped_by_cluster.map(lambda r: (r[0], compute_cov_mat(r[1]))).collect()

        centroids_delta_dist = sum(np.sum((centroids[iK] - p) ** 2) for (iK, p) in new_points)

        for (iK, p) in new_points:
            centroids[iK] = p

    print("Final centroids:")
    pprint(centroids)
    spark.stop()


if __name__ == '__main__':
    main()