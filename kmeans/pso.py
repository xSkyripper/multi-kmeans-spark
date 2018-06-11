import click
import numpy as np
import math
import time

from pyspark import StorageLevel
from sklearn.neighbors import NearestNeighbors
from pyspark.mllib.clustering import KMeans
from pyspark.sql import SparkSession
from pprint import pprint


def parse_vector(line):
    return np.array([float(x) for x in line.split(',')])


def euclidean_distance(x1, x2):
    return np.linalg.norm(x1 - x2)


def compute_ns(n, k):
    return max(int(0.1 * (n / k)), 10)


def compute_nearest_neighbors(sc, data_items, ns):
    knn_model = NearestNeighbors(n_neighbors=ns + 1, algorithm='kd_tree')
    knn_model.fit(data_items.collect())
    bc_knn_model = sc.broadcast(knn_model)

    zipped = data_items.zipWithIndex()
    ns_neighbors = zipped.map(lambda x: (x[1],
                                         bc_knn_model.value.kneighbors(X=[x[0]], return_distance=False)[0][1:]))
    return ns_neighbors.collect()


def compute_squared_sigma(data_items, n, kmeans_model):
    centroids = kmeans_model.clusterCenters
    squared_distances = data_items.map(
        lambda x: (euclidean_distance(x, centroids[kmeans_model.predict(x)]) ** 2))
    sum_term = squared_distances.sum()
    return sum_term / n


def compute_clusters(points, kmeans_model):
    k = kmeans_model.k
    clusters = [[] for _ in range(k)]

    for point_idx, point in enumerate(points):
        cluster_idx = kmeans_model.predict(point)
        clusters[cluster_idx].append(point_idx)

    return clusters


def compute_personal_best(nearest_neighbors_indexes, positions):
    sum_term = np.sum([positions[idx]
                       for idx in nearest_neighbors_indexes], axis=0)
    return sum_term / len(nearest_neighbors_indexes)


def compute_global_best(point, kmeans_model):
    centroids = kmeans_model.clusterCenters
    return centroids[kmeans_model.predict(point)]


def compute_position_velocity(position, velocity, personal_best, global_best, squared_sigma):
    dist = euclidean_distance(position, global_best)
    weight = 1 if dist < 0.125 * math.sqrt(squared_sigma) else 0

    new_velocity = np.array(velocity) + \
                   (np.array(personal_best) - np.array(position)) + \
                   (np.array(global_best) - np.array(position)) * weight

    new_position = np.array(position) + new_velocity

    return list(new_position), list(new_velocity)


def stop_condition(k, clusters, new_clusters, itr):
    for cluster_idx in range(k):
        if set(clusters[cluster_idx]) != set(new_clusters[cluster_idx]):
            return 0

    return itr + 1


def computer_nn(model, point):
    model = model.value
    return model.kneighbors(X=[point], return_distance=False)[0][1:]


def computer_clusters2(bc_kmeans_model, point_index, point):
    return bc_kmeans_model.value.predict(point), point_index


def computer_personal_best2(point, neighbours, ns):
    return point, np.sum(neighbours, axis=0) / ns


def computer_global_best2(bc_kmeans_model, point, centroids):
    return centroids[bc_kmeans_model.value.predict(point)]


def computer_velocity(data, squared_sigma):
    velocity, point, personal_best, global_best = data
    dist = euclidean_distance(point, global_best)
    weight = 1 if dist < 0.125 * math.sqrt(squared_sigma) else 0

    new_velocity = velocity + (personal_best - point) + weight * (global_best - point)
    return new_velocity


def computer_new_point_position(point, velocity):
    new_position = point + velocity
    return new_position


NUM_PARTITIONS = 16


@click.command()
@click.option('-f', '--file', required=True)
@click.option('-k', '--no-clusters', required=True, type=click.INT)
@click.option('-i', '--max-iterations', type=click.INT)
@click.option('--itr', required=True, type=click.INT)
@click.option('--ns', type=click.INT)
def main(file, no_clusters, max_iterations, ns, itr):
    start_time = time.time()
    spark = SparkSession.builder.appName("KMeans - PSO").getOrCreate()
    lines = spark.read.text(file).rdd.map(lambda r: r[0])
    data_items = lines.map(parse_vector).persist()
    zipped_date_items = data_items \
        .zipWithIndex() \
        .map(lambda x: (x[1], x[0])) \
        .persist()

    n = data_items.count()
    ns = ns or compute_ns(n, no_clusters)

    knn_model = NearestNeighbors(n_neighbors=ns + 1, algorithm='kd_tree')
    knn_model.fit(data_items.collect())
    knn_model = spark.sparkContext.broadcast(knn_model)

    nearest_neighbors = data_items \
        .zipWithIndex() \
        .map(lambda x: (x[1], computer_nn(knn_model, x[0]))) \
        .persist(StorageLevel.MEMORY_AND_DISK)

    kmeans_model = KMeans.train(data_items, no_clusters,
                                maxIterations=max_iterations or 100, initializationMode='random')
    centroids = kmeans_model.clusterCenters

    bc_kmeans_model = spark.sparkContext.broadcast(kmeans_model)
    squared_sigma = compute_squared_sigma(data_items, n, kmeans_model)

    max_iterations = max_iterations or np.inf
    iterations = 0
    convergence_iterations = 0
    velocities = spark.sparkContext.parallelize([(i, 0.0) for i in range(n)])
    clusters = zipped_date_items \
        .map(lambda point: computer_clusters2(bc_kmeans_model, point[0], point[1])) \
        .groupByKey(numPartitions=NUM_PARTITIONS) \
        .map(lambda x: list(x[1])) \
        .persist(StorageLevel.DISK_ONLY)

    while True:
        personal_bests = nearest_neighbors \
            .flatMap(lambda row: [(row[0], y) for y in row[1]]) \
            .map(lambda x: (x[1], x[0])) \
            .join(zipped_date_items, numPartitions=NUM_PARTITIONS) \
            .map(lambda x: (x[1][0], x[1][1])) \
            .groupByKey(numPartitions=NUM_PARTITIONS) \
            .mapValues(lambda points: list(points)) \
            .map(lambda x: computer_personal_best2(x[0], x[1], ns)) \
            .cache()


        global_bests = zipped_date_items \
            .mapValues(lambda point: computer_global_best2(bc_kmeans_model, point, centroids)) \
            # .cache()

        velocities = velocities \
            .join(zipped_date_items, numPartitions=NUM_PARTITIONS) \
            .join(personal_bests, numPartitions=NUM_PARTITIONS) \
            .mapValues(lambda x: (x[0][0], x[0][1], x[1])) \
            .join(global_bests, numPartitions=NUM_PARTITIONS) \
            .mapValues(lambda x: (x[0][0], x[0][1], x[0][2], x[1])) \
            .mapValues(lambda x: computer_velocity(x, squared_sigma)) \
            # .cache()

        zipped_date_items = zipped_date_items \
            .join(velocities, numPartitions=NUM_PARTITIONS) \
            .mapValues(lambda x: computer_new_point_position(x[0], x[1])) \
            .cache()

        new_positions = zipped_date_items \
            .map(lambda x: x[1])

        new_kmeans_model = KMeans.train(new_positions, no_clusters, maxIterations=1, initialModel=kmeans_model)
        bc_new_kmeans_model = spark.sparkContext.broadcast(new_kmeans_model)
        new_clusters = zipped_date_items \
            .map(lambda point: computer_clusters2(bc_new_kmeans_model, point[0], point[1])) \
            .groupByKey(numPartitions=NUM_PARTITIONS) \
            .map(lambda x: list(x[1])) \
            .persist(StorageLevel.MEMORY_AND_DISK)

        convergence_iterations = stop_condition(no_clusters, clusters.collect(), new_clusters.collect(),
                                                convergence_iterations)
        iterations += 1
        if convergence_iterations >= itr or iterations >= max_iterations:
            break

        clusters = new_clusters
        kmeans_model = new_kmeans_model
        print("Iteration: {}".format(iterations))
        print("Iteration Time: {}".format(time.time() - start_time))
        start_time = time.time()

    print("Final PSO k-means centroids")
    pprint(kmeans_model.clusterCenters)
    print("Clusters")
    pprint(clusters)
    spark.stop()


if __name__ == '__main__':
    main()
