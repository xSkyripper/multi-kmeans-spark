import click
import heapq

import math
import numpy as np
from pyspark.sql import SparkSession
from pyspark.mllib.clustering import KMeans
from pprint import pprint


def parse_vector(line):
    return np.array([float(x) for x in line.split(' ')])


def euclidean_distance(x1, x2):
    return np.linalg.norm(x1 - x2)


def get_ns_neighbors(distances, ns):
    max_heap = distances[:ns]
    heapq._heapify_max(max_heap)
    for i in range(ns, len(distances)):
        number = distances[i]
        if number < max_heap[0]:
            max_heap[0] = number
            heapq._heapify_max(max_heap)
    return sorted(max_heap)


def compute_nearest_neighbors(points, ns):
    nearest_neighbors = []
    for idx, point in enumerate(points):
        distances = [(i, euclidean_distance(point, x))
                     for i, x in enumerate(points) if tuple(x) != tuple(point)]
        distances = list(map(lambda x: (x[1], x[0]), distances))
        ns_neighbors = get_ns_neighbors(distances, ns)
        nearest_neighbors.append((idx, list(map(lambda x: x[1], ns_neighbors))))

    return nearest_neighbors


def compute_squared_sigma(points, kmeans_model):
    centroids = kmeans_model.clusterCenters
    sum_term = sum([euclidean_distance(point, centroids[kmeans_model.predict(point)]) ** 2
                    for point in points])
    return sum_term / len(points)


def compute_ns(n, k):
    return max(int(0.1 * (n / k)), 10)


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


def compute_clusters(points, kmeans_model):
    k = kmeans_model.k
    clusters = [[] for _ in range(k)]

    for point_idx, point in enumerate(points):
        cluster_idx = kmeans_model.predict(point)
        clusters[cluster_idx].append(point_idx)

    return clusters


@click.command()
@click.option('-f', '--file', required=True)
@click.option('-k', '--no-clusters', required=True, type=click.INT)
@click.option('-i', '--max-iterations', required=True, type=click.INT)
@click.option('--itr', required=True, type=click.INT)
def main(file, no_clusters, max_iterations, itr):
    spark = SparkSession.builder.appName("PythonKMeans").getOrCreate()
    lines = spark.read.text(file).rdd.map(lambda r: r[0])
    data = lines.map(parse_vector).cache()
    points = data.collect()
    n = len(points)
    # ns = compute_ns(n, no_clusters)
    ns = 2

    nearest_neighbors = compute_nearest_neighbors(points, ns)
    print('Nearest neighbors')
    pprint(nearest_neighbors)

    kmeans_model = KMeans.train(data, no_clusters, maxIterations=max_iterations, initializationMode="random")
    initial_centroids = kmeans_model.clusterCenters[:]
    print('Initial centroids')
    pprint(initial_centroids)

    squared_sigma = compute_squared_sigma(points, kmeans_model)
    print('Squared sigma')
    print(squared_sigma)

    positions = points[:]
    velocities = [0.0 for _ in range(n)]
    personal_bests = [None for _ in range(n)]
    global_bests = [None for _ in range(n)]

    iterations = 0
    convergence_iterations = 0
    clusters = compute_clusters(positions, kmeans_model)
    initial_clusters = clusters[:]

    while True:
        for idx, point in enumerate(points):
            personal_bests[idx] = compute_personal_best(nearest_neighbors[idx][1], positions)
            global_bests[idx] = compute_global_best(point, kmeans_model)
            positions[idx], velocities[idx] = compute_position_velocity(
                positions[idx], velocities[idx],
                personal_bests[idx], global_bests[idx], squared_sigma
            )

        # print("\nPositions:")
        # pprint(positions)
        # print("\nVelocities:")
        # pprint(velocities)
        # print("\nPersonal bests:")
        # pprint(personal_bests)
        # print("\nGlobal bests:")
        # pprint(global_bests)

        new_kmeans_model = KMeans.train(spark.sparkContext.parallelize(positions),
                                        no_clusters, maxIterations=1, initialModel=kmeans_model)
        new_clusters = compute_clusters(positions, new_kmeans_model)

        convergence_iterations = stop_condition(no_clusters, clusters, new_clusters, convergence_iterations)

        iterations += 1
        print("=" * 70, "Finished iteration {}".format(iterations))
        print('\nOld clusters')
        pprint(clusters)
        print('\nNew clusters')
        pprint(new_clusters)
        print('\nClusters identical for {} convergence iterations'.format(convergence_iterations))
        if convergence_iterations >= itr or iterations >= max_iterations:
            break

        kmeans_model = new_kmeans_model
        clusters = new_clusters[:]

    print("\n\n======= Finished in {} iterations =======".format(iterations))
    print('\nInitial k-means centroids:')
    pprint(initial_centroids)
    print("\nInitial k-means clusters:")
    pprint(initial_clusters)
    print('-' * 70)
    print("\nFinal PSO k-means centroids")
    pprint(kmeans_model.clusterCenters)
    print("\nFinal PSO k-means clusters:")
    pprint(clusters)


if __name__ == '__main__':
    main()
