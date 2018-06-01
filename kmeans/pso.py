import click
import heapq

import math
import numpy as np
from pyspark.sql import SparkSession
from pyspark.mllib.clustering import KMeans, KMeansModel
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
        distances = [(i, euclidean_distance(point, x)) for i, x in enumerate(points) if tuple(x) != tuple(point)]
        distances = list(map(lambda x: (x[1], x[0]), distances))
        ns_neighbors = get_ns_neighbors(distances, ns)
        nearest_neighbors.append(
            (idx, list(map(lambda x: x[1], ns_neighbors)))
        )

    return nearest_neighbors


def compute_squared_sigma(points, clusters):
    centroids = clusters.clusterCenters
    sum_term = sum([euclidean_distance(point, centroids[clusters.predict(point)]) ** 2
                    for point in points])
    return sum_term / len(points)


def compute_ns(n, k):
    return max(int(0.1 * (n / k)), 10)


def compute_personal_best(nearest_neighbors_indexes, positions):
    sum_term = np.sum([positions[idx]
                       for idx in nearest_neighbors_indexes], axis=0)
    return sum_term / len(nearest_neighbors_indexes)


def compute_global_best(point, clusters):
    centroids = clusters.clusterCenters
    return centroids[clusters.predict(point)]


def compute_position_velocity(position, velocity, personal_best, global_best, squared_sigma):
    dist = euclidean_distance(position, global_best)
    weight = 1 if dist < 0.125 * math.sqrt(squared_sigma) else 0

    new_velocity = np.array(velocity) + \
                   (np.array(personal_best) - np.array(position)) + \
                   (np.array(global_best) - np.array(position)) * weight
    new_position = np.array(position) + new_velocity

    return list(new_position), list(new_velocity)


def stop_condition(centroids, new_centroids, k, iterations, max_iterations):
    if iterations >= max_iterations:
        return True

    for i in range(k):
        if set(centroids[i]) != set(new_centroids[i]):
            return False

    return True

@click.command()
@click.option('-f', '--file', required=True)
@click.option('-k', '--no-clusters', required=True, type=click.INT)
@click.option('-p', '--no-particles', required=True, type=click.INT)
@click.option('-i', '--max-iterations', required=True, type=click.INT)
def main(file, no_clusters, no_particles, max_iterations):
    spark = SparkSession.builder.appName("PythonKMeans").getOrCreate()
    lines = spark.read.text(file).rdd.map(lambda r: r[0])
    data = lines.map(parse_vector).cache()
    points = data.collect()
    n = len(points)
    # ns = compute_ns(n, no_clusters)
    ns = 2

    nearest_neighbors = compute_nearest_neighbors(points, ns)
    pprint(nearest_neighbors)
    clusters = KMeans.train(data, no_clusters, maxIterations=max_iterations, initializationMode="random")
    pprint(clusters.clusterCenters)

    squared_sigma = compute_squared_sigma(points, clusters)
    print(squared_sigma)

    positions = points[:]
    velocities = [0.0 for _ in range(n)]
    personal_bests = [None for _ in range(n)]
    global_bests = [None for _ in range(n)]

    iterations = 0
    while True:
        for idx, point in enumerate(points):
            personal_bests[idx] = compute_personal_best(nearest_neighbors[idx][1], positions)
            global_bests[idx] = compute_global_best(point, clusters)
            positions[idx], velocities[idx] = compute_position_velocity(
                positions[idx], velocities[idx],
                personal_bests[idx], global_bests[idx], squared_sigma
            )

        print("\nPositions:")
        pprint(positions)
        print("\nVelocities:")
        pprint(velocities)
        print("\nPersonal bests:")
        pprint(personal_bests)
        print("\nGlobal bests:")
        pprint(global_bests)

        kmeans_model = KMeansModel(clusters.clusterCenters)
        new_clusters = KMeans.train(spark.sparkContext.parallelize(positions),
                                    no_clusters, maxIterations=1, initialModel=kmeans_model)
        print("\nNew centroids:")
        pprint(new_clusters.clusterCenters)

        iterations += 1
        print("=" * 70, " FINISHED ITERATION {}".format(iterations))
        if stop_condition(clusters.clusterCenters, new_clusters.clusterCenters, no_clusters, iterations, max_iterations):
            break

        clusters = new_clusters

    print("\n\nFinal centroids:")
    pprint(clusters.clusterCenters)


if __name__ == '__main__':
    main()