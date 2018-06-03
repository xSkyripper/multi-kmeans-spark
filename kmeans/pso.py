import click
import numpy as np
import math

from sklearn.neighbors import NearestNeighbors
from pyspark.mllib.clustering import KMeans
from pyspark.sql import SparkSession
from pprint import pprint


def parse_vector(line):
    return np.array([float(x) for x in line.split(' ')])


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


@click.command()
@click.option('-f', '--file', required=True)
@click.option('-k', '--no-clusters', required=True, type=click.INT)
@click.option('-i', '--max-iterations', required=True, type=click.INT)
@click.option('--itr', required=True, type=click.INT)
@click.option('--ns', type=click.INT)
def main(file, no_clusters, max_iterations, ns, itr):
    spark = SparkSession.builder.appName("PythonKMeans").getOrCreate()
    lines = spark.read.text(file).rdd.map(lambda r: r[0])
    data_items = lines.map(parse_vector).cache()
    n = data_items.count()
    ns = ns or compute_ns(n, no_clusters)

    nearest_neighbors = compute_nearest_neighbors(spark.sparkContext, data_items, ns)
    kmeans_model = KMeans.train(data_items, no_clusters,
                                maxIterations=max_iterations, initializationMode='random')
    squared_sigma = compute_squared_sigma(data_items, n, kmeans_model)

    iterations = 0
    convergence_iterations = 0
    positions = data_items.collect()
    velocities = [0.0 for _ in range(n)]
    personal_bests = [None for _ in range(n)]
    global_bests = [None for _ in range(n)]
    clusters = compute_clusters(positions, kmeans_model)
    initial_clusters = clusters[:]
    initial_centroids = kmeans_model.clusterCenters[:]

    while True:
        for idx in range(no_clusters):
            personal_bests[idx] = compute_personal_best(nearest_neighbors[idx][1], positions)
            global_bests[idx] = compute_global_best(positions[idx], kmeans_model)
            positions[idx], velocities[idx] = compute_position_velocity(
                positions[idx], velocities[idx],
                personal_bests[idx], global_bests[idx], squared_sigma
            )

        new_kmeans_model = KMeans.train(spark.sparkContext.parallelize(positions),
                                        no_clusters, maxIterations=1, initialModel=kmeans_model)
        new_clusters = compute_clusters(positions, new_kmeans_model)
        convergence_iterations = stop_condition(no_clusters, clusters, new_clusters, convergence_iterations)
        iterations += 1
        # print("=" * 70, "Finished iteration {}".format(iterations))
        # print('\nOld clusters')
        # pprint(clusters)
        # print('\nNew clusters')
        # pprint(new_clusters)
        # print('\nClusters identical for {} convergence iterations'.format(convergence_iterations))
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
