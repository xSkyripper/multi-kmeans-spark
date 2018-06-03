import click
import numpy as np
import math
import heapq

from sklearn.neighbors import NearestNeighbors
from pyspark.mllib.clustering import KMeans
from pyspark.sql import SparkSession
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


def compute_nearest_neighbors(bc_data_items, nrbs, ns):
    bc_data_items = bc_data_items.value
    ns_neighbour = map(lambda x: (x[0], nrbs.kneighbors(X=[x[1]], return_distance=False)[0][1:]), bc_data_items)
    return list(ns_neighbour)


def compute_squared_sigma(data_items, n, kmeans_model):
    centroids = kmeans_model.clusterCenters
    squared_distances = data_items.map(
        lambda x: (euclidean_distance(x, centroids[kmeans_model.predict(x)]) ** 2))
    sum_term = squared_distances.sum()
    return sum_term / n


@click.command()
@click.option('-f', '--file', required=True)
@click.option('-k', '--no-clusters', required=True, type=click.INT)
@click.option('-i', '--max-iterations', required=True, type=click.INT)
@click.option('--itr', required=True, type=click.INT)
def main(file, no_clusters, max_iterations, itr):
    spark = SparkSession.builder.appName("PythonKMeans").getOrCreate()
    lines = spark.read.text(file).rdd.map(lambda r: r[0])
    data_items = lines.map(parse_vector).cache()
    bc_data_items = spark.sparkContext.broadcast(
        data_items
            .zipWithIndex()
            .map(lambda x: (x[1], x[0]))
            .collect()
    )
    n = data_items.count()
    # ns = compute_ns(n, no_clusters)
    ns = 2
    nrbs = NearestNeighbors(n_neighbors=ns + 1, algorithm="kd_tree")
    nrbs.fit(data_items.collect())
    spark.sparkContext.broadcast(nrbs)

    nearest_neighbors = compute_nearest_neighbors(bc_data_items, nrbs, ns)
    bc_nearest_neighbors = spark.sparkContext.broadcast(nearest_neighbors)
    print('Nearest neighbors')
    with open("out.txt", mode="wt") as file_object:
        for row in bc_nearest_neighbors.value:
            file_object.write("{}\n".format(str(row)))

    kmeans_model = KMeans.train(data_items, no_clusters,
                                maxIterations=max_iterations, initializationMode='random')

    squared_sigma = compute_squared_sigma(data_items, n, kmeans_model)
    print("Squared sigma: {}".format(squared_sigma))



if __name__ == '__main__':
    main()
