import click
import numpy as np
import math
import heapq

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


def compute_nearest_neighbors(data_items, bc_data_items, ns):
    zipped = data_items.zipWithIndex().map(lambda x: (x[1], x[0]))
    crossed = zipped.map(lambda x: (x, bc_data_items.value))
    flattened = crossed.flatMap(lambda p_points: ((p_points[0], x) for x in p_points[1]))
    filtered = flattened.filter(lambda x: x[0][0] != x[1][0])
    distances = filtered.map(lambda x1_x2: (x1_x2[0], x1_x2[1], euclidean_distance(x1_x2[0][1], x1_x2[1][1])))
    indexes_distances = distances.map(lambda x: (x[0][0], (x[2], x[1][0])))
    grouped = indexes_distances.groupByKey().map(lambda x: (x[0], list(x[1])))
    ns_neighbors = grouped.map(lambda point_dists: (point_dists[0], get_ns_neighbors(point_dists[1], ns)))
    ns_neighbors_indexes = ns_neighbors.map(lambda x: (x[0], [i[1] for i in x[1]]))

    return ns_neighbors_indexes


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
        data_items.zipWithIndex().map(lambda x: (x[1], x[0]))
        .collect()
    )
    n = data_items.count()
    # ns = compute_ns(n, no_clusters)
    ns = 2

    nearest_neighbors = compute_nearest_neighbors(data_items, bc_data_items, ns)
    bc_nearest_neighbors = spark.sparkContext.broadcast(nearest_neighbors.collect())
    print('Nearest neighbors')
    pprint(bc_nearest_neighbors.value)


if __name__ == '__main__':
    main()
