import click
import numpy as np
from pprint import pprint
from pyspark.sql import SparkSession


def parse_vector(line):
    return np.array([float(x) for x in line.split(' ')])


def euclidean_distance(x1, x2):
    return np.linalg.norm(x1 - x2)


def closest_point(point, centroids, k):
    best_index = 0
    smallest_dist = np.inf

    for idx in range(k):
        temp_dist = euclidean_distance(point - centroids[idx])
        if temp_dist < smallest_dist:
            smallest_dist = temp_dist
            best_index = idx

    return best_index


def default_kmeans(input_file, k, convergence_dist):
    spark = SparkSession.builder.appName('PythonKMeans').getOrCreate()
    lines = spark.read.text(input_file).rdd.map(lambda r: r[0])
    data_items = lines.map(parse_vector).cache()
    centroids = data_items.takeSample(False, k)

    iterations = 0
    centroids_delta_dist = 1.0

    while centroids_delta_dist > convergence_dist:
        closest = data_items.map(lambda p: (closest_point(p, centroids, k), (p, 1)))
        point_stats = closest.reduceByKey(lambda p1_c1, p2_c2: (p1_c1[0] + p2_c2[0], p1_c1[1] + p2_c2[1]))
        new_points = point_stats.map(lambda st: (st[0], st[1][0] / st[1][1])).collect()

        centroids_delta_dist = sum(np.sum((centroids[iK] - p) ** 2) for (iK, p) in new_points)

        for (iK, p) in new_points:
            centroids[iK] = p

        iterations += 1

    print("Final centroids in {} iterations:".format(iterations))
    pprint(centroids)
    spark.stop()


@click.command()
@click.option('-f', '--file', required=True)
@click.option('-k', '--no-clusters', required=True, type=click.INT)
@click.option('-c', '--convergence-dist', required=True, type=click.FLOAT)
def main(file, no_clusters, convergence_dist):
    default_kmeans(file, no_clusters, convergence_dist)


if __name__ == '__main__':
    main()

