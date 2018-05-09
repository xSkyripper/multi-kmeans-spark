from __future__ import print_function

import click
import numpy as np
from pprint import pprint
from pyspark.sql import SparkSession


def parse_vector(line):
    return np.array([float(x) for x in line.split(' ')])


def closest_point(point, centroids):
    best_index = 0
    closest = float('+inf')

    for i in range(0, len(centroids)):
        temp_dist = np.sum((point - centroids[i]) ** 2)
        if temp_dist < closest:
            closest = temp_dist
            best_index = i

    return best_index


@click.command()
@click.option('-f', '--file', required=True)
@click.option('-k', '--no-clusters', required=True)
@click.option('-c', '--converge-dist', required=True)
def main(file, no_clusters, converge_dist):
    spark = SparkSession.builder.appName('PythonKMeans').getOrCreate()
    lines = spark.read.text(file).rdd.map(lambda r: r[0])
    data = lines.map(parse_vector).cache()
    k = int(no_clusters)
    converge_dist = float(converge_dist)

    centroids = data.takeSample(False, k)
    print("initial centroids:")
    pprint(centroids)
    # centroids = np.array([(1.90, 0.97), (3.17, 4.96)])
    centroids_delta_dist = 1.0

    iterations = 0
    while centroids_delta_dist > converge_dist:
        closest = data.map(lambda p: (closest_point(p, centroids), (p, 1)))
        point_stats = closest.reduceByKey(lambda p1_c1, p2_c2: (p1_c1[0] + p2_c2[0], p1_c1[1] + p2_c2[1]))
        new_points = point_stats.map(lambda st: (st[0], st[1][0] / st[1][1])).collect()

        centroids_delta_dist = sum(np.sum((centroids[iK] - p) ** 2) for (iK, p) in new_points)

        for (iK, p) in new_points:
            centroids[iK] = p

        iterations += 1

    print("Final centroids in {} iterations:".format(iterations))
    pprint(centroids)
    spark.stop()


if __name__ == '__main__':
    main()

