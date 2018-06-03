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
@click.option('-k', '--no-clusters', required=True, type=click.INT)
@click.option('-c', '--converge-dist', required=True, type=click.FLOAT)
def main(file, no_clusters, converge_dist):
    spark = SparkSession.builder.appName('PythonKMeans').getOrCreate()
    lines = spark.read.text(file).rdd.map(lambda r: r[0])
    data_items = lines.map(parse_vector).cache()

    centroids = data_items.takeSample(False, no_clusters)

    iterations = 0
    centroids_delta_dist = 1.0

    while centroids_delta_dist > converge_dist:
        closest = data_items.map(lambda p: (closest_point(p, centroids), (p, 1)))
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

