import click
import numpy as np
from pprint import pprint
from pyspark.sql import SparkSession
from kmeans.utils import plot_clusters


def parse_vector(line, delimiter):
    return np.array([float(x) for x in line.split(delimiter)])


def euclidean_distance(x1, x2):
    return np.linalg.norm(x1 - x2)


def closest_point(point, centroids, k):
    best_index = 0
    smallest_dist = np.inf

    for idx in range(k):
        temp_dist = euclidean_distance(point, centroids[idx])
        if temp_dist < smallest_dist:
            smallest_dist = temp_dist
            best_index = idx

    return best_index


def default(input_file, delimiter, k, convergence_dist, max_iterations, plot):
    spark = SparkSession.builder.appName('KMeans - Default PySpark-less').getOrCreate()
    lines = spark.read.text(input_file).rdd.map(lambda r: r[0])
    data_items = lines.map(lambda l: parse_vector(l, delimiter)).cache()
    max_iterations = max_iterations or np.inf

    centroids = data_items.takeSample(False, k)
    iterations = 0
    centroids_delta_dist = 1.0

    closest = None
    while centroids_delta_dist > convergence_dist and iterations < max_iterations:
        closest = data_items.map(lambda p: (closest_point(p, centroids, k), (p, 1)))
        point_stats = closest.reduceByKey(lambda p1_c1, p2_c2: (p1_c1[0] + p2_c2[0], p1_c1[1] + p2_c2[1]))
        new_points = point_stats.map(lambda st: (st[0], st[1][0] / st[1][1])).collect()

        centroids_delta_dist = sum(np.sum((centroids[iK] - p) ** 2) for (iK, p) in new_points)

        for (iK, p) in new_points:
            centroids[iK] = p

        iterations += 1
        print("Finished iteration {}".format(iterations))

    print("Final centroids in {} iterations:".format(iterations))
    pprint(centroids)

    def plot_default(data_items, centroids, clusters, k):
        # preparing data for plotting
        data_items_indexed = data_items\
            .zipWithIndex()\
            .map(lambda x: (x[1], x[0]))\
            .collect()

        centroids_indexed = list(zip([i for i in range(k)], centroids))

        clusters_indexed = clusters\
            .zipWithIndex()\
            .map(lambda x: (x[0][0], x[1]))\
            .groupByKey()\
            .map(lambda x: (x[0], list(x[1])))\
            .collect()

        plot_clusters(data_items_indexed, centroids_indexed, clusters_indexed,
                      'Default PySparkless')

    if plot:
        plot_default(data_items, centroids, closest, k)

    spark.stop()


@click.command()
@click.option('-f', '--input-file', required=True)
@click.option('-d', '--delimiter', default=' ')
@click.option('-k', '--no-clusters', required=True, type=click.INT)
@click.option('-c', '--convergence-dist', required=True, type=click.FLOAT)
@click.option('-i', '--max-iterations', type=click.INT)
@click.option('--plot', is_flag=True)
def main(input_file, delimiter, no_clusters, convergence_dist, max_iterations, plot):
    default(input_file, delimiter, no_clusters, convergence_dist, max_iterations, plot)


if __name__ == '__main__':
    main()
