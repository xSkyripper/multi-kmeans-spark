import numpy as np, numpy.random
from pyspark.sql import SparkSession
from pprint import pprint
import click


# class FuzzyKMeans(object):
#     NAME = "FuzzyKMeans"
#
#     def __init__(self, file, number_of_clusters, convergence_distance, fuzziness_level):
#         self.file = file
#         self.number_of_cluster = number_of_clusters
#         self.convergence_distance = convergence_distance
#         self.fuzziness_level = fuzziness_level
#
#     def _parse_vector(self, line):
#         return np.array([float(x) for x in line.split(' ')])
#
#     def _initialize_membership_instance(self, n):
#         return np.random.random((n, self.number_of_cluster))
#
#
#     def run(self):
#         spark = SparkSession.builder.appName(self.NAME).getOrCreate()
#         lines = spark.read.text(self.file).rdd(lambda line: line[0])
#         data = lines.map(self._parse_vector).cache()
#
#         centroids = data.takeSample(False, self.number_of_cluster)
#         print("Initial centroids")
#         pprint(centroids)
#
#         centroids_delta_distance = 1.0
#         number_of_points = data.collect().__len__()
#         membership_matrix = self._initialize_membership_instance(number_of_points)
#
#         while centroids_delta_distance > self.convergence_distance:
#             vertical_merged_data = data.map(lambda point: (point, centroids))
#             vertical_merged_data = vertical_merged_data.groupByKey().map(lambda x: (x[0], list(x[1])))
#
#             new_centroids = []
#             for j in range(len(centroids)):
#                 new_centroids[j] = np.divide(
#                     np.sum([membership_matrix[i][j] ** self.fuzziness_level * data[i] for i in
#                             range(0, number_of_points)]),
#                     np.sum([membership_matrix[i][j] ** self.fuzziness_level])
#                 )


def parse_vector(line):
    return np.array([float(x) for x in line.split(' ')])


def initialize_membership_instance(n, number_of_clusters):
    membership_matrix = np.random.dirichlet(np.ones(number_of_clusters), size=n)
    return membership_matrix



@click.command()
@click.option('-f', '--file', required=True)
@click.option('-k', '--number-of-clusters', required=True)
@click.option('-c', '--convergence-distance', required=True)
@click.option('-m', '--fuzziness-level', required=True)
def main(file, number_of_clusters, convergence_distance, fuzziness_level):
    spark = SparkSession.builder.appName("Fuzzy").getOrCreate()
    lines = spark.read.text(file).rdd.map(lambda line: line[0])
    data = lines.map(parse_vector).cache()
    number_of_clusters = int(number_of_clusters)
    convergence_distance = float(convergence_distance)

    centroids = data.takeSample(False, number_of_clusters)

    centroids_delta_distance = 1.0
    number_of_points = len(data.collect())
    membership_matrix = initialize_membership_instance(number_of_points, number_of_clusters)
    membership_matrix = spark.sparkContext.parallelize(membership_matrix)
    pprint(membership_matrix.collect())

    membership_matrix = membership_matrix.zipWithIndex()
    membership_matrix = membership_matrix.flatMap(lambda row: [(row[1], (u, k)) for k, u in enumerate(row[0])])
    # pprint(membership_matrix.collect())

    data = data.zipWithIndex().map(lambda p: (p[1], p[0]))
    # pprint(data.collect())

    joined = data.join(membership_matrix)
    # pprint(joined.collect())

    mapped = joined.map(lambda r: (r[1][1][1], (r[1][0], r[1][1][0])))
    # pprint(mapped.collect())

    grouped = mapped.groupByKey().map(lambda r: (r[0], list(r[1])))
    pprint(grouped.collect())

    # while centroids_delta_distance > convergence_distance:
    #
    #     for j in range(number_of_clusters):
    #         vertical_merged_data = data.map(lambda point: (j, (point, np.stack(membership_matrix.T)[j])))
    #
    #     pprint(vertical_merged_data.collect())
    #     vertical_merged_data = vertical_merged_data.groupByKey().map(lambda x: (x[0], list(x[1])))
    #     pprint(vertical_merged_data.collect())
    #     break

    spark.stop()


if __name__ == '__main__':
    main()