from pprint import pprint

import click
import numpy as np
from pyspark.sql import SparkSession


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


def compute_centroid(partial_data, m, dim):
    upper_sum = np.zeros(shape=(1, dim))
    lower_sum = np.zeros(shape=(1, dim))
    m = float(m)

    for tpl in partial_data:
        point = tpl[0]
        u = tpl[1]
        upper_sum += (u ** m) * point
        lower_sum += (u ** m)

    return upper_sum / lower_sum


def generate_computes(distances):
    results = []
    for distance in distances:
        results.append((distance, distances))

    return results


def compute_membership(line, m):
    results = []
    for cell in line:
        result = 0
        d_i_j = cell[0]
        for d_k_j in cell[1]:
            result += (d_i_j / d_k_j) ** (2 / m - 1)

        results.append(1 / result)

    return np.array(results)


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
    fuzziness_level = float(fuzziness_level)

    # centroids = data.takeSample(False, number_of_clusters)
    # centroids_delta_distance = 1.0
    #
    number_of_points = len(data.collect())
    dimensions = len(data.collect()[0])
    membership_matrix = initialize_membership_instance(number_of_points, number_of_clusters)

    membership_matrix = spark.sparkContext.parallelize(membership_matrix)
    # print("\nInitial membership matrix:")
    # pprint(membership_matrix.collect())

    data = data.zipWithIndex().map(lambda p: (p[1], p[0]))
    # print("\nWith index points:")
    # pprint(data.collect())

    membership_matrix = membership_matrix.zipWithIndex()
    previous_membership_matrix = membership_matrix.map(lambda x: (x[1], x[0]))
    # print("\nWith index membership matrix:")
    # pprint(membership_matrix.collect())

    # membership_matrix = membership_matrix.flatMap(lambda row: [(row[1], (u, k)) for k, u in enumerate(row[0])])
    # print("\nFlattened membership matrix:")
    # pprint(membership_matrix.collect())
    #
    # data = data.zipWithIndex().map(lambda p: (p[1], p[0]))
    # print("\nWith index points:")
    # pprint(data.collect())
    #
    # joined = data.join(membership_matrix)
    # print("\nJoined points - membership matrix:")
    # pprint(joined.collect())
    #
    # mapped = joined.map(lambda r: (r[1][1][1], (r[1][0], r[1][1][0])))
    # print("\nRemapped join:")
    # pprint(mapped.collect())
    #
    # grouped = mapped.groupByKey().map(lambda r: (r[0], list(r[1])))
    # print("\nGrouped:")
    # pprint(grouped.collect())
    #
    # centroids_data = grouped.map(lambda r: (compute_centroid(r[1], fuzziness_level, dimensions)))
    # print("\nCentroids matrix:")
    # pprint(centroids_data.collect())

    iterations = 0
    while iterations < 10:

        membership_matrix = membership_matrix.flatMap(lambda row: [(row[1], (u, k)) for k, u in enumerate(row[0])])
        # print("\nFlattened membership matrix:")
        # pprint(membership_matrix.collect())

        joined = data.join(membership_matrix)
        # print("\nJoined points - membership matrix:")
        # pprint(joined.collect())

        mapped = joined.map(lambda r: (r[1][1][1], (r[1][0], r[1][1][0])))
        # print("\nRemapped join:")
        # pprint(mapped.collect())

        grouped = mapped.groupByKey().map(lambda r: (r[0], list(r[1])))
        # print("\nGrouped:")
        # pprint(grouped.collect())

        centroids_data = grouped.map(lambda r: (compute_centroid(r[1], fuzziness_level, dimensions)))
        # print("\nCentroids matrix:")
        # pprint(centroids_data.collect())

        cross_data_centroids = data.cartesian(centroids_data)
        # print("Cartesian data - centroids")
        # pprint(cross_data_centroids.collect())

        distances = cross_data_centroids \
            .map(lambda i_p_c: (i_p_c[0][0], (np.linalg.norm(i_p_c[0][1] - i_p_c[1])))) \
            .reduceByKey(lambda d1, d2: (d1, d2))
        # print("Distances")
        # pprint(distances.collect())

        new_membership = distances \
            .mapValues(lambda value: generate_computes(value)) \
            .mapValues(lambda value: compute_membership(value, fuzziness_level))

        # print("New membership computes")
        # pprint(new_membership.collect())
        #
        # print("Old Membership Matrix")
        # pprint(previous_membership_matrix.collect())

        previous_current_membership = new_membership \
            .join(previous_membership_matrix)

        # print("Joined membership")
        # pprint(previous_current_membership.collect())

        previous_current_difference_membership = previous_current_membership \
            .mapValues(lambda value: np.linalg.norm(value[0] - value[1], ord=1))
        # print("Difference Membership")
        # pprint(previous_current_difference_membership.collect())

        max_difference = previous_current_difference_membership.max(lambda x: x[1])
        print("Max difference")
        print(max_difference)
        if max_difference[1] > convergence_distance:
            break

        previous_membership_matrix = membership_matrix
        membership_matrix = new_membership.map(lambda x: (x[1], x[0]))
        iterations += 1

    spark.stop()


if __name__ == '__main__':
    main()
