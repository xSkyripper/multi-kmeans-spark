import click
import numpy as np
import time
from pprint import pprint
from pyspark import StorageLevel
from pyspark.sql import SparkSession
from kmeans.utils import plot_clusters


def parse_vector(line, delim):
    return np.array([float(x) for x in line.rstrip().split(delim)])


def initialize_membership_instance(n, number_of_clusters):
    membership_matrix = np.random.dirichlet(np.ones(number_of_clusters), size=n)
    return membership_matrix


def compute_centroid(partial_data, m, dim):
    upper_sum = np.zeros(shape=(1, dim))
    lower_sum = np.zeros(shape=(1, dim))

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
            # print("{} ======= {}".format(d_i_j, d_k_j))
            result += (d_i_j / d_k_j) ** (2 / m - 1)

        results.append(1 / result)

    return np.array(results)


def compute(d1, d2):
    if type(d1) == np.float64 and type(d2) == np.float64:
        result = (d1, d2)
    elif type(d1) != type(d2):
        result = d1 + (float(d2),)
    else:
        result = d1 + d2

    return result


NUM_PARTITIONS = 4


def fuzzy(input_file, delimiter, no_clusters, convergence_distance, fuzziness_level, max_iterations, plot):
    start_time = time.time()
    spark = SparkSession.builder.appName("KMeans - Fuzzy").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    lines = spark.read.text(input_file).rdd.map(lambda line: line[0]).persist()
    data_items = lines \
        .map(lambda x: parse_vector(x, delimiter)) \
        .persist()
    n = data_items.count()
    dimensions = len(data_items.first())

    data_items = data_items \
        .zipWithIndex() \
        .map(lambda p: (p[1], p[0])) \
        .persist(StorageLevel.MEMORY_AND_DISK)

    max_iterations = max_iterations or np.inf

    membership_matrix = initialize_membership_instance(n, no_clusters)
    membership_matrix = spark \
        .sparkContext \
        .parallelize(membership_matrix) \
        .zipWithIndex() \
        .persist()
    # print("\nInitial membership matrix:")
    # pprint(membership_matrix.collect())

    # print("\nWith index points:")
    # pprint(data.collect())

    previous_membership_matrix = membership_matrix.map(lambda x: (x[1], x[0]))
    # print("\nWith index membership matrix:")
    # pprint(membership_matrix.collect())

    iterations = 0
    while True:

        membership_matrix = membership_matrix.flatMap(lambda row: [(row[1], (u, k)) for k, u in enumerate(row[0])])
        # print("\nFlattened membership matrix:")
        # pprint(membership_matrix.collect())

        joined = data_items.join(membership_matrix, numPartitions=NUM_PARTITIONS)
        # print("\nJoined points - membership matrix:")
        # pprint(joined.collect())

        mapped = joined.map(lambda r: (r[1][1][1], (r[1][0], r[1][1][0])))
        # print("\nRemapped join:")
        # pprint(mapped.collect())

        grouped = mapped.groupByKey(numPartitions=NUM_PARTITIONS).map(lambda r: (r[0], list(r[1])))
        # print("\nGrouped:")
        # pprint(grouped.collect())

        centroids_data = grouped.map(lambda r: (compute_centroid(r[1], fuzziness_level, dimensions)))
        # print("\nCentroids matrix:")
        # pprint(centroids_data.collect())

        cross_data_centroids = data_items.cartesian(centroids_data)
        # print("Cartesian data - centroids")
        # pprint(cross_data_centroids.collect())

        distances = cross_data_centroids \
            .map(lambda i_p_c: (i_p_c[0][0], (np.linalg.norm(i_p_c[0][1] - i_p_c[1])))) \
            .reduceByKey(compute, numPartitions=NUM_PARTITIONS)

        new_membership = distances \
            .mapValues(lambda value: generate_computes(value)) \
            .mapValues(lambda value: compute_membership(value, fuzziness_level))

        #
        # print("Old Membership Matrix")
        # pprint(previous_membership_matrix.collect())

        previous_current_membership = new_membership \
            .join(previous_membership_matrix, numPartitions=NUM_PARTITIONS)

        # print("Joined membership")
        # pprint(previous_current_membership.collect())

        previous_current_difference_membership = previous_current_membership \
            .mapValues(lambda value: np.linalg.norm(value[0] - value[1], ord=1))
        # print("Difference Membership")
        # pprint(previous_current_difference_membership.collect())

        max_difference = previous_current_difference_membership.max(lambda x: x[1])
        print("Max difference")
        print(max_difference)
        if max_difference[1] < convergence_distance or iterations >= max_iterations:
            break

        previous_membership_matrix = new_membership
        membership_matrix = new_membership \
            .map(lambda x: (x[1], x[0])) \
            .persist(StorageLevel.MEMORY_AND_DISK)
        iterations += 1

        print("Finished iteration: {}".format(iterations))
        print("Iteration Time: {}".format(time.time() - start_time))
        start_time = time.time()

    print("Finished iteration: {}".format(iterations))
    print("Iteration Time: {}".format(time.time() - start_time))

    def plot_fuzzy(data_items, centroids, membership_matrix, k):
        print('Data items indexed')
        data_items_indexed = data_items \
            .collect()
        pprint(data_items_indexed)

        print("Membership matrix")
        clusters_indexed = membership_matrix \
            .groupByKey() \
            .map(lambda x: (max(list(x[1]))[1], x[0])) \
            .groupByKey() \
            .map(lambda x: (x[0], list(x[1]))) \
            .collect()
        pprint(clusters_indexed)

        print("Centroids indexed")
        centroids_indexed = centroids \
            .zipWithIndex() \
            .map(lambda x: (x[1], x[0])) \
            .collect()
        pprint(centroids_indexed)

        plot_clusters(data_items_indexed, centroids_indexed, clusters_indexed,
                      'Fuzzy')

    if plot:
        plot_fuzzy(data_items, centroids_data, membership_matrix, no_clusters)

    spark.stop()


@click.command()
@click.option('-f', '--input-file', required=True)
@click.option('-d', '--delimiter', default=' ')
@click.option('-k', '--no-clusters', required=True, type=click.INT)
@click.option('-c', '--convergence-distance', required=True, type=click.FLOAT)
@click.option('-m', '--fuzziness-level', required=True, type=click.FLOAT)
@click.option('-i', '--max-iterations', type=click.INT)
@click.option('--plot', is_flag=True)
def main(input_file, delimiter, no_clusters, convergence_distance, fuzziness_level, max_iterations, plot):
    fuzzy(input_file, delimiter, no_clusters, convergence_distance, fuzziness_level, max_iterations, plot)


if __name__ == '__main__':
    main()
