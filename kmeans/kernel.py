import numpy as np
import numpy.random
import random
from pyspark.sql import SparkSession
from pprint import pprint
import click
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm


def parse_vector(line):
    return np.array([float(x) for x in line.split(' ')])


def init_random_clusters(data, k):
    clusters = data.zipWithIndex()
    clusters = clusters.map(lambda p_i: (np.random.randint(k), p_i[1]))
    return clusters


def init_kernel_matrix(data, sigma):
    cross_term1 = data.zipWithIndex().map(lambda r: (r[1], r[0]))
    cross_term2 = data.zipWithIndex().map(lambda r: (r[1], r[0]))

    crossed = cross_term1.cartesian(cross_term2)
    crossed = crossed.sortBy(lambda r: (r[0][0], r[1][0]))
    crossed = crossed.map(lambda r: (r[0][1], r[1][1]))

    kernel_values = crossed.map(lambda pi_pj: ker_gaussian(pi_pj[0], pi_pj[1], sigma))
    return kernel_values


def ker_gaussian(var_1, var_2, sigma):
    upper_term = - np.linalg.norm(var_1 - var_2, ord=2) ** 2
    lower_term = 2 * sigma ** 2
    return np.exp(upper_term / lower_term)


def compute_term_3(cluster_indexes, kernel_matrix):
    upper_term = 0.0
    lower_term = len(cluster_indexes) ** 2
    for j in cluster_indexes:
        for l in cluster_indexes:
            upper_term += kernel_matrix[j][l]
    return upper_term / lower_term


def compute_terms_2(cluster_indexes, kernel_matrix, n):
    terms_2 = []
    lower_term = len(cluster_indexes)

    for i in range(n):
        upper_term = 0.0
        for j in cluster_indexes:
            upper_term += kernel_matrix[i][j]
        term_2 = 2 * upper_term / lower_term
        terms_2.append((i, term_2))

    return terms_2


def compute_distances_point_cluster(cluster_index, group):
    # group has form (cluster_term_3, cluster_points_terms_2)
    # where cluster_points_terms_2 elem has form (point_index, term_2)
    distances_with_clusters = []
    cluster_term_3 = group[0]
    cluster_points_terms_2 = group[1]

    for point_index, term_2 in cluster_points_terms_2:
        distance = 1 - term_2 + cluster_term_3
        distances_with_clusters.append(
            (point_index, (cluster_index, distance))
        )

    return distances_with_clusters


def stop_condition(clusters, new_clusters, iteration, max_iterations, k):
    clusters_data = clusters.collect()
    new_clusters_data = new_clusters.collect()

    # pprint(clusters_data)
    # pprint(new_clusters_data)
    sorted_clusters = [[] for i in range(k)]
    sorted_new_clusters = [[] for i in range(k)]

    for group in clusters_data:
        sorted_clusters[group[0]].extend(group[1])

    for group in new_clusters_data:
        sorted_new_clusters[group[0]].extend(group[1])

    converged = True
    for cluster_index in range(k):
        cluster = sorted_clusters[cluster_index]
        new_cluster = sorted_new_clusters[cluster_index]

        if len(cluster) != len(new_cluster):
            converged = False
            break
        else:
            converged_cluster = set(cluster) == set(new_cluster)

        converged = converged and converged_cluster
        if not converged:
            break

    # print("Converged? ", converged)

    return converged or max_iterations <= iteration


def compute_centroids(clusters, points):
    flattened_clusters = clusters.flatMapValues(lambda x: x).map(lambda r: (r[1], r[0]))
    points_with_index = points.zipWithIndex().map(lambda r: (r[1], r[0]))
    joined = flattened_clusters.join(points_with_index).map(lambda r: (r[1][0], (r[1][1], 1)))

    sums = (
        joined
        .reduceByKey(lambda p1_o1, p2_o2: (p1_o1[0] + p2_o2[0], p1_o1[1] + p2_o2[1]))
    )
    centroids = sums.map(lambda r: (r[0], (r[1][0][0] / r[1][1], r[1][0][1] / r[1][1])))

    return centroids, joined


def plot_clusters(maps, centroids, n, k):
    clusters_data = maps.collect()
    centroids_data = centroids.collect()

    final_centroids = [[] for i in range(k)]
    for centroid_index, centroid in centroids_data:
        final_centroids[centroid_index].extend(centroid)

    final_clusters = [[] for i in range(k)]
    for centroid_index, point_ in clusters_data:
        final_clusters[centroid_index].append(point_[0])

    # pprint(final_centroids)
    # pprint(final_clusters)

    colors_iter = iter(cm.rainbow(np.linspace(0, 1, k)))
    plt.figure("Kernel K-Means: Final clusters")
    colors = cm.rainbow(np.linspace(0, 1, k * 2))
    colors = random.sample(list(colors), k)
    for cluster_index in range(k):
        cluster_color = colors[cluster_index]
        cluster_matrix = np.asmatrix(final_clusters[cluster_index])
        centroids_matrix = np.asmatrix(final_centroids[cluster_index])

        plt.scatter(
            x=np.ravel(cluster_matrix[:, 0]),
            y=np.ravel(cluster_matrix[:, 1]),
            marker='.',
            s=100,
            c=cluster_color
        )

        plt.scatter(
            x=np.ravel(centroids_matrix[:, 0]),
            y=np.ravel(centroids_matrix[:, 1]),
            marker='*',
            s=400,
            c=cluster_color,
            edgecolors="black"
        )
    plt.show()


@click.command()
@click.option('-f', '--file', required=True)
@click.option('-k', '--no-clusters', required=True)
@click.option('-i', '--max-iterations', required=True)
def main(file, no_clusters, max_iterations):
    spark = SparkSession.builder.appName('PythonKMeans').getOrCreate()
    lines = spark.read.text(file).rdd.map(lambda r: r[0])
    data = lines.map(parse_vector).cache()
    k = int(no_clusters)
    n = len(data.collect())
    max_iterations = int(max_iterations)

    kernel_values = init_kernel_matrix(data, 4)
    kernel_matrix = np.array(kernel_values.collect()).reshape(n, n)
    bc_kernel_matrix = spark.sparkContext.broadcast(kernel_matrix)
    # print("\nKernel matrix")
    # pprint(kernel_matrix)

    clusters = init_random_clusters(data, k)
    # print("\nInit random Clusters")
    # pprint(clusters.collect())

    # grouped_by_cluster = clusters.groupByKey().map(lambda c_pts: (c_pts[0], list(c_pts[1])))
    # print("\nGrouped by cluster")
    # pprint(grouped_by_cluster.collect())
    #
    # term3_by_cluster = grouped_by_cluster.map(lambda c_pts: (c_pts[0], compute_term_3(c_pts[1], bc_kernel_matrix.value)))
    # print("\nTerm 3 by cluster")
    # pprint(term3_by_cluster.collect())
    #
    # terms2_by_cluster = grouped_by_cluster.map(lambda c_pts: (c_pts[0], compute_terms_2(c_pts[1], bc_kernel_matrix.value, n)))
    # print("\nTerms 2 by cluster - all points")
    # pprint(terms2_by_cluster.collect())
    #
    # joined = term3_by_cluster.join(terms2_by_cluster)
    # print("\nJoined terms2 - terms 3")
    # pprint(joined.collect())
    #
    # distances = joined.flatMap(lambda c_g: compute_distances_point_cluster(c_g[0], c_g[1]))
    # print("\nDistances")
    # pprint(distances.collect())
    #
    # new_assigns = distances.reduceByKey(lambda c1_d1, c2_d2: min(c1_d1, c2_d2, key=lambda c_d: c_d[1]))
    # print("\nExperiment")
    # pprint(new_assigns.collect())

    clusters = clusters.groupByKey().map(lambda c_pts: (c_pts[0], list(c_pts[1])))
    # print("\nGrouped by cluster")
    # pprint(clusters.collect())

    iteration = 0
    while True:
        print("Iteration {} ...".format(iteration))
        term3_by_cluster = clusters.map(
            lambda c_pts: (c_pts[0], compute_term_3(c_pts[1], bc_kernel_matrix.value)))
        # print("\nTerm 3 by cluster")
        # pprint(term3_by_cluster.collect())

        terms2_by_cluster = clusters.map(
            lambda c_pts: (c_pts[0], compute_terms_2(c_pts[1], bc_kernel_matrix.value, n)))
        # print("\nTerms 2 by cluster - all points")
        # pprint(terms2_by_cluster.collect())

        joined = term3_by_cluster.join(terms2_by_cluster)
        # print("\nJoined terms2 - terms 3")
        # pprint(joined.collect())

        distances = joined.flatMap(lambda c_g: compute_distances_point_cluster(c_g[0], c_g[1]))
        # print("\nDistances")
        # pprint(distances.collect())

        new_assigns = distances.reduceByKey(lambda c1_d1, c2_d2: min(c1_d1, c2_d2, key=lambda c_d: c_d[1])).collect()
        # print('\nNew assigns')
        # pprint(new_assigns)

        new_clusters = [[i, []] for i in range(k)]
        for point_index, cluster_distance in new_assigns:
            new_clusters[cluster_distance[0]][1].append(point_index)

        # print('\nNew clusters')
        new_clusters = spark.sparkContext.parallelize(new_clusters)
        # pprint(new_clusters.collect())

        iteration += 1
        # print("\n\n==========================================================")
        if stop_condition(clusters, new_clusters, iteration, max_iterations, k):
            break

        clusters = new_clusters

    centroids, maps = compute_centroids(clusters, data)
    # pprint(centroids.collect())
    # pprint(maps.collect())
    plot_clusters(maps, centroids, n, k)




if __name__ == '__main__':
    main()
