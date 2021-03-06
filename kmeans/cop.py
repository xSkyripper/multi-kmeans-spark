import click
import numpy as np
import time
from pyspark.sql import SparkSession
from pprint import pprint
from kmeans.utils import plot_clusters

MUST_LINK = 1
CANNOT_LINK = -1

dimension = 0


def parse_vector(line, delimiter):
    return np.array([float(x) for x in line.split(delimiter)])


def parse_constraints(line):
    return np.array([int(x) for x in line.split(' ')])


def dfs(vertex, graph: dict, visited: list, trace: list):
    visited[vertex] = True
    for adjacent_vertex in graph[vertex]:
        if not visited[adjacent_vertex]:
            dfs(adjacent_vertex, graph, visited, trace)

    trace.append(vertex)


def add(graph: dict, i, j):
    graph[i].add(j)
    graph[j].add(i)


def transitive_closure(constraints: dict, n):
    must_link = constraints.get(MUST_LINK, {})
    cannot_link = constraints.get(CANNOT_LINK, {})

    must_link_graph = {i: set() for i in range(n)}
    cannot_link_graph = {i: set() for i in range(n)}

    for (a, b) in must_link:
        add(must_link_graph, a, b)

    visited = [False] * n
    for i in range(n):
        if not visited[i]:
            trace = []
            dfs(i, must_link_graph, visited, trace)
            for vertex in trace:
                for transitive_vertex in trace:
                    if vertex != transitive_vertex:
                        must_link_graph[vertex].add(transitive_vertex)

    for point1, point2 in cannot_link:
        add(cannot_link_graph, point1, point2)

        for should_link in must_link_graph[point2]:
            add(cannot_link_graph, point1, should_link)

        for should_link in must_link_graph[point1]:
            add(cannot_link_graph, should_link, point2)
            for should_link2 in must_link_graph[point2]:
                add(cannot_link_graph, should_link, should_link2)

    for point in must_link_graph:
        for should_link in must_link_graph[point]:
            if point != should_link and should_link in cannot_link_graph[point]:
                raise Exception("Inconsistent constraints between: {} - {}".format(point, should_link))

    return must_link_graph, cannot_link_graph


def distance_to_centroids(point, centroids):
    distances = dict()
    for i in range(len(centroids)):
        distance = np.sum((point[1] - centroids[i][1]) ** 2)
        distances[i] = distance

    return sorted(distances.items(), key=lambda x: x[1])


def violates_constraints(point_index, cluster_index, point_to_cluster_assignment, must_link_graph, cannot_link_graph):
    for adjacent_point in must_link_graph[point_index]:
        # Point was assigned and the must link nodes have been assigned to other clusters => Violation
        if point_to_cluster_assignment[adjacent_point] != -1 \
                and point_to_cluster_assignment[adjacent_point] != cluster_index:
            return True

    for adjacent_point in cannot_link_graph[point_index]:
        # Adjacent point was assigned to the cluster and those two cannot be in the same cluster
        if point_to_cluster_assignment[adjacent_point] == cluster_index:
            return True

    return False


def compute_new_centroids(points):
    new_point = np.zeros(shape=(1, dimension))
    for i in range(len(points)):
        new_point += points[i]

    return new_point / len(points)


def cop(input_file, delimiter, constraints_file, no_clusters, convergence_distance, max_iterations, plot):
    start_time = time.time()
    spark = SparkSession.builder.appName("KMeans - COP").getOrCreate()
    max_iterations = max_iterations or np.inf
    points = spark.read.text(input_file).rdd \
        .map(lambda r: r[0]) \
        .map(lambda l: parse_vector(l, delimiter)) \
        .zipWithIndex() \
        .map(lambda x: (x[1], x[0])) \
        .cache()
    global dimension
    dimension = len(points.first()[1])
    print("Number of dimensions: {}".format(dimension))

    centroids = points.takeSample(False, no_clusters)
    print("Initial centroids")
    pprint(centroids)

    constraints_point = spark.read.text(constraints_file).rdd \
        .map(lambda r: r[0]) \
        .map(parse_constraints) \
        .map(lambda x: (x[2], (x[0], x[1]))) \
        .groupByKey() \
        .map(lambda x: (x[0], np.array(list(x[1])))) \
        .persist()

    constrais_p = constraints_point.collectAsMap()
    must_link_graph, cannot_link_graph = transitive_closure(constrais_p, points.count())

    count1 = 0
    for links in must_link_graph.values():
        count1 += len(links)

    count2 = 0
    for links in cannot_link_graph.values():
        count2 += len(links)

    iterations = 0
    previous_converge_distance = np.inf
    point_to_cluster_assignment = [-1] * points.count()

    while iterations < max_iterations:
        point_to_centroids = points.map(lambda point: (point[0], distance_to_centroids(point, centroids)))
        aux = point_to_centroids.collect()

        point_to_cluster_assignment = [-1] * points.count()
        for point_index, point_to_cluster_distances in aux:
            if point_to_cluster_assignment[point_index] == -1:
                counter = 0
                while counter < len(point_to_cluster_distances):
                    cluster_index = point_to_cluster_distances[counter]
                    if not violates_constraints(point_index, cluster_index, point_to_cluster_assignment,
                                                must_link_graph, cannot_link_graph):
                        point_to_cluster_assignment[point_index] = cluster_index
                        for adjacent_node in must_link_graph[point_index]:
                            point_to_cluster_assignment[adjacent_node] = cluster_index

                        break

                    counter += 1

                if counter == len(point_to_cluster_assignment):
                    print("Point cannot be assigned to a cluster due to the constraints.")
                    spark.stop()
                    return

        point_to_cluster_assignment = spark.sparkContext.parallelize(point_to_cluster_assignment)
        new_centroids = point_to_cluster_assignment \
            .zipWithIndex() \
            .map(lambda x: (x[1], x[0][0])) \
            .join(points) \
            .map(lambda x: (x[1][0], np.array(x[1][1]))) \
            .groupByKey() \
            .map(lambda x: (x[0], list(x[1]))) \
            .map(lambda x: [x[0], compute_new_centroids(x[1])]) \
            .collect()

        print("New Centroids")
        pprint(new_centroids)

        centroids_delta_dist = sum(np.sum((centroids[index][1] - p) ** 2) for index, p in new_centroids)
        for index, point in new_centroids:
            print(type(point))
            if type(point) is float:
                point = np.zeros(shape=(1, dimension))

            new_centroids[index][1] = point

        iterations += 1
        print(centroids_delta_dist)
        if centroids_delta_dist < convergence_distance or centroids_delta_dist == previous_converge_distance:
            break

        centroids = new_centroids  # If this line is commented, points that will swap the clusters
        previous_converge_distance = centroids_delta_dist

        print("Finished iteration: {}".format(iterations))
        print("Iteration Time: {}".format(time.time() - start_time))
        start_time = time.time()

    print("Finished iteration: {}".format(iterations))
    print("Iteration Time: {}".format(time.time() - start_time))

    def plot_cop(data_items, centroids, clusters):
        data_items = data_items.collect()

        from sklearn.manifold import TSNE
        points_embedded = list(map(lambda x: x[1], data_items))
        points_embedded = TSNE(n_components=2).fit_transform(points_embedded)
        data_items_indexed = []
        for index, point in enumerate(points_embedded):
            data_items_indexed.append((index, point))

        empty_clusters = no_clusters - len(centroids)
        for i in range(empty_clusters):
            centroids.append(np.zeros(shape=(1, dimension)))

        centroids_embedded = list(map(lambda x: x[1][0], centroids))
        print("Centroids")
        pprint(centroids_embedded)
        centroids_embedded = TSNE(n_components=2).fit_transform(centroids_embedded)
        centroids_indexed = []
        for index, centroid in enumerate(centroids_embedded):
            centroids_indexed.append((index, centroid))

        point_to_cluster_assignment = clusters \
            .zipWithIndex() \
            .map(lambda x: (x[0][0], x[1])) \
            .groupByKey() \
            .mapValues(lambda indexes: list(indexes)) \
            .collect()

        plot_clusters(data_items, centroids, point_to_cluster_assignment,
                      'Constraints Based')

    if plot:
        plot_cop(points, centroids, point_to_cluster_assignment)

    spark.stop()


@click.command()
@click.option("-f", "--input-file", required=True)
@click.option('-d', '--delimiter', default=' ')
@click.option("-cop", "--constraints-file", required=True)
@click.option("-k", "--no_clusters", required=True, type=click.INT)
@click.option("-c", "--convergence-distance", required=True, type=click.FLOAT)
@click.option("-i", "--max-iterations", type=click.INT)
@click.option('--plot', is_flag=True)
def main(input_file, delimiter, no_clusters, convergence_distance, constraints_file, max_iterations, plot):
    cop(input_file, delimiter, constraints_file, no_clusters, convergence_distance, max_iterations, plot)


if __name__ == '__main__':
    main()
