from __future__ import print_function

import sys
from pprint import pprint

import click
import numpy as np
from pyspark.sql import SparkSession

MUST_LINK = 1
CANNOT_LINK = -1


def parse_vector(line):
    return np.array([float(x) for x in line.split(' ')])


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
    new_point = 0
    for i in range(1, len(points)):
        new_point += points[i]

    return new_point / len(points)


@click.command()
@click.option("-f", "--file", required=True)
@click.option("-k", "--number-of-clusters", required=True)
@click.option("-c", "--convergence-distance", required=True)
@click.option("-cop", "--constraints-file", required=True)
def main(file, number_of_clusters, convergence_distance, constraints_file):
    k = int(number_of_clusters)
    convergence_distance = float(convergence_distance)

    spark = SparkSession.builder.appName("COP-K-Means").getOrCreate()

    points = spark.read.text(file).rdd \
        .map(lambda r: r[0]) \
        .map(parse_vector) \
        .cache()

    centroids = points.takeSample(False, k)
    print("Initial centroids")
    pprint(centroids)
    points = points \
        .zipWithIndex() \
        .map(lambda x: (x[1], x[0])) \
        .cache()
    # print("Points")
    # pprint(points.collect())
    # pprint("Number of points:{}".format(points.count()))

    constraints_point = spark.read.text(constraints_file).rdd \
        .map(lambda r: r[0]) \
        .map(parse_constraints) \
        .map(lambda x: (x[2], (x[0], x[1]))) \
        .groupByKey() \
        .map(lambda x: (x[0], np.array(list(x[1])))) \
        .cache()
    # print("Constraints")
    constrais_p = constraints_point.collectAsMap()
    must_link_graph, cannot_link_graph = transitive_closure(constrais_p, points.count())
    # print("Must Link Graph")
    # pprint(must_link_graph)
    # print("============================")
    # print("Cannot Link Graph")
    # pprint(cannot_link_graph)

    count1 = 0
    for links in must_link_graph.values():
        count1 += len(links)

    count2 = 0
    for links in cannot_link_graph.values():
        count2 += len(links)

    iterations = 0
    while iterations < 20:
        print("Iteration: {}".format(iterations))
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
                    sys.exit(1)

        point_to_cluster_assignment = spark.sparkContext.parallelize(point_to_cluster_assignment)
        new_centroids = point_to_cluster_assignment \
            .zipWithIndex() \
            .map(lambda x: (x[1], x[0][0])) \
            .join(points) \
            .map(lambda x: (x[1][0], np.array(x[1][1]))) \
            .groupByKey() \
            .map(lambda x: (x[0], list(x[1]))) \
            .map(lambda x: (x[0], compute_new_centroids(x[1]))) \
            .collect()

        print("New Centroids")
        pprint(new_centroids)
        pprint(len(new_centroids))

        centroids_delta_dist = sum(np.sum((centroids[index] - p) ** 2) for index, p in new_centroids)
        for index, point in new_centroids:
            if type(point) is float:
                point = np.zeros(shape=(1, 4))
            centroids[index] = point

        iterations += 1
        print(centroids_delta_dist)
        if centroids_delta_dist < convergence_distance:
            break

    spark.stop()


if __name__ == '__main__':
    main()
