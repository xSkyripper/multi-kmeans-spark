from __future__ import print_function

import sys
from pprint import pprint

import numpy as np

from pyspark.sql import SparkSession


def parse_vector(line):
    return np.array([float(x) for x in line.split(' ')])


if __name__ == "__main__":
    """
    Usage: transitive_closure [partitions]
    """
    spark = SparkSession \
        .builder \
        .appName("PythonTransitiveClosure") \
        .getOrCreate()

    constraints_file = "iris_constraints.txt"
    partitions = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    tc = spark.read.text(constraints_file).rdd \
        .map(lambda r: r[0]) \
        .map(parse_vector) \
        .map(lambda x: (x[2], (x[0], x[1]))) \
        .groupByKey() \
        .map(lambda x: (x[0], list(x[1]))) \
        .cache()
    aux = tc.collectAsMap()
    tc = spark.sparkContext.parallelize(aux.get(1), partitions).cache()
    tc = spark.sparkContext.parallelize(aux.get(-1), partitions).cache()
    # Linear transitive closure: each round grows paths by one edge,
    # by joining the graph's edges with the already-discovered paths.
    # e.g. join the path (y, z) from the TC with the edge (x, y) from
    # the graph to obtain the path (x, z).

    # Because join() joins on keys, the edges are stored in reversed order.
    edges = tc.map(lambda x_y: (x_y[1], x_y[0]))

    oldCount = 0
    nextCount = tc.count()
    while True:
        oldCount = nextCount
        # Perform the join, obtaining an RDD of (y, (z, x)) pairs,
        # then project the result to obtain the new (x, z) paths.
        new_edges = tc.join(edges).map(lambda __a_b: (__a_b[1][1], __a_b[1][0]))
        tc = tc.union(new_edges).distinct().cache()
        nextCount = tc.count()
        if nextCount == oldCount:
            break

    print("TC has %i edges" % tc.count())
    pprint(tc.collect())

    spark.stop()
