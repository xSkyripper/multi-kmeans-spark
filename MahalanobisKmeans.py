import sys
import numpy as np
from pyspark.sql import SparkSession


def parse_vector(line):
    return np.array([float(x) for x in line.split(" ")])


def mahanobu_dist(point, centers):
    best_index = 0
    min_distance = float("+inf")
    identity_matrix = np.identity(2)
    for i in range(len(centers)):
        inverse_id_identity_matrix = np.linalg.inv(identity_matrix)
        distance = np.sqrt(np.dot(np.dot(np.transpose(point-centers[i]), inverse_id_identity_matrix), point-centers[i]))
        if distance < min_distance:
            min_distance, best_index = distance, i

    return best_index


def main():
    # if len(sys.argv) != 4:
    #     print("Usage: kmeans <file> <k> <convergeDist>", file=sys.stderr)
    #     sys.exit(-1)

    spark = SparkSession.builder.appName("PythonKMeans").getOrCreate()

    file = "input.txt"
    k = 2
    converge_distance = 0.1

    lines = spark.read.text(file).rdd.map(lambda line: line[0])
    data = lines.map(parse_vector).cache()

    k = int(k)
    converge_distance = float(converge_distance)

    centers = np.array([(1.90, 0.97), (3.17, 4.96)])

    temp_distance = 1.0

    while temp_distance > converge_distance:
        #
        closest = data.map(lambda point: (mahanobu_dist(point, centers), (point, 1)))
        point_stats = closest.reduceByKey(lambda point1, point2: (point1[0] + point2[0], point1[1] + point2[1]))
        new_centers = point_stats.map(lambda point: (point[0], point[1][0] / point[1][1])).collect()
        #

        temp_distance = sum(np.sum((centers[i] - new_center) ** 2) for i, new_center in new_centers)
        for i, point in new_centers:
            centers[i] = point

    print("Centers" + str(centers))


if __name__ == '__main__':
    main()