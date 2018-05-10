import sys
import numpy as np
from pyspark.sql import SparkSession
import random
import operator
import math


def parse_vector(line):
    return np.array([float(x) for x in line.split(" ")])


def euclidean_distance(point, centers):
    best_index = 0
    min_distance = float("+inf")
    for i in range(len(centers)):
        distance = np.sum((point - centers[i]) ** 2)
        if distance < min_distance:
            min_distance, best_index = distance, i
    return best_index


def initialize_membership_matrix(n, k):
    membership_mat = list()
    for i in range(n):
        random_num_list = [random.random() for i in range(k)]
        summation = sum(random_num_list)
        temp_list = [x/summation for x in random_num_list]
        membership_mat.append(temp_list)
    return membership_mat


def calculate_cluster_center(points, membership_mat, n, k, fuzzy_factor):
    cluster_mem_val = list(zip(*membership_mat))
    cluster_centers = list()
    for j in range(k):
        x = list(cluster_mem_val[j])
        xraised = [e ** fuzzy_factor for e in x]
        denominator = sum(xraised)
        temp_num = list()
        for i in range(n):
            data_point = points[i]
            prod = [xraised[i] * val for val in data_point]
            temp_num.append(prod)
        numerator = map(sum, zip(*temp_num))
        center = [z/denominator for z in numerator]
        cluster_centers.append(center)
    return cluster_centers


def update_membership_value(points, membership_mat, n, k, fuzzy_factor, cluster_centers):
    p = float(2/(fuzzy_factor-1))
    for i in range(n):
        x = list(points[i])
        distances = [np.linalg.norm(list(map(operator.sub, x, cluster_centers[j]))) for j in range(k)]
        for j in range(k):
            den = sum([math.pow(float(distances[j]/distances[c]), p) for c in range(k)])
            membership_mat[i][j] = float(1/den)
    return membership_mat


def main():
    # if len(sys.argv) != 4:
    #     print("Usage: kmeans <file> <k> <convergeDist>", file=sys.stderr)
    #     sys.exit(-1)

    spark = SparkSession.builder.appName("PythonKMeans").getOrCreate()
    file = "input.txt"
    k = 2
    lines = spark.read.text(file).rdd.map(lambda line: line[0])
    data = lines.map(parse_vector).cache()
    k = int(k)
    curr = 0
    MAX_ITER = 100
    count_points = data.collect().__len__()
    membership_mat = initialize_membership_matrix(count_points, k)
    print(membership_mat)
    points = data.collect()
    while curr < MAX_ITER:
        cluster_centers = calculate_cluster_center(points, membership_mat, count_points, k, 1.5)
        membership_mat = update_membership_value(points, membership_mat,count_points, k, 1.5, cluster_centers)
        curr += 1
        print(cluster_centers)


if __name__ == '__main__':
    main()