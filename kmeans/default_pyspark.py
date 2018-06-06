import click
import numpy as np
import time
from pyspark.mllib.clustering import KMeans
from pyspark.sql import SparkSession


def parse_vector(line, delimiter):
    return np.array([float(x) for x in line.split(delimiter)])


def default_pyspark(input_file, delimiter, k, max_iterations):
    spark = SparkSession.builder.appName('KMeans - Default').getOrCreate()
    lines = spark.read.text(input_file).rdd.map(lambda r: r[0])
    data_items = lines.map(lambda x: parse_vector(x, delimiter)).cache()

    start_time = time.time()
    _ = KMeans.train(data_items, k, maxIterations=max_iterations)
    end_time = time.time()
    print("Finished PySpark's KMeans|| clustering for k={}, max_iterations={} in {} seconds"
          .format(k, max_iterations, end_time - start_time))


@click.command()
@click.option('-f', '--input-file', required=True)
@click.option('-d', '--delimiter', required=True, default=',')
@click.option('-k', '--no-clusters', required=True, type=click.INT)
@click.option('-i', '--max-iterations', type=click.INT, default=100)
def main(input_file, delimiter, no_clusters, max_iterations):
    default_pyspark(input_file, delimiter, no_clusters, max_iterations)


if __name__ == '__main__':
    main()
