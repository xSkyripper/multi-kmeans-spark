== Project structure ==
The project contains the following:
    - 'kmeans': a Python module that contains the main implementations and utils
        - main implementations: default, mahalanobis, fuzzy, kernel, cop, pso
        - utils:
            - utils: contains the main generic plotting function
            - generate_ds: CLI for generating artificial dataset
            - subsample_ds: CLI for subsampling for different parameters

    - 'data': a folder containing the testing & evaluating datasets
        - ds_3d_spatial_network_huge: a 450 000 instances, 4 dimensional dataset
        - ds_cop_small: a 150 instances, 4 dimensional COP dataset
            - ds_cop_constraints_small: a 60 constaints COP file
        - ds_cop_big: a 100 000 instances, 4 dimensional COP dataset (extended from small)
            - ds_cop_constraints_big: a 33 000 constaints COP file (extended from small)
        - ds_kernel_small: a 400 instances, 2 dimensional Kernel dataset
        - ds_kernel_big: a 100 000 instances, 2 dimensional Kernel dataset (extended from small)
        - ds_mlbook_small: a 10 instances, 2 dimensional test dataset

== Setup & Commands ==
    The entire development has been done inside a Python virtual environment.
There is a set of necessary steps that need to be done before running the K-Means implementations

= Local =
1. Make sure python3 and pip for python3 are installed
2. Clone the repository and 'cd' into it
3. Create venv:
    $ python3 -m venv venv
4. Activate venv:
    $ source venv/bin/activate
5. Install requirements:
    $ pip install -r requirements.txt
6. Run the algorithms. Examples:
    $ python -m kmeans.default -f data/ds_cop_small.txt -k 2 -c 0.05
    $ python -m kmeans.mahalanobis -k 2 -c 0.1 -f data/ds_kernel_small.txt -i 50
    $ python -m kmeans.fuzzy -k 2 -c 0.01 -m 1.25 -f data/ds_kernel_small.txt
    $ python -m kmeans.cop -f data/ds_cop_small.txt -k 2 -cop data/ds_cop_constraints_small.txt -c 0.05
    $ python -m kmeans.kernel -f data/ds_kernel_small.txt -k 2
    $ python -m kmeans.pso -f data/ds_kernel_small.txt -k 2 --itr 2 -i 50

    # Every command supports '--help' for explanations
    # Every command also supports '--plot' for plotting the clusters of 2-dimensional datasets

= Dataproc =
1. Install google-cloud-sdk and setup the profile and project
2. Create the Google Storage buckets and upload input data there (multi-kmeans is ours)
3. Run the cluster creation command based on the custom Conda script
(after you upload it to Google Storage):
      $ gcloud dataproc clusters create multi-kmeans-cluster \
            --bucket multi-kmeans \
            --master-boot-disk-size 50GB \
            --zone europe-west1-b \
            --worker-boot-disk-size 50GB \
            --num-worker-local-ssds 1 \
            --master-machine-type n1-standard-8 \
            --worker-machine-type n1-standard-8 \
            --num-workers 4 \
            --initialization-actions gs://multi-kmeans/src/create_my_cluster.sh
4. SSH on the master node (multi-kmeans-cluster-m)
5. Clone the repository and 'cd' into it
6. Run the commands.
    # All of the commands below are almost the same as the local ones
    (only difference is the file execution and not python module and the input data is from gs://)
    # Examples:
    $ spark-submit kmeans/mahalanobis.py -f gs://multi-kmeans/data/ds_kernel_big.txt -k 2 -c 0.01 -i 100 -d ' '
    $ spark-submit kmeans/fuzzy.py -f gs://multi-kmeans/data/ds_kernel_big.txt -k 2 -c 0.01 -m 1.25 -d ' '
