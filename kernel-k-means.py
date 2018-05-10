import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import time

filePath2 = "input.txt"
dataTesting2 = np.loadtxt(filePath2, delimiter=" ")


#params
k = 2 #number of cluster
var = 5 #var in RFB kernel
iterationCounter = 0
input = dataTesting2
initMethod = "byOriginDistance" #options = random, byCenterDistance, byOriginDistance


def init_cluster(dataInput, nCluster, method):
    list_cluster_member = [[] for i in range(nCluster)]
    if method == "random":
        shuffled_data_in = dataInput
        np.random.shuffle(shuffled_data_in)
        for i in range(0, dataInput.shape[0]):
            list_cluster_member[i%nCluster].append(dataInput[i, :])

    if method == "byCenterDistance":
        center = np.matrix(np.mean(dataInput, axis=0))
        repeated_cent = np.repeat(center, dataInput.shape[0], axis=0)
        delta_matrix = abs(np.subtract(dataInput, repeated_cent))
        euclidean_matrix = np.sqrt(np.square(delta_matrix).sum(axis=1))
        data_new = np.array(np.concatenate((euclidean_matrix, dataInput), axis=1))
        data_new = data_new[np.argsort(data_new[:, 0])]
        data_new = np.delete(data_new, 0, 1)
        divider = dataInput.shape[0]/nCluster
        for i in range(0, dataInput.shape[0]):
            list_cluster_member[np.int(np.floor(i/divider))].append(data_new[i,:])

    if method == "byOriginDistance":
        origin = np.matrix([[0,0]])
        repeated_cent = np.repeat(origin, dataInput.shape[0], axis=0)
        delta_matrix = abs(np.subtract(dataInput, repeated_cent))
        euclidean_matrix = np.sqrt(np.square(delta_matrix).sum(axis=1))
        data_new = np.array(np.concatenate((euclidean_matrix, dataInput), axis=1))
        data_new = data_new[np.argsort(data_new[:, 0])]
        data_new = np.delete(data_new, 0, 1)
        divider = dataInput.shape[0]/nCluster
        for i in range(0, dataInput.shape[0]):
            list_cluster_member[np.int(np.floor(i/divider))].append(data_new[i,:])

    return list_cluster_member


def rbf_kernel(data1, data2, sigma):
    delta = abs(np.subtract(data1, data2))
    squared_euclidean = (np.square(delta).sum(axis=1))
    result = np.exp(-squared_euclidean/(2*sigma**2))
    return result


def plot_result(list_cluster_members, centroid, iteration, converged):
    n = list_cluster_members.__len__()
    color = iter(cm.rainbow(np.linspace(0, 1, n)))
    plt.figure("result")
    plt.clf()
    plt.title("iteration-" + iteration)
    for i in range(n):
        col = next(color)
        member_cluster = np.asmatrix(list_cluster_members[i])
        plt.scatter(np.ravel(member_cluster[:, 0]), np.ravel(member_cluster[:, 1]), marker=".", s=100, c=col)
    color = iter(cm.rainbow(np.linspace(0, 1, n)))
    for i in range(n):
        col = next(color)
        plt.scatter(np.ravel(centroid[i, 0]), np.ravel(centroid[i, 1]), marker="*", s=400, c=col, edgecolors="black")
    if (converged == 0):
        plt.ion()
        plt.show()
        plt.pause(0.1)
    if (converged == 1):
        plt.show(block=True)


def third_term(member_cluster):
    result = 0
    for i in range(0, member_cluster.shape[0]):
        for j in range(0, member_cluster.shape[0]):
            result = result + rbf_kernel(member_cluster[i, :], member_cluster[j, :], var)
    result = result / (member_cluster.shape[0] ** 2)
    return result


def second_term(data_i, member_cluster):
    result = 0
    for i in range(0, member_cluster.shape[0]):
        result = result + rbf_kernel(data_i, member_cluster[i, :], var)
    result = 2 * result / member_cluster.shape[0]
    return result


def k_means_kernel(data, init_method):
    global iterationCounter
    member_init = init_cluster(data, k, init_method)
    n_cluster = member_init.__len__()
    #looping until converged
    while True:
        # calculate centroid, only for visualization purpose
        centroid = np.ndarray(shape=(0, data.shape[1]))
        for i in range(0, n_cluster):
            member_cluster = np.asmatrix(member_init[i])
            centroid_cluster = member_cluster.mean(axis=0)
            centroid = np.concatenate((centroid, centroid_cluster), axis=0)
        #plot result in every iteration
        #plotResult(memberInit, centroid, str(iterationCounter), 0)
        old_time = np.around(time.time(), decimals=0)
        kernel_result_cluster_all_cluster = np.ndarray(shape=(data.shape[0], 0))
        #assign data to cluster whose centroid is the closest one
        for i in range(0, n_cluster): #repeat for all cluster
            term3 = third_term(np.asmatrix(member_init[i]))
            matrix_term3 = np.repeat(term3, data.shape[0], axis=0)
            matrix_term3 = np.asmatrix(matrix_term3)
            matrix_term2 = np.ndarray(shape=(0,1))
            for j in range(0, data.shape[0]): #repeat for all data
                term2 = second_term(data[j, :], np.asmatrix(member_init[i]))
                matrix_term2 = np.concatenate((matrix_term2, term2), axis=0)
            matrix_term2 = np.asmatrix(matrix_term2)
            kernel_result_cluster_i = np.add(-1*matrix_term2, matrix_term3)
            kernel_result_cluster_all_cluster =\
                np.concatenate((kernel_result_cluster_all_cluster, kernel_result_cluster_i), axis=1)
        cluster_matrix = np.ravel(np.argmin(np.matrix(kernel_result_cluster_all_cluster), axis=1))
        list_cluster_member = [[] for l in range(k)]
        for i in range(0, data.shape[0]):#assign data to cluster regarding cluster matrix
            list_cluster_member[np.asscalar(cluster_matrix[i])].append(data[i, :])
        for i in range(0, n_cluster):
            print("Cluster member numbers-", i, ": ", list_cluster_member[0].__len__())
        #break when converged
        bool_acc = True
        for m in range(0, n_cluster):
            prev = np.asmatrix(member_init[m])
            current = np.asmatrix(list_cluster_member[m])
            if prev.shape[0] != current.shape[0]:
                bool_acc = False
                break
            if prev.shape[0] == current.shape[0]:
                bool_per_cluster = (prev == current).all()
            bool_acc = bool_acc and bool_per_cluster
            if not bool_acc:
                break
        if bool_acc:
            break
        iterationCounter += 1

        #update new cluster member
        member_init = list_cluster_member
        new_time = np.around(time.time(), decimals=0)
        print("iteration-", iterationCounter, ": ", new_time - old_time, " seconds")

    return list_cluster_member, centroid


clusterResult, centroid = k_means_kernel(input, initMethod)
#plotResult(clusterResult, centroid, str(iterationCounter) + ' (converged)', 1)
print(centroid)
print("converged!")
