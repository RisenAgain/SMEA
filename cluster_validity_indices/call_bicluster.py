import numpy as np
from cluster_validity_indices.MSR_RV_BI import MSR_RV_BI
from cluster_validity_indices.bi_clustering_size import BS

def cal_objective_functions(Bicluster, MSR_threshold, size_of_dataset):
    objective1 = []  # list to contain value of MSR of each bicluster that have MSR value less than threshold MSR value
    objective2 = []  # list to contain value of Row variance of each bicluster that have MSR value less than threshold MSR value
    objective3 = []  # list to contain value of Bicluster Index of each bicluster that have MSR value less than threshold MSR value
    objective4 = []  # list to contain value of Bicluster size of each bicluster
    for i in range(len(Bicluster)):
        P = len(Bicluster[i])
        Q = len(Bicluster[i][0])
        #print("**************************************")
        #print("Dimension of Bicluster %d:" % (i + 1), P, Q)
        np_bicluster = np.array(Bicluster[i])
        MSR_value_lessthan_threshold, Row_variance_value, BI = MSR_RV_BI(np_bicluster,P,Q,MSR_threshold)
        objective1.append(MSR_value_lessthan_threshold)
        objective2.append(Row_variance_value)
        objective3.append(BI)
        size = BS(P,Q,size_of_dataset)
        objective4.append(size)
    # print("The MSR value of those bicluster that have MSR value less than threshold MSR value: ", objective1)
    # print("The Row variance value of those bicluster that have MSR value less than threshold MSR value: ", objective2)
    # print("The Bicluster index of those bicluster that have MSR value less than threshold MSR value: ",objective3)
    # print("The Bicluster size:", objective4)
    return objective1,objective2,objective3,objective4
