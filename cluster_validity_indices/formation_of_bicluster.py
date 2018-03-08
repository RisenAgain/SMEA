import numpy as np
 
def formation_of_bicluster(cluster_gene, gene_data, label_gene, cond_cluster, cond_data, label_cond):
    """
       :param cluster_gene: # no. of gene clusters
       :param gene_data: # the whole data for formation of gene cluster
       :param label_gene:  # list that contain label of gene cluster
       :param cond_cluster: # no. of condition cluster
       :param cond_data:  # the whole data for formation of condition cluster
       :param label_cond: # no. of gene cluster
       :param threshold_value:  # the threshold value of MSR
       :return: a list of bicluster
       """
 
    pop_gene = [[] for i in range(cluster_gene)]   # list to store all condition clusters that belong to same label
    for i in range(len(label_gene)):
        pop_gene[label_gene[i]].append(gene_data[i])
    #print(pop_gene)
    pop_cond = [[] for i in range(cond_cluster)]   # list to store all gene clusters that belong to same label
    for i in range(len(label_cond)):
        pop_cond[label_cond[i]].append(cond_data[i])
    #print(pop_cond)
 
    cond_index = [[] for j in range(cond_cluster)]  # store index of condition column that belong to same label
    # for formation of bicluster
    for i in range(len(label_cond)):
        cond_index[label_cond[i]].append(i)
    #print(cond_index)
 
 
    arr = []  # store the value of starting location of each condtion cluster in original data in first gene cluster
    arr.append(len(pop_cond[0]))
    for i in range(1, len(pop_cond)):
        arr.append(arr[i - 1] + (len(pop_cond[i])))
    #print("Length of each condition cluster:", arr)
    again = [0]    # store the value of starting location of each condtion cluster in original data in all gene cluster
    new = []  # store length of condition cluster
    again.extend(arr)
    #print(again)
    for i in range(cond_cluster):
        new.append(len(pop_cond[i]))
    #print(new)
    for i in range(1, cluster_gene):
        for j in new:
            flag = again[-1]
            flag += j
            again.append(flag)
    #print(again)
 
    max = []  # store all gene cluster in transpose manner
    for s in pop_gene:
        for sub in cond_index:
            for i in sub:
                k = np.array(s).T.tolist()
                max.append((k[i]))
    #print(max)
 
    # to finally select biclusters from max list
    Bicluster = []  # store bicluster of all possible combination of gene and condition cluster
    for i in range(1, len(again)):
        Bicluster.append(max[again[i - 1]:again[i]])
    #print("All biclusters:", Bicluster)
    return Bicluster
