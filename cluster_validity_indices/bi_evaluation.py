
import numpy as np
import math
import sys

def bicluster_evaluation(cluster_gene, gene_data, label_gene, cond_cluster, cond_data, label_cond, MSR_threshold):
    """
    :param cluster_gene: no. of gene clusters
    :param gene_data:  original data
    :param label_gene:  cluster labels of gene data
    :param cond_cluster: no. of gcondition clusters
    :param cond_data:original data
    :param label_cond: cluster labels of condition data
    :return:  no. of biclusters, (MSR, RV, BI size, BI index) of the biclusters
    """
    bicluster={} #STORE ALL possible biclusters
    bicluster_MSR=[]           #store all bicluster MSR
    delta_bicluster_MSR=[]         #store delta-bicluster MSR
    bicluster_RV=[]     #store row variance
    bicluster_volume=[]
    delta_bicluster_volume = []
    delta_bicluster_RV=[] #store delta bicluster row variance
    bicluster_BIsize=[]   #store all bicluster size
    delta_bicluster_BIsize=[]  #store delta  bicluster size
    bicluster_BI_index=[]  #store all bicluster BI index
    delta_bicluster_BI_index=[]  #store delta bicluster BI index
    # print "-------------------Inside bicluster------------------"
    # print "clus gene and cond : ", cluster_gene, cond_cluster
    # print "gene labels : ", label_gene
    # print "cond labels : ",  label_cond
    # print "GENE data:", gene_data
    gene_index = [[] for j in range(cluster_gene)]  # store index of gene cluster indices that belong to same label
    for i in range(len(label_gene)):
        gene_index[label_gene[i]].append(i)

    cond_index = [[] for j in range(cond_cluster)]  # store index of condition column indices that belong to same label
    for i in range(len(label_cond)):
        cond_index[label_cond[i]].append(i)
    # print "gene index : ", gene_index
    # print "cond index:", cond_index, len(cond_index)
    biclusterno=0
    """Form biclusters"""
    for kk in range(len(gene_index)):
        a = gene_index[kk]    #gene cluster data indices
        for jj in range(len(cond_index)):
            b = cond_index[jj]  #condition cluster data indices
            bicluster[biclusterno]=np.zeros((len(a),len(b)))
            for index1 in range(len(a)):
                for index2 in range(len(b)):
                    bicluster[biclusterno][index1][index2] =gene_data[a[index1]][b[index2]]
            # print "bilcuster number  {0} : ".format(biclusterno),bicluster[biclusterno]
            # print "row means:", cal_row_means(bicluster[biclusterno]), type(cal_row_means(bicluster[biclusterno]))
            # print "col means:", cal_col_means(bicluster[biclusterno]), type( cal_col_means(bicluster[biclusterno]))
            # print "bicluster mean:", bicluster_means(bicluster[biclusterno]), type(bicluster_means(bicluster[biclusterno]))

            MSR_avg, RV_avg, BI_size, BI_index=cal_MSR_RV_BIsize_BIindex(bicluster[biclusterno], gene_data, MSR_threshold)

            bicluster_MSR.append(MSR_avg)
            bicluster_RV.append(RV_avg)
            bicluster_BIsize.append(BI_size)
            bicluster_BI_index.append(BI_index)
            bicluster_volume.append(len(a) * len(b))
            biclusterno+=1
    #return biclusterno, np.mean(bicluster_MSR), np.mean(bicluster_RV), np.mean(bicluster_BIsize), np.mean(bicluster_BI_index), np.mean(bicluster_volume)
    # """calculate objective functions"""
    no_of_delta_bicluster=0
    for i in range(len(bicluster_MSR)):
        if bicluster_MSR[i]>0 and bicluster_MSR[i]<MSR_threshold:
            no_of_delta_bicluster+=1
            delta_bicluster_MSR.append(bicluster_MSR[i])  #deltaMSR
            delta_bicluster_RV.append(bicluster_RV[i])  #DELTA rv
            delta_bicluster_BIsize.append(bicluster_BIsize[i])  #DELTA BICLUSTER SIZE
            delta_bicluster_BI_index.append(bicluster_BI_index[i])  #DELTA BI-INDEX VALUE
            delta_bicluster_volume.append(bicluster_volume[i])   #DELTA BI-VOLUME

    if no_of_delta_bicluster!=0 and no_of_delta_bicluster>2:
        return no_of_delta_bicluster, min(delta_bicluster_MSR),np.mean(delta_bicluster_MSR), max(delta_bicluster_RV),np.mean(delta_bicluster_RV), max(delta_bicluster_BIsize), np.mean(delta_bicluster_BIsize), min(delta_bicluster_BI_index), np.mean(delta_bicluster_BI_index), max(delta_bicluster_volume), np.mean(delta_bicluster_volume)
    else:
         return 0, sys.maxint, sys.maxint, (-sys.maxsize -1), (-sys.maxsize -1), (-sys.maxsize -1),(-sys.maxsize -1), sys.maxint, sys.maxint, (-sys.maxsize -1), (-sys.maxsize -1)
    #



def cal_row_means(bicluster):
    """
    :param bicluster: one bilcuster
    :return: row means
    """
    row_means=bicluster.mean(axis=1)
    return row_means

def cal_col_means(bicluster):
    """
    :param bicluster: bilcuster
    :return: column means
    """
    col_means = bicluster.mean(axis=0)
    return col_means

def bicluster_means(bicluster):
    """
    :param row_mean:  single bilcuster
    :return: mean of bicluster
    """
    mean=np.mean(bicluster)
    return mean

def cal_MSR_RV_BIsize_BIindex(bicluster, gene_data,MSR_threshold):
    """
    :param bicluster:
    :return:
    """
    row,cols=bicluster.shape
    if row * cols > 4:
        row_mean = cal_row_means(bicluster)
        col_mean = cal_col_means(bicluster)
        mean = bicluster_means(bicluster)

        sum_msr=0
        sum_rv=0
        for i in range(row):
            for j in range(cols):
                #if bicluster[i][j] !=0:
                    temp=bicluster[i][j] - row_mean[i] - col_mean[j] + mean
                    value=math.pow(temp,2)
                    sum_msr+=value          #element squared residue

                    temp1=bicluster[i][j]-row_mean[i]   #element square variance
                    val_rv=math.pow(temp1,2)
                    sum_rv +=val_rv
                    #bicluster_RV[i][j]=val_rv
        MSR= float(sum_msr)/(row*cols)
        RV=float(sum_rv)/(row*cols)
        BI_size=float(row*cols)/(len(gene_data)*len(gene_data[0]))
        BI_index=  float(MSR)/(1+RV)
        return MSR, RV,  BI_size, BI_index
    else:
        return sys.maxint, (-sys.maxsize - 1), (-sys.maxsize - 1), sys.maxint
    # if MSR< MSR_threshold:
    #    return float(MSR)/MSR_threshold, (float(1)/(1+RV)), (float(1)/BI_size), BI_index
    #






