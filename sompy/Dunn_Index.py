import numpy as np
from collections import defaultdict
from sklearn.preprocessing import normalize, scale


def calculate_dunn_index(cluster, data, label):

    d1=[]
    d2=[]
    d3=[]
    cluster_data=defaultdict(list)
    for i in range (cluster):
        for j in range (len(label)):
            if label[j]==i:
                cluster_data[i].append(data[j])
                #print cluster_data
    for k,v in cluster_data.iteritems():

        cluster_list=cluster_data[k]
        for i in range (len(cluster_list)):
            temp1=cluster_list[i]
            for j in range(len(cluster_list)):
                temp2 = cluster_list[j]

                dist = np.linalg.norm(np.asarray(temp1) - np.asarray(temp2))
                d1.insert(j,dist)

            d2.insert(i,max(d1))
            d1=[]

        d3.insert(k,max(d2))
        d2=[]
    xmax= max(d3)
    #################################################################################################
    #Calcuation of minimum distance
    d1=[]
    d2=[]
    d3=[]
    d4=[]
    for k, v in cluster_data.iteritems():
        cluster_list=cluster_data[k]
        #print cluster_list
        for j in range(len(cluster_list)):
            temp1=cluster_list[j]
            for key,value in cluster_data.iteritems():
                if not key==k:
                    other_cluster_list=cluster_data[key]
                    for index in range ((len(other_cluster_list))):
                        temp2=other_cluster_list[index]
                        dist=np.linalg.norm(np.asarray(temp1)-np.asarray(temp2))
                        d1.insert(index,dist)

                    d2.insert(key,min(d1))
                    d1=[]
            d3.insert(j,min(d2))
            d2=[]
        d4.insert(k,min(d3))
    xmin=min(d4)
    dunn_index= xmin/xmax
    return dunn_index
