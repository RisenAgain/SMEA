import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from numpy import genfromtxt
from sklearn.cluster import KMeans,AgglomerativeClustering
from pbm_Index import cal_pbm_index
from sklearn.metrics import silhouette_score
from Evolutionary.Evoution1 import calculate_dunn_index


Idata = genfromtxt('/home/naveen.pcs16/Experiments/Text_clustering_SMEA/SMEA_Text_auto/data_set/BBC/BBC_gensim/BBC_gensim_300d.csv', delimiter='\t' ,usecols=range(0,300))
Idata_label=genfromtxt('/home/naveen.pcs16/Experiments/Text_clustering_SMEA/SMEA_Text_auto/data_set/BBC/BBC_gensim/BBC_gensim_300d.csv', delimiter='\t' ,usecols=(300))
print "first doc vector : ", Idata[0]
print "data shape : ", Idata.shape
print "Idata labels :", Idata_label
print "No of lablesl : ", len(Idata_label)


#from sklearn.decomposition import TruncatedSVD
#svd = TruncatedSVD(n_components=50, n_iter=7, random_state=42)
#Idata=svd.fit_transform(Idata)



def cal_center(data, T):
    """
    :param features: data points
    :param T: Cluster label of data points
    :return:
    """
    lens = {}      # will contain the lengths for each cluster
    centroids = {} # will contain the centroids of each cluster
    for idx,clno in enumerate(T):
        centroids.setdefault(clno,np.zeros(len(data[0])))
        centroids[clno] += data[idx,:]
        lens.setdefault(clno,0)
        lens[clno] += 1
    # Divide by number of observations in each cluster to get the centroid
    for clno in centroids:
        centroids[clno] /= float(lens[clno])
    return centroids
#
# Idata=np.array([[1,2,3],[3,2,1],[4,5,6],[6,5,4],[66,74,67],[74,67,66],[44,26,34],[967,589,122]])

K=24
print "No of clusters : ", K
cluster = KMeans(n_clusters=K)
cluster.fit(Idata)
label = cluster.predict(Idata)
centers = cluster.cluster_centers_
#print "Kmeans center : ", centers
#print "cal centers : ", cal_center(Idata,label)
dn=calculate_dunn_index(Idata, label, n_cluster=K)
print "Kmeans Siloutte score : ", silhouette_score(Idata, label)
print "PBM index : ", cal_pbm_index(K, Idata, centers, label)
print "Dunn index : ", dn
print "-------------------------------------------------------------\n"

agglomerative1= AgglomerativeClustering(n_clusters=K, linkage='ward', affinity='euclidean')
idx= agglomerative1.fit_predict(Idata)
hlabels=agglomerative1.labels_
centers = cal_center(Idata,hlabels)
dn=calculate_dunn_index(Idata, hlabels,n_cluster=K)
print "Agg ward Siloutte score : ", silhouette_score(Idata, hlabels)
print "PBM index : ", cal_pbm_index(K, Idata, centers, hlabels)
print "Dunn index : ", dn
print "---------------------------------------------------------\n"


agglomerative= AgglomerativeClustering(n_clusters=K, linkage='average', affinity='euclidean')
agglomerative.fit_predict(Idata)
hlabels=agglomerative.labels_
centers = cal_center(Idata,hlabels)
dn=calculate_dunn_index(Idata, hlabels,n_cluster=K)
print "AGG average Siloutte score : ", silhouette_score(Idata, hlabels)
print "PBM index : ", cal_pbm_index(K, Idata, centers, hlabels)
print "Dunn index : ", dn
print "-------------------------------------------------------------\n"



agglomerative2= AgglomerativeClustering(n_clusters=K, linkage='complete', affinity='euclidean')
agglomerative2.fit_predict(Idata)
hlabels=agglomerative2.labels_
centers = cal_center(Idata,hlabels)
dn=calculate_dunn_index(Idata, hlabels,n_cluster=K)
print "Agg completer Siloutte score : ", silhouette_score(Idata, hlabels)
print "PBM index : ", cal_pbm_index(K, Idata, centers, hlabels)
print "Dunn index : ", dn
#print "hlabels : ", hlabels
print "---------------------------------------------------------\n"
print "---------------------------------------------------------\n"
print "---------------------------------------------------------\n"
# K=4
# cluster = KMeans(n_clusters=K)
# cluster.fit(Idata)
# label = cluster.predict(Idata)
# centers = cluster.cluster_centers_
# #print "Kmeans center : ", centers
# #print "cal centers : ", cal_center(Idata,label)
# dn=calculate_dunn_index(Idata, label, n_cluster=K)
# print "Kmeans Siloutte score : ", silhouette_score(Idata, label)
# print "PBM index : ", cal_pbm_index(K, Idata, centers, label)
# print "Dunn index : ", dn
# print "-------------------------------------------------------------\n"
#
# agglomerative1= AgglomerativeClustering(n_clusters=K, linkage='ward', affinity='euclidean')
# idx= agglomerative1.fit_predict(Idata)
# hlabels=agglomerative1.labels_
# centers = cal_center(Idata,hlabels)
# dn=calculate_dunn_index(Idata, hlabels,n_cluster=K)
# print "Agg ward Siloutte score : ", silhouette_score(Idata, hlabels)
# print "PBM index : ", cal_pbm_index(K, Idata, centers, hlabels)
# print "Dunn index : ", dn
# print "---------------------------------------------------------\n"
#
#
# agglomerative= AgglomerativeClustering(n_clusters=K, linkage='average', affinity='euclidean')
# agglomerative.fit_predict(Idata)
# hlabels=agglomerative.labels_
# centers = cal_center(Idata,hlabels)
# dn=calculate_dunn_index(Idata, hlabels,n_cluster=K)
# print "AGG average Siloutte score : ", silhouette_score(Idata, hlabels)
# print "PBM index : ", cal_pbm_index(K, Idata, centers, hlabels)
# print "Dunn index : ", dn
# print "-------------------------------------------------------------\n"
#
#
#
# agglomerative2= AgglomerativeClustering(n_clusters=K, linkage='complete', affinity='euclidean')
# agglomerative2.fit_predict(Idata)
# hlabels=agglomerative2.labels_
# centers = cal_center(Idata,hlabels)
# dn=calculate_dunn_index(Idata, hlabels,n_cluster=K)
# print "Agg completer Siloutte score : ", silhouette_score(Idata, hlabels)
# print "PBM index : ", cal_pbm_index(K, Idata, centers, hlabels)
# print "Dunn index : ", dn
# #print "hlabels : ", hlabels
# print "---------------------------------------------------------\n"
#
