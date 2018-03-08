import numpy as np
from sklearn.cluster import AgglomerativeClustering
from pbm_Index import cal_pbm_index
from sklearn.metrics import silhouette_score


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
        #print "Centroid :", centroids.keys()
        centroids[clno] += data[idx,:]
        lens.setdefault(clno,0)
        lens[clno] += 1
    # Divide by number of observations in each cluster to get the centroid
    for clno in centroids:
        centroids[clno] /= float(lens[clno])
    return np.asarray(centroids.values())
#

def hie_clustering(K,Idata):
    agglomerative1= AgglomerativeClustering(n_clusters=K, linkage='ward', affinity='euclidean')
    agglomerative1.fit_predict(Idata)
    hlabels=agglomerative1.labels_
    #print "hlabels : ", hlabels
    centers = cal_center(Idata,hlabels)
    ss = silhouette_score(Idata, hlabels)
    # di = calculate_dunn_index(Idata,label,n_cluster=i+1)
    pbm = cal_pbm_index(K, Idata, centers, hlabels)
    return centers, hlabels, ss,pbm