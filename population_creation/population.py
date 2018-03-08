import numpy as np

def pop_create(center_gene,  center_cond):
    #pop_create(max_cluster, feature, K[i], x[i], max_cluster_cond, feature_cond,  K_cond[j], x_cond[j])
    """
    :param max_gene_cluster: maximum number no. of gene clusters
    :param gene_features: no. of features(condition) in one gene
    :param cluster_gene: no. of gene cluster(K) generated
    :param u_gene: center of K gene clusters
    :param max_cluster: maximum number no. of gene clusters
    :param features: no. of features(genes) in one condition
    :param cluster: no. of condition cluster(K) generated
    :param center: center of K condition clusters
    :return: combine gene cluster centers nad condition cluster centers
    """

    # center=np.zeros((len(center_gene)))
    # for pos in range(len(center_gene)):
    #     center[pos]=center_gene[pos]

    center=np.append(center_gene, center_cond)
    return center