# SMEA
A Self-Organizing Multi-objective Evolutionary Algorithm implemented to find coherent bi-clusters in the data

This is a customized implementation of a Self-Organizing Multi-Objective Evolutionary Algorithm presented [here](http://ieeexplore.ieee.org/abstract/document/7393537/). 
SMEA is used to find the coherent bi-clusters in training dataset. A bi-cluster consist of a cluster of some training
instances and a cluster of some features. A bi-cluster is coherent if the features in feature cluster are important for the training 
instances in the training instance cluster. Thus both clusters vary coherently.

The goodness of a bi-cluster is measured with multiple metrics including Mean Square Residue (MSR), Row Variance (RV), Bi-cluster 
Index (BI), Bi-cluster Size (BS), Bi-cluster Volume (BV). Thus to optimize these multiple objective functions, we use SMEA. The
bi-clusters can then be used to determine which features are important for a particular category of training instances.

The algorithm is mainly implemented in file [sompy/main_bi.py](https://github.com/chiragiitp/SMEA/blob/master/sompy/main_bi.py).

Sompy is the implementation of Self-organizing map which have been taken and modified for this problem from [here](https://github.com/sevamoo/SOMPY).
