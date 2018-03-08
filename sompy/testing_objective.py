import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from random import uniform
from numpy import genfromtxt
from sklearn.cluster import KMeans
from population_creation.population import pop_create
import matplotlib.pyplot as plt

from Evolutionary.Evoution1 import Mapsize, generate_matingPool,Generate,mapping, calculate_dunn_index
from nsga2.main import Select
from normalization import NormalizatorFactory
from random import randint
#from cluster_validity_indices.formation_of_bicluster import formation_of_bicluster
from cluster_validity_indices.test_obj import form_bicluster_and_cal_objectives

#
# file1=open('/home/naveen.pcs16/Experiments/Text_clustering_SMEA/SMEA_Text_auto/Output/Output_NIPS2015_gensim_300d_generation_details_run4.txt','w')
# file2=open('/home/naveen.pcs16/Experiments/Text_clustering_SMEA/SMEA_Text_auto/Output/Output_NIPS2015_gensim_300d_run4.txt','w')
import math
from som_train import build_lattice, closest_neuron,locate_neighbouring_neurons,update_training_parameters,update_weights


#np.set_printoptions(threshold='nan')
Iflag=0       #Flag
x_gene = []        #store all gene cluster centers
x_cond=[]     #store all condition cluster centers

counter=0
TDS=[0]
prev_ss=0
prev_dunn=0


MSR=[]        #List of ari against each solution in population
RV=[]    #List of Silhoutte Score angainst each solution in population
BI_size=[]


#Idata_gene = genfromtxt('/home/acer/PycharmProjects/biclustering_SMEA/SMEA_Text_auto/data_set/Breast_Cancer_temp.csv', delimiter=',' , skip_header=1,usecols=range(2,11))
Idata_gene = genfromtxt('/home/acer/PycharmProjects/biclustering_SMEA/SMEA_Text_auto/data_set/biclustering_data_Sets/colon_cancer.txt', delimiter='')

# normalizer = NormalizatorFactory.build('var')
# Idata_gene=normalizer.normalize(Idata_gene)
Idata_condition = Idata_gene.transpose()

# print "first gene vector : ", Idata_gene[0]
# print "first condition vector : ", Idata_condition[0]
print "gene data shape : ", Idata_gene.shape
print "condition data shape : ", Idata_condition.shape


"""-------------------Maximum cluster possible for gene and condition data-------------------"""
max_cluster_gene= int(math.floor(math.sqrt(len(Idata_gene))))     #Max. cluster in population
max_cluster_cond= int(math.floor(math.sqrt(len(Idata_condition))))     #Max. cluster in population
print "Max. no. of cluster in gene and condition data  : ",max_cluster_gene,",", max_cluster_cond

"""------------------Maximum chromosome length for gene and condition separately---------------"""
max_choromosome_length_gene = (max_cluster_gene)*len(Idata_gene[0])
max_choromosome_length_cond = (max_cluster_cond)*len(Idata_condition[0])
print "Max. length of chromosome for gene and condition: ", max_choromosome_length_gene, max_choromosome_length_cond

"""No. of features in gene and condition data"""
feature_gene=len(Idata_gene[0])      #no. of features present in gene data
feature_cond=len(Idata_condition[0])      #no. of features present in condition data

K_gene=[]      #store no. of clusters in gene solution
K_cond=[] #store no. of clusters in condition solution
K_population={}

Initial_Label_gene=[]       #store label of gene data present in gene solution
Initial_Label_cond=[]   #store label of condition data present in condition solution
Initial_Label_population={}

CH=input("enter No. of chromosome for gene clusters solution: ")
print "Start Forming Gene cluster solution"

for i in range(1,CH+1):
    """--------------------------Create gene clusters population------------------------"""
    counter+=1
    pop = []
    n=randint(2, max_cluster_gene)
    K_gene.insert(i,n)
    cluster = KMeans(n_clusters=n, init='random', max_iter=1, n_init=1)
    cluster.fit(Idata_gene)
    label = cluster.predict(Idata_gene)
    centers = cluster.cluster_centers_
    centers = np.reshape(centers, n*feature_gene)
    z = int(max_choromosome_length_gene - (n* feature_gene))  ##Calculate number of zeros to append(Maximum cluster dimension - Current cluster dimension)
    for j in range(z):
        centers = np.append(centers, [0])
    Initial_Label_gene.insert(i, label.tolist())
    x_gene.insert(i,centers)


CH_cond = input("enter No. of chromosome for condition clusters solution:  ")
print "Start Forming condition cluster solution"

for i in range(1,CH_cond+1):
    """---------------------Create condition clusters population-----------------"""
    pop_cond=[]
    n1 = randint(2, max_cluster_cond)
    K_cond.insert(i, n1)
    cluster1 = KMeans(n_clusters=n1, init='random', max_iter=1, n_init=1)
    cluster1.fit(Idata_condition)
    label1 = cluster1.predict(Idata_condition)
    centers1 = cluster1.cluster_centers_
    centers1 = np.reshape(centers1, n1 * feature_cond)
    z = int(max_choromosome_length_cond - (n1 * feature_cond))  ##Calculate number of zeros to append(Maximum cluster dimension - Current cluster dimension)
    for k in range(z):
        centers1 = np.append(centers1, [0])
    Initial_Label_cond.insert(i, label1.tolist())
    x_cond.insert(i, centers1)

print "population K : ", K_gene,K_cond
population=[]
counter=0
print "Start Forming population"
for i in range(len(x_gene)):
    for j in range(len(x_cond)):
        new_center = pop_create(x_gene[i],  x_cond[j])
        population.insert(counter, new_center)
        K_population[counter]=[K_gene[i], K_cond[j]]
        Initial_Label_population[counter]=[Initial_Label_gene[i], Initial_Label_cond[j]]
        # print "clus:", K_gene[i], K_cond[j]
        # print "labels : ", Initial_Label_gene[i], Initial_Label_cond[j]
        print type(Initial_Label_gene[i]), type(Initial_Label_cond[j])
        no_of_biclusters, msr, rv, BIsize,Biindex = form_bicluster_and_cal_objectives(K_gene[i], Idata_gene, Initial_Label_gene[i], K_cond[j], Idata_condition,Initial_Label_cond[j], MSR_threshold=500)
        print "No. of biclusters, MSR, RV, BI_size, BI_index of {0}th solution ".format(counter),  no_of_biclusters, msr, rv, BIsize,Biindex

        counter+=1