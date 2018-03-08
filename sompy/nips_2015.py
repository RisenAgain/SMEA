import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))



import numpy as np
from numpy import genfromtxt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster.supervised import adjusted_rand_score
from minkowski_score import cal_minkowski_score
from sklearn.preprocessing import normalize
#from plotter import plot_max_Sil_Score
import sompy
from Evolutionary.Evolution2 import Mapsize, generate_matingPool,Generate,mapping, calculate_dunn_index
from nsga2.main import Select
from normalization import NormalizatorFactory
from sklearn import cluster
from sklearn.metrics import normalized_mutual_info_score
from random import randint
from plotter import plot_max_Sil_Score

#
# file1=open('/home/naveen.pcs16/Experiments/Text_clustering_SMEA/SMEA_Text_auto/Output/Output_NIPS2015_gensim_300d_generation_details_run4.txt','w')
# file2=open('/home/naveen.pcs16/Experiments/Text_clustering_SMEA/SMEA_Text_auto/Output/Output_NIPS2015_gensim_300d_run4.txt','w')
import math
from som_train import build_lattice, closest_neuron,locate_neighbouring_neurons,update_training_parameters,update_weights
from pbm_Index import cal_pbm_index

#np.set_printoptions(threshold='nan')
Iflag=0       #Flag
x = []        #store all center
PBM=[]        #List of ari against each solution in population
sil_sco=[]    #List of Silhoutte Score angainst each solution in population
counter=0
actual_label=[]
TDS=[0]
prev_ss=0
prev_dunn=0



#/media/acer/New Volume/server_26_25/Experiments/Text_clustering_SMEA/SMEA_Text_auto/data_set/NIPS_2015/nips_count_term_docvector_183d.csv
Idata = genfromtxt('/home/acer/PycharmProjects/Text_clustering_SMEA/SMEA_Text_auto/data_set/Breast_Cancer_temp.csv', delimiter=',' , skip_header=1,usecols=range(2,11))
#Idata = genfromtxt('/home/acer/PycharmProjects/Text_clustering_SMEA/SMEA_Text_auto/data_set/biclustering_data_Sets/colon_cancer.txt', delimiter='' )
print "first doc vector : ", Idata[0]
print "data shape : ", Idata.shape

#normalizer = NormalizatorFactory.build('var')
#Idata=normalizer.normalize(Idata)

#print "new data shape : ", Idata.shape
max_cluster= int(math.floor(math.sqrt(len(Idata))))     #Max. cluster in population
print "Max. no. of cluster : ",max_cluster
Initial_Label=[]
max_choromosome_length=(max_cluster)*len(Idata[0])
print "Max. length of chromosome : ", max_choromosome_length
CH=input("enter No. of chromosome : ")
T=int(input("Enter no. of generation-  "))

K=[]
for i in range(1,CH+1):
    counter+=1
    pop = []
    n=randint(2, max_cluster)
    K.insert(i,n)
    print "no. of cluster : ", n
    cluster = KMeans(n_clusters=n)
    cluster.fit(Idata)
    label = cluster.predict(Idata)
    centers = cluster.cluster_centers_
    a=centers.tolist()
    for j in range(len(a)):
        for k in range(len(Idata[0])):
            pop.append(a[j][k])
    if not max_choromosome_length-len(pop)== 0:
        extra_zero= max_choromosome_length-len(pop)
        pop.extend(0 for x in range(extra_zero))
    x.insert(i,pop)
    ss=silhouette_score(Idata, label)
    pbm = cal_pbm_index(n, Idata, centers, label)
    sil_sco.insert(i, ss)
    PBM.insert(i,pbm)
    Initial_Label.insert(i,label.tolist())

Final_label=list(Initial_Label)
print "Chromosome clusers :", K
print "Chromosomes PBM Index  :", PBM
print "Chromosome SI  : ", sil_sco
print "Max PBM, Sil   : ", max(PBM), ",", max(sil_sco)
print "clusters corresponding to Max PBM, Sil : ", K[PBM.index(max(PBM))], ", ", K[sil_sco.index(max(sil_sco))]

# file2.write("\n Initial clusters : "+ str(K) + '\n' )
# file2.write("Chromosome initial PBM index: "+ str(PBM) + '\n' )
# file2.write("Chromosome initial Sil score : " +str(sil_sco)+ '\n')
# file2.write("Chromosome initial Dunn : " +str(ini_Dunn)+'\n')
# file2.write("\n Max PBM, Sil, Dunn : " + str(max(PBM)) +" , " + str(max(sil_sco)) + " , "+ str(max(ini_Dunn)))
# file2.write("\n clusters corresponding to Max PBM, Sil, Dunn : "+ str(K[PBM.index(max(PBM))])+","+str(K[sil_sco.index(max(sil_sco))])+","+str(K[ini_Dunn.index(max(ini_Dunn))]))
# file2.write("\n Max DUNN and corresponding cluster, PBM, Sil : " + str(K[index]) + ", "+str(PBM[index])+","+str(sil_sco[index]) )
# file2.write("\n ============================================================================" +'\n')


print "############################################"
k1,k2 = Mapsize(counter)                  # Get SOM Map size
som = None
TDS = [0]
self_pop = None
zdt_definitions = None
plotter = None
problem = None
evolution = None
snot=2 # Initial value of Sigma=2
tnot=0.1 # Initial value of tau=0.1 Take 0.9  as tau and check result
t=0 # Initial Generation t= 0

H= 15#int(input("Enter the size of neighbourhood mating pool: ")) #Size of mating pool


population=np.array(x)  #Initial population

data=np.copy(population)     #Training data for SOM
data_K=np.copy(K)            #clusters corresponding to each training data


neuron_weight=np.copy(population) # Assigning Initial weight vector to neurons same as data
neuron_K = np.copy(K)        # clusters corresponding to each neuron weight
lattice=build_lattice(neuron_weight,k2)    # Lattice coordinates of SOM
feature=len(Idata[0])      #no. of features present in actual data

print "unique : ", np.unique(population)

while t<T:
    """---------------------SOM Training-------------------------------------"""
    print "Generation number : ", t
    for s in range(len(data)):
        s_value=np.copy(data[s])
        new_snot, new_tnot = update_training_parameters (T, len(data),t+1,s+1,snot,tnot) #Initial values, T=10(Constant), t=0,s=0,t=2,tnot=0.1
        winning_neuron = closest_neuron(s_value, data_K[s], neuron_weight, neuron_K,feature)
        neighbour_neuron=locate_neighbouring_neurons(winning_neuron, lattice, new_snot)
        neuron_weight=update_weights(neuron_weight, neuron_K, neighbour_neuron, new_tnot, s_value, data_K[s],feature)


    A=np.copy(population)  #archive copy of the population
    A_K=np.copy(K)         #archive copy of clusters of the population

    """-------------Performing one to one mapping of solutions to neurons weight------------"""
    neuron_weight, neuron_K = mapping(A, A_K, neuron_weight, neuron_K,feature)
    #print "Neuron weight after one to one mapping : ", neuron_weight

    t+=1             # Increment the generation
    pop_max_list = population.max(0)
    pop_min_list = population.min(0)

    for i in range(len(A)):
        MatingPool, flag = generate_matingPool(H, i, A[i], A_K[i], feature, neuron_weight, neuron_K, beta=0.7)  # Generate mating pool
        #print "Mating Pool and flag value for solution {0} is : ".format(i), MatingPool
        if flag==0:
            #print "No. of cluster before generating new solution : ", A_K[i]
            upd_sol, upd_sol_K=Generate(Idata, MatingPool, neuron_weight, H, A, i,A[i], A_K[i],feature,flag, max_choromosome_length, pop_max_list, pop_min_list, CR=0.8,F=0.8,mutation_prob=0.6,eta=20) #Generate new solution/Children
        else:
            #print "No. of cluster before generating new solution : ", A_K[i]
            upd_sol, upd_sol_K = Generate(Idata, MatingPool, neuron_weight, H, A, i, A[i], A_K[i], feature, flag, max_choromosome_length, pop_max_list, pop_min_list, CR=0.8, F=0.8, mutation_prob=0.6, eta=20)    # Generate new solution/Children

        nsol=upd_sol[: upd_sol_K*feature]  #without appended zero
        #print "new solution length : ", len(nsol)
        cluster_center = np.reshape(nsol, (-1, Idata.shape[1]))
        #print "NSOL and cluster value\n ", nsol
        # if   cluster_center.shape[0] == upd_sol_K:
        #     print "Cluster size of new solution : ",upd_sol_K
        cluster = KMeans(n_clusters=cluster_center.shape[0])
        cluster.fit(Idata)
        new_center=cluster.cluster_centers_
        new_label = cluster.predict(Idata)
        new_ss = silhouette_score(Idata, new_label)
        new_pbm = cal_pbm_index(cluster_center.shape[0], Idata, new_center, new_label)
        #print "pbm and sil score of new solution : ", new_pbm, new_ss
        #print "max pbm and sil score of old sol : ", max(PBM), max(sil_sco)
        if max(PBM)<new_pbm :
            print "solution good "
        print "=============================================="
        lat_updated_nsol=[]              #new solution after appending zero
        for j in range(len(new_center)):
            for k in range(len(Idata[0])):
                lat_updated_nsol.append(new_center[j][k])
    #     #print "current solution {0} is  : ".format(i), A[i]
    #     #print "updated sol for solution {0} : ".format(i), lat_updated_nsol[0:8]
        if not max_choromosome_length - len(lat_updated_nsol) == 0:
            extra_zero = max_choromosome_length - len(lat_updated_nsol)
            lat_updated_nsol.extend(0 for x in range(extra_zero))
        #print "Ini clusters : ", K
        U_population, objectives, self_population, zdt_definitions, plotter, problem, evolution, Final_label , K= Select(population, sil_sco, PBM, K, lat_updated_nsol, new_pbm, new_ss, upd_sol_K, t, self_pop, zdt_definitions, plotter, problem, evolution,Final_label, new_label)

        #print "New clusters : ", K
        #print "---------------------------"
        self_pop=self_population
        population= np.array(U_population)
        #print "Data",data
    print "---------------End of generation  {0}---------------".format(t)
    # Updation of SOM data and its corresponding K
    num_rows, num_cols = population.shape
    b = np.copy(data)            #SOM data
    b=b.tolist()
    data_K = []           #empty previous SOM data clusters
    data=[]               #empty previous SOM data
    count=0
    for j in range(num_rows):
        a=population[j].tolist()
        if a not in b:
            data.insert(count,a)    #adding new SOM data for Training
            data_K.insert(count,K[j])  #clusters corresponding to new data for SOM Training
        count+=1
    #data = np.array([])
    data=np.array(data)
    # file1.write("\nGeneration No. :  "+str(t))
    # file1.write("\nChromosome clusters : " + str(K))
    # #file1.write("\nPredicted labels : " +str(Final_label))
    # sl=[item[0] for item in objectives]
    # pb=[item[1] for item in objectives]
    # file1.write("\nSil score :  "+str(sl))
    # file1.write("\nPBM : "+str(pb))
    # file1.write("---------------------------------------------\n")
    # data=population
    # data_K=list(K)
    #if t==1:
    #break
    if not data.tolist():
         break  # If Training DataSet(TDS) is empty, no training takes place and training stops


"""========================================================================="""
for i in range(len(population)):
    file1.write("chromosome {0} : ".format(i)+ str(population[i]))
    file1.write("\n Objectives (Sil, PBM) : "+str(objectives[i]))
    file1.write("\nCluster size{0} :".format(i)+str(K[i])+'\n')
    file1.write("\nPredicted Label for {0}: ".format(i)+ str(Final_label[i]))
    file1.write("\n-------------------------------------------------------------------------------------------------------------------\n")

# print "Returned pop  : ",population
# print "Final Clusters : ", K
# print "Objectives  : ",objectives
# print "No. of generation run : ", t
# return_ss = [item[0] for item in objectives]
# return_pbm = [item[1] for item in objectives]
# print "PBM : ",return_pbm
# print "SS  : ",return_ss
#
# final_Dunn=[]
# for i in range(len(population)):
#     sol_feat=K[i]*feature
#     sol=population[i][  :sol_feat ]  #without appended zero
#     #print "new solution length : ", len(nsol)
#     cluster_center = np.reshape(sol, (-1, Idata.shape[1]))
#     dd=calculate_dunn_index(Idata, Final_label[i],n_cluster=K[i])
#     final_Dunn.insert(i,dd)
#
#
# index1= return_ss.index(max(return_ss))
# index2= return_pbm.index(max(return_pbm))
# index3=final_Dunn.index(max(final_Dunn))
#
# print "No. of generation run : ", t
# print "Max PBM and No of clusters detected automatically coressponding to maximum PBM : " , K[index2]," , ",max(return_pbm)
# print "Max Silhouette score  and No of clusters detected automatically coressponding to maximum Sil : " , K[index1],",", max(return_ss)
# print "Max Dunn index and No of clusters detected automatically  and coressponding PBM, Silh. score : " ,  final_Dunn[index3],",",K[index3],",", return_pbm[index3],",",return_ss[index3]
#
#
# file2.write("----------------------Final Result-----------------------------------------\n")
# file2.write("\n Max PBM and corresponding clusters : "+ str(max(return_pbm))+','+ str(K[index2]) +'\n')
# file2.write("\n Max Silh. score and corresponding clusters : "+ str(max(return_ss))+' , '+ str(K[index1]) +'\n')
# file2.write("\n Max Dunn index and corresponding clusters : "+ str(final_Dunn[index3])+','+ str(K[index3]) +'\n')
# file2.write("Final label of data for max PBM : " +str( Final_label[index2])+'\n')
# file2.write("Final label of data for max Sil score : " +str( Final_label[index1])+'\n')
# file2.write("Final label of data for max Dunn Index : " +str( Final_label[index3])+'\n')
#
#
# import matplotlib.pyplot as plt
# plt.plot(return_ss, return_pbm, 'bo' )
# #plt.grid(True)
# plt.grid(True, color='0.75', linestyle='-')
# plt.xlabel('Silhouette score')
# plt.ylabel('PBM index')
# #plt.axis([0, max(return_ss), 0, max(return_pbm)])
# plt.savefig('/home/naveen.pcs16/Experiments/Text_clustering_SMEA/SMEA_Text_auto/Output/PF_gensim_300d_run4.jpeg')
# plt.show()
#

