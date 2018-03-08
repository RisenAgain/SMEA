from sklearn.metrics.cluster import v_measure_score
import numpy as np
from numpy import genfromtxt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster.supervised import adjusted_rand_score
from minkowski_score import cal_minkowski_score
from sklearn.preprocessing import normalize
#from plotter import plot_max_Sil_Score
import sompy
from Evolutionary.Evoution1 import Mapsize, generate_matingPool,Generate,mapping,calculate_dunn_index
from nsga2.main import Select
from normalization import NormalizatorFactory
from sklearn import cluster
from sklearn.metrics import normalized_mutual_info_score
from random import randint
from plotter import plot_max_Sil_Score
file=open('/home/acer/PycharmProjects/Text_clustering_SMEA/SMDEA_CLUST /SOMPY-master/Output/spherical_6_2_output.txt','w')
import math
from som_train import build_lattice, closest_neuron,locate_neighbouring_neurons,update_training_parameters,update_weights
from pbm_Index import cal_pbm_index


Iflag=0       #Flag
x = []        #store all center
PBM=[]        #List of ari against each solution in population
sil_sco=[]    #List of Silhoutte Score angainst each solution in population
counter=0
actual_label=[]
TDS=[0]
prev_ss=0
prev_dunn=0
data = genfromtxt('/home/acer/PycharmProjects/SOMBatch_Variable_latest/SMDEA_CLUST /SOMPY-master/data_set/spherical62.csv', delimiter='\t',usecols=(3))
for i in range(len(data)):
    actual_label.insert(i,data[i])
Idata = genfromtxt('/home/acer/PycharmProjects/SOMBatch_Variable_latest/SMDEA_CLUST /SOMPY-master/data_set/spherical62.csv', delimiter='\t',usecols=(1,2))
#print Idata
normalizer = NormalizatorFactory.build('var')
Idata=normalizer.normalize(Idata)
max_cluster= int(math.floor(math.sqrt(len(Idata))))   #Max. cluster in population
print "Max. no. of cluster : ",max_cluster+1
Initial_Label=[]
max_choromosome_length=(max_cluster+1)*len(Idata[0])
print "Max. length of chromosome : ", max_choromosome_length
CH=input("enter No. of chromosome : ")

K=[]
for i in range(1,CH+1):
    print "----------------------------------------"
    counter+=1
    pop = []
    n=randint(2, max_cluster)
    K.insert(i,n)
    print "no. of cluster : ", n
    cluster = KMeans(n_clusters=n,init='random',max_iter=1)
    cluster.fit(Idata)
    label = cluster.predict(Idata)
    centers = cluster.cluster_centers_
    print "center is  : ", centers
    print "labels : ", label
    print "no. of labels :", len(label)
    a=centers.tolist()

    for j in range(len(a)):
        for k in range(len(Idata[0])):
            pop.append(a[j][k])

    if not max_choromosome_length-len(pop)== 0:
        extra_zero= max_choromosome_length-len(pop)
        pop.extend(0 for x in range(extra_zero))
    print "center with appended zero : ", pop
    print "length of center : ", len(pop)
    x.insert(i,pop)
    ss=silhouette_score(Idata, label)
    #di = calculate_dunn_index(Idata,label,n_cluster=i+1)
    pbm = cal_pbm_index(n, Idata, centers, label)
    sil_sco.insert(i, ss)
    PBM.insert(i,pbm)
    Initial_Label.append(label.tolist())
Final_label=list(Initial_Label)
print "Chromosomes PBM Index  : ",PBM
print "Chromosome SI  : ",sil_sco
print "Predicted labels of chromosomes : ", Final_label
#x=[[1.0,2,3,4,0,0,0,0],[5,6,7,8,3,5,0,0],[3,1,2,5,6,4,5,2]]


#som_build=sompy.SOMFactory()
print "############################################"
k1,k2 = Mapsize(counter) # Get SOM Map size
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
T= 10#int(input("Enter no. of generation-  "))
H= 5#int(input("Enter the size of neighbourhood mating pool: ")) #Size of mating pool


population=np.array(x)  #Initial population

data=np.copy(population)     #Training data for SOM
data_K=np.copy(K)            #clusters corresponding to each training data

final_label=[]

neuron_weight=np.copy(data) # Assigning Initial weight vector to neurons same as data
neuron_K=np.copy(K)        #clusters corresponding to each neuron weight
lattice=build_lattice(neuron_weight,k2)    # Lattice coordinates of SOM
feature=len(Idata[0])      #no. of features present in actual data



while t<T:
    #snot=2 # Initial value of Sigma=2
    #tnot=0.1 #Initial value of tau=0.1 Take 0.9  as tau and check result  #initial learning rate
    TDS = [] #Empty the training dataset
    """SOM Training starts here up to length of data"""
    for s in range(len(data)):
        s_value=np.copy(data[s])
        new_snot, new_tnot = update_training_parameters (T, len(data),t+1,s+1,snot,tnot) #Initial values, T=10(Constant), t=0,s=0,snot=2,tnot=0.1
        print "updated learning rate and sigma :  ",new_tnot, new_snot
        winning_neuron = closest_neuron(s_value, data_K[s], neuron_weight, neuron_K,feature)
        neighbour_neuron=locate_neighbouring_neurons(winning_neuron, lattice, new_snot)
        neuron_weight=update_weights(neuron_weight, neuron_K, neighbour_neuron, new_tnot, s_value, data_K[s],feature)
    print "Updated neuron weight  :  \n ",neuron_weight
    A=np.copy(population)
    A_K=np.copy(K)
    neuron_weight,neuron_K = mapping(A, A_K, neuron_weight, neuron_K,feature)
    print "Neuron weight after one to one mapping : ", neuron_weight
    t+=1 # Increment the generation
    print "In generation ",t
    print "LenPOP,Len(mapp)",len(population),len(neuron_weight)
    print "actual K ", K
    print "neuron K : ", neuron_K

    for i in range(len(A)):
        #nsol=[]
        MatingPool,flag = generate_matingPool(H,i, A[i], A_K[i], feature, neuron_weight, neuron_K, beta=0.7)  # Generate mating pool
        upd_sol, upd_sol_K=Generate(Idata,MatingPool, neuron_weight, H, A, i,A[i], A_K[i],feature,flag,max_choromosome_length,CR=0.8,F=0.8,mutation_prob=0.6,eta=20) #Generate new solution/Children
        print "Nsol generated  :   ", upd_sol
        # for k in range (len(upd_sol)):
        #     if not upd_sol[k]==0:
        #         nsol.insert(k,upd_sol[k]) # If Generated solution(upd_sol) does not contain 0, keep that value in nsol.
        nsol=upd_sol[  : upd_sol_K*feature]  #without appended zero
        print "new solution length : ", len(nsol)
        cluster_center = np.reshape(nsol, (-1, Idata.shape[1]))
        print "NSOL and cluster value\n ", nsol
        print "Cluster size- ", cluster_center.shape[0]
        cluster = KMeans(n_clusters=cluster_center.shape[0],init=cluster_center)
        cluster.fit(Idata)
        new_center=cluster.cluster_centers_
        label = cluster.predict(Idata)
        new_ss = silhouette_score(Idata, label)
        #new_dunn = calculate_dunn_index(Idata,label,n_cluster=cluster_center.shape[0])
        new_pbm = cal_pbm_index(cluster_center.shape[0], Idata, new_center, label)
        print "New Silhouette score :  " ,new_ss
        print "New  PBM index : ", new_pbm
        #updated_center=new_center.tolist()
        lat_updated_nsol=[]              #new solution after appending zero
        print "new center is  : ", new_center
        print "updated center length : ", len(new_center), type(new_center)
        for j in range(len(new_center)):
            for k in range(len(Idata[0])):
                lat_updated_nsol.append(new_center[j][k])
        if not max_choromosome_length - len(lat_updated_nsol) == 0:
            extra_zero = max_choromosome_length - len(lat_updated_nsol)
            lat_updated_nsol.extend(0 for x in range(extra_zero))
        print "max ch length : ", max_choromosome_length
        print "updated center length : ", len(lat_updated_nsol)
        #U_population,objectives,self_population,zdt_definitions,plotter,problem,evolution,Final_label = Select(population,sil_sco,PBM,upd_sol,new_pbm,new_ss,t,self_pop,zdt_definitions,plotter,problem,evolution,Final_label,label)
        U_population, objectives, self_population, zdt_definitions, plotter, problem, evolution, Final_label , K= Select( population, sil_sco, PBM, K, lat_updated_nsol, new_pbm, new_ss, upd_sol_K, t, self_pop, zdt_definitions, plotter, problem, evolution,Final_label, label)
        self_pop=self_population
        population= np.array(U_population)
        print "Data",data
    data_K=[]
    num_rows, num_cols = population.shape
    b = data.tolist()
    for j in range(num_rows):
        a=population[j].tolist()
        if a not in b:
            TDS.insert(j,a)    #adding new SOM data for Training
            data_K.insert(j,K[j])  #clusters corresponding to new data for SOM Training
    data = np.array([])
    data=np.array(TDS)
    if not TDS:
        break  # If Training DataSet(TDS) is empty, no training takes place and training stops




for i in range(len(population)):
    file.write("chromosome {0} : ".format(i)+ str(population[i]))
    file.write("\n Objectives (Sil, PBM) : "+str(objectives[i]))
    file.write("\nCluster size{0} :".format(i)+str(K[i])+'\n')
    file.write("Predicted Label for {0}: ".format(i)+ str(Final_label[i]))
    ari=adjusted_rand_score(actual_label,Final_label[i])
    file.write("\n ARI Score {0} : ".format(i)+ str(ari))
    file.write("\nMinkowski score {0} : ".format(i)+str(cal_minkowski_score(actual_label, Final_label[i])))
    file.write("\n----------------------------------------------------------------------------------------------------------------------------\n")

print "Returned pop  : ",population
print "Objectives  : ",objectives
return_ss = [item[0] for item in objectives]
return_di = [item[1] for item in objectives]
print "PBM : ",return_di
print "SS  : ",return_ss
print "Clusters : ", K
print "Max PBM : ",max(return_di)
print "Max Silhouette score : ",max(return_ss)


all_ari=[]
all_mink=[]
for i in range (len(return_ss)):
    x=adjusted_rand_score(actual_label, Final_label[i])
    #print "ARI ",i, x
    all_ari.insert(i,x)
    all_mink.insert(i,cal_minkowski_score(actual_label, Final_label[i]))
print "No. of generation run : ", t
print "Max ARI : ",max(all_ari)
index= all_ari.index(max(all_ari))
print "No of clusters detected automatically : " , K[index]
print "Mink score coressponding to maximum ARI : ", all_mink[index]
file.write("\n Maximum ari for {0} chromosome : ".format(index)+ str(all_ari[index])+'\n')
file.write("Final label max ARI for {0} chromosome : ".format(index)+ str(Final_label[index])+'\n')
file.write("Actual label of data : " +str(actual_label))
file.write("\n No of clusters detected automatically : " +str(K[index]))
file.write("\n For chromosome : " +str(index))
