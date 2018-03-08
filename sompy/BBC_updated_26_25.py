

import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))



import numpy as np
from sklearn.metrics.cluster.supervised import adjusted_rand_score
from numpy import genfromtxt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from Evolutionary.Evoution1 import Mapsize, generate_matingPool,Generate,mapping,calculate_dunn_index
from nsga2.main import Select
from random import randint


run=input("Enter run No. : ")
file1=open('/home/naveen.pcs16/Experiments/Text_clustering_SMEA/SMEA_Text_auto/Output/BBC/BBC_tfidf_term/Output_BBC_tfidf_term_generation_details_run{0}.txt'.format(run),'w')
file2=open('/home/naveen.pcs16/Experiments/Text_clustering_SMEA/SMEA_Text_auto/Output/BBC/BBC_tfidf_term/Output_BBC_tfidf_terms_run{0}.txt'.format(run),'w')



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

MAX_ARI=[]    #store generation wise Max ARI
MAX_DUNN=[]   #store generation wise Max DUNN
GEN_COUNT=[]  #Generation No.


Idata = genfromtxt('/home/naveen.pcs16/Experiments/Text_clustering_SMEA/SMEA_Text_auto/data_set/BBC/BBC_tf_idf_and_count/BBC_tfidf_terms.csv', delimiter='\t' ,usecols=range(0,9159))
Idata_label=genfromtxt('/home/naveen.pcs16/Experiments/Text_clustering_SMEA/SMEA_Text_auto/data_set/BBC/BBC_tf_idf_and_count/BBC_tfidf_terms.csv', delimiter='\t' ,usecols=(9159))
print "first doc vector : ", Idata[0]
print "data shape : ", Idata.shape
print "Idata labels :", Idata_label
print "No of lablesl : ", len(Idata_label)


from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=50, n_iter=7, random_state=42)
Idata=svd.fit_transform(Idata)


print "new data shape : ", Idata.shape
max_cluster= int(math.floor(math.sqrt(len(Idata))))     #Max. cluster in population
print "Max. no. of cluster : ",max_cluster+1
Initial_Label=[]
max_choromosome_length=(max_cluster+1)*len(Idata[0])
print "Max. length of chromosome : ", max_choromosome_length
CH=input("enter No. of chromosome : ")

K=[]
ini_Dunn=[]
for i in range(1,CH+1):
    #print "----------------------------------------"
    counter+=1
    pop = []
    n=randint(2, max_cluster)
    K.insert(i,n)
    print "no. of cluster : ", n
    cluster = KMeans(n_clusters=n)
    cluster.fit(Idata)
    label = cluster.predict(Idata)
    centers = cluster.cluster_centers_
    #print "center is  : ", centers
    #print "labels : ", label
    #print "no. of labels :", len(label)
    a=centers.tolist()
    for j in range(len(a)):
        for k in range(len(Idata[0])):
            pop.append(a[j][k])
    if not max_choromosome_length-len(pop)== 0:
        extra_zero= max_choromosome_length-len(pop)
        pop.extend(0 for x in range(extra_zero))
    #print "center with appended zero : ", pop
    #print "length of center : ", len(pop)
    x.insert(i,pop)
    ss=silhouette_score(Idata, label)
    #di = calculate_dunn_index(Idata,label,n_cluster=i+1)
    pbm = cal_pbm_index(n, Idata, centers, label)
    sil_sco.insert(i, ss)
    PBM.insert(i,pbm)
    dn=calculate_dunn_index(Idata, label,n)
    ini_Dunn.insert(i,dn)
    Initial_Label.insert(i,label.tolist())
Final_label=list(Initial_Label)


index=ini_Dunn.index(max(ini_Dunn))
print "Chromosome clusers : ", K
print "Chromosomes PBM Index  : ",PBM
print "Chromosome SI  : ",sil_sco
print "Max PBM, Sil and Dunn   : ",max(PBM), ",",max(sil_sco),",", max(ini_Dunn)
print "clusters corresponding to Max PBM, Sil , Dunn:  ", K[PBM.index(max(PBM))], ", ",K[sil_sco.index(max(sil_sco))],", ", K[ini_Dunn.index(max(ini_Dunn))]
print "Max DUNN and corresponding cluster, PBM, Sil :  " , ini_Dunn[index],",",K[index], ", ", PBM[index], ",",sil_sco[index]


file2.write("\n Initial clusters : "+ str(K) + '\n' )
file2.write("Chromosome initial PBM index: "+ str(PBM) + '\n' )
file2.write("Chromosome initial Sil score : " +str(sil_sco)+ '\n')
file2.write("Chromosome initial Dunn : " +str(ini_Dunn)+'\n')
file2.write("Max PBM, Sil, Dunn : " + str(max(PBM)) +" , " + str(max(sil_sco)) + " , "+ str(max(ini_Dunn)))
file2.write("\nclusters corresponding to Max PBM, Sil, Dunn : "+ str(K[PBM.index(max(PBM))])+","+str(K[sil_sco.index(max(sil_sco))])+","+str(K[ini_Dunn.index(max(ini_Dunn))]))
file2.write("\nMax DUNN and corresponding cluster, PBM, Sil : " + str(ini_Dunn[index])+str(K[index]) + ", "+str(PBM[index])+","+str(sil_sco[index]) )
file2.write("\n============================================================================" +'\n')


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
T= 50#int(input("Enter no. of generation-  "))
H= 5#int(input("Enter the size of neighbourhood mating pool: ")) #Size of mating pool


population=np.array(x)  #Initial population

data=np.copy(population)     #Training data for SOM
data_K=np.copy(K)            #clusters corresponding to each training data

#final_label=[]

neuron_weight=np.copy(population) # Assigning Initial weight vector to neurons same as data
neuron_K=np.copy(K)        #clusters corresponding to each neuron weight
lattice=build_lattice(neuron_weight,k2)    # Lattice coordinates of SOM
feature=len(Idata[0])      #no. of features present in actual data



while t<T:
    #break
    #snot=2 # Initial value of Sigma=2
    #tnot=0.1 #Initial value of tau=0.1 Take 0.9  as tau and check result  #initial learning rate
    #TDS = [] #Empty the training dataset
    #SOM Training starts here up to length of data
    for s in range(len(data)):
        s_value=np.copy(data[s])
        new_snot, new_tnot = update_training_parameters (T, len(data),t+1,s+1,snot,tnot) #Initial values, T=10(Constant), t=0,s=0,snot=2,tnot=0.1
        #print "updated learning rate and sigma :  ",new_tnot, new_snot
        winning_neuron = closest_neuron(s_value, data_K[s], neuron_weight, neuron_K,feature)
        neighbour_neuron=locate_neighbouring_neurons(winning_neuron, lattice, new_snot)
        neuron_weight=update_weights(neuron_weight, neuron_K, neighbour_neuron, new_tnot, s_value, data_K[s],feature)
    #print "Updated neuron weight  :  \n ",neuron_weight

    A=np.copy(population)
    A_K=np.copy(K)
    neuron_weight, neuron_K = mapping(A, A_K, neuron_weight, neuron_K,feature)
    #print "Neuron weight after one to one mapping : ", neuron_weight
    t+=1 # Increment the generation
    pop_max_list=[]; pop_min_list=[]
    neuron_max_list=[]; neuron_min_list=[]
    for i in range(len(A)):
        nn=A[i]
        n = max(A[i])
        pop_max_list.insert(i, n)
        m = min(k for k in A[i] if k > 0 or k < 0)
        pop_min_list.insert(i, m)
    for i in range(len(neuron_weight)):
        n = max(neuron_weight[i])
        neuron_max_list.insert(i, n)
        m = min(k for k in neuron_weight[i] if k > 0 or k < 0)
        neuron_min_list.insert(i, m)
    #print pop_max_list, neuron_max_list
    #print pop_min_list, neuron_min_list
    pop_max=max(pop_max_list)
    pop_min=min(pop_min_list)
    neuron_max=max(neuron_max_list)
    neuron_min=min(neuron_min_list)

    for i in range(len(A)):
        #nsol=[]
        #print "current solution no. in generation {0} and its cluster :".format(t), i, ",", A_K[i]
        MatingPool,flag = generate_matingPool(H,i, A[i], A_K[i], feature, neuron_weight, neuron_K, beta=0.7)  # Generate mating pool
        #print "Mating Pool and flag value is : ", MatingPool
        if flag==0:
            upd_sol, upd_sol_K=Generate(Idata,MatingPool, neuron_weight, H, A, i,A[i], A_K[i],feature,flag,max_choromosome_length,neuron_max, neuron_min, CR=0.8,F=0.8,mutation_prob=0.6,eta=20) #Generate new solution/Children
        else:
            upd_sol, upd_sol_K = Generate(Idata, MatingPool, neuron_weight, H, A, i, A[i], A_K[i], feature, flag, max_choromosome_length, pop_max, pop_min, CR=0.8, F=0.8, mutation_prob=0.6, eta=20)  # Generate new solution/Children
        #print "Nsol generated  :   ", upd_sol
        nsol=upd_sol[  : upd_sol_K*feature]  #without appended zero
        #print "new solution length : ", len(nsol)
        cluster_center = np.reshape(nsol, (-1, Idata.shape[1]))
        #print "NSOL and cluster value\n ", nsol
        #print "Cluster size of new solution : ", cluster_center.shape[0]
        #cluster = KMeans(n_clusters=cluster_center.shape[0],init=cluster_center)
        cluster = KMeans(n_clusters=cluster_center.shape[0], init=cluster_center)
        cluster.fit(Idata)
        new_center=cluster.cluster_centers_
        new_label = cluster.predict(Idata)
        #print "New label type :", type(new_label)
        new_ss = silhouette_score(Idata, new_label)
        #new_dunn = calculate_dunn_index(Idata,label,n_cluster=cluster_center.shape[0])
        new_pbm = cal_pbm_index(cluster_center.shape[0], Idata, new_center, new_label)
        lat_updated_nsol=[]              #new solution after appending zero
        #print "new center is  : ", new_center
        #print "updated center length : ", len(new_center), type(new_center)
        for j in range(len(new_center)):
            for k in range(len(Idata[0])):
                lat_updated_nsol.append(new_center[j][k])
        #print "current solution {0} is  : ".format(i), A[i]
        #print "updated sol for solution {0} : ".format(i), lat_updated_nsol[0:8]
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


    # file1.write("\nPredicted labels : " +str(Final_label))
    sl = [item[0] for item in objectives]
    pb = [item[1] for item in objectives]

    ARI = []
    DUNN=[]
    for i in range(len(population)):
        ARI.insert(i, adjusted_rand_score(Idata_label, Final_label[i]))
        DUNN.insert(i, calculate_dunn_index( Idata, Final_label[i],K[i]))

    indexari = ARI.index(max(ARI))
    MAX_ARI.append(ARI[indexari])
    indexdunn=DUNN.index(max(DUNN))
    MAX_DUNN.append(DUNN[indexdunn])
    GEN_COUNT.append(t)

    file1.write("\nGeneration No. :  " + str(t))
    file1.write("\nChromosome clusters : " + str(K))
    file1.write("\nSil score :  " + str(sl))
    file1.write("\nPBM : " + str(pb))
    file1.write("\nARI : " + str(ARI))
    file1.write("\nDUNN: " + str(DUNN))
    file1.write("\n Max DUnn and Max ARI at index {0}, {1} : ".format(indexdunn,indexari)+str(DUNN[indexdunn])+" , "+str(ARI[indexari]))
    file1.write("\nPredicted Label for Max Dunn : " + str(Final_label[indexdunn]))
    file1.write("\nPredicted Label for Max ARI : "+str(Final_label[indexari]))
    file1.write("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++-\n")
    import matplotlib.pyplot as plt
    plt.plot(sl, pb, 'bo')
    # plt.grid(True)
    plt.grid(True, color='0.75', linestyle='-')
    plt.xlabel('Silhouette score')
    plt.ylabel('PBM index')
    # plt.axis([0, max(return_ss), 0, max(return_pbm)])
    name1 = 'PF_BBC_tfidf_term_run{0}_genration_{1}.jpeg'.format(run,t)
    plt.savefig('/home/naveen.pcs16/Experiments/Text_clustering_SMEA/SMEA_Text_auto/Output/BBC/BBC_tfidf_term/generation_wise_graph_run{0}/'.format(run)+name1)
    #plt.show()
    if not data.tolist():
        break  # If Training DataSet(TDS) is empty, no training takes place and training stops

print "Type of Final LIst : ", type(Final_label)
#Final_label=Final_label.tolist()
for i in range(len(population)):
    #file1.write("----------------------Fianl Results----------------------------------")
    file1.write("\nChromosome No. {0} : ".format(i)+ str(population[i]))
    file1.write("\nObjectives (Sil, PBM) : " + str(objectives[i]))
    file1.write("\nNo. of clusters :  " +str(K[i])+'\n')
    file1.write("\nPredicted Label : "+ str(Final_label[i]))
    file1.write("\n-------------------------------------------------------------------------------------------------------------------\n")

#print "Returned pop  : ",population
print "Final Clusters : ", K
print "Objectives  : ",objectives
print "No. of generation run : ", t
return_ss = [item[0] for item in objectives]
return_pbm = [item[1] for item in objectives]
print "PBM : ",return_pbm
print "SS  : ",return_ss


final_Dunn=[]
for i in range(len(population)):
    sol_feat=K[i]*feature
    sol=population[i][  :sol_feat ]  #without appended zero
    #print "new solution length : ", len(nsol)
    cluster_center = np.reshape(sol, (-1, Idata.shape[1]))
    dd=calculate_dunn_index( Idata, Final_label[i],K[i])
    final_Dunn.insert(i,dd)


index1= return_ss.index(max(return_ss))
index2= return_pbm.index(max(return_pbm))
index3=final_Dunn.index(max(final_Dunn))

print "No. of generation run : ", t
print "Max PBM and No of clusters detected automatically coresponding to maximum PBM : " , K[index2]," , ",max(return_pbm)
print "Max Silhouette score  and No of clusters detected automatically corresponding to maximum Sil : " , K[index1],",", max(return_ss)
from sklearn.metrics.cluster.supervised import adjusted_rand_score
adjust=adjusted_rand_score(Idata_label.tolist(), Final_label[index3])
print "Adjusted Rand score corresponding to max Dunn: ", adjust
print "Max Dunn index and No of clusters detected automatically  and coressponding PBM, Silh. score : " ,  final_Dunn[index3],",",K[index3],",", return_pbm[index3],",",return_ss[index3]

file2.write("----------------------Fianl Results----------------------------------")
file2.write("\n Max PBM and corresponding clusters : "+ str(return_pbm[index2])+','+ str(K[index2]) +'\n')
file2.write("\n Max Silh. score and corresponding clusters : "+ str(return_ss[index1])+' , '+ str(K[index1]) +'\n')
file2.write("\n Max Dunn index and corresponding clusters : "+ str(final_Dunn[index3])+','+ str(K[index3]) +'\n')
file2.write("\n ARI corresponding to max Dunn : "+ str(adjust) +'\n')
file2.write("Final label of data for max PBM : " +str( Final_label[index2].tolist())+'\n')
file2.write("Final label of data for max Sil score : " +str( Final_label[index1].tolist())+'\n')
file2.write("Final label of data for max Dunn Index : " +str( Final_label[index3].tolist())+'\n')
ari_max_index=MAX_ARI.index(max(MAX_ARI))
dunn_max_index=MAX_DUNN.index(max(MAX_DUNN))
file2.write("----------------------Max ARI and Max Dunn found@Generation details----------------------------------")
file2.write("\n Maximum ARI found at genartion {0}  is : ".format(GEN_COUNT[ari_max_index])+ str(MAX_ARI[ari_max_index]) +'\n')
file2.write("\n Maximum DUNN found at genartion {0}  is : ".format(GEN_COUNT[dunn_max_index])+ str(MAX_ARI[dunn_max_index]) +'\n')
print  "Maximum ARI at genartion {0} is: ".format(GEN_COUNT[ari_max_index]),MAX_ARI[ari_max_index]
print  "Maximum Dunn at genartion {0} is: ".format(GEN_COUNT[dunn_max_index]),MAX_DUNN[dunn_max_index]

return_ss=list(return_ss)
return_pbm=list(return_pbm)
#print type(return_pbm), type(return_ss)
#print return_pbm
import matplotlib.pyplot as plt
plt.plot(return_ss, return_pbm, 'bo' )
plt.grid(True, color='0.75', linestyle='-')
plt.xlabel('Silhouette score')
plt.ylabel('PBM index')
plt.savefig('/home/naveen.pcs16/Experiments/Text_clustering_SMEA/SMEA_Text_auto/Output/BBC/BBC_tfidf_term/PF_BBC_tfidf_terms_run{0}.jpeg'.format(run))
plt.show()



