import sys
import os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from random import uniform
from numpy import genfromtxt
from sklearn.cluster import KMeans
from population_creation.population import pop_create
from population_creation.find_unique import find_unique_elements
import matplotlib.pyplot as plt

from Evolutionary.Evoution1 import Mapsize, generate_matingPool,Generate,mapping, calculate_dunn_index
from nsga2.main import Select
from normalization import NormalizatorFactory
from random import randint
from cluster_validity_indices.test_obj import form_bicluster_and_cal_objectives
from cluster_validity_indices.bi_evaluation import bicluster_evaluation

import math
from som_train import build_lattice, closest_neuron,locate_neighbouring_neurons,update_training_parameters,update_weights
import pdb

#np.set_printoptions(threshold='nan')
Iflag=0       #Flag
x_gene = []        #store all gene cluster centers (Store all gene chromosomes [Gchrom1, Gchrom2,..])
x_cond=[]     #store all condition cluster centers (Store all condition chromosomes [CChrom1, CChrom2,..])


TDS=[0]
prev_ss=0
prev_dunn=0

MSR=[]        #List of MSR against each solution in population
RV=[]    #List of row varianace angainst each solution in population
BI_size=[]
vol=[]
no_of_delta_biclusters=[]


#Idata_gene = genfromtxt('/home/acer/PycharmProjects/biclustering_SMEA/SMEA_Text_auto/data_set/Breast_Cancer_temp.csv', delimiter=',' , skip_header=1,usecols=range(2,11))
#Idata_gene = genfromtxt('/home/acer/PycharmProjects/biclustering_SMEA/SMEA_Text_auto/data_set/biclustering_data_Sets/colon_cancer.txt', delimiter='')
# Idata_gene = genfromtxt('/home/acer/PycharmProjects/biclustering_SMEA/SMEA_Text_auto/data_set/biclustering_data_Sets/leukemia-preproc.txt', delimiter='')
Idata_gene = genfromtxt('/home/chirag/btp_project/SMEA_Text_biclustering/data_set/biclustering_data_Sets/sms_features.txt', delimiter='')
#Idata_gene=Idata_gene[:50, :]
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
pdb.set_trace()

"""No. of features in gene and condition data"""
feature_gene=len(Idata_gene[0])      #no. of features present in gene data
feature_cond=len(Idata_condition[0])      #no. of features present in condition data

K_gene=[]      #store no. of clusters in gene solution (Store no. of clusters in the gene chromosome)
K_cond=[] #store no. of clusters in condition solution (Store no. of clusters in the condition chromosome)
K_population={}	#store no. of gene clusters and no. of condition clusters in a chromosome in population K_population[i] = [K_gene[i],K_cond[i]].

Initial_Label_gene=[]       #store label of gene data present in gene solution (Store cluster no.(label) of gene clusters in the gene chromosome)
Initial_Label_cond=[]   #store label of condition data present in condition solution (Store cluster no.(label) of condition clusters in the condition chromosome)
Initial_Label_population={}

CH=input("enter No. of chromosome for gene clusters solution: ")
print "Start Forming Gene cluster solution"

for i in range(1,CH+1):
    """--------------------------Create gene clusters population------------------------"""

    n=randint(2, max_cluster_gene)
    K_gene.insert(i,n)
    cluster = KMeans(n_clusters=n, init='random', max_iter=1, n_init=1)
    cluster.fit(Idata_gene)
    label = cluster.labels_
    centers = cluster.cluster_centers_
    centers = np.reshape(centers, n*feature_gene)
    z = int(max_choromosome_length_gene - (n* feature_gene))  ##Calculate number of zeros to append(Maximum cluster dimension - Current cluster dimension)
    centers = np.append(centers,np.zeros(z))
    # for j in range(z):
    #     centers = np.append(centers, [0])
    Initial_Label_gene.insert(i, label.tolist())
    x_gene.insert(i,centers)


CH_cond = input("enter No. of chromosome for condition clusters solution:  ")
print "Start Forming condition cluster solution"
for i in range(1,CH_cond+1):
    """---------------------Create condition clusters population-----------------"""
    n1 = randint(2, max_cluster_cond)
    K_cond.insert(i, n1)
    cluster1 = KMeans(n_clusters=n1, init='random', max_iter=1, n_init=1)
    cluster1.fit(Idata_condition)
    label1 = cluster1.labels_
    centers1 = cluster1.cluster_centers_
    centers1 = np.reshape(centers1, n1 * feature_cond)
    z = int(max_choromosome_length_cond - (n1 * feature_cond))  ##Calculate number of zeros to append(Maximum cluster dimension - Current cluster dimension)
    centers1 = np.append(centers1,np.zeros(z))
    # for k in range(z):
    #     centers1 = np.append(centers1, [0])
    Initial_Label_cond.insert(i, label1.tolist())
    x_cond.insert(i, centers1)

print "no. of gene solutions : ", len(x_gene)
print "no. of condition solutions : ", len(x_cond)

population=[]
counter=0

print "Gene clusters in gene chromosomes :", K_gene
print "Condition clusters in condition chromosomes :",  K_cond
print "Start Forming population"
for i in range(len(x_gene)):
    for j in range(len(x_cond)):
        new_center = pop_create(x_gene[i],  x_cond[j])
        population.insert(counter, new_center)
        K_population[counter]=[K_gene[i], K_cond[j]]
        Initial_Label_population[counter]=[Initial_Label_gene[i], Initial_Label_cond[j]]
        delta_biclusters, min_msr, delta_msr, max_rv, delta_rv, max_bisize, delta_BIsize, min_biindex, delta_Biindex, max_vol, delta_volume = form_bicluster_and_cal_objectives(K_gene[i], Idata_gene, Initial_Label_gene[i], K_cond[j], Idata_condition, Initial_Label_cond[j],MSR_threshold=1200)
        MSR.insert(counter,delta_msr)      #should be minimum
        RV.insert(counter,delta_rv)   #should be minimum (1/(1+rv))
        BI_size.insert(counter, delta_BIsize)  # should be minimum  (1/BIsize)
        no_of_delta_biclusters.append(delta_biclusters)
        vol.append(delta_volume)
        counter += 1
        #print "++++++++++++++++++++++++++++++++++++++++++++++++++++++"

print "No. of solutions in the population :", counter
print "Chromosome gene clusters :", K_gene
print "Chromosome condition clusters :",  K_cond
print "Chromosomes (delta)MSR  :", MSR
print "Chromosome (delta)RV: ", RV
print "Chromosome (delta)BI size: ", BI_size
print "Chromosome (delta)volume : ", vol
print "delta biclusters :", no_of_delta_biclusters
print "Min MSR and corresponding RV, BI_size  (MSR,RV,BI_SIZE): ", min(MSR), ",", RV[MSR.index(min(MSR))],",",BI_size[MSR.index(min(MSR))]
print "Max RV and corresponding MSR, Bi size  (MSR,RV,BI_SIZE):",  MSR[RV.index(max(RV))],",", max(RV), ",",BI_size[RV.index(max(RV))]
print "Max BI size and corresponding RV, MSR  (MSR,RV,BI_SIZE):", MSR[BI_size.index(max(BI_size))],",",RV[BI_size.index(max(BI_size))],",", max(BI_size)

print "Maximum length of chromosome : ", len(population[0])
T=int(input("Enter no. of generation-  "))
k1_gene,k2_gene = Mapsize(CH)                  # Get SOM Map size for gene solutions
k1_cond,k2_cond = Mapsize(CH_cond)                  # Get SOM Map size  for condition solutions
print "SOM Map size for gene solutions : ", k1_gene, k2_gene
print "SOM Map size for condition solutions : ", k1_cond, k2_cond
#som = None
#TDS = [0]
self_pop = None
zdt_definitions = None
plotter = None
problem = None
evolution = None

"""SOM parameters"""
snot_gene=float(k1_gene/2)   # Initial value of Sigma=2 for gene solutions
snot_cond=float(k1_cond/2)   # Initial value of Sigma=2 for condition solutions
tnot=0.1  # Initial value of tau=0.1 Take 0.9  as tau and check result
t=0   # Initial Generation t= 0
H= 5  # int(input("Enter the size of neighbourhood mating pool: ")) #Size of mating pool

"""Initial population for gene and condition clusters"""
population_gene=np.array(x_gene)  #Initial gene population
population_cond=np.array(x_cond)  #Initial condition population
population=np.array(population)

print "Size of population : ", len(population)

"""Making cluster labels as final labels of solutions in the population for gene and condition clusters"""
Final_label_gene=list(Initial_Label_gene)
Final_label_cond=list(Initial_Label_cond)
Final_label_population=dict.copy(Initial_Label_population)


"""Initialize gene and condition training data and corresponding clusters (gene and conditions) for different SOM """
SOM_data_gene=np.copy(population_gene)     #Training gene data for SOM
SOM_data_K_gene=np.copy(K_gene)            #no. of gene clusters corresponding to each training data
SOM_data_cond=np.copy(population_cond)     #Training condition data for SOM
SOM_data_K_cond=np.copy(K_cond)            #no. of condition clusters corresponding to each training data


"""Initialize neurons weight vector using gene and condition population for different SOM"""
neuron_weight_gene=np.copy(population_gene) # Assigning Initial weight vector to neurons same as data
neuron_K_gene = np.copy(K_gene)        # clusters corresponding to each neuron weight
neuron_weight_cond=np.copy(population_cond) # Assigning Initial weight vector to neurons same as data
neuron_K_cond = np.copy(K_cond)        # clusters corresponding to each neuron weight

lattice_gene=build_lattice(neuron_weight_gene,k2_gene)    # Lattice coordinates of SOM (for gene)
lattice_cond=build_lattice(neuron_weight_cond,k2_cond)    # Lattice coordinates of SOM (for condition)

print "gene latticle : ", lattice_gene
print "cond lattice:", lattice_cond

while t<T:
    print "Generation no. {0}".format(t)
    """-----------------------------SOM Training using gene solutions-------------------------------------"""
    for s in range(len(SOM_data_gene)):
        #print "SOM Training for gene data "
        s_value=np.copy(SOM_data_gene[s])
        new_snot, new_tnot = update_training_parameters (T, len(SOM_data_gene),t+1,s,snot_gene,tnot) #Initial values, T=10(Constant), t=0,s=0,snot=2,tnot=0.1
        winning_neuron = closest_neuron(s_value, SOM_data_K_gene[s], neuron_weight_gene, neuron_K_gene,feature_gene)
        neighbour_neuron=locate_neighbouring_neurons(winning_neuron, lattice_gene, new_snot)
        print("\n\n Neighbour_neurons:\n",neighbour_neuron,"\n\n")
        neuron_weight_gene=update_weights(neuron_weight_gene, neuron_K_gene, neighbour_neuron, new_tnot, s_value, SOM_data_K_gene[s],feature_gene)
    #print "Updated neuron weight  after training using gene solutions:  \n ",neuron_weight_gene

    """-----------------------------SOM Training using condition solutions-------------------------------------"""
    for s in range(len(SOM_data_cond)):
        #print "SOM Training for condition data "
        s_value1 = np.copy(SOM_data_cond[s])
        new_snot1, new_tnot1 = update_training_parameters(T, len(SOM_data_cond), t + 1, s, snot_cond, tnot)  # Initial values, T=10(Constant), t=0,s=0,snot=2,tnot=0.1
        winning_neuron1 = closest_neuron(s_value1, SOM_data_K_cond[s], neuron_weight_cond, neuron_K_cond, feature_cond)
        neighbour_neuron1 = locate_neighbouring_neurons(winning_neuron1, lattice_cond, new_snot1)
        neuron_weight_cond = update_weights(neuron_weight_cond, neuron_K_cond, neighbour_neuron1, new_tnot1, s_value1, SOM_data_K_cond[s], feature_cond)
    #print "Updated neuron weight after training using condition solutions: :  \n ", neuron_weight_cond


    """Making archive copy of  solutions in the population and their corresponding clusters"""
    A_population = np.copy(population)                            #archive copy of the gene population
    A_K_population =dict.copy(K_population)                           #archive copy of clusters of the gene population
    A_Final_label_population=dict.copy(Final_label_population)     #archive (dictionary) copy of cluster Labels  of the solutions in population


    """Making archive copy of gene and condition solutions in the population and their corresponding clusters"""
    pop_gene_data=(np.copy(population[:, :len(population_gene[0])])).tolist()
    pop_gene_K=[K_population[j][0] for j in range(len(K_population))]
    pop_cond_data=(np.copy(population[:, len(population_gene[0]) : (len(population_gene[0]) + len(population_cond[0]))])).tolist()
    pop_cond_K=[K_population[j][1] for j in range(len(K_population))]

    A_gene, A_gene_K=find_unique_elements(pop_gene_data, pop_gene_K, CH )  #find unique gene solution in population
    A_cond, A_cond_K = find_unique_elements(pop_cond_data, pop_cond_K, CH_cond)    #find unique condition solution in population
    pop_gene_data=np.asarray(pop_gene_data)
    pop_cond_data=np.asarray(pop_cond_data)

    """-------------Performing one to one mapping of solutions to neurons weight------------"""
    neuron_weight_gene, neuron_K_gene = mapping(A_gene, A_gene_K, neuron_weight_gene, neuron_K_gene, feature_gene)
    neuron_weight_cond, neuron_K_cond= mapping(A_cond, A_cond_K, neuron_weight_cond, neuron_K_cond, feature_cond)


    pop_gene_max_list = population[:, :len(population_gene[0])].max(0)
    pop_gene_min_list = population[:, :len(population_gene[0])].min(0)

    pop_cond_max_list = population[:, len(population_gene[0]) : (len(population_gene[0]) + len(population_cond[0]))].max(0)
    pop_cond_min_list = population[:, len(population_gene[0]) : (len(population_gene[0]) + len(population_cond[0]))].min(0)

    for i in range(len(A_population)):
        #print "SOLUTION Number :  {0}".format(i)
        """------------------------solution generation for gene solution------------"""
        temp1_gene= A_population[i][ :len(population_gene[0])]
        temp1_gene_K=A_K_population[i][0]
        MatingPool_gene, flag_gene = generate_matingPool(H, i,temp1_gene, temp1_gene_K, feature_gene, neuron_weight_gene, neuron_K_gene, len(pop_gene_data), beta=0.7)
        if flag_gene == 0:
                upd_sol, upd_sol_K = Generate(Idata_gene, MatingPool_gene, neuron_weight_gene, H, pop_gene_data, i, temp1_gene, temp1_gene_K, feature_gene, flag_gene, max_choromosome_length_gene, pop_gene_max_list, pop_gene_min_list, CR=0.8, F=0.8, mutation_prob=0.6, eta=20)  # Generate new solution/Children
        else:
                upd_sol, upd_sol_K = Generate(Idata_gene, MatingPool_gene, neuron_weight_gene, H, pop_gene_data, i, temp1_gene, temp1_gene_K, feature_gene, flag_gene, max_choromosome_length_gene, pop_gene_max_list, pop_gene_min_list, CR=0.8, F=0.8, mutation_prob=0.6, eta=20)  # Generate new solution/Children
        nsol = upd_sol[: upd_sol_K * feature_gene]  # without appended zero
        cluster_center = np.reshape(nsol, (-1, Idata_gene.shape[1]))
        cluster = KMeans(n_clusters=cluster_center.shape[0], init=cluster_center)
        cluster.fit(Idata_gene)
        new_center = cluster.cluster_centers_
        new_label = cluster.labels_
        lat_updated_nsol = []  # new solution after appending zero
        for j in range(len(new_center)):
            for k in range(len(Idata_gene[0])):
                lat_updated_nsol.append(new_center[j][k])
        if not max_choromosome_length_gene - len(lat_updated_nsol) == 0:
            extra_zero = max_choromosome_length_gene - len(lat_updated_nsol)
            lat_updated_nsol.extend(0 for x in range(extra_zero))

        """------------------------solution generation for condition solution------------"""
        temp1_cond = A_population[i][len(population_gene[0]) : (len(population_gene[0]) + len(population_cond[0]))]
        temp1_cond_K=A_K_population[i][1]
        MatingPool_cond, flag_cond = generate_matingPool(H, i, temp1_cond, temp1_cond_K, feature_cond, neuron_weight_cond, neuron_K_cond, len(pop_cond_data), beta=0.8)
        if flag_cond == 0:
            upd_sol_cond, upd_sol_K_cond= Generate(Idata_condition, MatingPool_cond, neuron_weight_cond, H, pop_cond_data, i, temp1_cond, temp1_cond_K, feature_cond, flag_cond, max_choromosome_length_cond, pop_cond_max_list, pop_cond_min_list, CR=0.8, F=0.8, mutation_prob=0.6, eta=20)  # Generate new solution/Children
        else:
            upd_sol_cond, upd_sol_K_cond = Generate(Idata_condition, MatingPool_cond, neuron_weight_cond, H, pop_cond_data, i, temp1_cond, temp1_cond_K, feature_cond, flag_cond, max_choromosome_length_cond, pop_cond_max_list, pop_cond_min_list, CR=0.8, F=0.8, mutation_prob=0.6, eta=20)  # Generate new solution/Children
        #print "updated generated sol:", upd_sol, len(upd_sol)
        nsol1 = upd_sol_cond[: upd_sol_K_cond * feature_cond]  # without appended zero
        #print "extract fea from gene sol",nsol, len(nsol)
        cluster_center1 = np.reshape(nsol1, (-1, Idata_condition.shape[1]))
        cluster = KMeans(n_clusters=cluster_center1.shape[0], init=cluster_center1)
        cluster.fit(Idata_condition)
        new_center1 = cluster.cluster_centers_
        new_label1 = cluster.labels_
        lat_updated_nsol1 = []  # new solution after appending zero
        for j in range(len(new_center1)):
            for k in range(len(Idata_condition[0])):
                lat_updated_nsol1.append(new_center1[j][k])
        if not max_choromosome_length_cond - len(lat_updated_nsol1) == 0:
            extra_zero = max_choromosome_length_cond - len(lat_updated_nsol1)
            lat_updated_nsol1.extend(0 for x in range(extra_zero))

        Bicluster, min_MSR, new_MSR_avg, max_RV, new_RVavg, max_new_BIsize, new_BIsize, min_new_BI_index, new_BI_index, max_new_vol, avg_vol = form_bicluster_and_cal_objectives(upd_sol_K, Idata_gene, new_label, upd_sol_K_cond, Idata_condition, new_label1,MSR_threshold=1200)

        lat_updated_nsol.extend(lat_updated_nsol1)         #new solution(list) with new gene and comdition solution (merge)
        lat_updated_nsol_K = [upd_sol_K, upd_sol_K_cond]   #list of clusters corresponding to new gene and condition solution
        lat_updated_nsol_label = [new_label, new_label1]     #list of clusters labels corresponding to new gene and condition solution
        U_population, U_objectives, self_population, zdt_definitions, plotter, problem, evolution, Final_label_population, K_population = Select(population, MSR, RV, BI_size, K_population, lat_updated_nsol, new_MSR_avg, new_RVavg, new_BIsize,lat_updated_nsol_K, t, self_pop, zdt_definitions, plotter, problem, evolution, Final_label_population, lat_updated_nsol_label)

        self_pop = self_population
        population= np.array(U_population)
        U_objectives=np.asarray(U_objectives)
        MSR = [item[0] for item in U_objectives]
        RV = [item[1] for item in U_objectives]
        BI_size= [item[2] for item in U_objectives]


    """---------New gene data for  SOM Training--------"""
    b = np.copy(SOM_data_gene)  #SOM data
    b = b.tolist()
    b1=np.copy(population[:, :len(population_gene[0])])
    SOM_data_K_gene = []  # empty previous SOM data clusters
    SOM_data_gene= []  # empty previous SOM data
    count = 0
    for j in range(len(b1)):
        a = b1[j].tolist()
        if a not in b:
            SOM_data_gene.insert(count, a)  #adding new SOM data for Training
            SOM_data_K_gene.insert(count, K_population[j][0])  # clusters corresponding to new data for SOM Training
        count += 1
    SOM_data_gene=np.asarray(SOM_data_gene)

    """New condition data for  SOM Training"""
    b11 = np.copy(SOM_data_cond)  # SOM data
    b11 = b11.tolist()
    b12 = np.copy(population[:, len(population_gene[0]) : (len(population_gene[0]) + len(population_cond[0]))])
    SOM_data_K_cond = []  # empty previous SOM data clusters
    SOM_data_cond = []  # empty previous SOM data
    count = 0
    for j in range(len(b12)):
        a11 = b12[j].tolist()
        if a11 not in b11:
            SOM_data_cond.insert(count, a11)  # adding new SOM data for Training
            SOM_data_K_cond.insert(count, K_population[j][1])  # clusters corresponding to new data for SOM Training
        count += 1
    SOM_data_cond=np.asarray(SOM_data_cond)
    #print "length of new SOM data for gene and condition : ", len(SOM_data_gene),",", len(SOM_data_cond)
    print "-------------------Generation {0}th Completed".format(t)

    MSR = [item[0] for item in U_objectives]
    RV = [item[1] for item in U_objectives]
    BIsize= [item[2] for item in U_objectives]
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('MSR')
    ax.set_ylabel('RV')
    ax.set_zlabel('B_Size')
    ax.grid(True)
    ax.scatter(MSR, RV, BIsize,  zdir='z', s=20, color='b', marker='.')
    name1 = 'PF_leukemia_genration_{0}.jpeg'.format(t)
    plt.savefig('/home/chirag/btp_project/SMEA_Text_biclustering/pareto_front_plots/sms/run1/'+ name1)
    t += 1  # Increment the generation
    #break

BI_index=[]
BI_MSR=[]
BI_RV=[]
BI_SIZE=[]
BI_volume=[]

BI_min_index=[]
BI_min_MSR=[]
BI_max_RV=[]
BI_max_SIZE=[]
BI_max_volume=[]

file1=open('/home/chirag/btp_project/SMEA_Text_biclustering/Output/sms/output_sms_run1.txt','w')
file1.write("\nSize of population = "+str(len(x_gene))+"*"+str(len(x_cond))+"="+str(len(population)))
file1.write("\nMaximum generation: "+str(T))
file1.write("\nMSR threshold: "+str(1200))
file1.write("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
for i in range(len(population)):
    file1.write("\nChromosome No. {0} : ".format(i)+ str(population[i].tolist()))
    file1.write("\nObjectives (MSR, RV, BI_size) : " + str(U_objectives[i]))
    file1.write("\nNo. of clusters :  " +str(K_population[i]))
    file1.write("\nPredicted Label for gene: "+ str(Final_label_population[i][0]))
    file1.write("\nPredicted Label for condition: " + str(Final_label_population[i][1]))
    file1.write("\n-----------------------------------------------------------------------------------------------")
    #bicluster, min_msr, msr, max_rv, rv, max_bisize, bisize, min_biindex, biindex, max_volume, volume=form_bicluster_and_cal_objectives(K_population[i][0], Idata_gene, Final_label_population[i][0], K_population[i][1], Idata_condition, Final_label_population[i][1], MSR_threshold=500)
    bicluster, min_msr, msr, max_rv, rv, max_bisize, bisize, min_biindex, biindex, max_volume, volume =  bicluster_evaluation(K_population[i][0], Idata_gene, Final_label_population[i][0], K_population[i][1], Idata_condition,  Final_label_population[i][1], MSR_threshold=1200)

    BI_MSR.insert(i,msr)
    BI_SIZE.insert(i,bisize)
    BI_RV.insert(i,rv)
    BI_index.insert(i, biindex)
    BI_volume.insert(i,  volume)

    BI_min_MSR.insert(i,min_msr)
    BI_max_RV.insert(i, max_rv)
    BI_max_SIZE.insert(i, max_bisize)
    BI_min_index.insert(i,min_biindex)
    BI_max_volume.insert(i, max_volume)
    file1.write("\nAvg. MSR, RV, Bicluster size, BI Index, BI volume : " + str(msr)+ ","+str(rv)+ ","+str(bisize)+","+str(biindex)+","+str(volume))
    file1.write("\nMin. MSR, Max RV, Max. Bicluster size, Min. BI Index, Max volume : " + str(min_msr) + "," + str(max_rv) + "," + str(max_bisize) + "," + str(min_biindex)+","+str(max_volume))
    file1.write("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")


file1.write("\n=====================Final evaluation nased on **AVERAGE**  BI index===================================")
index=BI_index.index(min(BI_index))
file1.write("\nAvg. Lower BI-index found for {0}th chromosome: ".format(index)+str(BI_index[index]))
file1.write("\nCorresponding Avg MSR: "+str(BI_MSR[index]))
file1.write("\nCorresponding Avg. RV: "+str(BI_RV[index]))
file1.write("\nCorresponding Avg. BI size: "+str(BI_SIZE[index]))
file1.write("\nCorresponding Avg. volume: "+str(BI_volume[index]))
print "\n=========Final evaluation based on Avg. minimum BI index================="
print "Lower BI-index found for {0}th chromosome: ".format(index), BI_index[index]
print "Corresponding avg. MSR: ", BI_MSR[index]
print "Corresponding avg. RV: ",BI_RV[index]
print "Corresponding avg. BI-size : ", BI_SIZE[index]
print "Corresponding Avg. volume: ", BI_volume[index]



file1.write("\n=====================Final evaluation nased on **BEST** BI index ===================================")
index=BI_min_index.index(min(BI_min_index))
file1.write("\nMInimum BI-index found for {0}th chromosome: ".format(index)+str(BI_min_index[index]))
file1.write("\nCorresponding min MSR: "+str(BI_min_MSR[index]))
file1.write("\nCorresponding Max RV: "+str(BI_max_RV[index]))
file1.write("\nCorresponding Max BI size: "+str(BI_max_SIZE[index]))
file1.write("\nCorresponding max. volume: "+str(BI_max_volume[index]))
print "\n=========Final evaluation nased on minimum BI index================="
print "Min BI-index found for {0}th chromosome: ".format(index), BI_min_index[index]
print "Corresponding min. MSR: ", BI_min_MSR[index]
print "Corresponding Max. RV: ",BI_max_RV[index]
print "Corresponding Max. BI-size : ", BI_max_SIZE[index]
print "Corresponding Max. volume: ", BI_max_volume[index]


file1.write("\n\n=====================Final evaluation based on **AVERAGE** minimum MSR ===================================")
index1=BI_MSR.index(min(BI_MSR))
file1.write("\nAvg. Lower MSR found for {0}th chromosome: ".format(index1)+str(BI_MSR[index1]))
file1.write("\nCorresponding avg. BI-index: "+str(BI_index[index1]))
file1.write("\nCorresponding avg. RV: "+str(BI_RV[index1]))
file1.write("\nCorresponding avg.  BI size: "+str(BI_SIZE[index1]))
file1.write("\nCorresponding avg.  volume: "+str(BI_volume[index1]))
print "=====================Final evaluation based on avg. minimum MSR==============="
print "Avg. Lower MSR found for {0}th chromosome: ".format(index1), BI_MSR[index1]
print "Corresponding min. BI-index: ", BI_index[index1]
print "Corresponding max. RV: ",BI_RV[index1]
print "Corresponding max. BI-size : ",BI_SIZE[index1]
print "\nCorresponding max.  volume: ", BI_volume[index1]


file1.write("\n\n=====================Final evaluation based on **BEST** minimum MSR  ===================================")
index1=BI_min_MSR.index(min(BI_min_MSR))
file1.write("\nMinimum MSR found for {0}th chromosome: ".format(index1)+str(BI_min_MSR[index1]))
file1.write("\nCorresponding min. BI-index: "+str(BI_min_index[index1]))
file1.write("\nCorresponding max. RV: "+str(BI_max_RV[index1]))
file1.write("\nCorresponding max.  BI size: "+str(BI_max_SIZE[index1]))
file1.write("\nCorresponding max.  volume: "+str(BI_max_volume[index1]))
print "=====================Final evaluation based on avg. minimum MSR======================"
print "Minimum MSR found for {0}th chromosome: ".format(index1), BI_min_MSR[index1]
print "Corresponding min. BI-index: ", BI_min_index[index1]
print "Corresponding max. RV: ",BI_max_RV[index1]
print "Corresponding max. BI-size : ",BI_max_SIZE[index1]
print "Corresponding max.  volume: ", BI_max_volume[index1]


