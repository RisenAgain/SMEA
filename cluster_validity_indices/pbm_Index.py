import numpy as np
import itertools
from scipy.spatial import distance
def cal_Db(centers):
    distance_list = []
    for a, b in itertools.combinations(centers, 2):
        #print "a,b  : ", a,b
        d1 = distance.euclidean(a, b)
        #print "distance : ", d1
        #print "----------------------"
        distance_list.append(d1)
    Db = max(distance_list)
    return Db

def cal_Ew(Idata, label, centers):
    Ew=0
    for i in range(len(label)):
        Ew+= distance.euclidean(Idata[i], centers[label[i]])
    return Ew


def cal_Et(Idata ):
    Et=0
    barycenter=np.mean(Idata)
    #print "means of Idata : ", barycenterprint "oooo"
    for i in Idata:
        d = distance.euclidean(i,barycenter)
        Et+=d
    return Et

def cal_pbm_index(K, Idata, obtained_centers, obtained_label):
    Db=cal_Db(obtained_centers)
    Ew=cal_Ew(Idata, obtained_label, obtained_centers)
    Et=cal_Et(Idata)
    #print Db,Ew, Et
    x=(1/float(K))*(Et/Ew)*((Db))
    #print x
    pbm_index=x*x
    return pbm_index

