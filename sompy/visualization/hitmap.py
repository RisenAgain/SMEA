from view import MatplotView
from matplotlib import pyplot as plt
import numpy as np


class HitMapView(MatplotView):

    def _set_labels(self, cents, ax, labels):
        for i, txt in enumerate(labels):
            ax.annotate(labels[i], (cents[i, 1], cents[i, 0]), size=10, va="center")
            print "text  : ", txt  , "center{0}[1], center{1}[0]".format(i,i)

    def show(self, som, data,no_clusters=9):
        codebook = getattr(som, 'cluster_labels', som.cluster(n_clusters=no_clusters))
        print "=======-------", codebook
        msz = som.codebook.mapsize

        self.prepare()
        #ax = self._fig.add_subplot(111)
        ax=self._fig.add_subplot(111)
        if data is not None:
            proj = som.project_data(data)             #will identify bmu_index
            print "projected data bmu index  : ", proj
            cents = som.bmu_ind_to_xy(proj)           #will map the bmu_index with (x,y) cordinate
            print "coordinate of bmu index according to data : ", cents
            self._set_labels(cents, ax, codebook[proj])

        else:
            cents = som.bmu_ind_to_xy(np.arange(0, msz[0]*msz[1]))
            print "coordinate of bmu index in serial : ", cents
            self._set_labels(cents, ax, codebook)

        plt.imshow(codebook.reshape(msz[0], msz[1])[::], alpha=.5)
        plt.show()

        return cents
