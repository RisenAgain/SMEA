import itertools

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import matplotlib.pyplot as plt
import pylab
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

class Plotter():
    def __init__(self, problem):
        self.directory = 'plots'
        self.problem = problem
        self._f1_lim = None
        self._f2_lim = None
        self._f3_lim = None

    def plot_population_best_front(self, population, generation_number):

        filename = "{}/generation{}.png".format(self.directory, str(generation_number))
        figure = plt.figure()
        axes = figure.add_subplot(111)
        colors = itertools.cycle([".r", ".b", ".c", ".m", ".y", ".k"])

        self.__create_directory_if_not_exists()
        for i in range (len(population.fronts)-1):
            computed_pareto_front = population.fronts[i]
            computed_f1 = map(lambda individual: individual.objectives[0], computed_pareto_front)
            computed_f2 = map(lambda individual: individual.objectives[1], computed_pareto_front)
            axes.plot(computed_f1, computed_f2, next(colors), label='Computed Pareto Front')
            xmin, xmax = axes.get_xlim()
            ymin, ymax = axes.get_ylim()

            axes.set_xticks(np.arange(xmin, xmax, (xmax-xmin)/50))
            axes.set_yticks(np.arange(ymin, ymax, (ymax-ymin)/50))
            axes.set_title('Computed Pareto front @ Generation {:}'.format(generation_number))
        plt.grid()


        plt.savefig(filename)
        plt.close(figure)


    def plot_x_y(self, x, y, x_label, y_label, title, filename):

        filename = "{}/{}.png".format(self.directory, filename)
        self.__create_directory_if_not_exists()
        figure = plt.figure()
        axes = figure.add_subplot(111)
        axes.plot(x, y, 'r')
        axes.set_xlabel(x_label)
        axes.set_ylabel(y_label)
        axes.set_title(title)
        #plt.savefig(filename)
        #plt.show()
        plt.close(figure)

    def __create_directory_if_not_exists(self):
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def __plot_front(self, front, generation_number):
        filename = "{}/generation{}.png".format(self.directory, str(generation_number))
        figure = plt.figure()
        axes = figure.add_subplot(111)

        computed_f1 = map(lambda individual: individual.objectives[0], front)
        computed_f2 = map(lambda individual: individual.objectives[1], front)

        axes.plot(computed_f1, computed_f2, 'r.', label='Computed Pareto Front')

        '''
        perfect_pareto_front_f1, perfect_pareto_front_f2 = self.problem.perfect_pareto_front()
        print "PPF",perfect_pareto_front_f1,"\n",perfect_pareto_front_f2
        axes.plot(perfect_pareto_front_f1, perfect_pareto_front_f2, 'b.', label='Perfect Pareto Front')

        if self._f1_lim is None:
            self._f1_lim = (min(0, min(perfect_pareto_front_f1)), max(computed_f1))
        if self._f2_lim is None:
            self._f2_lim = (min(0, min(perfect_pareto_front_f2)), max(computed_f2))

        axes.set_xlabel('$f1$')
        axes.set_ylabel('$f2$')

        axes.set_xlim(self._f1_lim[0], self._f1_lim[1])
        axes.set_ylim(self._f2_lim[0], self._f2_lim[1])
        axes.set_title('Computed Pareto front @ Generation {:}'.format(generation_number))
        plt.legend(loc='upper left')
        plt.show()
        plt.savefig(filename)
        plt.close(figure)
        '''

def plot_max_Sil_Score(Idata, n, Initial_Label, plot_label, sil_sco):
    n+=1
    import matplotlib.pyplot as plt
    if sil_sco is not None:
        max_score = max(sil_sco)
        maxIndex = sil_sco.index(max_score)
        Max_Label = Initial_Label[maxIndex]
    else:
        Max_Label=Initial_Label

    colors = np.array(["r", "b", "k", "y", "m", "g"])
    no_rows,no_cols=Idata.shape
    #no_cols=2
    if no_cols>2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        for i in range(len(Max_Label)):
            count = 0
            while not count == n:
                if Max_Label[i] == count:
                    ax.scatter(Idata[i, 0], Idata[i, 1], Idata[i, 2], c=colors[count])
                    break
                else:
                    count += 1
    else:
        figure = plt.figure()
        axes = figure.add_subplot(111)
        for i in range(len(Max_Label)):
            count = 0
            while not count == n:
                if Max_Label[i] == count:
                    axes.scatter(Idata[i, 0], Idata[i, 1], c=colors[count%len(colors)],s=10)
                    break
                else:
                    count += 1

    plt.savefig(plot_label)
    plt.show()


