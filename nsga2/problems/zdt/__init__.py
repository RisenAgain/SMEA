"""Module with definition of ZDT problem interface"""
from nsga2.individual import Individual
from nsga2.problems import Problem
import random
import inspect
import functools

class ZDT(Problem):

    def __init__(self, zdt_definitions):
        self.zdt_definitions = zdt_definitions
        self.max_objectives = [None, None,None]
        self.min_objectives = [None, None, None]
        self.problem_type = None
        self.n = None

    def __dominates(self, individual2, individual1):
        worse_than_other = (self.zdt_definitions.f1(individual1) >= self.zdt_definitions.f1(individual2) ) and (self.zdt_definitions.f2(individual1) <= self.zdt_definitions.f2(individual2)) and (self.zdt_definitions.f3(individual1) <= self.zdt_definitions.f3(individual2))
        better_than_other = self.zdt_definitions.f1(individual1) > self.zdt_definitions.f1(individual2) or self.zdt_definitions.f2(individual1) < self.zdt_definitions.f2(individual2) or self.zdt_definitions.f3(individual1) < self.zdt_definitions.f3(individual2)
        return worse_than_other and better_than_other

    def generateIndividual(self,new_msr,new_rv, new_bisize, nsol_K, nsol_label, nsol):
        individual = Individual()
        individual.features = []
        individual.ob1 = new_msr
        individual.ob2 = new_rv
        individual.ob3 = new_bisize
        individual.K = nsol_K
        individual.label= nsol_label
        for i in range(len(nsol)):
            individual.features.append(nsol[i])
        individual.dominates = functools.partial(self.__dominates, individual1=individual)
        self.calculate_objectives(individual)
        return individual

    def calculate_objectives(self, individual):
        individual.objectives = []
        individual.objectives.append(self.zdt_definitions.f1(individual))
        individual.objectives.append(self.zdt_definitions.f2(individual))
        individual.objectives.append(self.zdt_definitions.f3(individual))
        for i in range(3):
        #for i in range(2):
            if self.min_objectives[i] is None or individual.objectives[i] < self.min_objectives[i]:
                self.min_objectives[i] = individual.objectives[i]
            if self.max_objectives[i] is None or individual.objectives[i] > self.max_objectives[i]:
                self.max_objectives[i] = individual.objectives[i]


