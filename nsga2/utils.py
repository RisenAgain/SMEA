"""NSGA-II related functions"""

import functools
from nsga2.population import Population
import random


class NSGA2Utils(object):
    def __init__(self, problem, num_of_individuals, mutation_strength=0.2, num_of_genes_to_mutate=5,
                 num_of_tour_particips=2):

        self.problem = problem
        self.num_of_individuals = num_of_individuals
        self.mutation_strength = mutation_strength
        self.number_of_genes_to_mutate = num_of_genes_to_mutate
        self.num_of_tour_particips = num_of_tour_particips

    def fast_nondominated_sort(self, population):
        population.fronts = []
        #print "population inside fas_nondominated sort : ", population
        population.fronts.append([])
        for individual in population:
            individual.domination_count = 0
            individual.dominated_solutions = set()

            for other_individual in population:
                if other_individual.dominates(individual):
                    individual.dominated_solutions.add(other_individual)
                elif individual.dominates(other_individual):
                    individual.domination_count += 1
            if individual.domination_count == 0:
                population.fronts[0].append(individual)
                individual.rank = 0
        i = 0
        while len(population.fronts[i]) > 0:
            temp = []
            for individual in population.fronts[i]:
                for other_individual in individual.dominated_solutions:
                    other_individual.domination_count -= 1
                    if other_individual.domination_count == 0:
                        other_individual.rank = i + 1
                        temp.append(other_individual)
            i = i + 1
            population.fronts.append(temp)

    def __sort_objective(self, val1, val2, m):
        return cmp(val1.objectives[m], val2.objectives[m])

    def calculate_crowding_distance(self, front):
        if len(front) > 0:
            solutions_num = len(front)
            for individual in front:
                individual.crowding_distance = 0

            for m in range(len(front[0].objectives)):
                front = sorted(front, cmp=functools.partial(self.__sort_objective, m=m))
                front[0].crowding_distance = self.problem.max_objectives[m]
                front[solutions_num - 1].crowding_distance = self.problem.max_objectives[m]
                for index, value in enumerate(front[1:solutions_num - 1]):
                    front[index].crowding_distance = (front[index + 1].crowding_distance - front[
                        index - 1].crowding_distance) / (self.problem.max_objectives[m] - self.problem.min_objectives[m])

    def crowding_operator(self, individual, other_individual):
        if (individual.rank < other_individual.rank) or \
                ((individual.rank == other_individual.rank) and (
                    individual.crowding_distance > other_individual.crowding_distance)):
            return 1
        else:
            return -1

    def create_initial_population(self, MSR, RV, BI_size, nsol_K, nsol_label, nsol):
        population = Population()
        for count in range(self.num_of_individuals):
            individual = self.problem.generateIndividual(MSR[count],RV[count], BI_size[count],nsol_K[count], nsol_label[count],nsol[count])
            self.problem.calculate_objectives(individual)
            population.population.append(individual)
        return population
    
    def create_children(self, nsol, new_msr, new_rv, new_bisize, child_label, new_K):

        children = []
        child=self.problem.generateIndividual(new_msr, new_rv,  new_bisize, new_K, child_label, nsol)
        children.append(child)

        return children

