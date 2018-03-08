"""Module with main parts of NSGA-II algorithm.
Contains main loop"""
from nsga2 import individual
from nsga2.utils import NSGA2Utils
from nsga2.population import Population



class Evolution(object):

    def __init__(self, problem, num_of_generations, num_of_individuals):
        self.utils = NSGA2Utils(problem, num_of_individuals)

        self.population = None

        self.num_of_generations = num_of_generations
        self.on_generation_finished = []
        self.num_of_individuals = num_of_individuals

    def register_on_new_generation(self, fun):
        self.on_generation_finished.append(fun)
        
    def evolve(self,solution,MSR, RV, BI_size, solution_K, new_solution, new_msr, new_rv, new_bisize, new_K, generation, self_population, population_label, children_label):
    #population, MSR, RV, BI_size, population_K,nsol, new_rv, new_msr, new_K,generation,self_population,population_label,children_label)

        if self_population is None:

            #self.population = self.utils.create_initial_population(sil_sco,DunnIndex,solution)
            self.population = self.utils.create_initial_population(MSR, RV, BI_size, solution_K, population_label, solution)
            self.utils.fast_nondominated_sort(self.population)
            for front in self.population.fronts:
                self.utils.calculate_crowding_distance(front)

        else:
            #print "SelfPOP", self_population, len(self_population), type(self_population), self_population.fronts
            self.population=self_population

        children = self.utils.create_children(new_solution,new_msr,new_rv, new_bisize, children_label,new_K)
        self.population.extend(children)
        self.utils.fast_nondominated_sort(self.population)

        new_population = Population()
        front_num = 0

        while len(new_population) + len(self.population.fronts[front_num]) <= self.num_of_individuals:
            self.utils.calculate_crowding_distance(self.population.fronts[front_num])
            new_population.extend(self.population.fronts[front_num])
            front_num += 1

        sorted(self.population.fronts[front_num], cmp=self.utils.crowding_operator)
        new_population.extend(self.population.fronts[front_num][0:self.num_of_individuals - len(new_population)])
        returned_population = self.population
        self.population=new_population

        nsol = []
        objective = []
        new_pop_label = {}
        new_pop_K = {}
        cc = 0

        for _ in new_population:
            x = (getattr(_, 'features'))
            obj = (getattr(_, 'objectives'))
            label=(getattr(_, 'label'))
            K1=(getattr(_, 'K'))
            nsol.insert(cc, x)
            objective.insert(cc, obj)
            new_pop_label[cc]= label
            new_pop_K[cc]=K1
            cc += 1
        for fun in self.on_generation_finished:
            fun(returned_population, generation)

        # population_label_old=dict(population_label)
        # solution_K_old=dict(solution_K)
        # solution = list(solution)
        # new_solution = list(new_solution)
        #
        # for k in range(len(solution)):
        #     y = list(solution[k])
        #     if y in nsol:
        #         get_index = nsol.index(y)
        #         population_label[get_index] = population_label_old[k]
        #         solution_K[get_index]=solution_K_old[k]
        #
        #     elif (new_solution in nsol) and (y not in nsol):
        #         get_index = nsol.index(new_solution)
        #         population_label[get_index] = children_label
        #         solution_K[get_index] = new_K
        #         print "new solution replaced {0}th solution in population, old and new objectives:".format(get_index), "(",[MSR[k],RV[k],BI_size[k]], ")", objective[get_index]

        return nsol, objective, self.population, new_pop_label, new_pop_K
