from nsga2.evolution import Evolution
from nsga2.problems.zdt import ZDT
from nsga2.problems.zdt.zdt3_definitions import ZDT3Definitions

from metrics.problems.zdt import ZDT3Metrics
from plotter import Plotter

collected_metrics = {}
def collect_metrics(population, generation_num):
    pareto_front = population.fronts[0]
    metrics = ZDT3Metrics()
    hv = metrics.HV(pareto_front)
    hvr = metrics.HVR(pareto_front)
    collected_metrics[generation_num] = hv, hvr


def Select(population, MSR, RV, BI_size, population_K ,nsol, new_msr, new_rv,new_bisize, new_K, generation,self_population,Zdt_definitions,PPlotter,PProblem,EEvolution,population_label,children_label):
    if (Zdt_definitions is None) and (PPlotter is None) and (PProblem is None) and (EEvolution is None):

        zdt_definitions = ZDT3Definitions()
        plotter = Plotter(zdt_definitions)
        problem = ZDT(zdt_definitions)
        evolution = Evolution(problem, 1, len(population))
        evolution.register_on_new_generation(plotter.plot_population_best_front)
        evolution.register_on_new_generation(collect_metrics)
    else:
        zdt_definitions=Zdt_definitions
        plotter=PPlotter
        problem=PProblem
        evolution=EEvolution


    new_pop,objectives,self_population,Final_label,K= evolution.evolve(population,MSR, RV, BI_size, population_K,nsol, new_msr, new_rv, new_bisize, new_K,generation,self_population,population_label,children_label)

    return new_pop,objectives,self_population,zdt_definitions,plotter,problem,evolution,Final_label,K
