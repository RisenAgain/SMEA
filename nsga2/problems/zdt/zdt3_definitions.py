import math
from nsga2 import seq
from nsga2.problems.problem_definitions import ProblemDefinitions

class ZDT3Definitions(ProblemDefinitions):

    def __init__(self):
        self.n = None

    def f1(self, individual):
        return individual.ob1

    def f2(self, individual):

        return individual.ob2

    def f3(self, individual):
        return individual.ob3
