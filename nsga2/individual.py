"""Module with main parts of NSGA-II algorithm.
It contains individual definition"""

class Individual(object):
    """Represents one individual"""
    
    def __init__(self):
        self.rank = None
        self.crowding_distance = None
        self.dominated_solutions = set()
        self.features = None
        self.objectives = None
        self.dominates = None
        self.ob1 = None
        self.ob2 = None
        self.ob3 = None
        self.K=None
        self.label=None


    def set_objectives(self, objectives):
        self.objectives = objectives
        
