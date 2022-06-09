import math
from random import randint, random
from model.data_model import Demand, Link, Specimen
from typing import List, Tuple

class Evolution:
    
    def __init__(self, demands : List[Demand], links : List[Link], modulity : int, 
        population_size : int, crossover_prob : float, mutation_prob : float,
        mutation_power : float, tournament_size : int, 
        target_fitness : float, max_epochs : int, stale_epochs_limit : int):
        
        self.demands = demands
        self.links = links
        self.modulity = modulity

        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.mutation_power = mutation_power
        self.tournament_size = tournament_size

        self.target_fitness  = target_fitness
        self.max_epochs = max_epochs
        self.stale_epochs_limit = stale_epochs_limit
        
        self.population : List[Specimen] = []
        self.best_fitness = -math.inf
        self.current_generation = 0
        self.log = List[List[Specimen]] = []
        self.stale_generations_count = 0


    def run(self):
        self.population = self.create_init_population()

        while self.continue_condition():
            next_pop : List[Specimen] = []

            for _ in range(self.population_size):
                if self.should_crossover():
                    parent1 = self.select()
                    parent2 = self.select()
                    
                    next_spec = self.mutation(self.crossover((parent1, parent2)))
                    next_pop.append(next_spec)

                else:
                    next_spec = self.mutation(self.select())
                    next_pop.append(next_spec)
            
            self.evaluate_population(next_pop)

            self.log.append(self.population)
            self.population = next_pop
            self.current_generation += 1    

    def continue_condition(self) -> bool:
        return True

    def calc_fitness(self, demands) -> int:
        modules = 0

        for link in self.links:
            load = 0
            for demand in demands:
                for path in demand:
                    if link in path:
                        load += path.value * demand.value

            modules += self.edge_capacity(self.modulity, load)
            
        return modules

    def calc_fitness_aggregate(self, demands) -> int:
        modules = 0

        for link in self.links: 
            load = 0
            for demand in demands:
                path = self.chosen_path(demand)
                if link in path:
                    load += demand.value
            modules += self.edge_capacity(load)        

        return modules

    def chosen_path(self, demand) -> int:
        for i in range(0, 6):
            if demand[i]:
                return i
        return 0
    
    def edge_capacity(self, o : int) -> int:
        if self.modulity <= 0:
            raise ValueError("modulity must be positive")
        return math.ceil(o / self.modulity)
    
    def select(self) -> Specimen:
        tournament : List[Specimen] = []

        for _ in range(self.tournament_size):
            specimen_index = randint(0, self.population_size)
            tournament.append(self.population[specimen_index])            
        
        tournament.sort(key=lambda spec: spec.fitness, reverse=True)
        
        return tournament[0]

    def mutation(self, specimen : Specimen) -> Specimen:
        return None

    def crossover(self, pair : Tuple[Specimen, Specimen]) -> Specimen:
        return None

    def create_init_population(self) -> List[Specimen]:
        return []

    def should_crossover(self) -> bool:
        return random() <= self.crossover_prob

    def should_mutate(self) -> bool:
        return random() <= self.mutation_prob

    def evaluate_population(self, population : List[Specimen]) -> None:        
        for spec in population:
            spec.fitness = self.calc_fitness_aggregate(self.demands)
            self.best_fitness = max(self.best_fitness, spec.fitness)

