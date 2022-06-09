from audioop import avg
import math
from random import randint, random
from typing import List, Tuple
from numpy import random
from data_model import Demand, Link, Specimen


class Evolution:
    
    def __init__(self, 
        demands : List[Demand], links : List[Link], modularity : int, aggregation : bool,
        population_size : int, crossover_prob : float, tournament_size : int,
        mutation_prob : float, mutation_power : float, mutation_range : int, 
        target_fitness : float, max_epochs : int, stale_epochs_limit : int):
        
        self.demands = demands
        self.links = links
        self.modularity = modularity
        self.aggregation = aggregation

        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.mutation_power = mutation_power
        self.mutation_range = mutation_range
        self.tournament_size = tournament_size

        self.target_fitness  = target_fitness
        self.max_epochs = max_epochs
        self.stale_epochs_limit = stale_epochs_limit
        
        self.population : List[Specimen] = []
        self.best_fitness = -math.inf
        self.current_generation = 0
        self.log : List[List[Specimen]] = []
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
        if self.best_fitness >= self.target_fitness:
            return False

        if self.current_generation >= self.max_epochs: 
            return False

        prev_best_fitness = self.log[len(self.log) - 1][0].fitness
        best_got_better = self.best_fitness > prev_best_fitness

        if best_got_better:
            self.stale_epochs_limit = 0
        elif self.stale_generations_count + 1 >= self.stale_epochs_limit:
            return False
        else:
            self.stale_generations_count += 1 

        return True

    def calc_fitness(self, demands) -> int:
        modules = 0

        for link in self.links:
            load = 0
            for demand in demands:
                for path in demand:
                    if link in path:
                        load += path.value * demand.value

            modules += self.edge_capacity(self.modularity, load)
            
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
        if self.modularity <= 0:
            raise ValueError("modularity must be positive")
        return math.ceil(o / self.modularity)
    
    def select(self) -> Specimen:
        tournament : List[Specimen] = []

        for _ in range(self.tournament_size):
            specimen_index = randint(0, self.population_size)
            tournament.append(self.population[specimen_index])            
        
        tournament.sort(key=lambda spec: spec.fitness, reverse=True)
        
        return tournament[0]

    def mutation(self, specimen : Specimen) -> Specimen:
        return self.mutation_aggregate(specimen) if self.aggregation else self.mutation_no_aggregate(specimen)

    def get_demands_to_mutate(self, specimen : Specimen) -> List[int]:
        demand_ids = list(range(0, len(specimen.demands)))
        demands_to_mutate : List[int] = []
        
        for _ in range(self.mutation_range):
            chosen_demand_index = randint(0, len(demand_ids) - 1)
            chosen_demand_id = demand_ids[chosen_demand_index]

            demand_ids.remove(chosen_demand_index)
            demands_to_mutate.append(chosen_demand_id)
        return demands_to_mutate

    def mutation_aggregate(self, specimen : Specimen) -> Specimen:
        demands_to_mutate = self.get_demands_to_mutate(specimen)
        
        demands = specimen.demands

        for demand_to_mutate in demands_to_mutate:
            DEMAND_PATHS = len(demands[demand_to_mutate])

            new_path = randint(0, DEMAND_PATHS - 1)
            
            new_demand = [0.0] * DEMAND_PATHS
            new_demand[new_path] = 1.0

            demands[demand_to_mutate] = new_demand

        return specimen

    def mutation_no_aggregate(self, specimen: Specimen) -> Specimen:
        demands_to_mutate = self.get_demands_to_mutate(specimen)
        DEMAND_PATHS = len(specimen.demands[0])

        for demand_to_mutate_index in demands_to_mutate:
            mutation_vector = [0.0] * DEMAND_PATHS

            for path_index in range(0, DEMAND_PATHS):
                mutation_vector[path_index] = random.normal(0, self.mutation_power)

            _, demand = specimen.demands[demand_to_mutate_index]

            for path_index in range(0, len(demand)):
                demand[path_index] = abs(demand[path_index] + mutation_vector[path_index])            
            
            specimen.demands[demand_to_mutate_index] = self.normalize_demand(demand)
            
        return specimen

    def crossover(self, pair : Tuple[Specimen, Specimen]) -> Specimen:
        return self.crossover_aggregate(pair) if self.aggregation else self.crossover_no_aggregate(pair)

    def crossover_aggregate(self, pair : Tuple[Specimen, Specimen]) -> Specimen:
        parent1, parent2 = pair
        
        crossover_geonome  : List[Tuple[str, List[float]]] = []

        for demand_index in range(0, len(parent1.demands)):

            if random() > 0.5:
                crossover_geonome.append(parent1[demand_index])
            else:
                crossover_geonome.append(parent2[demand_index])
                
        return Specimen(crossover_geonome, None)

    def crossover_no_aggregate(self,  pair : Tuple[Specimen, Specimen]) -> Specimen:
        parent1, parent2 = pair
        demands1 = parent1.demands
        demands2 = parent2.demands

        crossover_genome : List[Tuple[str, List[float]]] = []

        for demand_index in range(0, len(demands1)):
            crossover_demand : List[float] = []
            _, demand_id = demands1[demand_index]
    
            for path_index in len(parent1.demands[0]):
                path_usage = (demands1[demand_index][path_index] + demands2[demand_index][path_index]) / 2   
                crossover_demand.append(path_usage)    
            
            crossover_genome.append((crossover_demand, demand_id))

        return Specimen(crossover_genome, None)

    def normalize_demand(demand : List[float]) -> List[float]:
        demand_sum = sum(demand)
        
        for index in range(0, len(demand)):
            demand[index] /= demand_sum
        
        return demand

    def create_init_population(self) -> List[Specimen]:
        return self.init_population_aggregate() if self.aggregation else self.init_population_no_aggregate()

    def init_population_aggregate(self) -> List[Specimen]:
        return []

    def init_population_no_aggregate(self) -> List[Specimen]:
        return []

    def should_crossover(self) -> bool:
        return random() <= self.crossover_prob

    def should_mutate(self) -> bool:
        return random() <= self.mutation_prob

    def evaluate_population(self, population : List[Specimen]) -> None:        
        
        for spec in population:
            spec.fitness = self.calc_fitness_aggregate(self.demands)
            self.best_fitness = max(self.best_fitness, spec.fitness)

if __name__ == "__main__":
    print("Evolution Algorithm")