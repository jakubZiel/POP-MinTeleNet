import json
import math
import sys
from pathlib import Path
from random import randint, random
from typing import Dict, List, Tuple

import numpy as np

from data_model import Demand, Link, LinkResult, Result, Specimen
from parsing import NetworkParser


class Evolution:
    def __init__(
        self,
        demands: List[Demand],
        links: List[Link],
        modularity: int,
        aggregation: bool,
        population_size: int,
        crossover_prob: float,
        tournament_size: int,
        mutation_prob: float,
        mutation_power: float,
        mutation_range: int,
        target_fitness: float,
        max_epochs: int,
        stale_epochs_limit: int,
    ):
        self.demands = {demand.id: demand for demand in demands}
        self.links = {link.id: link for link in links}
        self.modularity = modularity
        self.aggregation = aggregation

        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.mutation_power = mutation_power
        self.mutation_range = mutation_range
        self.tournament_size = tournament_size

        self.target_fitness = target_fitness
        self.max_epochs = max_epochs
        self.stale_epochs_limit = stale_epochs_limit

        self.population: List[Specimen] = []
        self.best_specimen = Specimen([], sys.maxsize)
        self.prev_best_fitness = math.inf
        self.current_generation = 0
        self.log: List[List[Specimen]] = []
        self.stale_generations_count = 0

    def run(self) -> None:
        self.population = self.create_init_population()
        self.evaluate_population()
        self.population.sort(key=lambda spec: spec.fitness)
        self.log.append(self.population)

        while self.continue_condition():
            next_pop: List[Specimen] = []
            for _ in range(self.population_size):
                if self.should_crossover():
                    parent1 = self.select()
                    parent2 = self.select()

                    next_spec = self.mutation(self.crossover((parent1, parent2)))
                    next_pop.append(next_spec)

                else:
                    next_spec = self.mutation(self.select())
                    next_pop.append(next_spec)

            self.population = next_pop
            self.current_generation += 1
            self.evaluate_population()
            self.population.sort(key=lambda spec: spec.fitness)
            self.log.append(self.population)

    def continue_condition(self) -> bool:
        if self.best_specimen.fitness <= self.target_fitness:
            return False

        if self.current_generation >= self.max_epochs:
            return False

        best_got_better = self.best_specimen.fitness < self.prev_best_fitness
        self.prev_best_fitness = min(self.prev_best_fitness, self.best_specimen.fitness)

        if best_got_better:
            self.stale_generations_count = 0
        elif self.stale_generations_count + 1 >= self.stale_epochs_limit:
            return False
        else:
            self.stale_generations_count += 1

        return True

    def calc_fitness(self, spec: Specimen) -> int:
        modules = 0
        link_loads: Dict[str, float] = {link.id: 0.0 for link in self.links.values()}

        for demand_id, demand_uses in spec.demands:
            demand: Demand = self.demands[demand_id]
            paths: List[List[str]] = demand.admissable_paths.paths
            for i in range(len(demand_uses)):
                links: List[str] = paths[i]
                use: float = demand_uses[i]
                for link in links:
                    link_loads[link] += use * demand.demand_value

        for load in link_loads.values():
            modules += self.edge_capacity(load)

        return modules

    def edge_capacity(self, load: float) -> int:
        if self.modularity <= 0:
            raise ValueError("modularity must be positive")
        return math.ceil(load / self.modularity)

    def select(self) -> Specimen:
        tournament: List[Specimen] = []

        for _ in range(self.tournament_size):
            specimen_index = randint(0, self.population_size - 1)
            tournament.append(self.population[specimen_index])

        tournament.sort(key=lambda spec: spec.fitness)

        return tournament[0]

    def mutation(self, specimen: Specimen) -> Specimen:
        return (
            self.mutation_aggregate(specimen)
            if self.aggregation
            else self.mutation_no_aggregate(specimen)
        )

    def get_demands_to_mutate(self, specimen: Specimen) -> List[int]:
        demand_ids = list(range(0, len(specimen.demands)))
        demands_to_mutate: List[int] = []

        for _ in range(self.mutation_range):
            chosen_demand_index = randint(0, len(demand_ids) - 1)
            chosen_demand_id = demand_ids[chosen_demand_index]

            demand_ids.remove(chosen_demand_index)
            demands_to_mutate.append(chosen_demand_id)
        return demands_to_mutate

    def mutation_aggregate(self, specimen: Specimen) -> Specimen:
        demands_to_mutate = self.get_demands_to_mutate(specimen)

        demands = specimen.demands

        for demand_to_mutate in demands_to_mutate:
            demand_paths = len(demands[demand_to_mutate])

            new_path = randint(0, demand_paths - 1)

            new_demand = [0.0] * demand_paths
            new_demand[new_path] = 1.0

            demands[demand_to_mutate] = (demands[demand_to_mutate][0], new_demand)

        return specimen

    def mutation_no_aggregate(self, specimen: Specimen) -> Specimen:
        demands_to_mutate = self.get_demands_to_mutate(specimen)
        demand_paths = len(specimen.demands[0])

        for demand_to_mutate_index in demands_to_mutate:
            mutation_vector = [0.0] * demand_paths

            for path_index in range(demand_paths):
                mutation_vector[path_index] = np.random.normal(0, self.mutation_power)

            _, demand = specimen.demands[demand_to_mutate_index]

            for path_index in range(len(demand)):
                demand[path_index] = abs(
                    demand[path_index] + mutation_vector[path_index]
                )

            specimen.demands[demand_to_mutate_index] = (
                specimen.demands[demand_to_mutate_index][0],
                Evolution.normalize_demand(demand),
            )

        return specimen

    def crossover(self, pair: Tuple[Specimen, Specimen]) -> Specimen:
        return (
            self.crossover_aggregate(pair)
            if self.aggregation
            else self.crossover_no_aggregate(pair)
        )

    def crossover_aggregate(self, pair: Tuple[Specimen, Specimen]) -> Specimen:
        parent1, parent2 = pair

        crossover_geonome: List[Tuple[str, List[float]]] = []

        for demand_index in range(len(parent1.demands)):

            if random() > 0.5:
                crossover_geonome.append(parent1.demands[demand_index])
            else:
                crossover_geonome.append(parent2.demands[demand_index])

        return Specimen(crossover_geonome, sys.maxsize)

    def crossover_no_aggregate(self, pair: Tuple[Specimen, Specimen]) -> Specimen:
        parent1, parent2 = pair
        demands1 = parent1.demands
        demands2 = parent2.demands

        crossover_genome: List[Tuple[str, List[float]]] = []

        for demand_index in range(len(demands1)):
            crossover_demand: List[float] = []
            demand_id = demands1[demand_index][0]

            for path_index in range(len(parent1.demands[0])):
                path_usage = (
                    demands1[demand_index][1][path_index]
                    + demands2[demand_index][1][path_index]
                ) / 2
                crossover_demand.append(path_usage)

            crossover_genome.append((demand_id, crossover_demand))

        return Specimen(crossover_genome, sys.maxsize)

    @staticmethod
    def normalize_demand(demand: List[float]) -> List[float]:
        demand_sum = sum(demand)

        for index in range(len(demand)):
            demand[index] /= demand_sum

        return demand

    def create_init_population(self) -> List[Specimen]:
        return (
            self.init_population_aggregate()
            if self.aggregation
            else self.init_population_no_aggregate()
        )

    def init_population_aggregate(self) -> List[Specimen]:
        demands = list(self.demands.values())
        paths = len(demands[0].admissable_paths.paths)
        init_population: List[Specimen] = []

        for _ in range(self.population_size):
            new_genome: List[Tuple[str, List[float]]] = []

            for i_demand in range(len(self.demands)):
                new_gene = [0.0] * paths
                random_index = randint(0, paths - 1)
                new_gene[random_index] = 1.0
                new_genome.append((demands[i_demand].id, new_gene))

            init_population.append(Specimen(new_genome, sys.maxsize))

        return init_population

    def init_population_no_aggregate(self) -> List[Specimen]:
        demands = list(self.demands.values())
        paths = len(demands[0].admissable_paths.paths)
        init_population: List[Specimen] = []

        for _ in range(self.population_size):
            new_genome: List[Tuple[str, List[float]]] = []

            for i_demand in range(len(self.demands)):
                new_gene = np.random.uniform(0.0, 1.0, paths).tolist()
                new_gene = self.normalize_demand(new_gene)
                new_genome.append((demands[i_demand].id, new_gene))

            init_population.append(Specimen(new_genome, sys.maxsize))

        return init_population

    def should_crossover(self) -> bool:
        return random() <= self.crossover_prob

    def should_mutate(self) -> bool:
        return random() <= self.mutation_prob

    def evaluate_population(self) -> None:
        for spec in self.population:
            spec.fitness = self.calc_fitness(spec)
            if spec.fitness < self.best_specimen.fitness:
                self.best_specimen = spec

    def present_specimen(self, spec: Specimen) -> List[LinkResult]:
        link_loads: Dict[str, float] = {link.id: 0.0 for link in self.links.values()}
        for demand_id, demand_uses in spec.demands:
            demand: Demand = self.demands[demand_id]
            paths: List[List[str]] = demand.admissable_paths.paths
            for i in range(len(demand_uses)):
                links: List[str] = paths[i]
                use: float = demand_uses[i]
                for link in links:
                    link_loads[link] += use * demand.demand_value
        return [
            {
                "id": self.links[link_id].id,
                "source": self.links[link_id].source,
                "target": self.links[link_id].target,
                "modules": self.edge_capacity(load),
            }
            for link_id, load in link_loads.items()
        ]

    def get_result(self) -> Result:
        return {
            "log": [[s.fitness for s in specs] for specs in self.log],
            "links": self.present_specimen(self.best_specimen),
            "modules": self.best_specimen.fitness,
        }


if __name__ == "__main__":
    parser = NetworkParser(Path("polska/polska.xml"))
    evo = Evolution(
        demands=parser.demands(),
        links=parser.links(),
        modularity=10,
        aggregation=True,
        population_size=25,
        crossover_prob=0.1,
        tournament_size=2,
        mutation_prob=0.33,
        mutation_power=1,
        mutation_range=1,
        target_fitness=0,
        max_epochs=10000,
        stale_epochs_limit=1000,
    )
    evo.run()
    Path("result.json").write_text(json.dumps(evo.get_result()))
