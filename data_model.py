from dataclasses import dataclass
from typing import List, Tuple, TypedDict

# @dataclass
# class Node:
#     city: str
#     longitude: int
#     latitude: int


@dataclass
class AdmissablePaths:
    # demand_id : str
    paths: List[List[str]]


@dataclass
class Demand:
    id: str
    # source : str
    # target : str
    # routing_unit : int
    demand_value: float
    admissable_paths: AdmissablePaths


@dataclass
class Link:
    id: str
    source: str
    target: str
    # pre_installed_capacity: int
    # pre_installed_capacity_cost: int
    # routing_cost: int
    # setup_cost: int
    # modules: List[Tuple[int, int]]


@dataclass
class Specimen:
    demands: List[Tuple[str, List[float]]]
    fitness: int


class LinkResult(TypedDict):
    id: str
    source: str
    target: str
    modules: int


class Result(TypedDict):
    log: List[List[float]]
    links: List[LinkResult]
    modules: int

@dataclass
class Network:
    demands: List[Demand]
    links: List[Link]
    modularity: int
    aggregation: bool

@dataclass
class AlgorithmParameters:
    population_size: int
    crossover_prob: float
    tournament_size: int
    mutation_prob: float
    mutation_power: float
    mutation_range: int
    target_fitness: float
    max_epochs: int
    stale_epochs_limit: int
