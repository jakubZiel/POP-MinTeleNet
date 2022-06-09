from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Node:
    city : str
    longitude : int
    latitude : int

@dataclass
class Demand:
    id : str
    source : str
    target : str
    routing_unit : int
    demand_value : float

@dataclass
class Link:
    id : str
    source : str
    target : int
    pre_installed_capacity : int
    pre_installed_capacity_cost : int
    routing_cost : int
    setup_cost : int
    modules : List[Tuple[int, int]]

@dataclass
class AdmissablePaths:
    demand_id : str
    paths : List[str]

@dataclass
class Specimen:
    demands: List[Tuple[str, List[float]]]
    fitness: float