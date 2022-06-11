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
