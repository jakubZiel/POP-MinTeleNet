from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class Node:
    city : str
    longitude : int
    latitude : int

def parseNodes(file : str) -> List[Node] :

    nodes = []
    with open(file) as f_handle:
        
        for line in f_handle:
            if "NODES" in line:
                break

        for line in f_handle:
            if ")\n" == line:
                break
            else:
                filter_params = filter(lambda x: x not in "()", line.split())
                params =  list(filter_params)
                nodes.append(
                    Node(
                        params[0],
                        float(params[1]), 
                        float(params[2])
                    )
                )
    return nodes


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


def parseLinks(file : str) -> List[Link] :

    links = []
    with open(file) as f_handle:
        
        for line in f_handle:
            if "LINKS" in line:
                break

        for line in f_handle:
            if ")\n" == line:
                break
            else:
                filter_params = filter(lambda x: x not in "()", line.split())
                params =  list(filter_params)

                modules = [(params[7], params[8]), (params[9], params[10])]

                links.append(
                    Link(
                        params[0][5:],
                        params[1], 
                        params[2],
                        float(params[3]), 
                        float(params[4]),
                        float(params[5]),
                        float(params[6]),
                        modules
                    )
                )
    return links


@dataclass
class Demand:
    id : str
    source : str
    target : str
    routing_unit : int
    demand_value : float

def parseDemands(file : str) -> List[Demand] :

    demands = []
    with open(file) as f_handle:
        
        for line in f_handle:
            if "DEMANDS" in line:
                break

        for line in f_handle:
            if ")\n" == line:
                break
            else:
                filter_params = filter(lambda x: x not in "()", line.split())
                params =  list(filter_params)
                demands.append(
                    Demand(
                        params[0][7:],
                        params[1],
                        params[2], 
                        int(params[3]),
                        float(params[4])
                    )
                )
    return demands 

@dataclass
class Path:
    links : List[Link]

@dataclass
class AdmissablePaths:
    demand_id : str
    paths : List[str]


def parseAdmissablePaths(file) ->  None: 
    all_admissable_paths = []
    with open(file) as f_handle:
        
        for line in f_handle:
            if "ADMISSIBLE_PATHS" in line:
                break

        for line in f_handle:
            if ")\n" == line:
                break
            
            if "Demand" in line:
                demand_id = line.replace("(", "").strip()[7:]
                paths = []
                for line in f_handle:
                    if "  )\n" == line:
                        break
                    else:
                        links = []
                        for elem in line.split():
                            if elem not in "()" and "P_" not in elem:
                                links.append(elem[5:])

                        paths.append(links)

                all_admissable_paths.append(
                    AdmissablePaths(
                        demand_id,
                        paths
                        )
                    )
                                            


    return all_admissable_paths 


parseNodes("./polska/polska.txt")
parseLinks("./polska/polska.txt")
parseDemands("./polska/polska.txt")
parseAdmissablePaths("./polska/polska.txt")