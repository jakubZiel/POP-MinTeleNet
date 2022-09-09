from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator, Sequence

from data_model import AlgorithmParameters, EvolutionResult, NaiveResult


@dataclass
class EvoResult:
    parameters: str
    modularity: int
    aggregation: bool
    log_of_best: Sequence[float]
    score: float


def main():
    naive = parse_naive_results()
    evo = parse_evolution_results()

    make_naive_report(naive)
    print("\n")
    make_evo_report(evo)




def make_naive_report(results: Sequence[NaiveResult]):
    print("Naive report")
    for result in results:
        print(f"Naive for modularity {result['modularity']} is {result['modules']}")


def make_evo_report(results: Sequence[EvoResult]):
    print("Evolution report")
    by_aggr = group_results_by_aggregation(results)
    print()
    make_evo_aggr_report(by_aggr[True])
    print()
    make_evo_no_aggr_report(by_aggr[False])


def make_evo_aggr_report(results: Sequence[EvoResult]):
    print("With aggregation")
    make_report(results)


def make_evo_no_aggr_report(results: Sequence[EvoResult]):
    print("Without aggregation")
    make_report(results)


def make_report(results: Sequence[EvoResult]):
    by_modularity = group_results_by_modularity(results)
    by_modularity_by_params = {
        mod: group_results_by_parameters(results)
        for mod, results in by_modularity.items()
    }

    for mod in by_modularity_by_params.keys():
        avgres = sum([len(x) for x in by_modularity_by_params[mod].values()]) / len(by_modularity_by_params[mod].keys())
        print(f"for mod {mod} there is {len(by_modularity_by_params[mod].keys())} params with average number of results {avgres}")

    averagized_by_modularity_by_params = {
        mod: {
            params: averigize_results(results)
            for params, results in by_params.items()
        }
        for mod, by_params in by_modularity_by_params.items()
    }
    score_by_modularity_by_params = {
        mod: {
            params: score_result(result)
            for params, result in by_params.items()
        }
        for mod, by_params in averagized_by_modularity_by_params.items()
    }
    for mod in score_by_modularity_by_params.keys():
        scored = sorted(score_by_modularity_by_params[mod].items(), key=lambda x:x[1])
        print(f"Top results for modularity {mod}")
        for i in range(3):
            print(f"{i+1} - score {scored[i][1]} params {scored[i][0]}")


def parse_evolution_results() -> Sequence[EvoResult]:
    results: list[EvoResult] = []
    for file in Path("results").iterdir():
        if file.name.startswith("polska-evolution"):
            result: EvolutionResult = json.loads(file.read_text())
            results.append(
                EvoResult(
                    parameters=parameters_to_string_aggr(result["parameters"])
                    if result["aggregation"]
                    else parameters_to_string_no_aggr(result["parameters"]),
                    modularity=result["modularity"],
                    aggregation=result["aggregation"],
                    log_of_best=[max(gen) for gen in result["log"]],
                    score=result["modules"],
                )
            )
    return results


def parse_naive_results() -> Sequence[NaiveResult]:
    results: list[NaiveResult] = []
    for file in Path("results").iterdir():
        if file.name.startswith("polska-naive"):
            results.append(json.loads(file.read_text()))
    return results


def group_results_by_aggregation(
    results: Sequence[EvoResult],
) -> dict[bool, Sequence[EvoResult]]:
    aggr: list[EvoResult] = []
    no_aggr: list[EvoResult] = []
    for result in results:
        if result.aggregation:
            aggr.append(result)
        else:
            no_aggr.append(result)
    return {True: aggr, False: no_aggr}


def group_results_by_modularity(
    results: Sequence[EvoResult],
) -> dict[int, Sequence[EvoResult]]:
    bins: dict[int, list[EvoResult]] = {}
    for result in results:
        if bins.get(result.modularity) is None:
            bins[result.modularity] = []
        bins[result.modularity].append(result)
    return {id: bin for id, bin in bins.items()}


def group_results_by_parameters(
    results: Sequence[EvoResult],
) -> dict[str, Sequence[EvoResult]]:
    bins: dict[str, list[EvoResult]] = {}
    for result in results:
        if bins.get(result.parameters) is None:
            bins[result.parameters] = []
        bins[result.parameters].append(result)
    return {id: bin for id, bin in bins.items()}


def averigize_results(results: Sequence[EvoResult]) -> EvoResult:
    score = sum(result.score for result in results) / len(results)
    log: list[float] = []
    for i in range(len(results[0].log_of_best)):
        log.append(sum(x.log_of_best[i] for x in results) / len(results))
    return EvoResult(
        parameters=results[0].parameters,
        modularity=results[0].modularity,
        aggregation=results[0].aggregation,
        log_of_best=log,
        score=score,
    )

def score_result(result: EvoResult) -> float:
    assert(len(result.log_of_best) == 1001)
    return sum(result.log_of_best) / len(result.log_of_best)


def parameters_to_string_aggr(p: AlgorithmParameters) -> str:
    keys = [
        "population_size",
        "crossover_prob",
        "tournament_size",
        "mutation_prob",
        "mutation_range",
    ]
    vals: Iterator[object] = (p[key] for key in keys)
    return ",".join(map(str, vals))


def parameters_to_string_no_aggr(p: AlgorithmParameters) -> str:
    return parameters_to_string_aggr(p) + f",{p['mutation_power']}"


if __name__ == "__main__":
    main()
