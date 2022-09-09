import json
from dataclasses import dataclass
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
    score_scaled: float
    best_ever: float


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
        avgres = sum([len(x) for x in by_modularity_by_params[mod].values()]) / len(
            by_modularity_by_params[mod].keys()
        )
        print(
            f"for mod {mod} there is {len(by_modularity_by_params[mod].keys())} params with average number of results {avgres}"
        )

    averagized_by_modularity_by_params = {
        mod: {
            params: averigize_results(results) for params, results in by_params.items()
        }
        for mod, by_params in by_modularity_by_params.items()
    }
    for mod, res in averagized_by_modularity_by_params.items():
        print(f"Best results for modularity {mod}")
        by_score = sorted(res.items(), key=lambda x: x[1].score)
        print(
            f"Best score {by_score[0][1].score} with scaled score {by_score[0][1].score_scaled} with best ever {by_score[0][1].best_ever} for params {by_score[0][0]}"
        )
        by_scaled = sorted(res.items(), key=lambda x: x[1].score_scaled)
        print(
            f"Best scaled score {by_scaled[0][1].score_scaled} with score {by_scaled[0][1].score} with best ever {by_score[0][1].best_ever} for params {by_scaled[0][0]}"
        )


def parse_evolution_results() -> Sequence[EvoResult]:
    results: list[EvoResult] = []
    for file in Path("results").iterdir():
        if file.name.startswith("polska-evolution"):
            result: EvolutionResult = json.loads(file.read_text())

            log_of_best = [max(gen) for gen in result["log"]]
            assert len(log_of_best) == 1001

            best_until_now = 1000000000
            for i in range(len(log_of_best)):
                if log_of_best[i] <= best_until_now:
                    best_until_now = log_of_best[i]
                else:
                    log_of_best[i] = best_until_now

            score_scaled = 0
            for i in range(len(log_of_best)):
                score_scaled += log_of_best[i] * i / len(log_of_best) / 500

            results.append(
                EvoResult(
                    parameters=parameters_to_string_aggr(result["parameters"])
                    if result["aggregation"]
                    else parameters_to_string_no_aggr(result["parameters"]),
                    modularity=result["modularity"],
                    aggregation=result["aggregation"],
                    log_of_best=log_of_best,
                    score=sum(log_of_best) / len(log_of_best),
                    score_scaled=score_scaled,
                    best_ever=min(log_of_best),
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
    score_scaled = sum(result.score_scaled for result in results) / len(results)
    best_ever = sum(result.best_ever for result in results) / len(results)
    log: list[float] = []
    for i in range(len(results[0].log_of_best)):
        log.append(sum(x.log_of_best[i] for x in results) / len(results))
    return EvoResult(
        parameters=results[0].parameters,
        modularity=results[0].modularity,
        aggregation=results[0].aggregation,
        log_of_best=log,
        score=score,
        score_scaled=score_scaled,
        best_ever=best_ever,
    )


def parameters_to_string_aggr(p: AlgorithmParameters) -> str:
    keys = [
        "population_size",
        "crossover_prob",
        "tournament_size",
        "mutation_prob",
        "mutation_range",
    ]
    vals: Iterator[object] = (p[key] for key in keys)
    return ", ".join(map(str, vals))


def parameters_to_string_no_aggr(p: AlgorithmParameters) -> str:
    return parameters_to_string_aggr(p) + f",{p['mutation_power']}"


if __name__ == "__main__":
    main()
