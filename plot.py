from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import seaborn  # type: ignore

from results import (
    averigize_results,
    group_results_by_aggregation,
    group_results_by_modularity,
    group_results_by_parameters,
    parse_evolution_results,
    parse_naive_results,
)


@dataclass(frozen=True)
class Plotdata:
    naive: dict[int, int]  # modules by modularity
    evo: dict[
        bool, dict[int, dict[str, Sequence[float]]]
    ]  # modules by aggregation, by modularity, by parameters, by generation


def make_plotdata() -> Plotdata:
    naiveres = parse_naive_results()
    naive = {res["modularity"]: res["modules"] for res in naiveres}
    evores = parse_evolution_results()
    evo = {
        aggr: {
            mod: {
                param: averigize_results(res3).log_of_best
                for param, res3 in group_results_by_parameters(res2).items()
            }
            for mod, res2 in group_results_by_modularity(res).items()
        }
        for aggr, res in group_results_by_aggregation(evores).items()
    }
    return Plotdata(naive, evo)


def make_plot_dataset(
    plotdata: Plotdata, params: Sequence[str], aggregation: bool, modularity: int
) -> dict[str, list[Any]]:
    data: dict[str, list[Any]] = {"parameters": [], "generation": [], "modules": []}
    for i in range(1001):
        data["parameters"].append("naive")
        data["generation"].append(i)
        data["modules"].append(plotdata.naive[modularity])
    for i in range(1001):
        for param in params:
            data["parameters"].append(param)
            data["generation"].append(i)
            data["modules"].append(plotdata.evo[aggregation][modularity][param][i])
    return data


def make_plot(
    name: str,
    plotdata: Plotdata,
    params: Sequence[str],
    aggregation: bool,
    modularity: int,
    zoom: tuple[float, float] | None,
):
    dataset = make_plot_dataset(plotdata, params, aggregation, modularity)
    seaborn.set_theme(style="darkgrid")  # type: ignore
    plot = seaborn.relplot(data=dataset, x="generation", y="modules", hue="parameters", kind="line")  # type: ignore
    Path("charts").mkdir(exist_ok=True)
    plot.savefig("charts/" + name + "-full.png")  # type: ignore
    if zoom is not None:
        plot.set(ylim=zoom)  # type: ignore
        plot.savefig("charts/" + name + "-zoom.png")  # type: ignore


def main():
    plotdata = make_plotdata()
    make_plot(
        "aggr_mod1",
        plotdata,
        params=["40, 0.2, 2, 0.25, 1", "100, 1.0, 8, 0.1, 1", "40, 1.0, 4, 0.1, 1"],
        modularity=1,
        aggregation=True,
        zoom=(22400, 23500),
    )
    make_plot(
        "aggr_mod50",
        plotdata,
        params=[
            "40, 0.2, 2, 0.25, 1",
            "100, 1.0, 4, 0.1, 1",
            "40, 1.0, 4, 0.1, 1",
            "10, 0.5, 8, 0.5, 1",
        ],
        modularity=50,
        aggregation=True,
        zoom=(450, 490),
    )
    make_plot(
        "aggr_mod500",
        plotdata,
        params=["40, 0.2, 2, 0.25, 1", "100, 1.0, 2, 0.1, 1"],
        modularity=500,
        aggregation=True,
        zoom=None,
    )
    make_plot(
        "no-aggr_mod1",
        plotdata,
        params=["40, 0.2, 3, 0.1, 1", "100, 0.0, 9, 0.5, 1"],
        modularity=1,
        aggregation=False,
        zoom=None,
    )
    make_plot(
        "no-aggr_mod50",
        plotdata,
        params=["40, 0.2, 3, 0.1, 1", "100, 0.0, 9, 1.0, 1"],
        modularity=50,
        aggregation=False,
        zoom=None,
    )
    make_plot(
        "no-aggr_mod500",
        plotdata,
        params=["40, 0.2, 3, 0.1, 1", "100, 0.0, 9, 1.0, 1"],
        modularity=500,
        aggregation=False,
        zoom=None,
    )


if __name__ == "__main__":
    main()
