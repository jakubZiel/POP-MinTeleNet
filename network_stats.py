from pathlib import Path

from parsing import NetworkParser


def main():
    parser = NetworkParser(Path("networks/polska.xml"))
    print("polska.xml")

    links = parser.links()
    nodes = {link.source for link in links}.union({link.target for link in links})
    print(f"number of nodes: {len(nodes)}")
    print(f"number of links: {len(links)}")

    demands = parser.demands()
    print(f"number of demands: {len(demands)}")
    print(
        f"average demand value: {sum([demand.demand_value for demand in demands]) / len(demands)}"
    )


if __name__ == "__main__":
    main()
