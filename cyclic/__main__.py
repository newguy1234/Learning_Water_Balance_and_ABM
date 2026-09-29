"""python -m cyclic [report | ui | blocks | points | paper | identify | routing | bench | copernicus]

Agents are registered outside the package: python example.py

Copernicus (Sentinel-2 over Imbanuma)::

    python -m cyclic copernicus register
    python -m cyclic copernicus status
    python -m cyclic copernicus search 2022
    python -m cyclic copernicus download 2022 --max-cloud 40
"""

import sys


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    command = argv[0] if argv else "ui"
    if command == "report":
        from . import report

        report.main(agent=argv[1] if len(argv) > 1 else None)
    elif command in ("copernicus", "cdse", "sentinel"):
        from . import copernicus

        raise SystemExit(copernicus.main(argv))
    elif command == "ui":
        from . import serve

        serve()
    elif command == "points":
        from . import points

        points.main(argv[1:])
    elif command == "bench":
        from . import benchmark

        benchmark.main(argv[1:])
        return
    elif command == "paper":
        from . import methods

        methods.main(argv[1:])
        return
    elif command == "routing":
        from .identify import routing_delivery

        year = int(argv[1]) if len(argv) > 1 else 2022
        result = routing_delivery(year)
        print(result.text)
        print("wrote", result.path)
        return
    elif command == "identify":
        if len(argv) > 1 and argv[1] in ("qrc", "reservoir"):
            from .qrc import train_qrc
            backend = argv[2] if len(argv) > 2 else "classical"
            if backend == "spatial":
                from .qrc import train_spatial_qrc
                hw = "classical"
                year = 2022
                if len(argv) > 3 and argv[3] in ("ibm", "classical", "auto"):
                    hw = argv[3]
                    year = int(argv[4]) if len(argv) > 4 else year
                elif len(argv) > 3:
                    year = int(argv[3])
                print("spatial QRC hardware:", hw)
                result = train_spatial_qrc(year=year, backend=hw)
                ibm = result.ibm
                print(result.equation_text)
                print("spatial clusters", result.n_clusters, "paddies", result.n_paddies,
                      "days", result.n_days)
                print("rmse", result.rmse, "stay_weighted", result.stay_weighted,
                      "stay_target", result.stay_target)
                print("ibm backend", ibm.backend, "job", ibm.job_id)
                print("saved", result.path)
                if getattr(result, "equation_path", None):
                    print("equation", result.equation_path)
                return
            year = int(argv[3]) if len(argv) > 3 else 2022
            other = 2021 if year == 2022 else 2022
            result = train_qrc(year=year, backend=backend, holdout_year=other)
            print(result.equation_text)
            print("saved", result.path)
            return
        if len(argv) > 1 and argv[1] in ("graph", "qubo-graph"):
            from .data import paddies
            from .canal import distance_to_canal_m
            from .graph import leftover_graph
            from .lateral import knn_edges
            year = int(argv[2]) if len(argv) > 2 else 2022
            frame = paddies()
            cyclic = frame["cyclic"].to_numpy(dtype=bool)
            lon = frame["lon"].to_numpy(dtype=float)
            lat = frame["lat"].to_numpy(dtype=float)
            dist = distance_to_canal_m(lon, lat)
            edges = knn_edges(lon, lat, k=4)
            graph = leftover_graph(dist, edges, cyclic, one_way=False)
            print("leftover graph nodes", graph.n, "directed edges", graph.i.size,
                  "undirected soil edges", len(graph.undirected), "one_way", graph.one_way)
            if argv[1] == "qubo-graph":
                from .quantum import omalley_qubo
                encoding = omalley_qubo(year, n_nodes=12)
                print("coarsened QUBO n_var", encoding.n_var, "terms", len(encoding.Q),
                      "backend-ready (O'Malley clusters, not 4925 paddies)")
            return
        if len(argv) > 1 and argv[1] in ("quantum", "ibm", "qaoa", "omalley"):
            from .identify import run_finder
            backend = argv[2] if len(argv) > 2 else "auto"
            nodes = int(argv[3]) if len(argv) > 3 else 8
            method = "ibm" if backend == "ibm" else "omalley"
            report = run_finder(span="tables", method=method, n_nodes=nodes,
                                quantum_backend=backend)
            quantum = report.quantum
            print("backend", quantum.backend, "n_var", quantum.n_var, "energy", quantum.energy)
            print("physics", report.params)
            print("capture mean", report.params["kx_ratio"] + report.params["ky_ratio"])
            print("saved", report.saved)
            return
        from .identify import HydrologyProblem, solve_classical

        days = int(argv[2]) if len(argv) > 2 else 8
        n_eval = int(argv[1]) if len(argv) > 1 else 6
        problem = HydrologyProblem()
        result = solve_classical(problem, n_eval=n_eval, days=days)
        print("names", result.names, "later", result.later)
        print("best", result.best.params, "loss", result.best.loss)
        print("objectives", result.best.objectives)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
