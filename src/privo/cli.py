import argparse
from .app import run, benchmark


def main() -> None:
    """Command-line interface um den Assistenten in verschiedenen Modi zu starten."""
    parser = argparse.ArgumentParser(prog="privo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Startet Privo")
    run_parser.set_defaults(func=lambda args: run(debug=False))

    debug_parser = subparsers.add_parser(
        "debug",
        help="Startet Privo im Debug-Modus und speichert alle Audio- und Textdaten in debug/",
    )
    debug_parser.set_defaults(func=lambda args: run(debug=True))

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Führt einen Benchmark des Assistenten mit vordefinierten Audiodateien aus tests/samples durch",
    )
    benchmark_parser.set_defaults(func=lambda args: benchmark())

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
