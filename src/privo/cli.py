import argparse
from privo.app.runner import run

def main() -> None:
    parser = argparse.ArgumentParser(prog="privo")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("run", help="Start the assistant")
    subparsers.add_parser("benchmark", help="Benchmark the assistant with predefined audio samples")

    args = parser.parse_args()

    if args.command == "run":
        run()
    elif args.command == "benchmark":
        print("Benchmark startet...")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()