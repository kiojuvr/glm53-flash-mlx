from __future__ import annotations

import argparse
import json
import sys

from .manifest import ManifestError, inspect_checkpoint


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "serve":
        from .server import main as server_main
        return server_main(argv[1:])
    parser = argparse.ArgumentParser(prog="glm53")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect_p = sub.add_parser("inspect", help="validate a source or converted checkpoint")
    inspect_p.add_argument("model")
    sub.add_parser("serve", help="run the OpenAI-compatible server").add_argument(
        "args", nargs=argparse.REMAINDER
    )
    ns = parser.parse_args(argv)
    if ns.command == "inspect":
        try:
            print(json.dumps(inspect_checkpoint(ns.model).to_dict(), indent=2))
            return 0
        except ManifestError as exc:
            print(f"glm53 inspect: {exc}", file=sys.stderr)
            return 2
    raise AssertionError(f"unhandled command: {ns.command}")


if __name__ == "__main__":
    raise SystemExit(main())
