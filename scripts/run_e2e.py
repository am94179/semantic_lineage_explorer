"""Run the isolated, credential-free initial implementation end-to-end verification flow."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from semantic_lineage.config import DEFAULT_QDRANT_URL
from semantic_lineage.e2e import DEFAULT_E2E_COLLECTION, E2EError, run_offline_e2e

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--collection", default=DEFAULT_E2E_COLLECTION)
    parser.add_argument("--workspace", type=Path, help="Preserve generated E2E artifacts here.")
    parser.add_argument("--output", type=Path, help="Write the E2E summary JSON here.")
    arguments = parser.parse_args()
    if arguments.workspace:
        return _run(arguments, arguments.workspace)
    with tempfile.TemporaryDirectory(prefix="semantic_lineage_e2e_") as temporary_directory:
        return _run(arguments, Path(temporary_directory))


def _run(arguments: argparse.Namespace, workspace: Path) -> int:
    try:
        report = run_offline_e2e(
            qdrant_url=arguments.qdrant_url,
            workspace=workspace,
            project_root=PROJECT_ROOT,
            collection_name=arguments.collection,
        )
    except E2EError as exc:
        _write_report(exc.report, arguments.output)
        print(f"E2E verification failed at '{exc.stage}': {exc.__cause__}")
        if exc.stage == "qdrant_readiness":
            print("Start Qdrant with `docker compose up -d`, then rerun this command.")
        else:
            print(
                "Inspect the failed stage in the JSON report and correct the reported prerequisite or assertion."
            )
        return 1
    _write_report(report, arguments.output)
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return 0


def _write_report(report: object, output: Path | None) -> None:
    if output is None:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
