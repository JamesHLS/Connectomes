"""Execute notebooks in fresh Python kernels and save their regenerated outputs."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".execution"
for name, folder in {
    "JUPYTER_RUNTIME_DIR": "jupyter",
    "IPYTHONDIR": "ipython",
    "MPLCONFIGDIR": "matplotlib",
}.items():
    directory = STATE / folder
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[name] = str(directory)
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")
os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
os.environ["PYTHONIOENCODING"] = "utf-8"

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


class ProgressClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        result = super().process_message(msg, cell, cell_index)
        if msg["msg_type"] == "stream":
            value = msg["content"]["text"].strip()
            if value and len(value) < 400:
                print(value, flush=True)
        return result


def run(path):
    notebook = nbformat.read(path, as_version=4)
    notebook.nbformat_minor = max(5, notebook.nbformat_minor)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
            cell.metadata.pop("execution", None)
    started = time.time()
    record = {
        "notebook": path.name,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    record_path = STATE / f"{path.stem}.json"
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"
    ]

    def cell_start(cell, cell_index):
        if cell.cell_type == "code" and cell.source.strip():
            print(f"{path.name}: cell {cell_index + 1}/{len(notebook.cells)}", flush=True)

    def checkpoint(cell, cell_index, execute_reply):
        nbformat.write(notebook, STATE / f"{path.stem}.progress.ipynb")

    client = ProgressClient(
        notebook, km=manager, timeout=None, allow_errors=False,
        resources={"metadata": {"path": str(ROOT)}},
        on_cell_start=cell_start, on_cell_executed=checkpoint,
    )
    print(f"START {path.name}", flush=True)
    try:
        client.execute()
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        record["status"] = "passed"
        record["executed_cells"] = sum(
            c.cell_type == "code" and c.execution_count is not None
            for c in notebook.cells
        )
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = str(exc)
        nbformat.write(notebook, STATE / f"{path.stem}.failed.ipynb")
        print(f"FAILED {path.name}: {exc}", flush=True)
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        record["duration_seconds"] = round(time.time() - started, 2)
        record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"{record['status'].upper()} {path.name} ({record['duration_seconds']}s)", flush=True)
    return record["status"] == "passed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", help="Notebook filenames; defaults to all.")
    args = parser.parse_args()
    paths = [ROOT / name for name in args.notebooks] if args.notebooks else sorted(ROOT.glob("*.ipynb"))
    success = True
    for path in paths:
        success = run(path) and success
    sys.exit(0 if success else 1)
