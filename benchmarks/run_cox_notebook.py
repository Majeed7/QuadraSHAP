"""Execute the Cox tutorial with this Python interpreter and persist cell outputs."""
import os
import argparse
from pathlib import Path
import sys
from time import perf_counter

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("JUPYTER_RUNTIME_DIR", "/private/tmp/quadrashap-cox-jupyter")
os.environ.setdefault("IPYTHONDIR", "/private/tmp/quadrashap-cox-ipython")
Path(os.environ["JUPYTER_RUNTIME_DIR"]).mkdir(parents=True, exist_ok=True)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--notebook", default="tutorials/cox_survival.ipynb")
parser.add_argument("--timeout", type=int, default=1200, help="Maximum seconds per code cell")
args = parser.parse_args()
path = ROOT / args.notebook
notebook = nbformat.read(path, as_version=4)


def completed(cell, cell_index, **kwargs):
    nbformat.write(notebook, path)
    print(f"Completed cell {cell_index + 1}/{len(notebook.cells)}", flush=True)
    for output in cell.get("outputs", []):
        if output.output_type == "stream":
            print(output.text.rstrip(), flush=True)


manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
client = NotebookClient(notebook, km=manager, timeout=args.timeout, record_timing=True,
                        resources={"metadata": {"path": str(ROOT)}}, on_cell_executed=completed)
started = perf_counter()
try:
    client.execute(cleanup_kc=True)
finally:
    notebook.metadata["execution_wall_seconds"] = perf_counter() - started
    nbformat.write(notebook, path)
print(f"Executed notebook in {notebook.metadata['execution_wall_seconds']:.2f}s", flush=True)
