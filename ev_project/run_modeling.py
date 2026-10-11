"""Execute modeling and export an offline HTML reading copy using this Python."""

import argparse
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=900, help="Maximum seconds for each cell.")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be a positive number of seconds.")
    root = Path(__file__).resolve().parents[1]
    notebook_path = root / "notebooks" / "02_modeling_and_prioritization.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    # Use the interpreter that launched this runner without registering a global kernel.
    with tempfile.TemporaryDirectory(prefix="ev-modeling-kernel-") as directory:
        kernel_dir = Path(directory) / "kernels" / "ev-modeling"
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "EV modeling temporary", "language": "python",
        }), encoding="utf-8")
        specs = KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)])
        manager = KernelManager(kernel_name="ev-modeling", kernel_spec_manager=specs)
        client = NotebookClient(
            notebook, km=manager, timeout=args.timeout, allow_errors=False,
            resources={"metadata": {"path": str(root)}},
        )
        # Notebook plots use a writable temporary cache on any operating system.
        cache_dir = Path(directory) / "matplotlib"
        cache_dir.mkdir()
        kernel_env = dict(os.environ, MPLCONFIGDIR=str(cache_dir))
        print("Executing modeling notebook...", flush=True)
        client.execute(env=kernel_env)
    notebook.metadata["kernelspec"] = {
        "display_name": "MSBA265 | EV Purchases", "language": "python", "name": "msba265-ev-purchases",
    }
    notebook.metadata["language_info"]["version"] = sys.version.split()[0]
    nbformat.validate(notebook)
    nbformat.write(notebook, notebook_path)
    body, _ = HTMLExporter(template_name="lab", require_js_url="", mathjax_url="").from_notebook_node(notebook)
    body = re.sub(r'<script\b[^>]*\bsrc=""[^>]*>.*?</script>', "", body, flags=re.S)
    html_path = root / "docs" / "reports" / "02_modeling_and_prioritization.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(body, encoding="utf-8")
    print("Saved executed notebook and docs/reports/02_modeling_and_prioritization.html.", flush=True)


if __name__ == "__main__":
    main()
