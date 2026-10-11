"""Execute the EDA notebook and produce an offline HTML reading copy."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
import re

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=300, help="Maximum seconds for each cell.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks" / "01_data_understanding_and_cleaning.ipynb"
    notebook = nbformat.read(path, as_version=4)
    # Use this interpreter without installing or changing a global kernel.
    with tempfile.TemporaryDirectory(prefix="ev-eda-kernel-") as directory:
        kernel_dir = Path(directory) / "kernels" / "ev-eda"
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "EV EDA temporary", "language": "python",
        }))
        specs = KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)])
        manager = KernelManager(kernel_name="ev-eda", kernel_spec_manager=specs)
        client = NotebookClient(
            notebook, km=manager, timeout=args.timeout, allow_errors=False,
            resources={"metadata": {"path": str(root)}},
        )
        print("Executing notebook...", flush=True)
        client.execute()
    notebook.metadata["kernelspec"] = {
        "display_name": "MSBA265 | EV Purchases",
        "language": "python",
        "name": "msba265-ev-purchases",
    }
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    body, _ = HTMLExporter(
        template_name="lab", require_js_url="", mathjax_url=""
    ).from_notebook_node(notebook)
    body = re.sub(r'<script\b[^>]*\bsrc=""[^>]*>.*?</script>', '', body, flags=re.S)
    html_path = root / "docs" / "reports" / "01_data_understanding_and_cleaning.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(body, encoding="utf-8")
    print("Saved executed notebook and docs/reports/01_data_understanding_and_cleaning.html.")

if __name__ == "__main__":
    main()
