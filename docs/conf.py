"""Documentation-only build: no GPU runtime or simulation execution required."""
from pathlib import Path
import os

project = "ns-blowup"
author = "ns-blowup contributors"
release = "0.2.0"
extensions = ["myst_parser", "sphinx.ext.mathjax"]
myst_enable_extensions = ["dollarmath", "colon_fence"]
source_suffix = {".md": "markdown"}
exclude_patterns = ["conversation.md", "_build", "requirements.txt"]
html_theme = "sphinx_rtd_theme"
html_title = "ns-blowup — Singularity Observatory"
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
# Preserve source artifacts as downloads without parsing the imported conversation.
root = Path(__file__).parent
html_extra_path = [p.name for p in root.iterdir() if p.suffix in (".json", ".txt", ".html")
                   and p.name != "requirements.txt"] + ["conversation.md"]
