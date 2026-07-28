# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import subprocess
import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "gatoh"
copyright = "2025, Manuel Munizaga Sepúlveda"
author = "Manuel Munizaga Sepúlveda"
version = "2026.07.0"
release = "2026.07.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "sphinx.ext.todo",
    "jupyter_sphinx",
    "reno.sphinxext",
    "sphinx.ext.intersphinx",
    "sphinxemoji.sphinxemoji",
    "sphinx_reredirects",
    "sphinxcontrib.bibtex",
]

bibtex_bibfiles = [
    "references.bib"
]

bibtex_default_style = "unsrt"

templates_path = ["_templates"]
exclude_patterns = ["_build"]

pygments_style = "colorful"

add_module_names = False

modindex_common_prefix = ["gatoh."]

todo_include_todos = True

source_suffix = [".rst", ".md"]

master_doc = "index"

# Autosummary options
autosummary_generate = True
autosummary_generate_overwrite = False
autoclass_content = "both"

# Intersphinx configuration
intersphinx_mapping = {
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
}

if not os.getenv("GATOH_DEV_DOCS", None):
    rst_prolog = """
.. raw::html

    <br><br><br>

""".format(release)
else:
    rst_prolog = """
.. raw::html

    <br><br><br>

.. note::

    This is the documentation for the current state of the development branch
    of gatoh. The documentation or APIs here can change prior to being
    released.

"""

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = f"{project} {release}"
htmlhelp_basename = "gatoh"

# Latex options
latex_elements = {}
latex_documents = []

# Jupyter Sphinx options
jupyter_execute_default_kernel = "python3"

# Texinfo options
texinfo_documents = []

redirects = {}
with open("sources.txt", "r") as fd:
    for source_str in fd:
        redirects[f"stubs/{source_str}"] = f"../apiref/{source_str}"


# Version extensions
def _get_versions(app, config):
    context = config.html_context
    start_version = (0, 1, 0)
    proc = subprocess.run(["git", "describe", "--tags"], capture_output=True)
    proc.check_returncode()
    current_version = proc.stdout.decode("utf8")
    current_version_info = current_version.split(".")
    if current_version_info[0] == "0":
        version_list = [
            "0.%s" % x
            for x in range(start_version[1], int(current_version_info[1]) + 1)
        ]
    else:
        version_list = []
        pass
    context["version_list"] = version_list
    context["version_label"] = _get_version_label(current_version)


def _get_version_label(current_version):
    if not os.getenv("GATOH_DEV_DOCS", None):
        current_version_info = current_version.split(".")
        return ".".join(current_version_info[:-1])
    else:
        return "Development"


def avoid_duplicate_in_dispatch(app, obj, bound_method):
    if (
        hasattr(obj, "dispatch")
        and hasattr(obj, "register")
        and obj.dispatch.__module__ == "functools"
    ):
        obj.dipatch.__module__ = "gatoh"


def setup(app):
    app.connect("config-inited", _get_versions)
    app.connect("autodoc-before-process-signature", avoid_duplicate_in_dispatch)
