import jupytext, nbformat, sys
nb = jupytext.read("NCKU1A_traffic_law_rag.py")
nb = nbformat.from_dict(jupytext.writes(nb, fmt="ipynb") and nbformat.reads(jupytext.writes(nb, fmt="ipynb"), 4))
for c in nb.cells:
    if c.cell_type == "code" and c.source.startswith("# %pip"):
        c.source = c.source[2:]
nb.metadata = {"colab": {"provenance": [], "gpuType": "T4"}, "accelerator": "GPU",
               "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}
nbformat.validate(nb)
nbformat.write(nb, "NCKU1A_traffic_law_rag.ipynb")
print("built", len(nb.cells), "cells")
