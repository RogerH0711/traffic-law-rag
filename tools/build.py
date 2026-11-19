import jupytext, nbformat, sys
SRC = sys.argv[1] if len(sys.argv) > 1 else "NCKU1A_traffic_law_rag.py"
nb = jupytext.read(SRC)
nb = nbformat.from_dict(jupytext.writes(nb, fmt="ipynb") and nbformat.reads(jupytext.writes(nb, fmt="ipynb"), 4))
for c in nb.cells:
    if c.cell_type == "code":
        c.source = "\n".join(l[2:] if l.startswith(("# %pip", "# !")) else l for l in c.source.splitlines())
nb.metadata = {"colab": {"provenance": [], "gpuType": "T4"}, "accelerator": "GPU",
               "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}
nbformat.validate(nb)
nbformat.write(nb, SRC.replace(".py", ".ipynb"))
print("built", len(nb.cells), "cells")
