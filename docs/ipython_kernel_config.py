# IPython kernel configuration used when executing the tutorial notebooks.
#
# The docs build (docs/Makefile) and the notebook tests point IPYTHONDIR at a
# throwaway profile directory containing a copy of this file, so the matplotlib
# display settings live here rather than in a per-notebook preamble cell.
#
# Note: this only applies where we launch the kernel (the docs build and the
# tests). Notebooks opened on Google Colab or locally do not see it and fall
# back to the usual matplotlib defaults.
c.InteractiveShellApp.matplotlib = "inline"
c.InlineBackend.figure_format = "svg"
c.InlineBackend.rc = {"figure.figsize": (11.0, 6.0)}
