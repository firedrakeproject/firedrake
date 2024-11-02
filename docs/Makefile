# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = FIREDRAKE_BUILDING_DOCS=1 sphinx-build
PAPER         =
BUILDDIR      = build

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -W -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) source
# the i18n builder cannot share the environment and doctrees with the others
I18NSPHINXOPTS  = $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) source

.PHONY: help clean html dirhtml singlehtml pickle json htmlhelp qthelp \
devhelp epub latex latexpdf text man changes linkcheck doctest gettext \
publish publishpdf serve reset_notebooks

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"
	@echo "  serve      to launch a local web server to serve up documentation"
	@echo "  dirhtml    to make HTML files named index.html in directories"
	@echo "  singlehtml to make a single large HTML file"
	@echo "  pickle     to make pickle files"
	@echo "  json       to make JSON files"
	@echo "  htmlhelp   to make HTML files and a HTML help project"
	@echo "  qthelp     to make HTML files and a qthelp project"
	@echo "  devhelp    to make HTML files and a Devhelp project"
	@echo "  epub       to make an epub"
	@echo "  latex      to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  latexpdf   to make LaTeX files and run them through pdflatex"
	@echo "  text       to make text files"
	@echo "  man        to make manual pages"
	@echo "  texinfo    to make Texinfo files"
	@echo "  info       to make Texinfo files and run them through makeinfo"
	@echo "  gettext    to make PO message catalogs"
	@echo "  changes    to make an overview of all changed/added/deprecated items"
	@echo "  linkcheck  to check all external links for integrity"
	@echo "  doctest    to run all doctests embedded in the documentation (if enabled)"

TARGETS = html dirhtml singlehtml pickle json htmlhelp qthelp devhelp epub latex latexpdf text man changes linkcheck doctest gettext

GENERATED_FILES = source/team.rst source/demos source/element_list.csv source/firedrake_citations.rst

publishpdf:
	env FIREDRAKE_MANUAL_RELEASE=`date +%Y-%m` $(MAKE) latexpdf

validate-bibtex:
	../firedrake/scripts/firedrake_preprocess_bibtex.py --validate source/_static/bibliography.bib
	../firedrake/scripts/firedrake_preprocess_bibtex.py --validate source/_static/firedrake-apps.bib
	../firedrake/scripts/firedrake_preprocess_bibtex.py --validate source/_static/references.bib

livehtml:
	sphinx-autobuild -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html

serve: html
	cd $(BUILDDIR)/html; python -m SimpleHTTPServer $(PORT)

.PHONY: copy_demos

source/demos: copy_demos

copy_demos:
	install -d source/demos
	cp ../demos/*/*.rst ../demos/demo_references.bib ../demos/*/*.geo ../demos/*/*.msh ../demos/*/*.png source/demos
	for file in source/demos/*.py.rst; do pylit $$file; done

reset_notebooks:
	jupyter-nbconvert --clear-output ./notebooks/*.ipynb
	env OMP_NUM_THREADS=1 jupyter-nbconvert \
		--execute \
		--ClearMetadataPreprocessor.enabled=True \
		--allow-errors \
		--to notebook \
		--inplace \
		./notebooks/*.ipynb

source/element_list.csv: source/element_list.py
	cd source; python element_list.py

source/team.rst: source/team2.py
	cd source; python team2.py; mv AUTHORS.rst CITATION.rst ../..

source/firedrake_citations.rst:
	sphinx-apidoc -f -T -o source/ ../firedrake_citations
	echo -e ":orphan:\n\n" | cat - source/firedrake_citations.rst > source/firedrake_citations.tmp
	mv source/firedrake_citations.tmp source/firedrake_citations.rst

apidoc: $(GENERATED_FILES) copy_demos
	sphinx-apidoc -f -T -o source/ ../firedrake ../firedrake/cython/*.so

clean:
	-rm -rf $(BUILDDIR)/*
	-rm -rf $(GENERATED_FILES)

buildhtml: apidoc
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html

html: apidoc buildhtml
	cp source/demos/*.py source/demos/*.geo source/demos/*.msh $(BUILDDIR)/html/demos
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

dirhtml: apidoc
	$(SPHINXBUILD) -b dirhtml $(ALLSPHINXOPTS) $(BUILDDIR)/dirhtml
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/dirhtml."

singlehtml: apidoc
	$(SPHINXBUILD) -b singlehtml $(ALLSPHINXOPTS) $(BUILDDIR)/singlehtml
	@echo
	@echo "Build finished. The HTML page is in $(BUILDDIR)/singlehtml."

pickle: apidoc
	$(SPHINXBUILD) -b pickle $(ALLSPHINXOPTS) $(BUILDDIR)/pickle
	@echo
	@echo "Build finished; now you can process the pickle files."

json: apidoc
	$(SPHINXBUILD) -b json $(ALLSPHINXOPTS) $(BUILDDIR)/json
	@echo
	@echo "Build finished; now you can process the JSON files."

htmlhelp: apidoc
	$(SPHINXBUILD) -b htmlhelp $(ALLSPHINXOPTS) $(BUILDDIR)/htmlhelp
	@echo
	@echo "Build finished; now you can run HTML Help Workshop with the" \
	      ".hhp project file in $(BUILDDIR)/htmlhelp."

qthelp: apidoc
	$(SPHINXBUILD) -b qthelp $(ALLSPHINXOPTS) $(BUILDDIR)/qthelp
	@echo
	@echo "Build finished; now you can run "qcollectiongenerator" with the" \
	      ".qhcp project file in $(BUILDDIR)/qthelp, like this: apidoc"
	@echo "# qcollectiongenerator $(BUILDDIR)/qthelp/Firedrake.qhcp"
	@echo "To view the help file: apidoc"
	@echo "# assistant -collectionFile $(BUILDDIR)/qthelp/Firedrake.qhc"

devhelp: apidoc
	$(SPHINXBUILD) -b devhelp $(ALLSPHINXOPTS) $(BUILDDIR)/devhelp
	@echo
	@echo "Build finished."
	@echo "To view the help file:"
	@echo "# mkdir -p $$HOME/.local/share/devhelp/Firedrake"
	@echo "# ln -s $(BUILDDIR)/devhelp $$HOME/.local/share/devhelp/Firedrake"
	@echo "# devhelp"

epub: apidoc
	$(SPHINXBUILD) -b epub $(ALLSPHINXOPTS) $(BUILDDIR)/epub
	@echo
	@echo "Build finished. The epub file is in $(BUILDDIR)/epub."

latex: apidoc
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo
	@echo "Build finished; the LaTeX files are in $(BUILDDIR)/latex."
	@echo "Run \`make' in that directory to run these through (pdf)latex" \
	      "(use \`make latexpdf' here to do that automatically)."

latexpdf: apidoc
	$(SPHINXBUILD) -b latex $(ALLSPHINXOPTS) $(BUILDDIR)/latex
	@echo "Running LaTeX files through pdflatex..."
	$(MAKE) -C $(BUILDDIR)/latex all-pdf
	@echo "pdflatex finished; the PDF files are in $(BUILDDIR)/latex."

text: apidoc
	$(SPHINXBUILD) -b text $(ALLSPHINXOPTS) $(BUILDDIR)/text
	@echo
	@echo "Build finished. The text files are in $(BUILDDIR)/text."

man: apidoc
	$(SPHINXBUILD) -b man $(ALLSPHINXOPTS) $(BUILDDIR)/man
	@echo
	@echo "Build finished. The manual pages are in $(BUILDDIR)/man."

texinfo: apidoc
	$(SPHINXBUILD) -b texinfo $(ALLSPHINXOPTS) $(BUILDDIR)/texinfo
	@echo
	@echo "Build finished. The Texinfo files are in $(BUILDDIR)/texinfo."
	@echo "Run \`make' in that directory to run these through makeinfo" \
	      "(use \`make info' here to do that automatically)."

info: apidoc
	$(SPHINXBUILD) -b texinfo $(ALLSPHINXOPTS) $(BUILDDIR)/texinfo
	@echo "Running Texinfo files through makeinfo..."
	make -C $(BUILDDIR)/texinfo info
	@echo "makeinfo finished; the Info files are in $(BUILDDIR)/texinfo."

gettext: apidoc
	$(SPHINXBUILD) -b gettext $(I18NSPHINXOPTS) $(BUILDDIR)/locale
	@echo
	@echo "Build finished. The message catalogs are in $(BUILDDIR)/locale."

changes: apidoc
	$(SPHINXBUILD) -b changes $(ALLSPHINXOPTS) $(BUILDDIR)/changes
	@echo
	@echo "The overview file is in $(BUILDDIR)/changes."

linkcheck: apidoc
	$(SPHINXBUILD) -b linkcheck $(ALLSPHINXOPTS) $(BUILDDIR)/linkcheck
	@echo
	@echo "Link check complete; look for any errors in the above output " \
	      "or in $(BUILDDIR)/linkcheck/output.txt."

doctest: apidoc
	$(SPHINXBUILD) -b doctest $(ALLSPHINXOPTS) $(BUILDDIR)/doctest
	@echo "Testing of doctests in the sources finished, look at the " \
	      "results in $(BUILDDIR)/doctest/output.txt."
