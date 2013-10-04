IMAGES=linear_1d quadratic_1d linear_1d_mesh linear_1d_basis mesh \
cq_stencil quadratic_1d_basis numerical_integration linear_1d_derivative \
linear_1d_galerkin local_coords_tri triangle_numbering quad_numbering \
hex_numbering tet_numbering interval_numbering



femtools_manual.pdf: femtools_manual.tex notation.tex \
	$(addsuffix .pdftex_t, $(IMAGES)) \
	$(addsuffix _tex.pdf, $(IMAGES)) \

femtools_manual.html: femtools_manual.tex notation.tex \
	$(addsuffix .png, $(IMAGES))
	htlatex $< "femtools_manual"

#linear_1d_mesh.png: linear_1d_mesh.fig

%.tex: %.pstex_t %.pstex
	./bin/wrap_pstex $<

%.dvi: %.tex
	pdflatex -output-format dvi $<

%.png: %.dvi
	dvipng -T tight  -D 100 -bg Transparent $<  -o $@

PSFILES=$(addsuffix .pdftex_t, $(IMAGES)) \
	$(addsuffix _tex.pdf, $(IMAGES)) \
	$(addsuffix .png, $(IMAGES))


%.ps: %.dvi
	dvips -o $@ $^

%.pdf: %.tex
	pdflatex -shell-escape $<
	if fgrep "Rerun to" $*.log; then $(MAKE) --assume-new $^ $@;fi
ifndef NOBIBTEX
	if fgrep "There were undefined"  $*.log;\
	then bibtex $*; \
	$(MAKE) NOBIBTEX=true --assume-new $^ $@;fi
endif

bibliography.bib:
	ln -s $(CD)$@ .

%.pdftex: %.fig
	fig2dev -L pdftex $^ $@

%_tex.pdf: %.fig
	fig2dev -L pdftex $^ $@

%.pdftex_t: %.fig
	fig2dev -L pdftex_t -p $*_tex.pdf $^ $@

%.pstex: %.fig
	fig2dev -L pstex $^ $@

%.pstex_t: %.fig
	fig2dev -L pstex_t -p $*.pstex $^ $@

%.pdf: %.fig
	fig2dev -L pdf $^ $@

#%.png: %.fig
#	fig2dev -L png $^ $@

.PHONY: clean

# We don't rm *.ps because that might clobber image files. Instead, put any
# ps files which should go in $(PSFILES)
clean: htclean_femtools_manual
	rm $(TARGETS) $(PSFILES) *.dvi *.aux *.log *.bbl *.blg *.toc \
*.out *.pdftex *_tex.pdf *.pdftex_t *.pstex *.pstex_t *.png 2>/dev/null||true

htclean_%:
	rm $*.4ct $*.4tc $*.css $*.idv $*.lg $*.tmp $*.xref *.html 2>/dev/null||true