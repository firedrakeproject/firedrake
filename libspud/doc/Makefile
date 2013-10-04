default: spud_manual.pdf

%.pdf: %.tex
	pdflatex $^
	if fgrep "Rerun to" $*.log; then $(MAKE) --assume-new $^ $@;fi
ifndef NOBIBTEX
	if fgrep "There were undefined"  $*.log;\
	then bibtex $*; \
	$(MAKE) NOBIBTEX=true --assume-new $^ $@;fi
endif

# We don't rm *.ps *.pdf because that might clobber image files. Instead, put any
# ps files which should go in $(PSFILES)
clean:
	rm $(TARGETS) $(PSFILES) *.eps *.dvi *.aux *.log *.bbl *.blg *.out *.toc 2>/dev/null||true
