WGET = wget --no-check-certificate
BASEURL = https://spo.doc.ic.ac.uk/meshes/
PROCS = 0 1 2
MESHES = MMS_A MMS_B MMS_C MMS_D
FILES = $(foreach mesh, $(MESHES), $(foreach proc, $(PROCS), $(mesh).$(proc).pickle.gz))

input: clean $(FILES)
	@echo $(FILES)

%.pickle.gz:
	$(WGET) $(BASEURL)$@

.PHONY: clean input
clean:
	@rm -f *.out *.pyc
