WGET = wget --no-check-certificate
BASEURL = https://spo.doc.ic.ac.uk/meshes/
PROCS = 0 1 2
MMS_MESHES = $(foreach mesh, MMS_A MMS_B MMS_C MMS_D, $(foreach proc, $(PROCS), $(mesh).$(proc).pickle.gz))
HDF5_MESHES = new_grid.h5 FE_grid.h5
TRIANGLE_MESHES = $(foreach mesh, small medium large, $(foreach ext, edge ele node, $(mesh).$(ext)))

.PHONY : meshes

%.pickle.gz %.h5:
	$(WGET) $(BASEURL)$@

small.%:
	./generate_mesh small 10

medium.%:
	./generate_mesh medium 20

large.%:
	./generate_mesh large 40

meshes: $(MMS_MESHES) $(HDF5_MESHES) $(TRIANGLE_MESHES)
