MESH = wave_tank
SCALES = 1.0 0.707 0.577 0.5 0.408 0.354 0.316 0.289 0.25 0.204 0.177 0.144 0.125 0.102

MSH_MESHES := $(foreach scale, $(SCALES), $(MESH)_$(scale).msh)

all: $(MSH_MESHES)

$(MESH)_%.msh: $(MESH).geo
	gmsh -2 -clscale $* -o $@ $<
