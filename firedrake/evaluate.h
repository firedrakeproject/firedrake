#ifndef _EVALUATE_H
#define _EVALUATE_H

#include <petsc.h>

#ifdef __cplusplus
extern "C" {
#endif

struct Function {
	/* Number of cells */
	int n_cells;

	/* Coordinate values and node mapping */
	PetscScalar *coords;
	PetscInt *coords_map;

	/* Field values and node mapping */
	PetscScalar *f;
	PetscInt *f_map;
	int f_offset;

	/* Spatial index */
	void *sidx;

	/*
	 * TODO:
	 * - cell orientation
	 */
};

typedef PetscReal (*ref_cell_l1_dist)(void *data_,
				struct Function *f,
				int cell,
				double *x);

extern int locate_cell(struct Function *f,
		       double *x,
		       int dim,
		       ref_cell_l1_dist try_candidate,
		       void *temp_ref_coords,
		       void *found_ref_coords,
		       double *found_ref_cell_dist_l1,
			   size_t ncells_ignore,
			   int* cells_ignore);

extern int evaluate(struct Function *f,
		    double *x,
		    PetscScalar *result);

#ifdef __cplusplus
}
#endif

#endif /* _EVALUATE_H */
