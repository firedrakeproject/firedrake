#ifndef _EVALUATE_H
#define _EVALUATE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <petscsys.h>
#include <petscerror.h>
#include <rtree-capi.h>

#ifdef __cplusplus
extern "C" {
#endif

struct Function {
	/* Coordinate values and node mapping */
	PetscScalar *coords;
	PetscInt *coords_map;

	/* Field values and node mapping */
	PetscScalar *f;
	PetscInt *f_map;

	/* rtree */
	void *rtree;

	/*
	 * TODO:
	 * - cell orientation
	 */
};

typedef PetscReal (*ref_cell_l1_dist)(void *data_,
				struct Function *f,
				PetscInt cell,
				double *x);

extern PetscErrorCode locate_cell_from_coordinates(struct Function *f,
		       double *x,
		       ref_cell_l1_dist try_candidate,
		       void *temp_ref_coords,
		       void *found_ref_coords,
		       PetscReal *found_ref_cell_dist_l1,
			   size_t nids,
			   const int64_t *ids,
			   size_t ncells_ignore,
			   const PetscInt *cells_ignore,
			   PetscInt *cell_out);

extern int evaluate(struct Function *f,
		    double *x,
		    PetscScalar *result);

#ifdef __cplusplus
}
#endif

#endif /* _EVALUATE_H */
