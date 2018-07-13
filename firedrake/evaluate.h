#ifndef _EVALUATE_H
#define _EVALUATE_H

#include <petsc.h>

#ifdef __cplusplus
extern "C" {
#endif

struct Function {
	/* Number of cells in the base mesh */
	int n_cols;

	/* 1 if extruded, 0 if not */
	int extruded;

	/* number of layers for extruded, otherwise 1 */
	int n_layers;

	/* Coordinate values and node mapping */
	double *coords;
	PetscInt *coords_map;

	/* Field values and node mapping */
	double *f;
	PetscInt *f_map;

	/* Spatial index */
	void *sidx;

	/*
	 * TODO:
	 * - cell orientation
	 */
};

typedef int (*inside_predicate)(void *data_,
				struct Function *f,
				int cell,
				double *x);

typedef int (*inside_predicate_xtr)(void *data_,
				struct Function *f,
				int cell,
				int layer,
				double *x);

extern int locate_cell(struct Function *f,
		       double *x,
		       int dim,
		       inside_predicate try_candidate,
		       inside_predicate_xtr try_candidate_xtr,
		       void *data_);

extern int evaluate(struct Function *f,
		    double *x,
		    double *result);

#ifdef __cplusplus
}
#endif

#endif /* _EVALUATE_H */
