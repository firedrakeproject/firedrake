#ifndef _EVALUATE_H
#define _EVALUATE_H

#ifdef __cplusplus
extern "C" {
#endif

struct Function {
	/* Number of cells in the base mesh */
	int n_cols;

	/* Number of layers for extruded, otherwise 1 */
	int n_layers;

	/* Coordinate values and node mapping */
	double *coords;
	int *coords_map;

	/* Field values and node mapping */
	double *f;
	int *f_map;

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


extern int locate_cell(struct Function *f,
		       double *x,
		       int dim,
		       inside_predicate try_candidate,
		       void *data_);

extern int evaluate(struct Function *f,
		    double *x,
		    double *result);

#ifdef __cplusplus
}
#endif

#endif /* _EVALUATE_H */
