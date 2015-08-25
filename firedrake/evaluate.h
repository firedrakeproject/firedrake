#ifndef _EVALUATE_H
#define _EVALUATE_H


struct Function {
	/* Number of cells in mesh */
	int n_cells;

	/* Coordinate values and node mapping */
	double *coords;
	int *coords_map;

	/* Field values and node mapping */
	double *f;
	int *f_map;

	/*
	 * TODO:
	 * - cell orientation
	 * - spatial index
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


/* Is point inside the domain? */
#define inside(f, x) (!evaluate((f), (x), NULL))


#endif /* _EVALUATE_H */
