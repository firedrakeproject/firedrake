#ifndef _FUNCTION_H
#define _FUNCTION_H

struct Function {
	int n_cells;
	double *coords;
	int *coords_map;
	double *f;
	int *f_map;
};

#endif /* _FUNCTION_H */
