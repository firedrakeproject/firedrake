#include <evaluate.h>

int locate_cell(struct Function *f,
		double *x,
		int dim,
		inside_predicate try_candidate,
		void *data_)
{
	for (int c = 0; c < f->n_cols * f->n_layers; c++)
		if ((*try_candidate)(data_, f, c, x))
			return c;
	return -1;
}
