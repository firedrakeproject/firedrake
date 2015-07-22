#include <locate.h>
#include <function.h>

int locate_cell(struct Function *f, double *x,
		inside_p try_candidate, void *data_)
{
	int c;
	for (c = 0; c < f->n_cells; c++) {
		if ((*try_candidate)(data_, f, c, x)) {
			return c;
		}
	}
	return -1;
}
