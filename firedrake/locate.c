#include <stdio.h>
#include <stdlib.h>
#include <spatialindex/capi/sidx_api.h>

#include <evaluate.h>

int locate_cell(struct Function *f,
		double *x,
		int dim,
		inside_predicate try_candidate,
		void *data_)
{
	RTError err;
	int cell = -1;

	if (f->sidx) {
		int64_t *ids = NULL;
		uint64_t nids = 0;
		err = Index_Intersects_id(f->sidx, x, x, dim, &ids, &nids);
		if (err != RT_None) {
			fputs("ERROR: Index_Intersects_id failed in libspatialindex!", stderr);
			return -1;
		}

		for (int i = 0; i < nids; i++) {
			if ((*try_candidate)(data_, f, ids[i], x)) {
				cell = ids[i];
				break;
			}
		}
		free(ids);
	} else {
		for (int c = 0; c < f->n_cols * f->n_layers; c++)
			if ((*try_candidate)(data_, f, c, x)) {
				cell = c;
				break;
			}
	}
	return cell;
}
