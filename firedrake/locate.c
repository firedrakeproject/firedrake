#include <stdio.h>
#include <stdlib.h>
#include <spatialindex/capi/sidx_api.h>

#include <evaluate.h>

int locate_cell(struct Function *f,
        double *x,
        int dim,
        inside_predicate try_candidate,
        inside_predicate_xtr try_candidate_xtr,
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
        if (f->extruded == 0) {
            for (int i = 0; i < nids; i++) {
                if ((*try_candidate)(data_, f, ids[i], x)) {
                    cell = ids[i];
                    break;
                }
            }
        }
        else {
            for (int i = 0; i < nids; i++) {
                int nlayers = f->n_layers;
                int c = ids[i] / nlayers;
                int l = ids[i] % nlayers;
                if ((*try_candidate_xtr)(data_, f, c, l, x)) {
                    cell = ids[i];
                    break;
                }
            }
        }
        free(ids);
    } else {
        if (f->extruded == 0) {
            for (int c = 0; c < f->n_cols; c++) {
                if ((*try_candidate)(data_, f, c, x)) {
                    cell = c;
                    break;
                }
            }
        }
        else {
            for (int c = 0; c < f->n_cols; c++) {
                for (int l = 0; l < f->n_layers; l++)
                    if ((*try_candidate_xtr)(data_, f, c, l, x)) {
                        cell = l;
                        break;
                    }
                if (cell != -1) {
                    cell = c * f->n_layers + cell;
                    break;
                }
            }
        }
    }
    return cell;
}
