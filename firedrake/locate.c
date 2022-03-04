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
printf("f->sidx\n");
        int64_t *ids = NULL;
        uint64_t nids = 0;
        err = Index_Intersects_id(f->sidx, x, x, dim, &ids, &nids);
        if (err != RT_None) {
            fputs("ERROR: Index_Intersects_id failed in libspatialindex!", stderr);
            return -1;
        }
        if (f->extruded == 0) {
            for (uint64_t i = 0; i < nids; i++) {
printf("i = %d\n", i);
printf("ids[i] = %d\n", ids[i]);
printf("x = [%f, %f]\n", x[0], x[1]);
int return_value;
struct ReferenceCoords result_[1];
wrap_to_reference_coords(result_, x, &return_value, ids[i], ids[i]+1, f->coords, f->coords_map);
printf("return_value = %d\n", return_value);
printf("X = [%f, %f]\n", result_->X[0], result_->X[1]);
                if ((*try_candidate)(data_, f, ids[i], x)) {
                    cell = ids[i];
printf("cell updated\n");
                    break;
                }
            }
        }
        else {
            for (uint64_t i = 0; i < nids; i++) {
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
printf("not f->sidx\n");
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
