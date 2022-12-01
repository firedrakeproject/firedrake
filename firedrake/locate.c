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
        /* Assume that data_ is a ReferenceCoords object */
        struct ReferenceCoords *ref_coords = (struct ReferenceCoords *) data_;
        double closest_ref_coord = 0.0;
        double current_closest_ref_coord = 0.5;
        /* We treat our list of candidate cells (ids) from libspatialindex's
            Index_Intersects_id  as our source of truth: the point must be in
            one of the cells. */
        err = Index_Intersects_id(f->sidx, x, x, dim, &ids, &nids);
        if (err != RT_None) {
            fputs("ERROR: Index_Intersects_id failed in libspatialindex!", stderr);
            return -1;
        }
        if (f->extruded == 0) {
            for (uint64_t i = 0; i < nids; i++) {
                if ((*try_candidate)(data_, f, ids[i], x)) {
                    /* Found cell! */
                    cell = ids[i];
                    break;
                }
                else {
                    /* Cell not found, but could be on cell boundary. We therefore look
                       at our reference coordinates and find the point closest to being
                       inside the reference cell. If we don't find a cell using try_candidate
                       we assume that this process has found our cell. */
                    /* First we must find the the reference coordinate which is closest to
                       the interval [0, 1] without being in it (if all reference coordinates
                       were in the interval we would have already found our cell!). We do
                       this by storing the distance of each coordinate dimension from the
                       interval. Note that ref_coords was updated as data_ by try_candidate. */
                    current_closest_ref_coord = -1.0;
                    for (uint64_t j = 0; j < dim; j++) {
                        if(ref_coords->X[j] < 0.0 && current_closest_ref_coord < ref_coords->X[j]) {
                            current_closest_ref_coord = ref_coords->X[j];
                        }
                        else if(1.0 - ref_coords->X[j] < 0.0 && current_closest_ref_coord < 1.0 - ref_coords->X[j]) {
                            current_closest_ref_coord = 1.0 - ref_coords->X[j];
                        }
                    }
                    /* Then, if it's lower than the last, we assume we must have found our
                       cell until told otherwise */
                    if(current_closest_ref_coord < closest_ref_coord){
                        closest_ref_coord = current_closest_ref_coord;
                        cell = ids[i];
                    }
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
