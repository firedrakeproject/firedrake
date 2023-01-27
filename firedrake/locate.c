#include <stdio.h>
#include <stdlib.h>
#include <spatialindex/capi/sidx_api.h>
#include <float.h>
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
    /* Assume that data_ is a ReferenceCoords object */
    struct ReferenceCoords *ref_coords = (struct ReferenceCoords *) data_;
    double closest_ref_coord = DBL_MAX;
    double current_closest_ref_coord =  -0.5;
    /* NOTE: `tolerance`, which is used throughout this funciton, is a static
       variable defined outside this function when putting together all the C
       code that needs to be compiled - see pointquery_utils.py */

    if (f->sidx) {
        int64_t *ids = NULL;
        uint64_t nids = 0;
        /* We treat our list of candidate cells (ids) from libspatialindex's
            Index_Intersects_id as our source of truth: the point must be in
            one of the cells. */
        err = Index_Intersects_id(f->sidx, x, x, dim, &ids, &nids);
        if (err != RT_None) {
            fputs("ERROR: Index_Intersects_id failed in libspatialindex!", stderr);
            return -1;
        }
        if (f->extruded == 0) {
            for (uint64_t i = 0; i < nids; i++) {
                current_closest_ref_coord = (*try_candidate)(data_, f, ids[i], x);
                if (current_closest_ref_coord <= 0.0) {
                    /* Found cell! */
                    cell = ids[i];
                    break;
                }
                else if (current_closest_ref_coord < closest_ref_coord && current_closest_ref_coord < tolerance) {
                    /* Close to cell within tolerance so could be this cell */
                    closest_ref_coord = current_closest_ref_coord;
                    cell = ids[i];
                }
            }
        }
        else {
            for (uint64_t i = 0; i < nids; i++) {
                int nlayers = f->n_layers;
                int c = ids[i] / nlayers;
                int l = ids[i] % nlayers;
                current_closest_ref_coord = (*try_candidate_xtr)(data_, f, c, l, x);
                if (current_closest_ref_coord <= 0.0) {
                    /* Found cell! */
                    cell = ids[i];
                    break;
                }
                else if (current_closest_ref_coord < closest_ref_coord && current_closest_ref_coord < tolerance) {
                    /* Close to cell within tolerance so could be this cell */
                    closest_ref_coord = current_closest_ref_coord;
                    cell = ids[i];
                }
            }
        }
        free(ids);
    } else {
        if (f->extruded == 0) {
            for (int c = 0; c < f->n_cols; c++) {
                current_closest_ref_coord = (*try_candidate)(data_, f, c, x);
                if (current_closest_ref_coord <= 0.0) {
                    cell = c;
                    break;
                }
                else if (current_closest_ref_coord < closest_ref_coord && current_closest_ref_coord < tolerance) {
                    closest_ref_coord = current_closest_ref_coord;
                    cell = c;
                }
            }
        }
        else {
            for (int c = 0; c < f->n_cols; c++) {
                for (int l = 0; l < f->n_layers; l++) {
                    current_closest_ref_coord = (*try_candidate_xtr)(data_, f, c, l, x);
                    if (current_closest_ref_coord <= 0.0) {
                        cell = l;
                        break;
                    }
                    else if (current_closest_ref_coord < closest_ref_coord && current_closest_ref_coord < tolerance) {
                       closest_ref_coord = current_closest_ref_coord;
                        cell = l;
                    }
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
