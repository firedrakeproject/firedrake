#include <stdio.h>
#include <stdlib.h>
#include <spatialindex/capi/sidx_api.h>
#ifdef COMPUTE_DISTANCE_TO_CELL
#include <float.h>
#include <assert.h>
#endif
#include <evaluate.h>

int locate_cell(struct Function *f,
        double *x,
        int dim,
        inside_predicate try_candidate,
        inside_predicate_xtr try_candidate_xtr,
        void *data_,
        double tolerance)
{
    RTError err;
    int cell = -1;
    /* COMPUTE_DISTANCE_TO_CELL is defined when we provide a
       compute_distance_to_cell function. This is done in pointquery_utils.py */
#ifdef COMPUTE_DISTANCE_TO_CELL
    /* Assume that data_ is a ReferenceCoords object */
    struct ReferenceCoords *ref_coords = (struct ReferenceCoords *) data_;
    double closest_ref_coord = DBL_MAX;
    double current_closest_ref_coord =  -0.5;
#endif

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
                if ((*try_candidate)(data_, f, ids[i], x)) {
                    /* Found cell! */
                    cell = ids[i];
                    break;
                }
#ifdef COMPUTE_DISTANCE_TO_CELL
                else {
                    /* Cell not found, but could be on cell boundary. We therefore look
                       at our reference coordinates and find the point closest to being
                       inside the reference cell. If we don't find a cell using try_candidate
                       we assume that this process has found our cell. */
                    /* We use the compute_distance_to_cell function prepended to this file
                       which is specialised to the reference cell the local coordinates are
                       defined on. The function returns a negative value if the point is
                       inside the reference cell. Note that ref_coords was updated as data_
                       by try_candidate. */
                    current_closest_ref_coord = compute_distance_to_cell(ref_coords->X, dim);
                    /* If current_closest_ref_coord were in the reference cell it would
-                       already have been found! */
                    assert(0.0 < current_closest_ref_coord);
                    if (current_closest_ref_coord < closest_ref_coord && current_closest_ref_coord < tolerance) {
                        closest_ref_coord = current_closest_ref_coord;
                        cell = ids[i];
                    }
                }
#endif
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
#ifdef COMPUTE_DISTANCE_TO_CELL
                else {
                    /* As above, but for the case where we don't have a spatial index. */
                    current_closest_ref_coord = compute_distance_to_cell(ref_coords->X, dim);
                    assert(0.0 < current_closest_ref_coord);
                    if (current_closest_ref_coord < closest_ref_coord && current_closest_ref_coord < tolerance) {
                        closest_ref_coord = current_closest_ref_coord;
                        cell = c;
                    }
                }
#endif
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
