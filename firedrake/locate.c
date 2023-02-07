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
        void *temp_ref_coords,
        void *found_ref_coords)
{
    RTError err;
    int cell = -1;
    /* NOTE: temp_ref_coords and found_ref_coords are actually of type
    struct ReferenceCoords but can't be declared as such in the function
    signature because the dimensions of the reference coordinates in the
    ReferenceCoords struct are defined by python when the code which
    surrounds this is declared in pointquery_utils.py. We cast when we use the
    ref_coords_copy function and trust that the underlying memory which the
    pointers refer to is updated as necessary. */
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
                current_closest_ref_coord = (*try_candidate)(temp_ref_coords, f, ids[i], x);
                if (current_closest_ref_coord <= 0.0) {
                    /* Found cell! */
                    cell = ids[i];
                    memcpy(found_ref_coords, temp_ref_coords, sizeof(struct ReferenceCoords));
                    break;
                }
                else if (current_closest_ref_coord < closest_ref_coord) {
                    /* getting closer... */
                    closest_ref_coord = current_closest_ref_coord;
                    if (closest_ref_coord < tolerance) {
                        /* Close to cell within tolerance so could be this cell */
                        cell = ids[i];
                        memcpy(found_ref_coords, temp_ref_coords, sizeof(struct ReferenceCoords));
                    }
                }
            }
        }
        else {
            for (uint64_t i = 0; i < nids; i++) {
                int nlayers = f->n_layers;
                int c = ids[i] / nlayers;
                int l = ids[i] % nlayers;
                current_closest_ref_coord = (*try_candidate_xtr)(temp_ref_coords, f, c, l, x);
                if (current_closest_ref_coord <= 0.0) {
                    /* Found cell! */
                    cell = ids[i];
                    memcpy(found_ref_coords, temp_ref_coords, sizeof(struct ReferenceCoords));
                    break;
                }
                else if (current_closest_ref_coord < closest_ref_coord) {
                    /* getting closer... */
                    closest_ref_coord = current_closest_ref_coord;
                    if (closest_ref_coord < tolerance) {
                        /* Close to cell within tolerance so could be this cell */
                        cell = ids[i];
                        memcpy(found_ref_coords, temp_ref_coords, sizeof(struct ReferenceCoords));
                    }
                }
            }
        }
        free(ids);
    } else {
        if (f->extruded == 0) {
            for (int c = 0; c < f->n_cols; c++) {
                current_closest_ref_coord = (*try_candidate)(temp_ref_coords, f, c, x);
                if (current_closest_ref_coord <= 0.0) {
                    /* Found cell! */
                    cell = c;
                    memcpy(found_ref_coords, temp_ref_coords, sizeof(struct ReferenceCoords));
                    break;
                }
                else if (current_closest_ref_coord < closest_ref_coord) {
                    /* getting closer... */
                    closest_ref_coord = current_closest_ref_coord;
                    if (closest_ref_coord < tolerance) {
                        /* Close to cell within tolerance so could be this cell */
                        cell = c;
                        memcpy(found_ref_coords, temp_ref_coords, sizeof(struct ReferenceCoords));
                    }
                }
            }
        }
        else {
            for (int c = 0; c < f->n_cols; c++) {
                for (int l = 0; l < f->n_layers; l++) {
                    current_closest_ref_coord = (*try_candidate_xtr)(temp_ref_coords, f, c, l, x);
                    if (current_closest_ref_coord <= 0.0) {
                        /* Found cell! */
                        cell = l;
                        memcpy(found_ref_coords, temp_ref_coords, sizeof(struct ReferenceCoords));
                        break;
                    }
                    else if (current_closest_ref_coord < closest_ref_coord) {
                        /* getting closer... */
                        closest_ref_coord = current_closest_ref_coord;
                        if (closest_ref_coord < tolerance) {
                            /* Close to cell within tolerance so could be this cell */
                            cell = l;
                            memcpy(found_ref_coords, temp_ref_coords, sizeof(struct ReferenceCoords));
                        }
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
