#include <stdio.h>
#include <stdlib.h>
#include <spatialindex/capi/sidx_api.h>
#include <float.h>
#include <evaluate.h>

bool check_cell(int64_t current_cell,
                double current_ref_cell_dist_l1,
                int num_owned_cells,
                int64_t *found_cell,
                bool *found_cell_is_owned,
                void *found_ref_coords,
                void *temp_ref_coords,
                double *found_ref_cell_dist_l1)
{
    if (current_cell < num_owned_cells) {
        if (current_ref_cell_dist_l1 <= 0) {
            // found the cell
            *found_cell = current_cell;
            memcpy(found_ref_coords, temp_ref_coords, sizeof(struct ReferenceCoords));
            *found_ref_cell_dist_l1 = current_ref_cell_dist_l1;
            return true;
        }
        else if (current_ref_cell_dist_l1 < tolerance &&
                 current_ref_cell_dist_l1 < *found_ref_cell_dist_l1)
        {
            /* Close to cell within tolerance so could be this cell */
            *found_cell_is_owned = true;

            *found_cell = current_cell;
            memcpy(found_ref_coords, temp_ref_coords, sizeof(struct ReferenceCoords));
            *found_ref_cell_dist_l1 = current_ref_cell_dist_l1;
        }
    }
    else {
        if (!(*found_cell_is_owned) &&
            current_ref_cell_dist_l1 < tolerance &&
            current_ref_cell_dist_l1 < *found_ref_cell_dist_l1)
        {
            *found_cell = current_cell;
            memcpy(found_ref_coords, temp_ref_coords, sizeof(struct ReferenceCoords));
            *found_ref_cell_dist_l1 = current_ref_cell_dist_l1;
        }
    }
    return false;
}

int locate_cell(struct Function *f,
                double *x,
                int dim,
                int num_owned_cells,
                ref_cell_l1_dist try_candidate,
                ref_cell_l1_dist_xtr try_candidate_xtr,
                void *temp_ref_coords,
                void *found_ref_coords,
                double *found_ref_cell_dist_l1)
{
    /* NOTE: temp_ref_coords and found_ref_coords are actually of type
       struct ReferenceCoords but can't be declared as such in the function
       signature because the dimensions of the reference coordinates in the
       ReferenceCoords struct are defined by python when the code which
       surrounds this is declared in pointquery_utils.py. We cast when we use the
       ref_coords_copy function and trust that the underlying memory which the
       pointers refer to is updated as necessary. */
    /* NOTE: `tolerance`, which is used throughout this funciton, is a static
       variable defined outside this function when putting together all the C
       code that needs to be compiled - see pointquery_utils.py */

    *found_ref_cell_dist_l1 = DBL_MAX;

    RTError err;
    int64_t found_cell = -1;
    bool found_cell_is_owned = false;

    if (f->sidx) {
        /* We treat our list of candidate cells (ids) from libspatialindex's
           Index_Intersects_id as our source of truth: the point must be in
           one of the cells. */
        int64_t *ids = NULL;
        uint64_t nids = 0;
        err = Index_Intersects_id(f->sidx, x, x, dim, &ids, &nids);
        if (err != RT_None) {
            fputs("ERROR: Index_Intersects_id failed in libspatialindex!", stderr);
            return -1;
        }

        if (!f->extruded) {
            for (uint64_t i = 0; i < nids; i++) {
                int64_t current_cell = ids[i];
                double current_ref_cell_dist_l1 = (*try_candidate)(temp_ref_coords, f, current_cell, x);
                bool found = check_cell(current_cell,
                                        current_ref_cell_dist_l1,
                                        num_owned_cells,
                                        &found_cell,
                                        &found_cell_is_owned,
                                        found_ref_coords,
                                        temp_ref_coords,
                                        found_ref_cell_dist_l1);
                if (found) break;
            }
        }
        else {
            for (uint64_t i = 0; i < nids; i++) {
                int64_t current_cell = ids[i];

                int nlayers = f->n_layers;
                int c = current_cell / nlayers;
                int l = current_cell % nlayers;
                double current_ref_cell_dist_l1 = (*try_candidate_xtr)(temp_ref_coords, f, c, l, x);

                bool found = check_cell(current_cell,
                                        current_ref_cell_dist_l1,
                                        num_owned_cells,
                                        &found_cell,
                                        &found_cell_is_owned,
                                        found_ref_coords,
                                        temp_ref_coords,
                                        found_ref_cell_dist_l1);
                if (found) break;
            }
        }
        free(ids);
    } else {
        if (!f->extruded) {
            for (int c = 0; c < f->n_cols; c++) {
                double current_ref_cell_dist_l1 = (*try_candidate)(temp_ref_coords, f, c, x);
                bool found = check_cell(c,
                                        current_ref_cell_dist_l1,
                                        num_owned_cells,
                                        &found_cell,
                                        &found_cell_is_owned,
                                        found_ref_coords,
                                        temp_ref_coords,
                                        found_ref_cell_dist_l1);
                if (found) break;
            }
        }
        else {
            for (int c = 0; c < f->n_cols; c++) {
                for (int l = 0; l < f->n_layers; l++) {
                    int64_t current_cell = c * f->n_layers + l;
                    double current_ref_cell_dist_l1 = (*try_candidate_xtr)(temp_ref_coords, f, c, l, x);
                    bool found = check_cell(current_cell,
                                            current_ref_cell_dist_l1,
                                            num_owned_cells,
                                            &found_cell,
                                            &found_cell_is_owned,
                                            found_ref_coords,
                                            temp_ref_coords,
                                            found_ref_cell_dist_l1);
                    if (found) break;
                }
                if (found_cell != -1 && found_cell_is_owned) break;
            }
        }
    }
    return found_cell;
}
