#include <stdio.h>
#include <stdlib.h>
#include <rstar-capi.h>
#include <float.h>
#include <evaluate.h>

int locate_cell(struct Function *f,
        double *x,
        int dim,
        ref_cell_l1_dist try_candidate,
        ref_cell_l1_dist_xtr try_candidate_xtr,
        void *temp_ref_coords,
        void *found_ref_coords,
        double *found_ref_cell_dist_l1,
        size_t ncells_ignore,
        int* cells_ignore)
{
    RTreeError err;
    int cell = -1;
    int cell_ignore_found = 0;
    /* NOTE: temp_ref_coords and found_ref_coords are actually of type
    struct ReferenceCoords but can't be declared as such in the function
    signature because the dimensions of the reference coordinates in the
    ReferenceCoords struct are defined by python when the code which
    surrounds this is declared in pointquery_utils.py. We cast when we use the
    ref_coords_copy function and trust that the underlying memory which the
    pointers refer to is updated as necessary. */
    double ref_cell_dist_l1 = DBL_MAX;
    double current_ref_cell_dist_l1 =  -0.5;
    /* NOTE: `tolerance`, which is used throughout this funciton, is a static
       variable defined outside this function when putting together all the C
       code that needs to be compiled - see pointquery_utils.py */

    if (f->sidx) {
        size_t *ids = NULL;
        size_t nids = 0;
        err = rtree_locate_all_at_point((const struct RTreeH *)f->sidx, x, &ids, &nids);
        if (err != Success) {
            fputs("ERROR: RTree_LocateAllAtPoint failed in rstar-capi!\n", stderr);
            return -1;
        }
        if (f->extruded == 0) {
            for (size_t i = 0; i < nids; i++) {
                current_ref_cell_dist_l1 = (*try_candidate)(temp_ref_coords, f, ids[i], x);
                for (size_t j = 0; j < ncells_ignore; j++) {
                    if (ids[i] == cells_ignore[j]) {
                        cell_ignore_found = 1;
                        break;
                    }
                }
                if (cell_ignore_found) {
                    cell_ignore_found = 0;
                    continue;
                }
                if (current_ref_cell_dist_l1 <= 0.0) {
                    /* Found cell! */
                    cell = ids[i];
                    memcpy(found_ref_coords, temp_ref_coords, sizeof(struct ReferenceCoords));
                    found_ref_cell_dist_l1[0] = current_ref_cell_dist_l1;
                    break;
                }
                else if (current_ref_cell_dist_l1 < ref_cell_dist_l1) {
                    /* getting closer... */
                    ref_cell_dist_l1 = current_ref_cell_dist_l1;
                    if (ref_cell_dist_l1 < tolerance) {
                        /* Close to cell within tolerance so could be this cell */
                        cell = ids[i];
                        memcpy(found_ref_coords, temp_ref_coords, sizeof(struct ReferenceCoords));
                        found_ref_cell_dist_l1[0] = ref_cell_dist_l1;
                    }
                }
            }
        }
        else {
            for (size_t i = 0; i < nids; i++) {
                int nlayers = f->n_layers;
                int c = ids[i] / nlayers;
                int l = ids[i] % nlayers;
                current_ref_cell_dist_l1 = (*try_candidate_xtr)(temp_ref_coords, f, c, l, x);
                for (size_t j = 0; j < ncells_ignore; j++) {
                    if (ids[i] == cells_ignore[j]) {
                        cell_ignore_found = 1;
                        break;
                    }
                }
                if (cell_ignore_found) {
                    cell_ignore_found = 0;
                    continue;
                }
                if (current_ref_cell_dist_l1 <= 0.0) {
                    /* Found cell! */
                    cell = ids[i];
                    memcpy(found_ref_coords, temp_ref_coords, sizeof(struct ReferenceCoords));
                    found_ref_cell_dist_l1[0] = current_ref_cell_dist_l1;
                    break;
                }
                else if (current_ref_cell_dist_l1 < ref_cell_dist_l1) {
                    /* getting closer... */
                    ref_cell_dist_l1 = current_ref_cell_dist_l1;
                    if (ref_cell_dist_l1 < tolerance) {
                        /* Close to cell within tolerance so could be this cell */
                        cell = ids[i];
                        memcpy(found_ref_coords, temp_ref_coords, sizeof(struct ReferenceCoords));
                        found_ref_cell_dist_l1[0] = ref_cell_dist_l1;
                    }
                }
            }
        }
        rtree_free_ids(ids, nids);
    }
    return cell;
}
