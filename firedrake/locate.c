#include <evaluate.h>

PetscErrorCode locate_cell_from_candidates(struct Function *f,
        double *x,
        ref_cell_l1_dist try_candidate,
        ref_cell_l1_dist_xtr try_candidate_xtr,
        void *temp_ref_coords,
        void *found_ref_coords,
        PetscReal *found_ref_cell_dist_l1,
        size_t nids,
        const int64_t *ids,
        size_t ncells_ignore,
        const PetscInt *cells_ignore,
        const PetscInt *cell_owner_ranks,
        PetscInt *cell_out,
        PetscInt *owner_out)
{
    bool cell_ignore_found = false;
    /* NOTE: temp_ref_coords and found_ref_coords are actually of type
    struct ReferenceCoords but can't be declared as such in the function
    signature because the dimensions of the reference coordinates in the
    ReferenceCoords struct are defined by python when the code which
    surrounds this is declared in pointquery_utils.py. We cast when we use the
    ref_coords_copy function and trust that the underlying memory which the
    pointers refer to is updated as necessary. */
    PetscReal best_distance = PETSC_MAX_REAL;
    PetscInt best_owner = -1;
    PetscInt best_cell = -1;
    /* NOTE: `tolerance`, which is used throughout this function, is a static
       variable defined outside this function when putting together all the C
       code that needs to be compiled - see pointquery_utils.py */

    *cell_out = -1;
    if (owner_out) {
        *owner_out = -1;
    }
    for (size_t i = 0; i < nids; ++i) {
        /* Check that casting the ids from int64 to PetscInt is safe (for 32 bit petsc builds). 
        Since the ids are mesh cell ids this *should* always be safe, but better to check
        and return an error if something goes wrong. */
        if (ids[i] > (int64_t)PETSC_MAX_INT) {
            return PETSC_ERR_ARG_OUTOFRANGE;
        }
        PetscInt candidate = (PetscInt)ids[i];
        for (size_t j = 0; j < ncells_ignore; j++) {
            if (candidate == cells_ignore[j]) {
                cell_ignore_found = true;
                break;
            }
        }
        if (cell_ignore_found) {
            cell_ignore_found = false;
            continue;
        }

        PetscReal distance;
        PetscInt owner;
        if (f->extruded) {
            PetscInt nlayers = f->n_layers;
            PetscInt c = candidate / nlayers;
            PetscInt l = candidate % nlayers;
            distance = (*try_candidate_xtr)(temp_ref_coords, f, c, l, x);
            owner = cell_owner_ranks ? cell_owner_ranks[c] : 0;
        }
        else {
            distance = (*try_candidate)(temp_ref_coords, f, candidate, x);
            owner = cell_owner_ranks ? cell_owner_ranks[candidate] : 0;
        }
        /* Select owning cell by minimum L1 distance, with ties broken by highest owning rank. */
        if (distance < best_distance || (distance == best_distance && owner > best_owner)) {
            best_distance = distance;
            best_owner = owner;
            best_cell = candidate;
            memcpy(found_ref_coords, temp_ref_coords, sizeof(struct ReferenceCoords));
        }
    }

    if (best_cell != -1 && (best_distance <= 0.0 || best_distance < tolerance)) {
        *cell_out = best_cell;
        if (owner_out) {
            *owner_out = best_owner;
        }
        *found_ref_cell_dist_l1 = best_distance;
    }
    return PETSC_SUCCESS;
}
