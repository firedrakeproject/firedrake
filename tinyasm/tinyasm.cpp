#include <numeric>

#include <petsc.h>
#include <petsc/private/pcimpl.h>
#include <petsc/private/kernels/blockinvert.h>
#include <petscblaslapack.h>
#include <petsc/private/hashseti.h>
#include <petsc/private/hashmapi.h>
#include <petscpc.h>

#include <petsc4py/petsc4py.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


#define MY_PETSC_VERSION_LT(MAJOR,MINOR,SUBMINOR)       \
  (PETSC_VERSION_MAJOR < (MAJOR) ||                     \
   (PETSC_VERSION_MAJOR == (MAJOR) &&                   \
    (PETSC_VERSION_MINOR < (MINOR) ||                   \
     (PETSC_VERSION_MINOR == (MINOR) &&                 \
      (PETSC_VERSION_SUBMINOR < (SUBMINOR))))))

using namespace std;
namespace py = pybind11;

PetscLogEvent PC_tinyasm_SetASMLocalSubdomains, PC_tinyasm_apply, PC_tinyasm_setup;

PetscErrorCode mymatinvert(PetscInt* n, PetscScalar* mat, PetscInt* piv, PetscInt* info, PetscScalar* work);

class BlockJacobi {
    public:
        vector<vector<PetscInt>> dofsPerBlock;
        vector<vector<PetscInt>> globalDofsPerBlock;
        vector<PetscScalar> worka;
        vector<PetscScalar> workb;
        vector<PetscScalar> localb;
        vector<PetscScalar> localx;
        PetscSF sf;
        Mat *localmats_aij;
        Mat *localmats;

        vector<IS> dofis;
        vector<PetscInt> piv;
        vector<PetscScalar> fwork;

        BlockJacobi(vector<vector<PetscInt>> _dofsPerBlock, vector<vector<PetscInt>> _globalDofsPerBlock, int localSize, PetscSF _sf)
            : dofsPerBlock(_dofsPerBlock), globalDofsPerBlock(_globalDofsPerBlock), sf(_sf) {

                int numBlocks = dofsPerBlock.size();
                PetscInt dof;
                int biggestBlock = 0;
                for(int p=0; p<numBlocks; p++) {
                    dof = dofsPerBlock[p].size();
                    biggestBlock = max(biggestBlock, dof);
                }
                worka = vector<PetscScalar>(biggestBlock, 0);
                workb = vector<PetscScalar>(biggestBlock, 0);
                localb = vector<PetscScalar>(localSize, 0);
                localx = vector<PetscScalar>(localSize, 0);
                piv = vector<PetscInt>(biggestBlock, 0.);
                iota(piv.begin(), piv.end(), 1);
                fwork = vector<PetscScalar>(biggestBlock, 0.);
                localmats_aij = NULL;
                dofis = vector<IS>(numBlocks);
                auto ierr = PetscMalloc1(numBlocks, &localmats);CHKERRV(ierr);
                for(int p=0; p<numBlocks; p++) {
                    auto ndof = dofsPerBlock[p].size();
                    localmats[p] = NULL;
                    ISCreateGeneral(MPI_COMM_SELF, globalDofsPerBlock[p].size(), globalDofsPerBlock[p].data(), PETSC_USE_POINTER, dofis.data() + p);
                }
            }

        ~BlockJacobi() {
            int numBlocks = dofsPerBlock.size();
            for(int p=0; p<numBlocks; p++) {
                ISDestroy(&dofis[p]);
            }
            if(localmats_aij) {
                PetscErrorCode ierr;
                ierr = MatDestroySubMatrices(numBlocks, &localmats_aij);CHKERRV(ierr);
            }
            if (localmats) {
                PetscErrorCode ierr;
                for (int p=0; p<numBlocks; p++) {
                    ierr = MatDestroy(&localmats[p]);CHKERRV(ierr);
                }
                ierr = PetscFree(localmats);CHKERRV(ierr);
            }
            PetscSFDestroy(&sf);
        }

        PetscInt updateValuesPerBlock(Mat P) {
            PetscInt ierr, dof;
            int numBlocks = dofsPerBlock.size();
            ierr = MatCreateSubMatrices(P, numBlocks, dofis.data(), dofis.data(), localmats_aij ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX, &localmats_aij);CHKERRQ(ierr);
            PetscInt info;
            PetscScalar *vv;
            for(int p=0; p<numBlocks; p++) {
                ierr = MatConvert(localmats_aij[p], MATDENSE, localmats[p] ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX,&localmats[p]);CHKERRQ(ierr);
                PetscInt dof = dofsPerBlock[p].size();
                ierr = MatDenseGetArrayWrite(localmats[p],&vv);CHKERRQ(ierr);
                if (dof) mymatinvert(&dof, vv, piv.data(), &info, fwork.data());
                ierr = MatDenseRestoreArrayWrite(localmats[p],&vv);CHKERRQ(ierr);
            }
            return 0;
        }


        PetscInt solve(const PetscScalar* __restrict b, PetscScalar* __restrict x) {
            PetscInt dof, ierr;
            PetscScalar dOne = 1.0;
            PetscInt one = 1;
            PetscScalar dZero = 0.0;

            const PetscScalar *matvalues;
            for(int p=0; p<dofsPerBlock.size(); p++) {
                dof = dofsPerBlock[p].size();
                auto dofmap = dofsPerBlock[p];
                ierr = MatDenseGetArrayRead(localmats[p],&matvalues);CHKERRQ(ierr);
                for(int j=0; j<dof; j++)
                    workb[j] = b[dofmap[j]];
                if(dof < 6)
                    for(int i=0; i<dof; i++)
                        for(int j=0; j<dof; j++)
                            x[dofmap[i]] += matvalues[i*dof + j] * workb[j];
                else {
                    PetscCallBLAS("BLASgemv",BLASgemv_("N", &dof, &dof, &dOne, matvalues, &dof, workb.data(), &one, &dZero, worka.data(), &one));
                    for(int i=0; i<dof; i++)
                        x[dofmap[i]] += worka[i];
                }
                ierr = MatDenseRestoreArrayRead(localmats[p],&matvalues);CHKERRQ(ierr);
            }
            return 0;
        }
};

PetscErrorCode CreateCombinedSF(PC pc, const std::vector<PetscSF>& sf, const std::vector<PetscInt>& bs, PetscSF *newsf)
{
    PetscInt       i;
    PetscErrorCode ierr;
    auto n = sf.size();

    PetscFunctionBegin;
    if (n == 1 && bs[0] == 1) {
        *newsf= sf[0];
        ierr = PetscObjectReference((PetscObject) *newsf);CHKERRQ(ierr);
    } else {
        PetscInt     allRoots = 0, allLeaves = 0;
        PetscInt     leafOffset = 0;
        PetscInt    *ilocal = NULL;
        PetscSFNode *iremote = NULL;
        PetscInt    *remoteOffsets = NULL;
        PetscInt     index = 0;
        PetscHMapI   rankToIndex;
        PetscInt     numRanks = 0;
        PetscSFNode *remote = NULL;
        PetscSF      rankSF;
        PetscInt    *ranks = NULL;
        PetscInt    *offsets = NULL;
        MPI_Datatype contig;
        PetscHSetI   ranksUniq;

        /* First figure out how many dofs there are in the concatenated numbering.
         * allRoots: number of owned global dofs;
         * allLeaves: number of visible dofs (global + ghosted).
         */
        for (i = 0; i < n; ++i) {
            PetscInt nroots, nleaves;

            ierr = PetscSFGetGraph(sf[i], &nroots, &nleaves, NULL, NULL);CHKERRQ(ierr);
            allRoots  += nroots * bs[i];
            allLeaves += nleaves * bs[i];
        }
        ierr = PetscMalloc1(allLeaves, &ilocal);CHKERRQ(ierr);
        ierr = PetscMalloc1(allLeaves, &iremote);CHKERRQ(ierr);
        // Now build an SF that just contains process connectivity.
        ierr = PetscHSetICreate(&ranksUniq);CHKERRQ(ierr);
        for (i = 0; i < n; ++i) {
            const PetscMPIInt *ranks = NULL;
            PetscInt           nranks, j;

            ierr = PetscSFSetUp(sf[i]);CHKERRQ(ierr);
            ierr = PetscSFGetRootRanks(sf[i], &nranks, &ranks, NULL, NULL, NULL);CHKERRQ(ierr);
            // These are all the ranks who communicate with me.
            for (j = 0; j < nranks; ++j) {
                ierr = PetscHSetIAdd(ranksUniq, (PetscInt) ranks[j]);CHKERRQ(ierr);
            }
        }
        ierr = PetscHSetIGetSize(ranksUniq, &numRanks);CHKERRQ(ierr);
        ierr = PetscMalloc1(numRanks, &remote);CHKERRQ(ierr);
        ierr = PetscMalloc1(numRanks, &ranks);CHKERRQ(ierr);
        ierr = PetscHSetIGetElems(ranksUniq, &index, ranks);CHKERRQ(ierr);

        ierr = PetscHMapICreate(&rankToIndex);CHKERRQ(ierr);
        for (i = 0; i < numRanks; ++i) {
            remote[i].rank  = ranks[i];
            remote[i].index = 0;
            ierr = PetscHMapISet(rankToIndex, ranks[i], i);CHKERRQ(ierr);
        }
        ierr = PetscFree(ranks);CHKERRQ(ierr);
        ierr = PetscHSetIDestroy(&ranksUniq);CHKERRQ(ierr);
        ierr = PetscSFCreate(PetscObjectComm((PetscObject) pc), &rankSF);CHKERRQ(ierr);
        ierr = PetscSFSetGraph(rankSF, 1, numRanks, NULL, PETSC_OWN_POINTER, remote, PETSC_OWN_POINTER);CHKERRQ(ierr);
        ierr = PetscSFSetUp(rankSF);CHKERRQ(ierr);
        /* OK, use it to communicate the root offset on the remote
         * processes for each subspace. */
        ierr = PetscMalloc1(n, &offsets);CHKERRQ(ierr);
        ierr = PetscMalloc1(n*numRanks, &remoteOffsets);CHKERRQ(ierr);

        offsets[0] = 0;
        for (i = 1; i < n; ++i) {
            PetscInt nroots;

            ierr = PetscSFGetGraph(sf[i-1], &nroots, NULL, NULL, NULL);CHKERRQ(ierr);
            offsets[i] = offsets[i-1] + nroots*bs[i-1];
        }
        /* Offsets are the offsets on the current process of the
         * global dof numbering for the subspaces. */
        ierr = MPI_Type_contiguous(n, MPIU_INT, &contig);CHKERRQ(ierr);
        ierr = MPI_Type_commit(&contig);CHKERRQ(ierr);

#if MY_PETSC_VERSION_LT(3, 14, 4)
        ierr = PetscSFBcastBegin(rankSF, contig, offsets, remoteOffsets);CHKERRQ(ierr);
        ierr = PetscSFBcastEnd(rankSF, contig, offsets, remoteOffsets);CHKERRQ(ierr);
#else
        ierr = PetscSFBcastBegin(rankSF, contig, offsets, remoteOffsets, MPI_REPLACE);CHKERRQ(ierr);
        ierr = PetscSFBcastEnd(rankSF, contig, offsets, remoteOffsets, MPI_REPLACE);CHKERRQ(ierr);
#endif
        ierr = MPI_Type_free(&contig);CHKERRQ(ierr);
        ierr = PetscFree(offsets);CHKERRQ(ierr);
        ierr = PetscSFDestroy(&rankSF);CHKERRQ(ierr);
        /* Now remoteOffsets contains the offsets on the remote
         * processes who communicate with me.  So now we can
         * concatenate the list of SFs into a single one. */
        index = 0;
        for (i = 0; i < n; ++i) {
            const PetscSFNode *remote = NULL;
            const PetscInt    *local  = NULL;
            PetscInt           nroots, nleaves, j;

            ierr = PetscSFGetGraph(sf[i], &nroots, &nleaves, &local, &remote);CHKERRQ(ierr);
            for (j = 0; j < nleaves; ++j) {
                PetscInt rank = remote[j].rank;
                PetscInt idx, rootOffset, k;

                ierr = PetscHMapIGet(rankToIndex, rank, &idx);CHKERRQ(ierr);
                if (idx == -1) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Didn't find rank, huh?");
                //[> Offset on given rank for ith subspace <]
                rootOffset = remoteOffsets[n*idx + i];
                for (k = 0; k < bs[i]; ++k) {
                    ilocal[index]        = (local ? local[j] : j)*bs[i] + k + leafOffset;
                    iremote[index].rank  = remote[j].rank;
                    iremote[index].index = remote[j].index*bs[i] + k + rootOffset;
                    ++index;
                }
            }
            leafOffset += nleaves * bs[i];
        }
        ierr = PetscHMapIDestroy(&rankToIndex);CHKERRQ(ierr);
        ierr = PetscFree(remoteOffsets);CHKERRQ(ierr);
        ierr = PetscSFCreate(PetscObjectComm((PetscObject)pc), newsf);CHKERRQ(ierr);
        ierr = PetscSFSetGraph(*newsf, allRoots, allLeaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER);CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
}


PetscErrorCode PCSetup_TinyASM(PC pc) {
    PetscInt ierr;
    ierr = PetscLogEventBegin(PC_tinyasm_setup, pc, 0, 0, 0);CHKERRQ(ierr);
    auto P = pc -> pmat;
    auto blockjacobi = (BlockJacobi *)pc->data;
    blockjacobi -> updateValuesPerBlock(P);
    ierr = PetscLogEventEnd(PC_tinyasm_setup, pc, 0, 0, 0);CHKERRQ(ierr);
    return 0;
}

PetscErrorCode PCApply_TinyASM(PC pc, Vec b, Vec x) {
    PetscInt ierr;
    ierr = PetscLogEventBegin(PC_tinyasm_apply, pc, 0, 0, 0);CHKERRQ(ierr);
    ierr = VecSet(x, 0.0);CHKERRQ(ierr);
    auto blockjacobi = (BlockJacobi *)pc->data;

    const PetscScalar *globalb;
    PetscScalar *globalx;

    ierr = VecGetArrayRead(b, &globalb);CHKERRQ(ierr);
#if MY_PETSC_VERSION_LT(3, 14, 4)
    ierr = PetscSFBcastBegin(blockjacobi->sf, MPIU_SCALAR, globalb, &(blockjacobi->localb[0]));CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(blockjacobi->sf, MPIU_SCALAR, globalb, &(blockjacobi->localb[0]));CHKERRQ(ierr);
#else
    ierr = PetscSFBcastBegin(blockjacobi->sf, MPIU_SCALAR, globalb, &(blockjacobi->localb[0]), MPI_REPLACE);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(blockjacobi->sf, MPIU_SCALAR, globalb, &(blockjacobi->localb[0]), MPI_REPLACE);CHKERRQ(ierr);
#endif
    ierr = VecRestoreArrayRead(b, &globalb);CHKERRQ(ierr);

    std::fill(blockjacobi->localx.begin(), blockjacobi->localx.end(), 0);

    blockjacobi->solve(blockjacobi->localb.data(), blockjacobi->localx.data());
    ierr = VecGetArray(x, &globalx);CHKERRQ(ierr);
    ierr = PetscSFReduceBegin(blockjacobi->sf, MPIU_SCALAR, &(blockjacobi->localx[0]), globalx, MPI_SUM);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(blockjacobi->sf, MPIU_SCALAR, &(blockjacobi->localx[0]), globalx, MPI_SUM);CHKERRQ(ierr);
    ierr = VecRestoreArray(x, &globalx);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(PC_tinyasm_apply, pc, 0, 0, 0);CHKERRQ(ierr);
    return 0;
}

PetscErrorCode PCDestroy_TinyASM(PC pc) {
    if(pc->data)
        delete (BlockJacobi *)pc->data;
    return 0;
}

PetscErrorCode PCView_TinyASM(PC pc, PetscViewer viewer) {
    PetscBool isascii;
    PetscInt ierr;
    ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
    if(pc->data) {
        auto blockjacobi = (BlockJacobi *)pc->data;
        PetscInt nblocks = blockjacobi->dofsPerBlock.size();
        std::vector<PetscInt> blocksizes(nblocks);
        std::transform(blockjacobi->dofsPerBlock.begin(), blockjacobi->dofsPerBlock.end(), blocksizes.begin(), [](std::vector<PetscInt> &v){ return v.size(); });
        PetscInt biggestblock = *std::max_element(blocksizes.begin(), blocksizes.end());
        PetscScalar avgblock = std::accumulate(blocksizes.begin(), blocksizes.end(), 0.)/nblocks;
        ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "TinyASM (block Jacobi) preconditioner with %d blocks\n", nblocks);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "Average block size %f \n", avgblock);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer, "Largest block size %d \n", biggestblock);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    }
    return 0;
}

PetscErrorCode PCCreate_TinyASM(PC pc) {
    pc->data = NULL;
    pc->ops->apply = PCApply_TinyASM;
    pc->ops->setup = PCSetup_TinyASM;
    pc->ops->destroy = PCDestroy_TinyASM;
    pc->ops->view = PCView_TinyASM;
    return 0;
}
// pybind11 casters for PETSc/petsc4py objects, copied from dolfinx repo
// Import petsc4py on demand
#define VERIFY_PETSC4PY(func)                                                                                          \
    if (!func) {                                                                                                       \
        if (import_petsc4py() != 0)                                                                                    \
            throw std::runtime_error("Error when importing petsc4py");                                                 \
    }

// Macro for casting between PETSc and petsc4py objects
#define PETSC_CASTER_MACRO(TYPE, P4PYTYPE, NAME)                                                                       \
    template <> class type_caster<_p_##TYPE> {                                                                         \
      public:                                                                                                          \
        PYBIND11_TYPE_CASTER(TYPE, _(#NAME));                                                                          \
        bool load(handle src, bool) {                                                                                  \
            if (src.is_none()) {                                                                                       \
                value = nullptr;                                                                                       \
                return true;                                                                                           \
            }                                                                                                          \
            VERIFY_PETSC4PY(PyPetsc##P4PYTYPE##_Get);                                                                  \
            if (PyObject_TypeCheck(src.ptr(), &PyPetsc##P4PYTYPE##_Type) == 0)                                         \
                return false;                                                                                          \
            value = PyPetsc##P4PYTYPE##_Get(src.ptr());                                                                \
            return true;                                                                                               \
        }                                                                                                              \
                                                                                                                       \
        static handle cast(TYPE src, pybind11::return_value_policy policy, handle parent) {                            \
            VERIFY_PETSC4PY(PyPetsc##P4PYTYPE##_New);                                                                  \
            auto obj = PyPetsc##P4PYTYPE##_New(src);                                                                   \
            if (policy == pybind11::return_value_policy::take_ownership)                                               \
                PetscObjectDereference((PetscObject)src);                                                              \
            return pybind11::handle(obj);                                                                              \
        }                                                                                                              \
                                                                                                                       \
        operator TYPE() { return value; }                                                                              \
    }

namespace pybind11
{
    namespace detail
    {
        PETSC_CASTER_MACRO(PC, PC, pc);
        PETSC_CASTER_MACRO(IS, IS, is);
        PETSC_CASTER_MACRO(PetscSF, SF, petscsf);
    }
}


PYBIND11_MODULE(_tinyasm, m) {
    PCRegister("tinyasm", PCCreate_TinyASM);
    PetscLogEventRegister("PCTinyASMSetASMLocalSubdomains", PC_CLASSID, &PC_tinyasm_SetASMLocalSubdomains);
    PetscLogEventRegister("PCTinyASMSetup", PC_CLASSID, &PC_tinyasm_setup);
    PetscLogEventRegister("PCTinyASMApply", PC_CLASSID, &PC_tinyasm_apply);
    m.def("SetASMLocalSubdomains",
          [](PC pc, std::vector<IS> ises, std::vector<PetscSF> sfs, std::vector<PetscInt> blocksizes, int localsize) {
              PetscInt ierr, i, p, size, numDofs, blocksize;
              ierr = PetscLogEventBegin(PC_tinyasm_SetASMLocalSubdomains, pc, 0, 0, 0);CHKERRQ(ierr);
              auto P = pc->pmat;
              ISLocalToGlobalMapping lgr;
              ISLocalToGlobalMapping lgc;
              MatGetLocalToGlobalMapping(P, &lgr, &lgc);

              int numBlocks = ises.size();
              vector<vector<PetscInt>> dofsPerBlock(numBlocks);
              vector<vector<PetscInt>> globalDofsPerBlock(numBlocks);
              const PetscInt* isarray;

              for (p = 0; p < numBlocks; p++) {
                  ierr = ISGetSize(ises[p], &numDofs);CHKERRQ(ierr);
                  ierr = ISGetIndices(ises[p], &isarray);CHKERRQ(ierr);

                  dofsPerBlock[p] = vector<PetscInt>();
                  dofsPerBlock[p].reserve(numDofs);
                  globalDofsPerBlock[p] = vector<PetscInt>(numDofs, 0);

                  for (i = 0; i < numDofs; i++) {
                      dofsPerBlock[p].push_back(isarray[i]);
                  }
                  ierr = ISRestoreIndices(ises[p], &isarray);CHKERRQ(ierr);
                  ISLocalToGlobalMappingApply(lgr, numDofs, &dofsPerBlock[p][0], &globalDofsPerBlock[p][0]);
              }
              DM dm, plex;
              ierr = PCGetDM(pc, &dm);CHKERRQ(ierr);

              PetscSF newsf;
              ierr = CreateCombinedSF(pc, sfs, blocksizes, &newsf);CHKERRQ(ierr);
              auto blockjacobi = new BlockJacobi(dofsPerBlock, globalDofsPerBlock, localsize, newsf);
              pc->data = (void*)blockjacobi;
              ierr = PetscLogEventEnd(PC_tinyasm_SetASMLocalSubdomains, pc, 0, 0, 0);CHKERRQ(ierr);
              return ierr;
          });
}
