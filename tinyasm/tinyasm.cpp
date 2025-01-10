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

PetscErrorCode mymatinvert(PetscBLASInt* n, PetscScalar* mat, PetscBLASInt* piv, PetscBLASInt* info, PetscScalar* work);

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
        vector<PetscBLASInt> piv;
        vector<PetscScalar> fwork;

        BlockJacobi(vector<vector<PetscInt>> _dofsPerBlock, vector<vector<PetscInt>> _globalDofsPerBlock, int localSize, PetscSF _sf)
            : dofsPerBlock(_dofsPerBlock), globalDofsPerBlock(_globalDofsPerBlock), sf(_sf) {

                int numBlocks = dofsPerBlock.size();
                PetscInt dof;
                PetscInt biggestBlock = 0;
                for(int p=0; p<numBlocks; p++) {
                    dof = dofsPerBlock[p].size();
                    biggestBlock = max(biggestBlock, dof);
                }
                worka = vector<PetscScalar>(biggestBlock, 0);
                workb = vector<PetscScalar>(biggestBlock, 0);
                localb = vector<PetscScalar>(localSize, 0);
                localx = vector<PetscScalar>(localSize, 0);
                piv = vector<PetscBLASInt>(biggestBlock, 0.);
                iota(piv.begin(), piv.end(), 1);
                fwork = vector<PetscScalar>(biggestBlock, 0.);
                localmats_aij = NULL;
                dofis = vector<IS>(numBlocks);
                PetscCallVoid(PetscMalloc1(numBlocks, &localmats));
                for(int p=0; p<numBlocks; p++) {
                    localmats[p] = NULL;
                    PetscCallVoid(ISCreateGeneral(MPI_COMM_SELF, globalDofsPerBlock[p].size(), globalDofsPerBlock[p].data(), PETSC_USE_POINTER, dofis.data() + p));
                }
            }

        ~BlockJacobi() {
            int numBlocks = dofsPerBlock.size();
            for(int p=0; p<numBlocks; p++) {
                PetscCallVoid(ISDestroy(&dofis[p]));
            }
            if(localmats_aij) {
                PetscCallVoid(MatDestroySubMatrices(numBlocks, &localmats_aij));
            }
            if (localmats) {
                for (int p=0; p<numBlocks; p++) {
                    PetscCallVoid(MatDestroy(&localmats[p]));
                }
                PetscCallVoid(PetscFree(localmats));
            }
            PetscCallVoid(PetscSFDestroy(&sf));
        }

        PetscErrorCode updateValuesPerBlock(Mat P) {
            PetscBLASInt dof, info;
            int numBlocks = dofsPerBlock.size();
            PetscCall(MatCreateSubMatrices(P, numBlocks, dofis.data(), dofis.data(), localmats_aij ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX, &localmats_aij));
            PetscScalar *vv;
            for(int p=0; p<numBlocks; p++) {
                PetscCall(MatConvert(localmats_aij[p], MATDENSE, localmats[p] ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX,&localmats[p]));
                PetscCall(PetscBLASIntCast(dofsPerBlock[p].size(), &dof));
                PetscCall(MatDenseGetArrayWrite(localmats[p],&vv));
                if (dof) PetscCall(mymatinvert(&dof, vv, piv.data(), &info, fwork.data()));
                PetscCall(MatDenseRestoreArrayWrite(localmats[p],&vv));
            }
            PetscFunctionReturn(PETSC_SUCCESS);
        }


        PetscErrorCode solve(const PetscScalar* __restrict b, PetscScalar* __restrict x) {
            PetscScalar dOne = 1.0;
            PetscBLASInt dof, one = 1;
            PetscScalar dZero = 0.0;

            const PetscScalar *matvalues;
            for(size_t p=0; p<dofsPerBlock.size(); p++) {
                dof = dofsPerBlock[p].size();
                auto dofmap = dofsPerBlock[p];
                PetscCall(MatDenseGetArrayRead(localmats[p],&matvalues));
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
                PetscCall(MatDenseRestoreArrayRead(localmats[p],&matvalues));
            }
            PetscFunctionReturn(PETSC_SUCCESS);
        }
};

PetscErrorCode CreateCombinedSF(PC pc, const std::vector<PetscSF>& sf, const std::vector<PetscInt>& bs, PetscSF *newsf)
{
    auto n = sf.size();

    PetscFunctionBegin;
    if (n == 1 && bs[0] == 1) {
        *newsf= sf[0];
        PetscCall(PetscObjectReference((PetscObject) *newsf));
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
        for (size_t i = 0; i < n; ++i) {
            PetscInt nroots, nleaves;

            PetscCall(PetscSFGetGraph(sf[i], &nroots, &nleaves, NULL, NULL));
            allRoots  += nroots * bs[i];
            allLeaves += nleaves * bs[i];
        }
        PetscCall(PetscMalloc1(allLeaves, &ilocal));
        PetscCall(PetscMalloc1(allLeaves, &iremote));
        // Now build an SF that just contains process connectivity.
        PetscCall(PetscHSetICreate(&ranksUniq));
        for (size_t i = 0; i < n; ++i) {
            const PetscMPIInt *ranks = NULL;
            PetscMPIInt        nranks, j;

            PetscCall(PetscSFSetUp(sf[i]));
            PetscCall(PetscSFGetRootRanks(sf[i], &nranks, &ranks, NULL, NULL, NULL));
            // These are all the ranks who communicate with me.
            for (j = 0; j < nranks; ++j) {
                PetscCall(PetscHSetIAdd(ranksUniq, (PetscInt) ranks[j]));
            }
        }
        PetscCall(PetscHSetIGetSize(ranksUniq, &numRanks));
        PetscCall(PetscMalloc1(numRanks, &remote));
        PetscCall(PetscMalloc1(numRanks, &ranks));
        PetscCall(PetscHSetIGetElems(ranksUniq, &index, ranks));

        PetscCall(PetscHMapICreate(&rankToIndex));
        for (PetscInt i = 0; i < numRanks; ++i) {
            remote[i].rank  = ranks[i];
            remote[i].index = 0;
            PetscCall(PetscHMapISet(rankToIndex, ranks[i], i));
        }
        PetscCall(PetscFree(ranks));
        PetscCall(PetscHSetIDestroy(&ranksUniq));
        PetscCall(PetscSFCreate(PetscObjectComm((PetscObject) pc), &rankSF));
        PetscCall(PetscSFSetGraph(rankSF, 1, numRanks, NULL, PETSC_OWN_POINTER, remote, PETSC_OWN_POINTER));
        PetscCall(PetscSFSetUp(rankSF));
        /* OK, use it to communicate the root offset on the remote
         * processes for each subspace. */
        PetscCall(PetscMalloc1(n, &offsets));
        PetscCall(PetscMalloc1(n*numRanks, &remoteOffsets));

        offsets[0] = 0;
        for (size_t i = 1; i < n; ++i) {
            PetscInt nroots;

            PetscCall(PetscSFGetGraph(sf[i-1], &nroots, NULL, NULL, NULL));
            offsets[i] = offsets[i-1] + nroots*bs[i-1];
        }
        /* Offsets are the offsets on the current process of the
         * global dof numbering for the subspaces. */
        PetscCallMPI(MPI_Type_contiguous(n, MPIU_INT, &contig));
        PetscCallMPI(MPI_Type_commit(&contig));

#if MY_PETSC_VERSION_LT(3, 14, 4)
        PetscCall(PetscSFBcastBegin(rankSF, contig, offsets, remoteOffsets));
        PetscCall(PetscSFBcastEnd(rankSF, contig, offsets, remoteOffsets));
#else
        PetscCall(PetscSFBcastBegin(rankSF, contig, offsets, remoteOffsets, MPI_REPLACE));
        PetscCall(PetscSFBcastEnd(rankSF, contig, offsets, remoteOffsets, MPI_REPLACE));
#endif
        PetscCallMPI(MPI_Type_free(&contig));
        PetscCall(PetscFree(offsets));
        PetscCall(PetscSFDestroy(&rankSF));
        /* Now remoteOffsets contains the offsets on the remote
         * processes who communicate with me.  So now we can
         * concatenate the list of SFs into a single one. */
        index = 0;
        for (size_t i = 0; i < n; ++i) {
            const PetscSFNode *remote = NULL;
            const PetscInt    *local  = NULL;
            PetscInt           nroots, nleaves, j;

            PetscCall(PetscSFGetGraph(sf[i], &nroots, &nleaves, &local, &remote));
            for (j = 0; j < nleaves; ++j) {
                PetscInt rank = remote[j].rank;
                PetscInt idx, rootOffset, k;

                PetscCall(PetscHMapIGet(rankToIndex, rank, &idx));
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
        PetscCall(PetscHMapIDestroy(&rankToIndex));
        PetscCall(PetscFree(remoteOffsets));
        PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)pc), newsf));
        PetscCall(PetscSFSetGraph(*newsf, allRoots, allLeaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}


PetscErrorCode PCSetup_TinyASM(PC pc) {
    PetscCall(PetscLogEventBegin(PC_tinyasm_setup, pc, 0, 0, 0));
    auto P = pc -> pmat;
    auto blockjacobi = (BlockJacobi *)pc->data;
    blockjacobi -> updateValuesPerBlock(P);
    PetscCall(PetscLogEventEnd(PC_tinyasm_setup, pc, 0, 0, 0));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCApply_TinyASM(PC pc, Vec b, Vec x) {
    PetscCall(PetscLogEventBegin(PC_tinyasm_apply, pc, 0, 0, 0));
    PetscCall(VecSet(x, 0.0));
    auto blockjacobi = (BlockJacobi *)pc->data;

    const PetscScalar *globalb;
    PetscScalar *globalx;

    PetscCall(VecGetArrayRead(b, &globalb));
#if MY_PETSC_VERSION_LT(3, 14, 4)
    PetscCall(PetscSFBcastBegin(blockjacobi->sf, MPIU_SCALAR, globalb, &(blockjacobi->localb[0])));
    PetscCall(PetscSFBcastEnd(blockjacobi->sf, MPIU_SCALAR, globalb, &(blockjacobi->localb[0])));
#else
    PetscCall(PetscSFBcastBegin(blockjacobi->sf, MPIU_SCALAR, globalb, &(blockjacobi->localb[0]), MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(blockjacobi->sf, MPIU_SCALAR, globalb, &(blockjacobi->localb[0]), MPI_REPLACE));
#endif
    PetscCall(VecRestoreArrayRead(b, &globalb));

    std::fill(blockjacobi->localx.begin(), blockjacobi->localx.end(), 0);

    blockjacobi->solve(blockjacobi->localb.data(), blockjacobi->localx.data());
    PetscCall(VecGetArray(x, &globalx));
    PetscCall(PetscSFReduceBegin(blockjacobi->sf, MPIU_SCALAR, &(blockjacobi->localx[0]), globalx, MPI_SUM));
    PetscCall(PetscSFReduceEnd(blockjacobi->sf, MPIU_SCALAR, &(blockjacobi->localx[0]), globalx, MPI_SUM));
    PetscCall(VecRestoreArray(x, &globalx));
    PetscCall(PetscLogEventEnd(PC_tinyasm_apply, pc, 0, 0, 0));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCDestroy_TinyASM(PC pc) {
    if(pc->data)
        delete (BlockJacobi *)pc->data;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCView_TinyASM(PC pc, PetscViewer viewer) {
    PetscBool isascii;
    PetscCall(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii));
    if(pc->data) {
        auto blockjacobi = (BlockJacobi *)pc->data;
        PetscInt nblocks = blockjacobi->dofsPerBlock.size();
        std::vector<PetscInt> blocksizes(nblocks);
        std::transform(blockjacobi->dofsPerBlock.begin(), blockjacobi->dofsPerBlock.end(), blocksizes.begin(), [](std::vector<PetscInt> &v){ return v.size(); });
        PetscInt biggestblock = *std::max_element(blocksizes.begin(), blocksizes.end());
        PetscScalar avgblock = std::accumulate(blocksizes.begin(), blocksizes.end(), 0.)/nblocks;
        PetscCall(PetscViewerASCIIPushTab(viewer));
        PetscCall(PetscViewerASCIIPrintf(viewer, "TinyASM (block Jacobi) preconditioner with %" PetscInt_FMT " blocks\n", nblocks));
        PetscCall(PetscViewerASCIIPrintf(viewer, "Average block size %f \n", avgblock));
        PetscCall(PetscViewerASCIIPrintf(viewer, "Largest block size %" PetscInt_FMT " \n", biggestblock));
        PetscCall(PetscViewerASCIIPopTab(viewer));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCCreate_TinyASM(PC pc) {
    pc->data = NULL;
    pc->ops->apply = PCApply_TinyASM;
    pc->ops->setup = PCSetup_TinyASM;
    pc->ops->destroy = PCDestroy_TinyASM;
    pc->ops->view = PCView_TinyASM;
    PetscFunctionReturn(PETSC_SUCCESS);
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
              PetscInt p, numDofs;

              MPI_Comm comm = PetscObjectComm((PetscObject) pc);

              PetscCallAbort(comm, PetscLogEventBegin(PC_tinyasm_SetASMLocalSubdomains, pc, 0, 0, 0));
              auto P = pc->pmat;
              ISLocalToGlobalMapping lgr;
              ISLocalToGlobalMapping lgc;
              PetscCallAbort(comm, MatGetLocalToGlobalMapping(P, &lgr, &lgc));

              int numBlocks = ises.size();
              vector<vector<PetscInt>> dofsPerBlock(numBlocks);
              vector<vector<PetscInt>> globalDofsPerBlock(numBlocks);
              const PetscInt* isarray;

              for (p = 0; p < numBlocks; p++) {
                  PetscCallAbort(comm, ISGetSize(ises[p], &numDofs));
                  PetscCallAbort(comm, ISGetIndices(ises[p], &isarray));

                  dofsPerBlock[p] = vector<PetscInt>();
                  dofsPerBlock[p].reserve(numDofs);
                  globalDofsPerBlock[p] = vector<PetscInt>(numDofs, 0);

                  for (PetscInt i = 0; i < numDofs; i++) {
                      dofsPerBlock[p].push_back(isarray[i]);
                  }
                  PetscCallAbort(comm, ISRestoreIndices(ises[p], &isarray));
                  PetscCallAbort(comm, ISLocalToGlobalMappingApply(lgr, numDofs, &dofsPerBlock[p][0], &globalDofsPerBlock[p][0]));
              }
              DM dm;
              PetscCallAbort(comm, PCGetDM(pc, &dm));

              PetscSF newsf;
              PetscCallAbort(comm, CreateCombinedSF(pc, sfs, blocksizes, &newsf));
              auto blockjacobi = new BlockJacobi(dofsPerBlock, globalDofsPerBlock, localsize, newsf);
              pc->data = (void*)blockjacobi;
              PetscCallAbort(comm, PetscLogEventEnd(PC_tinyasm_SetASMLocalSubdomains, pc, 0, 0, 0));
              return 0;
          });
}
