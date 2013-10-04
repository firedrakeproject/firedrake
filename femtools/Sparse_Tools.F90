!    Copyright (C) 2006 Imperial College London and others.
!
!    Please see the AUTHORS file in the main source directory for a full list
!    of copyright holders.
!
!    Prof. C Pain
!    Applied Modelling and Computation Group
!    Department of Earth Science and Engineering
!    Imperial College London
!
!    amcgsoftware@imperial.ac.uk
!
!    This library is free software; you can redistribute it and/or
!    modify it under the terms of the GNU Lesser General Public
!    License as published by the Free Software Foundation,
!    version 2.1 of the License.
!
!    This library is distributed in the hope that it will be useful,
!    but WITHOUT ANY WARRANTY; without even the implied warranty of
!    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
!    Lesser General Public License for more details.
!
!    You should have received a copy of the GNU Lesser General Public
!    License along with this library; if not, write to the Free Software
!    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307
!    USA
#include "fdebug.h"
module sparse_tools
  !!< This module implements abstract data types for sparse matrices and
  !!< operations on them.
  use FLDebug
  use Futils
  use Reference_Counting
  use Global_Parameters, only: FIELD_NAME_LEN
  use Halo_data_types
  use halos_allocates
  use memory_diagnostics
  use ieee_arithmetic
  use data_structures

  implicit none

  private

  type csr_sparsity
     !!< Encapsulating type for the sparsity patter of a sparse matrix.

     !! Findrm is the indices of the row starts in colm.
     integer, dimension(:), pointer :: findrm=>null()
     !! Centrm is the indices of the main diagonal in colm.
     integer, dimension(:), pointer :: centrm=>null()
     !! Colm is the list of matrix j values.
     integer, dimension(:), pointer :: colm=>null()
     !! Number of columns in matrix.
     integer :: columns
     !! The halos associated with the rows and columns of the matrix.
     type(halo_type), pointer :: row_halo => null(), column_halo => null()
     !! Reference counting
     type(refcount_type), pointer :: refcount => null()
     !! Name
     character(len=FIELD_NAME_LEN) :: name=""
     !! Flag to indicate whether a matrix was allocated or wrapped.
     logical :: wrapped=.false.
     !! Flag to indicate whether each row in colm is sorted in ascending j
     !! order. If true this enables a faster binary search for entries
     !! during matrix accesses.
     logical :: sorted_rows=.false.
  end type csr_sparsity

  type csr_sparsity_pointer
    type(csr_sparsity), pointer :: ptr => null()
  end type csr_sparsity_pointer

  ! construct to avoid mem. leaks if people add an inactive array to a borrowed reference
  type logical_array_ptr
    logical, dimension(:), pointer :: ptr => null()
  end type

  type csr_matrix
     !!< Encapsulating type for a sparse matrix.

     !! The sparsity pattern for this matrix.
     type(csr_sparsity) :: sparsity
     !! The values of nonzero real entries
     real, dimension(:), pointer :: val=>null()
     !! The values of nonzero integer entries
     integer, dimension(:), pointer :: ival=>null()
     !! Flag to indicate whether a matrix was allocated or cloned.
     logical :: clone=.false.
     !! Flag to indicate value space has been externally supplied
     !! so it shouldn't be deallocated (only used for clone==.true.)
     logical :: external_val=.false.
     !! for nodes with inactive%ptr(node)==.true. the rows and columns
     !! will be left out of the matrix equation solved in petsc_solve()
     !! NOTE: %inactive should always be allocated for any allocated csr_matrix
     !! %inactive%ptr may not be allocated, in which case all nodes are "active"
     !! As %inactive is directly allocated from the start the %inactive%ptr pointer and its
     !! association status are always the same for all references of the matrix.
     type(logical_array_ptr), pointer:: inactive
     !! Reference counting
     type(refcount_type), pointer :: refcount => null()
     !! Name
     character(len=FIELD_NAME_LEN) :: name=""
  end type csr_matrix

  type csr_matrix_pointer
    type(csr_matrix), pointer :: ptr => null()
  end type csr_matrix_pointer

  type block_csr_matrix
     !!< Encapsulating type for a matrix with block sparse structure. The
     !!< blocks are stored in row major order. For example:
     !!<  +----------+
     !!<  | B1 B2 B3 |
     !!<  | B4 B5 B6 |
     !!<  | B7 B8 B9 |
     !!<  +----------+

     !! The sparsity pattern for this matrix.
     type(csr_sparsity) :: sparsity
     !! The values of the nonzero real entries.
     type(real_vector), dimension(:,:), pointer :: val=>null()
     !! Pointer to continuous memory if exists
     real, dimension(:), pointer :: contiguous_val=>null()
     !! The values of the nonzero integer entries.
     type(integer_vector), dimension(:,:), pointer :: ival=>null()
     !! The number of rows and columns of blocks.
     integer, dimension(2) :: blocks=(/0,0/)
     !! Flag to indicate whether a matrix was allocated or cloned.
     logical :: clone=.false.
     !! Flag to indicate value space has been externally supplied
     !! so it shouldn't be deallocated (only used for clone==.true.)
     logical :: external_val=.false.
     !! Number of columns in each block.
     integer :: columns
     !! Reference counting
     type(refcount_type), pointer :: refcount => null()
     !! Name
     character(len=FIELD_NAME_LEN) :: name=""
     !! Whether only the diagonal blocks are allocated or not
     logical :: diagonal=.false.
     !! Whether all diagonal blocks point to the same bit of memory
     logical :: equal_diagonal_blocks=.false.
  end type block_csr_matrix

  type block_csr_matrix_pointer
    type(block_csr_matrix), pointer :: ptr => null()
  end type block_csr_matrix_pointer

  type dynamic_csr_matrix
     !!< Dynamically sized CSR matrix.
     !! colm is the list of j values. In this case there is 1 colm per row.
     type(integer_vector), dimension(:), pointer :: colm=>null()
     !! The values of nonzero real entries
     type(real_vector), dimension(:), pointer :: val=>null()
     !! Number of columns in the matrix.
     integer :: columns
     !! Reference counting
     type(refcount_type), pointer :: refcount => null()
     !! Name
     character(len=FIELD_NAME_LEN) :: name=""
  end type dynamic_csr_matrix

  type block_dynamic_csr_matrix
     !!< A block matrix whose blocks are dynamically sized. Clearly the
     !!< blocks generally have differing sparsities.
     type(dynamic_csr_matrix), dimension(:,:), pointer :: blocks=>null()
     !! Reference counting
     type(refcount_type), pointer :: refcount => null()
     !! Name
     character(len=FIELD_NAME_LEN) :: name=""
  end type block_dynamic_csr_matrix

  public :: real_vector, integer_vector, csr_matrix, block_csr_matrix,&
       & dynamic_csr_matrix, block_dynamic_csr_matrix, dcsr2csr, csr2dcsr, &
       & mult,mult_T, zero_column, addref, incref, decref, has_references, &
       & csr_matrix_pointer, block_csr_matrix_pointer, &
       & csr_sparsity, csr_sparsity_pointer, logical_array_ptr,&
       & initialise_inactive, has_inactive, mult_addto, mult_t_addto

  TYPE node
     !!< A node in a linked list
     INTEGER :: ID                !! id number of node
     TYPE (node), POINTER :: next !! next node
  END TYPE node

  TYPE row
     !!< A matrix row comprising a linked list.
     TYPE (node), POINTER :: row
  END TYPE row

  interface allocate
     module procedure allocate_csr_matrix, allocate_block_csr_matrix,&
          & allocate_dcsr_matrix, allocate_block_dcsr_matrix,&
          & allocate_csr_sparsity
  end interface

  interface deallocate
     module procedure deallocate_csr_matrix, deallocate_block_csr_matrix,&
          & deallocate_dcsr_matrix, deallocate_block_dcsr_matrix,&
          & deallocate_csr_sparsity
  end interface

  interface attach_block
     module procedure block_csr_attach_block
  end interface

  interface unclone
     module procedure unclone_csr_matrix
  end interface

  interface size
     module procedure csr_size, block_csr_size, dcsr_size, block_dcsr_size,&
          & sparsity_size
  end interface

  interface block
     module procedure csr_block, dcsr_block
  end interface

  interface block_size
     module procedure block_csr_block_size, block_dcsr_block_size
  end interface

  interface blocks
     module procedure blocks_withdim, blocks_nodim, &
          &           dcsr_blocks_withdim, dcsr_blocks_nodim
  end interface

  interface entries
     module procedure csr_entries, dcsr_entries, sparsity_entries
  end interface

  interface pos
     module procedure csr_pos, block_csr_pos, dcsr_pos, dcsr_pos_noadd, &
       csr_sparsity_pos
  end interface

  private ::  pos

  interface row_m
     module procedure csr_row_m, block_csr_row_m, dcsr_row_m,&
          & sparsity_row_m
  end interface

  interface row_m_ptr
     module procedure csr_row_m_ptr, block_csr_row_m_ptr, &
       dcsr_row_m_ptr, sparsity_row_m_ptr
  end interface

  interface row_val
    module procedure csr_row_val, block_csr_row_val
  end interface

  interface row_val_ptr
     module procedure csr_row_val_ptr, block_csr_row_val_ptr, &
       dcsr_row_val_ptr
  end interface

  interface row_ival_ptr
     module procedure csr_row_ival_ptr, block_csr_row_ival_ptr
  end interface

  interface diag_val_ptr
     module procedure csr_diag_val_ptr
  end interface

  interface row_length
     module procedure csr_row_length, block_csr_block_row_length, &
       dcsr_row_length, csr_sparsity_row_length
  end interface

  interface zero
     module procedure csr_zero, block_csr_zero, dcsr_zero
  end interface

  interface zero_row
     module procedure csr_zero_row, block_csr_zero_row, block_csr_zero_single_row
  end interface

  interface zero_column
     module procedure dcsr_zero_column
  end interface

  interface addto
     module procedure csr_addto, csr_iaddto, csr_vaddto, &
          block_csr_addto, block_csr_vaddto, block_csr_blocks_addto, &
          block_csr_baddto, block_csr_bvaddto, &
          dcsr_addto, dcsr_vaddto, dcsr_vaddto1, dcsr_vaddto2,&
          dcsr_dcsraddto, csr_csraddto
  end interface

  interface addto_diag
     module procedure csr_addto_diag, csr_vaddto_diag, &
          block_csr_addto_diag, block_csr_vaddto_diag
  end interface

  interface set
     module procedure csr_set, csr_vset, csr_iset, block_csr_set, &
          dcsr_set, dcsr_vset, dcsr_set_row, dcsr_set_col, csr_csr_set, &
          block_csr_vset, block_csr_bset, csr_rset, csr_block_csr_set
  end interface

  interface set_diag
     module procedure csr_set_diag
  end interface

  interface scale
     module procedure csr_scale, block_csr_scale
  end interface

  interface val
     module procedure csr_val, block_csr_val, dcsr_val
  end interface

  interface ival
     module procedure csr_ival
  end interface

  interface dense
     module procedure csr_dense, block_csr_dense, dcsr_dense,&
          & block_dcsr_dense
  end interface

  interface dense_i
     module procedure csr_dense_i
  end interface

  interface wrap
     module procedure wrap_csr_matrix, block_wrap_csr_matrix,&
          & wrap_csr_sparsity
  end interface

  interface mult
     module procedure csr_mult
  end interface

  interface mult_addto
     module procedure csr_mult_addto
  end interface

  interface mult_T
     module procedure csr_mult_T
  end interface

  interface mult_T_addto
     module procedure csr_mult_T_addto
  end interface

  interface matmul
     module procedure csr_matmul, &
       block_csr_matmul, csr_sparsity_matmul
  end interface

  interface matmul_addto
     module procedure csr_matmul_addto, block_csr_matmul_addto
  end interface

  interface matmul_T
     module procedure dcsr_matmul_T, csr_matmul_T
  end interface

  interface set_inactive
     module procedure csr_set_inactive_rows, csr_set_inactive_row
  end interface set_inactive

  interface get_inactive_mask
     module procedure csr_get_inactive_mask
  end interface get_inactive_mask

  interface matrix2file
     module procedure csr_matrix2file, dcsr_matrix2file,&
          & block_csr_matrix2file, block_dcsr_matrix2file, &
          & dense_matrix2file
  end interface

  interface mmwrite
     module procedure csr_mmwrite, dcsr_mmwrite
  end interface

  interface transpose
     module procedure csr_sparsity_transpose, csr_transpose, block_csr_transpose
  end interface

  interface mmread
     module procedure dcsr_mmread
  end interface

  interface initialise_inactive
     module procedure csr_initialise_inactive
  end interface

  interface reset_inactive
     module procedure csr_reset_inactive
  end interface

  interface has_solver_cache
     module procedure csr_has_solver_cache, block_csr_has_solver_cache
  end interface

  interface destroy_solver_cache
     module procedure csr_destroy_solver_cache, block_csr_destroy_solver_cache
  end interface

  interface is_symmetric
     module procedure sparsity_is_symmetric
  end interface

  interface is_sorted
    module procedure sparsity_is_sorted
  end interface

  interface write_minmax
    module procedure csr_write_minmax, block_csr_write_minmax
  end interface

#include "Reference_count_interface_csr_sparsity.F90"
#include "Reference_count_interface_csr_matrix.F90"
#include "Reference_count_interface_block_csr_matrix.F90"
#include "Reference_count_interface_dynamic_csr_matrix.F90"
#include "Reference_count_interface_block_dynamic_csr_matrix.F90"

  !! Parameters enabling the selection of matrix entry type.
  integer, public, parameter :: CSR_REAL=0, CSR_INTEGER=1, CSR_NONE=2

  ! maximum line length in MatrixMarket files
  integer, private, parameter :: MMmaxlinelen=1024
  character(len=*), parameter :: MMlineformat='(1024a)'

  public :: allocate, deallocate, attach_block, &
       unclone, size, block, block_size, blocks, entries, row_m, row_val, &
       & row_m_ptr, row_val_ptr, row_ival_ptr, diag_val_ptr, row_length, zero, zero_row, addto,&
       & addto_diag, set_diag,  set, val, ival, dense, dense_i, wrap, matmul, matmul_addto, matmul_T,&
       & matrix2file, mmwrite, mmread, transpose, sparsity_sort,&
       & sparsity_merge, scale, set_inactive, get_inactive_mask, &
       & reset_inactive, has_solver_cache, destroy_solver_cache, is_symmetric, is_sorted, &
       & write_minmax

  public :: posinm

contains

  !! This subroutine works out the sparsity pattern of the matrix:
  !!     <--------- NNodes 1 ----------->
  !!     ^
  !!     |      non-zero in row i,colume j iff these exists an element
  !!     |      index, k, where element k in mesh1 contains j and element
  !!     |      k in mesh 2 contains i
  !!  NNodes 2
  !!     |
  !!     |
  !!     |
  !!     v
  !!
  SUBROUTINE posinm(sparsity, TOTELE, NNodes1, NLoc1, NDGLNO1,&
       NNodes2, NLoc2, NDGLNO2, diag, name)
    type(csr_sparsity), intent(out) :: sparsity
    INTEGER, INTENT(IN)::NNodes1, NNodes2, TOTELE, NLoc1, NLoc2
    INTEGER, INTENT(IN)::NDGLNO1(TOTELE*NLoc1), NDGLNO2(TOTELE*NLoc2)
    logical, intent(in), optional ::  diag
    character(len=*), intent(in):: name

    INTEGER ELE,GLOBI,GLOBJ,LOCI,LOCJ,I
    ! Count of nonzero entries.
    integer :: entries
    ! Count of diagonal entries.
    integer :: diag_cnt

    ! Whether val and diag should be allocated. If diag is false then the
    ! diagonal will be totally excluded from the matrix.
    logical :: ldiag

    TYPE(row), DIMENSION(:), ALLOCATABLE::lMatrix
    TYPE(node), POINTER::List, Current, Next

    if(present(diag)) then
       ldiag=diag
    else
       ldiag=.true.
    end if

    ewrite(2, *) "SUBROUTINE POSINM()"

    ! Initalise the linked lists
    ALLOCATE( lMatrix(NNodes2) )
    DO I=1, NNodes2
       ALLOCATE( List )
       List%ID = -1
       NULLIFY( List%next )

       lMatrix(I)%row => List
       NULLIFY(List)
    END DO

    ewrite(2, *) "Constructing lMatrix using linked-lists"

    ! The first entry on each row is already present.
    entries=NNodes2
    diag_cnt=0

    DO ELE=1,TOTELE
       DO LOCI=1,NLoc2
          GLOBI=NDGLNO2((ELE-1)*NLoc2+LOCI)
          List => lMatrix(GLOBI)%row

          DO LOCJ=1,NLoc1
             GLOBJ=NDGLNO1((ELE-1)*NLoc1+LOCJ)

             ! Check if the list is initalised
             IF(List%ID.EQ.-1) THEN
                List%ID = GLOBJ

                ! Count diagonal entries.
                if (GLOBI==GLOBJ) diag_cnt=diag_cnt+1

                CYCLE
             END IF

             IF(GLOBJ.LT.List%ID) THEN
                ! Insert at start of list
                ALLOCATE(Current)
                entries=entries+1
                Current%ID = GLOBJ
                Current%next => List

                lMatrix(GLOBI)%row => Current
                List => lMatrix(GLOBI)%row

                ! Count diagonal entries.
                if (GLOBI==GLOBJ) diag_cnt=diag_cnt+1
             ELSE
                Current => List
                DO WHILE ( ASSOCIATED(Current) )
                   IF(GLOBJ.EQ.Current%ID) THEN
                      ! Already have this node
                      exit
                   ELSE IF(.NOT.ASSOCIATED(Current%next)) THEN
                      ! End of list - insert this node
                      ALLOCATE(Current%next)
                      entries=entries+1
                      NULLIFY(Current%next%next)
                      Current%next%ID = GLOBJ

                      ! Count diagonal entries.
                      if (GLOBI==GLOBJ) diag_cnt=diag_cnt+1

                      exit
                   ELSE IF(GLOBJ.LT.Current%next%ID) THEN
                      ! Insert new node here
                      ALLOCATE(Next)
                      entries=entries+1
                      Next%ID = GLOBJ
                      Next%next => Current%next
                      Current%Next => Next

                      ! Count diagonal entries.
                      if (GLOBI==GLOBJ) diag_cnt=diag_cnt+1

                      exit
                   END IF
                   Current => Current%next
                END DO
             END IF
          END DO
       END DO
    END DO

    ewrite(2, *) "Compressing matrix"

    ! Exclude the diagonal if needed.
    if (.not.ldiag) then
       entries=entries-diag_cnt
    end if

    call allocate(sparsity, rows=NNodes2, columns=NNodes2, entries=entries,&
         & diag=diag, name=name)

    call compress_sparsity(nnodes2, sparsity, ldiag,lmatrix, entries)

    DEALLOCATE( lMatrix )

    ewrite(2, *) "END SUBROUTINE POSINM"
    RETURN
  END SUBROUTINE posinm

  subroutine row_insert(row_in, value, entries)
    ! Insert value into list.
    type(row), intent(inout) :: row_in
    integer, intent(in) :: value
    integer, intent(inout) :: entries

    type(node), pointer :: list, current, next

    list=>row_in%row

    ! Check if the list is initalised
    IF(List%ID.EQ.-1) THEN
       List%ID = value

       return
    END IF

    IF(value.LT.List%ID) THEN
       ! Insert at start of list
       ALLOCATE(Current)
       entries=entries+1
       Current%ID = value
       Current%next => List

       row_in%row => Current
       List => row_in%row

    ELSE
       Current => List
       DO WHILE ( ASSOCIATED(Current) )
          IF(value.EQ.Current%ID) THEN
             ! Already have this node
             exit
          ELSE IF(.NOT.ASSOCIATED(Current%next)) THEN
             ! End of list - insert this node
             ALLOCATE(Current%next)
             entries=entries+1
             NULLIFY(Current%next%next)
             Current%next%ID = value

             exit
          ELSE IF(value.LT.Current%next%ID) THEN
             ! Insert new node here
             ALLOCATE(Next)
             entries=entries+1
             Next%ID = value
             Next%next => Current%next
             Current%Next => Next

             exit
          END IF
          Current => Current%next
       END DO
    END IF

  end subroutine row_insert

  subroutine compress_sparsity(nnodes2, sparsity, ldiag,lmatrix, entries)

    integer, intent(in)::nnodes2,entries
    logical, intent(in)::ldiag
    type(csr_sparsity), intent(inout) :: sparsity

    TYPE(row), DIMENSION(:) ::lMatrix

    !local variables
    integer::ptr,irow
    TYPE(node), POINTER::Current, Next

    ewrite(2,*) "subroutine compress_sparsity"

    ! From sparsity write COLM, FINDRM and CENTRM
    ! linked list as we go
    PTR = 1

    DO IROW=1,NNodes2

       sparsity%FINDRM(IROW) = PTR

       if (ldiag) then
          sparsity%CENTRM(IROW) = -1
       end if

       Current => lMatrix(IROW)%row

       DO WHILE ( ASSOCIATED(Current) )

          ASSERT(PTR.LE.entries+1) ! Sanity check the calculation of entries.

          IF(Current%ID.EQ.IROW) THEN
             if (ldiag) then
                sparsity%CENTRM(IROW) = PTR
             else
                ! Exclude this element completely.
                goto 42
             end if
          END IF

          sparsity%COLM(PTR) = Current%ID
          IF(Current%ID==-1) THEN
             ewrite(-1,*) "ERROR: POSINM() seriously unhappy with node",IROW
             FLAbort("Mesh contains nodes that are not associated with any elements.")
          END IF

          PTR = PTR + 1

42        Next => Current%next
          DEALLOCATE(Current)
          Current => Next

       END DO
    END DO

    ASSERT(PTR==entries+1) ! Sanity check the calculation of entries.

    sparsity%FINDRM(NNodes2+1) = entries+1

    ewrite(2,*)"END subroutine compress_sparsity"

  end subroutine compress_sparsity

  pure function blockstart(matrix, blocki, blockj)
    !!< local auxillary function that determines start of a block in matrix%val
    !!< This is almost obsolete now - it is only used to position pointers.
    integer blockstart
    type(block_csr_matrix), intent(in):: matrix
    integer, intent(in):: blocki, blockj

    blockstart=((blocki-1)*matrix%blocks(2)+blockj-1)*size(matrix%sparsity%colm)+1

  end function blockstart

  subroutine allocate_csr_sparsity(sparsity, rows, columns, entries, nnz, diag,&
       & name, stat)
    type(csr_sparsity), intent(out) :: sparsity
    !! Rows is the number of rows.
    integer, intent(in) :: rows, columns
    !! Entries is the number of nonzero entries. Either 'entries' or 'nnz' is required
    integer, intent(in), optional :: entries
    !! nnz number of nonzero entries for each row
    integer, dimension(:), intent(in), optional :: nnz
    !! Diag can be used to not allocate the diagonal.
    logical, intent(in), optional :: diag
    character(len=*), intent(in) :: name
    integer, intent(out), optional :: stat

    logical :: ldiag
    integer :: lstat, totalmem, lentries
    integer :: i, k

    if(present(diag)) then
       ldiag=diag
    else
       ldiag=.true.
    end if

    if(present(entries)) then
       lentries=entries
    else if (present(nnz)) then
       lentries=sum(nnz)
    else
       FLAbort("In allocate_csr_sparsity need to provide either entries or nnz argument")
    end if

    sparsity%name = name

    sparsity%wrapped=.false.

    sparsity%columns=columns

    nullify(sparsity%refcount)
    call addref(sparsity)

    allocate(sparsity%findrm(rows+1), sparsity%colm(lentries), stat=lstat)
    if (lstat/=0) goto 42
    totalmem=rows+1 + lentries

    if (ldiag) then
       allocate(sparsity%centrm(min(rows, columns)), stat=lstat)
       if (lstat/=0) goto 42
       totalmem=totalmem + size(sparsity%centrm)
    else
       ! fix for 'old' gfortran bug:
       nullify(sparsity%centrm)
    end if

42  if (present(stat)) then
       stat=lstat
    else
       if (lstat/=0) then
          FLAbort("Failed to allocate sparsity.")
       end if
    end if

#ifdef HAVE_MEMORY_STATS
    call register_allocation("csr_sparsity", "integer", &
         totalmem, name=name)
#endif

    if (present(nnz)) then
      ! fill in %findrm from nnz:
      k=1
      do i=1, size(nnz)
        sparsity%findrm(i)=k
        k=k+nnz(i)
      end do
      sparsity%findrm(i)=k
      assert( k==lentries+1 )
    end if

  end subroutine allocate_csr_sparsity

  subroutine allocate_csr_matrix(matrix, sparsity, val, type, name, stat)
    type(csr_matrix), intent(out) :: matrix
    type(csr_sparsity), intent(in) :: sparsity
    !! Val can be used to not allocate the values.
    logical, intent(in), optional :: val
    !! Real or integer matrix.
    integer, intent(in), optional :: type
    character(len=*), intent(in), optional :: name
    integer, intent(out), optional :: stat

    integer :: lstat, ltype
    character(len=FIELD_NAME_LEN) :: lname

    if (present(name)) then
       lname=name
    else
       lname=""
    end if
    matrix%name = lname

    if (present(type)) then
       ltype=type
    else
       ltype=CSR_REAL
    end if

    matrix%clone=.false.

    nullify(matrix%refcount) ! Hack for gfortran component initialisation
    !                         bug.
    call addref(matrix)

    matrix%sparsity=sparsity
    call incref(matrix%sparsity)

    ! this is a temp. measure as long as gfortran does not do it automatically
    nullify(matrix%val)
    nullify(matrix%ival)
    ! should always be allocated, so that matrix%inactive%ptr is the same for
    ! all references of the matrix:
    allocate(matrix%inactive)
    nullify(matrix%inactive%ptr)

    select case (ltype)
    case (CSR_REAL)
       allocate(matrix%val(size(sparsity%colm)), stat=lstat)
#ifdef HAVE_MEMORY_STATS
       call register_allocation("csr_matrix", "real", &
              size(sparsity%colm), name=name)
#endif

    case (CSR_INTEGER)
       allocate(matrix%ival(size(sparsity%colm)), stat=lstat)
#ifdef HAVE_MEMORY_STATS
       call register_allocation("csr_matrix", "integer", &
              size(sparsity%colm), name=name)
#endif

    case default
       FLAbort("Unknown matrix data type.")
    end select

    if (present(stat)) then
       stat=lstat
    else
       if (lstat/=0) then
          FLAbort("Failed to allocate matrix.")
       end if
    end if

  end subroutine allocate_csr_matrix

  subroutine deallocate_csr_sparsity(sparsity, stat)
    type(csr_sparsity), intent(inout) :: sparsity
    integer, intent(out), optional :: stat

    integer :: lstat, totalmem

    lstat = 0

    call decref(sparsity)
    if (has_references(sparsity)) then
      goto 42
    end if

    if (.not.sparsity%wrapped) then
       totalmem=size(sparsity%findrm) + size(sparsity%colm)
       deallocate(sparsity%findrm, sparsity%colm, stat=lstat)
       if (lstat/=0) goto 42

       ! centrm may legitimately not be allocated.
       if (associated(sparsity%centrm)) then
          totalmem=totalmem+size(sparsity%centrm)
          deallocate(sparsity%centrm, stat=lstat)
          if (lstat/=0) goto 42
       end if

#ifdef HAVE_MEMORY_STATS
       call register_deallocation("csr_sparsity", "integer", &
            totalmem, name=sparsity%name)
#endif
    end if

    if (associated(sparsity%row_halo)) then
       call deallocate(sparsity%row_halo)
       deallocate(sparsity%row_halo)
    end if

    if (associated(sparsity%column_halo)) then
       call deallocate(sparsity%column_halo)
       deallocate(sparsity%column_halo)
    end if

42  if (present(stat)) then
       stat=lstat
    else
       if (lstat/=0) then
          FLAbort("Failed to deallocate matrix.")
       end if
    end if

  end subroutine deallocate_csr_sparsity

  subroutine deallocate_csr_matrix(matrix, stat)
    type(csr_matrix), intent(inout) :: matrix
    integer, intent(out), optional :: stat

    integer :: lstat

    lstat = 0

    call decref(matrix)
    if (has_references(matrix)) then
      goto 42
    end if

    call deallocate(matrix%sparsity)

    if (.not. (matrix%clone .and. matrix%external_val)) then
      if (associated(matrix%val)) then
#ifdef HAVE_MEMORY_STATS
         call register_deallocation("csr_matrix", "real", &
              size(matrix%val), name=matrix%name)
#endif
#ifdef DDEBUG
         matrix%val=ieee_value(0.0, ieee_quiet_nan)
#endif

         deallocate(matrix%val, stat=lstat)
         if (lstat/=0) goto 42
      end if
      if (associated(matrix%ival)) then
#ifdef HAVE_MEMORY_STATS
         call register_deallocation("csr_matrix", "integer", &
              size(matrix%ival), name=matrix%name)
#endif
         deallocate(matrix%ival, stat=lstat)
         if (lstat/=0) goto 42
      end if
    end if

    if(can_have_inactive(matrix)) then
      if(has_inactive(matrix)) then
        deallocate(matrix%inactive%ptr, stat=lstat)
        if (lstat/=0) goto 42
      end if
      deallocate(matrix%inactive, stat=lstat)
      if (lstat/=0) goto 42
    end if

42  if (present(stat)) then
       stat=lstat
    else
       if (lstat/=0) then
          FLAbort("Failed to deallocate matrix.")
       end if
    end if


  end subroutine deallocate_csr_matrix

  subroutine allocate_block_csr_matrix(matrix, sparsity, blocks, data, name, &
       diagonal, equal_diagonal_blocks, stat)
    type(block_csr_matrix), intent(out) :: matrix
    type(csr_sparsity), intent(in) :: sparsity
    integer, intent(in), dimension(2) :: blocks
    !! Whether the blocks should be filled with data.
    logical, intent(in), optional :: data
    character(len=*), optional :: name
    !! Whether to allocate just the diagonal blocks or not (default is not)
    logical, intent(in), optional :: diagonal
    !! Together with diagonal this means all diagonal blocks will be the same,
    !! i.e. they point at the same bit of memory
    logical, intent(in), optional :: equal_diagonal_blocks
    integer, intent(out), optional :: stat

    integer :: lstat, i, j
    character(len=FIELD_NAME_LEN) :: lname

    lstat = 0

    if (present(name)) then
      lname = name
    else
      lname = ""
    end if
    matrix%name = lname

    if(present_and_true(diagonal).and.(blocks(1)/=blocks(2))) then
      FLAbort("Attempt made to allocate a non-square diagonal block_csr_matrix!")
    end if
    matrix%diagonal = present_and_true(diagonal)
    matrix%equal_diagonal_blocks = present_and_true(equal_diagonal_blocks)

    nullify(matrix%refcount) ! Hack for gfortran component initialisation
    !                         bug.
    call addref(matrix)

    matrix%sparsity=sparsity

    call incref(matrix%sparsity)
    matrix%blocks=blocks

    allocate(matrix%val(blocks(1),blocks(2)), stat=lstat)
    if (lstat/=0) goto 42

    if (present_and_false(data)) then

      ! no data to be allocated at all:
      do i=1, blocks(1)
        do j=1, blocks(2)
          nullify(matrix%val(i,j)%ptr)
        end do
      end do
      matrix%external_val=.true.

    else if (matrix%diagonal) then

      ! only allocate diagonal blocks

      do i=1, blocks(1)
        do j=1, blocks(2)
          nullify(matrix%val(i,j)%ptr)
        end do
      end do

      if (matrix%equal_diagonal_blocks) then
        allocate(matrix%val(1,1)%ptr(size(sparsity%colm)), stat=lstat)
#ifdef HAVE_MEMORY_STATS
        call register_allocation("csr_matrix", "real", &
                  size(sparsity%colm), name=name)
#endif
        if (lstat/=0) goto 42
        do i=2, blocks(1)
          matrix%val(i,i)%ptr => matrix%val(1,1)%ptr
        end do
      else
        do i=1, blocks(1)
          allocate(matrix%val(i,i)%ptr(size(sparsity%colm)), stat=lstat)
#ifdef HAVE_MEMORY_STATS
          call register_allocation("csr_matrix", "real", &
                  size(sparsity%colm), name=name)
#endif
          if (lstat/=0) goto 42
        end do
      end if
      matrix%external_val=.false.

    else

      ! normal case: allocate all blocks
      do i=1,blocks(1)
         do j=1,blocks(2)
           allocate(matrix%val(i,j)%ptr(size(sparsity%colm)), stat=lstat)
#ifdef HAVE_MEMORY_STATS
           call register_allocation("csr_matrix", "real", &
                  size(sparsity%colm), name=name)
#endif
           if (lstat/=0) goto 42
         end do
      end do
      matrix%external_val=.false.

    end if

42  if (present(stat)) then
       stat=lstat
    else
       if (lstat/=0) then

          FLAbort("Failed to allocate matrix.")
       end if
    end if

  end subroutine allocate_block_csr_matrix

  subroutine deallocate_block_csr_matrix(matrix, stat)
    type(block_csr_matrix), intent(inout) :: matrix
    integer, intent(out), optional :: stat

    integer :: lstat, i, j

    lstat=0

    call decref(matrix)
    if (has_references(matrix)) then
      goto 42
    end if

    call deallocate(matrix%sparsity)

    if (associated(matrix%val)) then
      if (.not. (matrix%clone .and. matrix%external_val)) then
        if (matrix%equal_diagonal_blocks) then
#ifdef HAVE_MEMORY_STATS
          call register_deallocation("csr_matrix", "real", &
                    size(matrix%val(1,1)%ptr), name=matrix%name)
#endif
#ifdef DDEBUG
          matrix%val(1,1)%ptr=ieee_value(0.0, ieee_quiet_nan)
#endif
          deallocate(matrix%val(1,1)%ptr, stat=lstat)
          if (lstat/=0) goto 42
        elseif (matrix%diagonal) then
          do i=1, matrix%blocks(1)
#ifdef HAVE_MEMORY_STATS
            call register_deallocation("csr_matrix", "real", &
                    size(matrix%val(i,i)%ptr), name=matrix%name)
#endif
#ifdef DDEBUG
            matrix%val(i,i)%ptr=ieee_value(0.0, ieee_quiet_nan)
#endif
            deallocate(matrix%val(i,i)%ptr, stat=lstat)
          end do
        else
          do i=1,matrix%blocks(1)
            do j=1,matrix%blocks(2)
#ifdef HAVE_MEMORY_STATS
               call register_deallocation("csr_matrix", "real", &
                    size(matrix%val(i,j)%ptr), name=matrix%name)
#endif
#ifdef DDEBUG
               matrix%val(i,j)%ptr=ieee_value(0.0, ieee_quiet_nan)
#endif
               deallocate(matrix%val(i,j)%ptr, stat=lstat)
               if (lstat/=0) goto 42
            end do
          end do
        end if
      end if
      ! the val pointer-array is always allocated by us:
      deallocate(matrix%val, stat=lstat)
      if (lstat/=0) goto 42
    end if

    if (associated(matrix%ival)) then
      if (.not. (matrix%clone .and. matrix%external_val)) then
        if (matrix%equal_diagonal_blocks) then
#ifdef HAVE_MEMORY_STATS
          call register_deallocation("csr_matrix", "integer", &
                    size(matrix%ival(1,1)%ptr), name=matrix%name)
#endif
          deallocate(matrix%ival(1,1)%ptr, stat=lstat)
          if (lstat/=0) goto 42
        elseif (matrix%diagonal) then
          do i=1, matrix%blocks(1)
#ifdef HAVE_MEMORY_STATS
            call register_deallocation("csr_matrix", "integer", &
                    size(matrix%ival(i,i)%ptr), name=matrix%name)
#endif
            deallocate(matrix%ival(i,i)%ptr, stat=lstat)
          end do
        else
          do i=1,matrix%blocks(1)
            do j=1,matrix%blocks(2)
#ifdef HAVE_MEMORY_STATS
               call register_deallocation("csr_matrix", "integer", &
                    size(matrix%ival(i,j)%ptr), name=matrix%name)
#endif
               deallocate(matrix%ival(i,j)%ptr, stat=lstat)
               if (lstat/=0) goto 42
            end do
          end do
        end if
      end if
      ! the val pointer-array is always allocated by us:
      deallocate(matrix%ival, stat=lstat)
      if (lstat/=0) goto 42
    end if

42  if (present(stat)) then
       stat=lstat
    else
       if (lstat/=0) then
          FLAbort("Failed to deallocate matrix.")
       end if
    end if

  end subroutine deallocate_block_csr_matrix

  subroutine allocate_dcsr_matrix(matrix, rows, columns, name, stat)
    !!< Allocate the core of a dynamic csr matrix. Due to the dynamic
    !!< nature of these matrices, further allocation will occur as the
    !!< matrix is constructed.
    type(dynamic_csr_matrix), intent(inout) :: matrix
    integer, intent(in) :: rows
    integer, intent(in) :: columns
    character(len=*), optional :: name
    integer, intent(out), optional :: stat

    integer :: lstat, i
    character(len=FIELD_NAME_LEN) :: lname

    if (present(name)) then
      lname = name
    else
      lname = ""
    end if
    matrix%name = lname


    nullify(matrix%refcount) ! Hack for gfortran component initialisation
    !                         bug.
    call addref(matrix)

    matrix%columns=columns

    allocate(matrix%colm(rows), matrix%val(rows), stat=lstat)

    if (lstat/=0) goto 666

    do i=1, rows
       allocate(matrix%colm(i)%ptr(0), matrix%val(i)%ptr(0), stat=lstat)
       if (lstat/=0) goto 666
    end do

666 if (present(stat)) then
       stat=lstat
    else
       if (lstat/=0) then
          FLAbort("Failed to allocate matrix.")
       end if
    end if

  end subroutine allocate_dcsr_matrix

  subroutine deallocate_dcsr_matrix(matrix, stat)
    type(dynamic_csr_matrix), intent(inout) :: matrix
    integer, intent(out), optional :: stat

    integer :: lstat, i

    call decref(matrix)
    if (has_references(matrix)) then
      goto 666
    end if

    do i=1,size(matrix%colm)
#ifdef DDEBUG
       matrix%val(i)%ptr=ieee_value(0.0, ieee_quiet_nan)
#endif
       deallocate(matrix%colm(i)%ptr, matrix%val(i)%ptr, stat=lstat)
       if (lstat/=0) goto 666
    end do

    deallocate(matrix%colm, matrix%val, stat=lstat)

666 if (present(stat)) then
       stat=lstat
    else
       if (lstat/=0) then
          FLAbort("Failed to deallocate matrix.")
       end if
    end if

  end subroutine deallocate_dcsr_matrix

  subroutine allocate_block_dcsr_matrix(matrix, blocks, rows, columns, name, stat)
    type(block_dynamic_csr_matrix), intent(inout) :: matrix
    !! Number of rows and columns of blocks.
    integer, dimension(2), intent(in) :: blocks
    !! Number of rows in each block row.
    integer, dimension(blocks(1)), intent(in) :: rows
    !! Number of rows in each block column.
    integer, dimension(blocks(2)), intent(in) :: columns
    character(len=*), optional :: name
    integer, intent(out), optional :: stat

    integer :: lstat, i, j
    character(len=FIELD_NAME_LEN) :: lname

    if (present(name)) then
      lname = name
    else
      lname = ""
    end if
    matrix%name = lname

    nullify(matrix%refcount) ! Hack for gfortran component initialisation
    !                         bug.
    call addref(matrix)

    allocate(matrix%blocks(blocks(1), blocks(2)), stat=lstat)

    if (lstat/=0) goto 666

    do i=1,blocks(1)
       do j=1,blocks(2)
          call allocate(matrix%blocks(i,j),rows(i),columns(j), stat=lstat)
          if (lstat/=0) goto 666
       end do
    end do

666 if (present(stat)) then
       stat=lstat
    else
       if (lstat/=0) then
          FLAbort("Failed to allocate matrix.")
       end if
    end if

  end subroutine allocate_block_dcsr_matrix

  subroutine deallocate_block_dcsr_matrix(matrix, stat)
    type(block_dynamic_csr_matrix), intent(inout) :: matrix
    integer, intent(out), optional :: stat

    integer :: i, j, lstat

    call decref(matrix)
    if (has_references(matrix)) then
      goto 666
    end if

    do i=1,size(matrix%blocks,1)
       do j=1,size(matrix%blocks,2)
          call deallocate(matrix%blocks(i,j), stat=lstat)
          if (lstat/=0) goto 666
       end do
    end do

    deallocate(matrix%blocks, stat=lstat)

666 if (present(stat)) then
       stat=lstat
    else
       if (lstat/=0) then
          FLAbort("Failed to deallocate matrix.")
       end if
    end if

  end subroutine deallocate_block_dcsr_matrix

  function csr_block(matrix, block_i, block_j) result (block_out)
    !!< Extract block block_i, block_j from matrix.
    !!< This is the only case where the returned matrix
    !!< is not to be deallocated!!!
    type(csr_matrix) :: block_out
    type(block_csr_matrix), intent(in) :: matrix
    integer, intent(in) :: block_i, block_j

    if(matrix%diagonal.and.(block_i/=block_j)) then
      FLAbort("Attempting to extract an off-diagonal block from a diagonal block_csr_matrix.")
    end if

    block_out%clone=.true.

    block_out%sparsity=matrix%sparsity

    if (associated(matrix%val)) then
       block_out%val => matrix%val(block_i,block_j)%ptr
       ! should only be deallocated in the deallocate() call for the orig. matrix
       block_out%external_val=.true.
    end if
    if (associated(matrix%ival)) then
       block_out%ival=> matrix%ival(block_i,block_j)%ptr
       ! should only be deallocated in the deallocate() call for the orig. matrix
       block_out%external_val=.true.
    end if

    ! "Borrowed" matrices cannot have inactive nodes
    nullify(block_out%inactive)

  end function csr_block

  function dcsr_block(matrix, block_i, block_j) result (block_out)
    !!< Extract block block_i, block_j from matrix.
    type(dynamic_csr_matrix) :: block_out
    type(block_dynamic_csr_matrix), intent(in) :: matrix
    integer, intent(in) :: block_i, block_j

    block_out=matrix%blocks(block_i, block_j)

  end function dcsr_block

  function wrap_csr_matrix(sparsity, val, ival, name, stat) result (matrix)
    !!< Create a matrix using sparsity and the val or ival provided.
    !!< The wrapping matrix must be deallocated after use!!!
    type(csr_matrix) :: matrix
    type(csr_sparsity), intent(in) :: sparsity
    real, dimension(size(sparsity%colm)), intent(in), target, optional :: val
    integer, dimension(size(sparsity%colm)), intent(in), target, optional ::&
         & ival
    character(len=*), intent(in):: name
    integer, intent(out), optional :: stat

    integer :: lstat

    matrix%clone=.true.
    matrix%name=name

    nullify(matrix%refcount)
    call addref(matrix)

    matrix%sparsity=sparsity
    call incref(sparsity)

    ! This is a workaround for a gfortran initialisation bug.
    matrix%val=>null()
    matrix%ival=>null()

    if (present(val)) then
       assert(size(sparsity%colm)==size(val))
       matrix%val=>val
       ! avoid deallocation of val in deallocate():
       matrix%external_val=.true.
       lstat=0
    else if (present(ival)) then
       assert(size(sparsity%colm)==size(ival))
       matrix%ival=>ival
       ! avoid deallocation of ival in deallocate():
       matrix%external_val=.true.
       lstat=0
    else
       FLAbort("Either val or ival must be provided to wrap_matrix.")
    end if

    ! Wrapped matrices can have inactive nodes
    allocate( matrix%inactive, stat=lstat )
    nullify( matrix%inactive%ptr )

    if (present(stat)) then
       stat=lstat
    else
       if (lstat/=0) then
          FLAbort("Failed to wrap matrix.")
       end if
    end if

  end function wrap_csr_matrix

  function block_wrap_csr_matrix(sparsity, blocks, &
       val, ival, name, stat) result (matrix)
    !!< Return a matrix with the same structure but different data space to
    !!< matrix. If val is present then it is used as the data space.
    !!< Otherwise, a new space is allocated.
    !!< The wrapping matrix must be deallocated after use!!!
    type(block_csr_matrix) :: matrix
    type(csr_sparsity), intent(in) :: sparsity
    integer, dimension(2), intent(in)::blocks
    real, dimension(size(sparsity%colm)*product(blocks)), &
         intent(in), target, optional :: val
    integer, dimension(size(sparsity%colm)*product(blocks)), &
         intent(in), target, optional :: ival
    character(len=*), intent(in):: name
    integer, intent(out), optional :: stat

    integer :: lstat, i, j, bs

    lstat=0
    matrix%clone=.true.
    matrix%name=name

    nullify(matrix%refcount)
    call addref(matrix)
    matrix%sparsity=sparsity
    call incref(sparsity)

    matrix%blocks=blocks
    matrix%diagonal=.false.

    ! this is a temp. measure as long as gfortran does not do it automatically
    nullify(matrix%val)
    nullify(matrix%ival)

    if (present(val)) then
       allocate(matrix%val(blocks(1),blocks(2)), stat=lstat)
       do i=1,blocks(1)
          do j=1,blocks(2)
             bs=blockstart(matrix, i, j)
             matrix%val(i,j)%ptr=>val(bs:bs+size(matrix%sparsity%colm)-1)
          end do
       end do
       lstat=0
       ! avoid deallocation of val in deallocate():
       matrix%external_val=.true.
    else if (present(ival)) then
       allocate(matrix%val(blocks(1),blocks(2)), stat=lstat)
       do i=1,blocks(1)
          do j=1,blocks(2)
             bs=blockstart(matrix, i, j)
             matrix%val(i,j)%ptr=>val(bs:bs+size(matrix%sparsity%colm)-1)
          end do
       end do
       lstat=0
       ! avoid deallocation of val in deallocate():
       matrix%external_val=.true.
    else
       ! No data space.
       allocate(matrix%val(blocks(1),blocks(2)), stat=lstat)
       do i=1,blocks(1)
          do j=1,blocks(2)
             nullify(matrix%val(i,j)%ptr)
          end do
       end do
       lstat=0
       ! avoid deallocation of val in deallocate():
       matrix%external_val=.true.
    end if

    if (present(stat)) then
       stat=lstat
    else
       if (lstat/=0) then
          FLAbort("Failed to wrap matrix.")
       end if
    end if

  end function block_wrap_csr_matrix

  subroutine unclone_csr_matrix(matrix)
    !!< Specify that matrix is no longer a clone. This is useful for memory
    !!< management but be careful not to shoot yourself in the foot!
    type(csr_matrix), intent(inout) :: matrix

    if (matrix%clone .and. matrix%external_val) then
      FLAbort("Can't unclone this matrix as it is using externally stored values.")
    end if
    matrix%clone=.false.

  end subroutine unclone_csr_matrix

  subroutine block_csr_attach_block(matrix, blocki, blockj, val)
    !!< Having cloned a csr matrix without data, insert val as one of the
    !!< blocks.
    type(block_csr_matrix), intent(inout) :: matrix
    integer, intent(in) :: blocki, blockj
    real, dimension(size(matrix%sparsity%colm)), intent(in), target :: val

    if (.not.associated(matrix%val)) then
       allocate(matrix%val(matrix%blocks(1), matrix%blocks(2)))
    else if (.not. matrix%external_val) then
      FLAbort("Can't attach block of data as value memory has been allocated internally.")
    end if

    matrix%val(blocki,blockj)%ptr=>val

    ! avoid deallocation of val in deallocate():
    matrix%external_val=.true.

  end subroutine block_csr_attach_block

  function wrap_csr_sparsity(findrm, centrm, colm, name, &
      row_halo, column_halo) result(sparsity)
    !!< Wrap a csr_matrix around the sparsity pattern defined by the input
    !!< arguments.
    type(csr_sparsity) :: sparsity
    integer, dimension(:), intent(in), target :: findrm, colm
    integer, dimension(:), intent(in), target, optional :: centrm
    character(len=*), intent(in):: name
    type(halo_type), optional, intent(in):: row_halo, column_halo

    sparsity%name=name

    sparsity%findrm=>findrm
    if (present(centrm)) then
       sparsity%centrm=>centrm
    else
       sparsity%centrm=>null()
    end if
    sparsity%colm=>colm

    nullify(sparsity%refcount)
    call addref(sparsity)
    sparsity%wrapped=.true.

    ! Attempt to work out columns by voodoo. Not totally safe!
    sparsity%columns=maxval(colm)

    if (present(row_halo)) then
       allocate(sparsity%row_halo)
       sparsity%row_halo=row_halo
       call incref(row_halo)
    end if

    if (present(column_halo)) then
       allocate(sparsity%column_halo)
       sparsity%column_halo=column_halo
       call incref(column_halo)
    end if

  end function wrap_csr_sparsity

  pure function sparsity_size(sparsity, dim)
    !!< Clone of size function.
    integer :: sparsity_size
    type(csr_sparsity), intent(in) :: sparsity
    integer, optional, intent(in) :: dim

    integer, dimension(2) :: shape

    shape(1)=size(sparsity%findrm)-1
    shape(2)=sparsity%columns

    if (present(dim)) then
       sparsity_size=shape(dim)
    else
       sparsity_size=product(shape)
    end if

  end function sparsity_size

  pure function csr_size(matrix, dim)
    !!< Clone of size function.
    integer :: csr_size
    type(csr_matrix), intent(in) :: matrix
    integer, optional, intent(in) :: dim

    csr_size=sparsity_size(matrix%sparsity, dim)

  end function csr_size

  pure function block_csr_size(matrix, dim)
    !!< Clone of size function.
    integer :: block_csr_size
    type(block_csr_matrix), intent(in) :: matrix
    integer, optional, intent(in) :: dim

    integer, dimension(2) :: shape

    shape(1)=size(matrix%sparsity%findrm)-1
    shape(2)=matrix%sparsity%columns

    if (.not.present(dim)) then
       block_csr_size = product(shape)*product(matrix%blocks)
    else
       block_csr_size = shape(dim)*matrix%blocks(dim)
    end if

  end function block_csr_size

  pure function block_csr_block_size(matrix, dim) result (block_size)
    !!< Clone of size function. Assumes matrix is blocks of square matrices.
    integer :: block_size
    type(block_csr_matrix), intent(in) :: matrix
    integer, optional, intent(in) :: dim

    block_size=sparsity_size(matrix%sparsity, dim)

  end function block_csr_block_size

  pure function block_dcsr_block_size(matrix, block_i, block_j, dim) &
       result (block_size)
    !!< Size function for an individual matrix block.
    integer :: block_size
    type(block_dynamic_csr_matrix), intent(in) :: matrix
    integer, intent(in) :: block_i, block_j
    integer, optional, intent(in) :: dim

    block_size=size(matrix%blocks(block_i, block_j), dim)

  end function block_dcsr_block_size

  pure function dcsr_size(matrix, dim)
    !!< Clone of size function. Assumes matrix is square.
    integer :: dcsr_size
    type(dynamic_csr_matrix), intent(in) :: matrix
    integer, optional, intent(in) :: dim

    integer, dimension(2) :: shape

    shape(1)=size(matrix%colm)
    shape(2)=matrix%columns

    if (present(dim)) then
       dcsr_size=shape(dim)
    else
       dcsr_size=product(shape)
    end if

  end function dcsr_size

  pure function block_dcsr_size(matrix, dim) result(dcsr_size)
    !!< Clone of size function. Assumes matrix is square.
    integer :: dcsr_size
    type(block_dynamic_csr_matrix), intent(in) :: matrix
    integer, optional, intent(in) :: dim

    integer, dimension(2) :: shape
    integer :: i, j

    shape=0

    do i=1,size(matrix%blocks,1)
       shape(1)=shape(1)+size(matrix%blocks(i,1),1)
    end do

    do j=1,size(matrix%blocks,2)
       shape(2)=shape(2)+size(matrix%blocks(1,j),2)
    end do

    if (present(dim)) then
       dcsr_size=shape(dim)
    else
       dcsr_size=product(shape)
    end if

  end function block_dcsr_size

  pure function blocks_nodim(matrix)
  integer, dimension(2):: blocks_nodim
  type(block_csr_matrix), intent(in):: matrix

    blocks_nodim=matrix%blocks

  end function blocks_nodim

  pure function blocks_withdim(matrix, dim)
  integer blocks_withdim
  type(block_csr_matrix), intent(in):: matrix
  integer, intent(in):: dim

    blocks_withdim=matrix%blocks(dim)

  end function blocks_withdim

  pure function dcsr_blocks_nodim(matrix)
    integer, dimension(2):: dcsr_blocks_nodim
    type(block_dynamic_csr_matrix), intent(in):: matrix

    dcsr_blocks_nodim=shape(matrix%blocks)

  end function dcsr_blocks_nodim

  pure function dcsr_blocks_withdim(matrix, dim)
    integer :: dcsr_blocks_withdim
    type(block_dynamic_csr_matrix), intent(in):: matrix
    integer, intent(in):: dim

    dcsr_blocks_withdim=size(matrix%blocks,dim)

  end function dcsr_blocks_withdim

  pure function sparsity_entries(sparsity)
    !!< Return the number of (potentially) non-zero entries in matrix.
    integer :: sparsity_entries
    type(csr_sparsity), intent(in) :: sparsity

    sparsity_entries=count(sparsity%colm/=0)

  end function sparsity_entries

  pure function csr_entries(matrix)
    !!< Return the number of (potentially) non-zero entries in matrix.
    integer :: csr_entries
    type(csr_matrix), intent(in) :: matrix

    csr_entries=count(matrix%sparsity%colm/=0)

  end function csr_entries

  pure function dcsr_entries(matrix)
    !!< Return the number of (potentially) non-zero entries in matrix.
    integer :: dcsr_entries
    type(dynamic_csr_matrix), intent(in) :: matrix

    integer i, c

    c=0
    do i=1, size(matrix,1)
      c=c+size(matrix%colm(i)%ptr)
    end do
    dcsr_entries=c

  end function dcsr_entries

  pure function sparsity_row_m(sparsity, i)
    !!< Return the m indices of the ith row of sparsity.
    type(csr_sparsity), intent(in) :: sparsity
    integer, intent(in) :: i
    integer, dimension(sparsity%findrm(i+1)-sparsity%findrm(i)) :: sparsity_row_m

    sparsity_row_m=sparsity%colm(sparsity%findrm(i):sparsity%findrm(i+1)-1)

  end function sparsity_row_m

  pure function csr_row_m(matrix, i)
    !!< Return the m indices of the ith row of matrix.
    type(csr_matrix), intent(in) :: matrix
    integer, intent(in) :: i
    integer, dimension(matrix%sparsity%findrm(i+1)-matrix%sparsity&
         &%findrm(i)) :: csr_row_m

    csr_row_m=matrix%sparsity%colm&
         (matrix%sparsity%findrm(i):matrix%sparsity%findrm(i+1)-1)

  end function csr_row_m

  pure function dcsr_row_m(matrix, i)
    !!< Return the m indices of the ith row of matrix.
    type(dynamic_csr_matrix), intent(in) :: matrix
    integer, intent(in) :: i
    integer, dimension(size(matrix%colm(i)%ptr)) :: dcsr_row_m

    dcsr_row_m=matrix%colm(i)%ptr

  end function dcsr_row_m

  pure function block_csr_row_m(matrix, i)
    !!< Return the m indices of the ith row of matrix. Since all rows in a
    !!< blockmatrix are the same, we do not have to specify the block
    type(block_csr_matrix), intent(in) :: matrix
    integer, intent(in) :: i
    integer, dimension(matrix%sparsity%findrm(i+1)-matrix%sparsity&
         &%findrm(i)) :: block_csr_row_m

    block_csr_row_m=matrix%sparsity%colm&
         (matrix%sparsity%findrm(i):matrix%sparsity%findrm(i+1)-1)

  end function block_csr_row_m

  function sparsity_row_m_ptr(sparsity, i)
    !!< Return a pointer to the m indices of the ith row of matrix.
    type(csr_sparsity), intent(in) :: sparsity
    integer, intent(in) :: i
    integer, dimension(:), pointer :: sparsity_row_m_ptr

    sparsity_row_m_ptr=>sparsity%colm(sparsity%findrm(i):sparsity%findrm(i+1)-1)

  end function sparsity_row_m_ptr

  function csr_row_m_ptr(matrix, i)
    !!< Return a pointer to the m indices of the ith row of matrix.
    type(csr_matrix), intent(in) :: matrix
    integer, intent(in) :: i
    integer, dimension(:), pointer :: csr_row_m_ptr

    csr_row_m_ptr=>matrix%sparsity%colm(matrix%sparsity%findrm(i):matrix%sparsity%findrm(i+1)-1)

  end function csr_row_m_ptr

  function block_csr_row_m_ptr(matrix, i)
    !!< Return a pointer to the m indicies of the ith row of matrix. Since all
    !!< rows in a blockmatrix are the same, we do not have to specify the block
    type(block_csr_matrix), intent(in) :: matrix
    integer, intent(in) :: i
    integer, dimension(:), pointer :: block_csr_row_m_ptr

    block_csr_row_m_ptr=>matrix%sparsity%colm(matrix%sparsity%findrm(i):matrix%sparsity%findrm(i+1)-1)

  end function block_csr_row_m_ptr

  function dcsr_row_m_ptr(matrix, i)
    !!< Return a pointer to the m indices of the ith row of matrix.
    !!< For dynamic sparse matrices this remains valid until
    !!< the matrix is changed. After that call row_m_ptr again.
    type(dynamic_csr_matrix), intent(in) :: matrix
    integer, intent(in) :: i
    integer, dimension(:), pointer :: dcsr_row_m_ptr

    dcsr_row_m_ptr => matrix%colm(i)%ptr

  end function dcsr_row_m_ptr

  pure function csr_row_val(matrix, i)
    !!< Return the values of the ith row of matrix.
    type(csr_matrix), intent(in) :: matrix
    integer, intent(in) :: i
    real, dimension(matrix%sparsity%findrm(i+1)-matrix%sparsity%findrm(i)) :: csr_row_val

    csr_row_val=matrix%val( matrix%sparsity%findrm(i):matrix%sparsity%findrm(i+1)-1 )

  end function csr_row_val

  pure function block_csr_row_val(matrix, blocki, blockj, i)
    !!< Return the values of the ith row of (blocki,blockj) of matrix.
    type(block_csr_matrix), intent(in) :: matrix
    integer, intent(in) :: blocki, blockj, i
    real, dimension(matrix%sparsity%findrm(i+1)-matrix%sparsity%findrm(i)) :: block_csr_row_val

    if(.not.matrix%diagonal.or.(blocki==blockj)) then
      block_csr_row_val=matrix%val(blocki,blockj)%ptr( &
          matrix%sparsity%findrm(i):matrix%sparsity%findrm(i+1)-1)
    else
      block_csr_row_val = 0.0
    end if

  end function block_csr_row_val

  pure function block_csr_fullrow_val(matrix, blocki, i)
    !!< Return the values of all ith rows of the blocki-th row of blocks
    type(block_csr_matrix), intent(in) :: matrix
    integer, intent(in) :: blocki, i
    real, dimension((matrix%sparsity%findrm(i+1)-matrix%sparsity%findrm(i))&
         &*matrix%blocks(2)) :: block_csr_fullrow_val

    integer :: blockj, k, rowlen

    block_csr_fullrow_val = 0.0

    rowlen=matrix%sparsity%findrm(i+1)-matrix%sparsity%findrm(i)

    k=1
    do blockj=1, matrix%blocks(2)
      if(.not.matrix%diagonal.or.(blocki==blockj)) then
        block_csr_fullrow_val(k:k+rowlen-1)= &
            matrix%val(blocki,blockj)%ptr(matrix%sparsity%findrm(i):matrix%sparsity%findrm(i+1)-1 )
      end if
      k=k+rowlen
    end do

  end function block_csr_fullrow_val

  function csr_row_val_ptr(matrix, i)
    !!< Return a pointer to the values of the ith row of matrix.
    type(csr_matrix), intent(in) :: matrix
    integer, intent(in) :: i
    real, dimension(:), pointer :: csr_row_val_ptr

    csr_row_val_ptr=>matrix%val(matrix%sparsity%findrm(i):matrix%sparsity%findrm(i+1)-1)

  end function csr_row_val_ptr

  function dcsr_row_val_ptr(matrix, i)
    !!< Return a pointer to the values of the ith row of matrix.
    type(dynamic_csr_matrix), intent(in) :: matrix
    integer, intent(in) :: i
    real, dimension(:), pointer :: dcsr_row_val_ptr

    dcsr_row_val_ptr => matrix%val(i)%ptr

  end function dcsr_row_val_ptr

  function block_csr_row_val_ptr(matrix, blocki, blockj, i)
    !!< Return a pointer to the values of the ith row of matrix.
    type(block_csr_matrix), intent(in) :: matrix
    integer, intent(in) :: blocki, blockj, i
    real, dimension(:), pointer :: block_csr_row_val_ptr

    if(matrix%diagonal.and.(blocki/=blockj)) then
      FLAbort("Attempting to retrieve values in an-off diagonal block of a diagonal block_csr_matrix!")
    end if

    block_csr_row_val_ptr=> &
         matrix%val(blocki, blockj)%ptr(matrix%sparsity%findrm(i):matrix%sparsity%findrm(i+1)-1)

  end function block_csr_row_val_ptr

  function csr_row_ival_ptr(matrix, i)
    !!< Return a pointer to the values of the ith row of matrix.
    type(csr_matrix), intent(in) :: matrix
    integer, intent(in) :: i
    integer, dimension(:), pointer :: csr_row_ival_ptr

    csr_row_ival_ptr=>matrix%ival(matrix%sparsity%findrm(i):matrix%sparsity%findrm(i+1)-1)

  end function csr_row_ival_ptr

  function block_csr_row_ival_ptr(matrix, blocki, blockj, i)
    !!< Return a pointer to the values of the ith row of matrix.
    type(block_csr_matrix), intent(in) :: matrix
    integer, intent(in) :: blocki, blockj, i
    integer, dimension(:), pointer :: block_csr_row_ival_ptr

    if(matrix%diagonal.and.(blocki/=blockj)) then
      FLAbort("Attempting to retrieve values in an off-diagonal block of a diagonal block_csr_matrix!")
    end if

    block_csr_row_ival_ptr=> &
         matrix%ival(blocki, blockj)%ptr(matrix%sparsity%findrm(i):matrix%sparsity%findrm(i+1)-1)

  end function block_csr_row_ival_ptr

  function csr_diag_val_ptr(matrix, i)
    !!< Return a pointer to the values of the diagonal of ith row of matrix.
    type(csr_matrix), intent(in) :: matrix
    integer, intent(in) :: i
    real, pointer :: csr_diag_val_ptr

    csr_diag_val_ptr=>matrix%val(matrix%sparsity%centrm(i))

  end function csr_diag_val_ptr

  pure function csr_row_length(matrix, i)
    !!< Return the row length of the ith row of matrix.
    type(csr_matrix), intent(in) :: matrix
    integer, intent(in) :: i
    integer :: csr_row_length

    csr_row_length=matrix%sparsity%findrm(i+1)-matrix%sparsity%findrm(i)

  end function csr_row_length

  pure function csr_sparsity_row_length(sparsity, i)
    !!< Return the row length of the ith row of matrix.
    type(csr_sparsity), intent(in) :: sparsity
    integer, intent(in) :: i
    integer :: csr_sparsity_row_length

    csr_sparsity_row_length=sparsity%findrm(i+1)-sparsity%findrm(i)
  end function csr_sparsity_row_length

  pure function dcsr_row_length(matrix, i)
    !!< Return the row length of the ith row of matrix.
    type(dynamic_csr_matrix), intent(in) :: matrix
    integer, intent(in) :: i
    integer :: dcsr_row_length

    dcsr_row_length=size(matrix%colm(i)%ptr)

  end function dcsr_row_length

  pure function block_csr_block_row_length(matrix, i)
    !!< Return the row length of the ith row (within a block) of matrix.
    type(block_csr_matrix), intent(in) :: matrix
    integer, intent(in) :: i
    integer :: block_csr_block_row_length

    block_csr_block_row_length=matrix%sparsity%findrm(i+1)-matrix%sparsity%findrm(i)

  end function block_csr_block_row_length

  subroutine csr_initialise_inactive(matrix)
    !!< Initialises the administration for registration of inactive rows.
    !!< All rows start out as active (i.e. not inactive)
    !!< May be called as many times as you like.
    type(csr_matrix), intent(inout):: matrix

    if (.not. can_have_inactive(matrix)) then
      ewrite(1,*) "Matrix: ", trim(matrix%name)
      FLAbort("This matrix cannot have inactive rows set.")
    end if

    if (.not. has_inactive(matrix)) then
      allocate( matrix%inactive%ptr(1:size(matrix,1)) )
      matrix%inactive%ptr=.false.
    end if

  end subroutine csr_initialise_inactive

  subroutine csr_reset_inactive(matrix)
    !!< Makes all rows "active" again
    type(csr_matrix), intent(inout):: matrix

    if(has_inactive(matrix)) then
      deallocate( matrix%inactive%ptr )
    end if

  end subroutine csr_reset_inactive


  subroutine csr_set_inactive_row(matrix, row)
    !!< Registers a single row to be "inactive" this can be used for
    !!< strong boundary conditions and reference nodes.
    type(csr_matrix), intent(inout):: matrix
    integer, intent(in):: row
    character(len=255) :: buf

    call csr_initialise_inactive(matrix)

    if (row > size(matrix%inactive%ptr)) then
      buf = "Error: attempting to set row " // int2str(row) // " to be inactive, but only " // &
          &  int2str(size(matrix%inactive%ptr)) // " rows. Check your reference pressure node?"
      FLExit(trim(buf))
    end if
    matrix%inactive%ptr(row)=.true.

  end subroutine csr_set_inactive_row

  subroutine csr_set_inactive_rows(matrix, rows)
    !!< Registers a number of rows to be "inactive" this can be used for
    !!< strong boundary conditions and reference nodes.
    type(csr_matrix), intent(inout):: matrix
    integer, dimension(:), intent(in):: rows

    call csr_initialise_inactive(matrix)

    matrix%inactive%ptr(rows)=.true.

  end subroutine csr_set_inactive_rows

  function csr_get_inactive_mask(matrix)
    !!< Returns a pointer to a logical array that indicates inactive rows
    !!< May return a null pointer, in which case no rows are inactive
    logical, dimension(:), pointer:: csr_get_inactive_mask
    type(csr_matrix), intent(in):: matrix

    if (associated(matrix%inactive)) then
      csr_get_inactive_mask => matrix%inactive%ptr
    else
      nullify(csr_get_inactive_mask)
    end if

  end function csr_get_inactive_mask

  pure function can_have_inactive(matrix)
    type(csr_matrix), intent(in) :: matrix

    logical :: can_have_inactive

    can_have_inactive = associated(matrix%inactive)

  end function can_have_inactive

  pure function has_inactive(matrix)
    type(csr_matrix), intent(in) :: matrix

    logical :: has_inactive

    if(can_have_inactive(matrix)) then
      has_inactive = associated(matrix%inactive%ptr)
    else
      has_inactive = .false.
    end if

  end function has_inactive

  pure function csr_has_solver_cache(matrix)
    logical :: csr_has_solver_cache
    type(csr_matrix), intent(in) :: matrix
    ! this should only be possible for a csr_matrix returned from block()
    csr_has_solver_cache = .false.


  end function csr_has_solver_cache

  pure function block_csr_has_solver_cache(matrix)
    logical :: block_csr_has_solver_cache
    type(block_csr_matrix), intent(in) :: matrix

      ! don't think this is possible, but hey
      block_csr_has_solver_cache = .false.


  end function block_csr_has_solver_cache

  subroutine csr_destroy_solver_cache(matrix)
    type(csr_matrix), intent(inout) :: matrix

    integer:: ierr

    return

  end subroutine csr_destroy_solver_cache

  subroutine block_csr_destroy_solver_cache(matrix)
    type(block_csr_matrix), intent(inout) :: matrix

    integer:: ierr

    return

  end subroutine block_csr_destroy_solver_cache

  subroutine csr_zero(matrix)
    !!< Zero the entries of a csr matrix.
    type(csr_matrix), intent(inout) :: matrix

    if (associated(matrix%val)) then
       matrix%val=0.0
    end if
    if (associated(matrix%ival)) then
       matrix%ival=0
    end if
    if (has_inactive(matrix)) then
       deallocate(matrix%inactive%ptr)
       nullify(matrix%inactive%ptr)
    end if

    ! this invalidates the solver context
    call destroy_solver_cache(matrix)

  end subroutine csr_zero

  subroutine block_csr_zero(matrix)
    !!< Zero the entries of a csr matrix.
    type(block_csr_matrix), intent(inout) :: matrix

    integer :: i,j

    if (associated(matrix%val)) then
       if(matrix%equal_diagonal_blocks) then
         matrix%val(1,1)%ptr=0.0
       else if(matrix%diagonal) then
         do i=1,matrix%blocks(1)
           matrix%val(i,i)%ptr=0.0
         end do
       else
         do i=1,matrix%blocks(1)
            do j=1,matrix%blocks(2)
               matrix%val(i,j)%ptr=0.0
            end do
         end do
       end if
    end if
    if (associated(matrix%ival)) then
       if(matrix%equal_diagonal_blocks) then
         matrix%ival(1,1)%ptr=0.0
       else if(matrix%diagonal) then
         do i=1,matrix%blocks(1)
           matrix%ival(i,i)%ptr=0.0
         end do
       else
         do i=1,matrix%blocks(1)
            do j=1,matrix%blocks(2)
               matrix%ival(i,j)%ptr=0.0
            end do
         end do
       end if
    end if

    ! this invalidates the solver context
    call destroy_solver_cache(matrix)

  end subroutine block_csr_zero

  subroutine dcsr_zero(matrix)
    !!< Zero the entries of a dynamic csr matrix.
    type(dynamic_csr_matrix), intent(inout) :: matrix

    integer :: i

    do i=1,size(matrix,1)
       matrix%val(i)%ptr=0.0
    end do

  end subroutine dcsr_zero

  subroutine csr_zero_row(matrix, i)
    !!< Zero the entries of a particular row of a csr matrix.
    type(csr_matrix), intent(inout) :: matrix
    integer, intent(in) :: i

    real, dimension(:), pointer :: val
    integer, dimension(:), pointer :: ival

    if (associated(matrix%val)) then
       val => row_val_ptr(matrix, i)
       val = 0.0
    end if
    if (associated(matrix%ival)) then
       ival => row_ival_ptr(matrix, i)
       ival = 0.0
    end if

  end subroutine csr_zero_row

  subroutine block_csr_zero_single_row(matrix, blocki, i)
    !!< Zero the entries of a particular row of a block csr matrix.
    type(block_csr_matrix), intent(inout) :: matrix
    integer, intent(in) :: blocki, i
    integer :: k

    real, dimension(:), pointer :: val
    integer, dimension(:), pointer :: ival

    if (associated(matrix%val)) then
       do k=1,matrix%blocks(2)
          if(matrix%diagonal.and.(blocki/=k)) cycle
          val => row_val_ptr(matrix, blocki, k, i)
          val = 0.0
       end do
    end if
    if (associated(matrix%ival)) then
       do k=1,matrix%blocks(2)
          if(matrix%diagonal.and.(blocki/=k)) cycle
          ival => row_ival_ptr(matrix, blocki, k, i)
          ival = 0
       end do
    end if

  end subroutine block_csr_zero_single_row

  subroutine block_csr_zero_row(matrix, i)
    !!< Zero the entries of a particular row in all blocks of a block csr matrix.
    type(block_csr_matrix), intent(inout) :: matrix
    integer, intent(in) :: i
    integer :: j, k

    real, dimension(:), pointer :: val
    integer, dimension(:), pointer :: ival

    if (associated(matrix%val)) then
       do j=1,matrix%blocks(1)
          do k=1,matrix%blocks(2)
             if(matrix%diagonal.and.(j/=k)) cycle
             val => row_val_ptr(matrix, j, k, i)
             val = 0.0
          end do
       end do
    end if
    if (associated(matrix%ival)) then
       do j=1,matrix%blocks(1)
          do k=1,matrix%blocks(2)
             if(matrix%diagonal.and.(j/=k)) cycle
             ival => row_ival_ptr(matrix, j, k, i)
             ival = 0.0
          end do
       end do
    end if

  end subroutine block_csr_zero_row

  subroutine dcsr_zero_column(matrix,column)
    !!< Zero the entries of a dynamic csr matrix.
    type(dynamic_csr_matrix), intent(inout) :: matrix
    integer, intent(in) :: column

    integer, dimension(:), pointer :: row_ptr
    integer :: i,j

    do  i =1 ,size(matrix,1)

       row_ptr => matrix%colm(i)%ptr

       if(any(row_ptr==column)) then
          do j=1,size(row_ptr)
             if(row_ptr(j)==column) then
                matrix%val(i)%ptr(j)=0.0
             end if
          end do
       end if

    end do

  end subroutine dcsr_zero_column

  function csr_sparsity_pos(sparsity, i, j, save_pos)
    !!< Return the location in sparsity of element (i,j)
    integer :: csr_sparsity_pos
    type(csr_sparsity), intent(in) :: sparsity
    integer, intent(in) :: i,j
    ! an attempt at optimisation...
    ! if save_pos is present, test to see if it's correct, if yes then return it
    ! if no, then carry on as normal but save the position and return it as save_pos
    integer, intent(inout), optional :: save_pos

    integer, dimension(:), pointer :: row
    integer :: rowpos, base
    integer :: lower_pos, lower_j
    integer :: upper_pos, upper_j
    integer :: this_pos, this_j

    if (present(save_pos)) then
      if (save_pos>=sparsity%findrm(i) .and. save_pos<sparsity%findrm(i+1)) then
        if (sparsity%colm(save_pos)==j) then
          csr_sparsity_pos=save_pos
          return
        end if
      end if
    end if

    row => row_m_ptr(sparsity,i)

    if (sparsity%sorted_rows) then
       ! The j values in row are sorted in ascending order so we can do a
       ! fast bisection search.

       ! Base is the last position in colm of the previous row.
       base=sparsity%findrm(i)-1

       upper_pos=size(row)
       upper_j=row(upper_pos)
       lower_pos=1
       lower_j=row(1)

       if (upper_j<j) then
          csr_sparsity_pos=0
          goto 42
       else if (upper_j==j) then
          csr_sparsity_pos=upper_pos+base
          goto 42
       else if (lower_j>j) then
          csr_sparsity_pos=0
          goto 42
       else if(lower_j==j) then
          csr_sparsity_pos=lower_pos+base
          goto 42
       end if

       bisection_loop: do while (upper_pos-lower_pos>1)
          this_pos=(upper_pos+lower_pos)/2
          this_j=row(this_pos)

          if(this_j == j) then
            csr_sparsity_pos=this_pos+base
            goto 42
          else if(this_j > j) then
            ! this_j>j
            upper_j=this_j
            upper_pos=this_pos
          else
            ! this_j<j
            lower_j=this_j
            lower_pos=this_pos
          end if

!          select case(this_j-j)
!          case(0)
!             csr_sparsity_pos=this_pos+base
!             goto 42
!          case(1:)
!             ! this_j>j
!             upper_j=this_j
!             upper_pos=this_pos
!          case(:-1)
!             ! this_j<j
!             lower_j=this_j
!             lower_pos=this_pos
!          end select

       end do bisection_loop

       csr_sparsity_pos=0

    else
       ! Slower, more general case.

       ! Under f2003 setting rowpos to zero is not required.
       rowpos=0

       rowpos=minloc(row, 1, mask=(row==j))

       if (rowpos==0) then
          ! i,j is not in matrix.
          csr_sparsity_pos=0
       else
          csr_sparsity_pos=sparsity%findrm(i)-1 +rowpos
       end if
    end if

42  if(present(save_pos)) save_pos = csr_sparsity_pos

  end function csr_sparsity_pos

  function csr_pos(matrix, i, j, save_pos)
    !!< Return the location in matrix of element (i,j)
    integer :: csr_pos
    type(csr_matrix), intent(in) :: matrix
    integer, intent(in) :: i,j
    integer, intent(inout), optional :: save_pos

    csr_pos = csr_sparsity_pos(matrix%sparsity, i, j, save_pos=save_pos)

  end function csr_pos

  function block_csr_pos(matrix, i, j, save_pos)
    !!< Return the location in matrix of element (i,j)
    !!< in some block.
    integer :: block_csr_pos
    type(block_csr_matrix), intent(in) :: matrix
    integer, intent(in) :: i,j
    integer, intent(inout), optional :: save_pos

    block_csr_pos = csr_sparsity_pos(matrix%sparsity, i, j, save_pos=save_pos)

  end function block_csr_pos

  function dcsr_pos_noadd(matrix, i, j)
    !!< Return the location in matrix of element (i,j)
    integer :: dcsr_pos_noadd
    type(dynamic_csr_matrix), intent(in) :: matrix

    integer, intent(in) :: i,j
    integer, dimension(:), pointer :: row
    integer, dimension(1) :: rowpos

    row=>matrix%colm(i)%ptr

    if (.not.any(j==row)) then
       ! i,j is not in matrix.
       dcsr_pos_noadd=0
       return
    end if

    rowpos=0
    rowpos=minloc(row, row==j)

    dcsr_pos_noadd=rowpos(1)

  end function dcsr_pos_noadd

  function dcsr_pos(matrix, i, j, add)
    !!< Return the location in matrix of element (i,j)
    integer :: dcsr_pos
    type(dynamic_csr_matrix), intent(inout) :: matrix
    integer, intent(in) :: i,j
    !! Flag which determines whether new entries are added.
    logical, intent(in) :: add

    integer, dimension(:), pointer :: row
    real, dimension(:), pointer :: val
    integer, dimension(1) :: rowpos

    row=>matrix%colm(i)%ptr
    val=>matrix%val(i)%ptr

    if (.not.any(j==row)) then
       if (.not.add) then
          ! i,j is not in matrix.
          dcsr_pos=0
          return
       end if

       rowpos=0
       if (size(row)>0) then
          if (all(j>row)) then
             rowpos(1)=size(row)
          else
             rowpos=minloc(row, mask=row>j)-1
          end if
       end if

       ! Lengthen the row by one place
       allocate(matrix%colm(i)%ptr(size(row)+1), &
            &   matrix% val(i)%ptr(size(row)+1))

       ! Copy the old row in place
       if (rowpos(1)>0) then
          matrix%colm(i)%ptr(:rowpos(1))=row(:rowpos(1))
          matrix%val(i)%ptr(:rowpos(1))=val(:rowpos(1))
       end if
       if (rowpos(1)<size(row)) then
          matrix%colm(i)%ptr(rowpos(1)+2:)=row(rowpos(1)+1:)
          matrix%val(i)%ptr(rowpos(1)+2:)=val(rowpos(1)+1:)
       end if
       ! Destroy old memory.
#ifdef DDEBUG
       val=ieee_value(0.0, ieee_quiet_nan)
#endif
       deallocate(row, val)

       row=>matrix%colm(i)%ptr
       val=>matrix%val(i)%ptr

       ! Insert the new location.
       row(rowpos(1)+1)=j
       val(rowpos(1)+1)=0.0
    end if

    rowpos=0
    rowpos=minloc(row, row==j)

    dcsr_pos=rowpos(1)

  end function dcsr_pos

  subroutine csr_addto(matrix, i, j, val, save_pos)
    !!< Add val to matrix(i,j)
    type(csr_matrix), intent(inout) :: matrix
    integer, intent(in) :: i,j
    real, intent(in) :: val
    integer, intent(inout), optional :: save_pos

    integer :: mpos

    if (val==0) return ! No point doing nothing.

    mpos = pos(matrix,i,j,save_pos=save_pos)

    if (associated(matrix%val)) then
       if(mpos==0) then
          FLAbort("Attempting to set value in matrix outside sparsity pattern.")
       end if
       matrix%val(mpos)=matrix%val(mpos)+val
    else if (associated(matrix%ival)) then
       matrix%ival(mpos)=matrix%ival(mpos)+val
    else
       FLAbort("Attempting to set value in a matrix with no value space.")
    end if

  end subroutine csr_addto

  subroutine csr_iaddto(matrix, i, j, val, save_pos)
    !!< Add val to matrix(i,j)
    type(csr_matrix), intent(inout) :: matrix
    integer, intent(in) :: i,j
    integer, intent(in) :: val
    integer, intent(inout), optional :: save_pos

    integer :: mpos

    mpos = pos(matrix,i,j,save_pos=save_pos)

    if (val==0) return ! No point doing nothing.

    if (associated(matrix%val)) then
       matrix%val(mpos)=matrix%val(mpos)+val
    else if (associated(matrix%ival)) then
       matrix%ival(mpos)=matrix%ival(mpos)+val
    else
       FLAbort("Attempting to set value in a matrix with no value space.")
    end if

  end subroutine csr_iaddto

  subroutine csr_vaddto(matrix, i, j, val, mask)
    !!< Add val to matrix(i,j)
    type(csr_matrix), intent(inout) :: matrix
    integer, dimension(:), intent(in) :: i,j
    real, dimension(size(i),size(j)), intent(in) :: val
    logical, dimension(size(i), size(j)), intent(in), optional :: mask

    integer :: iloc, jloc
    logical, dimension(size(i), size(j)) :: l_mask

    if(present(mask)) then
      l_mask = mask
    else
      l_mask = .true.
    end if

    do iloc=1,size(i)
       do jloc=1,size(j)
          if(.not.l_mask(iloc,jloc)) cycle
          call addto(matrix, i(iloc), j(jloc), val(iloc,jloc))
       end do
    end do

  end subroutine csr_vaddto

  subroutine block_csr_addto(matrix, blocki, blockj, i, j, val, save_pos)
    !!< Add val to matrix(i,j)
    type(block_csr_matrix), intent(inout) :: matrix
    integer, intent(in) :: blocki,blockj,i,j
    real, intent(in) :: val
    integer, intent(inout), optional :: save_pos

    integer :: mpos

    if(matrix%diagonal.and.(blocki/=blockj)) then
      FLAbort("Attempting to set value in an off-diagonal block of a diagonal block_csr_matrix.")
    end if

    mpos = pos(matrix, i, j, save_pos=save_pos)

    if (associated(matrix%val)) then
       matrix%val(blocki, blockj)%ptr(mpos)&
            =matrix%val(blocki, blockj)%ptr(mpos)+val

    else if (associated(matrix%ival)) then
       matrix%ival(blocki, blockj)%ptr(mpos)&
            =matrix%ival(blocki, blockj)%ptr(mpos)+val
    else
       FLAbort("Attempting to set value in a matrix with no value space.")
    end if

  end subroutine block_csr_addto

  subroutine block_csr_vaddto(matrix, blocki, blockj, i, j, val, mask)
    !!< Add val to matrix(i,j)
    type(block_csr_matrix), intent(inout) :: matrix
    integer, intent(in) :: blocki,blockj
    integer, dimension(:), intent(in) :: i,j
    real, dimension(size(i),size(j)), intent(in) :: val
    logical, dimension(size(i), size(j)), intent(in), optional :: mask

    integer :: iloc, jloc
    logical, dimension(size(i), size(j)) :: l_mask

    if(matrix%diagonal.and.(blocki/=blockj)) then
      FLAbort("Attempting to set value in an off-diagonal block of a diagonal block_csr_matrix.")
    end if

    if(present(mask)) then
      l_mask = mask
    else
      l_mask = .true.
    end if

    do iloc=1,size(i)
       do jloc=1,size(j)
          if(.not.l_mask(iloc, jloc)) cycle
          call addto(matrix, blocki,blockj, &
               &                 i(iloc),j(jloc), val(iloc,jloc))
       end do
    end do

  end subroutine block_csr_vaddto

  subroutine block_csr_blocks_addto(matrix, i, j, val, block_mask)
    !!< Add the (blocki, blockj, :, :) th matrix of val onto the (blocki, blockj) th
    !!< block of the block csr matrix, for all blocks of the block csr matrix.

    type(block_csr_matrix), intent(inout) :: matrix
    integer, dimension(:), intent(in) :: i
    integer, dimension(:), intent(in) :: j
    real, dimension(matrix%blocks(1), matrix%blocks(2), size(i), size(j)), intent(in) :: val
    logical, dimension(matrix%blocks(1), matrix%blocks(2)), intent(in), optional :: block_mask

    integer, dimension(size(i), size(j)) :: positions
    logical, dimension(matrix%blocks(1), matrix%blocks(2)) :: l_block_mask

    integer :: blocki, blockj, iloc, jloc

    if(present(block_mask)) then
      l_block_mask = block_mask
    else
      l_block_mask = .true.
    end if

    ! this is optimised so that row searches are only done once
    ! we do however want to want to keep the block loops on the outside
    ! to improve data locality.

    do iloc = 1, size(positions, 1)
      do jloc = 1, size(positions, 2)
        if (all(val(:, :, iloc, jloc)==0.0)) cycle
        positions(iloc,jloc)=pos(matrix%sparsity,i(iloc),j(jloc))
      end do
    end do

    do blocki = 1, matrix%blocks(1)
      do blockj = 1, matrix%blocks(2)
        if(.not.l_block_mask(blocki, blockj)) cycle
        do iloc = 1, size(positions, 1)
          do jloc = 1, size(positions, 2)
            ! Don't add zeros into the matrix, especially as these may be
            ! at invalid locations.
            if(val(blocki, blockj, iloc, jloc)==0) cycle
            matrix%val(blocki, blockj)%ptr(positions(iloc,jloc))= &
               matrix%val(blocki, blockj)%ptr(positions(iloc,jloc)) + &
               val(blocki, blockj, iloc, jloc)
          end do
        end do
      end do
    end do

  end subroutine block_csr_blocks_addto

  subroutine block_csr_baddto(matrix, blocki, blockj, mblock, scalar)
    !!< Add csr_matrix to a block_csr_matrix, where the csr_matrix has the same
    !!< sparsity (or a subset of it) as the blocks of the block_csr_matrix
    type(block_csr_matrix), intent(inout) :: matrix
    integer, intent(in) :: blocki,blockj
    type(csr_matrix), intent(in):: mblock
    !! if present add scalar*mblock:
    real, optional, intent(in):: scalar

    real, pointer:: blockijval(:), val_ptr(:)
    integer, pointer:: col_ptr(:)
    real lscalar
    integer row, col

    if(matrix%diagonal.and.(blocki/=blockj)) then
      FLAbort("Attempting to set value in an off-diagonal block of a diagonal block_csr_matrix.")
    end if

    if (mblock%clone .or. matrix%clone) then
       ! if one of matrix and mblock is a clone of the other, or both are clones
       ! of the same original, we only have to copy the values:
       if (associated(matrix%sparsity%findrm, mblock%sparsity%findrm) .and. &
            associated(matrix%sparsity%colm, mblock%sparsity%colm)) then

          blockijval => matrix%val(blocki, blockj)%ptr
          if (present(scalar)) then
             blockijval=blockijval+mblock%val*scalar
          else
             blockijval=blockijval+mblock%val
          end if
          return
       end if
    end if

    ! the safe way: all entries are add in one by one...
    if (present(scalar)) then
      lscalar=scalar
    else
      lscalar=1.0
    end if

    do row=1, size(mblock,1)
      col_ptr => row_m_ptr(mblock, row)
      val_ptr => row_val_ptr(mblock, row)
      do col=1, size(col_ptr)
        call addto(matrix, blocki, blockj, row, col_ptr(col), &
          val_ptr(col)*lscalar)
      end do
    end do

  end subroutine block_csr_baddto

  subroutine block_csr_bset(matrix, blocki, blockj, mblock)
    !!< Assign csr_matrix to a block_csr_matrix, where the csr_matrix has the same
    !!< sparsity
    type(block_csr_matrix), intent(inout) :: matrix
    integer, intent(in) :: blocki,blockj
    type(csr_matrix), intent(in):: mblock

    real, pointer:: blockijval(:), val_ptr(:)
    integer, pointer:: col_ptr(:)
    integer row, col

    if(matrix%diagonal.and.(blocki/=blockj)) then
      FLAbort("Attempting to set value in an off-diagonal block of a diagonal block_csr_matrix.")
    end if

    if (associated(matrix%sparsity%findrm, mblock%sparsity%findrm) .and. &
         associated(matrix%sparsity%colm, mblock%sparsity%colm)) then

       blockijval => matrix%val(blocki, blockj)%ptr
       blockijval=mblock%val
       return
     else
       do row=1, size(mblock,1)
         col_ptr => row_m_ptr(mblock, row)
         val_ptr => row_val_ptr(mblock, row)
         do col=1, size(col_ptr)
           call set(matrix, blocki, blockj, row, col_ptr(col), &
             val_ptr(col))
         end do
       end do
    end if
  end subroutine block_csr_bset

  subroutine csr_scale(matrix, scale)
    !!< Scale matrix by scale.
    type(csr_matrix), intent(inout) :: matrix
    real, intent(in) :: scale

    matrix%val=matrix%val*scale

  end subroutine csr_scale

  subroutine block_csr_scale(matrix, scale)
    !!< Scale matrix by scale.
    type(block_csr_matrix), intent(inout) :: matrix
    real, intent(in) :: scale
    !
    integer :: d1,d2

    do d1 = 1, matrix%blocks(1)
       do d2 = 1, matrix%blocks(2)
          matrix%val(d1,d2)%ptr=matrix%val(d1,d2)%ptr*scale
       end do
    end do

  end subroutine block_csr_scale

  subroutine block_csr_bvaddto(matrixA, blocki, blockj, matrixB, scalar)
    !!< Add all blocks of a block_csr_matrix to another block_csr_matrix,
    !!< where the same sparsity pattern but possibly more blocks
    type(block_csr_matrix), intent(inout) :: matrixA
    type(block_csr_matrix), intent(in):: matrixB
    ! The compiler on AIX has some unreasonable obkection to the blocks
    ! function.
!!$    integer, dimension(blocks(matrixB,1)), intent(in) :: blocki
!!$    integer, dimension(blocks(matrixB,2)), intent(in) :: blockj
       integer, dimension(matrixB%blocks(1)), intent(in) :: blocki
       integer, dimension(matrixB%blocks(2)), intent(in) :: blockj
    !! if present add scalar*matrixB:
    real, optional, intent(in):: scalar

    type(csr_matrix) blockij
    integer :: i, j

    do i=1, size(blocki)
       do j=1, size(blockj)
          if(matrixB%diagonal.and.(i/=j)) then
            FLAbort("Attempting to retrive an off-diagonal block of a diagonal block_csr_matrix.")
          end if
          if(matrixA%diagonal.and.(blocki(i)/=blockj(j))) then
            FLAbort("Attempting to set values in an off-diagonal block of a diagonal block_csr_matrix.")
          end if

          blockij=block(matrixB, i, j)
          call addto(matrixA, blocki(i), blockj(j), blockij, scalar=scalar)
       end do
    end do

  end subroutine block_csr_bvaddto

  subroutine dcsr_addto(matrix, i, j, val)
    !!< Add val to matrix(i,j)
    type(dynamic_csr_matrix), intent(inout) :: matrix
    integer, intent(in) :: i, j
    real, intent(in) :: val

    integer :: rowpos

    ! Because pos has side effects, it is a very good idea to call it
    ! before the assemble.
    rowpos=pos(matrix, i, j, add=.true.)

    matrix%val(i)%ptr(rowpos)=matrix%val(i)%ptr(rowpos)+val

  end subroutine dcsr_addto

  subroutine dcsr_dcsraddto(matrix1,matrix2)
    !!< Add matrix2 to matrix1
    !!< could probably do with optimizing
    type(dynamic_csr_matrix), intent(inout) :: matrix1
    type(dynamic_csr_matrix), intent(in) :: matrix2

    !locals
    integer :: i,j

    assert(size(matrix1,1)==size(matrix2,1))
    assert(size(matrix1,2)==size(matrix2,2))

    do i = 1,size(matrix2%colm)
       do j = 1,size(matrix2%colm(i)%ptr)
          call addto(matrix1,i,matrix2%colm(i)%ptr(j),matrix2%val(i)%ptr(j))
       end do
    end do

  end subroutine dcsr_dcsraddto

  subroutine csr_csraddto(matrix1,matrix2, scale)
    !!< Add matrix2 to matrix1: sparsity must be the same though
    type(csr_matrix), intent(inout) :: matrix1
    type(csr_matrix), intent(in) :: matrix2
    real, optional, intent(in) :: scale

    !locals
    integer :: i,j
    real :: l_scale

    assert(size(matrix1,1)==size(matrix2,1))
    assert(size(matrix1,2)==size(matrix2,2))

    if(present(scale)) then
      l_scale=scale
    else
      l_scale=1.0
    end if

    do i = 1,size(matrix2%sparsity%findrm)-1
       do j = matrix2%sparsity%findrm(i),matrix2%sparsity%findrm(i+1)-1
          call addto(matrix1,i,matrix2%sparsity%colm(j),l_scale*matrix2%val(j))
       end do
    end do

  end subroutine csr_csraddto

  subroutine dcsr_vaddto(matrix, i, j, val)
    !!< Add val to matrix(i,j)
    type(dynamic_csr_matrix), intent(inout) :: matrix
    integer, dimension(:), intent(in) :: i, j
    real, dimension(size(i),size(j)), intent(in) :: val

    integer :: iloc, jloc

    do iloc=1,size(i)
       do jloc=1,size(j)
          call addto(matrix, i(iloc), j(jloc), val(iloc,jloc))
       end do
    end do

  end subroutine dcsr_vaddto

  subroutine dcsr_vaddto1(matrix, i, j, val)
    !!< Add val to matrix(i,j)
    type(dynamic_csr_matrix), intent(inout) :: matrix
    integer, intent(in) :: i
    integer, dimension(:), intent(in) :: j
    real, dimension(size(j)), intent(in) :: val

    integer :: jloc

    do jloc=1,size(j)
       call addto(matrix, i, j(jloc), val(jloc))
    end do

  end subroutine dcsr_vaddto1

  subroutine dcsr_vaddto2(matrix, i, j, val)
    !!< Add val to matrix(i,j)
    type(dynamic_csr_matrix), intent(inout) :: matrix
    integer, dimension(:), intent(in) :: i
    integer, intent(in) :: j
    real, dimension(size(i)), intent(in) :: val

    integer :: iloc

    do iloc=1,size(i)
       call addto(matrix, i(iloc), j, val(iloc))
    end do

  end subroutine dcsr_vaddto2

  subroutine dcsr_set(matrix, i, j, val)
    !!< Add val to matrix(i,j)
    type(dynamic_csr_matrix), intent(inout) :: matrix
    integer, intent(in) :: i, j
    real, intent(in) :: val

    integer :: rowpos

    ! Because pos has side effects, it is a very good idea to call it
    ! before the assemble.
    rowpos=pos(matrix, i, j, add=.true.)

    matrix%val(i)%ptr(rowpos)=val

  end subroutine dcsr_set

  subroutine dcsr_vset(matrix, i, j, val)
    !!< Add val to matrix(i,j)
    type(dynamic_csr_matrix), intent(inout) :: matrix
    integer, dimension(:), intent(in) :: i, j
    real, dimension(size(i),size(j)), intent(in) :: val

    integer :: iloc, jloc

    do iloc=1,size(i)
       do jloc=1,size(j)
          call set(matrix, i(iloc), j(jloc), val(iloc,jloc))
       end do
    end do

  end subroutine dcsr_vset

  subroutine dcsr_set_row(matrix, i, j, val)
    !!< Add val to matrix(i,j)
    type(dynamic_csr_matrix), intent(inout) :: matrix
    integer, intent(in) :: i
    integer, dimension(:), intent(in) :: j
    real, dimension(size(j)), intent(in) :: val

    integer :: jloc

    do jloc=1,size(j)
       call set(matrix, i, j(jloc), val(jloc))
    end do

  end subroutine dcsr_set_row

  subroutine dcsr_set_col(matrix, i, j, val)
    !!< Add val to matrix(i,j)
    type(dynamic_csr_matrix), intent(inout) :: matrix
    integer, dimension(:), intent(in) :: i
    integer, intent(in) :: j
    real, dimension(size(i)), intent(in) :: val

    integer :: iloc

    do iloc=1,size(i)
       call set(matrix, i(iloc), j, val(iloc))
    end do

  end subroutine dcsr_set_col

  subroutine csr_addto_diag(matrix, i, val, scale, save_pos)
    !!< Add val to matrix(i,i)
    type(csr_matrix), intent(inout) :: matrix
    integer, intent(in) :: i
    real, intent(in) :: val
    real, intent(in), optional ::scale
    integer, intent(inout), optional :: save_pos

    integer :: mpos

    if(associated(matrix%sparsity%centrm)) then
      mpos = matrix%sparsity%centrm(i)
    else
      mpos = pos(matrix,i,i,save_pos=save_pos)
    end if

    if (associated(matrix%val)) then
       if(present(scale)) then
         matrix%val(mpos)=matrix%val(mpos)+val*scale
       else
         matrix%val(mpos)=matrix%val(mpos)+val
       end if
    else if (associated(matrix%ival)) then
       if(present(scale)) then
         matrix%ival(mpos)=matrix%ival(mpos)+val*scale
       else
         matrix%ival(mpos)=matrix%ival(mpos)+val
       end if
    else
       FLAbort("Attempting to set value in a matrix with no value space.")
    end if

  end subroutine csr_addto_diag

  subroutine csr_vaddto_diag(matrix, i, val, scale)
    !!< Add val to matrix(i,i)
    type(csr_matrix), intent(inout) :: matrix
    integer, dimension(:), intent(in) :: i
    real, dimension(size(i)), intent(in) :: val
    real, intent(in), optional :: scale

    integer :: iloc

    do iloc=1,size(i)
          call addto_diag(matrix, i(iloc), val(iloc), scale=scale)
    end do

  end subroutine csr_vaddto_diag

  subroutine block_csr_addto_diag(matrix, blocki, blockj, i, val, scale, save_pos)
    !!< Add val to matrix(i,i)
    !!< Adding to the diagonal of a non-diagonal block is supported.
    type(block_csr_matrix), intent(inout) :: matrix
    integer, intent(in) :: blocki,blockj, i
    real, intent(in) :: val
    real, intent(in), optional :: scale
    integer, intent(inout), optional :: save_pos

    integer :: mpos

    if(matrix%diagonal.and.(blocki/=blockj)) then
      FLAbort("Attempting to set value in an off-diagonal block of a diagonal block_csr_matrix.")
    end if

    if(associated(matrix%sparsity%centrm)) then
      mpos = matrix%sparsity%centrm(i)
    else
      mpos = pos(matrix,i,i,save_pos=save_pos)
    end if

    if (associated(matrix%val)) then
       if(present(scale)) then
          matrix%val(blocki,blockj)%ptr(mpos)&
                =matrix%val(blocki, blockj)%ptr(mpos)+val*scale
       else
          matrix%val(blocki,blockj)%ptr(mpos)&
                =matrix%val(blocki, blockj)%ptr(mpos)+val
       end if
    else if (associated(matrix%ival)) then
       if(present(scale)) then
          matrix%ival(blocki, blockj)%ptr(mpos)&
                =matrix%ival(blocki, blockj)%ptr(mpos)+val*scale
       else
          matrix%ival(blocki, blockj)%ptr(mpos)&
                =matrix%ival(blocki, blockj)%ptr(mpos)+val
       end if
    else
       FLAbort("Attempting to set value in a matrix with no value space.")
    end if

  end subroutine block_csr_addto_diag

  subroutine block_csr_vaddto_diag(matrix, blocki, blockj, i, val, scale)
    !!< Add val to matrix(i,i)
    type(block_csr_matrix), intent(inout) :: matrix
    integer, intent(in) :: blocki, blockj
    integer, dimension(:), intent(in) :: i
    real, dimension(size(i)), intent(in) :: val
    real, intent(in), optional :: scale

    integer :: iloc

    if(matrix%diagonal.and.(blocki/=blockj)) then
      FLAbort("Attempting to set value in an off-diagonal block of a diagonal block_csr_matrix.")
    end if

    do iloc=1,size(i)
          call addto_diag(matrix, blocki, blockj, i(iloc), val(iloc), scale=scale)
    end do

  end subroutine block_csr_vaddto_diag

  subroutine csr_set(matrix, i, j, val, save_pos)
    !!< Add val to matrix(i,j)
    type(csr_matrix), intent(inout) :: matrix
    integer, intent(in) :: i,j
    real, intent(in) :: val
    integer, intent(inout), optional :: save_pos

    integer :: mpos

    mpos = pos(matrix,i,j,save_pos=save_pos)
    !In debugging mode, check that the entry actually exists.
    assert(mpos>0)

    if (associated(matrix%val)) then
       matrix%val(mpos)=val
    else if (associated(matrix%ival)) then
       matrix%ival(mpos)=val
    else
       FLAbort("Attempting to set value in a matrix with no value space.")
    end if

  end subroutine csr_set

  subroutine csr_csr_set(out_matrix, in_matrix)
    !!< Set out_matrix to in_matrix. This will only work if the matrices have
    !!< the same sparsity.
    type(csr_matrix), intent(inout) :: out_matrix
    type(csr_matrix), intent(in) :: in_matrix

    !
    integer :: row, i
    integer, dimension(:), pointer:: cols
    real, dimension(:), pointer :: vals
#ifdef DDEBUG
    logical :: matrix_same_shape
#endif

    if(in_matrix%sparsity%refcount%id==out_matrix%sparsity%refcount%id) then
       !Code for the same sparsity
       if (associated(out_matrix%ival)) then
          assert(associated(in_matrix%ival))
       else
          assert((associated(out_matrix%val).and.associated(in_matrix%val)))
       end if

       if(associated(out_matrix%val)) then
          out_matrix%val=in_matrix%val
       else if (associated(out_matrix%ival)) then
          out_matrix%ival=in_matrix%ival
       else
          FLAbort("Attempting to set value in a matrix with no value space.")
       end if
    else
       ewrite(-1,*) 'Warning, not same sparsity'
       !Code for different sparsity, we assume that in_matrix%sparsity
       !is contained in out_matrix%sparsity
#ifdef DDEBUG
       matrix_same_shape=size(out_matrix,1)==size(in_matrix,1)
       assert(matrix_same_shape)
       matrix_same_shape=size(out_matrix,2)==size(in_matrix,2)
       assert(matrix_same_shape)
#endif

       if (associated(out_matrix%ival)) then
          assert(associated(in_matrix%ival))
       else
          assert((associated(out_matrix%val).and.associated(in_matrix%val)))
       end if

       call zero(out_matrix)

       do row = 1, size(out_matrix,1)
          cols => row_m_ptr(in_matrix, row)
          vals => row_val_ptr(in_matrix, row)
          do i = 1, size(cols)
             call set(out_matrix,row,cols(i),vals(i))
          end do
       end do
    end if
  end subroutine csr_csr_set

  subroutine csr_block_csr_set(out_matrix, in_matrix, blocki, blockj)
    !!< Set out_matrix to (blocki, blockj) of in_matrix. This will
    !!< only work if the matrices have the same sparsity.
    type(csr_matrix), intent(inout) :: out_matrix
    type(block_csr_matrix), intent(in) :: in_matrix
    integer, intent(in) :: blocki, blockj

    !
    integer :: row, i
    integer, dimension(:), pointer:: cols
    real, dimension(:), pointer :: vals
#ifdef DDEBUG
    logical :: matrix_same_shape
#endif

    if(in_matrix%sparsity%refcount%id==out_matrix%sparsity%refcount%id) then
       !Code for the same sparsity
       if (associated(out_matrix%ival)) then
          assert(associated(in_matrix%ival))
       else
          assert((associated(out_matrix%val).and.associated(in_matrix%val)))
       end if

       if(associated(out_matrix%val)) then
          assert(associated(in_matrix%val(blocki,blockj)%ptr))
          out_matrix%val=in_matrix%val(blocki,blockj)%ptr
       else if (associated(out_matrix%ival)) then
          assert(associated(in_matrix%ival(blocki,blockj)%ptr))
          out_matrix%ival=in_matrix%ival(blocki,blockj)%ptr
       else
          FLAbort("Attempting to set value in a matrix with no value space.")
       end if
    else
       ewrite(-1,*) 'Warning, not same sparsity'
       !Code for different sparsity, we assume that in_matrix%sparsity
       !is contained in out_matrix%sparsity
#ifdef DDEBUG
       matrix_same_shape=size(out_matrix,1)==size(in_matrix,1)
       assert(matrix_same_shape)
       matrix_same_shape=size(out_matrix,2)==size(in_matrix,2)
       assert(matrix_same_shape)
#endif

       if (associated(out_matrix%ival)) then
          assert(associated(in_matrix%ival))
       else
          assert((associated(out_matrix%val).and.associated(in_matrix%val)))
       end if

       call zero(out_matrix)

       do row = 1, size(out_matrix,1)
          cols => row_m_ptr(in_matrix, row)
          vals => row_val_ptr(in_matrix, blocki, blockj, row)
          do i = 1, size(cols)
             call set(out_matrix,row,cols(i),vals(i))
          end do
       end do
    end if

  end subroutine csr_block_csr_set

  subroutine csr_vset(matrix, i, j, val)
    !!< Set val to matrix(i,j)
    type(csr_matrix), intent(inout) :: matrix
    integer, dimension(:), intent(in) :: i, j
    real, dimension(size(i),size(j)), intent(in) :: val

    integer :: iloc, jloc

    do iloc=1,size(i)
       do jloc=1,size(j)
          call set(matrix, i(iloc), j(jloc), val(iloc,jloc))
       end do
    end do

  end subroutine csr_vset

  subroutine csr_rset(matrix, i, j, val)
    !!< Set val to matrix(i,j)
    type(csr_matrix), intent(inout) :: matrix
    integer, dimension(:), intent(in) :: j
    integer, intent(in) :: i
    real, dimension(size(j)), intent(in) :: val

    integer :: jloc

    do jloc=1,size(j)
       call set(matrix, i, j(jloc), val(jloc))
    end do

  end subroutine csr_rset

  subroutine csr_iset(matrix, i, j, val, save_pos)
    !!< Add val to matrix(i,j)
    type(csr_matrix), intent(inout) :: matrix
    integer, intent(in) :: i,j
    integer, intent(in) :: val
    integer, intent(inout), optional :: save_pos

    integer :: mpos

    mpos = pos(matrix,i,j,save_pos=save_pos)
    !In debugging mode, check that the entry actually exists.
    assert(mpos>0)

    if (associated(matrix%val)) then
       matrix%val(mpos)=val
    else if (associated(matrix%ival)) then
       matrix%ival(mpos)=val
    else
       FLAbort("Attempting to set value in a matrix with no value space.")
    end if

  end subroutine csr_iset

  subroutine block_csr_vset(matrix, blocki, blockj, i, j, val)
    !!< Add val to matrix(i,j)
    type(block_csr_matrix), intent(inout) :: matrix
    integer, dimension(:), intent(in) :: i, j
    integer, intent(in) :: blocki, blockj
    real, dimension(size(i),size(j)), intent(in) :: val

    integer :: iloc, jloc

    if(matrix%diagonal.and.(blocki/=blockj)) then
      FLAbort("Attempting to set value in an off-diagonal block of a diagonal block_csr_matrix.")
    end if

    do iloc=1,size(i)
       do jloc=1,size(j)
          call block_csr_set(matrix, blocki, blockj, i(iloc), j(jloc), val(iloc,jloc))
       end do
    end do

  end subroutine block_csr_vset

  subroutine block_csr_set(matrix, blocki, blockj, i, j, val, save_pos)
    !!< Add val to matrix(i,j)
    type(block_csr_matrix), intent(inout) :: matrix
    integer, intent(in) :: blocki,blockj,i,j
    real, intent(in) :: val
    integer, intent(inout), optional :: save_pos

    integer :: mpos

    if(matrix%diagonal.and.(blocki/=blockj)) then
      FLAbort("Attempting to set value in an off-diagonal block of a diagonal block_csr_matrix.")
    end if

    mpos = pos(matrix,i,j,save_pos=save_pos)

    !In debugging mode, check that the entry actually exists.
    assert(mpos>0)
!    assert(blocki<=matrix%blocks(1).and.blockj<=matrix%blocks(j))

    if (associated(matrix%val)) then
       matrix%val(blocki,blockj)%ptr(mpos)=val
    else if (associated(matrix%ival)) then
       matrix%ival(blocki,blockj)%ptr(mpos)=val
    else
       FLAbort("Attempting to set value in a matrix with no value space.")
    end if

  end subroutine block_csr_set

  subroutine csr_set_diag(matrix, i, val, save_pos)
    !!< Set val to matrix(i,j)
    type(csr_matrix), intent(inout) :: matrix
    integer, intent(in) :: i
    real, intent(in) :: val
    integer, intent(inout), optional :: save_pos

    integer :: mpos

    if(associated(matrix%sparsity%centrm)) then
      mpos = matrix%sparsity%centrm(i)
    else
      mpos = pos(matrix,i,i,save_pos=save_pos)
    end if

    !In debugging mode, check that the entry actually exists.
    assert(mpos>0)

    if (associated(matrix%val)) then
       matrix%val(mpos)=val
    else if (associated(matrix%ival)) then
       matrix%ival(mpos)=val
    else
       FLAbort("Attempting to set value in a matrix with no value space.")
    end if

  end subroutine csr_set_diag

  function csr_val(matrix, i, j, save_pos) result(val)
    !!< Return the value at  matrix(i,j)
    real :: val
    type(csr_matrix), intent(in) :: matrix
    integer, intent(in) :: i,j
    integer, intent(inout), optional :: save_pos

    integer :: mpos

    mpos=pos(matrix,i,j,save_pos=save_pos)

    if (associated(matrix%val)) then
       if (mpos/=0) then
          val=matrix%val(mpos)
       else
          ! i,j not in nonzero part of matrix.
          val=0
       end if
    else if (associated(matrix%ival)) then
       if (mpos/=0) then
          val=matrix%ival(mpos)
       else
          ! i,j not in nonzero part of matrix.
          val=0
       end if

    else
       FLAbort("Attempting to extract value in a matrix with no value space.")
    end if

  end function csr_val

  function csr_ival(matrix, i, j, save_pos) result(val)
    !!< Return the value at  matrix(i,j)
    integer :: val
    type(csr_matrix), intent(in) :: matrix
    integer, intent(in) :: i,j
    integer, intent(inout), optional :: save_pos

    integer :: mpos

    mpos=pos(matrix,i,j,save_pos=save_pos)

    if (associated(matrix%val)) then
       if (mpos/=0) then
          val=matrix%val(mpos)
       else
          ! i,j not in nonzero part of matrix.
          val=0
       end if

    else if (associated(matrix%ival)) then
       if (mpos/=0) then
          val=matrix%ival(mpos)
       else
          ! i,j not in nonzero part of matrix.
          val=0
       end if

    else
       FLAbort("Attempting to extract value in a matrix with no value space.")
    end if

  end function csr_ival

  function block_csr_val(matrix, blocki, blockj, i, j, save_pos) result(val)
    !!< Return the value at  matrix(i,j)
    real :: val
    type(block_csr_matrix), intent(in) :: matrix
    integer, intent(in) :: blocki, blockj, i,j
    integer, intent(inout), optional :: save_pos

    integer :: mpos

    if(matrix%diagonal.and.(blocki/=blockj)) then
      FLAbort("Attempting to retrieve value in an off-diagonal block of a diagonal block_csr_matrix.")
    end if

    mpos=pos(matrix,i,j,save_pos=save_pos)

    if (associated(matrix%val)) then
       if (mpos/=0) then
          val=matrix%val(blocki,blockj)%ptr(mpos)
       else
          ! i,j not in nonzero part of matrix.
          val=0
       end if
    else if (associated(matrix%ival)) then
       if (mpos/=0) then
          val=matrix%ival(blocki,blockj)%ptr(mpos)
       else
          ! i,j not in nonzero part of matrix.
          val=0
       end if

    else
       FLAbort("Attempting to extract value in a matrix with no value space.")
    end if

  end function block_csr_val

  function dcsr_val(matrix, i, j) result (val)
    !!< Return the value at  matrix(i,j)
    real :: val
    type(dynamic_csr_matrix), intent(in) :: matrix
    integer, intent(in) :: i,j

    integer :: mpos

    mpos=pos(matrix,i,j)

    if (mpos/=0) then
       val=matrix%val(i)%ptr(mpos)
    else
       ! i,j not in nonzero part of matrix.
       val=0
    end if

  end function dcsr_val

  function csr_dense_i(matrix)
    !!< Return the dense form of matrix. WARNING! THIS CAN EASILY BE HUGE!!!
    type(csr_matrix), intent(in) :: matrix
    integer, dimension(size(matrix,1), size(matrix,2)) :: csr_dense_i

    integer :: i

    csr_dense_i=0

    do i=1,size(matrix,1)
       csr_dense_i(i,row_m_ptr(matrix,i))=csr_row_ival_ptr(matrix,i)
    end do

  end function csr_dense_i

  function csr_dense(matrix) result(dense)
    !!< Return the dense form of matrix. WARNING! THIS CAN EASILY BE HUGE!!!
    type(csr_matrix), intent(in) :: matrix
    real, dimension(size(matrix,1), size(matrix,2)) :: dense

    integer :: i

    dense=0.0

    do i=1,size(matrix,1)
       dense(i,row_m_ptr(matrix,i))=row_val_ptr(matrix,i)
    end do

  end function csr_dense

  function block_csr_dense(matrix) result(dense)
    !!< Return the dense form of matrix. WARNING! THIS CAN EASILY BE HUGE!!!
    type(block_csr_matrix), intent(in) :: matrix
    real, dimension(size(matrix,1), size(matrix,2)) :: dense

    integer :: i, blocki, blockj, blockstarti, blockstartj

    dense=0.0

    do blocki=1,matrix%blocks(1)

       blockstarti=(blocki-1)*block_size(matrix,1)

       do blockj=1,matrix%blocks(2)

          if(matrix%diagonal.and.(blocki/=blockj)) cycle

          blockstartj=(blockj-1)*block_size(matrix,2)

          do i=1,block_size(matrix,1)

             dense(blockstarti+i,blockstartj+row_m_ptr(matrix,i))&
                  =row_val_ptr(matrix,blocki, blockj, i)
          end do

       end do
    end do

  end function block_csr_dense

  function dcsr_dense(matrix)
    !!< Return the dense form of matrix. WARNING! THIS CAN EASILY BE HUGE!!!
    type(dynamic_csr_matrix), intent(in) :: matrix
    real, dimension(size(matrix,1), size(matrix,2)) :: dcsr_dense

    integer :: i

    dcsr_dense=0

    do i=1,size(matrix,1)
       dcsr_dense(i,matrix%colm(i)%ptr)=matrix%val(i)%ptr
    end do

  end function dcsr_dense

  function block_dcsr_dense(matrix) result (dense)
    !!< Return the dense form of matrix. WARNING! THIS CAN EASILY BE HUGE!!!
    type(block_dynamic_csr_matrix), intent(in) :: matrix
    integer, dimension(size(matrix,1), size(matrix,2)) :: dense

    integer :: i,j
    integer, dimension(blocks(matrix,1)+1) :: sizes_i
    integer, dimension(blocks(matrix,2)+1) :: sizes_j

    sizes_i(1)=0
    do i=2,blocks(matrix,1)+1
       sizes_i(i)=sizes_i(i-1)+block_size(matrix, i-1, 1, dim=1)
    end do

    sizes_j(1)=0
    do j=2,blocks(matrix,2)+1
       sizes_j(j)=sizes_j(j-1)+block_size(matrix, 1, j-1, dim=2)
    end do

    do i=1, blocks(matrix,1)
       do j=1,blocks(matrix,2)

          dense(sizes_i(i)+1:sizes_i(i+1),sizes_j(j)+1:sizes_j(j+1))=&
               dcsr_dense(matrix%blocks(i,j))

       end do
    end do
    ! call reset_debug_level() - This shouldn't be in the trunk!

  end function block_dcsr_dense

  function dcsr2csr(dcsr) result (csr)
    !!< Given a dcsr matrix return a csr matrix. The dcsr matrix is left
    !!< untouched.
    type(dynamic_csr_matrix), intent(in) :: dcsr
    type(csr_matrix) :: csr

    integer :: rows, columns, nentries, i, rowptr, rowlen
    integer, dimension(1) :: rowpos
    type(csr_sparsity) :: sparsity

    rows=size(dcsr,1)
    columns=size(dcsr,2)
    nentries=entries(dcsr)

    if (len_trim(dcsr%name)==0) then
       call allocate(sparsity, rows, columns, nentries, name="dcsr2csrSparsity")
    else
      call allocate(sparsity, rows, columns, nentries, &
        name=trim(dcsr%name)//'Sparsity')
    end if
    call allocate(csr, sparsity, name=dcsr%name)
    ! Drop the excess reference
    call deallocate(sparsity)

    rowptr=1
    do i=1,rows
       csr%sparsity%findrm(i)=rowptr

       rowlen=size(dcsr%colm(i)%ptr)

       csr%sparsity%colm(rowptr:rowptr+rowlen-1)=dcsr%colm(i)%ptr

       csr%val(rowptr:rowptr+rowlen-1)=dcsr%val(i)%ptr

       if (any(dcsr%colm(i)%ptr==i)) then

          rowpos=minloc(dcsr%colm(i)%ptr,mask=dcsr%colm(i)%ptr==i)

          csr%sparsity%centrm(i)=rowptr+rowpos(1)-1

       else
!          ewrite(1,*) "Missing diagonal element"

          csr%sparsity%centrm(i)=0
       end if

       rowptr = rowptr + rowlen

    end do

    csr%sparsity%findrm(rows+1) = rowptr

    ! dcsr rows are sorted (by construction in dcsr_rowpos_add)
    csr%sparsity%sorted_rows=.true.

  end function dcsr2csr

  function csr2dcsr(csr) result (dcsr)
    !!< Given a csr matrix return a dcsr matrix. The csr matrix is left
    !!< untouched.
    type(csr_matrix), intent(in) :: csr
    type(dynamic_csr_matrix) :: dcsr

    integer i, j, rows, columns

    rows=size(csr,1)
    columns=size(csr,2)

    call allocate(dcsr, rows, columns)

    do i=1,rows

      do j=csr%sparsity%findrm(i), csr%sparsity%findrm(i+1)-1

        call set(dcsr, i, csr%sparsity%colm(j), csr%val(j))

      end do

    end do

  end function csr2dcsr

  subroutine csr_mult(vector_out,mat,vector_in)
    !!< Multiply a csr_matrix by a vector,
    !!< result is written to vector_out

    !interface variables
    real, dimension(:), intent(in) :: vector_in
    type(csr_matrix), intent(in) :: mat
    real, dimension(:), intent(out) :: vector_out

    !local variables
    integer :: i, j

    assert(size(vector_in)==size(mat,2))
    assert(size(vector_out)==size(mat,1))

    do i = 1, size(vector_out)
      vector_out(i) = 0.0
      do j=mat%sparsity%findrm(i), mat%sparsity%findrm(i+1)-1
         vector_out(i) = vector_out(i) + mat%val(j) * vector_in(mat%sparsity%colm(j))
      end do
    end do

  end subroutine csr_mult

  subroutine csr_mult_addto(vector_out,mat,vector_in)
    !!< Multiply a csr_matrix by a vector,
    !!< result is added vector_out

    !interface variables
    real, dimension(:), intent(in) :: vector_in
    type(csr_matrix), intent(in) :: mat
    real, dimension(:), intent(inout) :: vector_out

    !local variables
    integer :: i, j

    assert(size(vector_in)==size(mat,2))
    assert(size(vector_out)==size(mat,1))

    do i = 1, size(vector_out)
      do j=mat%sparsity%findrm(i), mat%sparsity%findrm(i+1)-1
         vector_out(i) = vector_out(i) + mat%val(j) * vector_in(mat%sparsity%colm(j))
      end do
    end do

  end subroutine csr_mult_addto

  subroutine dcsr_mult(m,v,mv)
    type(dynamic_csr_matrix), intent(in) :: m
    real, dimension(:), intent(in) :: v
    real, dimension(:), intent(out) :: mv

    !locals
    integer :: i

    if(size(m,1).ne.size(mv)) FLAbort('Bad vector size out.')
    if(size(m,2).ne.size(v)) FLAbort('Bad vector size in.')

    mv = 0.

    do i = 1, size(mv)
       mv(i) = sum(v(row_m(m,i)) * row_val_ptr(m,i))
    end do

  end subroutine dcsr_mult


  subroutine csr_mult_T(vector_out,mat,vector_in)
    !!< Multiply the transpose of a csr_matrix by a vector,
    !!< result is written to vector_out

    !interface variables
    real, dimension(:), intent(in) :: vector_in
    type(csr_matrix), intent(in) :: mat
    real, dimension(:), intent(out) :: vector_out

    !local variables
    integer :: i, j, k

    ewrite(2,*) 'size(vector_in) = ', size(vector_in)
    ewrite(2,*) 'size(mat,1) = ', size(mat,1)
    assert(size(vector_in)==size(mat,1))
    ewrite(2,*) 'size(vector_out) = ', size(vector_out)
    ewrite(2,*) 'size(mat,2) = ', size(mat,2)
    assert(size(vector_out)==size(mat,2))

    vector_out=0
    do i = 1, size(vector_in)
      do j=mat%sparsity%findrm(i), mat%sparsity%findrm(i+1)-1
         k = mat%sparsity%colm(j)
         vector_out(k) = vector_out(k) + mat%val(j) * vector_in(i)
      end do
    end do

  end subroutine csr_mult_T

  subroutine csr_mult_T_addto(vector_out,mat,vector_in)
    !!< Multiply the transpose of a csr_matrix by a vector,
    !!< result is added to vector_out

    !interface variables
    real, dimension(:), intent(in) :: vector_in
    type(csr_matrix), intent(in) :: mat
    real, dimension(:), intent(out) :: vector_out

    !local variables
    integer :: i, j, k

    ewrite(2,*) 'size(vector_in) = ', size(vector_in)
    ewrite(2,*) 'size(mat,1) = ', size(mat,1)
    assert(size(vector_in)==size(mat,1))
    ewrite(2,*) 'size(vector_out) = ', size(vector_out)
    ewrite(2,*) 'size(mat,2) = ', size(mat,2)
    assert(size(vector_out)==size(mat,2))

    do i = 1, size(vector_in)
      do j=mat%sparsity%findrm(i), mat%sparsity%findrm(i+1)-1
         k = mat%sparsity%colm(j)
         vector_out(k) = vector_out(k) + mat%val(j) * vector_in(i)
      end do
    end do

  end subroutine csr_mult_T_addto

  subroutine dcsr_mult_T(m,v,mv)
    type(dynamic_csr_matrix), intent(in) :: m
    real, dimension(:), intent(in) :: v
    real, dimension(:), intent(out) :: mv

    !locals
    integer :: i

    if(size(m,2).ne.size(mv)) FLAbort('Bad vector size out.')
    if(size(m,1).ne.size(v)) FLAbort('Bad vector size in.')

    mv = 0.

    do i = 1, size(v)
       mv(row_m(m,i)) = mv(row_m(m,i)) &
            + v(i) * row_val_ptr(m,i)
    end do

  end subroutine dcsr_mult_T

  function dcsr_matmul_T(matrix1, matrix2, model,check) result (product)
    !!< Perform the matrix multiplication:
    !!<
    !!<     matrix1*matrix2^T
    !!<
    type(dynamic_csr_matrix), intent(in) :: matrix1, matrix2
    type(dynamic_csr_matrix) :: product
    type(dynamic_csr_matrix), intent(in), optional :: model
    logical, intent(in), optional :: check

    type(integer_vector), dimension(:), allocatable :: hitlist
    integer, dimension(:), allocatable :: size_hitlist

    integer, dimension(:), pointer :: row, col
    integer :: i,j,k1,k2,jrow,jcol
    real :: entry
    logical :: addflag
    logical :: l_check
    real , allocatable, dimension(:) :: vec, m2Tvec, m1m2tvec, productvec

    l_check = .false.
    if(present(check)) then
       L_check = check
    end if

    assert(size(matrix1,2)==size(matrix2,2))

    call allocate(product, size(matrix1,1), size(matrix2,1))
    call zero(product)

    ewrite(2,*) 'Measuring structure'

    allocate( hitlist(size(matrix2,2)), size_hitlist(size(matrix2,2)) )

    if(.not.present(model)) then
       size_hitlist = 0

       !count the number of rows of matrix2 which contain an element
       !in the column
       do j = 1, size(matrix2,1)
          col=>matrix2%colm(j)%ptr
          if(size(col)>0) then
             do k1 = 1, size(col)
                size_hitlist(col(k1)) = size_hitlist(col(k1)) + 1
             end do
          end if
       end do

       do j = 1, size(matrix1,2)
          allocate( hitlist(j)%ptr(size_hitlist(j)) )
       end do

       size_hitlist = 1

       !make a list of rows of matrix2 which contain element i in the column
       do j = 1, size(matrix2,1)
          col=>matrix2%colm(j)%ptr
          if(size(col)>0) then
             do k1 = 1, size(col)
                hitlist(col(k1))%ptr(size_hitlist(col(k1))) = j
                size_hitlist(col(k1)) = size_hitlist(col(k1)) + 1
             end do
          end if
       end do

       deallocate( size_hitlist )

       do i=1, size(matrix1,1)
          row=>matrix1%colm(i)%ptr
          if(size(row)>0) then
             do jrow = 1,size(row)
                !need to visit all points hit by i
                do jcol = 1,size(hitlist(row(jrow))%ptr)
                   j = hitlist(row(jrow))%ptr(jcol)
                   assert(j.ne.0)
                   col=>matrix2%colm(j)%ptr

                   if(size(col)>0) then
                      entry=0.0

                      addflag = .false.

                      k1 = 1
                      k2 = 1
                      do
                         if(addflag) exit
                         if((k1.gt.size(row)).or.(k2.gt.size(col))) exit
                         if(row(k1)<col(k2)) then
                            k1 = k1 + 1
                         else
                            if(row(k1)==col(k2)) then
                               ! Note the transpose in the second val call.
                               entry=entry+val(matrix1,i,row(k1))* &
                                    val(matrix2,j,row(k1))
                               addflag = .true.
                               k1 = k1 + 1
                               k2 = k2 + 1
                            else
                               k2 = k2 + 1
                            end if
                         end if
                      end do
                      if(addflag) then
                         call addto(product,i,j,0.0)
                      end if
                   end if
                end do
             end do
          end if
       end do

       deallocate( hitlist )

    else

       do i = 1, size(matrix1,1)
          allocate(product%colm(i)%ptr(size(model%colm(i)%ptr)))
          allocate(product%val(i)%ptr(size(model%colm(i)%ptr)))
          product%colm(i)%ptr = model%colm(i)%ptr
          product%val(i)%ptr = 0.0
       end do

    endif

    do i=1, size(matrix1,1)
       row=>matrix1%colm(i)%ptr

       if(size(row)>0) then
          do jcol = 1, size(product%colm(i)%ptr)
             j = product%colm(i)%ptr(jcol)
             col=>matrix2%colm(j)%ptr

             if(size(col)>0) then
                entry=0.0

                addflag = .false.

                k1 = 1
                k2 = 1
                do
                   if((k1.gt.size(row)).or.(k2.gt.size(col))) exit
                   if(row(k1)<col(k2)) then
                      k1 = k1 + 1
                   else
                      if(row(k1)==col(k2)) then
                         ! Note the transpose in the second val call.
                         entry=entry+val(matrix1,i,row(k1))* &
                              val(matrix2,j,row(k1))
                         addflag = .true.
                         k1 = k1 + 1
                         k2 = k2 + 1
                      else
                         k2 = k2 + 1
                      end if
                   end if
                end do
                if(addflag) then
                   call addto(product,i,j,entry)
                end if
             end if
          end do
       end if
    end do

    if(l_check) then
       allocate( vec(size(matrix2,1)) )
       allocate( m2tvec(size(matrix2,2)) )
       if(size(matrix1,2).ne.size(matrix2,2)) then
          FLAbort('Cannot perform multiplication when matrix sizes differ.')
       end if
       allocate( m1m2tvec(size(matrix1,1)) )
       allocate( productvec(size(matrix1,1)) )

       call random_number(vec)
       call dcsr_mult_T(matrix2,vec,m2tvec)
       call dcsr_mult(matrix1,m2tvec,m1m2tvec)
       call dcsr_mult(product,vec,productvec)

       if(any(abs(productvec-m1m2tvec)>abs(productvec)*1.0e-8)) then

          ewrite(2,*) maxval(abs(productvec-m1m2tvec))

          call dcsr_matrix2file('matrix1',matrix1)
          call dcsr_matrix2file('matrix2',matrix2)
          call dcsr_matrix2file('product',product)
          ewrite(2,*) size(matrix1,1), size(matrix1,2)
          ewrite(2,*) size(matrix2,1), size(matrix2,2)
          ewrite(2,*) size(product,1), size(product,2)
          FLAbort('Matmul_t error.')
       end if
    end if

  end function dcsr_matmul_T

  function csr_matmul_T(matrix1, matrix2, model) result (product)
    !!< Perform the matrix multiplication:
    !!<
    !!<     matrix1*matrix2^T
    !!<
    !!< Only works on csr matrices with monotonic row entries in colm
    type(csr_matrix), intent(in) :: matrix1, matrix2
    type(csr_sparsity), intent(in), optional :: model
    type(csr_matrix) :: product
    type(dynamic_csr_matrix) :: product_d

    type(integer_vector), dimension(:), allocatable :: hitlist
    integer, dimension(:), allocatable :: size_hitlist

    integer, dimension(:), pointer :: row, col
    integer :: i,j,k1,k2

    ewrite(1,*) 'Entering csr_matmul_T'

    assert(size(matrix1,2)==size(matrix2,2))
    if(.not.matrix1%sparsity%sorted_rows.or..not.matrix2%sparsity%sorted_rows) then
      FLAbort("csr_matmul_T assumes sorted rows.")
    end if

    if(.not.present(model)) then
       ewrite(2,*) 'Measuring structure'

       allocate( hitlist(size(matrix2,2)), size_hitlist(size(matrix2,2)) )
       size_hitlist = 0

       !count the number of rows of matrix2 which contain an element
       !in the column
       do j = 1, size(matrix2,1)
          col=>row_m_ptr(matrix2,j)
          do k1 = 1, size(col)
             size_hitlist(col(k1)) = size_hitlist(col(k1)) + 1
          end do
       end do

       do j = 1, size(matrix1,2)
          allocate( hitlist(j)%ptr(size_hitlist(j)) )
       end do

       size_hitlist = 1

       !make a list of rows of matrix2 which contain element i in the column
       do j = 1, size(matrix2,1)
          col=>row_m_ptr(matrix2,j)
          if(size(col)>0) then
             do k1 = 1, size(col)
                hitlist(col(k1))%ptr(size_hitlist(col(k1))) = j
                size_hitlist(col(k1)) = size_hitlist(col(k1)) + 1
             end do
          end if
       end do

       deallocate( size_hitlist )

       call allocate(product_d, size(matrix1,1), size(matrix2,1))
       do i=1, size(matrix1,1)
          row=>row_m_ptr(matrix1, i)
          do k1 = 1,size(row)
             do k2 = 1,size(hitlist(row(k1))%ptr)
                call addto(product_d,i,hitlist(row(k1))%ptr(k2),0.0)
             end do
          end do
       end do

       deallocate( hitlist )

       product = dcsr2csr(product_d)
       call deallocate( product_d)
    else
       call allocate(product, model)
    end if

    product%name="matmul_T"//trim(matrix1%name)//"*"//trim(matrix2%name)

    call csr_matmul_t_preallocated&
         (matrix1, matrix2, product = product, set_sparsity = .not. present(model))

  end function csr_matmul_T

  subroutine csr_matmul_t_preallocated(matrix1, matrix2, product, set_sparsity)
    !!< Perform the matrix multiplication:
    !!<
    !!<     matrix1*matrix2^T
    !!<
    !!< Only works on csr matrices with monotonic row entries in colm. Returns
    !!< the result in the pre-allocated csr matrix product.

    type(csr_matrix), intent(in) :: matrix1
    type(csr_matrix), intent(in) :: matrix2
    type(csr_matrix), intent(inout) :: product
    !! If present and .true., set the product sparsity as well as performing
    !! the product
    logical, optional, intent(in) :: set_sparsity

    integer, dimension(:), pointer :: row, col, row_product
    real, dimension(:), pointer :: row_val, col_val
    integer :: i,j,k1,k2,jcol
    real :: entry0
    integer :: nentry0
    logical :: addflag, lset_sparsity

    lset_sparsity = present_and_true(set_sparsity)

    ewrite(2,*) 'adding in data'

    assert(size(matrix1,2)==size(matrix2,2))
    if(.not.matrix1%sparsity%sorted_rows.or..not.matrix2%sparsity%sorted_rows) then
      FLAbort("csr_matmul_T assumes sorted rows.")
    end if

    call zero(product)

    nentry0 = 0

    do i=1, size(matrix1,1)
       row=>row_m_ptr(matrix1, i)
       row_val=>row_val_ptr(matrix1, i)

       if(size(row)>0) then
          row_product=>row_m_ptr(product, i)
          do jcol = 1, size(row_product)
             j = row_product(jcol)

             col=>row_m_ptr(matrix2, j)
             col_val=>row_val_ptr(matrix2, j)

             if(size(col)>0) then
                entry0=0.0

                addflag = .false.

                k1 = 1
                k2 = 1
                do
                   if((k1.gt.size(row)).or.(k2.gt.size(col))) exit
                   if(row(k1)<col(k2)) then
                      k1 = k1 + 1
                   else
                      if(row(k1)==col(k2)) then
                         ! Note the transpose in the second val call.
                         entry0=entry0+row_val(k1)* &
                              col_val(k2)
                         addflag = .true.
                         k1 = k1 + 1
                         k2 = k2 + 1
                      else
                         k2 = k2 + 1
                      end if
                   end if
                end do
                if(addflag) then
                   nentry0 = nentry0 + 1
                   product%val(nentry0) = entry0
                   if(lset_sparsity) then
                      product%sparsity%colm(nentry0) = j
                      if(i==j) product%sparsity%centrm(i) = nentry0
                   end if
                end if
             end if
          end do
       end if
    end do

  end subroutine csr_matmul_t_preallocated

  function csr_sparsity_matmul(A, B) result (C)
    !!< Computes the sparsity of the matrix product:
    !!<
    !!<     C_ij = \sum_j A_ik * B_kj
    !!<
    type(csr_sparsity), intent(in) :: A, B
    type(csr_sparsity) :: C

    type(integer_set):: iset
    integer, dimension(:), allocatable:: nnz
    integer, dimension(:), pointer:: rowA_i, rowB_k, rowC_i
    integer:: i, k

    ! work out number of nonzeros per row of C
    allocate(nnz(size(A,1)))
    do i=1, size(A, 1)
     rowA_i => row_m_ptr(A, i)
     call allocate(iset)
     do k=1, size(rowA_i)
       rowB_k => row_m_ptr(B, rowA_i(k))
       call insert(iset, rowB_k)
     end do
     nnz(i)=key_count(iset)
     call deallocate(iset)

    end do

    ! the sparsity for C
    call allocate(C, size(A,1), size(B,2), nnz=nnz, &
      name="matmul_"//trim(A%name)//"*"//trim(B%name))

    ! same thing, now actually filling in the column indices in the rows of C
    do i=1, size(A, 1)
     rowA_i => row_m_ptr(A, i)
     rowC_i => row_m_ptr(C, i)
     call allocate(iset)
     do k=1, size(rowA_i)
       rowB_k => row_m_ptr(B, rowA_i(k))
       call insert(iset, rowB_k)
     end do
     assert(key_count(iset)==size(rowC_i))
     rowC_i=set2vector(iset)
     call deallocate(iset)
    end do

  end function csr_sparsity_matmul

  function csr_matmul(A, B, model) result (C)
    !!< Perform the matrix multiplication:
    !!<
    !!<     C_ij = \sum_j A_ik * B_kj
    !!<
    type(csr_matrix), intent(in) :: A, B
    type(csr_sparsity), intent(in), optional :: model
    type(csr_matrix) :: C

    type(csr_sparsity):: sparsity

    ewrite(1,*) 'Entering csr_matmul'

    assert(size(A,2)==size(B,1))

    if(.not.present(model)) then
       sparsity = csr_sparsity_matmul(A%sparsity, B%sparsity)
       call allocate(C, sparsity)
    else
       call allocate(C, model)
    end if

    C%name="matmul_"//trim(A%name)//"*"//trim(B%name)

    call csr_matmul_preallocated(A, B, product = C)

  end function csr_matmul

  subroutine csr_matmul_preallocated(A, B, product)
    !!< Perform the matrix multiplication:
    !!<
    !!<     A*B
    !!<
    !!< Returns the result in the pre-allocated csr matrix product.

    type(csr_matrix), intent(in) :: A
    type(csr_matrix), intent(in) :: B
    ! we use intent(in) here as only the value space gets changed
    ! this allows eg. using block() as input
    type(csr_matrix), intent(inout) :: product

    ewrite(1,*) 'Entering csr_matmul_preallocated'

    call zero(product)
    call matmul_addto(A, B, product=product)

  end subroutine csr_matmul_preallocated

  subroutine csr_matmul_addto(A, B, product)
    !!< Perform the matrix multiplication:
    !!<
    !!<     C=C+A*B
    !!<
    !!< Returns the result in the pre-allocated csr matrix product.

    type(csr_matrix), intent(in) :: A
    type(csr_matrix), intent(in) :: B
    type(csr_matrix), intent(inout) :: product

    real, dimension(:), pointer:: A_i, B_k
    integer, dimension(:), pointer:: rowA_i, rowB_k
    integer:: i, j, k

    ewrite(1,*) 'Entering csr_matmul_preallocated_addto'

    assert(size(A,2)==size(B,1))
    assert(size(product,1)==size(A,1))
    assert(size(product,2)==size(B,2))

    ! perform C_ij=\sum_k A_ik B_kj

    do i=1, size(A, 1)
     A_i => row_val_ptr(A, i)
     rowA_i => row_m_ptr(A, i)
     do k=1, size(rowA_i)
       B_k => row_val_ptr(B, rowA_i(k))
       rowB_k => row_m_ptr(B, rowA_i(k))
       do j=1, size(rowB_k)
         call addto(product, i, rowB_k(j), A_i(k)*B_k(j))
       end do
     end do
    end do

  end subroutine csr_matmul_addto

  function block_csr_matmul(A, B, model) result (C)
    !!< Perform the matrix multiplication:
    !!<
    !!<     C_ij = \sum_j A_ik * B_kj
    !!<
    type(block_csr_matrix), intent(in) :: A, B
    type(csr_sparsity), intent(in), optional :: model
    type(block_csr_matrix) :: C

    type(csr_sparsity):: sparsity

    ewrite(1,*) 'Entering csr_matmul'

    assert(size(A,2)==size(B,1))
    assert(blocks(A,2)==blocks(B,1))

    if(.not.present(model)) then
       sparsity = csr_sparsity_matmul(A%sparsity, B%sparsity)
       call allocate(C, sparsity, blocks=(/ blocks(A,1), blocks(B,2) /))
    else
       call allocate(C, model, blocks=(/ blocks(A,1), blocks(B,2) /))
    end if

    C%name="matmul_"//trim(A%name)//"*"//trim(B%name)

    call block_csr_matmul_preallocated(A, B, product = C)

  end function block_csr_matmul

  subroutine block_csr_matmul_preallocated(A, B, product)
    !!< Perform the matrix multiplication:
    !!<
    !!<     A*B
    !!<
    !!< Returns the result in the pre-allocated csr matrix product.

    type(block_csr_matrix), intent(in) :: A
    type(block_csr_matrix), intent(in) :: B
    type(block_csr_matrix), intent(inout) :: product

    ewrite(1,*) 'Entering csr_matmul_preallocated'

    call zero(product)
    call matmul_addto(A, B, product=product)

  end subroutine block_csr_matmul_preallocated

  subroutine block_csr_matmul_addto(A, B, product)
    !!< Perform the matrix multiplication:
    !!<
    !!<     C=C+A*B
    !!<
    !!< Returns the result in the pre-allocated csr matrix product.

    type(block_csr_matrix), intent(in) :: A
    type(block_csr_matrix), intent(in) :: B
    type(block_csr_matrix), intent(inout) :: product

    real, dimension(:), pointer:: A_i, B_k
    integer, dimension(:), pointer:: rowA_i, rowB_k
    integer:: blocki, blockj, blockk, i, j, k

    ewrite(1,*) 'Entering csr_matmul_preallocated_addto'

    assert(size(A,2)==size(B,1))
    assert(blocks(A,2)==blocks(B,1))
    assert(size(product,1)==size(A,1))
    assert(size(product,2)==size(B,2))
    assert(blocks(product,1)==blocks(A,1))
    assert(blocks(product,2)==blocks(B,2))

    ! perform C_ij=C_ij+\sum_k A_ik B_kj

    do blocki=1, blocks(A,1)
      do blockk=1, blocks(A,2)
        do blockj=1, blocks(B,2)


          do i=1, size(A, 1)
           A_i = row_val_ptr(A, blocki, blockk, i)
           rowA_i => row_m_ptr(A, i)
           do k=1, size(rowA_i)
             B_k => row_val_ptr(B, blockk, blockj, rowA_i(k))
             rowB_k => row_m_ptr(B, rowA_i(k))
             do j=1, size(rowB_k)
               call addto(product, i, blocki, blockj, rowB_k(j), A_i(k)*B_k(j))
             end do
           end do
          end do

        end do
      end do
    end do

  end subroutine block_csr_matmul_addto

  function csr_sparsity_transpose(sparsity) result(sparsity_T)
  !!< Provides the transpose of the given sparsity
  type(csr_sparsity), intent(in):: sparsity
  type(csr_sparsity) sparsity_T

    integer, dimension(:), allocatable:: rowlen
    integer, dimension(:), pointer:: cols
    integer i, j, row, col, count
    logical have_diag

    have_diag=associated(sparsity%centrm)

    ! just swap n/o rows and cols
    call allocate(sparsity_T, size(sparsity,2), size(sparsity,1), &
       entries=size(sparsity%colm), diag=have_diag, &
       name=trim(sparsity%name)//"Transpose")

    ! Also swap the row and column halos if present.
    if (associated(sparsity%row_halo)) then
       allocate(sparsity_T%column_halo)
       sparsity_T%column_halo=sparsity%row_halo
       call incref(sparsity_T%column_halo)
    end if
    if (associated(sparsity%column_halo)) then
       allocate(sparsity_T%row_halo)
       sparsity_T%row_halo=sparsity%column_halo
       call incref(sparsity_T%row_halo)
    end if

    ! work out row lengths of the transpose
    allocate( rowlen(1:size(sparsity_T,1)) )
    rowlen=0
    do i=1, size(sparsity%colm)
      col=sparsity%colm(i)
      if (col>0) then
        rowlen(col)=rowlen(col)+1
      end if
    end do

    ! work out sparsity_T%findrm
    count=1
    do row=1, size(sparsity_T,1)
      sparsity_T%findrm(row)=count
      count=count+rowlen(row)
    end do
    sparsity_T%findrm(row)=count

    rowlen=0 ! use rowlen again as counter
    do row=1, size(sparsity,1)
      cols => row_m_ptr(sparsity, row)
      do j=1, size(cols)
         col=cols(j)
         if (col>0) then
           sparsity_T%colm(sparsity_T%findrm(col)+rowlen(col))=row
           rowlen(col)=rowlen(col)+1
         end if
      end do
    end do
    ! note that the above procedure inserts the entries in increasing order
    sparsity_T%sorted_rows=.true.

    if (have_diag) then
      do row=1, size(sparsity_T%centrm)
        sparsity_T%centrm(row)=csr_sparsity_pos(sparsity_T, row, row)
      end do
    end if

  end function csr_sparsity_transpose

  function block_csr_transpose(block_A, symmetric_sparsity) result (block_AT)
    type(block_csr_matrix), intent(in) :: block_A
    ! If the sparsity is symmetric, don't create a new one
    logical, intent(in), optional :: symmetric_sparsity

    type(block_csr_matrix) block_AT
    type(csr_matrix) :: A, AT
    type(csr_sparsity):: sparsity
    integer :: i, j

    if (present_and_true(symmetric_sparsity)) then
      call allocate(block_AT, block_A%sparsity, (/ block_A%blocks(2), block_A%blocks(1) /), name=trim(block_A%name) // "Transpose")
    else
      sparsity=transpose(block_A%sparsity)
      call allocate(block_AT, sparsity, (/ block_A%blocks(2), block_A%blocks(1) /), name=trim(block_A%name) // "Transpose")
      call deallocate(sparsity)
    end if

    do i = 1, blocks(block_A, 1)
      do j = 1, blocks(block_A, 2)
        A = block(block_A, i, j)
        AT = transpose(A, symmetric_sparsity=symmetric_sparsity)
        call set(block_AT, j, i, AT)
        call deallocate(AT)
      end do
    end do

  end function block_csr_transpose

  function csr_transpose(A, symmetric_sparsity) result (AT)
  !!< Provides the transpose of the given matrix
    type(csr_matrix), intent(in):: A
    ! If the sparsity is symmetric, don't create a new one
    logical, intent(in), optional :: symmetric_sparsity
    type(csr_matrix) AT

    type(csr_sparsity):: sparsity
    integer, dimension(:), allocatable:: rowlen
    integer, dimension(:), pointer:: cols
    real, dimension(:), pointer:: vals
    integer row, j, col

    if (present_and_true(symmetric_sparsity) .and. .not. A%sparsity%sorted_rows) then
      FLAbort("csr_tranpose on symmetric sparsities works only with sorted_rows.")
    end if

#ifdef DDEBUG
    ! Check that the supplied sparsity is indeed symmetric
    if (present_and_true(symmetric_sparsity)) then
      if (.not. is_symmetric(A%sparsity)) then
         FLAbort("The symmetric flag is supplied, but the sparsity is not symmetric.")
      end if
    end if
#endif

    if (present_and_true(symmetric_sparsity)) then
      call allocate(AT, A%sparsity, name=trim(A%name) // "Transpose")
    else
      sparsity=transpose(A%sparsity)
      call allocate(AT, sparsity, name=trim(A%name) // "Transpose")
      call deallocate(sparsity)
    end if

    ! we use the same insertion procedure as above in csr_sparsity_transpose
    ! rowlen is used to count the number of inserted entries thus far per row
    allocate( rowlen(1:size(AT,1)) )
    rowlen=0
    do row=1, size(A,1)
      cols => row_m_ptr(A, row)
      vals => row_val_ptr(A, row)
      do j=1, size(cols)
         col=cols(j)
         if (col>0) then
           AT%val(AT%sparsity%findrm(col)+rowlen(col))=vals(j)
           ! check that this is indeed the right column:
           assert(AT%sparsity%colm(AT%sparsity%findrm(col)+rowlen(col))==row)
           rowlen(col)=rowlen(col)+1
         else if (present_and_true(symmetric_sparsity)) then
           FLAbort("Found a zero entry in the colm of the given sparsity which is currently not supported if the symmetric flag.")
         end if
      end do
    end do

  end function csr_transpose

  subroutine sparsity_sort(sparsity)
  !!< In-place sort of the rows of the given sparsity to increasing column index
  !!< Only for internal usage within sparsity creating routines. Should not be
  !!< called after any matrix has been based upon it.
  type(csr_sparsity), intent(inout):: sparsity

    integer, dimension(:), pointer:: cols
    integer i, j, col
    logical sorted

    if (associated(sparsity%refcount)) then
      if (sparsity%refcount%count>1) then
        ewrite(-1,*) "For health and safety reasons sparsities should not"
        FLAbort("be sorted after they are referenced.")
      end if
    end if

    do i=1, size(sparsity,1)
       cols => row_m_ptr(sparsity, i)
       ! hurray for the bubble sort
       do
          sorted=.true. ! let's  be optimistic
          do j=1, size(cols)-1
             if (cols(j)>cols(j+1)) then
                col=cols(j)
                cols(j)=cols(j+1)
                cols(j+1)=col
                sorted=.false. ! not quite there yet
             end if
          end do
          if (sorted) exit
       end do
    end do

    sparsity%sorted_rows=.true.

  end subroutine sparsity_sort

  function sparsity_is_symmetric(sparsity) result(symmetric)
    !!< Checks if the given sparsity is symmetric
    type(csr_sparsity), intent(in):: sparsity

    integer, dimension(:), pointer :: cols, colsT
    integer :: row, col
    logical :: symmetric

    if (.not. size(sparsity,1) == size(sparsity,2)) then
      ! The dimensions dont even match
      symmetric = .false.
      return
    end if
    symmetric = .true.
    do row=1, size(sparsity,1)
       cols => row_m_ptr(sparsity, row)
       do col=1, size(cols)
          colsT => row_m_ptr(sparsity, cols(col))
          if (.not. any(colsT==row)) then
            ! There is a nonzero entry in row X cols(col),
            ! but not at cols(col) X row
            symmetric = .false.
            return
          end if
       end do
    end do

  end function sparsity_is_symmetric

  function sparsity_is_sorted(sparsity) result(sorted)
    !!< Checks if the rows of the given sparsity is sorted to increasing column index
    type(csr_sparsity), intent(in):: sparsity

    integer, dimension(:), pointer:: cols
    integer i, j
    logical sorted

    do i=1, size(sparsity,1)
       cols => row_m_ptr(sparsity, i)
       do j=1, size(cols)-1
          if (cols(j)>cols(j+1)) then
             sorted=.false.
             return
          end if
       end do
    end do

    sorted=.true.
  end function sparsity_is_sorted

  function sparsity_merge(sparsityA, sparsityB, name) result (sparsityC)
  !!< Merges sparsityA and sparsityB such that:
  !!< all (i,j) in either sparsityA or sparsityB are in sparsityC
    type(csr_sparsity), intent(in):: sparsityA, sparsityB
    character(len=*), optional, intent(in):: name

    type(csr_sparsity) sparsityC
    integer, dimension(:), allocatable:: colm, findrm
    integer, dimension(:), pointer:: colsA, colsB
    integer i, k, k1, k2, colA, colB, count
    logical have_diag

    if(.not.sparsityA%sorted_rows.or..not.sparsityB%sorted_rows) then
      FLAbort("sparsity_merge assumes sorted rows.")
    end if

    assert( size(sparsityA,1)==size(sparsityB,1) )

    ! allocate oversized, temp. sparsity:
    allocate( colm(1:size(sparsityA%colm)+size(sparsityB)), &
      findrm(1:size(sparsityA,1)+1) )
    count=0
    do i=1, size(sparsityA,1)
       findrm(i)=count+1
       colsA => row_m_ptr(sparsityA, i)
       colsB => row_m_ptr(sparsityB, i)
       k1=1
       k2=1
       if (k1<=size(colsA)) then
         colA=colsA(k1)
         if (k2<=size(colsB)) then
           colB=colsB(k2)
           ! this loop only if both rows are nonzero
           do
             count=count+1
             if (colA<colB) then
               colm(count)=colA
               k1=k1+1
               if (k1>size(colsA)) exit
               colA=colsA(k1)
             else if (colA>colB) then
               colm(count)=colB
               k2=k2+1
               if (k2>size(colsB)) exit
               colB=colsB(k2)
             else
               ! colA==colB
               colm(count)=colA
               k1=k1+1
               k2=k2+1
               if (k1>size(colsA) .or. k2>size(colsB)) exit
               colA=colsA(k1)
               colB=colsB(k2)
             end if
           end do
         end if
       end if
       ! now copy the left over bits from either colsA or colsB
       do k=k1, size(colsA)
          count=count+1
          colm(count)=colsA(k)
       end do
       do k=k2, size(colsB)
          count=count+1
          colm(count)=colsB(k)
       end do
    end do
    findrm(i)=count+1

    have_diag=associated(sparsityA%centrm) .or. associated(sparsityB%centrm)
    call allocate(sparsityC, size(sparsityA,1), &
       max(size(sparsityA,2), size(sparsityB,2)), &
       entries=count, diag=have_diag, name=name)
    sparsityC%findrm=findrm
    sparsityC%colm=colm(1:count)

    if (have_diag) then
      do i=1, size(sparsityC,1)
        sparsityC%centrm(i)=csr_sparsity_pos(sparsityC, i, i)
      end do
    end if

    sparsityC%sorted_rows=.true.

  end function sparsity_merge


  subroutine csr_matrix2file(filename, matrix)
    !!< Write the dense form of matrix to filename.
    !!<
    !!< WARNING! - The dense form of a sparse matrix can get bloody big.
    character(len=*), intent(in) :: filename
    type(csr_matrix), intent(in) :: matrix

    character(len=42) :: format
    integer :: unit

    unit=free_unit()

    open(unit=unit, file=filename, action="write")

    ! Construct the correct format for a matrix row.
    write(format,'(a,i0,a)')"(",size(matrix,2),"g22.8e4)"
    write(unit, format) transpose(dense(matrix))

    close(unit)

  end subroutine csr_matrix2file

  subroutine block_csr_matrix2file(filename, matrix)
    !!< Write the dense form of matrix to filename.
    !!<
    !!< WARNING! - The dense form of a sparse matrix can get bloody big.
    character(len=*), intent(in) :: filename
    type(block_csr_matrix), intent(in) :: matrix

    character(len=42) :: format
    integer :: unit

    unit=free_unit()

    open(unit=unit, file=filename, action="write")

    ! Construct the correct format for a matrix row.
    write(format,'(a,i0,a)')"(",size(matrix,2),"g22.8e4)"
    write(unit, format) transpose(dense(matrix))

    close(unit)

  end subroutine block_csr_matrix2file

  subroutine dcsr_matrix2file(filename, matrix)
    !!< Write the dense form of matrix to filename.
    !!<
    !!< WARNING! - The dense form of a sparse matrix can get bloody big.
    character(len=*), intent(in) :: filename
    type(dynamic_csr_matrix), intent(in) :: matrix

    character(len=42) :: format
    integer :: unit

    unit=free_unit()

    open(unit=unit, file=filename, action="write")

    ! Construct the correct format for a matrix row.
    write(format,'(a,i0,a)')"(",size(matrix,2),"g22.8e4)"
    write(unit, format) transpose(dense(matrix))

    close(unit)

  end subroutine dcsr_matrix2file

  subroutine block_dcsr_matrix2file(filename, matrix)
    !!< Write the dense form of matrix to filename.
    !!<
    !!< WARNING! - The dense form of a sparse matrix can get bloody big.
    character(len=*), intent(in) :: filename
    type(block_dynamic_csr_matrix), intent(in) :: matrix

    character(len=42) :: format
    integer :: unit

    unit=free_unit()

    open(unit=unit, file=filename, action="write")

    ! Construct the correct format for a matrix row.
    write(format,'(a,i0,a)')"(",size(matrix,2),"g22.8e4)"
    write(unit, format) transpose(dense(matrix))

    close(unit)

  end subroutine block_dcsr_matrix2file

  subroutine dense_matrix2file(filename, matrix)
    !!< Write  matrix to filename.
    character(len=*), intent(in) :: filename
    real, dimension(:,:), intent(in) :: matrix

    character(len=42) :: format
    integer :: unit

    unit=free_unit()

    open(unit=unit, file=filename, action="write")

    ! Construct the correct format for a matrix row.
    write(format,'(a,i0,a)')"(",size(matrix,2),"g22.8e4)"
    write(unit, format) transpose(matrix)

    close(unit)

  end subroutine dense_matrix2file

  ! I/O routines to read/write matrices in MatrixMarket format
  ! from http://math.nist.gov/MatrixMarket/formats.html:
  !-----------------------------------------------------------------------------------
  !   %%MatrixMarket matrix coordinate real general
  ! %=================================================================================
  ! %
  ! % This ASCII file represents a sparse MxN matrix with L
  ! % nonzeros in the following Matrix Market format:
  ! %
  ! % +----------------------------------------------+
  ! % |%%MatrixMarket matrix coordinate real general | <--- header line
  ! % |%                                             | <--+
  ! % |% comments                                    |    |-- 0 or more comment lines
  ! % |%                                             | <--+
  ! % |    M  N  L                                   | <--- rows, columns, entries
  ! % |    I1  J1  A(I1, J1)                         | <--+
  ! % |    I2  J2  A(I2, J2)                         |    |
  ! % |    I3  J3  A(I3, J3)                         |    |-- L lines
  ! % |        . . .                                 |    |
  ! % |    IL JL  A(IL, JL)                          | <--+
  ! % +----------------------------------------------+
  ! %
  ! % Indices are 1-based, i.e. A(1,1) is the first element.
  ! %
  ! %=================================================================================
  !   5  5  8
  !     1     1   1.000e+00
  !     2     2   1.050e+01
  !     3     3   1.500e-02
  !     1     4   6.000e+00
  !     4     2   2.505e+02
  !     4     4  -2.800e+02
  !     4     5   3.332e+01
  !     5     5   1.200e+01
  !
  !-----------------------------------------------------------------------------------

  subroutine csr_mmwrite(filename, matrix)
    character(len=*), intent(in) :: filename
    type(csr_matrix), intent(in) :: matrix

    real, dimension(:), pointer:: vals
    integer, dimension(:), pointer:: cols
    integer i, j, unit

    unit=free_unit()

    open(unit=unit, file=filename, action="write")

    ! this will include a leading blank that we will overwrite in the end.
    write(unit, *) "%MatrixMarket matrix coordinate real general"

    write(unit, *) size(matrix,1), size(matrix, 2), entries(matrix)

    do i=1, size(matrix,1)
      cols => row_m_ptr(matrix, i)
      vals => row_val_ptr(matrix, i)
      do j=1, size(cols)
        write(unit, *) i, cols(j), vals(j)
      end do
    end do

    close(unit)

    ! overwrite leading blank with %
    open(unit=unit, file=filename, access='direct', form='formatted', recl=1, &
      action='write')
    write(unit, '(a1)', rec=1) '%'
    close(unit)

  end subroutine csr_mmwrite

  subroutine dcsr_mmwrite(filename, matrix)
    character(len=*), intent(in) :: filename
    type(dynamic_csr_matrix), intent(in) :: matrix

    real, dimension(:), pointer:: vals
    integer, dimension(:), pointer:: cols
    integer i, j, unit

    unit=free_unit()

    open(unit=unit, file=filename, action="write")

    ! write header line
    ! this will include a leading blank that we will overwrite in the end.
    write(unit, *) "%MatrixMarket matrix coordinate real general"

    write(unit, *) size(matrix,1), size(matrix, 2), entries(matrix)

    do i=1, size(matrix,1)
      cols => row_m_ptr(matrix, i)
      vals => row_val_ptr(matrix, i)
      do j=1, size(cols)
        write(unit, *) i, cols(j), vals(j)
      end do
    end do

    close(unit)

    ! overwrite leading blank with %
    open(unit=unit, file=filename, access='direct', form='formatted', recl=1, &
      action='write')
    write(unit, '(a1)', rec=1) '%'
    close(unit)

  end subroutine dcsr_mmwrite

  subroutine dcsr_mmread(filename, matrix)
    character(len=*), intent(in) :: filename
    type(dynamic_csr_matrix), intent(out) :: matrix

    character(len=MMmaxlinelen) line
    integer unit, rows, cols, nnz, row, col
    real value

    call mmreadheader(filename, unit, rows, cols, nnz)

    call allocate(matrix, rows, cols)

    do
      read(unit, fmt=MMlineformat) line
      if (len_trim(line)==0) cycle
      read(line, *) row, col, value
      call set(matrix, row, col, value)
      nnz=nnz-1
      if (nnz==0) exit
    end do

    close(unit)

  end subroutine dcsr_mmread

  subroutine mmreadheader(filename, unit, rows, cols, nnz)
  ! Opens MatrixMarket file with returned unit number,
  ! reads in the header of the file and checks that
  ! it says 'coordinate real general' (only thing we support right now)
    character(len=*), intent(in) :: filename
    integer, intent(out):: unit, rows, cols, nnz

    character(len=*), parameter:: headerword(1:5)=(/ &
      '%%MatrixMarket', &
      'matrix        ', &
      'coordinate    ', &
      'real          ', &
      'general       ' /)

    character(len=MMmaxlinelen) line
    integer i, j

    unit=free_unit()

    ewrite(2, *) 'Opening MatrixMarket file: ', trim(filename)
    open(unit=unit, file=filename, action="read")

    ! read header line
    read(unit, fmt=MMlineformat) line

    i=1
    j=1
    do
      if (line(i:i)==' ') then
        i=i+1
        cycle
      end if

      if (i+len_trim(headerword(j))-1>len_trim(line) .or. &
        line(i:i+len_trim(headerword(j))-1)/=headerword(j)) then
        ewrite(-1,*) 'First line reads:'
        ewrite(-1,*) trim(line)
        ewrite(-1,*) "MatrixMarket file not in 'matrix coordinate", &
                      & "real general' format."
        FLAbort("MatrixMarket file cannot be generated.")
      end if

      i=i+len_trim(headerword(j))
      j=j+1
      if (j>size(headerword)) exit
    end do

    do
      read(unit, fmt=MMlineformat) line
      line=adjustl(line)
      if (len_trim(line)/=0 .and. line(1:1)/='%') exit
    end do

    read(line, *) rows, cols, nnz
    ewrite(2,*) 'rows, cols, nnz: ', rows, cols, nnz

  end subroutine mmreadheader

  subroutine csr_write_minmax(matrix, matrix_expression)
    ! the matrix to print its min and max of
    type(csr_matrix), intent(in):: matrix
    ! the actual matrix in the code
    character(len=*), intent(in):: matrix_expression

    ewrite(2,*) 'Min, max of '//trim(matrix_expression)//' "'// &
            trim(matrix%name)//'" = ', minval(matrix%val), maxval(matrix%val)

  end subroutine csr_write_minmax

  subroutine block_csr_write_minmax(matrix, matrix_expression)
    ! the matrix to print its min and max of
    type(block_csr_matrix), intent(in):: matrix
    ! the actual matrix in the code
    character(len=*), intent(in):: matrix_expression

    integer:: i, j

    do i=1, blocks(matrix, 1)
      do j=1, blocks(matrix, 2)
        if (associated(matrix%val(i,j)%ptr)) then
          ewrite(2,*) 'Min, max of '//trim(matrix_expression)//' "'// &
            trim(matrix%name)//'%'//int2str(i)//','//int2str(j)// &
            '" = ', minval(matrix%val(i,j)%ptr), maxval(matrix%val(i,j)%ptr)
        end if
      end do
    end do

  end subroutine block_csr_write_minmax

#include "Reference_count_csr_matrix.F90"
#include "Reference_count_csr_sparsity.F90"
#include "Reference_count_block_csr_matrix.F90"
#include "Reference_count_dynamic_csr_matrix.F90"
#include "Reference_count_block_dynamic_csr_matrix.F90"

end module sparse_tools
