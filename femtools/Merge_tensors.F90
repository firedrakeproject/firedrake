#include "fdebug.h"

module merge_tensors
    !!< This module contains code to merge two tensors representing
    !!< anisotropic mesh information together to form a new metric
    !!< satisfying both constraints.
    !!< See Gerard Gorman's thesis, section 2.4.

    use fields
    use vector_tools
    use metric_tools, only: aspect_ratio
    use unittest_tools
    implicit none

    private
    public :: merge_tensor, merge_tensor_fields, get_deformation_matrix
    contains

    subroutine merge_tensor(tensor1, tensor2, aniso_min)
      !!< Merge two tensors together, putting the result in tensor1.
      real, dimension(:, :), intent(inout), target :: tensor1
      real, dimension(:, :), intent(inout), target :: tensor2
      logical, optional :: aniso_min
      logical :: laniso_min
      real, dimension(size(tensor1, 1), size(tensor1, 1)), target :: v1, v2, Finv, T, F, store ! eigenvectors, temporary storage, deformation matrix
      real, dimension(size(tensor1, 1)), target :: a1, a2 ! eigenvalues
      integer :: i, dim, stat
      real, dimension(:, :), pointer :: sphere_t, other_t, sphere_v, other_v ! which one gets mapped to the sphere?
      real, dimension(:), pointer :: sphere_a, other_a
      real :: aspect1, aspect2
      logical :: eigenvalue_changed

      dim = size(tensor1, 1)

      ! I hate this aniso_min business, because it's really ugly.
      ! This is for when you're merging with the anisotropic minimum
      ! edge length tensor (OPHSAM == 1).
      ! Because you're given the bounding ellipse that everything
      ! has to be smaller than, you have to take the minimum
      ! of the eigenvalues, not the maximum.
      ! But coding it makes it look a bit ugly. If it helps, just
      ! ignore the case aniso_min/laniso_min = .true.

      if (present(aniso_min)) then
        laniso_min = aniso_min
      else
        laniso_min = .false.
      end if

      ! Step 1: decompose the two matrices.

      call eigendecomposition_symmetric(tensor1, v1, a1)
      call eigendecomposition_symmetric(tensor2, v2, a2)

      call vec_clean(a1, 1e-12)
      call vec_clean(a2, 1e-12)

      if (maxval(a1) == 0.0 .and. maxval(a2) == 0.0) then
        return
      end if

      do i=1,dim
        if (a1(i) .lt. 0.0) then
          a1(i) = 0.0
        end if
        if (a2(i) .lt. 0.0) then
          a2(i) = 0.0
        end if
      end do

      aspect1 = aspect_ratio(a1)
      aspect2 = aspect_ratio(a2)

      if (.not. laniso_min) then
        if (aspect1 .le. aspect2) then ! so aspect1 is mapped to the sphere
          sphere_t => tensor1; sphere_v => v1; sphere_a => a1
          other_t => tensor2; other_v => v2; other_a => a2
        else
          sphere_t => tensor2; sphere_v => v2; sphere_a => a2
          other_t => tensor1; other_v => v1; other_a => a1
        end if
      else
        sphere_t => tensor2; sphere_v => v2; sphere_a => a2
        other_t => tensor1; other_v => v1; other_a => a1
      end if

      store = other_t ! if no eigenvalues change, don't do the eigendecomposition/eigenrecomposition
      eigenvalue_changed = .false.

      ! Step 1: Map sphere_t to the sphere.
      ! Apply the same mapping to other_t.

      F = get_deformation_matrix(sphere_t, sphere_v, sphere_a)
      Finv = F; call invert(Finv, stat=stat)

      if (stat /= 0) then
        call write_matrix(sphere_t, "The tensor we are mapping to the sphere")
        call write_matrix(sphere_v, "Its eigenvectors")
        call write_vector(sphere_a, "Its eigenvalues")
        call write_matrix(F, "Deformation matrix")
        call write_matrix(other_t, "Other matrix")
        call write_matrix(other_v, "Its eigenvectors")
        call write_vector(other_a, "Its eigenvalues")
        call write_vector(a1, "First eigenvalues")
        call write_vector(a2, "Second eigenvalues")
        write (0,*) "aspect1 == ", aspect1, "; aspect2 == ", aspect2
        FLAbort("Error: inverting deformation matrix failed")
      end if

      T = transpose(Finv)
      other_t = matmul(matmul(T, other_t), transpose(T))
      !if (.not. mat_symmetric(other_t)) then
      !  ewrite(-1,*) "other_t == ", transpose(other_t)
      !  FLAbort("other_t not symmetric: something is seriously wrong.")
      !end if

      ! Step 2: Apply the eigendecomposition to other_t.
      ! Change its eigenvalues.
      ! Reform.

      call eigendecomposition_symmetric(other_t, other_v, other_a)
      if (laniso_min) then
        do i=1,dim
          if (other_a(i) .fgt. 1.0) eigenvalue_changed = .true.
          other_a(i) = min(other_a(i), 1.0)
        end do
      else
        do i=1,dim
          if (other_a(i) .flt. 1.0) eigenvalue_changed = .true.
          other_a(i) = max(other_a(i), 1.0)
        end do
      end if
      call eigenrecomposition(other_t, other_v, other_a)
      ! Step 3: Apply the inverse map to other_t.

      T = F
      tensor1 = matmul(matmul(transpose(T), other_t), T)

      if (.not. eigenvalue_changed) tensor1 = store ! ignore the eigendecomposition/recomposition
      ! Done.
    end subroutine merge_tensor

    function get_deformation_matrix(M, V, A) result(F)
      !! Compute F = A^(1/2) * V^T
      real, dimension(:, :), intent(in) :: M
      real, dimension(size(M, 1), size(M, 1)), intent(in), optional :: V
      real, dimension(size(M, 1)), intent(in), optional :: A
      real, dimension(size(M, 1), size(M, 1)) :: local_V, F
      real, dimension(size(M, 1)) :: local_A
      integer :: i

      if (present(V) .and. present(A)) then
        local_V = V
        local_A = A
      else
        call eigendecomposition_symmetric(M, local_V, local_A)
      end if

      do i=1,size(M, 1)
        local_A(i) = sqrt(local_A(i))
      end do

      F = matmul(get_mat_diag(local_A), transpose(local_V))
    end function get_deformation_matrix

    subroutine merge_tensor_fields(fielda, fieldb, aniso_min)
      !!< Loop through the two tensor fields and merge them nodewise.
      type(tensor_field), intent(inout) :: fielda, fieldb
      logical, intent(in), optional :: aniso_min
      integer :: i

      assert(fielda%mesh%nodes == fieldb%mesh%nodes)

      ewrite(2,*) "Merging tensor fields."

      do i=1,fielda%mesh%nodes
        call merge_tensor(fielda%val(:, :, i), fieldb%val(:, :, i), aniso_min)
      end do
    end subroutine
end module merge_tensors
