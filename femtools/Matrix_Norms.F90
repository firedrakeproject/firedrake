#include "fdebug.h"

module matrix_norms

  use fields
  use vector_tools
  use node_boundary
  implicit none

  interface one_norm
    module procedure one_norm_matrix, one_norm_field
  end interface

  interface two_norm
    module procedure two_norm_matrix, two_norm_field
  end interface

  interface inf_norm
    module procedure inf_norm_matrix, inf_norm_field
  end interface

  contains

  function one_norm_matrix(matrix) result(one)
    real, dimension(:, :), intent(in) :: matrix
    real :: one
    integer :: j

    one = 0.0

    do j=1,size(matrix, 2)
      one = max(one, sum(abs(matrix(:, j))))
    end do
  end function one_norm_matrix

  function one_norm_field(field, boundary) result(one)
    type(tensor_field), intent(in) :: field
    logical, intent(in), optional :: boundary
    real :: one
    integer :: node
    logical :: lboundary

    one = 0.0

    lboundary = .false.
    if (present(boundary)) then
      lboundary = boundary
    end if

    do node=1,node_count(field)
      if (lboundary .and. .not. node_lies_on_boundary(node)) then
        cycle
      end if
      one = one + one_norm_matrix(node_val(field, node))
    end do
    one = one / node_count(field)
  end function one_norm_field

  function two_norm_matrix(matrix) result(two)
    real, dimension(:, :), intent(in) :: matrix
    real :: two
    real, dimension(size(matrix, 1), size(matrix, 1)) :: evecs
    real, dimension(size(matrix, 1)) :: evals

    assert(size(matrix, 1) == size(matrix, 2))
    call eigendecomposition_symmetric(matmul(transpose(matrix), matrix), evecs, evals)
    two = sqrt(maxval(evals))
  end function two_norm_matrix

  function two_norm_field(field, boundary) result(two)
    type(tensor_field), intent(in) :: field
    logical, intent(in), optional :: boundary
    real :: two
    integer :: node
    logical :: lboundary

    two = 0.0

    lboundary = .false.
    if (present(boundary)) then
      lboundary = boundary
    end if

    do node=1,node_count(field)
      if (lboundary .and. .not. node_lies_on_boundary(node)) then
        cycle
      end if
      two = two + two_norm_matrix(node_val(field, node))**2
    end do
    two = two / node_count(field)
    two = sqrt(two)
  end function two_norm_field

  function inf_norm_matrix(matrix) result(inf)
    real, dimension(:, :), intent(in) :: matrix
    real :: inf
    integer :: i

    inf = 0.0

    do i=1,size(matrix, 1)
      inf = max(inf, sum(abs(matrix(i, :))))
    end do
  end function inf_norm_matrix

  function inf_norm_field(field, boundary) result(inf)
    type(tensor_field), intent(in) :: field
    logical, intent(in), optional :: boundary
    real :: inf
    integer :: node
    logical :: lboundary

    inf = 0.0

    lboundary = .false.
    if (present(boundary)) then
      lboundary = boundary
    end if

    do node=1,node_count(field)
      if (lboundary .and. .not. node_lies_on_boundary(node)) then
        cycle
      end if
      inf = max(inf, inf_norm_matrix(node_val(field, node)))
    end do
  end function inf_norm_field
end module matrix_norms
