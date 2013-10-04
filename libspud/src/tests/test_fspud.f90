!    Copyright (C) 2007 Imperial College London and others.
!
!    Please see the AUTHORS file in the main source directory for a full list
!    of copyright holders.
!
!    Applied Modelling and Computation Group
!    Department of Earth Science and Engineering
!    Imperial College London
!
!    David.Ham@Imperial.ac.uk
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

subroutine test_fspud

  use spud
  use unittest_tools

  implicit none

  integer, parameter :: D = kind(0.0D0)
  real(D), parameter :: tol = epsilon(0.0)

  print *, "*** Testing clear_options ***"
  call test_clear_options("/type_none")

  print *, "*** Testing set_option and get_option for real scalar ***"
  call test_set_and_get_real_scalar("/real_scalar", 42.0_D)

  print *, "*** Testing set_option and get_option for real vector ***"
  call test_set_and_get_real_vector("/real_vector", (/42.0_D, 43.0_D/))

  print *, "*** Testing set_option and get_option for real tensor ***"
  call test_set_and_get_real_tensor("/real_tensor", reshape((/42.0_D, 43.0_D, 44.0_D, 45.0_D, 46.0_D, 47.0_D/), (/2, 3/)))

  print *, "*** Testing set_option and get_option for integer scalar ***"
  call test_set_and_get_integer_scalar("/integer_scalar", 42)

  print *, "*** Testing set_option and get_option for integer vector ***"
  call test_set_and_get_integer_vector("/integer_vector", (/42, 43/))

  print *, "*** Testing set_option and get_option for integer tensor ***"
  call test_set_and_get_integer_tensor("/integer_tensor", reshape((/42, 43, 44, 45, 46, 47/), (/2, 3/)))

  print *, "*** Testing set_option and get_option for character ***"
  call test_set_and_get_character("/character", "Forty Two")

  print *, "*** Testing add_option and get_option ***"
  call test_set_and_get_type_none("/type_none")

  print *, "*** Testing set_option for integer scalar, with option name ***"
  call test_named_key("/integer_scalar", "name", 42)

  print *, "*** Testing set_option for integer scalar, with option name (containing symbols) ***"
  call test_named_key("/integer_scalar", 'tricky_name!"£$%^&*()', 42)

  print *, "*** Testing set_option for integer scalar, with option index ***"
  call test_indexed_key("/integer_scalar", 42)

  print *, "*** Testing move_option ***"
  call test_move_option("/type_none", "/type_none_2")

  print *, "*** Testing copy_option ***"
  call test_copy_option("/type_none", "/type_none_2")

contains

  subroutine test_key_errors(key)
    character(len = *), intent(in) :: key

    character(len = 255) :: test_char
    integer :: test_integer_scalar
    integer, dimension(3) :: test_integer_vector, integer_vector_default
    integer, dimension(3, 4) :: test_integer_tensor, integer_tensor_default
    real(D) :: test_real_scalar
    real(D), dimension(3) :: test_real_vector, real_vector_default
    real(D), dimension(3, 4) :: test_real_tensor, real_tensor_default
    integer :: rank, type, stat
    integer, dimension(2) :: shape

    integer :: i, j

    do i = 1, size(real_vector_default)
      real_vector_default = 42.0_D + i
    end do
    do i = 1, size(real_tensor_default, 1)
      do j = 1, size(real_tensor_default, 2)
        real_tensor_default = 42.0_D + i * size(real_tensor_default, 2) + j
      end do
    end do
    do i = 1, size(integer_vector_default)
      integer_vector_default = 42.0_D + i
    end do
    do i = 1, size(real_tensor_default, 1)
      do j = 1, size(integer_tensor_default, 2)
        integer_tensor_default = 42.0_D + i * size(integer_tensor_default, 2) + j
      end do
    end do

    call report_test("[Missing option]", have_option(key), .false., "Missing option reported present")
    type = option_type(key, stat)
    call report_test("[Key error when extracting option type]", stat /= SPUD_KEY_ERROR, .false., "Returned incorrect error code when retrieving option type")
    rank = option_rank(key, stat)
    call report_test("[Key error when extracting option rank]", stat /= SPUD_KEY_ERROR, .false., "Returned incorrect error code when retrieving option rank")
    shape = option_shape(key, stat)
    call report_test("[Key error when extracting option shape]", stat /= SPUD_KEY_ERROR, .false., "Returned incorrect error code when retrieving option shape")
    call get_option(key, test_real_scalar, stat)
    call report_test("[Key error when extracting option data]", stat /= SPUD_KEY_ERROR, .false., "Returned incorrect error code when retrieving option data")
    call get_option(key, test_real_scalar, stat, default = 42.0_D)
    call report_test("[Extracted option data with default argument]", stat /= SPUD_NO_ERROR, .false., "Returned error code when retrieving option data")
    call report_test("[Extracted correct option data (default)]", abs(test_real_scalar - 42.0_D) > tol, .false., "Retrieved incorrect option data")
    call get_option(key, test_real_vector, stat)
    call report_test("[Key error when extracting option data]", stat /= SPUD_KEY_ERROR, .false., "Returned incorrect error code when retrieving option data")
    call get_option(key, test_real_vector, stat, default = real_vector_default)
    call report_test("[Extracted option data with default argument]", stat /= SPUD_NO_ERROR, .false., "Returned error code when retrieving option data")
    call report_test("[Extracted correct option data (default)]", maxval(abs(test_real_vector - real_vector_default)) > tol, .false., "Retrieved incorrect option data")
    call get_option(key, test_real_tensor, stat)
    call report_test("[Key error when extracting option data]", stat /= SPUD_KEY_ERROR, .false., "Returned incorrect error code when retrieving option data")
    call get_option(key, test_real_tensor, stat, default = real_tensor_default)
    call report_test("[Extracted option data with default argument]", stat /= SPUD_NO_ERROR, .false., "Returned error code when retrieving option data")
    call report_test("[Extracted correct option data (default)]", maxval(abs(test_real_tensor - real_tensor_default)) > tol, .false., "Retrieved incorrect option data")
    call get_option(key, test_integer_scalar, stat)
    call report_test("[Key error when extracting option data]", stat /= SPUD_KEY_ERROR, .false., "Returned incorrect error code when retrieving option data")
    call get_option(key, test_integer_scalar, stat, default = 42)
    call report_test("[Extracted option data with default argument]", stat /= SPUD_NO_ERROR, .false., "Returned error code when retrieving option data")
    call report_test("[Extracted correct option data (default)]", test_integer_scalar /= 42, .false., "Retrieved incorrect option data")
    call get_option(key, test_integer_vector, stat)
    call report_test("[Key error when extracting option data]", stat /= SPUD_KEY_ERROR, .false., "Returned incorrect error code when retrieving option data")
    call get_option(key, test_integer_vector, stat, default = integer_vector_default)
    call report_test("[Extracted option data with default argument]", stat /= SPUD_NO_ERROR, .false., "Returned error code when retrieving option data")
    call report_test("[Extracted correct option data (default)]", count(test_integer_vector /= integer_vector_default) > 0, .false., "Retrieved incorrect option data")
    call get_option(key, test_integer_tensor, stat)
    call report_test("[Key error when extracting option data]", stat /= SPUD_KEY_ERROR, .false., "Returned incorrect error code when retrieving option data")
    call get_option(key, test_integer_tensor, stat, default = integer_tensor_default)
    call report_test("[Extracted option data with default argument]", stat /= SPUD_NO_ERROR, .false., "Returned error code when retrieving option data")
    call report_test("[Extracted correct option data (default)]", count(test_integer_tensor /= integer_tensor_default) > 0, .false., "Retrieved incorrect option data")
    call get_option(key, test_char, stat)
    call report_test("[Key error when extracting option data]", stat /= SPUD_KEY_ERROR, .false., "Returned incorrect error code when retrieving option data")
    call get_option(key, test_char, stat, default = "Forty Two")
    call report_test("[Extracted option data with default argument]", stat /= SPUD_NO_ERROR, .false., "Returned error code when retrieving option data")
    call report_test("[Extracted correct option data (default)]", test_char /= "Forty Two", .false., "Retrieved incorrect option data")

  end subroutine test_key_errors

  subroutine test_key_present(key)
    character(len = *), intent(in) :: key

    call report_test("[Option present]", .not. have_option(key), .false., "Present option reported missing")

  end subroutine test_key_present

  subroutine test_type(key, type)
    character(len = *), intent(in) :: key
    integer, intent(in) :: type

    integer :: stat, type_ret

    type_ret = option_type(key, stat)
    call report_test("[Extracted option type]", stat /= SPUD_NO_ERROR, .false., "Returned error code when retrieving option type")
    call report_test("[Correct option type]", type_ret /= type, .false., "Incorrect option type returned")

  end subroutine test_type

  subroutine test_rank(key, rank)
    character(len = *), intent(in) :: key
    integer, intent(in) :: rank

    integer :: stat, rank_ret

    rank_ret = option_rank(key, stat)
    call report_test("[Extracted option rank]", stat /= SPUD_NO_ERROR, .false., "Returned error code when retrieving option rank")
    call report_test("[Correct option rank]", rank_ret /= rank, .false., "Incorrect option rank returned")

  end subroutine test_rank

  subroutine test_shape(key, shape)
    character(len = *), intent(in) :: key
    integer, dimension(2), intent(in) :: shape

    integer :: stat
    integer, dimension(2) :: shape_ret

    shape_ret = option_shape(key, stat)
    call report_test("[Extracted option shape]", stat /= SPUD_NO_ERROR, .false., "Returned error code when retrieving option shape")
    call report_test("[Correct option shape]", count(shape_ret /= shape) /= 0, .false., "Incorrect option shape returned")

  end subroutine test_shape

  subroutine test_type_errors_real(key)
    character(len = *), intent(in) :: key

    integer :: stat
    real(D) :: real_scalar_val
    real(D), dimension(3) :: real_vector_default, real_vector_val
    real(D), dimension(3, 4) :: real_tensor_default, real_tensor_val

    call get_option(key, real_scalar_val, stat)
    call report_test("[Type error when extracting option data]", stat /= SPUD_TYPE_ERROR, .false., "Returned incorrect error code when retrieving option data")
    call get_option(key, real_scalar_val, stat, default = 0.0_D)
    call report_test("[Type error when extracting option data with default argument]", stat /= SPUD_TYPE_ERROR, .false., "Returned incorrect error code when retrieving option data")
    call get_option(key, real_vector_val, stat)
    call report_test("[Type error when extracting option data]", stat /= SPUD_TYPE_ERROR, .false., "Returned incorrect error code when retrieving option data")
    call get_option(key, real_vector_val, stat, default = real_vector_default)
    call report_test("[Type error when extracting option data with default argument]", stat /= SPUD_TYPE_ERROR, .false., "Returned incorrect error code when retrieving option data")
    call get_option(key, real_tensor_val, stat)
    call report_test("[Type error when extracting option data]", stat /= SPUD_TYPE_ERROR, .false., "Returned incorrect error code when retrieving option data")
    call get_option(key, real_tensor_val, stat, default = real_tensor_default)
    call report_test("[Type error when extracting option data with default argument]", stat /= SPUD_TYPE_ERROR, .false., "Returned incorrect error code when retrieving option data")

  end subroutine test_type_errors_real

  subroutine test_type_errors_integer(key)
    character(len = *), intent(in) :: key

    integer :: integer_scalar_val, stat
    integer, dimension(3) :: integer_vector_default, integer_vector_val
    integer, dimension(3, 4) :: integer_tensor_default, integer_tensor_val

    call get_option(key, integer_scalar_val, stat)
    call report_test("[Type error when extracting option data]", stat /= SPUD_TYPE_ERROR, .false., "Returned incorrect error code when retrieving option data")
    call get_option(key, integer_scalar_val, stat, default = 0)
    call report_test("[Type error when extracting option data with default argument]", stat /= SPUD_TYPE_ERROR, .false., "Returned incorrect error code when retrieving option data")
    call get_option(key, integer_vector_val, stat)
    call report_test("[Type error when extracting option data]", stat /= SPUD_TYPE_ERROR, .false., "Returned incorrect error code when retrieving option data")
    call get_option(key, integer_vector_val, stat, default = integer_vector_default)
    call report_test("[Type error when extracting option data with default argument]", stat /= SPUD_TYPE_ERROR, .false., "Returned incorrect error code when retrieving option data")
    call get_option(key, integer_tensor_val, stat)
    call report_test("[Type error when extracting option data]", stat /= SPUD_TYPE_ERROR, .false., "Returned incorrect error code when retrieving option data")
    call get_option(key, integer_tensor_val, stat, default = integer_tensor_default)
    call report_test("[Type error when extracting option data with default argument]", stat /= SPUD_TYPE_ERROR, .false., "Returned incorrect error code when retrieving option data")

  end subroutine test_type_errors_integer

  subroutine test_type_errors_character(key)
    character(len = *), intent(in) :: key

    character(len = 0) :: character_val
    integer :: stat

    call get_option(key, character_val, stat)
    call report_test("[Type error when extracting option data]", stat /= SPUD_TYPE_ERROR, .false., "Returned incorrect error code when retrieving option data")
    call get_option(key, character_val, stat, default = "")
    call report_test("[Type error when extracting option data with default argument]", stat /= SPUD_TYPE_ERROR, .false., "Returned incorrect error code when retrieving option data")

  end subroutine test_type_errors_character

  subroutine test_rank_errors_real_scalar(key)
    character(len = *), intent(in) :: key

    integer :: stat
    real(D) :: real_scalar_val

    call get_option(key, real_scalar_val, stat)
    call report_test("[Rank error when extracting option data]", stat /= SPUD_RANK_ERROR, .false., "Returned incorrect error code when retrieving option data")
    call get_option(key, real_scalar_val, stat, default = 0.0_D)
    call report_test("[Rank error when extracting option data with default argument]", stat /= SPUD_RANK_ERROR, .false., "Returned error code when retrieving option data")

  end subroutine test_rank_errors_real_scalar

  subroutine test_rank_errors_real_vector(key)
    character(len = *), intent(in) :: key

    integer :: stat
    real(D), dimension(3) :: real_vector_default, real_vector_val

    call get_option(key, real_vector_val, stat)
    call report_test("[Rank error when extracting option data]", stat /= SPUD_RANK_ERROR, .false., "Returned incorrect error code when retrieving option data")
    call get_option(key, real_vector_val, stat, default = real_vector_default)
    call report_test("[Rank error when extracting option data with default argument]", stat /= SPUD_RANK_ERROR, .false., "Returned error code when retrieving option data")

  end subroutine test_rank_errors_real_vector

  subroutine test_rank_errors_real_tensor(key)
    character(len = *), intent(in) :: key

    integer :: stat
    real(D), dimension(3, 4) :: real_tensor_default, real_tensor_val

    call get_option(key, real_tensor_val, stat)
    call report_test("[Rank error when extracting option data]", stat /= SPUD_RANK_ERROR, .false., "Returned incorrect error code when retrieving option data")
    call get_option(key, real_tensor_val, stat, default = real_tensor_default)
    call report_test("[Rank error when extracting option data with default argument]", stat /= SPUD_RANK_ERROR, .false., "Returned error code when retrieving option data")

  end subroutine test_rank_errors_real_tensor

  subroutine test_rank_errors_integer_scalar(key)
    character(len = *), intent(in) :: key

    integer :: integer_scalar_val, stat

    call get_option(key, integer_scalar_val, stat)
    call report_test("[Rank error when extracting option data]", stat /= SPUD_RANK_ERROR, .false., "Returned incorrect error code when retrieving option data")
    call get_option(key, integer_scalar_val, stat, default = 0)
    call report_test("[Rank error when extracting option data with default argument]", stat /= SPUD_RANK_ERROR, .false., "Returned error code when retrieving option data")

  end subroutine test_rank_errors_integer_scalar

  subroutine test_rank_errors_integer_vector(key)
    character(len = *), intent(in) :: key

    integer :: stat
    integer, dimension(3) :: integer_vector_default, integer_vector_val

    call get_option(key, integer_vector_val, stat)
    call report_test("[Rank error when extracting option data]", stat /= SPUD_RANK_ERROR, .false., "Returned incorrect error code when retrieving option data")
    call get_option(key, integer_vector_val, stat, default = integer_vector_default)
    call report_test("[Rank error when extracting option data with default argument]", stat /= SPUD_RANK_ERROR, .false., "Returned error code when retrieving option data")

  end subroutine test_rank_errors_integer_vector

  subroutine test_rank_errors_integer_tensor(key)
    character(len = *), intent(in) :: key

    integer :: stat
    integer, dimension(3, 4) :: integer_tensor_default, integer_tensor_val

    call get_option(key, integer_tensor_val, stat)
    call report_test("[Rank error when extracting option data]", stat /= SPUD_RANK_ERROR, .false., "Returned incorrect error code when retrieving option data")
    call get_option(key, integer_tensor_val, stat, default = integer_tensor_default)
    call report_test("[Rank error when extracting option data with default argument]", stat /= SPUD_RANK_ERROR, .false., "Returned error code when retrieving option data")

  end subroutine test_rank_errors_integer_tensor

  subroutine test_add_new_option(key)
    character(len = *), intent(in) :: key

    integer :: stat

    call add_option(key, stat)
    call report_test("[New option]", stat /= SPUD_NEW_KEY_WARNING, .false., "Failed to return new key warning when adding option")

  end subroutine test_add_new_option

  subroutine test_delete_option(key)
    character(len = *), intent(in) :: key

    integer :: stat

    call delete_option(key, stat)
    call report_test("[Deleted option]", stat /= SPUD_NO_ERROR, .false., "Returned error code when deleting option")

    call test_key_errors(key)

  end subroutine test_delete_option

  subroutine test_clear_options(key)
    character(len = *), intent(in) :: key

    call test_key_errors(key)
    call test_add_new_option(key)
    call clear_options()
    call test_key_errors(key)

  end subroutine test_clear_options

  subroutine test_set_and_get_real_scalar(key, test_real_scalar)
    character(len = *), intent(in) :: key
    real(D), intent(in) :: test_real_scalar

    integer :: i, stat
    real(D) :: ltest_real_scalar, real_scalar_val

    call test_key_errors(key)

    do i = 1, 2
      select case(i)
        case(1)
          ltest_real_scalar = test_real_scalar
          call set_option(key, ltest_real_scalar, stat)
          call report_test("[New option]", stat /= SPUD_NEW_KEY_WARNING, .false., "Failed to return new key warning when setting option")
        case default
          ltest_real_scalar = ltest_real_scalar * 1.1_D
          call set_option(key, ltest_real_scalar, stat)
          call report_test("[Set existing option]", stat /= SPUD_NO_ERROR, .false., "Returned error code when setting option")
      end select

      call test_key_present(key)
      call test_type(key, SPUD_REAL)
      call test_rank(key, 0)
      call test_shape(key, (/-1, -1/) )

      call get_option(key, real_scalar_val, stat)
      call report_test("[Extracted option data]", stat /= SPUD_NO_ERROR, .false., "Returned error code when retrieving option data")
      call report_test("[Extracted correct option data]", abs(real_scalar_val - ltest_real_scalar) > tol, .false., "Retrieved incorrect option data")
      call get_option(key, real_scalar_val, stat, default = ltest_real_scalar * 1.1_D)
      call report_test("[Extracted option data with default argument]", stat /= SPUD_NO_ERROR, .false., "Returned error code when retrieving option data")
      call report_test("[Extracted correct option data with default argument]", abs(real_scalar_val - ltest_real_scalar) > tol, .false., "Retrieved incorrect option data")

      call test_rank_errors_real_vector(key)
      call test_rank_errors_real_tensor(key)
      call test_type_errors_integer(key)
      call test_type_errors_character(key)

      call test_get_character(key // "/__value/rank", "0")
      call test_key_errors(key // "/__value/shape")

    end do

    call test_delete_option(key)

  end subroutine test_set_and_get_real_scalar

  subroutine test_set_and_get_real_vector(key, test_real_vector)
    character(len = *), intent(in) :: key
    real(D), dimension(:), intent(in) :: test_real_vector

    integer :: i, stat
    real(D), dimension(size(test_real_vector)) :: ltest_real_vector
    real(D), dimension(:), allocatable :: real_vector_default, real_vector_val

    call test_key_errors(key)

    do i = 1, 2
      select case(i)
        case(1)
          ltest_real_vector = test_real_vector
          call set_option(key, ltest_real_vector, stat)
          call report_test("[New option]", stat /= SPUD_NEW_KEY_WARNING, .false., "Failed to return new key warning when setting option")
        case default
          ltest_real_vector = ltest_real_vector * 1.1_D
          call set_option(key, ltest_real_vector, stat)
          call report_test("[Set existing option]", stat /= SPUD_NO_ERROR, .false., "Returned error code when setting option")
      end select

      call test_key_present(key)
      call test_type(key, SPUD_REAL)
      call test_rank(key, 1)
      call test_shape(key, (/size(ltest_real_vector), -1/))

      allocate(real_vector_val(size(ltest_real_vector)))
      allocate(real_vector_default(size(ltest_real_vector)))
      call get_option(key, real_vector_val, stat)
      call report_test("[Extracted option data]", stat /= SPUD_NO_ERROR, .false., "Returned error code when retrieving option data")
      call report_test("[Extracted correct option data]", maxval(abs(real_vector_val - ltest_real_vector)) > tol, .false., "Retrieved incorrect option data")
      call get_option(key, real_vector_val, stat, default = ltest_real_vector * 1.1_D)
      call report_test("[Extracted option data with default argument]", stat /= SPUD_NO_ERROR, .false., "Returned error code when retrieving option data")
      call report_test("[Extracted correct option data with default argument]", maxval(abs(real_vector_val - ltest_real_vector)) > tol, .false., "Retrieved incorrect option data")
      deallocate(real_vector_val)
      deallocate(real_vector_default)
      allocate(real_vector_val(size(ltest_real_vector) + 1))
      allocate(real_vector_default(size(ltest_real_vector) + 1))
      call get_option(key, real_vector_val, stat)
      call report_test("[Shape error when extracting option data]", stat /= SPUD_SHAPE_ERROR, .false., "Returned error code when retrieving option data")
      call get_option(key, real_vector_val, stat, default = real_vector_default)
      call report_test("[Shape error when extracting option data with default argument]", stat /= SPUD_SHAPE_ERROR, .false., "Returned error code when retrieving option data")
      deallocate(real_vector_val)
      deallocate(real_vector_default)

      call test_rank_errors_real_scalar(key)
      call test_rank_errors_real_tensor(key)
      call test_type_errors_integer(key)
      call test_type_errors_character(key)

      call test_get_character(key // "/__value/rank", "1")
      call test_get_character(key // "/__value/shape", int2str(size(test_real_vector)))

    end do

    call test_delete_option(key)

  end subroutine test_set_and_get_real_vector

  subroutine test_set_and_get_real_tensor(key, test_real_tensor)
    character(len = *), intent(in) :: key
    real(D), dimension(:, :), intent(in) :: test_real_tensor

    integer :: i, stat
    real(D), dimension(size(test_real_tensor, 1), size(test_real_tensor, 2)) :: ltest_real_tensor
    real(D), dimension(:, :), allocatable :: real_tensor_default, real_tensor_val

    call test_key_errors(key)

    do i = 1, 2
      select case(i)
        case(1)
          ltest_real_tensor = test_real_tensor
          call set_option(key, ltest_real_tensor, stat)
          call report_test("[New option]", stat /= SPUD_NEW_KEY_WARNING, .false., "Failed to return new key warning when setting option")
        case default
          ltest_real_tensor = ltest_real_tensor * 1.1_D
          call set_option(key, ltest_real_tensor, stat)
          call report_test("[Set existing option]", stat /= SPUD_NO_ERROR, .false., "Returned error code when setting option")
      end select

      call test_key_present(key)
      call test_type(key, SPUD_REAL)
      call test_rank(key, 2)
      call test_shape(key, (/size(ltest_real_tensor, 1), size(ltest_real_tensor, 2)/))

      allocate(real_tensor_val(size(ltest_real_tensor, 1), size(ltest_real_tensor, 2)))
      allocate(real_tensor_default(size(ltest_real_tensor, 1), size(ltest_real_tensor, 2)))
      call get_option(key, real_tensor_val, stat)
      call report_test("[Extracted option data]", stat /= SPUD_NO_ERROR, .false., "Returned error code when retrieving option data")
      call report_test("[Extracted correct option data]", maxval(abs(real_tensor_val - ltest_real_tensor)) > tol, .false., "Retrieved incorrect option data")
      call get_option(key, real_tensor_val, stat, default = ltest_real_tensor * 1.1_D)
      call report_test("[Extracted option data with default argument]", stat /= SPUD_NO_ERROR, .false., "Returned error code when retrieving option data")
      call report_test("[Extracted correct option data with default argument]", maxval(abs(real_tensor_val - ltest_real_tensor)) > tol, .false., "Retrieved incorrect option data")
      deallocate(real_tensor_val)
      deallocate(real_tensor_default)
      allocate(real_tensor_val(size(ltest_real_tensor, 1) + 1, size(ltest_real_tensor, 2) + 1))
      allocate(real_tensor_default(size(ltest_real_tensor, 1) + 1, size(ltest_real_tensor, 2) + 1))
      call get_option(key, real_tensor_val, stat)
      call report_test("[Shape error when extracting option data]", stat /= SPUD_SHAPE_ERROR, .false., "Returned error code when retrieving option data")
      call get_option(key, real_tensor_val, stat, default = real_tensor_default)
      call report_test("[Shape error when extracting option data with default argument]", stat /= SPUD_SHAPE_ERROR, .false., "Returned error code when retrieving option data")
      deallocate(real_tensor_val)
      deallocate(real_tensor_default)

      call test_rank_errors_real_scalar(key)
      call test_rank_errors_real_vector(key)
      call test_type_errors_integer(key)
      call test_type_errors_character(key)

      call test_get_character(key // "/__value/rank", "2")
      call test_get_character(key // "/__value/shape", int2str(size(test_real_tensor, 2)) // " " // int2str(size(test_real_tensor, 1)))

    end do

    call test_delete_option(key)

  end subroutine test_set_and_get_real_tensor

  subroutine test_get_integer_scalar(key, test_integer_scalar)
    character(len = *), intent(in) :: key
    integer, intent(in) :: test_integer_scalar

    integer :: integer_scalar_val, stat

    call test_key_present(key)
    call test_type(key, SPUD_INTEGER)
    call test_rank(key, 0)
    call test_shape(key, (/-1, -1/))

    call get_option(key, integer_scalar_val, stat)
    call report_test("[Extracted option data]", stat /= SPUD_NO_ERROR, .false., "Returned error code when retrieving option data")
    call report_test("[Extracted correct option data]", integer_scalar_val /= test_integer_scalar, .false., "Retrieved incorrect option data")
    call get_option(key, integer_scalar_val, stat, default = test_integer_scalar + 1)
    call report_test("[Extracted option data with default argument]", stat /= SPUD_NO_ERROR, .false., "Returned error code when retrieving option data")
    call report_test("[Extracted correct option data with default argument]", integer_scalar_val /= test_integer_scalar, .false., "Retrieved incorrect option data")

    call test_type_errors_real(key)
    call test_rank_errors_integer_vector(key)
    call test_rank_errors_integer_tensor(key)
    call test_type_errors_character(key)

  end subroutine test_get_integer_scalar

  subroutine test_set_and_get_integer_scalar(key, test_integer_scalar)
    character(len = *), intent(in) :: key
    integer, intent(in) :: test_integer_scalar

    integer :: i, ltest_integer_scalar, stat

    call test_key_errors(key)

    do i = 1, 2
      select case(i)
        case(1)
          ltest_integer_scalar = test_integer_scalar
          call set_option(key, ltest_integer_scalar, stat)
          call report_test("[New option]", stat /= SPUD_NEW_KEY_WARNING, .false., "Failed to return new key warning when setting option")
        case default
          ltest_integer_scalar = ltest_integer_scalar + 1
          call set_option(key, ltest_integer_scalar, stat)
          call report_test("[Set existing option]", stat /= SPUD_NO_ERROR, .false., "Returned error code when setting option")
      end select

      call test_get_integer_scalar(key, ltest_integer_scalar)
      call test_get_character(key // "/__value/rank", "0")
      call test_key_errors(key // "/__value/shape")

    end do

    call test_delete_option(key)

  end subroutine test_set_and_get_integer_scalar

  subroutine test_set_and_get_integer_vector(key, test_integer_vector)
    character(len = *), intent(in) :: key
    integer, dimension(:), intent(in) :: test_integer_vector

    integer :: i, stat
    integer, dimension(size(test_integer_vector)) :: ltest_integer_vector
    integer, dimension(:), allocatable :: integer_vector_default, integer_vector_val

    call test_key_errors(key)

    do i = 1, 2
      select case(i)
        case(1)
          ltest_integer_vector = test_integer_vector
          call set_option(key, ltest_integer_vector, stat)
          call report_test("[New option]", stat /= SPUD_NEW_KEY_WARNING, .false., "Failed to return new key warning when setting option")
        case default
          ltest_integer_vector = ltest_integer_vector + 1
          call set_option(key, ltest_integer_vector, stat)
          call report_test("[Set existing option]", stat /= SPUD_NO_ERROR, .false., "Returned error code when setting option")
      end select

      call test_key_present(key)
      call test_type(key, SPUD_INTEGER)
      call test_rank(key, 1)
      call test_shape(key, (/size(ltest_integer_vector), -1/))

      allocate(integer_vector_val(size(ltest_integer_vector)))
      allocate(integer_vector_default(size(ltest_integer_vector)))
      call get_option(key, integer_vector_val, stat)
      call report_test("[Extracted option data]", stat /= SPUD_NO_ERROR, .false., "Returned error code when retrieving option data")
      call report_test("[Extracted correct option data]", count(integer_vector_val /= ltest_integer_vector) > 1, .false., "Retrieved incorrect option data")
      call get_option(key, integer_vector_val, stat, default = ltest_integer_vector + 1)
      call report_test("[Extracted option data with default argument]", stat /= SPUD_NO_ERROR, .false., "Returned error code when retrieving option data")
      call report_test("[Extracted correct option data with default argument]", count(integer_vector_val /= ltest_integer_vector) > 1, .false., "Retrieved incorrect option data")
      deallocate(integer_vector_val)
      deallocate(integer_vector_default)
      allocate(integer_vector_val(size(ltest_integer_vector) + 1))
      allocate(integer_vector_default(size(ltest_integer_vector) + 1))
      call get_option(key, integer_vector_val, stat)
      call report_test("[Shape error when extracting option data]", stat /= SPUD_SHAPE_ERROR, .false., "Returned error code when retrieving option data")
      call get_option(key, integer_vector_val, stat, default = integer_vector_default)
      call report_test("[Shape error when extracting option data with default argument]", stat /= SPUD_SHAPE_ERROR, .false., "Returned error code when retrieving option data")
      deallocate(integer_vector_val)
      deallocate(integer_vector_default)

      call test_type_errors_real(key)
      call test_rank_errors_integer_scalar(key)
      call test_rank_errors_integer_tensor(key)
      call test_type_errors_character(key)

      call test_get_character(key // "/__value/rank", "1")
      call test_get_character(key // "/__value/shape", int2str(size(test_integer_vector)))

    end do

    call test_delete_option(key)

  end subroutine test_set_and_get_integer_vector

  subroutine test_set_and_get_integer_tensor(key, test_integer_tensor)
    character(len = *), intent(in) :: key
    integer, dimension(:, :), intent(in) :: test_integer_tensor

    integer :: i, stat
    integer, dimension(size(test_integer_tensor, 1), size(test_integer_tensor, 2)) :: ltest_integer_tensor
    integer, dimension(:, :), allocatable :: integer_tensor_default, integer_tensor_val

    call test_key_errors(key)

    do i = 1, 2
      select case(i)
        case(1)
          ltest_integer_tensor = test_integer_tensor
          call set_option(key, ltest_integer_tensor, stat)
          call report_test("[New option]", stat /= SPUD_NEW_KEY_WARNING, .false., "Failed to return new key warning when setting option")
        case default
          ltest_integer_tensor = ltest_integer_tensor + 1
          call set_option(key, ltest_integer_tensor, stat)
          call report_test("[Set existing option]", stat /= SPUD_NO_ERROR, .false., "Returned error code when setting option")
      end select

      call test_key_present(key)
      call test_type(key, SPUD_INTEGER)
      call test_rank(key, 2)
      call test_shape(key, (/size(ltest_integer_tensor, 1), size(ltest_integer_tensor, 2)/))

      allocate(integer_tensor_val(size(ltest_integer_tensor, 1), size(ltest_integer_tensor, 2)))
      allocate(integer_tensor_default(size(ltest_integer_tensor, 1), size(ltest_integer_tensor, 2)))
      call get_option(key, integer_tensor_val, stat)
      call report_test("[Extracted option data]", stat /= SPUD_NO_ERROR, .false., "Returned error code when retrieving option data")
      call report_test("[Extracted correct option data]", count(integer_tensor_val /= ltest_integer_tensor) > 0, .false., "Retrieved incorrect option data")
      call get_option(key, integer_tensor_val, stat, default = ltest_integer_tensor + 1)
      call report_test("[Extracted option data with default argument]", stat /= SPUD_NO_ERROR, .false., "Returned error code when retrieving option data")
      call report_test("[Extracted correct option data with default argument]", count(integer_tensor_val /= ltest_integer_tensor) > 0, .false., "Retrieved incorrect option data")
      deallocate(integer_tensor_val)
      deallocate(integer_tensor_default)
      allocate(integer_tensor_val(size(ltest_integer_tensor, 1) + 1, size(ltest_integer_tensor, 2) + 1))
      allocate(integer_tensor_default(size(ltest_integer_tensor, 1) + 1, size(ltest_integer_tensor, 2) + 1))
      call get_option(key, integer_tensor_val, stat)
      call report_test("[Shape error when extracting option data]", stat /= SPUD_SHAPE_ERROR, .false., "Returned error code when retrieving option data")
      call get_option(key, integer_tensor_val, stat, default = integer_tensor_default)
      call report_test("[Shape error when extracting option data with default argument]", stat /= SPUD_SHAPE_ERROR, .false., "Returned error code when retrieving option data")
      deallocate(integer_tensor_val)
      deallocate(integer_tensor_default)

      call test_type_errors_real(key)
      call test_rank_errors_integer_scalar(key)
      call test_rank_errors_integer_vector(key)
      call test_type_errors_character(key)

      call test_get_character(key // "/__value/rank", "2")
      call test_get_character(key // "/__value/shape", int2str(size(test_integer_tensor, 2)) // " " // int2str(size(test_integer_tensor, 1)))

    end do

    call test_delete_option(key)

  end subroutine test_set_and_get_integer_tensor

  subroutine test_get_character(key, test_character)
    character(len = *), intent(in) :: key
    character(len = *), intent(in) :: test_character

    character(len = 0) :: short_character
    character(len = len(test_character)) :: character_val
    integer :: stat

    call test_key_present(key)
    call test_type(key, SPUD_CHARACTER)
    call test_rank(key, 1)
    call test_shape(key, (/len_trim(test_character), -1/))

    call get_option(key, character_val, stat)
    call report_test("[Extracted option data]", stat /= SPUD_NO_ERROR, .false., "Returned error code when retrieving option data")
    call report_test("[Extracted correct option data]", trim(character_val) /= trim(test_character), .false., "Retrieved incorrect option data")
    call get_option(key, character_val, stat, default = trim(test_character) // " Plus One")
    call report_test("[Extracted option data with default argument]", stat /= SPUD_NO_ERROR, .false., "Returned error code when retrieving option data")
    call report_test("[Extracted correct option data with default argument]", trim(character_val) /= trim(test_character), .false., "Retrieved incorrect option data")
    if(len_trim(test_character) > 0) then
      call get_option(key, short_character, stat)
      call report_test("[Shape error when extracting option data]", stat /= SPUD_SHAPE_ERROR, .false., "Returned error code when retrieving option data")
      call get_option(key, short_character, stat, default = "")
      call report_test("[Shape error when extracting option data with default argument]", stat /= SPUD_SHAPE_ERROR, .false., "Returned error code when retrieving option data")
    else
      write(0, *) "Warning: Zero length test character supplied - character shape test skipped"
    end if

    call test_type_errors_real(key)
    call test_type_errors_integer(key)

  end subroutine test_get_character

  subroutine test_set_and_get_character(key, test_character)
    character(len = *), intent(in) :: key
    character(len = *), intent(in) :: test_character

    character(len = len_trim(test_character) + len(" Plus One")) :: ltest_character
    integer :: i, stat

    call test_key_errors(key)

    do i = 1, 2
      select case(i)
        case(1)
          ltest_character = trim(test_character)
          call set_option(key, ltest_character, stat)
          call report_test("[New option]", stat /= SPUD_NEW_KEY_WARNING, .false., "Failed to return new key warning when setting option")
        case default
          ltest_character = trim(ltest_character) // " Plus One"
          call set_option(key, ltest_character, stat)
          call report_test("[Set existing option]", stat /= SPUD_NO_ERROR, .false., "Returned error code when setting option")
      end select

      call test_get_character(key, ltest_character)
      call test_key_errors(key // "/__value/rank")
      call test_key_errors(key // "/__value/shape")

    end do

    call test_delete_option(key)

  end subroutine test_set_and_get_character

  subroutine test_set_and_get_type_none(key)
    character(len = *), intent(in) :: key

    integer :: i, stat

    call test_key_errors(key)

    do i = 1, 2
      select case(i)
        case(1)
          call test_add_new_option(key)
        case default
          call add_option(key, stat)
          call report_test("[Add existing option]", stat /= SPUD_NO_ERROR, .false., "Returned error code when adding option")
      end select

      call test_key_present(key)
      call test_type(key, SPUD_NONE)
      call test_rank(key, -1)
      call test_shape(key, (/-1, -1/))

      call test_type_errors_real(key)
      call test_type_errors_integer(key)
      call test_type_errors_character(key)

      call test_key_errors(key // "/__value/rank")
      call test_key_errors(key // "/__value/shape")

    end do

    call test_delete_option(key)

  end subroutine test_set_and_get_type_none

  subroutine test_named_key(key, name, test_integer)
    character(len = *), intent(in) :: key
    character(len = *), intent(in) :: name
    integer, intent(in) :: test_integer

    character(len = len_trim(name)) :: name_val
    integer :: integer_val, stat

    call set_option(trim(key) // "::" // name, test_integer, stat)
    call report_test("[New option]", stat /= SPUD_NEW_KEY_WARNING, .false., "Failed to return new key warning when setting option")
    call test_key_present(key)
    call report_test("[Add option name]", .not. have_option(key // "::" // name), .false., "Failed to add option name when adding option")

    call get_option(trim(key) // "::" // name, integer_val, stat)
    call report_test("[Extracted option data]", stat /= SPUD_NO_ERROR, .false., "Returned error code when retrieving option data")
    call report_test("[Extracted correct option data]", integer_val /= test_integer, .false., "Retrieved incorrect option data")
    call get_option(trim(key) // "::" // trim(name) // "/name", name_val, stat)
    call report_test("[Extracted option data]", stat /= SPUD_NO_ERROR, .false., "Returned error code when retrieving option data")
    call report_test("[Extracted correct option data]", trim(name_val) /= trim(name), .false., "Retrieved incorrect option data")

    call get_option(trim(key) // "[0]", integer_val, stat)
    call report_test("[Extracted option data]", stat /= SPUD_NO_ERROR, .false., "Returned error code when retrieving option data")
    call report_test("[Extracted correct option data]", integer_val /= test_integer, .false., "Retrieved incorrect option data")
    call get_option(trim(key) // "[0]" // "/name", name_val, stat)
    call report_test("[Extracted option data]", stat /= SPUD_NO_ERROR, .false., "Returned error code when retrieving option data")
    call report_test("[Extracted correct option data]", trim(name_val) /= trim(name), .false., "Retrieved incorrect option data")

    call get_option(trim(key) // "[0]trailingtext", integer_val, stat)
    call report_test("[Key error when extracting option data]", stat /= SPUD_KEY_ERROR, .false., "Returned incorrect error code when retrieving option data")
    call get_option(trim(key) // "[1]", integer_val, stat)
    call report_test("[Key error when extracting option data]", stat /= SPUD_KEY_ERROR, .false., "Returned incorrect error code when retrieving option data")

    call test_delete_option(key)

  end subroutine test_named_key

  subroutine test_indexed_key(key, test_integer)
    character(len = *), intent(in) :: key
    integer, intent(in) :: test_integer

    integer :: integer_val, stat

    call set_option(trim(key) // "[0]", test_integer, stat)
    call report_test("[New option]", stat /= SPUD_NEW_KEY_WARNING, .false., "Failed to return new key warning when setting option")
    call test_key_present(key)

    call set_option(trim(key) // "[1]", test_integer + 1, stat)
    call report_test("[New option]", stat /= SPUD_NEW_KEY_WARNING, .false., "Failed to return new key warning when setting option")
    call test_key_present(key)

    call get_option(trim(key) // "[0]", integer_val, stat)
    call report_test("[Extracted option data]", stat /= SPUD_NO_ERROR, .false., "Returned error code when retrieving option data")
    call report_test("[Extracted correct option data]", integer_val /= test_integer, .false., "Retrieved incorrect option data")

    call get_option(trim(key) // "[1]", integer_val, stat)
    call report_test("[Extracted option data]", stat /= SPUD_NO_ERROR, .false., "Returned error code when retrieving option data")
    call report_test("[Extracted correct option data]", integer_val /= test_integer + 1, .false., "Retrieved incorrect option data")

    call test_delete_option(trim(key) // "[1]")
    call test_delete_option(trim(key) // "[0]")

  end subroutine test_indexed_key

  subroutine test_move_option(key1, key2)
    character(len = *), intent(in) :: key1
    character(len = *), intent(in) :: key2

    integer :: stat

    call add_option(trim(key1), stat)
    call report_test("[New option]", stat /= SPUD_NEW_KEY_WARNING, .false., "Failed to return new key warning when setting option")
    call test_key_present(key1)

    call move_option(trim(key1), trim(key2))
    call test_key_errors(key1)
    call test_key_present(key2)

    call test_delete_option(key2)

    call add_option(trim(key1) // "::name", stat)
    call report_test("[New option]", stat /= SPUD_NEW_KEY_WARNING, .false., "Failed to return new key warning when setting option")
    call test_key_present(key1)
    call test_key_present(key1 // "::name")

    call move_option(trim(key1) // "::name", trim(key2) // "::name")
    call test_key_errors(key1)
    call test_key_errors(key1 // "::name")
    call test_key_present(key2)
    call test_key_present(key2 // "::name")

    call test_delete_option(key2)

    call add_option(trim(key1) // trim(key1), stat)
    call report_test("[New option]", stat /= SPUD_NEW_KEY_WARNING, .false., "Failed to return new key warning when setting option")
    call test_key_present(key1)
    call test_key_present(trim(key1) // key1)

    call move_option(trim(key1) // trim(key1), trim(key2) // trim(key2))
    call test_key_present(key1)
    call test_key_errors(trim(key1) // key1)
    call test_key_present(key2)
    call test_key_present(trim(key2) // key2)

    call test_delete_option(key1)
    call test_delete_option(key2)

  end subroutine test_move_option

  subroutine test_copy_option(key1, key2)
    character(len = *), intent(in) :: key1
    character(len = *), intent(in) :: key2

    integer :: stat

    call add_option(trim(key1), stat)
    call report_test("[New option]", stat /= SPUD_NEW_KEY_WARNING, .false., "Failed to return new key warning when setting option")
    call test_key_present(key1)

    call copy_option(trim(key1), trim(key2))
    call test_key_present(key1)
    call test_key_present(key2)

    call test_delete_option(key2)
    call test_delete_option(key1)

    call add_option(trim(key1) // "::name", stat)
    call report_test("[New option]", stat /= SPUD_NEW_KEY_WARNING, .false., "Failed to return new key warning when setting option")
    call test_key_present(key1)
    call test_key_present(key1 // "::name")

    call copy_option(trim(key1) // "::name", trim(key2) // "::name")
    call test_key_present(key2)
    call test_key_present(key2 // "::name")
    call test_key_present(key1)
    call test_key_present(key1 // "::name")

    call test_delete_option(key2)
    call test_delete_option(key1)

    call add_option(trim(key1) // trim(key1), stat)
    call report_test("[New option]", stat /= SPUD_NEW_KEY_WARNING, .false., "Failed to return new key warning when setting option")
    call test_key_present(key1)
    call test_key_present(trim(key1) // key1)

    call copy_option(trim(key1) // trim(key1), trim(key2) // trim(key2))
    call test_key_present(key1)
    call test_key_present(trim(key1) // key1)
    call test_key_present(key2)
    call test_key_present(trim(key2) // key2)

    call test_delete_option(key1)
    call test_delete_option(key2)

  end subroutine test_copy_option

end subroutine test_fspud
