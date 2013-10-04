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

module futils
  !!< Some generic fortran utility functions.

  use fldebug
  use global_parameters, only : real_digits_10

  implicit none

  interface real_format_len
    module procedure real_format_non_padded_len, real_format_padded_len
  end interface

  interface real_format
    module procedure real_format_non_padded, real_format_padded
  end interface real_format

  interface nullify
    module procedure nullify_integer_vector, nullify_integer_vector_vector, &
      & nullify_real_vector, nullify_real_vector_vector
  end interface nullify

  type real_vector
#ifdef DDEBUG
     real, dimension(:), pointer :: ptr=>null()
#else
     real, dimension(:), pointer :: ptr
#endif
  end type real_vector

  type real_matrix
#ifdef DDEBUG
     real, dimension(:,:), pointer :: ptr=>null()
#else
     real, dimension(:,:), pointer :: ptr
#endif
  end type real_matrix

  type integer_vector
#ifdef DDEBUG
     integer, dimension(:), pointer :: ptr=>null()
#else
     integer, dimension(:), pointer :: ptr
#endif
  end type integer_vector

contains

  subroutine nullify_integer_vector(vector)
    type(integer_vector), intent(inout) :: vector

    vector%ptr => null()

  end subroutine nullify_integer_vector

  subroutine nullify_integer_vector_vector(vector)
    type(integer_vector), dimension(:), intent(inout) :: vector

    integer :: i

    do i = 1, size(vector)
      vector(i)%ptr => null()
    end do

  end subroutine nullify_integer_vector_vector

  subroutine nullify_real_vector(vector)
    type(real_vector), intent(inout) :: vector

    vector%ptr => null()

  end subroutine nullify_real_vector

  subroutine nullify_real_vector_vector(vector)
    type(real_vector), dimension(:), intent(inout) :: vector

    integer :: i

    do i = 1, size(vector)
      vector(i)%ptr => null()
    end do

  end subroutine nullify_real_vector_vector

  pure function present_and_true(flag)
    logical :: present_and_true
    logical, intent(in), optional :: flag

    if (present(flag)) then
       present_and_true=flag
    else
       present_and_true=.false.
    end if

  end function present_and_true

  pure function present_and_false(flag)
    logical :: present_and_false
    logical, intent(in), optional :: flag

    if (present(flag)) then
       present_and_false=.not.flag
    else
       present_and_false=.false.
    end if

  end function present_and_false

  pure function present_and_zero(var)
    integer, optional, intent(in) :: var

    logical :: present_and_zero

    if(present(var)) then
      present_and_zero = (var == 0)
    else
      present_and_zero = .false.
    end if

  end function present_and_zero

  pure function present_and_nonzero(var)
    integer, optional, intent(in) :: var

    logical :: present_and_nonzero

    if(present(var)) then
      present_and_nonzero = (var /= 0)
    else
      present_and_nonzero = .false.
    end if

  end function present_and_nonzero

  pure function present_and_nonempty(var)
    character(len = *), optional, intent(in) :: var

    logical :: present_and_nonempty

    if(present(var)) then
      present_and_nonempty = (len_trim(var) > 0)
    else
      present_and_nonempty = .false.
    end if

  end function present_and_nonempty

  function free_unit()
    !!< Find a free unit number. Start from unit 10 in order to ensure that
    !!< we skip any preconnected units which may not be correctly identified
    !!< on some compilers.
    integer :: free_unit

    logical :: connected

    do free_unit=10, 99

       inquire(unit=free_unit, opened=connected)

       if (.not.connected) return

    end do

    FLAbort("No free unit numbers avalable")

  end function

  pure function real_format_non_padded_len() result(length)
    !!< Return the length of the format string for real data, without
    !!< padding characters

    integer :: length

    length = real_format_len(0)

  end function real_format_non_padded_len

  pure function real_format_padded_len(padding) result(length)
    !!< Return the length of the format string for real data, with
    !!< padding characters

    integer, intent(in) :: padding

    integer :: length

    ! See real_format comment
    length = 2 + int2str_len(real_digits_10 + 10 + padding) + int2str_len(real_digits_10 + 2) + 1 + int2str_len(3)

  end function real_format_padded_len

  function real_format_non_padded() result(format)
    !!< Return a format string for real data, without padding characters

    character(len = real_format_len()) :: format

    format = real_format(0)

  end function real_format_non_padded

  function real_format_padded(padding) result(format)
    !!< Return a format string for real data, with padding characters

    integer, intent(in) :: padding

    character(len = real_format_len()) :: format

    ! Construct:
    !    (real_digits_10 + 2) + 8 + padding . (real_digits_10 + 2) e3
    ! (real_digits_10 + 2) seems to give sufficient digits to preserve
    ! (1.0 + epsilon(1.0) > 1.0) in double precision writes. "8" (before ".")
    ! is the minimum number of additional characters allowing a general real
    ! to be written. 3 (after "e") is the minimum number of characters allowing
    ! a general real exponent to be written.
    format = "e" // int2str(real_digits_10 + 10 + padding) // "." // int2str(real_digits_10 + 2) // "e" // int2str(3)

  end function real_format_padded

  pure function nth_digit(number, digit)
    !!< Return the nth digit of number. Useful for those infernal
    !!< overloaded options.
    !!<
    !!< Digits are counted from the RIGHT.
    integer :: nth_digit
    integer, intent(in) :: number, digit

    ! The divisions strip the trailing digits while the mod strips the
    ! leading digits.
    nth_digit=mod(abs(number)/10**(digit-1), 10)

  end function nth_digit

  pure function count_chars(string, sep)
    character(len=*), intent(in) :: string
    character(len=1), intent(in) :: sep
    integer :: count_chars
    integer :: i

    count_chars = 0

    do i=1,len(string)
      if (string(i:i) == sep) then
        count_chars = count_chars + 1
      end if
    end do
  end function count_chars

  function multiindex(string, sep)
    character(len=*), intent(in) :: string
    character(len=1), intent(in) :: sep
    integer :: i, j
    integer, dimension(count_chars(string, sep)) :: multiindex

    multiindex=0
    j=0

    do i=1,len(string)
      if (string(i:i) == sep) then
        j = j + 1
        multiindex(j) = i
      end if
    end do
  end function multiindex

  pure function file_extension_len(filename) result(length)
    !!< Return the length of the file extension of the supplied filename
    !!< (including the ".")

    character(len = *), intent(in) :: filename

    integer :: length

    length = len(filename) - trim_file_extension_len(filename)

  end function file_extension_len

  function file_extension(filename)
    !!< Return the file extension of the supplied filename (including the ".")

    character(len = *), intent(in) :: filename

    character(len = file_extension_len(filename)) :: file_extension

    file_extension = filename(len(filename) - len(file_extension) + 1:)

  end function file_extension

  pure function trim_file_extension_len(filename) result(length)
    !!< Return the length of the supplied filename minus the file extension

    character(len = *), intent(in) :: filename

    integer :: length

    do length = len_trim(filename), 1, -1
      if(filename(length:length) == ".") then
        exit
      end if
    end do

    length = length - 1

  end function trim_file_extension_len

  function trim_file_extension(filename)
    !!< Trim the file extension from the supplied filename

    character(len = *), intent(in) :: filename

    character(len = trim_file_extension_len(filename)) :: trim_file_extension

    trim_file_extension = filename(:len(trim_file_extension))

  end function trim_file_extension

  function random_number_minmax(min, max) result(rand)
    real, intent(in) :: min, max
    real :: rand

    call random_number(rand)
    rand = (max - min) * rand
    rand = rand + min
  end function random_number_minmax

  pure function int2str_len(i)

    !!< Count number of digits in i.

    integer, intent(in) :: i
    integer :: int2str_len

    if(i==0) then
       int2str_len=1
    else if (i>0) then
       int2str_len = floor(log10(real(i)))+1
    else
       int2str_len = floor(log10(abs(real(i))))+2
    end if

  end function int2str_len

  function int2str (i)

    !!< Convert integer i into a string.
    !!< This should only be used when forming option strings.

    integer, intent(in) :: i
    character(len=int2str_len(i)) :: int2str

    write(int2str,"(i0)") i

  end function int2str

  pure function starts_with(string, start)
    !!< Auxillary function, returns .true. if 'string' starts with 'start'
    logical :: starts_with

    character(len=*), intent(in):: string, start

    if (len(start)>len(string)) then
      starts_with=.false.
    else
      starts_with= string(1:len(start))==start
    end if

  end function starts_with

  subroutine tokenize(string, tokens, delimiter)
    !!< Split the supplied string with the supplied delimiter. tokens is
    !!< allocated by this routine. Note that the whole of string is parsed and
    !!< compared with the whole of delimiter - it is the callers responsibility
    !!< to do any necessary trimming.

    character(len = *), intent(in) :: string
    character(len = *), dimension(:), allocatable, intent(out) :: tokens
    character(len = *), intent(in) :: delimiter

    integer :: end_index, i, tokens_size, start_index

    tokens_size = 1
    do i = 1, len(string) - len(delimiter) + 1
      if(string(i:i + len(delimiter) - 1) == delimiter) then
        tokens_size = tokens_size + 1
      end if
    end do
    allocate(tokens(tokens_size))

    start_index = 1
    end_index = -len(delimiter)
    do i = 1, tokens_size
      if(i == tokens_size) then
        end_index = len(string)
      else
        end_index = end_index + index(string(start_index:), delimiter) + len(delimiter) - 1
      end if
      tokens(i) = string(start_index:end_index)
      start_index = end_index + len(delimiter) + 1
      tokens(i) = adjustl(tokens(i))
    end do
    assert(start_index == len(string) + 1 + len(delimiter))

  end subroutine tokenize

end module futils
