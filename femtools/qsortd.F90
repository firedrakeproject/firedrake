#include "fdebug.h"

module quicksort

  use fldebug
  use global_parameters, only : real_4

  implicit none

  private

  public :: qsort, sort, count_unique, inverse_permutation,&
       & apply_permutation, apply_reverse_permutation, sorted

  interface qsort
    module procedure qsortd, qsortsp, qsorti
  end interface qsort

  interface sorted
     module procedure sortedi, sortedi_key
  end interface sorted

  interface sort
    module procedure sort_integer_array, sort_real_array
  end interface

  interface apply_permutation
    module procedure apply_permutation_integer_array, &
      & apply_permutation_real_array, apply_permutation_integer, &
      & apply_permutation_real
  end interface apply_permutation

  interface apply_reverse_permutation
    module procedure apply_reverse_permutation_real, apply_reverse_permutation_integer
  end interface apply_reverse_permutation

contains

! Retrieved from:
! http://users.bigpond.net.au/amiller/NSWC/qsortd.f90
! Believed to be public domain as it is the work of
! a US government employee.

SUBROUTINE qsortd(x, ind)

! Code converted using TO_F90 by Alan Miller
! Date: 2002-12-18  Time: 11:55:47

IMPLICIT NONE
INTEGER, PARAMETER  :: dp = SELECTED_REAL_KIND(12, 60)

REAL (dp), INTENT(IN)  :: x(:)
INTEGER, INTENT(OUT)   :: ind(:)
INTEGER :: n

!***************************************************************************

!                                                         ROBERT RENKA
!                                                 OAK RIDGE NATL. LAB.

!   THIS SUBROUTINE USES AN ORDER N*LOG(N) QUICK SORT TO SORT A REAL (dp)
! ARRAY X INTO INCREASING ORDER.  THE ALGORITHM IS AS FOLLOWS.  IND IS
! INITIALIZED TO THE ORDERED SEQUENCE OF INDICES 1,...,N, AND ALL INTERCHANGES
! ARE APPLIED TO IND.  X IS DIVIDED INTO TWO PORTIONS BY PICKING A CENTRAL
! ELEMENT T.  THE FIRST AND LAST ELEMENTS ARE COMPARED WITH T, AND
! INTERCHANGES ARE APPLIED AS NECESSARY SO THAT THE THREE VALUES ARE IN
! ASCENDING ORDER.  INTERCHANGES ARE THEN APPLIED SO THAT ALL ELEMENTS
! GREATER THAN T ARE IN THE UPPER PORTION OF THE ARRAY AND ALL ELEMENTS
! LESS THAN T ARE IN THE LOWER PORTION.  THE UPPER AND LOWER INDICES OF ONE
! OF THE PORTIONS ARE SAVED IN LOCAL ARRAYS, AND THE PROCESS IS REPEATED
! ITERATIVELY ON THE OTHER PORTION.  WHEN A PORTION IS COMPLETELY SORTED,
! THE PROCESS BEGINS AGAIN BY RETRIEVING THE INDICES BOUNDING ANOTHER
! UNSORTED PORTION.

! INPUT PARAMETERS -   N - LENGTH OF THE ARRAY X.

!                      X - VECTOR OF LENGTH N TO BE SORTED.

!                    IND - VECTOR OF LENGTH >= N.

! N AND X ARE NOT ALTERED BY THIS ROUTINE.

! OUTPUT PARAMETER - IND - SEQUENCE OF INDICES 1,...,N PERMUTED IN THE SAME
!                          FASHION AS X WOULD BE.  THUS, THE ORDERING ON
!                          X IS DEFINED BY Y(I) = X(IND(I)).

!*********************************************************************

! NOTE -- IU AND IL MUST BE DIMENSIONED >= LOG(N) WHERE LOG HAS BASE 2.

!*********************************************************************

INTEGER   :: iu(21), il(21)
INTEGER   :: m, i, j, k, l, ij, it, itt, indx
REAL      :: r
REAL (dp) :: t

n = size(x)

! LOCAL PARAMETERS -

! IU,IL =  TEMPORARY STORAGE FOR THE UPPER AND LOWER
!            INDICES OF PORTIONS OF THE ARRAY X
! M =      INDEX FOR IU AND IL
! I,J =    LOWER AND UPPER INDICES OF A PORTION OF X
! K,L =    INDICES IN THE RANGE I,...,J
! IJ =     RANDOMLY CHOSEN INDEX BETWEEN I AND J
! IT,ITT = TEMPORARY STORAGE FOR INTERCHANGES IN IND
! INDX =   TEMPORARY INDEX FOR X
! R =      PSEUDO RANDOM NUMBER FOR GENERATING IJ
! T =      CENTRAL ELEMENT OF X

IF (n <= 0) RETURN

! INITIALIZE IND, M, I, J, AND R

DO  i = 1, n
  ind(i) = i
END DO
m = 1
i = 1
j = n
r = .375

! TOP OF LOOP

20 IF (i >= j) GO TO 70
IF (r <= .5898437) THEN
  r = r + .0390625
ELSE
  r = r - .21875
END IF

! INITIALIZE K

30 k = i

! SELECT A CENTRAL ELEMENT OF X AND SAVE IT IN T

ij = i + r*(j-i)
it = ind(ij)
t = x(it)

! IF THE FIRST ELEMENT OF THE ARRAY IS GREATER THAN T,
!   INTERCHANGE IT WITH T

indx = ind(i)
IF (x(indx) > t) THEN
  ind(ij) = indx
  ind(i) = it
  it = indx
  t = x(it)
END IF

! INITIALIZE L

l = j

! IF THE LAST ELEMENT OF THE ARRAY IS LESS THAN T,
!   INTERCHANGE IT WITH T

indx = ind(j)
IF (x(indx) >= t) GO TO 50
ind(ij) = indx
ind(j) = it
it = indx
t = x(it)

! IF THE FIRST ELEMENT OF THE ARRAY IS GREATER THAN T,
!   INTERCHANGE IT WITH T

indx = ind(i)
IF (x(indx) <= t) GO TO 50
ind(ij) = indx
ind(i) = it
it = indx
t = x(it)
GO TO 50

! INTERCHANGE ELEMENTS K AND L

40 itt = ind(l)
ind(l) = ind(k)
ind(k) = itt

! FIND AN ELEMENT IN THE UPPER PART OF THE ARRAY WHICH IS
!   NOT LARGER THAN T

50 l = l - 1
indx = ind(l)
IF (x(indx) > t) GO TO 50

! FIND AN ELEMENT IN THE LOWER PART OF THE ARRAY WHCIH IS NOT SMALLER THAN T

60 k = k + 1
indx = ind(k)
IF (x(indx) < t) GO TO 60

! IF K <= L, INTERCHANGE ELEMENTS K AND L

IF (k <= l) GO TO 40

! SAVE THE UPPER AND LOWER SUBSCRIPTS OF THE PORTION OF THE
!   ARRAY YET TO BE SORTED

IF (l-i > j-k) THEN
  il(m) = i
  iu(m) = l
  i = k
  m = m + 1
  GO TO 80
END IF

il(m) = k
iu(m) = j
j = l
m = m + 1
GO TO 80

! BEGIN AGAIN ON ANOTHER UNSORTED PORTION OF THE ARRAY

70 m = m - 1
IF (m == 0) RETURN
i = il(m)
j = iu(m)

80 IF (j-i >= 11) GO TO 30
IF (i == 1) GO TO 20
i = i - 1

! SORT ELEMENTS I+1,...,J.  NOTE THAT 1 <= I < J AND J-I < 11.

90 i = i + 1
IF (i == j) GO TO 70
indx = ind(i+1)
t = x(indx)
it = indx
indx = ind(i)
IF (x(indx) <= t) GO TO 90
k = i

100 ind(k+1) = ind(k)
k = k - 1
indx = ind(k)
IF (t < x(indx)) GO TO 100

ind(k+1) = it
GO TO 90
END SUBROUTINE qsortd

SUBROUTINE qsortsp(x, ind)

! Code converted using TO_F90 by Alan Miller
! Date: 2002-12-18  Time: 11:55:47

IMPLICIT NONE

REAL (kind = real_4), INTENT(IN)  :: x(:)
INTEGER, INTENT(OUT)   :: ind(:)
INTEGER :: n

!***************************************************************************

!                                                         ROBERT RENKA
!                                                 OAK RIDGE NATL. LAB.

!   THIS SUBROUTINE USES AN ORDER N*LOG(N) QUICK SORT TO SORT A REAL (dp)
! ARRAY X INTO INCREASING ORDER.  THE ALGORITHM IS AS FOLLOWS.  IND IS
! INITIALIZED TO THE ORDERED SEQUENCE OF INDICES 1,...,N, AND ALL INTERCHANGES
! ARE APPLIED TO IND.  X IS DIVIDED INTO TWO PORTIONS BY PICKING A CENTRAL
! ELEMENT T.  THE FIRST AND LAST ELEMENTS ARE COMPARED WITH T, AND
! INTERCHANGES ARE APPLIED AS NECESSARY SO THAT THE THREE VALUES ARE IN
! ASCENDING ORDER.  INTERCHANGES ARE THEN APPLIED SO THAT ALL ELEMENTS
! GREATER THAN T ARE IN THE UPPER PORTION OF THE ARRAY AND ALL ELEMENTS
! LESS THAN T ARE IN THE LOWER PORTION.  THE UPPER AND LOWER INDICES OF ONE
! OF THE PORTIONS ARE SAVED IN LOCAL ARRAYS, AND THE PROCESS IS REPEATED
! ITERATIVELY ON THE OTHER PORTION.  WHEN A PORTION IS COMPLETELY SORTED,
! THE PROCESS BEGINS AGAIN BY RETRIEVING THE INDICES BOUNDING ANOTHER
! UNSORTED PORTION.

! INPUT PARAMETERS -   N - LENGTH OF THE ARRAY X.

!                      X - VECTOR OF LENGTH N TO BE SORTED.

!                    IND - VECTOR OF LENGTH >= N.

! N AND X ARE NOT ALTERED BY THIS ROUTINE.

! OUTPUT PARAMETER - IND - SEQUENCE OF INDICES 1,...,N PERMUTED IN THE SAME
!                          FASHION AS X WOULD BE.  THUS, THE ORDERING ON
!                          X IS DEFINED BY Y(I) = X(IND(I)).

!*********************************************************************

! NOTE -- IU AND IL MUST BE DIMENSIONED >= LOG(N) WHERE LOG HAS BASE 2.

!*********************************************************************

INTEGER   :: iu(21), il(21)
INTEGER   :: m, i, j, k, l, ij, it, itt, indx
REAL      :: r
REAL (kind = real_4) :: t

n = size(x)

! LOCAL PARAMETERS -

! IU,IL =  TEMPORARY STORAGE FOR THE UPPER AND LOWER
!            INDICES OF PORTIONS OF THE ARRAY X
! M =      INDEX FOR IU AND IL
! I,J =    LOWER AND UPPER INDICES OF A PORTION OF X
! K,L =    INDICES IN THE RANGE I,...,J
! IJ =     RANDOMLY CHOSEN INDEX BETWEEN I AND J
! IT,ITT = TEMPORARY STORAGE FOR INTERCHANGES IN IND
! INDX =   TEMPORARY INDEX FOR X
! R =      PSEUDO RANDOM NUMBER FOR GENERATING IJ
! T =      CENTRAL ELEMENT OF X

IF (n <= 0) RETURN

! INITIALIZE IND, M, I, J, AND R

DO  i = 1, n
  ind(i) = i
END DO
m = 1
i = 1
j = n
r = .375

! TOP OF LOOP

20 IF (i >= j) GO TO 70
IF (r <= .5898437) THEN
  r = r + .0390625
ELSE
  r = r - .21875
END IF

! INITIALIZE K

30 k = i

! SELECT A CENTRAL ELEMENT OF X AND SAVE IT IN T

ij = i + r*(j-i)
it = ind(ij)
t = x(it)

! IF THE FIRST ELEMENT OF THE ARRAY IS GREATER THAN T,
!   INTERCHANGE IT WITH T

indx = ind(i)
IF (x(indx) > t) THEN
  ind(ij) = indx
  ind(i) = it
  it = indx
  t = x(it)
END IF

! INITIALIZE L

l = j

! IF THE LAST ELEMENT OF THE ARRAY IS LESS THAN T,
!   INTERCHANGE IT WITH T

indx = ind(j)
IF (x(indx) >= t) GO TO 50
ind(ij) = indx
ind(j) = it
it = indx
t = x(it)

! IF THE FIRST ELEMENT OF THE ARRAY IS GREATER THAN T,
!   INTERCHANGE IT WITH T

indx = ind(i)
IF (x(indx) <= t) GO TO 50
ind(ij) = indx
ind(i) = it
it = indx
t = x(it)
GO TO 50

! INTERCHANGE ELEMENTS K AND L

40 itt = ind(l)
ind(l) = ind(k)
ind(k) = itt

! FIND AN ELEMENT IN THE UPPER PART OF THE ARRAY WHICH IS
!   NOT LARGER THAN T

50 l = l - 1
indx = ind(l)
IF (x(indx) > t) GO TO 50

! FIND AN ELEMENT IN THE LOWER PART OF THE ARRAY WHCIH IS NOT SMALLER THAN T

60 k = k + 1
indx = ind(k)
IF (x(indx) < t) GO TO 60

! IF K <= L, INTERCHANGE ELEMENTS K AND L

IF (k <= l) GO TO 40

! SAVE THE UPPER AND LOWER SUBSCRIPTS OF THE PORTION OF THE
!   ARRAY YET TO BE SORTED

IF (l-i > j-k) THEN
  il(m) = i
  iu(m) = l
  i = k
  m = m + 1
  GO TO 80
END IF

il(m) = k
iu(m) = j
j = l
m = m + 1
GO TO 80

! BEGIN AGAIN ON ANOTHER UNSORTED PORTION OF THE ARRAY

70 m = m - 1
IF (m == 0) RETURN
i = il(m)
j = iu(m)

80 IF (j-i >= 11) GO TO 30
IF (i == 1) GO TO 20
i = i - 1

! SORT ELEMENTS I+1,...,J.  NOTE THAT 1 <= I < J AND J-I < 11.

90 i = i + 1
IF (i == j) GO TO 70
indx = ind(i+1)
t = x(indx)
it = indx
indx = ind(i)
IF (x(indx) <= t) GO TO 90
k = i

100 ind(k+1) = ind(k)
k = k - 1
indx = ind(k)
IF (t < x(indx)) GO TO 100

ind(k+1) = it
GO TO 90
END SUBROUTINE qsortsp

SUBROUTINE qsorti(x, ind)

! Code converted using TO_F90 by Alan Miller
! Date: 2002-12-18  Time: 11:55:47

IMPLICIT NONE

INTEGER, INTENT(IN)  :: x(:)
INTEGER, INTENT(OUT)   :: ind(:)
INTEGER :: n

!***************************************************************************

!                                                         ROBERT RENKA
!                                                 OAK RIDGE NATL. LAB.

!   THIS SUBROUTINE USES AN ORDER N*LOG(N) QUICK SORT TO SORT AN INTEGER
! ARRAY X INTO INCREASING ORDER.  THE ALGORITHM IS AS FOLLOWS.  IND IS
! INITIALIZED TO THE ORDERED SEQUENCE OF INDICES 1,...,N, AND ALL INTERCHANGES
! ARE APPLIED TO IND.  X IS DIVIDED INTO TWO PORTIONS BY PICKING A CENTRAL
! ELEMENT T.  THE FIRST AND LAST ELEMENTS ARE COMPARED WITH T, AND
! INTERCHANGES ARE APPLIED AS NECESSARY SO THAT THE THREE VALUES ARE IN
! ASCENDING ORDER.  INTERCHANGES ARE THEN APPLIED SO THAT ALL ELEMENTS
! GREATER THAN T ARE IN THE UPPER PORTION OF THE ARRAY AND ALL ELEMENTS
! LESS THAN T ARE IN THE LOWER PORTION.  THE UPPER AND LOWER INDICES OF ONE
! OF THE PORTIONS ARE SAVED IN LOCAL ARRAYS, AND THE PROCESS IS REPEATED
! ITERATIVELY ON THE OTHER PORTION.  WHEN A PORTION IS COMPLETELY SORTED,
! THE PROCESS BEGINS AGAIN BY RETRIEVING THE INDICES BOUNDING ANOTHER
! UNSORTED PORTION.

! INPUT PARAMETERS -   N - LENGTH OF THE ARRAY X.

!                      X - VECTOR OF LENGTH N TO BE SORTED.

!                    IND - VECTOR OF LENGTH >= N.

! N AND X ARE NOT ALTERED BY THIS ROUTINE.

! OUTPUT PARAMETER - IND - SEQUENCE OF INDICES 1,...,N PERMUTED IN THE SAME
!                          FASHION AS X WOULD BE.  THUS, THE ORDERING ON
!                          X IS DEFINED BY Y(I) = X(IND(I)).

!*********************************************************************

! NOTE -- IU AND IL MUST BE DIMENSIONED >= LOG(N) WHERE LOG HAS BASE 2.

!*********************************************************************

INTEGER   :: iu(21), il(21)
INTEGER   :: m, i, j, k, l, ij, it, itt, indx
REAL      :: r
INTEGER   :: t

n = size(x)

! LOCAL PARAMETERS -

! IU,IL =  TEMPORARY STORAGE FOR THE UPPER AND LOWER
!            INDICES OF PORTIONS OF THE ARRAY X
! M =      INDEX FOR IU AND IL
! I,J =    LOWER AND UPPER INDICES OF A PORTION OF X
! K,L =    INDICES IN THE RANGE I,...,J
! IJ =     RANDOMLY CHOSEN INDEX BETWEEN I AND J
! IT,ITT = TEMPORARY STORAGE FOR INTERCHANGES IN IND
! INDX =   TEMPORARY INDEX FOR X
! R =      PSEUDO RANDOM NUMBER FOR GENERATING IJ
! T =      CENTRAL ELEMENT OF X

IF (n <= 0) RETURN

! INITIALIZE IND, M, I, J, AND R

DO  i = 1, n
  ind(i) = i
END DO
m = 1
i = 1
j = n
r = .375

! TOP OF LOOP

20 IF (i >= j) GO TO 70
IF (r <= .5898437) THEN
  r = r + .0390625
ELSE
  r = r - .21875
END IF

! INITIALIZE K

30 k = i

! SELECT A CENTRAL ELEMENT OF X AND SAVE IT IN T

ij = i + r*(j-i)
it = ind(ij)
t = x(it)

! IF THE FIRST ELEMENT OF THE ARRAY IS GREATER THAN T,
!   INTERCHANGE IT WITH T

indx = ind(i)
IF (x(indx) > t) THEN
  ind(ij) = indx
  ind(i) = it
  it = indx
  t = x(it)
END IF

! INITIALIZE L

l = j

! IF THE LAST ELEMENT OF THE ARRAY IS LESS THAN T,
!   INTERCHANGE IT WITH T

indx = ind(j)
IF (x(indx) >= t) GO TO 50
ind(ij) = indx
ind(j) = it
it = indx
t = x(it)

! IF THE FIRST ELEMENT OF THE ARRAY IS GREATER THAN T,
!   INTERCHANGE IT WITH T

indx = ind(i)
IF (x(indx) <= t) GO TO 50
ind(ij) = indx
ind(i) = it
it = indx
t = x(it)
GO TO 50

! INTERCHANGE ELEMENTS K AND L

40 itt = ind(l)
ind(l) = ind(k)
ind(k) = itt

! FIND AN ELEMENT IN THE UPPER PART OF THE ARRAY WHICH IS
!   NOT LARGER THAN T

50 l = l - 1
indx = ind(l)
IF (x(indx) > t) GO TO 50

! FIND AN ELEMENT IN THE LOWER PART OF THE ARRAY WHCIH IS NOT SMALLER THAN T

60 k = k + 1
indx = ind(k)
IF (x(indx) < t) GO TO 60

! IF K <= L, INTERCHANGE ELEMENTS K AND L

IF (k <= l) GO TO 40

! SAVE THE UPPER AND LOWER SUBSCRIPTS OF THE PORTION OF THE
!   ARRAY YET TO BE SORTED

IF (l-i > j-k) THEN
  il(m) = i
  iu(m) = l
  i = k
  m = m + 1
  GO TO 80
END IF

il(m) = k
iu(m) = j
j = l
m = m + 1
GO TO 80

! BEGIN AGAIN ON ANOTHER UNSORTED PORTION OF THE ARRAY

70 m = m - 1
IF (m == 0) RETURN
i = il(m)
j = iu(m)

80 IF (j-i >= 11) GO TO 30
IF (i == 1) GO TO 20
i = i - 1

! SORT ELEMENTS I+1,...,J.  NOTE THAT 1 <= I < J AND J-I < 11.

90 i = i + 1
IF (i == j) GO TO 70
indx = ind(i+1)
t = x(indx)
it = indx
indx = ind(i)
IF (x(indx) <= t) GO TO 90
k = i

100 ind(k+1) = ind(k)
k = k - 1
indx = ind(k)
IF (t < x(indx)) GO TO 100

ind(k+1) = it
GO TO 90
END SUBROUTINE qsorti

  recursive subroutine sort_integer_array(integer_array, permutation)
    !!< Sort integer_array along integer_array(:, 1), then integer_array(:, 2), etc.

    integer, dimension(:, :), intent(in) :: integer_array
    integer, dimension(size(integer_array, 1)), intent(out) :: permutation

    integer :: end_index, i, j, start_index
    integer, dimension(:), allocatable :: sub_permutation
    logical :: do_sub_permute
    integer, dimension(:, :), allocatable :: sorted_integer_array

    permutation = 0

    if(size(integer_array, 2) == 1) then
      ! Terminating case

      ! Sort along integer_array(:, 1)
      call qsort(integer_array(:, 1), permutation)
    else
      ! Recursing case

      ! Sort along integer_array(:, 1)
      call qsort(integer_array(:, 1), permutation)
      ! Now we need to sort equal consecutive entries in
      ! integer_array(permutation, 1) using integer_array(permutation, 2:)
      start_index = -1  ! When this is > 0, it indicates we're iterating over
                        ! equal consecutive entries in
                        ! integer_array(permutation, 1)
      do i = 2, size(integer_array, 1) + 1
        if(start_index < 0) then
          ! We haven't yet found equal consecutive entries in
          ! integer_array(permutation, 1) over which to sort

          if(i <= size(integer_array, 1)) then
            ! We're not yet at the end of the array
            if(abs(integer_array(permutation(i), 1) - integer_array(permutation(i - 1), 1)) == 0) then
              ! We've found equal entries in integer_array(permutation, 1) - this
              ! gives us a start index over which to sort
              start_index = i - 1
            end if
          end if
        else
          ! We're already iterating over equal entries in
          ! integer_array(permutation, 1)

          ! We've found an end index over which to sort if ...
          ! ... we're at the end of the array ...
          do_sub_permute = i == size(integer_array, 1) + 1
          if(.not. do_sub_permute) then
            ! ... or we've hit non-equal consecutive entries
            do_sub_permute = abs(integer_array(permutation(i), 1) - integer_array(permutation(i - 1), 1)) > 0
          end if
          if(do_sub_permute) then
            ! We've found an end index
            end_index = i - 1

            ! Sort using integer_array(permutation(start_index:end_index), 2:)
            allocate(sorted_integer_array(end_index - start_index + 1, size(integer_array, 2) - 1))
            do j = 1, size(sorted_integer_array, 1)
              assert(permutation(j + start_index - 1) >= 1)
              assert(permutation(j + start_index - 1) <= size(integer_array, 1))
              sorted_integer_array(j, :) = integer_array(permutation(j + start_index - 1), 2:)
            end do
            allocate(sub_permutation(end_index - start_index + 1))
            call sort(sorted_integer_array, sub_permutation)
            call apply_permutation(permutation(start_index:end_index), sub_permutation)
            deallocate(sub_permutation)
            deallocate(sorted_integer_array)

            ! Now we need to find a new start index
            start_index = -1
          end if
        end if
      end do
    end if

  end subroutine sort_integer_array

  recursive subroutine sort_real_array(real_array, permutation)
    !!< Sort real_array along real_array(:, 1), then real_array(:, 2), etc.

    real, dimension(:, :), intent(in) :: real_array
    integer, dimension(size(real_array, 1)), intent(out) :: permutation

    integer :: end_index, i, j, start_index
    integer, dimension(:), allocatable :: sub_permutation
    logical :: do_sub_permute
    real, dimension(:, :), allocatable :: sorted_real_array

    permutation = 0

    if(size(real_array, 2) == 1) then
      ! Terminating case

      ! Sort along real_array(:, 1)
      call qsort(real_array(:, 1), permutation)
    else
      ! Recursing case

      ! Sort along real_array(:, 1)
      call qsort(real_array(:, 1), permutation)
      ! Now we need to sort equal consecutive entries in
      ! real_array(permutation, 1) using real_array(permutation, 2:)
      start_index = -1  ! When this is > 0, it indicates we're iterating over
                        ! equal consecutive entries in
                        ! real_array(permutation, 1)
      do i = 2, size(real_array, 1) + 1
        if(start_index < 0) then
          ! We haven't yet found equal consecutive entries in
          ! real_array(permutation, 1) over which to sort

          if(i <= size(real_array, 1)) then
            ! We're not yet at the end of the array
            if(abs(real_array(permutation(i), 1) - real_array(permutation(i - 1), 1)) == 0.0) then
              ! We've found equal entries in real_array(permutation, 1) - this
              ! gives us a start index over which to sort
              start_index = i - 1
            end if
          end if
        else
          ! We're already iterating over equal entries in
          ! real_array(permutation, 1)

          ! We've found an end index over which to sort if ...
          ! ... we're at the end of the array ...
          do_sub_permute = i == size(real_array, 1) + 1
          if(.not. do_sub_permute) then
            ! ... or we've hit non-equal consecutive entries
            do_sub_permute = abs(real_array(permutation(i), 1) - real_array(permutation(i - 1), 1)) > 0.0
          end if
          if(do_sub_permute) then
            ! We've found an end index
            end_index = i - 1

            ! Sort using real_array(permutation(start_index:end_index), 2:)
            allocate(sorted_real_array(end_index - start_index + 1, size(real_array, 2) - 1))
            do j = 1, size(sorted_real_array, 1)
              assert(permutation(j + start_index - 1) >= 1 .and. permutation(j + start_index - 1) <= size(real_array, 1))
              sorted_real_array(j, :) = real_array(permutation(j + start_index - 1), 2:)
            end do
            allocate(sub_permutation(end_index - start_index + 1))
            call sort(sorted_real_array, sub_permutation)
            call apply_permutation(permutation(start_index:end_index), sub_permutation)
            deallocate(sub_permutation)
            deallocate(sorted_real_array)

            ! Now we need to find a new start index
            start_index = -1
          end if
        end if
      end do
    end if

  end subroutine sort_real_array

  function count_unique(int_array) result(unique)
    !!< Count the unique entries in the supplied array of integers

    integer, dimension(:), intent(in) :: int_array

    integer :: unique

    integer :: i
    integer, dimension(size(int_array)) :: permutation

    call qsort(int_array, permutation)

    unique = 0
    if(size(int_array) > 0) then
      unique = unique + 1
    end if
    do i = 2, size(int_array)
      if(int_array(permutation(i)) == int_array(permutation(i - 1))) cycle
      unique = unique + 1
    end do

  end function count_unique

  pure function inverse_permutation(permutation)
    !!< Return the inverse of the supplied permutation

    integer, dimension(:), intent(in) :: permutation

    integer, dimension(size(permutation)) :: inverse_permutation

    integer :: i

    do i = 1, size(permutation)
      inverse_permutation(permutation(i)) = i
    end do

  end function inverse_permutation

  subroutine apply_permutation_integer_array(permutation, applied_permutation)
    !!< Apply the given applied_permutation to the array permutation
    !!< Use this instead of permutation = permutation(applied_permutation), as
    !!< the inline form can cause intermittent errors on some compilers.

    integer, dimension(:, :), intent(inout) :: permutation
    integer, dimension(size(permutation, 1)), intent(in) :: applied_permutation

    integer :: i
    integer, dimension(size(permutation, 1), size(permutation, 2)) :: temp_permutation

    temp_permutation = permutation

    do i = 1, size(applied_permutation)
      assert(applied_permutation(i) >= 1 .and. applied_permutation(i) <= size(permutation))
      permutation(i, :) = temp_permutation(applied_permutation(i), :)
    end do

  end subroutine apply_permutation_integer_array

  subroutine apply_permutation_real_array(permutation, applied_permutation)
    !!< Apply the given applied_permutation to the array permutation
    !!< Use this instead of permutation = permutation(applied_permutation), as
    !!< the inline form can cause intermittent errors on some compilers.

    real, dimension(:, :), intent(inout) :: permutation
    integer, dimension(size(permutation, 1)), intent(in) :: applied_permutation

    integer :: i
    real, dimension(size(permutation, 1), size(permutation, 2)) :: temp_permutation

    temp_permutation = permutation

    do i = 1, size(applied_permutation)
      assert(applied_permutation(i) >= 1 .and. applied_permutation(i) <= size(permutation))
      permutation(i, :) = temp_permutation(applied_permutation(i), :)
    end do

  end subroutine apply_permutation_real_array

  subroutine apply_permutation_integer(permutation, applied_permutation)
    !!< Apply the given applied_permutation to the array permutation
    !!< Use this instead of permutation = permutation(applied_permutation), as
    !!< the inline form can cause intermittent errors on some compilers.

    integer, dimension(:), intent(inout) :: permutation
    integer, dimension(size(permutation)), intent(in) :: applied_permutation

    integer :: i
    integer, dimension(size(permutation)) :: temp_permutation

    temp_permutation = permutation

    do i = 1, size(applied_permutation)
      assert(applied_permutation(i) >= 1 .and. applied_permutation(i) <= size(permutation))
      permutation(i) = temp_permutation(applied_permutation(i))
    end do

  end subroutine apply_permutation_integer

  subroutine apply_permutation_real(permutation, applied_permutation)
    !!< Apply the given applied_permutation to the array permutation
    !!< Use this instead of permutation = permutation(applied_permutation), as
    !!< the inline form can cause intermittent errors on some compilers.

    real, dimension(:), intent(inout) :: permutation
    integer, dimension(size(permutation)), intent(in) :: applied_permutation

    integer :: i
    real, dimension(size(permutation)) :: temp_permutation

    temp_permutation = permutation

    do i = 1, size(applied_permutation)
      assert(applied_permutation(i) >= 1 .and. applied_permutation(i) <= size(permutation))
      permutation(i) = temp_permutation(applied_permutation(i))
    end do

  end subroutine apply_permutation_real

  subroutine apply_reverse_permutation_real(permutation, applied_permutation)
    !!< Apply the reverse of the given applied_permutation to the array permutation

    real, dimension(:), intent(inout) :: permutation
    integer, dimension(size(permutation)), intent(in) :: applied_permutation

    integer :: i, length
    real, dimension(size(permutation)) :: temp_permutation

    temp_permutation = permutation

    length = size(applied_permutation)+1
    do i = 1, size(applied_permutation)
      assert(applied_permutation(i) >= 1 .and. applied_permutation(i) <= size(permutation))
      permutation(i) = temp_permutation(applied_permutation(length-i))
    end do

  end subroutine apply_reverse_permutation_real

  subroutine apply_reverse_permutation_integer(permutation, applied_permutation)
    !!< Apply the reverse of the given applied_permutation to the array permutation

    integer, dimension(:), intent(inout) :: permutation
    integer, dimension(size(permutation)), intent(in) :: applied_permutation

    integer :: i, length
    integer, dimension(size(permutation)) :: temp_permutation

    temp_permutation = permutation

    length = size(applied_permutation)+1
    do i = 1, size(applied_permutation)
      assert(applied_permutation(i) >= 1 .and. applied_permutation(i) <= size(permutation))
      permutation(i) = temp_permutation(applied_permutation(length-i))
    end do

  end subroutine apply_reverse_permutation_integer

  function sortedi(list) result (sorted)
    !! Return the sorted version of list.
    integer, dimension(:) :: list
    integer, dimension(size(list)) :: sorted, perm

    call qsort(list, perm)

    sorted=list(perm)

  end function sortedi

  function sortedi_key(list, key) result (sorted)
    !! Return list sorted according to key.
    integer, dimension(:) :: list, key
    integer, dimension(size(list)) :: sorted, perm

    call qsort(key, perm)

    sorted=list(perm)

  end function sortedi_key



end module quicksort
