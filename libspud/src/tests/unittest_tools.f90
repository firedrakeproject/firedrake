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

module unittest_tools
  !!< This module contains utility functions for the libspud unit testing framework.

  implicit none

  private

  public :: report_test, int2str_len, int2str

contains

  subroutine report_test(title, fail, warn, msg)
    !!< This is the subroutine used by unit tests to report the output of
    !!< a test case.

    !! Title: the name of the test case.
    character(len = *), intent(in) :: title
    !! Msg: an explanatory message printed if the test case fails.
    character(len = *), intent(in) :: msg
    !! Has the test case failed, or triggered a warning? Set fail or warn to .true. if so.
    logical, intent(in) :: fail, warn

    if(fail) then
      print "('Fail: ',a,'; error: ',a)", title, msg
    else if(warn) then
      print "('Warn: ',a,'; error: ',a)", title, msg
    else
      print "('Pass: ',a)", title
    end if

  end subroutine report_test

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

end module unittest_tools
