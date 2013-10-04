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

program testpolynomials
  use polynomials
  ! Implicit none.

  type(polynomial), dimension(10) :: poly



  poly(1)=(/1.0, 1.0/)
  poly(2)=(/2.0, 1.0/)
  poly(3)=(/1.0, 1.0, 1.0/)

  print '(100f10.3)', poly(1)*poly(1)*poly(1)+poly(2)
  print '(100f10.3)', 2.0*poly(2)
  print '(100f10.3)', ddx(poly(1)*poly(1)*poly(1))

end program testpolynomials
