!    Copyright (C) 2006-2007 Imperial College London and others.
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

subroutine test_interpolation
  !!< Test that we can interpolate from one field to another
  use fields
  use read_triangle
  use unittest_tools
  use vtk_interfaces
  use conservative_interpolation
  implicit none

  type(vector_field) :: X_in, X_out
  type(scalar_field) :: T_in, T_out
  type(interpolator_type) :: interpolator
  real :: fmin_in, fmax_in, fnorm2_in, fintegral_in
  real :: fmin_out, fmax_out, fnorm2_out, fintegral_out
  logical :: fail

  X_in=read_triangle_files("square.1", quad_degree=4)
  X_out=read_triangle_files("square.2", quad_degree=4)

  call allocate(T_in, X_in%mesh, "tracer")
  call allocate(T_in, X_out%mesh, "tracer")

  call set_from_python_function(T_in, &
       "def val(X,t): import math; return math.cos(X[0])", X, 0.0)

  interpolator=make_interpolator(X_in, X_out)

  call interpolate_field(interpolator, T_in, T_out)

  call field_stats(T_in, X_in, fmin_in, fmax_in, fnorm2_in, fintegral_in)

  call field_stats(T_out, X_out, fmin_out, fmax_out, fnorm2_out, fintegral_out)

  print '(a10, 2a22)', " ","T_in","T_out"
  print '(a10, g22.8, g22.8)', "Minimum", fmin_in, fmin_out
  print '(a10, g22.8, g22.8)', "Maximum", fmax_in, fmax_out
  print '(a10, g22.8, g22.8)', "2-norm", fnorm2_in, fnorm2_out
  print '(a10, g22.8, g22.8)', "integral", integral_in, integral_out

  call vtk_write_fields("interpolation_in", 0, X_in, X_in%mesh, (/T_in/))
  call vtk_write_fields("interpolation_out", 0, X_in, X_in%mesh, (/T_out/))

end subroutine test_interpolation
