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

#include "fdebug.h"

subroutine test_python_2d
  !!< Test that we can set a field using python.
  use fields
  use read_triangle
  use unittest_tools
  use futils
  implicit none

  type(vector_field) :: X
  type(scalar_field) :: T
  type(tensor_field) :: Q
  logical :: fail

#ifdef HAVE_PYTHON
  X=read_triangle_files("data/square.1", quad_degree=4)

  call allocate(T, X%mesh, "tracer")

  call set_from_python_function(T, &
       "def val(X,t): import math; return math.cos(X[0]*X[1])", X, 0.0)

  fail=any(abs(T%val-cos(X%val(1,:)*X%val(2,:)))>1e-14)
  call report_test("[test_python 2D function fields]", fail, .false., &
       "python and fortran should produce the same answer.")

  call set_from_python_function(T%val, &
       "def val(X,t): import math; return math.cos(X[0]*X[1])", X%val(1,:),&
       & X%val(2,:), time=0.0)

  fail=any(abs(T%val-cos(X%val(1,:)*X%val(2,:)))>1e-14)
  call report_test("[test_python 2D function values]", fail, .false., &
       "python and fortran should produce the same answer.")

#ifdef HAVE_NUMPY

  call allocate(Q, X%mesh, "Tensor")

  call set_from_python_function(Q, &
       "def val(X,t): return [[1, 2], [3, 4]]", X, 0.0)

  fail=any(node_val(Q,1)/= reshape((/1.,3.,2.,4./),(/2,2/)))

  call report_test("[test_python 2D tensor field]", fail, .false., &
       "Tensor field value is set correctly.")


#endif
#endif

end subroutine test_python_2d
