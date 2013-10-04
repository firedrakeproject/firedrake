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

subroutine test_python_state

  use fields
  use fldebug
  use python_state
  use python_utils
  use read_triangle
  use state_module
  use unittest_tools

  implicit none

  integer :: dim, i, stat
  logical :: fail
  type(scalar_field) :: s_field
  type(state_type) :: state
  type(tensor_field) :: t_field
  type(vector_field) :: positions, v_field

  positions = read_triangle_files("data/interval", quad_degree = 1)
  dim = positions%dim
  call insert(state, positions, name = "Coordinate")
  call insert(state, positions%mesh, name = "CoordinateMesh")

  call allocate(s_field, positions%mesh, name = "ScalarField")
  call insert(state, s_field, name = s_field%name)

  call zero(s_field)

  fail = .false.
  do i = 1, node_count(s_field)
    if(node_val(s_field, i) .fne. 0.0) then
      fail = .true.
      exit
    end if
  end do
  call report_test("[Zero valued scalar]", fail, .false., "Scalar field is not zero valued")

  call python_add_state(state)
  call python_run_string('s_field = state.scalar_fields["ScalarField"]' // new_line("") // &
                       & 'for i in range(s_field.node_count):' // new_line("") // &
                       & '  s_field.set(i, i + 1)' // new_line(""), &
                       & stat = stat)
  call python_reset()

  call report_test("[python_run_string]", stat /= 0, .false., "python_run_string returned an error")

  fail = .false.
  do i = 1, node_count(s_field)
    if(node_val(s_field, i) .fne. float(i)) then
      fail = .true.
      exit
    end if
  end do
  call report_test("[Scalar field value set in python]", fail, .false., "Failed to set scalar field value")

  call allocate(v_field, positions%dim, positions%mesh, name = "VectorField")
  call insert(state, v_field, name = v_field%name)

  call zero(v_field)

  call allocate(t_field, positions%mesh, name = "TensorField")
  call insert(state, t_field, name = t_field%name)

  fail = .false.
  do i = 1, node_count(v_field)
    if(node_val(v_field, i) .fne. spread(0.0, 1, dim)) then
      fail = .true.
      exit
    end if
  end do
  call report_test("[Zero valued vector]", fail, .false., "Vector field is not zero valued")

  call python_add_state(state)
  call python_run_string('v_field = state.vector_fields["VectorField"]' // new_line("") // &
                       & 'for i in range(v_field.node_count):' // new_line("") // &
                       & '  v_field.set(i, numpy.array([i + 1]))' // new_line(""), &
                       & stat = stat)
  call python_reset()

  call report_test("[python_run_string]", stat /= 0, .false., "python_run_string returned an error")

  fail = .false.
  do i = 1, node_count(v_field)
    if(node_val(v_field, i) .fne. spread(float(i), 1, dim)) then
      fail = .true.
      exit
    end if
  end do
  call report_test("[Vector field value set in python]", fail, .false., "Failed to set vector field value")

  call zero(t_field)

  fail = .false.
  do i = 1, node_count(t_field)
    if(.not. mat_zero(node_val(t_field, i))) then
      fail = .true.
      exit
    end if
  end do
  call report_test("[Zero valued tensor]", fail, .false., "Tensor field is not zero valued")

  call python_add_state(state)
  call python_run_string('t_field = state.tensor_fields["TensorField"]' // new_line("") // &
                       & 'for i in range(t_field.node_count):' // new_line("") // &
                       & '  t_field.set(i, numpy.array([[i + 1]]))' // new_line(""), &
                       & stat = stat)
  call python_reset()

  call report_test("[python_run_string]", stat /= 0, .false., "python_run_string returned an error")

  fail = .false.
  do i = 1, node_count(t_field)
    if(node_val(t_field, i) .fne. reshape(spread(float(i), 1, dim * dim), (/dim, dim/))) then
      fail = .true.
      exit
    end if
  end do
  call report_test("[Tensor field value set in python]", fail, .false., "Failed to set tensor field value")

  call deallocate(state)
  call deallocate(s_field)
  call deallocate(v_field)
  call deallocate(t_field)
  call deallocate(positions)

  call report_test_no_references()

end subroutine test_python_state
