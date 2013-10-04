#include "confdefs.h"

subroutine test_conservative_interpolation_sa

  use fields
  use read_triangle
  use conservative_interpolation_module, only: interpolation_galerkin
  use unittest_tools
  use futils
  use solvers
  use state_module
  use supermesh_assembly

  implicit none

  interface
    function field_func_const(pos)
      real, dimension(:), intent(in) :: pos
      real :: solution
    end function
  end interface

  interface
    function field_func_linear(pos)
      real, dimension(:), intent(in) :: pos
      real :: solution
    end function
  end interface

  interface
    function field_func_quadratic(pos)
      real, dimension(:), intent(in) :: pos
      real :: solution
    end function
  end interface

  interface
    function field_func_cubic(pos)
      real, dimension(:), intent(in) :: pos
      real :: solution
    end function
  end interface

  interface
    function field_func_exp(pos)
      real, dimension(:), intent(in) :: pos
      real :: solution
    end function
  end interface

  type(vector_field) :: positionsA, positionsB
  type(scalar_field), dimension(5) :: fieldA, fieldB, fieldC
  real :: integralA, integralB
  logical :: fail
  integer :: field, no_field
  type(state_type), dimension(1) :: stateA, stateB, stateC
  type(element_type) :: field_element
  type(mesh_type) :: donor_mesh, target_mesh

  positionsA = read_triangle_files("data/pslgA", quad_degree=4, no_faces=.true.)
  positionsB = read_triangle_files("data/pslgB", quad_degree=4, no_faces=.true.)

  field_element = make_element_shape(positionsA%mesh%shape%ndof, positionsA%mesh%shape%dim, 2, positionsA%mesh%shape%quadrature)
  donor_mesh = make_mesh(positionsA%mesh, field_element, continuity=0, name="DonorMesh")
  target_mesh = make_mesh(positionsA%mesh, field_element, continuity=0, name="TargetMesh")

  no_field = size(fieldA)

  do field=1,no_field
    call allocate(fieldA(field), donor_mesh, "Field" // int2str(field))
    fieldA(field)%option_path = "/fieldA" // int2str(field) // "/prognostic/galerkin_projection/continuous"
    call set_solver_options(fieldA(field), ksptype='cg', pctype='eisenstat', rtol=1.0e-7, max_its=10000)
    fieldA(field)%option_path = "/fieldA" // int2str(field)
    call allocate(fieldB(field), target_mesh, "Field" // int2str(field))
    fieldB(field)%option_path = "/fieldB" // int2str(field) // "/prognostic/galerkin_projection/continuous"
    call set_solver_options(fieldB(field), ksptype='cg', pctype='eisenstat', rtol=1.0e-7, max_its=10000)
    fieldB(field)%option_path = "/fieldB" // int2str(field)
    call allocate(fieldC(field), target_mesh, "Field" // int2str(field))
    fieldC(field)%option_path = "/fieldC" // int2str(field) // "/prognostic/galerkin_projection/continuous"
    call set_solver_options(fieldC(field), ksptype='cg', pctype='eisenstat', rtol=1.0e-7, max_its=10000)
    fieldC(field)%option_path = "/fieldC" // int2str(field)
  end do

  call insert(stateA, positionsA, "Coordinate")
  call insert(stateA, positionsA%mesh, "CoordinateMesh")
  call insert(stateB, positionsB, "Coordinate")
  call insert(stateB, positionsB%mesh, "CoordinateMesh")
  call insert(stateC, positionsB, "Coordinate")
  call insert(stateC, positionsB%mesh, "CoordinateMesh")

  call set_from_function(fieldA(1), field_func_const, positionsA)
  call set_from_function(fieldA(2), field_func_linear, positionsA)
  call set_from_function(fieldA(3), field_func_quadratic, positionsA)
  call set_from_function(fieldA(4), field_func_cubic, positionsA)
  call set_from_function(fieldA(5), field_func_exp, positionsA)

  do field=1,no_field
    call zero(fieldB(field))
    call zero(fieldC(field))

    call insert(stateA(1), fieldA(field), name=trim(fieldA(field)%name))
    call insert(stateB(1), fieldB(field), name=trim(fieldB(field)%name))
    call insert(stateC(1), fieldC(field), name=trim(fieldC(field)%name))
  end do

  call galerkin_projection_scalars(stateA, positionsA, stateB, positionsB)
  call interpolation_galerkin(stateA, positionsA, stateC, positionsB)

  call deallocate(stateA(1))
  call deallocate(stateB(1))
  call deallocate(stateC(1))

  do field = 1, no_field
    call report_test("[Same result as interpolation_galerkin]", fieldB(field)%val .fne. fieldC(field)%val, .false., "Result differs from that returned by interpolation_galerkin")
  end do

  do field=1,no_field
    integralA = field_integral(fieldA(field), positionsA)
    integralB = field_integral(fieldB(field), positionsB)

    fail=(abs(integralA - integralB) > epsilon(0.0_4))
    call report_test("[conservative interpolation galerkin]", fail, .false., "")

    if (fail) then
      write(0,*) "integralA == ", integralA
      write(0,*) "integralB == ", integralB
      write(0,*) "integralB - integralA == ", integralB - integralA
    end if
  end do

  call deallocate(target_mesh)
  call deallocate(donor_mesh)
  call deallocate(field_element)
  call deallocate(positionsA)
  call deallocate(positionsB)
  do field=1,no_field
    call deallocate(fieldA(field))
    call deallocate(fieldB(field))
    call deallocate(fieldC(field))
  end do

  call report_test_no_references()

end subroutine test_conservative_interpolation_sa

function field_func_const(pos) result(f)
  real, dimension(:), intent(in) :: pos
  real :: f

  f = 1.0
end function field_func_const

function field_func_linear(pos) result(f)
  real, dimension(:), intent(in) :: pos
  real :: f

  f = pos(1) + pos(2)
end function field_func_linear

function field_func_quadratic(pos) result(f)
  real, dimension(:), intent(in) :: pos
  real :: f

  f = pos(1)**2 + 2.0 * pos(2) + 3.0
end function field_func_quadratic

function field_func_cubic(pos) result(f)
  real, dimension(:), intent(in) :: pos
  real :: f

  f = 5.0 * pos(2)**3 + pos(1)**2 + 2.0 * pos(2) + 3.0
end function field_func_cubic

function field_func_exp(pos) result(f)
  real, dimension(:), intent(in) :: pos
  real :: f

  f = exp(pos(1)**2 + 2.0 * pos(2))
end function field_func_exp
