#include "confdefs.h"

subroutine test_conservative_interpolation

  use fields
  use read_triangle
  use conservative_interpolation_module, only: interpolation_galerkin
  use unittest_tools
  use futils
  use solvers
  use state_module
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
  type(scalar_field), dimension(5) :: fieldA, fieldB
  real :: integralA, integralB
  logical :: fail
  integer :: field, no_field
  type(state_type), dimension(1) :: stateA, stateB

  positionsA = read_triangle_files("data/pslgA", quad_degree=4, no_faces=.true.)
  positionsB = read_triangle_files("data/pslgB", quad_degree=4, no_faces=.true.)

  no_field = 5

  do field=1,no_field
    call allocate(fieldA(field), positionsA%mesh, "Field" // int2str(field))
    fieldA(field)%option_path = "/fieldA" // int2str(field) // "/prognostic/galerkin_projection/continuous"
    call set_solver_options(fieldA(field), ksptype='cg', pctype='sor', rtol=1.0e-7, max_its=10000)
    fieldA(field)%option_path = "/fieldA" // int2str(field)
    call allocate(fieldB(field), positionsB%mesh, "Field" // int2str(field))
    fieldB(field)%option_path = "/fieldB" // int2str(field) // "/prognostic/galerkin_projection/continuous"
    call set_solver_options(fieldB(field), ksptype='cg', pctype='sor', rtol=1.0e-7, max_its=10000)
    fieldB(field)%option_path = "/fieldB" // int2str(field)
  end do

  call set_from_function(fieldA(1), field_func_const, positionsA)
  call set_from_function(fieldA(2), field_func_linear, positionsA)
  call set_from_function(fieldA(3), field_func_quadratic, positionsA)
  call set_from_function(fieldA(4), field_func_cubic, positionsA)
  call set_from_function(fieldA(5), field_func_exp, positionsA)

  do field=1,no_field
    call zero(fieldB(field))

    call insert(stateA(1), fieldA(field), name=trim(fieldA(field)%name))
    call insert(stateB(1), fieldB(field), name=trim(fieldB(field)%name))
  end do

  call interpolation_galerkin(stateA, positionsA, stateB, positionsB)

  call deallocate(stateA(1))
  call deallocate(stateB(1))

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

  do field=1,no_field
    call zero(fieldB(field))

    call insert(stateA(1), fieldA(field), name=trim(fieldA(field)%name))
    call insert(stateB(1), fieldB(field), name=trim(fieldB(field)%name))
  end do

  call interpolation_galerkin(stateA, positionsA, stateB, positionsB, force_bounded=.true.)

  call deallocate(stateA(1))
  call deallocate(stateB(1))

  do field=1,no_field
    integralA = field_integral(fieldA(field), positionsA)
    integralB = field_integral(fieldB(field), positionsB)

    fail=(abs(integralA - integralB) > epsilon(0.0_4))
    call report_test("[conservative interpolation bounded]", fail, .false., "")

    if (fail) then
      write(0,*) "integralA == ", integralA
      write(0,*) "integralB == ", integralB
      write(0,*) "integralB - integralA == ", integralB - integralA
    end if
  end do

end subroutine test_conservative_interpolation

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
