subroutine test_jacobian
  !!< test computation of jacobian for a 2D triangle embedded in 3D space

  use transform_elements
  use elements
  use quadrature
  use shape_functions
  use unittest_tools

  implicit none

  integer, parameter :: dim=2, loc=3, quad_degree=4

  type(element_type), pointer :: shape
  type(quadrature_type), pointer :: quad
  real, dimension(3,3) :: X
  real, allocatable, dimension(:,:,:) :: J
  real, allocatable, dimension(:) :: detwei
  logical :: fail

  allocate(quad)
  allocate(shape)

  quad=make_quadrature(loc, dim, quad_degree)
  shape=make_element_shape(loc, dim, 1, quad)
  allocate(J(dim,3,shape%ngi))
  allocate(detwei(shape%ngi))

  ! positions field
  X(:,1)=(/0,0,0/)
  X(:,2)=(/2,0,0/)
  X(:,3)=(/0,1,1/)

  ! compute jacobian
  call compute_jacobian(X, shape, J, detwei)

  fail = abs(sum(detwei)-sqrt(2.0))>1e-10
  call report_test("[compute_jacobian]", fail, .false., "Incorrect Jacobian")

end subroutine test_jacobian
