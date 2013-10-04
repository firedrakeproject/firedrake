subroutine test_quad_quadrature

  use unittest_tools
  use elements
  use fetools
  use shape_functions
  implicit none

  logical :: fail
  type(element_type) :: quad_shape, tri_shape
  type(quadrature_type) :: quad_quadrature, tri_quadrature
  real, dimension(2,4) :: X_quad
  real, dimension(2,6) :: X_tri
  real, dimension(:,:,:), allocatable :: J_quad, J_tri
  real, dimension(:), allocatable :: detwei_quad, detwei_tri
  real, dimension(4,4) :: quad_mass
  real, dimension(6,6) :: l_tri_mass
  real, dimension(4,4) :: global_tri_mass
  real, dimension(4,6) :: local2global
  integer :: i

  quad_quadrature = make_quadrature(vertices=4,dim=2,degree=2)
  tri_quadrature = make_quadrature(vertices=3,dim=2,degree=4)

  allocate(J_quad(2,2,quad_quadrature%ngi))
  allocate(J_tri(2,2,tri_quadrature%ngi))
  allocate(detwei_quad(quad_quadrature%ngi))
  allocate(detwei_tri(tri_quadrature%ngi))

  quad_shape=make_element_shape(vertices=4, dim=2, degree=1, &
       &quad= quad_quadrature)
  tri_shape=make_element_shape(vertices=3, dim=2, degree=2, &
       &quad= tri_quadrature)

  !This unit test is based on the fact that the Q1 space on a single
  !quadrilateral can be represented exactly by the P2 space on the same
  !quadrilateral subdivided into two quadratically-mapped triangles.  The
  !dividing line between the triangles is quadratic and passes through the
  !mean of the four vertices.

  !compute the mass matrix using quadrilateral quadrature
  !numbering: 3 4
  !           1 2
  quad_mass = 0.
  X_quad(:,1) = (/0.,0./)
  X_quad(:,2) = (/1.0,0./)
  X_quad(:,3) = (/0.2,1.2/)
  X_quad(:,4) = (/1.3,1.5/)
  call compute_jacobian(X_quad, quad_shape, J=J_quad,detwei=detwei_quad)
  quad_mass = shape_shape(quad_shape,quad_shape,detwei_quad)

  !compute the mass matrix using triangular quadrature
  global_tri_mass = 0.
  !Top-left triangle
  !numbering: 6 5 3         3  4
  !           4 2
  !           1             1  2
  X_tri(:,1) = X_quad(:,1)
  X_tri(:,2) = sum(X_quad,2)/4.0
  X_tri(:,3) = X_quad(:,4)
  X_tri(:,4) = (X_quad(:,1)+X_quad(:,3))/2.0
  X_tri(:,5) = (X_quad(:,3)+X_quad(:,4))/2.0
  X_tri(:,6) = X_quad(:,3)
  call compute_jacobian(X_tri, tri_shape, J=J_tri,detwei=detwei_tri)
  l_tri_mass = shape_shape(tri_shape,tri_shape,detwei_tri)
  !local2global(i,:) gives coefficients of expansion of Q1 basis function i
  !into P2 basis functions in this triangle
  local2global(1,:) = (/1.,0.25,0.,0.5,0.,0./)
  local2global(2,:) = (/0.,0.25,0.,0.,0.,0./)
  local2global(3,:) = (/0.,0.25,0.,0.5,0.5,1./)
  local2global(4,:) = (/0.,0.25,1.,0.,0.5,0./)
  global_tri_mass = global_tri_mass + &
       matmul(local2global,matmul(l_tri_mass,transpose(local2global)))
  !bottom-right triangle
  !numbering:     1       3    4
  !             2 4
  !           3 5 6       1    2
  X_tri(:,1) = X_quad(:,4)
  X_tri(:,2) = sum(X_quad,2)/4.0
  X_tri(:,3) = X_quad(:,1)
  X_tri(:,4) = (X_quad(:,4)+X_quad(:,2))/2.0
  X_tri(:,5) = (X_quad(:,1)+X_quad(:,2))/2.0
  X_tri(:,6) = X_quad(:,2)
  call compute_jacobian(X_tri, tri_shape, J=J_tri,detwei=detwei_tri)
  l_tri_mass = shape_shape(tri_shape,tri_shape,detwei_tri)
  !local2global(i,:) gives coefficients of expansion of Q1 basis function i
  !into P2 basis functions in this triangle
  local2global(1,:) = (/0.,0.25,1.,0.,0.5,0./)
  local2global(2,:) = (/0.,0.25,0.,0.5,0.5,1./)
  local2global(3,:) = (/0.,0.25,0.,0.,0.,0./)
  local2global(4,:) = (/1.,0.25,0.,0.5,0.,0./)
  global_tri_mass = global_tri_mass + &
       matmul(local2global,matmul(l_tri_mass,transpose(local2global)))

  fail = any(abs(quad_mass - global_tri_mass) > 1e-12)
  call report_test("[quad_quadrature]", fail, .false., "matrices not the same")

end subroutine test_quad_quadrature
