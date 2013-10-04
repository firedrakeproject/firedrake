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

module surface_integrals

  use quadrature
  use elements
  use field_options
  use fields
  use fldebug
  use global_parameters, only : OPTION_PATH_LEN
  use parallel_fields
  use spud
  use state_module
  !use smoothing_module

  implicit none

  private

  public :: calculate_surface_integral, gradient_normal_surface_integral, &
    & normal_surface_integral, surface_integral, surface_gradient_normal, &
    & surface_normal_distance_sele
  public :: diagnostic_body_drag

  interface integrate_over_surface_element
    module procedure integrate_over_surface_element_mesh, &
      & integrate_over_surface_element_scalar, &
      & integrate_over_surface_element_vector, &
      & integrate_over_surface_element_tensor
  end interface

  interface calculate_surface_integral
    module procedure calculate_surface_integral_scalar, &
      & calculate_surface_integral_vector
  end interface

contains

  function surface_integral(s_field, positions, surface_ids, normalise) result(integral)
    !!< Integrate the given scalar field over the surface of its mesh. The
    !!< surface elements integrated over are defined by
    !!< integrate_over_surface_element(...). If normalise is present and true,
    !!< then the surface integral is normalised by surface area.

    type(scalar_field), intent(in) :: s_field
    type(vector_field), target, intent(in) :: positions
    integer, dimension(:), optional, intent(in) :: surface_ids
    logical, optional, intent(in) :: normalise

    real :: integral

    integer :: i
    logical :: integrate_over_element
    real :: area, face_area, face_integral

    if(present_and_true(normalise)) then
      area = 0.0
    end if
    integral = 0.0

    do i = 1, surface_element_count(s_field)
      if(present(surface_ids)) then
        integrate_over_element = integrate_over_surface_element(s_field, i, surface_ids)
      else
        integrate_over_element = integrate_over_surface_element(s_field, i)
      end if

      if(integrate_over_element) then
        if(present_and_true(normalise)) then
          call surface_integral_face(s_field, i, positions, face_integral, area = face_area)
          area = area + face_area
        else
          call surface_integral_face(s_field, i, positions, face_integral)
        end if
        integral = integral + face_integral
      end if
    end do

    call allsum(integral, communicator = halo_communicator(s_field))
    if(present_and_true(normalise)) then
      call allsum(area, communicator = halo_communicator(s_field))
      integral = integral / area
    end if

  end function surface_integral

  subroutine surface_integral_face(s_field, face, positions, integral, area)
    type(scalar_field), intent(in) :: s_field
    type(vector_field), intent(in) :: positions
    integer, intent(in) :: face
    real, intent(out) :: integral
    real, optional, intent(out) :: area

    real, dimension(face_ngi(s_field, face)) :: detwei

    assert(face_ngi(s_field, face) == face_ngi(positions, face))

    call transform_facet_to_physical(positions, face, detwei)

    integral = dot_product(face_val_at_quad(s_field, face), detwei)
    if(present(area)) then
      area = sum(detwei)
    end if

  end subroutine surface_integral_face

  function normal_surface_integral(v_field, positions, surface_ids, normalise) result(integral)
    !!< Evaluate:
    !!<   /
    !!<   | v_field dot dn
    !!<   /
    !!< over the surface of v_field's mesh. The surface elements integrated
    !!< over are defined by integrate_over_surface_element(...). If normalise is
    !!< present and true, then the surface integral is normalised by surface
    !!< area.

    type(vector_field), intent(in) :: v_field
    type(vector_field), target, intent(in) :: positions
    integer, dimension(:), optional, intent(in) :: surface_ids
    logical, optional, intent(in) :: normalise

    real :: integral

    integer :: i
    logical :: integrate_over_element
    real :: area, face_area, face_integral

    if(present_and_true(normalise)) then
      area = 0.0
    end if
    integral = 0.0
    do i = 1, surface_element_count(v_field)
      if(present(surface_ids)) then
        integrate_over_element = integrate_over_surface_element(v_field, i, surface_ids)
      else
        integrate_over_element = integrate_over_surface_element(v_field, i)
      end if

      if(integrate_over_element) then
        if(present_and_true(normalise)) then
          call normal_surface_integral_face(v_field, i, positions, face_integral, area = face_area)
          area = area + face_area
        else
          call normal_surface_integral_face(v_field, i, positions, face_integral)
        end if
        integral = integral + face_integral
      end if
    end do

    call allsum(integral, communicator = halo_communicator(v_field))
    if(present_and_true(normalise)) then
      call allsum(area, communicator = halo_communicator(v_field))
      integral = integral / area
    end if

  end function normal_surface_integral

  subroutine normal_surface_integral_face(v_field, face, positions, integral, area)
    type(vector_field), intent(in) :: v_field
    integer, intent(in) :: face
    type(vector_field), intent(in) :: positions
    real, intent(out) :: integral
    real, optional, intent(out) :: area

    integer :: ele, i
    real, dimension(face_ngi(v_field, face)) :: detwei, v_dot_n_at_quad
    real, dimension(v_field%dim, face_ngi(v_field, face)) :: normal, v_at_quad
    type(element_type), pointer :: element_shape, face_element_shape

    ele = face_ele(v_field, face)

    assert(face_ngi(v_field, face) == face_ngi(positions, face))
    assert(ele_ngi(v_field, ele) == ele_ngi(positions, ele))
    assert(v_field%dim == positions%dim)

    face_element_shape => face_shape(v_field, face)
    element_shape => ele_shape(v_field, ele)

    call transform_facet_to_physical( &
      positions, face, detwei_f = detwei, normal = normal)

    ! Find value of v_field at the surface element quadrature points
    v_at_quad = face_val_at_quad(v_field, face)

    ! Find the value of v_field . normal at the surface element quadrature
    ! points
    do i = 1, face_ngi(v_field, face)
      v_dot_n_at_quad(i) = dot_product(v_at_quad(:, i), normal(:, i))
    end do

    ! Integrate over this surface element
    integral = dot_product(v_dot_n_at_quad, detwei)
    if(present(area)) then
      area = sum(detwei)
    end if

  end subroutine normal_surface_integral_face

  function gradient_normal_surface_integral(s_field, positions, surface_ids, normalise) result(integral)
    !!< Evaluate:
    !!<   /
    !!<   | grad s_field dot dn
    !!<   /
    !!< over the surface of s_field's mesh. The surface elements integrated
    !!< over are defined by integrate_over_surface_element(...). If normalise is
    !!< present and true, then the surface integral is normalised by surface
    !!< area.

    type(scalar_field), intent(in) :: s_field
    type(vector_field), target, intent(in) :: positions
    integer, dimension(:), optional, intent(in) :: surface_ids
    logical, optional, intent(in) :: normalise

    real :: integral

    integer :: i
    logical :: integrate_over_element
    real :: area, face_area, face_integral

    if(present_and_true(normalise)) then
      area = 0.0
    end if
    integral = 0.0

    do i = 1, surface_element_count(s_field)
      if(present(surface_ids)) then
        integrate_over_element = integrate_over_surface_element(s_field, i, surface_ids)
      else
        integrate_over_element = integrate_over_surface_element(s_field, i)
      end if

      if(integrate_over_element) then
        if(present_and_true(normalise)) then
          call gradient_normal_surface_integral_face(s_field, i, positions, face_integral, area = face_area)
          area = area + face_area
        else
          call gradient_normal_surface_integral_face(s_field, i, positions, face_integral)
        end if
        integral = integral + face_integral
      end if
    end do

    call allsum(integral, communicator = halo_communicator(s_field))
    if(present_and_true(normalise)) then
      call allsum(area, communicator = halo_communicator(s_field))
      integral = integral / area
    end if

  end function gradient_normal_surface_integral

  subroutine gradient_normal_surface_integral_face(s_field, face, positions, integral, area)
    type(scalar_field), intent(in) :: s_field
    integer, intent(in) :: face
    type(vector_field), intent(in) :: positions
    real, intent(out) :: integral
    real, optional, intent(out) :: area

    integer :: ele, i, j, l_face_number
    real, dimension(ele_loc(s_field, face_ele(s_field, face))) :: s_ele_val
    real, dimension(face_ngi(s_field, face)) :: detwei, grad_s_dot_n_at_quad
    real, dimension(ele_loc(s_field, face_ele(s_field, face)),face_ngi(s_field, face),mesh_dim(s_field)) :: ele_dshape_at_face_quad
    real, dimension(mesh_dim(s_field), face_ngi(s_field, face)) :: grad_s_at_quad, normal
    real, dimension(mesh_dim(s_field), mesh_dim(s_field), ele_ngi(s_field, face_ele(s_field, face))) :: invj
    real, dimension(mesh_dim(s_field), mesh_dim(s_field), face_ngi(s_field, face)) :: invj_face
    type(element_type) :: augmented_shape
    type(element_type), pointer :: element_shape, face_element_shape, &
      & positions_face_element_shape, positions_shape

    ele = face_ele(s_field, face)

    assert(face_ngi(s_field, face) == face_ngi(positions, face))
    assert(ele_ngi(s_field, ele) == ele_ngi(positions, ele))
    assert(mesh_dim(s_field) == positions%dim)

    face_element_shape => face_shape(s_field, face)
    positions_face_element_shape => face_shape(positions, face)

    element_shape => ele_shape(s_field, ele)
    positions_shape => ele_shape(positions, ele)

    assert(positions_shape%degree == 1)
    if(associated(element_shape%dn_s)) then
      augmented_shape = element_shape
      call incref(augmented_shape)
    else
      augmented_shape = make_element_shape(positions_shape%ndof, element_shape%dim, &
        & element_shape%degree, element_shape%quadrature, &
        & quad_s = face_element_shape%quadrature)
    end if

    call compute_inverse_jacobian( &
      & ele_val(positions, ele), positions_shape, &
      ! Output variables
      & invj = invj)
    assert(cell_family(positions_shape) == FAMILY_SIMPLEX)
    invj_face = spread(invj(:, :, 1), 3, size(invj_face, 3))

    call transform_facet_to_physical( &
      positions, face, detwei_f = detwei, normal = normal)

    ! As "strain" calculation in diagnostic_body_drag_new_options

    ! Get the local face number
    l_face_number = local_face_number(s_field, face)

    ! Evaluate the volume element shape function derivatives at the surface
    ! element quadrature points
    ele_dshape_at_face_quad = eval_volume_dshape_at_face_quad(augmented_shape, l_face_number, invj_face)

    ! Calculate grad s_field at the surface element quadrature points
    s_ele_val = ele_val(s_field, ele)
    forall(i = 1:mesh_dim(s_field), j = 1:face_ngi(s_field, face))
      grad_s_at_quad(i, j) = dot_product(s_ele_val, ele_dshape_at_face_quad(:, j, i))
    end forall

    ! Calculate grad s_field dot dn at the surface element quadrature points
    do i = 1, face_ngi(s_field, face)
      grad_s_dot_n_at_quad(i) = dot_product(grad_s_at_quad(:, i), normal(:, i))
    end do

    ! Integrate over the surface element
    integral = dot_product(grad_s_dot_n_at_quad, detwei)
    if(present(area)) then
      area = sum(detwei)
    end if

    call deallocate(augmented_shape)

  end subroutine gradient_normal_surface_integral_face

  subroutine surface_gradient_normal(source, positions, output, surface_ids)
    !!< Return a field containing:
    !!<   /
    !!<   | grad source dot dn
    !!<   /
    !!< The output field is P0 over the surface. Here, output is a volume field,
    !!< hence there will be errors at edges.

    type(scalar_field), intent(in) :: source
    type(vector_field), intent(in) :: positions
    type(scalar_field), intent(inout) :: output
    integer, dimension(:), optional, intent(in) :: surface_ids

    integer :: i
    real :: face_area, face_integral

    if(continuity(output) /= -1) then
      FLAbort("surface_gradient_normal requires a discontinuous mesh")
    end if

    call zero(output)
    do i = 1, surface_element_count(output)
      if(.not. include_face(i, source, surface_ids = surface_ids)) cycle
      call gradient_normal_surface_integral_face(source, i, positions, face_integral, area = face_area)
      call set(output, face_global_nodes(output, i), spread(face_integral / face_area, 1, face_loc(output, i)))
    end do

  contains

    function include_face(face, source, surface_ids)
      integer, intent(in) :: face
      type(scalar_field), intent(in) :: source
      integer, dimension(:), optional, intent(in) :: surface_ids

      logical :: include_face

      if(present(surface_ids)) then
        if(.not. associated(source%mesh%faces)) then
          include_face = .false.
        else if(.not. associated(source%mesh%faces%boundary_ids)) then
          include_face = .false.
        else
          include_face = any(surface_ids == surface_element_id(source, face))
        end if
      else
        include_face = .true.
      end if

    end function include_face

  end subroutine surface_gradient_normal

  function surface_normal_distance_sele(positions, sele, ele) result(h)
    ! calculate wall-normal element size
    type(vector_field), intent(in) :: positions
    integer, intent(in) :: ele, sele
    real :: h
    type(element_type), pointer :: shape
    integer :: i, dim
    real, dimension(face_ngi(positions,sele)) :: detwei_bdy
    real, dimension(positions%dim,positions%dim) :: J
    real, dimension(positions%dim,face_ngi(positions,sele)) :: normal_bdy

    shape => ele_shape(positions, ele)
    dim = positions%dim

    call transform_facet_to_physical(positions, sele, detwei_f=detwei_bdy, normal=normal_bdy)
    J = transpose(matmul(ele_val(positions, ele) , shape%dn(:, 1, :)))
    h = maxval((/( abs(dot_product(normal_bdy(:, 1), J(i, :))), i=1, dim)/))

  end function surface_normal_distance_sele

  function integrate_over_surface_element_mesh(mesh, face_number, surface_ids) result(integrate_over_element)
    !!< Return whether the given surface element should be integrated over when
    !!< performing a surface integral

    type(mesh_type), intent(in) :: mesh
    integer, intent(in) :: face_number
    integer, dimension(:), optional, intent(in) :: surface_ids

    logical :: integrate_over_element

    if(present(surface_ids)) then
      ! If surface_ids have been supplied, only integrate over the element
      ! if the surface element surface ID exists and is in the list of supplied
      ! surface IDs
      if(.not. associated(mesh%faces)) then
        integrate_over_element = .false.
        return
      else if(.not. associated(mesh%faces%boundary_ids)) then
        integrate_over_element = .false.
        return
      else if(.not. any(surface_ids == surface_element_id(mesh, face_number))) then
        integrate_over_element = .false.
        return
      end if
    end if

    if(isparallel()) then
      ! In parallel, only integrate over the surface element if it is owned by
      ! this process
      if(.not. surface_element_owned(mesh, face_number)) then
        integrate_over_element = .false.
        return
      end if
    end if

    integrate_over_element = .true.

  end function integrate_over_surface_element_mesh

  function integrate_over_surface_element_scalar(s_field, face_number, surface_ids) result(integrate_over_element)
    !!< Return whether the given surface element on the mesh for the given field
    !!< should be integrated over when performing a surface integral

    type(scalar_field), intent(in) :: s_field
    integer, intent(in) :: face_number
    integer, dimension(:), optional, intent(in) :: surface_ids

    logical :: integrate_over_element

    if(present(surface_ids)) then
      integrate_over_element = integrate_over_surface_element(s_field%mesh, face_number, surface_ids)
    else
      integrate_over_element = integrate_over_surface_element(s_field%mesh, face_number)
    end if

  end function integrate_over_surface_element_scalar

  function integrate_over_surface_element_vector(v_field, face_number, surface_ids) result(integrate_over_element)
    !!< Return whether the given surface element on the mesh for the given field
    !!< should be integrated over when performing a surface integral

    type(vector_field), intent(in) :: v_field
    integer, intent(in) :: face_number
    integer, dimension(:), optional, intent(in) :: surface_ids

    logical :: integrate_over_element

    if(present(surface_ids)) then
      integrate_over_element = integrate_over_surface_element(v_field%mesh, face_number, surface_ids)
    else
      integrate_over_element = integrate_over_surface_element(v_field%mesh, face_number)
    end if

  end function integrate_over_surface_element_vector

  function integrate_over_surface_element_tensor(t_field, face_number, surface_ids) result(integrate_over_element)
    !!< Return whether the given surface element on the mesh for the given field
    !!< should be integrated over when performing a surface integral

    type(tensor_field), intent(in) :: t_field
    integer, intent(in) :: face_number
    integer, dimension(:), optional, intent(in) :: surface_ids

    logical :: integrate_over_element

    if(present(surface_ids)) then
      integrate_over_element = integrate_over_surface_element(t_field%mesh, face_number, surface_ids)
    else
      integrate_over_element = integrate_over_surface_element(t_field%mesh, face_number)
    end if

  end function integrate_over_surface_element_tensor

  subroutine diagnostic_body_drag(state, force, surface_integral_name, pressure_force, viscous_force)
    type(state_type), intent(in) :: state
    real, dimension(:), intent(out) :: force
    character(len = FIELD_NAME_LEN), intent(in) :: surface_integral_name
    real, dimension(size(force)), optional, intent(out) :: pressure_force
    real, dimension(size(force)), optional, intent(out) :: viscous_force

    type(vector_field), pointer :: velocity, position
    type(tensor_field), pointer :: viscosity
    type(scalar_field), pointer :: pressure
    type(element_type), pointer :: x_f_shape, x_shape, u_shape, u_f_shape
    character(len=OPTION_PATH_LEN) :: option_path
    integer :: ele,sele,nloc,snloc,sngi,ngi,stotel,nfaces,meshdim, gi
    integer, dimension(:), allocatable :: surface_ids
    integer, dimension(2) :: shape_option
    real, dimension(:), allocatable :: face_detwei, face_pressure
    real, dimension(:,:), allocatable :: velocity_ele, normal, strain, force_at_quad
    real, dimension(:,:,:), allocatable :: dn_t,viscosity_ele, tau, invJ, vol_dshape_face, invJ_face
    real :: sarea
    integer :: l_face_number, stat
    type(element_type) :: augmented_shape
    logical :: have_viscosity

    ewrite(1,*) 'In diagnostic_body_drag'
    ewrite(1,*) 'Computing body forces for label "'//trim(surface_integral_name)//'"'

    position => extract_vector_field(state, "Coordinate")
    pressure => extract_scalar_field(state, "Pressure")
    velocity => extract_vector_field(state, "Velocity")
    viscosity=> extract_tensor_field(state, "Viscosity", stat)
    have_viscosity = stat == 0

    assert(size(force) == position%dim)

    meshdim = mesh_dim(velocity)
    x_shape => ele_shape(position, 1)
    u_shape => ele_shape(velocity, 1)
    x_f_shape => face_shape(position, 1)
    u_f_shape => face_shape(velocity, 1)
    nloc = ele_loc(velocity, 1)
    snloc = face_loc(velocity, 1)
    ngi   = ele_ngi(velocity, 1)
    sngi  = face_ngi(velocity, 1)
    stotel = surface_element_count(velocity)
    option_path = velocity%option_path
    shape_option = option_shape(trim(option_path)//'/prognostic/stat/compute_body_forces_on_surfaces::'//trim(surface_integral_name)//'/surface_ids')
    allocate( surface_ids(shape_option(1)), face_detwei(sngi), &
              dn_t(nloc, ngi, meshdim), &
              velocity_ele(meshdim, nloc), normal(meshdim, sngi), &
              face_pressure(sngi), viscosity_ele(meshdim, meshdim, sngi), &
              tau(meshdim, meshdim, sngi), strain(meshdim, meshdim), &
              force_at_quad(meshdim, sngi))

    allocate(invJ(meshdim, meshdim, ngi))
    allocate(vol_dshape_face(ele_loc(velocity, 1), face_ngi(velocity, 1),meshdim))
    allocate(invJ_face(meshdim, meshdim, face_ngi(velocity, 1)))

    call get_option(trim(option_path)//'/prognostic/stat/compute_body_forces_on_surfaces::'//trim(surface_integral_name)//'/surface_ids', surface_ids)
    ewrite(2,*) 'Calculating forces on surfaces with these IDs: ', surface_ids

    augmented_shape = make_element_shape(x_shape%ndof, u_shape%dim, u_shape%degree, u_shape%quadrature, &
                    & quad_s=u_f_shape%quadrature)

    sarea = 0.0
    nfaces = 0
    force = 0.0
    if(present(pressure_force)) pressure_force = 0.0
    if(present(viscous_force)) viscous_force = 0.0
    do sele=1,stotel
      if(integrate_over_surface_element(velocity, sele, surface_ids = surface_ids)) then
        ! Get face_detwei and normal
        ele = face_ele(velocity, sele)
          call transform_facet_to_physical( &
             position, sele, detwei_f=face_detwei, normal=normal)
          call transform_to_physical(position, ele, &
             shape=u_shape, dshape=dn_t, invJ=invJ)
          velocity_ele = ele_val(velocity,ele)

          ! Compute tau only if viscosity is present
          if(have_viscosity) then
            viscosity_ele = face_val_at_quad(viscosity,sele)

          !
          ! Form the stress tensor
          !

          ! If P1, Assume strain tensor is constant over
          ! the element.
          ! If it isn't, computing this becomes a whole lot more
          ! complicated. You have to compute the values of the
          ! derivatives of the volume basis functions at the
          ! quadrature points of the surface element.

            if (u_shape%degree == 1 .and. cell_family(u_shape) == FAMILY_SIMPLEX) then
              strain = matmul(velocity_ele, dn_t(:, 1, :))
              strain = (strain + transpose(strain)) / 2.0
              do gi=1,sngi
                tau(:, :, gi) = 2 * matmul(viscosity_ele(:, :, gi), strain)
              end do
            else
              ! Get the local face number.
              l_face_number = local_face_number(velocity, sele)

              ! Here comes the magic.
              if (x_shape%degree == 1 .and. cell_family(x_shape) == FAMILY_SIMPLEX) then
                invJ_face = spread(invJ(:, :, 1), 3, size(invJ_face, 3))
              else
                ewrite(-1,*) "If positions are nonlinear, then you have to compute"
                ewrite(-1,*) "the inverse Jacobian of the volume element at the surface"
                ewrite(-1,*) "quadrature points. Sorry ..."
                FLExit("Calculating the body drag not supported for nonlinear coordinates.")
              end if
              vol_dshape_face = eval_volume_dshape_at_face_quad(augmented_shape, l_face_number, invJ_face)

              do gi=1,sngi
                strain = matmul(velocity_ele, vol_dshape_face(:, gi, :))
                strain = (strain + transpose(strain)) / 2.0
                tau(:, :, gi) = 2 * matmul(viscosity_ele(:, :, gi), strain)
              end do
            end if

          end if

          face_pressure = face_val_at_quad(pressure, sele)
          nfaces = nfaces + 1
          sarea = sarea + sum(face_detwei)


          if(have_viscosity) then
            do gi=1,sngi
              force_at_quad(:, gi) = normal(:, gi) * face_pressure(gi) - matmul(normal(:, gi), tau(:, :, gi))
            end do
          else
            do gi=1,sngi
              force_at_quad(:, gi) = normal(:, gi) * face_pressure(gi)
            end do
          end if
          force = force + matmul(force_at_quad, face_detwei)

          if(present(pressure_force)) then
            do gi=1,sngi
              force_at_quad(:, gi) = normal(:, gi) * face_pressure(gi)
            end do
            pressure_force = pressure_force + matmul(force_at_quad, face_detwei)
          end if
          if(present(viscous_force)) then
            do gi=1,sngi
              force_at_quad(:, gi) = - matmul(normal(:, gi), tau(:, :, gi))
            end do
            viscous_force = viscous_force + matmul(force_at_quad, face_detwei)
          end if
       end if
    enddo

    call allsum(nfaces)
    call allsum(sarea)
    call allsum(force)
    if(present(pressure_force)) call allsum(pressure_force)
    if(present(viscous_force)) call allsum(viscous_force)

    ewrite(2,*) 'Integrated over this number of faces and total area: ', nfaces, sarea
    ewrite(2, *) "Force on surface: ", force
    if(present(pressure_force)) then
      ewrite(2,*) 'Pressure force on surface: ', pressure_force
    end if
    if(present(viscous_force)) then
      ewrite(2,*) 'Viscous force on surface: ', viscous_force
    end if

    call deallocate(augmented_shape)

    ewrite(1, *) "Exiting diagnostic_body_drag"

  end subroutine diagnostic_body_drag

  function calculate_surface_integral_scalar(s_field, positions, index) result(integral)
    !!< Calculates a surface integral for the specified scalar field based upon
    !!< options defined in the options tree

    type(scalar_field), intent(in) :: s_field
    type(vector_field), intent(in) :: positions
    integer, optional, intent(in) :: index

    real :: integral

    character(len = real_format_len() + 4) :: format_buffer
    character(len = FIELD_NAME_LEN) :: integral_name
    character(len = OPTION_PATH_LEN) :: path, integral_type
    integer :: lindex, max_index
    integer, dimension(2) :: shape
    integer, dimension(:), allocatable :: surface_ids
    logical :: normalise

    if(present(index)) then
      lindex = index
    else
      lindex = 0
    end if

    path = trim(complete_field_path(s_field%option_path)) // "/stat/surface_integral"

    max_index = option_count(trim(path)) - 1
    if(lindex < 0 .or. lindex > max_index) then
      ewrite(-1, "(a,i0,a,i0)") "Index: ", lindex, " - Max index: ", max_index
      FLAbort("Invalid option path index when calculating surface integral")
    end if

    path = trim(path) // "[" // int2str(lindex) // "]"

    if(have_option(trim(path) // "/surface_ids")) then
      shape = option_shape(trim(path) // "/surface_ids")
      assert(shape(1) >= 0)
      allocate(surface_ids(shape(1)))
      call get_option(trim(path) // "/surface_ids", surface_ids)
    end if

    normalise = have_option(trim(path) // "/normalise")

    call get_option(trim(path) // "/type", integral_type)
    select case(trim(integral_type))
      case("value")
        if(allocated(surface_ids)) then
          integral = surface_integral(s_field, positions, surface_ids = surface_ids, normalise = normalise)
        else
          integral = surface_integral(s_field, positions, normalise = normalise)
        end if
      case("gradient_normal")
        if(allocated(surface_ids)) then
          integral = gradient_normal_surface_integral(s_field, positions, surface_ids = surface_ids, normalise = normalise)
        else
          integral = gradient_normal_surface_integral(s_field, positions, normalise = normalise)
        end if
      case default
        FLAbort("Invalid scalar field surface integral type: " // trim(integral_type))
    end select

    if(allocated(surface_ids)) then
      deallocate(surface_ids)
    end if

    call get_option(trim(path) // "/name", integral_name)
    if(normalise) then
      ewrite(2, *) "Normalised surface integral of type " // trim(integral_type) // " for field " // trim(s_field%name) // ":"
    else
      ewrite(2, *) "Surface integral of type " // trim(integral_type) // " for field " // trim(s_field%name) // ":"
    end if
    format_buffer = "(a," // real_format() // ")"
    ewrite(2, format_buffer) trim(integral_name) // " = ", integral

  end function calculate_surface_integral_scalar

  function calculate_surface_integral_vector(v_field, positions, index) result(integral)
    !!< Calculates a surface integral for the specified vector field based upon
    !!< options defined in the options tree

    type(vector_field), intent(in) :: v_field
    type(vector_field), intent(in) :: positions
    integer, optional, intent(in) :: index

    real :: integral

    character(len = real_format_len() + 4) :: format_buffer
    character(len = FIELD_NAME_LEN) :: integral_name
    character(len = OPTION_PATH_LEN) :: path, integral_type
    integer :: lindex, max_index
    integer, dimension(2) :: shape
    integer, dimension(:), allocatable :: surface_ids
    logical :: normalise

    if(present(index)) then
      lindex = index
    else
      lindex = 0
    end if

    path = trim(complete_field_path(v_field%option_path)) // "/stat/surface_integral"

    max_index = option_count(trim(path)) - 1
    if(lindex < 0 .or. lindex > max_index) then
      ewrite(-1, "(a,i0,a,i0)") "Index: ", lindex, " - Max index: ", max_index
      FLAbort("Invalid option path index when calculating surface integral")
    end if

    path = trim(path) // "[" // int2str(lindex) // "]"

    if(have_option(trim(path) // "/surface_ids")) then
      shape = option_shape(trim(path) // "/surface_ids")
      assert(shape(1) >= 0)
      allocate(surface_ids(shape(1)))
      call get_option(trim(path) // "/surface_ids", surface_ids)
    end if

    normalise = have_option(trim(path) // "/normalise")

    call get_option(trim(path) // "/type", integral_type)
    select case(trim(integral_type))
      case("normal")
        if(allocated(surface_ids)) then
          integral = normal_surface_integral(v_field, positions, surface_ids = surface_ids, normalise = normalise)
        else
          integral = normal_surface_integral(v_field, positions, normalise = normalise)
        end if
      case default
        FLAbort("Invalid vector field surface integral type: " // trim(integral_type))
    end select

    if(allocated(surface_ids)) then
      deallocate(surface_ids)
    end if

    call get_option(trim(path) // "/name", integral_name)
    if(normalise) then
      ewrite(2, *) "Normalised surface integral of type " // trim(integral_type) // " for field " // trim(v_field%name) // ":"
    else
      ewrite(2, *) "Surface integral of type " // trim(integral_type) // " for field " // trim(v_field%name) // ":"
    end if
    format_buffer = "(a," // real_format() // ")"
    ewrite(2, format_buffer) trim(integral_name) // " = ", integral

  end function calculate_surface_integral_vector

end module surface_integrals
