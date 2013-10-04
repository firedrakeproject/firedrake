#include "fdebug.h"

module field_derivatives
    !!< This module contains code to compute the derivatives of
    !!< scalar fields.

    use elements
    use fetools, only: shape_shape, shape_dshape, dshape_outer_dshape
    use fields
    use halos
    use eventcounter
    use transform_elements
    use vector_tools
    use vector_set
    use node_boundary
    use surfacelabels
    use vtk_interfaces
    use state_module
    implicit none

    interface compute_hessian_real
      module procedure compute_hessian_var
    end interface

    interface differentiate_field_lumped
      module procedure differentiate_field_lumped_single, differentiate_field_lumped_multiple, &
          differentiate_field_lumped_vector
    end interface

    interface u_dot_nabla
      module procedure u_dot_nabla_scalar, &
        & u_dot_nabla_vector
    end interface u_dot_nabla

    interface grad
      module procedure grad_scalar, grad_vector, grad_vector_tensor
    end interface grad

    private

    public :: strain_rate, differentiate_field, grad, compute_hessian, &
      domain_is_2d, patch_type, get_patch_ele, get_patch_node, curl, &
      div, u_dot_nabla, differentiate_field_lumped

    public :: compute_hessian_var

    contains

    subroutine grad_scalar(infield, positions, gradient)
      !!< This routine computes the gradient of a field.
      !!< For a continuous gradient this lumps the mass matrix
      !!< in the Galerkin projection.
      type(scalar_field), intent(in) :: infield
      type(vector_field), intent(in) :: positions
      type(vector_field), intent(inout) :: gradient
      type(scalar_field), dimension(gradient%dim) :: pardiff
      logical, dimension(gradient%dim) :: derivatives
      integer :: i, dim

      dim = gradient%dim
      do i=1,dim
        pardiff(i) = extract_scalar_field(gradient, i)
      end do

      ! we need all derivatives
      derivatives = .true.

      call differentiate_field(infield, positions, derivatives, pardiff)

    end subroutine grad_scalar

    subroutine grad_vector(infield, positions, gradient)
      !!< This routine computes the gradient of a field.
      type(vector_field), intent(in) :: infield
      type(vector_field), intent(in) :: positions
      type(vector_field), dimension(infield%dim), intent(inout) :: gradient
      type(scalar_field), dimension(gradient(1)%dim) :: pardiff
      type(scalar_field) :: component
      logical, dimension(gradient(1)%dim) :: derivatives
      integer :: i, j, dim

      dim = gradient(1)%dim

      do j=1,infield%dim

        component = extract_scalar_field(infield, j)

        do i=1,dim
          pardiff(i) = extract_scalar_field(gradient(j), i)
        end do

        derivatives = .true.

        call differentiate_field(component, positions, derivatives, pardiff)
      end do

    end subroutine grad_vector

    subroutine grad_vector_tensor(infield,positions,t_field)
      !!< This routine computes the full (tensor) grad of an infield vector field
      type(vector_field), intent(in) :: infield
      type(vector_field), intent(in) :: positions
      type(tensor_field), intent(inout) :: t_field

      type(scalar_field), dimension(infield%dim) :: pardiff
      type(scalar_field) :: component

      real, dimension(t_field%dim(1),t_field%dim(2)) :: t
      logical, dimension(infield%dim) :: derivatives
      integer :: i, j
      integer :: node

      do j=1,infield%dim

        component = extract_scalar_field(infield, j)

        do i=1,infield%dim
          pardiff(i) = extract_scalar_field(t_field,i,j)
        end do

        derivatives = .true.

        call differentiate_field(component, positions, derivatives, pardiff)

      end do

    end subroutine grad_vector_tensor

    subroutine strain_rate(infield,positions,t_field)
      !!< This routine computes the strain rate of an infield vector field
      type(vector_field), intent(in) :: infield
      type(vector_field), intent(in) :: positions
      type(tensor_field), intent(inout) :: t_field

      type(scalar_field), dimension(infield%dim) :: pardiff
      type(scalar_field) :: component

      real, dimension(t_field%dim(1),t_field%dim(2)) :: t
      logical, dimension(infield%dim) :: derivatives
      integer :: i, j
      integer :: node

      do j=1,infield%dim

        component = extract_scalar_field(infield, j)

        do i=1,infield%dim
          pardiff(i) = extract_scalar_field(t_field,i,j)
        end do

        derivatives = .true.

        call differentiate_field(component, positions, derivatives, pardiff)

      end do

      ! Computing the final strain rate tensor

      do node=1,node_count(t_field)
           t=node_val(t_field, node)
           call set(t_field, node, (t+transpose(t))/2)
      end do

    end subroutine strain_rate

    subroutine compute_hessian_int(infield, positions, hessian)
    !!< This routine computes the hessian using integration by parts.
    !!< See Buscaglia and Dari, Int. J. Numer. Meth. Engng., 40, 4119-4136 (1997)
      type(scalar_field), intent(inout) :: infield
      type(vector_field), intent(in) :: positions
      type(tensor_field), intent(inout) :: hessian

      ! For now, assume only one element type in the mesh.

      real, dimension(ele_ngi(infield, 1)) :: detwei
      real, dimension(ele_loc(infield, 1), ele_ngi(infield, 1), mesh_dim(infield)) :: dt_t
      type(element_type), pointer :: t_shape
      real, dimension(mesh_dim(infield), mesh_dim(infield), ele_loc(infield, 1), ele_loc(infield, 1)) :: r
      real, dimension(mesh_dim(infield), mesh_dim(infield), ele_loc(infield, 1)) :: r_ele
      type(scalar_field), target  :: lumped_mass_matrix
      real, dimension(ele_loc(infield, 1), ele_loc(infield, 1)) :: mass_matrix
      type(mesh_type) :: mesh

      integer :: ele, node,i,j


      call zero(hessian)
      if (maxval(infield%val) == minval(infield%val)) then
        ewrite(2,*) "+++: Field constant; returning 0.0"
        return
      end if

      mesh = infield%mesh
      call add_nelist(mesh)
      call initialise_boundcount(infield%mesh, positions)

      call allocate(lumped_mass_matrix, infield%mesh, "Lumped mass matrix")
      call zero(lumped_mass_matrix)

      t_shape => ele_shape(infield, 1)

      do ele=1,element_count(infield)
        ! Compute detwei.
        call transform_to_physical(positions, ele, t_shape, dshape=dt_t, detwei=detwei)

        ! Compute the tensor representing grad(N) grad(N)
        r = dshape_outer_dshape(dt_t, dt_t, detwei)
        !r_ele = 0.5 * (tensormul(r, ele_val(infield, ele), 3) + tensormul(r, ele_val(infield, ele), 4))
        !r_ele = tensormul(r, ele_val(infield, ele), 4)

        r_ele = 0.
        do i = 1,size(r,1)
           do j = 1,size(r,2)
              r_ele(i,j,:) = r_ele(i,j,:) + &
                   matmul(r(i,j,:,:),ele_val(infield,ele))
           end do
        end do
        call addto(hessian, ele_nodes(infield, ele), r_ele)

        ! Lump the mass matrix
        mass_matrix = shape_shape(t_shape, t_shape, detwei)
        call addto(lumped_mass_matrix, ele_nodes(infield, ele), sum(mass_matrix, 2))
      end do

      do node=1,node_count(infield)
        hessian%val(:, :, node) = (-1.0 / node_val(lumped_mass_matrix, node)) * hessian%val(:, :, node)
        !hessian%val(:, :, node) = (-1) * hessian%val(:, :, node)
      end do

      call hessian_boundary_correction(hessian, positions, t_shape)
      call deallocate(lumped_mass_matrix)
    end subroutine compute_hessian_int

    subroutine differentiate_boundary_correction(pardiff, positions, t_shape, count)
      !!< Implement the boundary correction routine for first derivatives.
      type(scalar_field), dimension(:), intent(inout), target :: pardiff
      type(vector_field), intent(in) :: positions
      type(element_type), intent(in) :: t_shape
      integer, intent(in) :: count

      type(scalar_field) :: node_weights
      type(mesh_type), pointer :: mesh

      integer :: i, j, k, dim, node, nnode, ele
      integer, dimension(:), pointer :: neighbour_elements, neighbour_nodes

      real :: sum_weights
      real, dimension(ele_ngi(pardiff(1), 1)) :: detwei
      real, dimension(ele_loc(pardiff(1), 1), ele_loc(pardiff(1), 1)) :: mass_matrix
      real, dimension(mesh_dim(pardiff(1))) :: old_val

      logical :: has_neighbouring_interior_node
      type(patch_type) :: node_patch
      type(csr_sparsity), pointer :: nelist

      !type(scalar_field) :: boundcounts, node_numbers, patch_nodes

      mesh => pardiff(1)%mesh
      nelist => extract_nelist(mesh)
      call allocate(node_weights, mesh, "NodeWeights")

      dim = mesh_dim(mesh)

      call initialise_boundcount(mesh, positions)

      do i=1,dim
        do node=1,node_count(mesh)
          if (node_boundary_count(node) >= get_expected_boundcount() + i) then
            call zero(node_weights)
            has_neighbouring_interior_node = .false.
            ! First we need to compute the weights for each neighbouring node.
            neighbour_elements => row_m_ptr(nelist, node)
            do j=1,size(neighbour_elements)
              ele = neighbour_elements(j)
              neighbour_nodes => ele_nodes(mesh, ele)
              call transform_to_physical(positions, ele, detwei=detwei)
              mass_matrix = shape_shape(t_shape, t_shape, detwei)
              ! In words: find the row of the mass matrix corresponding to the node we're interested in,
              ! and stuff the integral of the shape functions into node_weights.
              call addto(node_weights, neighbour_nodes, mass_matrix(:, find(neighbour_nodes, node)))

              ! Also: find out if the node has /any/ neighbouring interior nodes.
              if (.not. has_neighbouring_interior_node) then
                do k=1,size(neighbour_nodes)
                  nnode = neighbour_nodes(k)
                  if (.not. node_lies_on_boundary(nnode)) then
                    has_neighbouring_interior_node = .true.
                    exit
                  end if
                end do
              end if
            end do

            ! Now that we have the weights, let us use them.

            node_patch = get_patch_node(mesh, node)
            sum_weights = 0.0
            forall (j=1:count)
              old_val(j) = pardiff(j)%val(node)
              pardiff(j)%val(node) = 0.0
            end forall
            do j=1,node_patch%count
              nnode = node_patch%elements(j)
              ! If it's on the boundary, no ...
              if (&
              (has_neighbouring_interior_node .and. &
              (.not. node_lies_on_boundary(nnode))) &
              .or. ((.not. has_neighbouring_interior_node) .and. node_boundary_count(nnode) < node_boundary_count(node))) then
                sum_weights = sum_weights + node_val(node_weights, nnode)
                do k=1,count
                  pardiff(k)%val(node) = pardiff(k)%val(node) +  &
                                         pardiff(k)%val(nnode) * node_val(node_weights, nnode)
                end do
              end if
            end do

            if (sum_weights == 0.0) then
              forall (k=1:count)
                pardiff(k)%val(node) = old_val(k)
              end forall
            else
              forall (k=1:count)
                pardiff(k)%val(node) = pardiff(k)%val(node) / sum_weights
              end forall
            end if
            deallocate(node_patch%elements)
          end if
        end do
      end do
      call deallocate(node_weights)
    end subroutine differentiate_boundary_correction

    subroutine hessian_boundary_correction(hessian, positions, t_shape)
      !!< Implement the hessian boundary correction routine.
      type(tensor_field), intent(inout), target :: hessian
      type(vector_field), intent(in) :: positions
      type(element_type), intent(in) :: t_shape

      type(scalar_field) :: node_weights
      type(mesh_type), pointer :: mesh

      integer :: i, j, k, dim, node, nnode, ele
      integer, dimension(:), pointer :: neighbour_elements, neighbour_nodes

      real :: sum_weights
      real, dimension(ele_ngi(hessian, 1)) :: detwei
      real, dimension(ele_loc(hessian, 1), ele_loc(hessian, 1)) :: mass_matrix
      real, dimension(hessian%dim(1), hessian%dim(2)) :: old_val

      logical :: has_neighbouring_interior_node
      type(patch_type) :: node_patch
      type(csr_sparsity), pointer :: nelist

      assert(hessian%dim(1)==hessian%dim(2))

      mesh => hessian%mesh
      nelist => extract_nelist(mesh)
      call allocate(node_weights, mesh, "Node weights")
      dim = hessian%dim(1)

      call initialise_boundcount(hessian%mesh, positions)

      do i=1,dim
        do node=1,node_count(hessian)
          if (node_boundary_count(node) >= get_expected_boundcount() + i) then
            call zero(node_weights)
            has_neighbouring_interior_node = .false.
            ! First we need to compute the weights for each neighbouring node.
            neighbour_elements => row_m_ptr(nelist, node)
            do j=1,size(neighbour_elements)
              ele = neighbour_elements(j)
              neighbour_nodes => ele_nodes(hessian, ele)
              call transform_to_physical(positions, ele, detwei=detwei)
              mass_matrix = shape_shape(t_shape, t_shape, detwei)
              ! In words: find the row of the mass matrix corresponding to the node we're interested in,
              ! and stuff the integral of the shape functions into node_weights.
              call addto(node_weights, neighbour_nodes, mass_matrix(:, find(neighbour_nodes, node)))

              ! Also: find out if the node has /any/ neighbouring interior nodes.
              if (.not. has_neighbouring_interior_node) then
                do k=1,size(neighbour_nodes)
                  nnode = neighbour_nodes(k)
                  if (.not. node_lies_on_boundary(nnode)) then
                    has_neighbouring_interior_node = .true.
                    exit
                  end if
                end do
              end if
            end do

            ! Now that we have the weights, let us use them.

            node_patch = get_patch_node(mesh, node)
            sum_weights = 0.0
            old_val = hessian%val(:, :, node)
            hessian%val(:, :, node) = 0.0
            do j=1,node_patch%count
              nnode = node_patch%elements(j)
              ! If it's on the boundary, no ...
              if (&
              (has_neighbouring_interior_node .and. &
              (.not. node_lies_on_boundary(nnode))) &
              .or. ((.not. has_neighbouring_interior_node) .and. node_boundary_count(nnode) < node_boundary_count(node))) then
                sum_weights = sum_weights + node_val(node_weights, nnode)
                hessian%val(:, :, node) = hessian%val(:, :, node) +  &
                                          hessian%val(:, :, nnode) * node_val(node_weights, nnode)
              end if
            end do

            if (sum_weights == 0.0) then
              hessian%val(:, :, node) = old_val
            else
              hessian%val(:, :, node) = hessian%val(:, :, node) / sum_weights
            end if
            deallocate(node_patch%elements)
          end if
        end do
      end do

      call deallocate(node_weights)
    end subroutine hessian_boundary_correction

    subroutine hessian_squash_pseudo2d(hessian)
      !!< Squash derivatives in directions where no dynamics occur.
      type(tensor_field), intent(inout) :: hessian

      integer :: node

      if (pseudo2d_coord == 0) then
        return
      end if

      do node=1,node_count(hessian)
        hessian%val(:, pseudo2d_coord, node) = 0.0
        hessian%val(pseudo2d_coord, :, node) = 0.0
      end do
    end subroutine hessian_squash_pseudo2d

    subroutine differentiate_squash_pseudo2d(pardiff)
      !!< Squash derivatives in directions where no dynamics occur.
      type(scalar_field), dimension(:), intent(inout) :: pardiff

      if (pseudo2d_coord == 0) then
        return
      end if

      call zero(pardiff(pseudo2d_coord))
    end subroutine differentiate_squash_pseudo2d

    function find(array, val) result(loc)
      !!< Find the first instance of val in array.
      integer, intent(in), dimension(:) :: array
      integer, intent(in) :: val
      integer :: i, loc

      loc = -1
      do i=1,size(array)
        if (array(i) == val) then
          loc = i
          return
        end if
      end do
    end function find

    subroutine compute_hessian_var(infield, positions, hessian)
    !!< This routine computes the hessian using a weak finite element formulation.
      type(scalar_field), intent(in) :: infield
      type(vector_field), intent(in) :: positions
      type(tensor_field), intent(inout), target :: hessian

      type(vector_field), target :: gradient
      type(mesh_type), pointer :: mesh

      real, dimension(ele_ngi(positions, 1)) :: detwei
      real, dimension(ele_loc(infield, 1), ele_ngi(infield, 1), mesh_dim(infield)) :: dt_t
      real, dimension(ele_loc(hessian, 1), ele_ngi(hessian, 1), mesh_dim(hessian)) :: dh_t
      type(element_type), pointer :: t_shape, h_shape
      real, dimension(mesh_dim(infield), ele_loc(hessian, 1), ele_loc(infield, 1)) :: r
      real, dimension(mesh_dim(infield), ele_loc(hessian, 1)) :: r_grad_ele
      real, dimension(mesh_dim(hessian), ele_loc(hessian, 1), ele_loc(hessian, 1)) :: r_hess
      real, dimension(mesh_dim(hessian), mesh_dim(hessian), ele_loc(hessian, 1)) :: r_hess_ele
      type(scalar_field)  :: lumped_mass_matrix
      real, dimension(ele_loc(hessian, 1), ele_loc(hessian, 1)) :: mass_matrix
      integer :: dim, i, j
      integer :: node, ele
      real, dimension(:, :), pointer :: hess_ptr

      mesh => hessian%mesh
      dim = mesh_dim(mesh)

      call zero(hessian)
      if (maxval(infield%val) == minval(infield%val)) then
        ewrite(2,*) "+++: Field constant; returning 0.0"
        return
      end if

      call allocate(lumped_mass_matrix, mesh, "Lumped mass matrix")
      call allocate(gradient, dim, mesh, "Gradient")

      call add_nelist(mesh)
      call initialise_boundcount(mesh, positions)

      call zero(lumped_mass_matrix)
      call zero(gradient)

      t_shape => ele_shape(infield, 1)
      h_shape => ele_shape(hessian, 1)

      ! First, compute gradient and mass matrix.
      do ele=1,element_count(infield)
        ! Compute detwei.
        call transform_to_physical(positions, ele, t_shape, dshape=dt_t, detwei=detwei)

        r = shape_dshape(h_shape, dt_t, detwei)
        r_grad_ele = tensormul(r, ele_val(infield, ele), 3)

        call addto(gradient, ele_nodes(gradient, ele), r_grad_ele)

        ! Lump the mass matrix
        mass_matrix = shape_shape(h_shape, h_shape, detwei)
        call addto(lumped_mass_matrix, ele_nodes(lumped_mass_matrix, ele), sum(mass_matrix, 2))
      end do

      do node=1,node_count(gradient)
        do i=1,dim
          gradient%val(i,node) = gradient%val(i,node) / node_val(lumped_mass_matrix, node)
        end do
      end do

      ! Testing: does this cause the lock exchange result to fail?
      !do i=1,dim
      !  grad_components(i) = extract_scalar_field(gradient, i)
      !end do
      !call differentiate_boundary_correction(grad_components, positions, x_shape, t_shape, dim)

      do ele=1,element_count(infield)
        call transform_to_physical(positions, ele, h_shape, dshape=dh_t, detwei=detwei)
        r_hess = shape_dshape(h_shape, dh_t, detwei)
        do i=1,dim
          r_hess_ele(i, :, :) = tensormul(r_hess, ele_val(gradient, i, ele), 3)
        end do
        call addto(hessian, ele_nodes(hessian, ele), r_hess_ele)
      end do

      do node=1,node_count(hessian)
        hess_ptr => hessian%val(:, :, node)
        hess_ptr = hess_ptr / node_val(lumped_mass_matrix, node)

        ! Now we need to make it symmetric, see?
        do i=1,dim
          do j=i+1,dim
            hess_ptr(i, j) = (hess_ptr(i, j) + hess_ptr(j, i)) / 2.0
            hess_ptr(j, i) = hess_ptr(i, j)
          end do
        end do
      end do

      call hessian_boundary_correction(hessian, positions, h_shape)

      call deallocate(lumped_mass_matrix)
      call deallocate(gradient)
    end subroutine compute_hessian_var

    subroutine differentiate_field_lumped_multiple(infields, positions, derivatives, pardiff)
    !!< This routine computes the first derivatives using a weak finite element formulation.
      type(scalar_field), dimension(:), intent(in) :: infields
      type(vector_field), intent(in) :: positions
      logical, dimension(:), intent(in) :: derivatives
      type(scalar_field), dimension(:,:), target, intent(inout) :: pardiff

      type(vector_field), dimension(size(infields)), target :: gradient
      type(mesh_type), pointer :: mesh

      type(scalar_field)  :: lumped_mass_matrix, inverse_lumped_mass
      logical, dimension( mesh_dim(infields(1)) ):: compute
      integer :: i, j, k
      integer :: ele

      mesh => pardiff(1, 1)%mesh

      do i=1, size(infields)
        ! don't compute if the field is constant
        compute(i)= (maxval(infields(i)%val) /= minval(infields(i)%val))
        ! check the infield is continuous!!!!
        if (infields(i)%mesh%continuity<0) then
          ewrite(0,*) "If the following error is directly due to user input"
          ewrite(0,*) "a check and a more helpful error message should be inserted in"
          ewrite(0,*) "the calling routine (outside field_derivatives) - please mantis this:"
          ewrite(0,*) "Error has occured in differentiate_field_lumped_multiple, with field, ", trim(infields(i)%name)
          FLAbort("The field_derivatives code cannot take the derivative of a discontinuous field")
        end if
      end do

      call allocate(lumped_mass_matrix, mesh, "LumpedMassMatrix")
      call zero(lumped_mass_matrix)

      do i=1, size(infields)
        if (compute(i)) then
          call allocate(gradient(i), positions%dim, mesh, "Gradient")
          call zero(gradient(i))
        end if
      end do


      ! First, compute gradient and mass matrix.
      do ele=1, element_count(mesh)
        call differentiate_field_ele(ele)
      end do

      do i=1, size(infields)
        if (compute(i)) then
          k=0
          do j=1, positions%dim
            if (derivatives(j)) then
              k=k+1
              call set( pardiff(k,i), gradient(i), dim=j )
            end if
          end do
        else
          do k=1, size(pardiff,1)
            call zero(pardiff(k,i))
          end do
        end if
      end do

      ! invert the lumped mass matrix
      call allocate(inverse_lumped_mass, mesh, "InverseLumpedMassMatrix")
      call invert(lumped_mass_matrix, inverse_lumped_mass)

      ! compute pardiff=M^-1*pardiff
      do i=1, size(infields)
        do k=1, size(pardiff,1)
          call scale(pardiff(k,i), inverse_lumped_mass)
        end do
      end do

      do i=1, size(infields)
        if (compute(i)) then
          call deallocate(gradient(i))
        end if
      end do
      call deallocate(lumped_mass_matrix)
      call deallocate(inverse_lumped_mass)

      contains

      subroutine differentiate_field_ele(ele)
        integer, intent(in):: ele

        real, dimension(mesh_dim(mesh), ele_loc(mesh, ele), ele_loc(infields(1), ele)) :: r
        real, dimension(ele_ngi(mesh, ele)) :: detwei
        real, dimension(ele_loc(infields(1), ele), ele_ngi(infields(1), ele), mesh_dim(infields(1))) :: dt_t
        real, dimension(ele_loc(mesh, ele), ele_loc(mesh, ele)) :: mass_matrix

        integer i

        ! Compute detwei.
        call transform_to_physical(positions, ele, &
           ele_shape(infields(1), ele), dshape=dt_t, detwei=detwei)

        r = shape_dshape(ele_shape(mesh, ele), dt_t, detwei)
        do i=1, size(infields)

          if (compute(i)) then
            call addto(gradient(i), ele_nodes(mesh, ele), &
               tensormul(r, ele_val(infields(i), ele), 3) )
          end if

        end do

        ! Lump the mass matrix
        mass_matrix = shape_shape(ele_shape(mesh, ele), ele_shape(mesh, ele), detwei)
        call addto(lumped_mass_matrix, ele_nodes(mesh, ele), sum(mass_matrix, 2))

      end subroutine differentiate_field_ele

    end subroutine differentiate_field_lumped_multiple

    subroutine differentiate_field_lumped_single(infield, positions, derivatives, pardiff)
    !!< This routine computes the first derivatives using a weak finite element formulation.
      type(scalar_field), intent(in), target :: infield
      type(vector_field), intent(in) :: positions
      logical, dimension(:), intent(in) :: derivatives
      type(scalar_field), dimension(:), intent(inout) :: pardiff

      type(scalar_field), dimension(size(pardiff),1) :: pardiffs

      pardiffs(:,1)=pardiff

      call differentiate_field_lumped_multiple( (/ infield /), positions, derivatives, pardiffs)

    end subroutine differentiate_field_lumped_single

    subroutine differentiate_field_lumped_vector(infield, positions, outfield)
    !!< This routine computes the derivatives of a vector field returning a tensor field
      type(vector_field), intent(in), target :: infield
      type(vector_field), intent(in) :: positions
      type(tensor_field), intent(inout) :: outfield

      logical, dimension( positions%dim ):: derivatives
      type(scalar_field), dimension( infield%dim ):: infields
      type(scalar_field), dimension( positions%dim, infield%dim ):: pardiffs
      integer i, j

      derivatives=.true.
      do i=1, infield%dim
        infields(i)=extract_scalar_field(infield, i)
        do j=1, positions%dim
          pardiffs(j, i)=extract_scalar_field(outfield, j, i)
        end do
      end do

      call differentiate_field_lumped_multiple( infields, positions, derivatives, pardiffs)

    end subroutine differentiate_field_lumped_vector

    subroutine differentiate_field(infield, positions, derivatives, pardiff)
      type(scalar_field), intent(in), target :: infield
      type(vector_field), intent(in) :: positions
      logical, dimension(:), intent(in) :: derivatives
      type(scalar_field), dimension(:), intent(inout) :: pardiff

      integer :: i
      type(mesh_type), pointer :: mesh

      if (infield%field_type == FIELD_TYPE_CONSTANT) then
        do i=1,count(derivatives)
          call zero(pardiff(i))
        end do
        return
      end if

      if (continuity(pardiff(1))<0) then
        call differentiate_field_discontinuous(infield, positions, derivatives, pardiff)
        return
      end if

      mesh => infield%mesh
      call add_nelist(mesh)
      call differentiate_field_lumped_single(infield, positions, derivatives, pardiff)

      if (pseudo2d_coord /= 0) then
        if (derivatives(pseudo2d_coord)) then
        ! which pardiff corresponds to dimension pseudo2d_coord?
        i = count(derivatives(1:pseudo2d_coord))
        call zero(pardiff(i))
        end if
      end if

    end subroutine differentiate_field

    subroutine differentiate_field_discontinuous(infield, positions, derivatives, pardiff)
      type(scalar_field), intent(in), target :: infield
      type(vector_field), intent(in) :: positions
      logical, dimension(:), intent(in) :: derivatives
      type(scalar_field), dimension(:), intent(inout) :: pardiff

      type(element_type) xshape, inshape, dershape
      real, dimension(ele_loc(infield,1), ele_ngi(infield,1), size(derivatives)):: dinshape
      real, dimension(size(derivatives), size(derivatives), ele_ngi(infield,1)):: invJ
      real, dimension(ele_loc(pardiff(1),1)):: r
      real, dimension(size(r), size(r)):: M
      real, dimension(size(derivatives), size(r), ele_loc(infield,1)):: Q
      real, dimension(ele_ngi(infield,1)):: detwei
      integer ele, gi, i, j

      if (infield%field_type == FIELD_TYPE_CONSTANT) then
        do i=1,count(derivatives)
          if (derivatives(i)) then
             call zero(pardiff(i))
          end if
        end do
        return
      end if

      ! only works if all pardiff fields are discontinuous:
      do i=1, count(derivatives)
        assert(pardiff(i)%mesh%continuity<0)
      end do
      ! and the infield is continuous!!!!
      if (infield%mesh%continuity<0) then
        ewrite(0,*) "If the following error is directly due to user input"
        ewrite(0,*) "a check and a more helpful error message should be inserted in"
        ewrite(0,*) "the calling routine (outside field_derivatives) - please mantis this:"
        ewrite(0,*) "Error has occured in differentiate_field_discontinuous, with field, ", trim(infield%name)
        FLAbort("The field_derivatives code cannot take the derivative of a discontinuous field")
      end if

      xshape=ele_shape(positions, 1)
      inshape=ele_shape(infield, 1)
      dershape=ele_shape(pardiff(1), 1)

      do ele=1, element_count(infield)

         ! calculate the transformed derivative of the shape function
         call compute_inverse_jacobian( ele_val(positions, ele), xshape, invJ, detwei=detwei)
         do gi=1, inshape%ngi
            do i=1, inshape%ndof
               dinshape(i,gi,:)=matmul(invJ(:,:,gi), inshape%dn(i,gi,:))
            end do
         end do

         M=shape_shape(dershape, dershape, detwei)
         Q=shape_dshape(dershape, dinshape, detwei)
         call invert(M)

         ! apply Galerkin projection M^{-1} Q \phi
         j=0
         do i=1, size(derivatives)
            if (derivatives(i)) then
               j=j+1
               r=matmul(M, matmul(Q(i,:,:), ele_val(infield, ele)))
               call set(pardiff(j), ele_nodes(pardiff(j), ele), r)
            end if
        end do

      end do

    end subroutine differentiate_field_discontinuous

    subroutine compute_hessian(infield, positions, hessian)
      type(scalar_field), intent(inout) :: infield
      type(vector_field), intent(in) :: positions
      type(tensor_field), intent(inout) :: hessian

      integer :: node

      if (infield%field_type == FIELD_TYPE_CONSTANT) then
        call zero(hessian)
        return
      end if

      call add_nelist(infield%mesh)
      call compute_hessian_real(infield, positions, hessian)

      if (pseudo2d_coord /= 0) then
        do node=1,node_count(hessian)
          hessian%val(pseudo2d_coord, :, node) = 0.0
          hessian%val(:, pseudo2d_coord, node) = 0.0
        end do
      end if
    end subroutine compute_hessian

    subroutine curl(infield, positions, curl_norm, curl_field)
      type(vector_field), intent(in) :: positions, infield
      type(scalar_field), intent(inout), optional :: curl_norm ! norm of curl_field
      type(vector_field), intent(inout), optional :: curl_field

      type(vector_field), dimension(positions%dim) :: grad_v
      integer :: i
      real :: w, a, b, c
      type(mesh_type) :: mesh

      assert(positions%dim == 3)

      mesh = infield%mesh
      call add_nelist(mesh)

      do i=1,positions%dim
        call allocate(grad_v(i), positions%dim, infield%mesh, "Grad V")
        call grad(extract_scalar_field(infield, i), positions, grad_v(i))
      end do

      if (present(curl_field)) then
        call zero(curl_field)
      end if

      if (present(curl_norm)) then
        call zero(curl_norm)
      end if

      do i=1,node_count(infield)
        a = grad_v(3)%val(2,i) - grad_v(2)%val(3,i) ! dw/dy - dv/dz
        b = grad_v(1)%val(3,i) - grad_v(3)%val(1,i) ! du/dz - dw/dx
        c = grad_v(2)%val(1,i) - grad_v(1)%val(2,i) ! dv/dx - du/dy
        if (present(curl_norm)) then
          w = sqrt(a**2 + b**2 + c**2)
          call addto(curl_norm, i, w)
        end if
        if (present(curl_field)) then
          call addto(curl_field, i, (/a, b, c/))
        end if
      end do

      do i=1,positions%dim
        call deallocate(grad_v(i))
      end do

    end subroutine curl

  subroutine u_dot_nabla_scalar(v_field, in_field, positions, out_field)
    !!< Calculates (u dot nabla) in_field for scalar fields

    type(vector_field), intent(in) :: v_field
    type(scalar_field), intent(in) :: in_field
    type(vector_field), intent(in) :: positions
    type(scalar_field), intent(inout) :: out_field

    integer :: i
    real, dimension(positions%dim) :: grad_val_at_node, &
      & v_field_val_at_node
    type(vector_field) :: gradient

    call allocate(gradient, positions%dim, in_field%mesh, "Gradient")

    call grad(in_field, positions, gradient)

    call zero(out_field)
    do i = 1, node_count(out_field)
      grad_val_at_node = node_val(gradient, i)
      v_field_val_at_node = node_val(v_field, i)
      call set(out_field, i, &
        & dot_product(v_field_val_at_node, grad_val_at_node))
    end do

    call deallocate(gradient)

  end subroutine u_dot_nabla_scalar

  subroutine u_dot_nabla_vector(v_field, in_field, positions, out_field)
    !!< Calculates (u dot nabla) in_field for vector fields

    type(vector_field), intent(in) :: v_field
    type(vector_field), intent(in) :: in_field
    type(vector_field), intent(in) :: positions
    type(vector_field), intent(inout) :: out_field

    integer :: i
    type(scalar_field) :: out_field_comp

    do i = 1, v_field%dim
      out_field_comp = extract_scalar_field(out_field, i)
      call u_dot_nabla(v_field, &
        & extract_scalar_field(in_field, i), positions, &
        & out_field_comp)
    end do

  end subroutine u_dot_nabla_vector

    subroutine div(infield, positions, divergence)
      !! Implement div() operator.
      type(vector_field), intent(in):: infield, positions
      type(scalar_field), intent(inout), target  :: divergence

      type(scalar_field) :: component
      type(scalar_field), dimension(1) :: derivative
      type(mesh_type), pointer :: mesh
      logical, dimension(mesh_dim(infield)) :: derivatives
      integer :: i

      mesh => divergence%mesh
      call allocate(derivative(1), mesh, "Derivative")

      call zero(divergence)
      derivatives = .false.

      do i=1,mesh_dim(infield)
        derivatives(i) = .true.
        component = extract_scalar_field(infield, i)
        call differentiate_field(component, positions, derivatives, derivative)
        call addto(divergence, derivative(1))
        derivatives = .false.
      end do

      call deallocate(derivative(1))
    end subroutine div

    function insphere_tet(positions) result(centre)
      !! dim x loc
      real, dimension(3, 4), intent(in) :: positions
      real, dimension(size(positions, 1)) :: centre

      real, dimension(size(positions, 1)) :: u, v, w, p, q, r, O1, O2, y, s
      real :: t

      u = positions(:, 2) - positions(:, 1)
      v = positions(:, 3) - positions(:, 1)
      w = positions(:, 4) - positions(:, 1)
      p = cross_product(u, v); p = p / norm2(p)
      q = cross_product(v, w); q = q / norm2(q)
      r = cross_product(w, u); r = r / norm2(r)

      O1 = p -q
      O2 = q - r
      y = cross_product(O1, O2)

      O1 = u - w
      O2 = v - w
      s = cross_product(O1, O2); s = -1 * (s / norm2(s))

      O1 = s - p
      t = dot_product(w, s) / dot_product(y, O1)
      centre = positions(:, 1) + t * y
    end function insphere_tet

end module field_derivatives
