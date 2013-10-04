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

module colouring
  use fields
  use fields_manipulation
  use fields_base
  use field_options, only : find_linear_parent_mesh
  use data_structures
  use sparse_tools
  use global_parameters, only : topology_mesh_name, NUM_COLOURINGS, &
       COLOURING_CG1, COLOURING_DG0, COLOURING_DG2, &
       COLOURING_DG1
  use state_module, only : state_type, extract_mesh
  use sparsity_patterns_meshes, only : get_csr_sparsity_secondorder, &
       get_csr_sparsity_firstorder
  implicit none

  public :: colour_sparsity, verify_colour_sparsity, verify_colour_ispsparsity
  public :: colour_sets, get_mesh_colouring

contains



  ! Converts the matrix sparsity to an isp sparsity which can then be coloured to reduce the number
  ! of function evaluations needed to compute a (sparse) Jacobian via differencing.
  ! This function returns S^T*S if S is the given sparsity matrix.
  ! The resulting sparsity matrix is symmetric.
  function mat_sparsity_to_isp_sparsity(sparsity_in) result(sparsity_out)
    type(csr_sparsity), intent(in) :: sparsity_in
    type(csr_sparsity) :: sparsity_out
    type(csr_sparsity) :: sparsity_in_T

    sparsity_in_T=transpose(sparsity_in)
    sparsity_out=matmul(sparsity_in_T, sparsity_in)
    call deallocate(sparsity_in_T)

  end function mat_sparsity_to_isp_sparsity

  ! Return a colouring of a mesh that is thread safe for a particular
  ! assembly type.  All elements with the same colour in the returned
  ! colouring are safe to assemble concurrently.
  !
  ! For a given mesh topology there are four possible colourings
  !
  ! o Level 1 node: For CG assembly
  !                 [COLOURING_CG1]
  ! o Level 0 element: For DG assembly without viscosity
  !                    [COLOURING_DG0]
  ! o Level 1 element: For assembly with cjc's trace elements
  !                    [COLOURING_DG1]
  ! o Level 2 element: For DG assembly with viscosity
  !                    [COLOURING_DG2]
  !
  ! These colourings don't change between adapts, so we cache them on
  ! the topology mesh on first construction and subsequently pull
  ! them out of the cache.
  subroutine get_mesh_colouring(state, mesh, colouring_type, colouring)
    type(state_type), intent(inout) :: state
    type(mesh_type), intent(inout) :: mesh
    integer, intent(in) :: colouring_type
    type(integer_set), dimension(:), pointer, intent(out) :: colouring
    type(mesh_type), pointer :: topology
    type(csr_sparsity), pointer :: sparsity
    type(mesh_type) :: p0_mesh
    integer :: ncolours
    integer :: stat
    integer :: i
    type(scalar_field) :: element_colours

    topology => extract_mesh(state, topology_mesh_name)

    colouring => topology%colourings(colouring_type)%sets
    if (associated(colouring)) return

    ! If we reach here then the colouring has not yet been constructed.

#ifdef _OPENMP
    ! Use the sparsity patterns to find the dependency stencil.
    ! Greedily colour the sparsity graph
    ! Map this colouring back onto elements
    p0_mesh = piecewise_constant_mesh(topology, "P0Mesh")

    select case(colouring_type)
    case(COLOURING_CG1)
       ! Level 1 node
       sparsity => get_csr_sparsity_secondorder(state, p0_mesh, topology)
    case(COLOURING_DG0)
       ! Easy, just one colour
       ! So nothing to do here.
    case(COLOURING_DG2)
       ! Level 2 element
       sparsity => get_csr_sparsity_secondorder(state, p0_mesh, p0_mesh)
    case(COLOURING_DG1)
       ! Level 1 element
       sparsity => get_csr_sparsity_firstorder(state, p0_mesh, p0_mesh)
    case default
       FLAbort('Invalid colouring type specified')
    end select
    ! Colour the resulting sparsity
    ! Need to special case for DG_NO_VISCOSITY
    if ( colouring_type .eq. COLOURING_DG0 ) then
       allocate(colouring(1))
       call allocate(colouring)
       do i=1, element_count(mesh)
          call insert(colouring(1), i)
       end do
    else
       call colour_sparsity(sparsity, p0_mesh, element_colours, ncolours)
       allocate(colouring(ncolours))
       colouring = colour_sets(sparsity, element_colours, ncolours)
       call deallocate(element_colours)
    end if
    call deallocate(p0_mesh)
    topology%colourings(colouring_type)%sets => colouring
#else
    allocate(colouring(1))
    call allocate(colouring)
    do i=1, element_count(mesh)
       call insert(colouring(1), i)
    end do
    topology%colourings(colouring_type)%sets => colouring
#endif

  end subroutine get_mesh_colouring

  ! This routine colours a graph using the greedy approach.
  ! It takes as argument the sparsity of the adjacency matrix of the graph
  ! (i.e. the matrix is node X nodes and symmetric for undirected graphs).
  subroutine colour_sparsity(sparsity, mesh, node_colour, no_colours)
    type(csr_sparsity), intent(in) :: sparsity
    type(mesh_type), intent(inout) :: mesh
    type(scalar_field), intent(out) :: node_colour
    integer, intent(out) :: no_colours

    integer, dimension(:), pointer:: cols
    type(integer_set) :: neigh_colours
    integer :: i, node

    call allocate(node_colour, mesh, "NodeColouring")

    ! Set the first node colour
    call set(node_colour, 1, 1.0)
    no_colours = 1

    ! Colour remaining nodes.
    do node=2, size(sparsity,1)
       call allocate(neigh_colours)
       ! Determine colour of neighbours.
       cols => row_m_ptr(sparsity, node)
       do i=1, size(cols)
          if(cols(i)<node) then
            call insert(neigh_colours, nint(node_val(node_colour,cols(i))))
          end if
       end do

       ! Find the lowest unused colour in neighbourhood.
       do i=1, no_colours+1
          if(.not.has_value(neigh_colours, i)) then
             call set(node_colour, node, float(i))
             if(i>no_colours) then
                no_colours = i
             end if
             exit
          end if
       end do
       call deallocate(neigh_colours)
    end do

  end subroutine colour_sparsity

  ! Checks if a sparsity colouring is valid.
  function verify_colour_sparsity(sparsity, node_colour) result(valid)
    type(csr_sparsity), intent(in) :: sparsity
    type(scalar_field), intent(in) :: node_colour
    logical :: valid
    integer :: i, node
    real :: my_colour
    integer, dimension(:), pointer:: cols

    valid=.true.
    do node=1, size(sparsity, 1)
      cols => row_m_ptr(sparsity, node)
      my_colour=node_val(node_colour, node)
      ! Each nonzero column is a neighbour of node, so lets make sure that they do not have the same colour.
      do i=1, size(cols)
        if (cols(i)<node .and. my_colour==node_val(node_colour, cols(i))) then
          valid=.false.
        end if
      end do
    end do
  end function verify_colour_sparsity

  ! Checks if a sparsity colouring of a matrix is valid for isp.
  ! This method checks that no two columns of the same colour have
  ! nonzeros at the same positions.
  function verify_colour_ispsparsity(mat_sparsity, node_colour) result(valid)
    type(csr_sparsity), intent(in) :: mat_sparsity
    type(scalar_field), intent(in) :: node_colour
    logical :: valid
    integer :: i, row
    integer, dimension(:), pointer:: cols
    type(integer_set) :: neigh_colours

    valid=.true.
    do row=1, size(mat_sparsity, 1)
      call allocate(neigh_colours)
      cols => row_m_ptr(mat_sparsity, row)
      do i=1, size(cols)
        if (has_value(neigh_colours, nint(node_val(node_colour, cols(i))))) then
          valid=.false.
        end if
        call insert(neigh_colours, nint(node_val(node_colour,cols(i))))
      end do
      call deallocate(neigh_colours)
    end do
  end function verify_colour_ispsparsity

  ! with above colour_sparsity, we get map:node_id --> colour
  ! now we want map: colour --> node_ids
  function colour_sets(sparsity, node_colour, no_colours) result(clr_sets)
    type(csr_sparsity), intent(in) :: sparsity
    type(scalar_field), intent(in) :: node_colour
    integer, intent(in) :: no_colours
    type(integer_set), dimension(no_colours) :: clr_sets
    integer :: node

    call allocate(clr_sets)
    do node=1, size(sparsity, 1)
       call insert(clr_sets(nint(node_val(node_colour, node))), node)
    end do

  end function colour_sets

end module colouring
