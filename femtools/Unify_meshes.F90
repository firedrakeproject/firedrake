!    Copyright (C) 2006 Imperial College London and others.
!
!    Please see the AUTHORS file in the main source directory for a full list
!    of copyright holders.
!
!    Prof. C Pain
!    Applied Modelling and Computation Group
!    Department of Earth Science and Engineeringp
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

module unify_meshes_module

  use fields_data_types
  use fields_manipulation, only: set, remap_field, set_ele_nodes
  use fields_allocates
  use fields_base
  use fldebug
  implicit none

  interface unify_meshes
    module procedure unify_meshes_linear
  end interface

  contains

  function unify_meshes_linear(meshes) result(union)
    type(vector_field), intent(in), dimension(:) :: meshes
    type(vector_field) :: union
    integer :: mesh, nodes
    integer :: total_elements, ele_accum
    integer :: ele_mesh, i
    type(mesh_type) :: union_mesh
    integer :: loc
    integer, dimension(:), pointer :: old_nodes
    integer, dimension(ele_loc(meshes(1), 1)) :: new_nodes

    total_elements = 0
    do mesh=1,size(meshes)
      total_elements = total_elements + ele_count(meshes(mesh))
    end do
    loc = ele_loc(meshes(1), 1)
    nodes = total_elements * loc

    call allocate(union_mesh, nodes, total_elements, meshes(1)%mesh%shape, "AccumulatedMesh")
    union_mesh%continuity = -1
    if (associated(meshes(1)%mesh%region_ids)) then
       call allocate_region_ids(union_mesh, total_elements)
    end if
    call allocate(union, meshes(1)%dim, union_mesh, "AccumulatedPositions")
    call deallocate(union_mesh)

    ele_accum = 1

    do mesh=1,size(meshes)
      assert(continuity(meshes(mesh)) < 0)
      assert(ele_loc(meshes(mesh), 1) == loc)
      if (associated(union%mesh%region_ids)) then
        assert(associated(meshes(mesh)%mesh%region_ids))
        union%mesh%region_ids(ele_accum:ele_accum + ele_count(meshes(mesh))-1) = meshes(mesh)%mesh%region_ids
      end if

      do ele_mesh=1,ele_count(meshes(mesh))
        new_nodes = (/ (i, i=loc * (ele_accum-1)+1,loc*ele_accum) /)
        call set_ele_nodes(union%mesh, ele_accum, new_nodes)

        old_nodes => ele_nodes(meshes(mesh), ele_mesh)
        do i=1,size(old_nodes)
          call set(union, new_nodes(i), node_val(meshes(mesh), old_nodes(i)))
        end do

        ele_accum = ele_accum + 1
      end do
    end do

  end function unify_meshes_linear

  subroutine unify_meshes_quadratic(posA, posB, posC)
  ! Given two volume-disjoint discontinuous positions fields,
  ! unify them together. For example, this is useful for
  ! stitching together the supermesh.
  ! For now, we assume the element types (triangles/quads etc)
  ! are the same.
    type(vector_field), intent(in) :: posA, posB
    type(vector_field), intent(out) :: posC

    type(mesh_type) :: meshA, meshB, meshC, tmp_meshC
    type(vector_field) :: lposA, lposB

    integer :: eles, nodes
    integer :: ele, ele_accum, node_accum, i
    integer, dimension(:), pointer :: old_nodes
    integer, dimension(ele_loc(posA, 1)) :: new_nodes

    ewrite(1,*) "Warning! This algorithm is quadratic"

    if (continuity(posA) < 0) then
      meshA = posA%mesh
      call incref(meshA)
      lposA = posA
      call incref(lposA)
    else
      meshA = make_mesh(posA%mesh, posA%mesh%shape, -1, 'DiscontinuousVersion')
      call allocate(lposA, posA%dim, meshA, "DiscontinuousPosA")
      call remap_field(posA, lposA)
    end if

    if (continuity(posB) < 0) then
      meshB = posB%mesh
      call incref(meshB)
      lposB = posB
      call incref(lposB)
    else
      meshB = make_mesh(posB%mesh, posB%mesh%shape, -1, 'DiscontinuousVersion')
      call allocate(lposB, posB%dim, meshB, "DiscontinuousPosA")
      call remap_field(posB, lposB)
    end if

    eles = ele_count(lposA) + ele_count(lposB)
    nodes = node_count(lposA) + node_count(lposB)
    call allocate(tmp_meshC, nodes, eles, posA%mesh%shape, "AccumulatedMesh")
    meshC = make_mesh(tmp_meshC, tmp_meshC%shape, -1, "DiscontinuousAccumulatedMesh")
    call deallocate(tmp_meshC)

    if (associated(meshA%region_ids) .and. associated(meshB%region_ids)) then
      call allocate_region_ids(meshC, eles)
      meshC%region_ids(1:ele_count(lposA)) = meshA%region_ids
      meshC%region_ids(ele_count(lposA)+1:) = meshB%region_ids
    end if

    call allocate(posC, posA%dim, meshC, "AccumulatedPositions")

    ! Now fill in the ndglno and the positions.
    ele_accum = 1
    node_accum = 1
    do ele=1,ele_count(lposA)
      new_nodes = (/ (i, i=posA%mesh%shape%ndof * (ele_accum-1)+1,posA%mesh%shape%ndof*ele_accum) /)
      call set_ele_nodes(meshC, ele_accum, new_nodes)

      old_nodes => ele_nodes(lposA, ele)
      do i=1,size(old_nodes)
        call set(posC, new_nodes(i), node_val(lposA, old_nodes(i)))
      end do

      ele_accum = ele_accum + 1
    end do
    do ele=1,ele_count(lposB)
      new_nodes = (/ (i, i=posA%mesh%shape%ndof * (ele_accum-1)+1,posA%mesh%shape%ndof*ele_accum) /)
      call set_ele_nodes(meshC, ele_accum, new_nodes)

      old_nodes => ele_nodes(lposB, ele)
      do i=1,size(old_nodes)
        call set(posC, new_nodes(i), node_val(lposB, old_nodes(i)))
      end do

      ele_accum = ele_accum + 1
    end do

    call deallocate(meshC)

    call deallocate(meshA)
    call deallocate(meshB)
    call deallocate(lposA)
    call deallocate(lposB)
  end subroutine unify_meshes_quadratic
end module unify_meshes_module
