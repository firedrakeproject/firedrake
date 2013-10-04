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
module MeshDiagnostics

  use transform_elements
  use elements
  use spud
  use fldebug
  use parallel_fields
  use parallel_tools
  use fields_base

  implicit none

  private

  public :: tetvol, triarea, &
          & simplex_volume, mesh_stats, pentahedron_vol

  interface mesh_stats
    module procedure mesh_stats_mesh, mesh_stats_scalar, mesh_stats_vector
  end interface mesh_stats

contains

  REAL FUNCTION PENTAHEDRON_VOL(X,Y,Z)
    !c
    !C Calculate the volume of a pentahedron, i.e. a polyhedron comprising 5 faces,
    !c (3 quadrilaterals and 2 triangles) and 6 vertices.
    !c Make the assumption that vertices 3,4,5,6 of X,Y,Z are one of
    !c these quadrilaterals. Volume calculated by splitting into 3 tetrahedra.
    !c To split into 3 tetrahedra here need to assume that node 1 is on the same
    !c face as nodes 3,4 and that node 2 is on the same face as nodes 5,6.
    !c
    REAL X(6), Y(6), Z(6)
    !C
    REAL XTET1(4), YTET1(4), ZTET1(4)
    REAL XTET2(4), YTET2(4), ZTET2(4)
    REAL XTET3(4), YTET3(4), ZTET3(4)
    !C
    XTET1(1) = X(6)
    XTET1(2) = X(4)
    XTET1(3) = X(3)
    XTET1(4) = X(2)

    YTET1(1) = Y(6)
    YTET1(2) = Y(4)
    YTET1(3) = Y(3)
    YTET1(4) = Y(2)

    ZTET1(1) = Z(6)
    ZTET1(2) = Z(4)
    ZTET1(3) = Z(3)
    ZTET1(4) = Z(2)

    XTET2(1) = X(6)
    XTET2(2) = X(3)
    XTET2(3) = X(5)
    XTET2(4) = X(2)

    YTET2(1) = Y(6)
    YTET2(2) = Y(3)
    YTET2(3) = Y(5)
    YTET2(4) = Y(2)

    ZTET2(1) = Z(6)
    ZTET2(2) = Z(3)
    ZTET2(3) = Z(5)
    ZTET2(4) = Z(2)

    XTET3(1) = X(4)
    XTET3(2) = X(3)
    XTET3(3) = X(2)
    XTET3(4) = X(1)

    YTET3(1) = Y(4)
    YTET3(2) = Y(3)
    YTET3(3) = Y(2)
    YTET3(4) = Y(1)

    ZTET3(1) = Z(4)
    ZTET3(2) = Z(3)
    ZTET3(3) = Z(2)
    ZTET3(4) = Z(1)

    PENTAHEDRON_VOL = abs(TETVOL(XTET1,YTET1,ZTET1)) &
         + abs(TETVOL(XTET2,YTET2,ZTET2)) &
         + abs(TETVOL(XTET3,YTET3,ZTET3))

    RETURN
  END FUNCTION PENTAHEDRON_VOL

  subroutine mesh_stats_mesh(mesh, nodes, elements, surface_elements, facets)
    !!< Parallel safe mesh statistics

    type(mesh_type), intent(in) :: mesh
    integer, optional, intent(out) :: nodes
    integer, optional, intent(out) :: elements
    integer, optional, intent(out) :: surface_elements
    integer, optional, intent(out) :: facets

    integer :: i, surface_facets

    if(present(nodes)) then
      if(isparallel()) then
        nodes = 0
        do i = 1, node_count(mesh)
          if(node_owned_mesh(mesh, i)) then
            nodes = nodes + 1
          end if
        end do
        call allsum(nodes)
      else
        nodes = node_count(mesh)
      end if
    end if

    if(present(elements)) then
      if(isparallel()) then
        elements = 0
        do i = 1, ele_count(mesh)
          if(element_owned(mesh, i)) then
            elements = elements + 1
          end if
        end do
        call allsum(elements)
      else
        elements = ele_count(mesh)
      end if
    end if

    if(present(surface_elements)) then
      if(isparallel()) then
        surface_elements = 0
        do i = 1, surface_element_count(mesh)
          if(surface_element_owned(mesh, i)) then
            surface_elements = surface_elements + 1
          end if
        end do
        call allsum(surface_elements)
      else
        surface_elements = surface_element_count(mesh)
      end if
    end if

    if(present(facets)) then
      if(isparallel()) then
        facets = 0
        do i = 1, face_count(mesh)
          if(surface_element_owned(mesh, i)) then
            facets = facets + 1
          end if
        end do
        if(present(surface_elements)) then
          ! this depends on facets being worked out after surface_elements
          surface_facets = surface_elements
        else
          surface_facets = 0
          do i = 1, surface_element_count(mesh)
            if(surface_element_owned(mesh, i)) then
              surface_facets = surface_facets + 1
            end if
          end do
          call allsum(surface_facets)
        end if
        facets = (facets-surface_facets)/2 + surface_facets
        call allsum(facets)
      else
        facets = (face_count(mesh)-surface_element_count(mesh))/2 &
                   + surface_element_count(mesh)
      end if
    end if

  end subroutine mesh_stats_mesh

  subroutine mesh_stats_scalar(s_field, nodes, elements, surface_elements)
    !!< Parallel safe mesh statistics

    type(scalar_field), intent(in) :: s_field
    integer, optional, intent(out) :: nodes
    integer, optional, intent(out) :: elements
    integer, optional, intent(out) :: surface_elements

    call mesh_stats(s_field%mesh, nodes = nodes, elements = elements, surface_elements = surface_elements)

  end subroutine mesh_stats_scalar

  subroutine mesh_stats_vector(v_field, nodes, elements, surface_elements)
    !!< Parallel safe mesh statistics

    type(vector_field), intent(in) :: v_field
    integer, optional, intent(out) :: nodes
    integer, optional, intent(out) :: elements
    integer, optional, intent(out) :: surface_elements

    call mesh_stats(v_field%mesh, nodes = nodes, elements = elements, surface_elements = surface_elements)

  end subroutine mesh_stats_vector

end module MeshDiagnostics
