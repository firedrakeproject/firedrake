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

module sparsity_patterns
  ! This module produces sparsity patterns for matrices.
  use linked_lists
  use elements
  use fields_base
  implicit none

contains

  function make_sparsity(rowmesh, colmesh, name) result (sparsity)
    ! Produce the sparsity of a first degree operator mapping from colmesh
    ! to rowmesh.
    type(ilist), dimension(:), pointer :: list_matrix
    type(csr_sparsity) :: sparsity
    type(mesh_type), intent(in) :: colmesh, rowmesh
    character(len=*), intent(in) :: name

    integer :: row_count, i

    row_count=node_count(rowmesh)

    allocate(list_matrix(row_count))

    list_matrix=make_sparsity_lists(rowmesh, colmesh)

    sparsity=lists2csr_sparsity(list_matrix, name)
    sparsity%columns=node_count(colmesh)
    sparsity%sorted_rows=.true.

    do i=1,row_count
       call flush_list(list_matrix(i))
    end do

    ! Since this is a degree one operator, use the first halo.
    if (associated(rowmesh%halos)) then
       allocate(sparsity%row_halo)
       sparsity%row_halo=rowmesh%halos(1)
       call incref(sparsity%row_halo)
    end if
    if (associated(colmesh%halos)) then
       allocate(sparsity%column_halo)
       sparsity%column_halo=colmesh%halos(1)
       call incref(sparsity%column_halo)
    end if

    deallocate(list_matrix)

  end function make_sparsity

  function make_sparsity_transpose(outsidemesh, insidemesh, name) result (sparsity)
    ! Produce the sparsity of a second degree operator formed by the
    ! operation C^TC. C is insidemesh by outsidemesh so the resulting
    ! sparsity is outsidemesh squared.
    type(csr_sparsity) :: sparsity
    type(mesh_type), intent(in) :: outsidemesh, insidemesh
    character(len=*), intent(in) :: name

    type(ilist), dimension(:), pointer :: list_matrix, list_matrix_out
    integer :: row_count, row_count_out, i, j, k
    integer, dimension(:), allocatable :: row

    row_count=node_count(insidemesh)
    row_count_out=node_count(outsidemesh)

    allocate(list_matrix(row_count))
    allocate(list_matrix_out(row_count_out))

    ! Generate the first order operator C.
    list_matrix=make_sparsity_lists(insidemesh, outsidemesh)


    ! Generate the sparsity of C^TC
    do i=1,row_count
       allocate(row(list_matrix(i)%length))

       row=list2vector(list_matrix(i))

       do j=1,size(row)
          do k=1,size(row)
             call insert_ascending(list_matrix_out(row(j)),row(k))
          end do
       end do

       deallocate(row)
    end do

    sparsity=lists2csr_sparsity(list_matrix_out, name)
    ! Note that the resulting sparsity is outsidemesh square.
    sparsity%columns=node_count(outsidemesh)
    sparsity%sorted_rows=.true.

    ! Second order operater so halo(2)
    if (associated(outsidemesh%halos)) then
       allocate(sparsity%row_halo)
       sparsity%row_halo=outsidemesh%halos(2)
       call incref(sparsity%row_halo)
       allocate(sparsity%column_halo)
       sparsity%column_halo=outsidemesh%halos(2)
       call incref(sparsity%column_halo)
    end if

    do i=1,row_count
       call flush_list(list_matrix(i))
    end do
    do i=1,row_count_out
       call flush_list(list_matrix_out(i))
    end do

    deallocate(list_matrix, list_matrix_out)

  end function make_sparsity_transpose

  function make_sparsity_mult(mesh1, mesh2, mesh3, name) result (sparsity)
    ! Produce the sparsity of a second degree operator formed by the
    ! operation A B, where A is mesh1 x mesh2 and B is mesh2 x mesh3
    type(csr_sparsity) :: sparsity
    type(mesh_type), intent(in) :: mesh1, mesh2, mesh3
    character(len=*), intent(in) :: name

    type(ilist), dimension(:), pointer :: list_matrix_1, list_matrix_3, list_matrix_out
    integer :: count_1, count_2, count_3, i, j, k
    integer, dimension(:), allocatable :: row_1, row_3

    count_1=node_count(mesh1)
    count_2=node_count(mesh2)
    count_3=node_count(mesh3)

    allocate(list_matrix_1(count_2))
    allocate(list_matrix_3(count_2))
    allocate(list_matrix_out(count_1))

    list_matrix_1=make_sparsity_lists(mesh2, mesh1)
    list_matrix_3=make_sparsity_lists(mesh2, mesh3)

    ! Generate the sparsity of A B
    do i=1,count_2
       allocate(row_1(list_matrix_1(i)%length))
       allocate(row_3(list_matrix_3(i)%length))

       row_1=list2vector(list_matrix_1(i))
       row_3=list2vector(list_matrix_3(i))

       do j=1,size(row_3)
          do k=1,size(row_1)
             call insert_ascending(list_matrix_out(row_1(k)),row_3(j))
          end do
       end do

       deallocate(row_1)
       deallocate(row_3)
    end do

    sparsity=lists2csr_sparsity(list_matrix_out, name)
    sparsity%columns=count_3
    sparsity%sorted_rows=.true.

    ! Second order operater so halo(2)
    if (associated(mesh1%halos)) then
       assert(associated(mesh3%halos))
       allocate(sparsity%row_halo)
       sparsity%row_halo=mesh1%halos(2)
       call incref(sparsity%row_halo)
       allocate(sparsity%column_halo)
       sparsity%column_halo=mesh3%halos(2)
       call incref(sparsity%column_halo)
    end if

    call deallocate(list_matrix_1)
    call deallocate(list_matrix_3)
    call deallocate(list_matrix_out)
    deallocate(list_matrix_1, list_matrix_3, list_matrix_out)

  end function make_sparsity_mult

  function make_sparsity_dg_mass(mesh) result (sparsity)
    !!< Produce the sparsity pattern of a DG mass matrix. These matrices
    !!< are block diagonal as there is no communication between the
    !!< elements.
    !!<
    !!< Note that this currently assumes that the mesh has uniform
    !!< elements.
    type(csr_sparsity) :: sparsity
    type(mesh_type), intent(in) :: mesh

    integer :: nonzeros, nodes, elements, nloc
    integer :: i,j,k

    nloc=mesh%shape%ndof ! Nodes per element
    nodes=node_count(mesh) ! Total nodes
    elements=element_count(mesh) ! Total elements
    nonzeros=nloc**2*elements

    call allocate(sparsity, rows=nodes, columns=nodes,&
         & entries=nonzeros, diag=.false., name="DGMassSparsity")

    forall (i=1:elements, j=1:nloc, k=1:nloc)
       sparsity%colm((i-1)*nloc**2+(j-1)*nloc+k)&
            =(i-1)*nloc+k
    end forall

    forall (i=1:nodes+1)
       sparsity%findrm(i)=(i-1)*nloc+1
    end forall
    sparsity%sorted_rows=.true.

    if (associated(mesh%halos)) then
       allocate(sparsity%row_halo)
       sparsity%row_halo=mesh%halos(1)
       call incref(sparsity%row_halo)
       allocate(sparsity%column_halo)
       sparsity%column_halo=mesh%halos(1)
       call incref(sparsity%column_halo)
    end if

  end function make_sparsity_dg_mass

  function make_sparsity_compactdgdouble(&
       mesh, name) result (sparsity)
    !!< Produce the sparsity pattern of a second order compact dg stencil.
    !!< This means: Each node is coupled to the nodes in that element,
    !!< plus the nodes in all other elements which share a face with that
    !!< element.
    !!<
    !!< Note that this currently assumes that the mesh has uniform
    !!< elements.
    type(ilist), dimension(:), pointer :: list_matrix
    type(csr_sparsity) :: sparsity
    type(mesh_type), intent(in) :: mesh
    character(len=*), intent(in) :: name

    integer :: row_count, i

    row_count=node_count(mesh)

    allocate(list_matrix(row_count))

    list_matrix=make_sparsity_lists(mesh,mesh, &
         & include_all_neighbour_element_nodes = .true.)

    sparsity=lists2csr_sparsity(list_matrix, name)
    sparsity%columns=node_count(mesh)
    sparsity%sorted_rows=.true.

    do i=1,row_count
       call flush_list(list_matrix(i))
    end do

    deallocate(list_matrix)

    ! Since the operator is compact, the level 1 halo should suffice.
    if (associated(mesh%halos)) then
       allocate(sparsity%row_halo)
       sparsity%row_halo=mesh%halos(1)
       call incref(sparsity%row_halo)
       allocate(sparsity%column_halo)
       sparsity%column_halo=mesh%halos(1)
       call incref(sparsity%column_halo)
    end if

  end function make_sparsity_compactdgdouble

  function make_sparsity_lists(rowmesh, colmesh, &
       & include_all_neighbour_element_nodes) result (list_matrix)
    ! Produce the sparsity of a first degree operator mapping from colmesh
    ! to rowmesh. Return a listmatrix
    ! Note this really ought to be mesh_type.
    type(mesh_type), intent(in) :: colmesh, rowmesh
    type(ilist), dimension(node_count(rowmesh)) :: list_matrix
    logical, intent(in), optional :: &
         & include_all_neighbour_element_nodes

    integer :: ele, i, j, face, neigh
    integer, dimension(:), pointer :: row_ele, col_ele, face_ele, col_neigh

    logical :: l_include_all_neighbour_element_nodes

    l_include_all_neighbour_element_nodes = .false.
    if(present(include_all_neighbour_element_nodes)) then
       l_include_all_neighbour_element_nodes = &
            & include_all_neighbour_element_nodes
    end if

    ! this should happen automatically through the initialisations
    ! statements, but not in old gcc4s:
    list_matrix%length=0

    do ele=1,element_count(rowmesh)
       row_ele=>ele_nodes(rowmesh, ele)
       col_ele=>ele_nodes(colmesh, ele)

       do i=1,size(row_ele)
          do j=1,size(col_ele)
             ! Every node in row_ele receives contribution from every node
             ! in col_ele
             call insert_ascending(list_matrix(row_ele(i)),col_ele(j))
          end do
       end do

    end do

    ! add in entries for boundary integrals if both row and column mesh are discontinuous
    ! if rowmesh is continuous then we're only interested in coupling between continuous face nodes
    !   and discontinuous nodes on the same face pair - the connection to the discontinuous nodes on the other
    !   side will be added from the adjacent element, so nothing extra to do in this case
    if (continuity(colmesh)<0 .and. (continuity(rowmesh)<0 .or. l_include_all_neighbour_element_nodes)) then
       assert(has_faces(colmesh))

       do ele=1,element_count(colmesh)
          row_ele=>ele_nodes(rowmesh, ele)
          col_neigh=>ele_neigh(colmesh, ele)

          do neigh=1,size(col_neigh)
             ! Skip external faces
             if (col_neigh(neigh)<=0) cycle

             face=ele_face(colmesh, col_neigh(neigh), ele)
             face_ele=>face_local_nodes(colmesh, face)
             col_ele=>ele_nodes(colmesh, col_neigh(neigh))

             if(l_include_all_neighbour_element_nodes) then
                do i=1,size(row_ele)
                   do j=1,size(col_ele)
                      call insert_ascending(list_matrix(row_ele(i))&
                           &,col_ele(j))
                   end do
                end do
             else

                do i=1,size(row_ele)
                   do j=1,size(face_ele)
                      call insert_ascending(list_matrix(row_ele(i))&
                           &,col_ele(face_ele(j)))
                   end do
                end do
             end if
          end do

       end do

    end if

  end function make_sparsity_lists

  function lists2csr_sparsity(lists, name) result (sparsity)
    ! Take a dynamically assembled set of row lists and return a sparsity.
    type(csr_sparsity) :: sparsity
    character(len=*), intent(in):: name
    type(ilist), dimension(:), intent(in) :: lists

    integer :: i, count, pos

    integer :: columns

    ! We have to figure out how many columns we have in this matrix
    columns = -1
    do i=1,size(lists)
      if(lists(i)%length/=0) columns = max(columns, maxval(lists(i)))
    end do

    call allocate(sparsity, rows=size(lists), columns=columns, &
         entries=sum(lists(:)%length), name=name)

    ! Lay out space for column indices.
    count=1
    do i=1,size(lists)
       sparsity%findrm(i)=count
       count=count+lists(i)%length
    end do
    sparsity%findrm(size(lists)+1)=count

    ! Insert column indices.
    do i=1,size(lists)
       sparsity%colm(sparsity%findrm(i):sparsity%findrm(i+1)-1)&
            &=list2vector(lists(i))
    end do

    ! Find diagonal.
    do i=1,size(sparsity%centrm)
       pos=0
       pos=minloc(row_m(sparsity,i), dim=1, mask=row_m(sparsity,i)==i)
       if (pos>0) then
          sparsity%centrm(i)=sparsity%findrm(i)+pos-1
       else
          sparsity%centrm(i)=0
          ! The following warning was removed as it produces too many
          ! spurious warnings.
          !ewrite(2,*) "Warning: missing diagonal in lists2csr_sparsity"
       end if
    end do

  end function lists2csr_sparsity

end module sparsity_patterns
