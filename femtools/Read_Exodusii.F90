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

module read_exodusii
  ! This module reads ExodusII files and results in a vector field of
  ! node coordinates, their connectivity, and node sets (grouped nodes
  ! given an ID, e.g. for setting physical boundaries)

  use iso_c_binding, only: C_INT, C_FLOAT, C_CHAR, C_NULL_CHAR
  use futils
  use elements
  use fields
  use state_module
  use spud
  use vtk_interfaces
  use exodusii_common
  use exodusii_f_interface
  use global_parameters, only : OPTION_PATH_LEN, real_4

  implicit none

  private

  interface read_exodusii_file
     module procedure  read_exodusii_file_to_field, &
                       read_exodusii_simple, &
                       read_exodusii_file_to_state
  end interface

  public :: read_exodusii_file, identify_exodusii_file

contains

  function read_exodusii_simple(filename, quad_degree, &
       quad_ngi, quad_family ) result (field)
    !!< A simpler mechanism for reading an ExodusII file into a field.
    character(len=*), intent(in) :: filename
    !! The degree of the quadrature.
    integer, intent(in), optional, target :: quad_degree
    !! The degree of the quadrature.
    integer, intent(in), optional, target :: quad_ngi
    !! What quadrature family to use
    integer, intent(in), optional :: quad_family
    type(vector_field) :: field
    type(quadrature_type) :: quad
    type(element_type) :: shape
    integer :: dim, ndof

    ewrite(1,*) "In read_exodusii_simple"

    if(isparallel()) then
       !call identify_exodusii_file(parallel_filename(filename), dim, ndof)
       FLExit("Currently we cannot read in a decomposed Exodus mesh file")
    else
       call identify_exodusii_file(filename, dim, ndof)
    end if

    if (present(quad_degree)) then
       quad = make_quadrature(ndof, dim, degree=quad_degree, family=quad_family)
    else if (present(quad_ngi)) then
       quad = make_quadrature(ndof, dim, ngi=quad_ngi, family=quad_family)
    else
       FLAbort("Need to specify either quadrature degree or ngi")
    end if

    shape=make_element_shape(ndof, dim, 1, quad)

    field=read_exodusii_file(filename, shape)

    call deallocate_element(shape)
    call deallocate(quad)

    ewrite(2,*) "Out of read_exodusii_simple"

  end function read_exodusii_simple

  ! -----------------------------------------------------------------
  ! ExodusII version of gmsh/triangle equivalent.
  subroutine identify_exodusii_file(filename, numDimenOut, locOut, &
       numElementsOut, boundaryFlagOut)
    ! Discover the dimension and size of the ExodusII mesh.
    ! Filename is the base name of the file without the
    ! ExodusII extension .e .exo .E .EXO

    character(len=*), intent(in) :: filename

    !! Number of vertices of elements.
    integer, intent(out), optional :: numDimenOut, locOut
    integer, intent(out), optional :: numElementsOut
    integer, intent(out), optional :: boundaryFlagOut

    integer :: boundaryFlag

    logical :: fileExists

    integer :: exoid, ierr
    real(kind=c_float) :: version
    integer(kind=c_int) :: comp_ws, io_ws, mode
    character(kind=c_char, len=OPTION_PATH_LEN) :: lfilename

    character(kind=c_char, len=OPTION_PATH_LEN) :: title
    integer :: num_dim, num_nodes, num_allelem, num_elem_blk
    integer :: num_node_sets, num_side_sets
    integer, allocatable, dimension(:) :: block_ids, num_elem_in_block, num_nodes_per_elem

    ewrite(1,*) "In identify_exodusii_file"

    call get_exodusii_filename(filename, lfilename, fileExists)
    if(.not. fileExists) then
       FLExit("None of the possible ExodusII files "//trim(filename)//".exo /.e /.EXO /.E were found")
    end if

    ewrite(2, *) "Opening " // trim(lfilename) // " for reading in database parameters"

    version = 0.0
    mode = 0; comp_ws=0; io_ws=0;
    exoid = f_read_ex_open(trim(lfilename)//C_NULL_CHAR, mode, comp_ws, io_ws, version)

    if (exoid <= 0) then
      FLExit("Unable to open "//trim(lfilename))
    end if

    ! Get database parameters from exodusII file
    ierr = f_ex_get_init(exoid, title, num_dim, num_nodes, &
                       num_allelem, num_elem_blk, num_node_sets, &
                       num_side_sets)
    if (ierr /= 0) then
       FLExit("Unable to read database parameters from "//trim(lfilename))
    end if

    ! Check for boundaries (internal boundaries currently not supported):
    if (num_side_sets /= 0) then
       boundaryFlag = 1 ! physical boundaries found
    else
       boundaryFlag = 0 ! no physical boundaries defined
    end if

    ! Get num_nodes_per_elem
    allocate(block_ids(num_elem_blk))
    allocate(num_elem_in_block(num_elem_blk))
    allocate(num_nodes_per_elem(num_elem_blk))
    ierr = f_ex_get_elem_block_parameters(exoid, num_elem_blk, block_ids, num_elem_in_block, num_nodes_per_elem)
    if (ierr /= 0) then
       FLExit("Unable to read in block parameters from "//trim(lfilename))
    end if

    ierr = f_ex_close(exoid)
    if (ierr /= 0) then
       FLExit("Unable to close file "//trim(lfilename))
    end if

    ! Return optional variables requested
    if(present(numDimenOut)) numDimenOut=num_dim
    ! numElementsOut is set to be the total number of volume and surface
    ! elements in the exodusII mesh, as only that is stored in the header
    ! of the mesh file. In Fluidity, only the volume elements of the mesh
    ! file are taken into account. Thus the actual number of volume
    ! elements is computed when the CoordinateMesh is assembled.
    if(present(numElementsOut)) numElementsOut=num_allelem
    ! Here we assume all (volume) elements of the mesh have the same
    ! number of vertices/nodes. Since the exodusII mesh could contain
    ! surface elements (as described above) we set locOut to be the
    ! max number of nodes per elements. Checking for hybrid meshes,
    ! that are currently not supported in Fluidity, is done when
    ! assembling the CoordinateMesh.
    if(present(locOut)) locOut=maxval(num_nodes_per_elem)
    if(present(boundaryFlagOut)) boundaryFlagOut=boundaryFlag

    ewrite(2,*) "Out of identify_exodusii_file"

  end subroutine identify_exodusii_file

  ! -----------------------------------------------------------------
  ! The main function for reading ExodusII files
  function read_exodusii_file_to_field(filename, shape) result (field)
    character(len=*), intent(in) :: filename
    type(element_type), intent(in), target :: shape
    type(vector_field)  :: field
    type(mesh_type) :: mesh

    logical :: fileExists
    logical :: haveRegionIDs, haveBoundaries

    type(EXOelement), pointer :: exo_element(:), exo_face(:), allelements(:)

    ! exodusii lib basic variables:
    integer :: exoid, ierr
    real(kind=c_float) :: version
    integer(kind=c_int) :: comp_ws, io_ws, mode
    character(kind=c_char, len=OPTION_PATH_LEN) :: lfilename
    character(kind=c_char, len=OPTION_PATH_LEN) :: title
    integer :: num_dim, num_nodes, num_allelem, num_elem_blk
    integer :: num_node_sets, num_side_sets

    ! exodusii lib variables:
    real(real_4), allocatable, dimension(:) :: coord_x, coord_y, coord_z
    integer, allocatable, dimension(:) :: node_map, elem_num_map, elem_order_map
    integer, allocatable, dimension(:) :: block_ids, num_elem_in_block, num_nodes_per_elem
    integer, allocatable, dimension(:) :: elem_type, elem_connectivity
    integer, allocatable, dimension(:) :: side_set_ids, num_elem_in_set
    integer, allocatable, dimension(:) :: total_side_sets_node_list, total_side_sets_elem_list
    integer, allocatable, dimension(:) :: total_side_sets_node_cnt_list

    ! variables for conversion to fluidity structure:
    real(real_4), allocatable, dimension(:,:) :: node_coord
    integer, allocatable, dimension(:) :: total_elem_node_list
    integer, allocatable, dimension(:) :: sndglno, boundaryIDs

    integer :: num_faces, num_elem, faceType
    integer :: ndof, sloc
    integer :: eff_dim, f

    ewrite(1,*) "In read_exodusii_file_to_field"

    ! First of all: Identify the filename:
    call get_exodusii_filename(filename, lfilename, fileExists)
    if(.not. fileExists) then
       FLExit("None of the possible ExodusII files "//trim(filename)//".exo /.e /.EXO /.E were found")
    end if

    ewrite(2, *) "Opening " // trim(lfilename) // " for reading in the mesh"

    version = 0.0
    mode = 0; comp_ws=0; io_ws=0;
    exoid = f_read_ex_open(trim(lfilename)//C_NULL_CHAR, mode, comp_ws, io_ws, version)

    if (exoid <= 0) then
      FLExit("Unable to open "//trim(lfilename))
    end if

    ! Get database parameters from exodusII file
    ierr = f_ex_get_init(exoid, title, num_dim, num_nodes, &
                       num_allelem, num_elem_blk, num_node_sets, &
                       num_side_sets)
    if (ierr /= 0) then
       FLExit("Unable to read database parameters from "//trim(lfilename))
    end if

    ! Catch user mistake of setting node sets instead of side sets:
    ! Give the user an error message, since node sets are not supported here, only side sets:
    if (num_node_sets > 0) then
       ! Maybe a warning might be better here, instead of FLExit:
       ewrite(-1,*) "You have specified node sets on your ExodusII meshfile '"//trim(lfilename)//"' but node sets are not supported by Fluidity. Please set your boundary conditions as side sets"
       FLExit("Node sets are not supported by Fluidity, use side sets instead.")
    end if

    ! read nodal coordinates values and names from database
    allocate(coord_x(num_nodes))
    allocate(coord_y(num_nodes))
    allocate(coord_z(num_nodes))
    coord_x=0.0; coord_y=0.0; coord_z=0.0
    ! Get coordinates from the mesh:
    ierr = f_ex_get_coord(exoid, coord_x, coord_y, coord_z)
    if (ierr /= 0) then
       FLExit("Unable to read in node coordinates "//trim(lfilename))
    end if

    ! Read node number map:
    allocate(node_map(num_nodes))
    ierr = f_ex_get_node_num_map(exoid, node_map)
    if (ierr /= 0) then
       FLExit("Unable to read in node number map from "//trim(lfilename))
    end if

    ! read element number map
    allocate(elem_num_map(num_allelem))
    elem_num_map = 0
    ierr = f_ex_get_elem_num_map(exoid, elem_num_map)
    if (ierr /= 0) then
       FLExit("Unable to read in element number map "//trim(lfilename))
    end if

    ! read element order map
    allocate(elem_order_map(num_allelem))
    elem_order_map = 0
    ierr = f_ex_get_elem_order_map(exoid, elem_order_map)
    if (ierr /= 0) then
       FLExit("Unable to read in element order map "//trim(lfilename))
    end if

    ! Get block ids:
    allocate(block_ids(num_elem_blk))
    ierr = f_ex_get_elem_blk_ids(exoid, block_ids)
    if (ierr /= 0) then
       FLExit("Unable to read in element block ids from "//trim(lfilename))
    end if

    ! Get block parameters:
    allocate(num_elem_in_block(num_elem_blk))
    allocate(num_nodes_per_elem(num_elem_blk))
    allocate(elem_type(num_elem_blk))
    call get_block_param(exoid, lfilename, block_ids, num_elem_blk, num_elem_in_block, num_nodes_per_elem, elem_type)

    ! Get faceType and give the user an error if he supplied a mesh with an unsupported combination of element types:
    call check_combination_face_element_types(num_dim, num_elem_blk, elem_type, lfilename, faceType)

    ! read element connectivity:
    allocate(elem_connectivity(0))
    call get_element_connectivity(exoid, block_ids, num_elem_blk, num_nodes_per_elem, num_elem_in_block, lfilename, elem_connectivity)

    ! Initialize logical variables:
    ! We have RegionIDs when there are blockIDs assigned to elements
    ! so basically always when supplying an exodusII mesh, as an blockID is assigned
    ! to all elements of the mesh if the user does not specify an blockID manually
    haveRegionIDs = .true. ! redundant for reasons stated above, but kept here to keep it consistent with gmshreader for now
    ! Boundaries: Boundaries are present if at least one side-set was supplied by the user:
    if (num_side_sets .gt. 0) then
       haveBoundaries = .true.
    else
       haveBoundaries = .false.
    end if
    ! Get side sets
    ! Side sets in exodusii are what physical lines/surfaces are in gmsh (so basically boundary-IDs)
    ! Allocate arrays for the side sets:
    ! Get Side SetIDs and parameters:
    if (haveBoundaries) then
       allocate(side_set_ids(num_side_sets))
       allocate(num_elem_in_set(num_side_sets)) ! There are the same # of elements as sides in a side set
       ! Allocate return arrays of the subroutine get_side_set_param:
       allocate(total_side_sets_elem_list(0)); allocate(total_side_sets_node_list(0)); allocate(total_side_sets_node_cnt_list(0))
       call get_side_set_param(exoid, num_side_sets, lfilename, side_set_ids, num_elem_in_set, total_side_sets_elem_list, total_side_sets_node_list, total_side_sets_node_cnt_list)
    end if

    ! Close ExodusII meshfile
    ierr = f_ex_close(exoid)
    if (ierr /= 0) then
       FLExit("Unable close file "//trim(lfilename))
    end if


    !---------------------------------
    ! At this point, all relevant data has been read in from the exodusii file
    ! Now construct within Fluidity data structures

    if( num_dim .eq. 2 .and. have_option("/geometry/spherical_earth/") ) then
       eff_dim = num_dim+1
    else
       eff_dim = num_dim
    end if

    ! Reorder element node numbering (if necessary):
    ! (allows for different element types)
    allocate(total_elem_node_list(0))
    call reorder_node_numbering(num_elem_blk, num_nodes_per_elem, num_elem_in_block, elem_connectivity, elem_type, total_elem_node_list)

    ! check if number of vertices/nodes are consistent with shape
    ndof = maxval(num_nodes_per_elem)
    assert(ndof==shape%ndof)

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! Coordinates              !
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! Loop around nodes copying across coords
    ! First, assemble array containing all node coordinates:
    allocate(node_coord(eff_dim, num_nodes))
    node_coord = 0
    node_coord(1,:) = coord_x(:)
    if (eff_dim .eq. 2 .or. eff_dim .eq. 3) then
       node_coord(2,:) = coord_y(:)
    end if
    if (eff_dim .eq. 3) then
       node_coord(3,:) = coord_z(:)
    end if

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! Assemble allelements (without faces) !
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    allocate(allelements(num_allelem))
    call assemble_allelements(num_elem_blk, block_ids, num_elem_in_block, &
                               num_nodes_per_elem, elem_order_map, elem_type, &
                               total_elem_node_list, allelements)
    ! At this stage 'allelements' contains all elements (faces and elements) of all blocks of the mesh
    ! Now, in case we have side sets/boundary ids in the mesh, assigns those ids to allelements:
    if (haveBoundaries) then
       call allelements_add_boundary_ids(num_side_sets, num_elem_in_set, total_side_sets_elem_list, side_set_ids, allelements)
    end if
   ! At this stage, the elements of 'allelements' have been correctly tagged,
   ! meaning they carry the side set ID(s) as tags, which later will
   ! become the boundary ID of their face(s)

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! Identify Faces           !
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! Now faces:
    ! First of all: Face elements in the mesh (-file) are neglected, thus get the
    ! number of elements (-minus number of face-elements in the mesh),
    ! and by 'elements' we mean no edges/faces in 2D/3D.
    ! And then subtract the amount of elements that carry at least one side-set-id
    ! In 2D: Faces are lines/edges
    ! In 3D: Faces are surfaces
    ! Find total number of such faces in all blocks
    ! loop over blocks, check for element type in block i,
    ! and depending on the mesh dimension, determine if element e is a face or element
    ! This does not support a 1D mesh,
    ! because you do NOT want to use fancy cubit to create a 1D mesh, do you?!
    ! Get number of elements and faces in the mesh:
    num_elem = 0; num_faces = 0
    sloc = 0
    call get_num_elem(num_dim, num_elem_blk, elem_type, num_elem_in_block, num_elem)

    ! Set sloc based on faceType:
    ! Only faceTypes 1, 2, and 3 are allowed (see above), their corresponding sloc is faceType+1,
    ! e.g. if faceType = 2 (triangle), it should have 3 nodes, which is faceType+1
    sloc = faceType+1
    ! Now check for site-set-id/physical-id, if element has numTags>0,
    ! than an edge/face will be generated next to that element,
    ! and the element's side-set-ID will be assigned to the newly generated edge/face
    if (haveBoundaries) then
       call get_num_faces(num_allelem, allelements, num_faces)
    end if

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! Setting Elements and faces !
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! Now actually set the elements and face-elements:
    ! assemble array with faces (exo_face contains element number (=element id of mesh))
    allocate(exo_element(num_elem))
    call assemble_actual_elements(num_dim, num_elem_blk, num_elem_in_block, elem_type, allElements, exo_element)


    ! Now derive the faces for fluidity, that are based on elements with side-set-ID:
    if (haveBoundaries) then
       allocate(exo_face(num_faces))
       call assemble_actual_face_elements(num_side_sets, num_elem_in_set, side_set_ids, &
                                           total_side_sets_node_cnt_list, total_side_sets_node_list, &
                                           exo_face)
    end if
    ! faces and elements are now all set

    ! Assemble the CoordinateMesh:
    call allocate(mesh, num_nodes, num_elem, shape, name="CoordinateMesh")
    call allocate(field, eff_dim, mesh, name="Coordinate")

    !!!!!!!!!!!!!!!!!!!!!!!!!!
    ! Copy Node IDs to field !
    !!!!!!!!!!!!!!!!!!!!!!!!!!
    call adding_nodes_to_field(eff_dim, num_nodes, node_map, node_coord, field)

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! Copy (only) Elements to the mesh !
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! RegionIDs in fluidity are blockIDs in exodusII:
    if (haveRegionIDs) then
      call allocate_region_ids(field%mesh, num_elem)
      field%mesh%region_ids = 0
    end if
    call adding_elements_to_field(num_dim, num_elem, exo_element, field)

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! Assemble array with faces and boundaryIDs !
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    allocate(sndglno(1:num_faces*sloc))
    sndglno=0
    if(haveBoundaries) then
       allocate(boundaryIDs(1:num_faces))
    end if
    do f=1, num_faces
       sndglno((f-1)*sloc+1:f*sloc) = exo_face(f)%nodeIDs(1:sloc)
       if(haveBoundaries) boundaryIDs(f) = exo_face(f)%tags(1)
    end do

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! Adding the face-elements to the mesh !
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if (haveBoundaries) then
       call add_faces(field%mesh, sndgln = sndglno(1:num_faces*sloc), boundary_ids = boundaryIDs(1:num_faces))
    else
       call add_faces(field%mesh, sndgln = sndglno(1:num_faces*sloc))
    end if

    ! At this point, the field 'CoordinateMesh' is assembled

    ! Deallocate arrays (exodusii arrays):
    deallocate(coord_x); deallocate(coord_y); deallocate(coord_z)
    deallocate(node_map); deallocate(elem_num_map); deallocate(elem_order_map);
    deallocate(block_ids); deallocate(num_elem_in_block); deallocate(num_nodes_per_elem);
    deallocate(elem_type)
    deallocate(elem_connectivity);

    ! Deallocate other arrays:
    deallocate(node_coord); deallocate(total_elem_node_list)
    if (haveBoundaries) then
       deallocate(side_set_ids);
       deallocate(total_side_sets_elem_list); deallocate(total_side_sets_node_list)
       deallocate(total_side_sets_node_cnt_list); deallocate(num_elem_in_set)
    end if
    deallocate(allelements)
    deallocate(exo_element)
    if (haveBoundaries) then
       deallocate(exo_face)
    end if
    call deallocate( mesh )

    if (haveBoundaries) then
       deallocate(boundaryIDs)
    end if
    deallocate(sndglno)

    ewrite(2,*) "Out of read_exodusii_file_to_field"

  end function read_exodusii_file_to_field

  ! -----------------------------------------------------------------

  subroutine get_block_param(exoid, lfilename, &
                              block_ids, &
                              num_elem_blk, &
                              num_elem_in_block, &
                              num_nodes_per_elem, &
                              elem_type)
    ! This subroutine get block specific data from the mesh file,
    ! e.g. the element type (Triangle/Quad/Tet/Hex...)
    ! number of elements (and nodes per element) per block:
    integer, intent(in) :: exoid
    character(kind=c_char, len=OPTION_PATH_LEN), intent(in) :: lfilename
    integer, dimension(:), intent(in) :: block_ids
    integer, intent(in) :: num_elem_blk
    integer, dimension(:), intent(inout) :: num_elem_in_block
    integer, dimension(:), intent(inout) :: num_nodes_per_elem
    integer, dimension(:), intent(inout) :: elem_type

    character(len=6) :: elem_type_char
    integer, allocatable, dimension(:) :: num_attr
    integer :: i, ierr

       ! Get block parameters:
       allocate(num_attr(num_elem_blk))
       ! Loop over the blocks in the mesh and get block specific data:
       do i=1, num_elem_blk
          ierr = f_ex_get_elem_block(exoid, block_ids(i), elem_type_char, &
                                     num_elem_in_block(i), &
                                     num_nodes_per_elem(i), &
                                     num_attr(i))
          ! assemble array to hold integers determining the element type
          ! element type names in exodusii are:
          ! Integer to element type relation (same as for gmsh):
          ! 1: BAR2 (line)
          ! 2: TRI3 (triangle)
          ! 3: SHELL4 (quad)
          ! 4: TETRA (tetrahedra)
          ! 5: HEX8 (hexahedron)
          ! assemble array to hold integers to identify element type of element block i:
          if (trim(elem_type_char(1:4)) .eq. "BAR2") then
             elem_type(i) = 1
          else if (trim(elem_type_char(1:4)) .eq. "TRI3") then
             elem_type(i) = 2
          else if (trim(elem_type_char(1:6)) .eq. "SHELL4") then
             elem_type(i) = 3
          else if (trim(elem_type_char(1:5)) .eq. "TETRA") then
             elem_type(i) = 4
          else if (trim(elem_type_char(1:4)) .eq. "HEX8") then
             elem_type(i) = 5
          else ! element type is not supported, give user an error
             ewrite(-1,*) "Mesh file "//trim(lfilename)//": Fluidity does not support meshes with elements of type"//trim(elem_type_char(1:6))//". Please provide a mesh with Triangles, Shells, Tetrahedras or Hexahedrons."
             FLExit("Element type of "//trim(elem_type_char(1:6))//" is not supported.")
          end if
       end do
       if (ierr /= 0) then
          FLExit("Unable to read in element block parameters from "//trim(lfilename))
       end if
       deallocate(num_attr)

  end subroutine get_block_param

  ! -----------------------------------------------------------------

  subroutine check_combination_face_element_types(num_dim, &
                                                   num_elem_blk, &
                                                   elem_type, &
                                                   lfilename, &
                                                   faceType)
    ! This subroutine get block specific data from the mesh file,
    ! e.g. the element type (Triangle/Quad/Tet/Hex...)
    ! number of elements (and nodes per element) per block:
    integer, intent(in) :: num_dim
    integer, intent(in) :: num_elem_blk
    integer, dimension(:), intent(in) :: elem_type
    character(kind=c_char, len=OPTION_PATH_LEN), intent(in) :: lfilename
    integer, intent(inout) :: faceType

    integer :: i, elementType

       elementType = 0; faceType = 0
       ! Practically looping over the blocks, and checking the combination of face/element types for
       ! each block in the supplied mesh, plus exit if dimension of mesh is 1!
       do i=1, num_elem_blk
          ! 2D meshes:
          if (num_dim == 2) then
             if (elem_type(i) .ne. 1) then !then it's no edge, but either triangle or shell
                if (elementType .ne. 0 .and. elementType .ne. elem_type(i)) then
                   ewrite(-1,*) "Mesh file "//trim(lfilename)//": You have generated a hybrid 2D mesh with Triangles and Shells which Fluidity does not support. Please choose either Triangles or Shells."
                   FLExit("Hybrid meshes are not supported by Fluidity.")
                end if
             end if
             ! the face type of 2D meshes are obviously edges, aka type '1'
             faceType = 1
          ! Now 3D meshes:
          else if (num_dim == 3) then
             if (elem_type(i) .ne. 2 .and. elem_type(i) .ne. 3) then !then it's not a triangle nor a shell
                if (elementType .ne. 0 .and. elementType .ne. elem_type(i)) then
                   ewrite(-1,*) "Mesh file "//trim(lfilename)//": You have generated a hybrid 3D mesh with Tetrahedras and Hexahedrons which Fluidity does not support. Please choose either Tetrahedras or Hexahedrons."
                   FLExit("Hybrid meshes are not supported by Fluidity.")
                end if
             end if
             if (elem_type(i) == 4) then ! tet
                ! Set faceType for tets
                faceType = 2
             else if (elem_type(i) == 5) then !hex
                ! Set faceType for hexas
                faceType = 3
             end if
          elementType = elem_type(i)
          else
             ewrite(-1,*) "Mesh file "//trim(lfilename)//": Fluidity currently does not support 1D exodusII meshes. But you do NOT want to use fancy cubit to create a 1D mesh, do you? GMSH or other meshing tools can easily be used to generate 1D meshes"
             FLExit("1D exodusII mesh files are not supported.")
          end if
       end do

  end subroutine check_combination_face_element_types

  ! -----------------------------------------------------------------

  subroutine get_element_connectivity(exoid, &
                                        block_ids, &
                                        num_elem_blk, &
                                        num_nodes_per_elem, &
                                        num_elem_in_block, &
                                        lfilename, &
                                        elem_connectivity)
    ! This subroutine gets the element connectivity of the given mesh file
    integer, intent(in) :: exoid
    integer, dimension(:), intent(in) :: block_ids
    integer, intent(in) :: num_elem_blk
    integer, dimension(:), intent(in) :: num_nodes_per_elem
    integer, dimension(:), intent(in) :: num_elem_in_block
    character(kind=c_char, len=OPTION_PATH_LEN), intent(in) :: lfilename
    integer, dimension(:), allocatable, intent(inout) :: elem_connectivity

    integer, dimension(:), allocatable :: elem_blk_connectivity
    integer :: i, ierr

      do i=1, num_elem_blk
         ! Get element connectivity of block 'i' and append to global element connectivity:
         allocate(elem_blk_connectivity(num_nodes_per_elem(i) * num_elem_in_block(i)))
         ierr = f_ex_get_elem_connectivity(exoid, block_ids(i), elem_blk_connectivity)
         call append_array(elem_connectivity, elem_blk_connectivity)
         deallocate(elem_blk_connectivity)
      end do
      if (ierr /= 0) then
         FLExit("Unable to read in element connectivity from "//trim(lfilename))
      end if

  end subroutine get_element_connectivity

  ! -----------------------------------------------------------------

  subroutine get_side_set_param(exoid, &
                                  num_side_sets, &
                                  lfilename, &
                                  side_set_ids, &
                                  num_elem_in_set, &
                                  total_side_sets_elem_list, &
                                  total_side_sets_node_list, &
                                  total_side_sets_node_cnt_list)
    integer, intent(in) :: exoid
    integer, intent(in) :: num_side_sets
    character(kind=c_char, len=OPTION_PATH_LEN), intent(in) :: lfilename
    integer, dimension(:), intent(inout) :: side_set_ids
    integer, dimension(:), intent(inout) :: num_elem_in_set
    integer, allocatable, dimension(:), intent(inout) :: total_side_sets_elem_list
    integer, allocatable, dimension(:), intent(inout) :: total_side_sets_node_list
    integer, allocatable, dimension(:), intent(inout) :: total_side_sets_node_cnt_list


    integer, allocatable, dimension(:) :: num_sides_in_set, num_df_in_set
    integer, allocatable, dimension(:) :: side_set_node_list, side_set_side_list
    integer, allocatable, dimension(:) :: side_set_elem_list, side_set_node_cnt_list
    integer, allocatable, dimension(:) :: elem_node_list

    integer :: i, e, n, ierr
    ! This subroutine gives back side set related data

       allocate(num_sides_in_set(num_side_sets)); allocate(num_df_in_set(num_side_sets));
       side_set_ids=0; num_sides_in_set=0; num_df_in_set=0;
       ierr = f_ex_get_side_set_ids(exoid, side_set_ids);
       if (ierr /= 0) then
          ewrite(2,*) "No side sets found in "//trim(lfilename)
       end if

      ! Get side set parameters:
       do i=1, num_side_sets
          ierr = f_ex_get_side_set_param(exoid, side_set_ids(i), num_sides_in_set(i), num_df_in_set(i));
       end do
       if (ierr /= 0) then
          FLExit("Unable to read in the side set parameters from "//trim(lfilename))
       end if

       num_elem_in_set = num_sides_in_set; ! There are as many elements in a side set, as there are sides in a side set
       ! Now let's finally get the side-set-ids!
       do i=1, num_side_sets
          ! Reset the node index to 1 for the ith side set:
          n = 1
          ! Arrays for side list and element list of side sets:
          allocate(side_set_elem_list(num_elem_in_set(i))); allocate(side_set_side_list(num_sides_in_set(i)))
          ! Arrays needed to obtain the node list:
          allocate(side_set_node_list(num_df_in_set(i))); allocate(side_set_node_cnt_list(num_elem_in_set(i)))

          ! Get side set ids, element list, side list
          ierr = f_ex_get_side_set(exoid, side_set_ids(i), side_set_elem_list, side_set_side_list)
          ! Get side set node list:
          ierr = f_ex_get_side_set_node_list(exoid, side_set_ids(i), side_set_node_cnt_list, side_set_node_list)

          ! In case the present element is a hexahedron, its face should be a quad... otherwise sth seriously went wrong
          ! Thus, renumber the node list of that quad before adding the node list to the global array:
          do e=1, num_elem_in_set(i)
             if (side_set_node_cnt_list(e) == 4) then
                allocate(elem_node_list(side_set_node_cnt_list(e)))
                ! Copy relevant nodes to a tmp array:
                elem_node_list(:) = side_set_node_list( n : n+side_set_node_cnt_list(e)-1 )
                ! Renumber the local node list of that face
                call toFluidityElementNodeOrdering( elem_node_list, 3 )
                ! After renumbering the face-nodes, copy them back into side_set_node_list:
                side_set_node_list( n : n+side_set_node_cnt_list(e)-1 ) = elem_node_list(:)
                deallocate(elem_node_list)
             end if
             ! Increase the node-index by number of nodes in element e of side-set i
             n = n + side_set_node_cnt_list(e)
          end do

          ! append the side set element list in global array for later:
          call append_array(total_side_sets_elem_list, side_set_elem_list)
          call append_array(total_side_sets_node_list, side_set_node_list)
          call append_array(total_side_sets_node_cnt_list, side_set_node_cnt_list)
          deallocate(side_set_elem_list); deallocate(side_set_side_list)
          deallocate(side_set_node_list); deallocate(side_set_node_cnt_list)
       end do

       ! Deallocate whatever we do not need anymore:
       deallocate(num_sides_in_set); deallocate(num_df_in_set)

  end subroutine get_side_set_param

  ! -----------------------------------------------------------------

  subroutine reorder_node_numbering(num_elem_blk, &
                                      num_nodes_per_elem, &
                                      num_elem_in_block, &
                                      elem_connectivity, &
                                      elem_type, &
                                      total_elem_node_list)
    integer, intent(in) :: num_elem_blk
    integer, dimension(:), intent(in) :: num_nodes_per_elem
    integer, dimension(:), intent(in) :: num_elem_in_block
    integer, dimension(:), intent(in) :: elem_connectivity
    integer, dimension(:), intent(in) :: elem_type
    integer, allocatable, dimension(:), intent(inout) :: total_elem_node_list

    integer, allocatable, dimension(:) :: elem_node_list

    integer :: i, e, n, z

    z = 0
    do i=1, num_elem_blk
       ! assemble element node list as we go:
       allocate(elem_node_list(num_nodes_per_elem(i)))
       do e=1, num_elem_in_block(i)
          do n=1, num_nodes_per_elem(i)
             elem_node_list(n) = elem_connectivity(n + z)
          end do
          call toFluidityElementNodeOrdering( elem_node_list, elem_type(i) )
          ! Now append elem_node_list to total_elem_node_list
          call append_array(total_elem_node_list, elem_node_list)
          z = z + num_nodes_per_elem(i)
       ! reset node list:
       elem_node_list = 0
       end do
       ! deallocate elem_node_list for next block
       deallocate(elem_node_list)
    end do

  end subroutine reorder_node_numbering

  ! -----------------------------------------------------------------

  subroutine assemble_allelements(num_elem_blk, &
                                    block_ids, &
                                    num_elem_in_block, &
                                    num_nodes_per_elem, &
                                    elem_order_map, &
                                    elem_type, &
                                    total_elem_node_list, &
                                    allelements)
    integer, intent(in) :: num_elem_blk
    integer, dimension(:), intent(in) :: block_ids
    integer, dimension(:), intent(in) :: num_elem_in_block
    integer, dimension(:), intent(in) :: num_nodes_per_elem
    integer, dimension(:), intent(in) :: elem_order_map
    integer, dimension(:), intent(in) :: elem_type
    integer, dimension(:), intent(in) :: total_elem_node_list
    type(EXOelement), pointer, dimension(:), intent(inout) :: allelements

    integer :: i, e, n, z, z2
    ! Subroutine to assemble a bucket full of element related data,
    ! e.g. element id, which block id it belongs to, its element type,
    ! its node ids.
    ! This is done for all elements of the mesh, e.g. also for surface
    ! elements of a 3D mesh. These surface elements won't be passed
    ! to the fluidity structure later on, but are added to allelements
    ! here.
    ! Also: Potential boundaryID numbers are added in the seperate
    ! subroutine 'allelements_add_boundary_ids'.

       ! Set elementIDs and blockIDs of to which the elements belong to
       allelements(:)%elementID = 0.0; allelements(:)%blockID = 0.0
       allelements(:)%type = 0.0; allelements(:)%numTags = 0.0
       z=0; z2=0;
       do i=1, num_elem_blk
          do e=1, num_elem_in_block(i)
             ! Set elementID:
             allelements(e+z)%elementID = elem_order_map(e+z)
             ! Set blockID of element e
             allelements(e+z)%blockID = block_ids(i)
             ! Set type of element:
             allelements(e+z)%type = elem_type(i)
             ! For nodeIDs:
             allocate( allelements(e+z)%nodeIDs(num_nodes_per_elem(i)) )
             do n=1, num_nodes_per_elem(i)
                ! copy the nodes of the element out of total_elem_node_list:
                allelements(e+z)%nodeIDs(n) = total_elem_node_list(n+z2)
             end do
             z2 = z2+num_nodes_per_elem(i)
          end do
          z = z + num_elem_in_block(i)
       end do

  end subroutine assemble_allelements

  ! -----------------------------------------------------------------

  subroutine allelements_add_boundary_ids(num_side_sets, &
                                            num_elem_in_set, &
                                            total_side_sets_elem_list, &
                                            side_set_ids, &
                                            allelements)
    integer, intent(in) :: num_side_sets
    integer, dimension(:), intent(in) :: num_elem_in_set
    integer, dimension(:), intent(in) :: total_side_sets_elem_list
    integer, dimension(:), intent(in) :: side_set_ids
    type(EXOelement), pointer, dimension(:), intent(inout) :: allelements

    integer :: elemID, num_tags_elem
    integer :: e, i, j, z

      z=1;
      do i=1, num_side_sets
         do e=1, num_elem_in_set(i)
            ! Get global element id:
            elemID = total_side_sets_elem_list(z)
            ! Set # of tags for this particular element
            allelements(elemID)%numTags = allelements(elemID)%numTags+1
            z = z+1
         end do
      end do
     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
     ! Setting tags to the elements with side-set-id !
     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      z=1;
      do i=1, num_side_sets
         do e=1, num_elem_in_set(i)
            ! Get global element id:
            elemID = total_side_sets_elem_list(z)
            num_tags_elem = allelements(elemID)%numTags
            ! Allocate array of tags for this particular element
            allocate(allelements(elemID)%tags(num_tags_elem))
            ! Initialize element%tag with a diabolic integer to indicate
            ! that the tag has not been 'correctly' set
            allelements(elemID)%tags(:) = -666
            z = z+1
         end do
      end do
      ! Now that the tags for all elements are allocated and uniquely marked, set them:
      z=1;
      do i=1, num_side_sets
         do e=1, num_elem_in_set(i)
            ! Get global element id:
            elemID = total_side_sets_elem_list(z)
            num_tags_elem = allelements(elemID)%numTags
            ! Set the side-set-id to this element
            do j=1, num_tags_elem
               ! Check for already existing tags in this element
               if ( allelements(elemID)%tags(j) == -666 ) then
                  allelements(elemID)%tags(j) = side_set_ids(i)
                  ! end exit the inner loop after setting this side sets id to the element
                  exit
               end if
            end do
            z = z+1
         end do
      end do

  end subroutine allelements_add_boundary_ids

  ! -----------------------------------------------------------------

  subroutine get_num_elem(num_dim, num_elem_blk, elem_type, num_elem_in_block, num_elem)
    integer, intent(in) :: num_dim
    integer, intent(in) :: num_elem_blk
    integer, dimension(:), intent(in) :: elem_type
    integer, dimension(:), intent(in) :: num_elem_in_block
    integer, intent(inout) :: num_elem

    integer :: i
    ! This subroutines computes the number of elements, which will be
    ! included in the fluidity mesh, thus num_elem is the number of
    ! elements of the mesh - the number of face-elements, e.g.
    ! in a 3D mesh the number of tetrahedras minus the number of
    ! triangles on surfaces.

    do i=1, num_elem_blk
       if (num_dim .eq. 2) then
          if (elem_type(i) .eq. 2 .or. elem_type(i) .eq. 3) then
             ! this is an element:
             num_elem = num_elem + num_elem_in_block(i)
          end if
       else if (num_dim .eq. 3) then
          if ( elem_type(i) .eq. 4 .or. elem_type(i) .eq. 5 ) then
             ! this is an element:
             num_elem = num_elem + num_elem_in_block(i)
          end if
       end if
    end do

  end subroutine get_num_elem

  ! -----------------------------------------------------------------

  subroutine get_num_faces(num_allelem, allelements, num_faces)
    integer, intent(in) :: num_allelem
    type(EXOelement), pointer, dimension(:), intent(in) :: allelements
    integer, intent(inout) :: num_faces

    integer :: i, elemID, num_tags_elem
    ! This subroutines computes the number of faces, which will be
    ! included in the fluidity mesh. These are the elements of the mesh
    ! which have a boundary ID/side set ID assigned to them.

       do i=1, num_allelem
          elemID = allelements(i)%elementID
          num_tags_elem = allelements(elemID)%numTags
          ! Is there at least one site set ID assigned to the element, it is a face:
          if (num_tags_elem > 0) then
             ! increase number of faces in the mesh...
             num_faces = num_faces + num_tags_elem
          end if
       end do

  end subroutine get_num_faces

  ! -----------------------------------------------------------------

  subroutine assemble_actual_elements(num_dim, num_elem_blk, num_elem_in_block, elem_type, allElements, exo_element)
    integer, intent(in) :: num_dim
    integer, intent(in) :: num_elem_blk
    integer, dimension(:), intent(in) :: num_elem_in_block
    integer, dimension(:), intent(in) :: elem_type
    type(EXOelement), pointer, dimension(:), intent(in) :: allelements
    type(EXOelement), pointer, dimension(:), intent(inout) :: exo_element

    integer :: b, e, i, exo_e
    ! This subroutine assembles a bucket (called exo_element) which corresponds
    ! to the actual elements of the mesh, meaning in a 3D mesh only to the volume
    ! elements, and in a 2D mesh only to the surface elements. For each elements
    ! that matches this description, the elementID, the block ID the element belongs
    ! to, its node IDs, and the element type are stored in this bucket.

    b=0; exo_e=1
    do i=1, num_elem_blk
       do e=1, num_elem_in_block(i)
          ! Distinguish between faces/edges and elements:
          if( .not. ( (num_dim .eq. 2 .and. elem_type(i) .eq. 1) .or. &
               (num_dim .eq. 3 .and. &
               (elem_type(i) .eq. 2 .or. elem_type(i) .eq. 3)) ) ) then
             ! these are elements (not edges/faces)
             allocate( exo_element(exo_e)%nodeIDs(size(allElements(e+b)%nodeIDs)))
             exo_element(exo_e)%elementID = allelements(e+b)%elementID
             exo_element(exo_e)%blockID = allelements(e+b)%blockID
             exo_element(exo_e)%nodeIDs = allelements(e+b)%nodeIDs
             exo_element(exo_e)%type = allelements(e+b)%type
             exo_e = exo_e + 1
          ! else
             ! These are edges/faces, thus do nothing
          end if
       ! next element e of block i
       end do
       b = b + num_elem_in_block(i) ! next block
    end do

  end subroutine assemble_actual_elements

  ! -----------------------------------------------------------------

  subroutine assemble_actual_face_elements(num_side_sets, &
                                             num_elem_in_set, &
                                             side_set_ids, &
                                             total_side_sets_node_cnt_list, &
                                             total_side_sets_node_list, &
                                             exo_face)
    integer, intent(in) :: num_side_sets
    integer, dimension(:), intent(in) :: num_elem_in_set
    integer, dimension(:), intent(in) :: side_set_ids
    integer, dimension(:), intent(in) :: total_side_sets_node_cnt_list
    integer, dimension(:), intent(in) :: total_side_sets_node_list
    type(EXOelement), pointer, dimension(:), intent(inout) :: exo_face

    integer :: num_nodes_face_ele
    integer :: i, e, n, m, exo_f

       n=1; exo_f=1;
       do i=1, num_side_sets
          do e=1, num_elem_in_set(i)
             num_nodes_face_ele = total_side_sets_node_cnt_list(e)
             allocate( exo_face(exo_f)%nodeIDs(num_nodes_face_ele))
             do m=1, num_nodes_face_ele
                exo_face(exo_f)%nodeIDs(m) = total_side_sets_node_list(n)
                n = n+1
             end do
             ! Set boundaryID to face:
             allocate(exo_face(exo_f)%tags(1))
             exo_face(exo_f)%tags = side_set_ids(i)
             exo_f = exo_f+1
          end do
       end do

  end subroutine assemble_actual_face_elements

  ! -----------------------------------------------------------------

  subroutine adding_nodes_to_field(eff_dim, num_nodes, node_map, node_coord, field)
    integer, intent(in) :: eff_dim
    integer, intent(in) :: num_nodes
    integer, allocatable, dimension(:), intent(in) :: node_map
    real(real_4), dimension(:,:), intent(in) :: node_coord
    type(vector_field), intent(inout) :: field

    type(EXOnode), pointer, dimension(:) :: exo_nodes
    integer :: d, n, nodeID
    ! This subroutine does what its name tells us: Adding the nodes of the mesh file
    ! to the actual field that describes the coordinate mesh in Fluidity

    ! Now set up nodes, their IDs and coordinates:
    ! Allocate exodus nodes
    allocate(exo_nodes(num_nodes))
    ! setting all node properties to zero
    exo_nodes(:)%nodeID = 0.0
    exo_nodes(:)%x(1)=0.0; exo_nodes(:)%x(2)=0.0; exo_nodes(:)%x(3)=0.0;
    ! copy coordinates into Coordinate field
    do n=1, num_nodes
       nodeID = node_map(n)
       exo_nodes(n)%nodeID = nodeID
       forall (d = 1:eff_dim)
          exo_nodes(n)%x(d) = node_coord(d,n)
          field%val(d,nodeID) = exo_nodes(n)%x(d)
       end forall
    end do
    ! Don't need those anymore:
    deallocate(exo_nodes);

  end subroutine adding_nodes_to_field

  ! -----------------------------------------------------------------

  subroutine adding_elements_to_field(num_dim, num_elem, exo_element, field)
    integer, intent(in) :: num_dim
    integer, intent(in) :: num_elem
    type(EXOelement), pointer, dimension(:), intent(in) :: exo_element
    type(vector_field), intent(inout) :: field

    integer :: elementType, num_nodes_per_elem_ele
    integer :: i, n, z, exo_e
    ! This subroutine now adds elements and regionIDs (which in an exodusII mesh
    ! are blockIDs) to the field

    z=0; exo_e=1;
    do i=1, num_elem
       elementType = exo_element(exo_e)%type
       if(.not.( (num_dim .eq. 2 .and. elementType .eq. 1) .or. &
         (num_dim .eq. 3 .and. &
         (elementType .eq. 2 .or. elementType .eq. 3)) ) ) then
         !these are normal elements:
          num_nodes_per_elem_ele = size(exo_element(exo_e)%nodeIDs)
          do n=1, num_nodes_per_elem_ele
             field%mesh%ndglno(n+z) = exo_element(exo_e)%nodeIDs(n)
          end do
          ! Set region_id of element (this will be its blockID in exodus)
          field%mesh%region_ids = exo_element(exo_e)%blockID
          exo_e = exo_e+1
          z = z+num_nodes_per_elem_ele
       end if
    end do

  end subroutine adding_elements_to_field

  ! -----------------------------------------------------------------
  ! Read ExodusII file to state object.
  function read_exodusii_file_to_state(filename, shape,shape_type,n_states) &
       result (result_state)
    ! Filename is the base name of the ExodusII file without file extension, e.g. .exo
    character(len=*), intent(in) :: filename
    type(element_type), intent(in), target :: shape
    logical , intent(in):: shape_type
    integer, intent(in), optional :: n_states
    type(state_type)  :: result_state

       FLAbort("read_exodusii_file_to_state() not implemented yet")

  end function read_exodusii_file_to_state

  ! -----------------------------------------------------------------

  subroutine append_array(array, array2)
     integer, allocatable, dimension(:), intent(inout) :: array
     integer, allocatable, dimension(:), intent(in) :: array2
     integer, allocatable, dimension(:) :: tmp
        allocate(tmp(size(array) + size(array2)))
        tmp(1:size(array)) = array
        tmp(size(array)+1:size(array)+size(array2)) = array2
        deallocate(array)
        allocate(array(size(tmp)))
        array = tmp
  end subroutine append_array

end module read_exodusii
