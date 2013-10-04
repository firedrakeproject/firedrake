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
!    License as published by the Free Software Foundation; either
!    version 2.1 of the License, or (at your option) any later version.
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

module write_gmsh

  use elements
  use fields
  use state_module
  use futils
  use parallel_tools
  use field_options
  use global_parameters, only : OPTION_PATH_LEN
  use gmsh_common

  implicit none

  private

  public :: write_gmsh_file

  interface write_gmsh_file
     module procedure write_mesh_to_gmsh, write_positions_to_gmsh
  end interface

  ! Writes to GMSH binary format - can set to ASCII (handy for debugging)
  logical, parameter  :: useBinaryGMSH=.true.

contains


  ! -----------------------------------------------------------------
  ! GMSH equivalents of write_triangle. Have been a bit
  ! naughty and assumed you want to write to binary
  ! GMSH format.
  ! -----------------------------------------------------------------


  subroutine write_mesh_to_gmsh( filename, state, mesh )

    character(len = *), intent(in) :: filename
    type(state_type), intent(in) :: state
    type(mesh_type), intent(in) :: mesh
    type(vector_field) :: positions

    positions = get_nodal_coordinate_field( state, mesh )

    call write_gmsh_file( filename, positions )

    ! Deallocate node and element memory structures
    call deallocate(positions)

    return

  end subroutine write_mesh_to_gmsh




  ! -----------------------------------------------------------------



  subroutine write_positions_to_gmsh(filename, positions)
    !!< Write out the mesh given by the position field in GMSH file:
    !!< In parallel, empty trailing processes are not written.
    character(len=*), intent(in):: filename
    type(vector_field), intent(in):: positions

    character(len=longStringLen) :: meshFile
    integer :: numParts, fileDesc

    ! How many processes contain data?
    numParts = get_active_nparts(ele_count(positions))

    ! Write out data only for those processes that contain data - SPMD requires
    ! that there be no early return
    if( getprocno() <= numParts ) then

      fileDesc=free_unit()

      meshFile = trim(filename) // ".msh"

      open( fileDesc, file=trim(meshFile), status="replace", access="stream", &
           action="write", err=101 )

    end if

    if( getprocno() <= numParts ) then
       ! Writing GMSH file header
       call write_gmsh_header( fileDesc, meshFile, useBinaryGMSH )
       call write_gmsh_nodes( fileDesc, meshFile, positions, useBinaryGMSH )
       call write_gmsh_faces_and_elements( fileDesc, meshFile, &
            positions%mesh, useBinaryGMSH )

       ! write columns data if present
       if (associated(positions%mesh%columns)) then
          call write_gmsh_node_columns( fileDesc, meshFile, positions, &
               useBinaryGMSH )
       end if

      ! Close GMSH file
      close( fileDesc )

    end if

    return

101 FLExit("Failed to open " // trim(meshFile) // " for writing")

  end subroutine write_positions_to_gmsh



  ! -----------------------------------------------------------------
  ! Write out GMSH header

  subroutine write_gmsh_header( fd, lfilename, useBinaryGMSH )
    integer :: fd
    character(len=*) :: lfilename
    logical :: useBinaryGMSH
    character(len=999) :: GMSHVersionStr, GMSHFileFormat, GMSHdoubleNumBytes

    integer, parameter :: oneInt = 1

    call ascii_formatting(fd, lfilename, "write")

    GMSHVersionStr="2.1"


    if(useBinaryGMSH) then
       ! GMSH binary format
       GMSHFileFormat="1"
    else
       GMSHFileFormat="0"
    end if

    write(GMSHdoubleNumBytes, *) doubleNumBytes
    write(fd, "(A)") "$MeshFormat"
    write(fd, "(A)") trim(GMSHVersionStr)//" "//trim(GMSHFileFormat)//" " &
         //trim(adjustl(GMSHdoubleNumBytes))

    if(useBinaryGMSH) then
       call binary_formatting(fd, lfilename, "write")

       ! The 32-bit integer "1", followed by a newline
       write(fd) oneInt, char(10)
       call ascii_formatting(fd, lfilename, "write")
    end if

    write(fd, "(A)") "$EndMeshFormat"

  end subroutine write_gmsh_header


  ! -----------------------------------------------------------------
  ! Write out GMSH nodes

  subroutine write_gmsh_nodes( fd, lfilename, field, useBinaryGMSH )
    ! Writes out nodes for the given position field
    integer fd
    character(len=*) :: lfilename
    character(len=longStringLen) :: charBuf
    character(len=longStringLen) :: idStr, xStr, yStr, zStr
    type(vector_field), intent(in):: field
    logical :: useBinaryGMSH
    integer numNodes, numDimen, numCoords, i, j
    real :: coords(3)

    numNodes = node_count(field)
    numDimen = mesh_dim(field)
    numCoords = field%dim

    ! Sanity check.
    if (numNodes==0) then
       FLAbort("write_gmsh_nodes(): no nodes to write out")
    end if

    ! header line: nodes, dim, no attributes, no boundary markers
    write(fd, "(A)", err=201) "$Nodes"
    write(fd, "(I0)", err=201) numNodes

    if( useBinaryGMSH) then
       ! Write out nodes in binary format
       call binary_formatting( fd, lfilename, "write" )
    end if

5959 format( I0, 999(X, F0.10) )

    do i=1, numNodes
       coords = 0
       coords(1:numCoords) = node_val(field, i)

       if(useBinaryGMSH) then
          write( fd ) i, coords
       else
          write(fd, 5959) i, coords
       end if
    end do

    if( useBinaryGMSH) then
       ! Write newline character
       write(fd) char(10)

       call ascii_formatting(fd, lfilename, "write")
    end if

    write( fd, "(A)" ) "$EndNodes"

    return

201 FLExit("Failed to write nodes to .msh file")

  end subroutine write_gmsh_nodes



  ! -----------------------------------------------------------------

  subroutine write_gmsh_faces_and_elements( fd, lfilename, mesh, &
       useBinaryGMSH )
    ! Writes out elements for the given mesh
    type(mesh_type), intent(in):: mesh
    logical :: useBinaryGMSH

    character(len=*) :: lfilename
    character(len=longStringLen) :: charBuf

    integer :: fd, numGMSHElems, numElements, numFaces
    integer :: numTags, nloc, sloc, faceType, elemType
    integer, pointer :: lnodelist(:)

    integer :: e, f, elemID
    character, parameter :: newLineChar=char(10)

    logical :: internal_boundaries

    ! Gather some info about the mesh
    numElements = ele_count(mesh)
    numFaces = surface_element_count(mesh)
    internal_boundaries = has_internal_boundaries(mesh)

    ! In the GMSH format, faces are also elements.
    numGMSHElems = numElements + numFaces

    ! Sanity check.
    if (numGMSHElems==0) then
       FLAbort("write_gmsh_faces_and_elements(): none of either!")
    end if


    ! Number of nodes for elements and faces
    nloc = ele_loc(mesh, 1)
    sloc = face_loc(mesh,1)

    ! Working out face and element types now
    faceType=0
    elemType=0

    select case(mesh_dim(mesh))
       ! Two dimensions
    case(2)
       if (nloc==3 .and. sloc==2) then
          faceType=1
          elemType=2
       else if(nloc==4 .and. sloc==2) then
          faceType=1
          elemType=4
       end if

       ! Three dimensions
    case(3)
       if(nloc==4 .and. sloc==3) then
          faceType=2
          elemType=4
       else if(nloc==8 .and. sloc==4) then
          faceType=3
          elemType=5
       end if
    end select

    ! If we've not managed to identify the element and faces, exit
    if(faceType==0 .and. elemType==0) then
       FLExit("Unknown combination of elements and faces.")
    end if

    ! Write out element label
    call ascii_formatting(fd, lfilename, "write")
    write(fd, "(A)") "$Elements"


    ! First, the number of GMSH elements (= elements+ faces)
    write(fd, "(I0)" ) numGMSHElems

    ! Faces written out first

    ! Number of tags associated with elements
    if(internal_boundaries) then
      ! write surface id and element owner
      numTags = 4
    else
      ! only surface id
      numTags = 2
    end if

    if(useBinaryGMSH) then
       call binary_formatting( fd, lfilename, "write" )
       write(fd) faceType, numFaces, numTags
    end if

    ! Correct format for ASCII mode element lines
6969 format (I0, 999(X,I0))

    do f=1, numFaces
       allocate( lnodelist(sloc) )

       lnodelist = face_global_nodes(mesh, f)
       call toGMSHElementNodeOrdering(lnodelist, faceType)

       ! Output face data
       select case(numTags)

       case (2)

          if(useBinaryGMSH) then
             write(fd, err=301) f, surface_element_id(mesh, f), 0, lnodelist
          else
             write(fd, 6969, err=301) f, faceType, numTags, surface_element_id(mesh, f), 0, lnodelist
          end if

       case (4)

          if(useBinaryGMSH) then
             write(fd, err=301) f, surface_element_id(mesh, f), 0, 0, face_ele(mesh, f), lnodelist
          else
             write(fd, 6969, err=301) f, faceType, numTags, surface_element_id(mesh,f), 0, 0, &
                  face_ele(mesh,f), lnodelist
          end if

       end select

       deallocate(lnodelist)
    end do

    ! Then regular GMSH elements (i.e. the real volume elements)

    if(useBinaryGMSH) then
       ! we always write 2 taqs - without region ids we just write an extra 0
       write(fd) elemType, numElements, 2
    end if

    do e=1, numElements
       elemID = e + numFaces
       allocate( lnodelist(nloc) )

       lnodelist = ele_nodes(mesh, e)
       call toGMSHElementNodeOrdering(lnodelist, elemType)

       ! Output element data
       if(associated(mesh%region_ids)) then

          if(useBinaryGMSH) then
             write(fd, err=301) elemID, ele_region_id(mesh, e), 0, lnodelist
          else
             write(fd, 6969, err=301) elemID, elemType, 2, &
                  ele_region_id(mesh, e), 0, lnodelist
          end if

       else

          if(useBinaryGMSH) then
             write(fd, err=301) elemID, 0, 0, lnodelist
          else
             write(fd, 6969, err=301) elemID, elemType, 2, &
                  0, 0, lnodelist
          end if

       end if

       deallocate(lnodelist)
    end do

    if(useBinaryGMSH) then
       write(fd, err=301) newLineChar
    end if

    ! Back to ASCII for end of elements section
    call ascii_formatting( fd, lfilename, "write" )
    write(fd, "(A)") "$EndElements"

    return

301 FLExit("Error while writing elements in .msh file.")

  end subroutine write_gmsh_faces_and_elements




  ! -----------------------------------------------------------------
  ! Write out node colum data

  subroutine write_gmsh_node_columns( fd, meshFile, field, useBinaryGMSH )
    integer :: fd
    character(len=*) :: meshFile
    type(vector_field), intent(in) :: field
    logical :: useBinaryGMSH

    character(len=longStringLen) :: charBuf
    integer :: numNodes, timeStepNum, numComponents, i
    real :: columnID

    numNodes = node_count(field)

    ! Not currently used
    timeStepNum = 0
    ! Number of field components for node (only 1 : column ID)
    numComponents = 1

    ! Sanity check.
    if (numNodes==0) then
       FLAbort("write_gmsh_node_columns(): no nodes to write out")
    end if

    call ascii_formatting(fd, meshFile, "write")

    write(fd, "(A)") "$NodeData"
    ! Telling GMSH we have one string tag ('node_column_ids')
    write(fd, "(I0)" ) 1
    write(fd, "(A)") "column_ids"

    ! No real number tag
    write(fd, "(I0)" ) 0

    ! And 3 integer tags (needed)
    write(fd, "(I0)") 3

    ! Which are:
    write(fd, "(I0)") timeStepNum
    write(fd, "(I0)") numComponents
    write(fd, "(I0)") numNodes

    ! Switch to binary format and write out node column IDs
    if(useBinaryGMSH) call binary_formatting(fd, meshFile, "write")
    do i=1, numNodes
       columnID = real(field%mesh%columns(i))
       if(useBinaryGMSH) then
          write(fd) i, columnID
       else
          write(fd, "(I0, X, F0.8)") i, columnID
       end if
    end do
    ! Newline

    if(useBinaryGMSH) then
       write(fd) char(10)
       call ascii_formatting(fd, meshFile, "write")
    end if

    ! Write out end tag and return
    write(fd, "(A)") "$EndNodeData"

  end subroutine write_gmsh_node_columns


end module write_gmsh
