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

module read_gmsh
  ! This module reads GMSH files and results in a vector field of
  ! positions.

  use futils
  use elements
  use fields
  use state_module
  use spud
  use gmsh_common
  use global_parameters, only : OPTION_PATH_LEN

  implicit none

  private

  interface read_gmsh_file
     module procedure read_gmsh_simple
  end interface

  public :: read_gmsh_file

  integer, parameter:: GMSH_LINE=1, GMSH_TRIANGLE=2, GMSH_QUAD=3, GMSH_TET=4, GMSH_HEX=5, GMSH_NODE=15

contains

  ! -----------------------------------------------------------------
  ! The main function for reading GMSH files

  function read_gmsh_simple( filename, quad_degree, &
       quad_ngi, quad_family, coord_dim ) &
       result (field)
    !!< Read a GMSH file into a coordinate field.
    !!< In parallel the filename must *not* include the process number.

    character(len=*), intent(in) :: filename
    !! The degree of the quadrature.
    integer, intent(in), optional, target :: quad_degree
    !! The degree of the quadrature.
    integer, intent(in), optional, target :: quad_ngi
    !! What quadrature family to use
    integer, intent(in), optional :: quad_family
    !! Actual coordinate size, useful for manifolds
    integer, intent(in), optional :: coord_dim
    !! result: a coordinate field
    type(vector_field) :: field

    type(quadrature_type):: quad
    type(element_type):: shape
    type(mesh_type):: mesh

    integer :: fd
    integer,  pointer, dimension(:) :: sndglno, boundaryIDs, faceOwner

    character(len = parallel_filename_len(filename)) :: lfilename
    integer :: loc, sloc
    integer :: numNodes, numElements, numFaces
    logical :: haveBounds, haveElementOwners, haveRegionIDs
    integer :: dim, coordinate_dim
    integer :: gmshFormat
    integer :: n, d, e, f, nodeID

    type(GMSHnode), pointer :: nodes(:)
    type(GMSHelement), pointer :: elements(:), faces(:)

    ! If running in parallel, add the process number
    if(isparallel()) then
       lfilename = trim(parallel_filename(filename)) // ".msh"
    else
       lfilename = trim(filename) // ".msh"
    end if

    fd = free_unit()

    ! Open node file
    ewrite(2, *) "Opening "//trim(lfilename)//" for reading."
    open( unit=fd, file=trim(lfilename), err=43, action="read", &
         access="stream", form="formatted" )

    ! Read in header information, and validate
    call read_header( fd, lfilename, gmshFormat )

    ! Read in the nodes
    call read_nodes_coords( fd, lfilename, gmshFormat, nodes )

    ! Read in elements
    call read_faces_and_elements( fd, lfilename, gmshFormat, &
         elements, faces, dim)

    call read_node_column_IDs( fd, lfilename, gmshFormat, nodes )

    ! According to fluidity/bin/gmsh2triangle, Fluidity doesn't need
    ! anything past $EndElements, so we close the file.
    close( fd )


    numNodes = size(nodes)
    numFaces = size(faces)
    numElements = size(elements)

    ! NOTE:  similar function 'boundaries' variable in Read_Triangle.F90
    ! ie. flag for boundaries and internal boundaries (period mesh bounds)
    if (numFaces>0) then
      ! do we have physical surface ids?
      haveBounds= faces(1)%numTags>0
      ! do we have element owners of faces?
      haveElementOwners = faces(1)%numTags==4

      ! if any (the first face) has them, then all should have them
      do f=2, numFaces
         if(faces(f)%numTags/=faces(1)%numTags) then
           ewrite(0,*) "In your gmsh input files all faces (3d)/edges (2d) should" // &
              & "  have the same number of tags"
           FLExit("Inconsistent number of face tags")
         end if
      end do

    else

      haveBounds=.false.
      haveElementOwners=.false.

    end if

    if (numElements>0) then

      haveRegionIDs = elements(1)%numTags>0
      ! if any (the first face) has them, then all should have them
      do e=2, numElements
         if(elements(e)%numTags/=elements(1)%numTags) then
           ewrite(0,*) "In your gmsh input files all elements should" // &
              & "  have the same number of tags"
           FLExit("Inconsistent number of element tags")
         end if
      end do

    else

      haveRegionIDs = .false.

    end if

    if(have_option("/geometry/spherical_earth/") ) then
      ! on the sphere the input mesh may be 2d (extrusion), or 3d but
      ! Coordinate is always 3-dimensional
      coordinate_dim  = 3
    else
      coordinate_dim  = dim
    end if

    if (present(coord_dim)) then
      if (coord_dim > 0) then
        coordinate_dim = coord_dim
      end if
    end if

    loc = size( elements(1)%nodeIDs )
    if (numFaces>0) then
      sloc = size( faces(1)%nodeIDs )
    else
      sloc = 0
    end if

    ! Now construct within Fluidity data structures

    if (present(quad_degree)) then
       quad = make_quadrature(loc, dim, degree=quad_degree, family=quad_family)
    else if (present(quad_ngi)) then
       quad = make_quadrature(loc, dim, ngi=quad_ngi, family=quad_family)
    else
       FLAbort("Need to specify either quadrature degree or ngi")
    end if
    shape=make_element_shape(loc, dim, 1, quad)
    call allocate(mesh, numNodes, numElements, shape, name="CoordinateMesh")
    call allocate( field, coordinate_dim, mesh, name="Coordinate")

    ! deallocate our references of mesh, shape and quadrature:
    call deallocate(mesh)
    call deallocate(shape)
    call deallocate(quad)


    if (haveRegionIDs) then
      call allocate_region_ids(field%mesh, numElements)
    end if

    if(nodes(1)%columnID>=0)  allocate(field%mesh%columns(1:numNodes))


    ! Loop round nodes copying across coords and column IDs to field mesh,
    ! if they exist
    do n=1, numNodes

       nodeID = nodes(n)%nodeID
       forall (d = 1:field%dim)
          field%val(d,nodeID) = nodes(n)%x(d)
       end forall

       ! If there's a valid node column ID, use it.
       if ( nodes(n)%columnID .ne. -1 ) then
          field%mesh%columns(nodeID) = nodes(n)%columnID
       end if

    end do

    ! Copy elements to field
    do e=1, numElements
       field%mesh%ndglno((e-1)*loc+1:e*loc) = elements(e)%nodeIDs
       if (haveRegionIDs) field%mesh%region_ids(e) = elements(e)%tags(1)
    end do

    ! Now faces
    allocate(sndglno(1:numFaces*sloc))
    sndglno=0
    if(haveBounds) then
      allocate(boundaryIDs(1:numFaces))
    end if
    if(haveElementOwners) then
      allocate(faceOwner(1:numFaces))
    end if

    do f=1, numFaces
       sndglno((f-1)*sloc+1:f*sloc) = faces(f)%nodeIDs(1:sloc)
       if(haveBounds) boundaryIDs(f) = faces(f)%tags(1)
       if(haveElementOwners) faceOwner(f) = faces(f)%tags(4)
    end do

    ! If we've got boundaries, do something
    if( haveBounds ) then
       if ( haveElementOwners ) then
          call add_faces( field%mesh, &
               sndgln = sndglno(1:numFaces*sloc), &
               boundary_ids = boundaryIDs(1:numFaces), &
               element_owner=faceOwner )
       else
          call add_faces( field%mesh, &
               sndgln = sndglno(1:numFaces*sloc), &
               boundary_ids = boundaryIDs(1:numFaces) )
       end if
    else
       ewrite(2,*) "WARNING: no boundaries in GMSH file "//trim(lfilename)
       call add_faces( field%mesh, sndgln = sndglno(1:numFaces*sloc) )
    end if

    ! Deallocate arrays
    deallocate(sndglno)
    if (haveBounds) deallocate(boundaryIDs)
    if (haveElementOwners) deallocate(faceOwner)

    deallocate(nodes)
    deallocate(faces)
    deallocate(elements)

    return

43  FLExit("Unable to open "//trim(lfilename))

  end function read_gmsh_simple

  ! -----------------------------------------------------------------
  ! Read through the head to decide whether binary or ASCII, and decide
  ! whether this looks like a GMSH mesh file or not.

  subroutine read_header( fd, lfilename, gmshFormat )
    integer fd, gmshFormat

    character(len=*) :: lfilename
    character(len=longStringLen) :: charBuf
    character :: newlineChar
    integer gmshFileType, gmshDataSize, one
    real versionNumber


    ! Error checking ...

    read(fd, *) charBuf
    if( trim(charBuf) .ne. "$MeshFormat" ) then
       FLExit("Error: can't find '$MeshFormat' (GMSH mesh file?)")
    end if

    read(fd, *) charBuf, gmshFileType, gmshDataSize

    read(charBuf,*) versionNumber
    if( versionNumber .lt. 2.0 .or. versionNumber .ge. 3.0 ) then
       FLExit("Error: GMSH mesh version must be 2.x")
    end if


    if( gmshDataSize .ne. doubleNumBytes ) then
       write(charBuf,*) doubleNumBytes
       FLExit("Error: GMSH data size does not equal "//trim(adjustl(charBuf)))
    end if



    ! GMSH binary format continues the integer 1, in binary.
    if( gmshFileType .eq. binaryFormat ) then
       call binary_formatting(fd, lfilename, "read")
       read(fd) one, newlineChar
       call ascii_formatting(fd, lfilename, "read")
    end if


    read(fd, *) charBuf
    if( trim(charBuf) .ne. "$EndMeshFormat" ) then
       FLExit("Error: can't find '$EndMeshFormat' (is this a GMSH mesh file?)")
    end if

    ! Done with error checking... set format (ie. ascii or binary)
    gmshFormat = gmshFileType

  end subroutine read_header



  ! -----------------------------------------------------------------
  ! read in GMSH mesh nodes' coords into temporary arrays

  subroutine read_nodes_coords( fd, filename, gmshFormat, nodes )
    integer :: fd, gmshFormat

    character(len=*) :: filename
    character(len=longStringLen) :: charBuf
    character :: newlineChar
    integer :: i, numNodes
    type(GMSHnode), pointer :: nodes(:)


    read(fd, *) charBuf
    if( trim(charBuf) .ne. "$Nodes" ) then
       FLExit("Error: cannot find '$Nodes' in GMSH mesh file")
    end if


    read(fd, *) numNodes

    if(numNodes .lt. 2) then
       FLExit("Error: GMSH number of nodes field < 2")
    end if

    allocate( nodes(numNodes) )

    select case(gmshFormat)
    case(asciiFormat)
       call ascii_formatting(fd, filename, "read")
    case(binaryFormat)
       call binary_formatting(fd, filename, "read")
    end select

    ! read in node data
    do i=1, numNodes
       if( gmshFormat .eq. asciiFormat ) then
          read(fd, * ) nodes(i)%nodeID, nodes(i)%x
       else
          read(fd) nodes(i)%nodeID, nodes(i)%x
       end if
       ! Set column ID to -1: this will be changed later if $NodeData exists
       nodes(i)%columnID = -1
    end do

    ! Skip newline character when in binary mode
    if( gmshFormat .eq. binaryFormat ) read(fd), newlineChar


    call ascii_formatting(fd, filename, "read")

    ! Read in end node section
    read(fd, *) charBuf
    if( trim(charBuf) .ne. "$EndNodes" ) then
       FLExit("Error: can't find '$EndNodes' in GMSH file '"//trim(filename)//"'")
    end if

  end subroutine read_nodes_coords



  ! -----------------------------------------------------------------
  ! read in GMSH mesh nodes' column IDs (if exists)

  subroutine read_node_column_IDs( fd, filename, gmshFormat, nodes )
    integer :: fd, gmshFormat
    character(len=*) :: filename
    type(GMSHnode), pointer :: nodes(:)

    character(len=longStringLen) :: charBuf
    character :: newlineChar

    integer :: numStringTags, numRealTags, numIntTags
    integer :: timeStep, numComponents, numNodes
    integer :: i, nodeIx, fileState
    real :: rval

    call ascii_formatting( fd, filename, "read" )

    ! If there's no $NodeData section, don't try to read in column IDs: return
    read(fd, iostat=fileState, fmt=*) charBuf
    if( trim(charBuf) .ne. "$NodeData" .or. fileState .lt. 0 ) then
       return
    end if

    ! Sanity checking
    read(fd, *) numStringTags
    if(numStringTags .ne. 1) then
       FLExit("Error: must have one string tag in GMSH file $NodeData part")
    end if
    read(fd, *) charBuf
    if( trim(charBuf) .ne. "column_ids") then
       FLExit("Error: GMSH string tag in $NodeData section != 'column_ids'")
    end if

    ! Skip over these, not used (yet)
    read(fd, *) numRealTags
    do i=1, numRealTags
       read(fd, *) rval
    end do

    read(fd,*) numIntTags
    ! This must equal 3
    if(numIntTags .ne. 3) then
       FLExit("Error: must be 3 GMSH integer tags in GMSH $NodeData section")
    end if

    read(fd, *) timeStep
    read(fd, *) numComponents
    read(fd, *) numNodes

    ! More sanity checking
    if(numNodes .ne. size(nodes) ) then
       FLExit("Error: number of nodes for column IDs doesn't match node array")
    end if

    ! Switch to binary if necessary
    if(gmshFormat == binaryFormat) then
       call binary_formatting(fd, filename, "read")
    end if


    ! Now read in the node column IDs
    do i=1, numNodes
       select case(gmshFormat)
       case(asciiFormat)
          read(fd, *) nodeIx, rval
       case(binaryFormat)
          read(fd ) nodeIx, rval
       end select
       nodes(i)%columnID = floor(rval)
    end do

    ! Skip newline character when in binary mode
    if( gmshFormat == binaryFormat ) read(fd), newlineChar

    call ascii_formatting(fd, filename, "read")

    ! Read in end node section
    read(fd, *) charBuf
    if( trim(charBuf) .ne. "$EndNodeData" ) then
       FLExit("Error: cannot find '$EndNodeData' in GMSH mesh file")
    end if

  end subroutine read_node_column_IDs



  ! -----------------------------------------------------------------
  ! Read in element header data and establish topological dimension

  subroutine read_faces_and_elements( fd, filename, gmshFormat, &
       elements, faces, dim)

    integer, intent(in) :: fd, gmshFormat
    character(len=*), intent(in) :: filename
    type(GMSHelement), pointer :: elements(:), faces(:)
    integer, intent(out) :: dim

    type(GMSHelement), pointer :: allElements(:)

    integer :: numAllElements
    character(len=longStringLen) :: charBuf
    character :: newlineChar
    integer :: numEdges, numTriangles, numQuads, numTets, numHexes
    integer :: numFaces, faceType, numElements, elementType
    integer :: e, i, numLocNodes, tmp1, tmp2, tmp3
    integer :: groupType, groupElems, groupTags


    read(fd,*) charBuf
    if( trim(charBuf)/="$Elements" ) then
       FLExit("Error: cannot find '$Elements' in GMSH mesh file")
    end if

    read(fd,*) numAllElements

    ! Sanity check.
    if(numAllElements<1) then
       FLExit("Error: number of elements in GMSH file < 1")
    end if

    allocate( allElements(numAllElements) )


    ! Read in GMSH elements, corresponding tags and nodes

    select case(gmshFormat)

       ! ASCII is straightforward
    case (asciiFormat)

       do e=1, numAllElements
          ! Read in whole line into a string buffer
          read(fd, "(a)", end=888) charBuf
          ! Now read from string buffer for main element info
888       read(charBuf, *) allElements(e)%elementID, allElements(e)%type, &
               allElements(e)%numTags

          numLocNodes = elementNumNodes(allElements(e)%type)
          allocate( allElements(e)%nodeIDs(numLocNodes) )
          allocate( allElements(e)%tags( allElements(e)%numTags) )

          ! Now read in tags and node IDs
          read(charBuf, *) tmp1, tmp2, tmp3, &
               allElements(e)%tags, allElements(e)%nodeIDs

       end do

    case (binaryFormat)
       ! Make sure raw stream format is on
       call binary_formatting( fd, filename, "read" )

       e=1

       ! GMSH groups elements by type:
       ! the code below reads in one type of element in a block, followed
       ! by other types until all the elements have been read in.
       do while( e .le. numAllelements )
          read(fd) groupType, groupElems, groupTags

          if( (e-1)+groupElems .gt. numAllElements ) then
             FLExit("GMSH element group contains more than the total")
          end if

          ! Read in elements in a particular type block
          do i=e, (e-1)+groupElems
             numLocNodes = elementNumNodes(groupType)
             allocate( allElements(i)%nodeIDs(numLocNodes) )
             allocate( allElements(i)%tags( groupTags ) )

             allElements(i)%type = groupType
             allElements(i)%numTags = groupTags

             read(fd) allElements(i)%elementID, allElements(i)%tags, &
                  allElements(i)%nodeIDs
          end do

          e = e+groupElems
       end do

    end select

    ! Skip final newline
    if(gmshFormat==binaryFormat) read(fd) newlineChar


    ! Run through final list of elements, reorder nodes etc.
    numEdges = 0
    numTriangles = 0
    numTets = 0
    numQuads = 0
    numHexes = 0


    ! Now we've got all our elements in memory, do some housekeeping.
    do e=1, numAllElements

       call toFluidityElementNodeOrdering( allElements(e)%nodeIDs, &
            allElements(e)%type )

       select case ( allElements(e)%type )
       case (GMSH_LINE)
          numEdges = numEdges+1
       case (GMSH_TRIANGLE)
          numTriangles = numTriangles+1
       case (GMSH_QUAD)
          numQuads = numQuads+1
       case (GMSH_TET)
          numTets = numTets+1
       case (GMSH_HEX)
          numHexes = numHexes+1
       case (GMSH_NODE)
          ! Do nothing
       case default
          ewrite(0,*) "element id,type: ", allElements(e)%elementID, allElements(e)%type
          FLExit("Unsupported element type in gmsh .msh file")
       end select

    end do

    ! Check for $EndElements tag
    call ascii_formatting( fd, filename, "read" )
    read(fd,*) charBuf
    if( trim(charBuf) .ne. "$EndElements" ) then
       FLExit("Error: cannot find '$EndElements' in GMSH mesh file")
    end if

    ! This decides which element types are faces, and which are
    ! regular elements, as per gmsh2triangle logic. Implicit in that logic
    ! is that faces can only be of one element type, and so the following
    ! meshes are verboten:
    !   tet/hex, tet/quad, triangle/hex and triangle/quad

    if (numTets>0) then
       numElements = numTets
       elementType = GMSH_TET
       numFaces = numTriangles
       faceType = GMSH_TRIANGLE
       dim = 3
       if (numQuads>0 .or. numHexes>0) then
         FLExit("Cannot combine hexes or quads with tetrahedrals in one gmsh .msh file")
       end if

    elseif (numTriangles>0) then
       numElements = numTriangles
       elementType = GMSH_TRIANGLE
       numFaces = numEdges
       faceType = GMSH_LINE
       dim = 2
       if (numQuads>0 .or. numHexes>0) then
         FLExit("Cannot combine hexes or quads with triangles in one gmsh .msh file")
       end if

    elseif (numHexes .gt. 0) then
       numElements = numHexes
       elementType = GMSH_HEX
       numFaces = numQuads
       faceType = GMSH_QUAD
       dim = 3

    elseif (numQuads .gt. 0) then
       numElements = numQuads
       elementType = GMSH_QUAD
       numFaces = numEdges
       faceType = GMSH_LINE
       dim = 2

    else
       FLExit("Unsupported mixture of face/element types")
    end if

    call copy_to_faces_and_elements( allElements, &
         elements, numElements, elementType, &
         faces, numFaces, faceType )


    ! We no longer need this
    call deallocateElementList( allElements )



  end subroutine read_faces_and_elements



  ! -----------------------------------------------------------------
  ! This copies elements from allElements(:) to elements(:) and faces(:),
  ! depending upon the element type definition of faces.

  subroutine copy_to_faces_and_elements( allElements, &
       elements, numElements, elementType, &
       faces, numFaces, faceType )

    type(GMSHelement), pointer :: allElements(:), elements(:), faces(:)
    integer :: numElements, elementType, numFaces, faceType

    integer :: allelementType
    integer :: e, fIndex, eIndex, numTags, numNodeIDs

    allocate( elements(numElements) )
    allocate( faces(numFaces) )

    fIndex=1
    eIndex=1

    ! Copy element data across. Only array pointers are copied, which
    ! is why we don't deallocate nodeIDs(:), etc.
    do e=1, size(allElements)
       allelementType = allElements(e)%type

       numTags = allElements(e)%numTags
       numNodeIDs = size(allElements(e)%nodeIDs)

       if(allelementType .eq. faceType) then

          faces(fIndex) = allElements(e)

          allocate( faces(fIndex)%tags(numTags) )
          allocate( faces(fIndex)%nodeIDs(numNodeIDs) )
          faces(fIndex)%tags = allElements(e)%tags
          faces(fIndex)%nodeIDs = allElements(e)%nodeIDs

          fIndex = fIndex+1
       else if (allelementType .eq. elementType) then

          elements(eIndex) = allElements(e)

          allocate( elements(eIndex)%tags(numTags) )
          allocate( elements(eIndex)%nodeIDs(numNodeIDs) )
          elements(eIndex)%tags = allElements(e)%tags
          elements(eIndex)%nodeIDs = allElements(e)%nodeIDs

          eIndex = eIndex+1
       end if
    end do

  end subroutine copy_to_faces_and_elements


end module read_gmsh
