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
#include "confdefs.h"


! ----------------------------------------------------------------------------
! This module acts as a wrapper for read/write routines for
! meshes of different formats.
!
! You should add code to support additional formats:
! - here (wrapper routines)
! - external modules (eg. Read_MeshFormat.F90, Write_MeshFormat.F90, etc.)
!
! This module call your own mesh read/writing routines - do not call them
! from the main Fluidity code.
! ----------------------------------------------------------------------------


module mesh_files
  use futils
  use elements
  use fields
  use state_module

  use global_parameters, only : OPTION_PATH_LEN

  use gmsh_common
  use read_gmsh
  use read_triangle
  use read_exodusii
  use write_gmsh
  use write_triangle

  use spud

  implicit none

  private

  interface read_mesh_files
     module procedure read_mesh_simple
  end interface

  interface write_mesh_files
     module procedure write_mesh_to_file, &
          write_positions_to_file
  end interface

  public :: read_mesh_files, write_mesh_files


contains


  ! --------------------------------------------------------------------------
  ! Read routines first
  ! --------------------------------------------------------------------------

  function read_mesh_simple(filename, format, quad_degree, &
       quad_ngi, quad_family, mdim, coord_dim) &
       result (field)

    ! A simpler mechanism for reading a mesh file into a field.
    ! In parallel the filename must *not* include the process number.

    character(len=*), intent(in) :: filename, format
    ! The degree of the quadrature.
    integer, intent(in), optional, target :: quad_degree
    ! The degree of the quadrature.
    integer, intent(in), optional, target :: quad_ngi
    ! What quadrature family to use
    integer, intent(in), optional :: quad_family
    ! Dimension of mesh
    integer, intent(in), optional :: mdim
    ! Force dimension of mesh in the case of manifolds
    integer, intent(in), optional :: coord_dim

    type(vector_field) :: field

    select case( trim(format) )
    case("triangle")
       field = read_triangle_files(filename, quad_degree=quad_degree, quad_ngi=quad_ngi, &
            quad_family=quad_family, mdim=mdim)

    case("gmsh")
       if (present(mdim)) then
         FLExit("Cannot specify dimension for gmsh format")
       end if
       field = read_gmsh_file(filename, quad_degree=quad_degree, quad_ngi=quad_ngi, &
            quad_family=quad_family, coord_dim=coord_dim)

    case("exodusii")
#ifdef HAVE_LIBEXOIIV2C
       field = read_exodusii_file(filename, quad_degree=quad_degree, quad_ngi=quad_ngi, &
            quad_family=quad_family)
#else
  FLExit("Fluidity was not configured with exodusII, reconfigure with '--with-exodusII'!")
#endif


       ! Additional mesh format subroutines go here

    case default
       FLExit("Reading mesh type "//format//" not supported within Fluidity")
    end select

  end function read_mesh_simple



  ! --------------------------------------------------------------------------
  ! Write routines here
  ! --------------------------------------------------------------------------


  subroutine write_mesh_to_file(filename, format, state, mesh)
    ! Write out the supplied mesh to the specified filename as mesh files.

    character(len = *), intent(in) :: filename
    character(len = *), intent(in) :: format
    type(state_type), intent(in) :: state
    type(mesh_type), intent(in) :: mesh

    select case(format)
    case("triangle")
       call write_triangle_files(filename, state, mesh)

    case("gmsh")
       call write_gmsh_file(filename, state, mesh )

    ! ExodusII write routines are not implemented at this point.
    ! Mesh is dumped as gmsh format for now.
    ! check subroutine 'insert_external_mesh' in Populate_State.F90,
    ! right after reading in external mesh files

       ! Additional mesh format subroutines go here

    case default
       FLExit("Writing to mesh type "//format//" not supported within Fluidity")
    end select


  end subroutine write_mesh_to_file

  ! --------------------------------------------------------------------------

  subroutine write_positions_to_file(filename, format, positions)
    !!< Write out the mesh given by the position field in mesh files
    !!< In parallel, empty trailing processes are not written.
    character(len=*), intent(in):: filename, format
    type(vector_field), intent(in):: positions

    select case( trim(format) )
    case("triangle")
       call write_triangle_files( trim(filename), positions)

    case("gmsh")
       call write_gmsh_file( trim(filename), positions)

    ! ExodusII write routines are not implemented at this point.
    ! Mesh is dumped as gmsh format for now.
    ! check subroutine 'insert_external_mesh' in Populate_State.F90,
    ! right after reading in external mesh files

       ! Additional mesh format subroutines go here

    case default
       FLExit("Writing to mesh type "//format//" not supported within Fluidity")
    end select

  end subroutine write_positions_to_file

end module mesh_files

