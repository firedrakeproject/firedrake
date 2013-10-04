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

module field_options
   !!< This module contains code that uses scalar/vector/tensor_fields
   !!< in combination with floptions. Anything schema specific.

   use quadrature
   use elements
   use spud
   use FLDebug
   use Fields_Data_Types
   use fields_base
   use fields_allocates
   use fields_manipulation
   use fields_calculations
   use Global_Parameters, only: OPTION_PATH_LEN, adaptivity_mesh_name
   use state_module
   use futils
   use metric_tools
   use data_structures
   use state_module

   implicit none

   interface adaptivity_options
      module procedure adaptivity_options_scalar, adaptivity_options_vector
   end interface

   interface interpolate_field
      module procedure interpolate_field_scalar, interpolate_field_vector, &
                     & interpolate_field_tensor, interpolate_options
   end interface

   interface needs_initial_mesh
      module procedure needs_initial_mesh_scalar, needs_initial_mesh_vector, &
                     & needs_initial_mesh_tensor, interpolate_options
   end interface

   interface constant_field
      module procedure constant_field_scalar, constant_field_vector, &
                     & constant_field_tensor
   end interface

   interface isotropic_field
      module procedure isotropic_field_tensor
   end interface

   interface diagonal_field
      module procedure diagonal_field_tensor
   end interface

   interface extract_pressure_mesh
      module procedure extract_pressure_mesh_from_state, extract_pressure_mesh_from_any_state
   end interface extract_pressure_mesh

   interface extract_velocity_mesh
      module procedure extract_velocity_mesh_from_state, extract_velocity_mesh_from_any_state
   end interface extract_velocity_mesh

   private

   public :: complete_mesh_path, complete_field_path, &
     & get_external_mesh, adaptivity_options, print_children, &
     & get_coordinate_field, select_fields_to_interpolate, &
     & find_mesh_to_adapt, &
     & adaptivity_bounds, find_linear_parent_mesh, &
     & interpolate_field, convergence_norm_integer, &
     & do_not_recalculate, needs_initial_mesh, &
     & get_external_coordinate_field, collect_fields_by_mesh, &
     & equation_type_index, field_options_check_options, &
     & constant_field, isotropic_field, diagonal_field, &
     & extract_pressure_mesh, extract_velocity_mesh, &
     & postprocess_periodic_mesh, get_diagnostic_coordinate_field, &
     & get_nodal_coordinate_field, extract_prognostic_pressure, &
     & extract_prognostic_velocity

  integer, parameter, public :: FIELD_EQUATION_UNKNOWN                   = 0, &
                                FIELD_EQUATION_ADVECTIONDIFFUSION        = 1, &
                                FIELD_EQUATION_CONSERVATIONOFMASS        = 2, &
                                FIELD_EQUATION_REDUCEDCONSERVATIONOFMASS = 3, &
                                FIELD_EQUATION_INTERNALENERGY            = 4, &
                                FIELD_EQUATION_HEATTRANSFER              = 5, &
                                FIELD_EQUATION_ELECTRICALPOTENTIAL       = 6, &
                                FIELD_EQUATION_KEPSILON       = 7

contains

  recursive subroutine print_children(path)

    use spud
    use global_parameters, only: OPTION_PATH_LEN

    implicit none

    character(len=*), intent(in)::path
    character(len=OPTION_PATH_LEN)::name, child_name
    integer :: i, nchildren

    name = " "
    child_name = " "

    call get_number_of_children(path, nchildren)
    print *, trim(path), ": number of children: ", nchildren
    if(nchildren==0) return
    do i=0, nchildren-1
       call get_child_name(path, i, name)
       print*, trim(path), " contains ", trim(name)
       child_name=path//"/"//trim(name)
       call print_children(trim(child_name))
    end do

  end subroutine print_children

  function complete_mesh_path(path, stat)
    !!< Auxillary function to add from_file/from_mesh to meshion path.

    character(len = *), intent(in) :: path
    integer, optional, intent(out) :: stat

    character(len = OPTION_PATH_LEN) :: complete_mesh_path

    if(present(stat)) then
      stat = 0
    end if

    if(have_option(trim(path) // "/from_mesh")) then
      complete_mesh_path = trim(path) // "/from_mesh"
    else if(have_option(trim(path) // "/from_file")) then
      complete_mesh_path = trim(path) // "/from_file"
    else if(present(stat)) then
      stat = 1
    else
      ewrite(-1, *) "For mesh option path: " // trim(path)
      FLAbort("Unknown mesh type or wrong mesh option path")
    end if

  end function complete_mesh_path

   character(len=OPTION_PATH_LEN) function complete_field_path(path, name, stat)
   !!< Auxillary function to add  prognostic/diagnostic/prescribed
   !!< to field option path. This version flaborts as opposed to the one
   !!< in populate_state_module.

   character(len=*), intent(in) :: path
   character(len=*), intent(in), optional :: name
   integer, intent(out), optional :: stat

    if (present(stat)) then
      stat = 0
    end if

    if (have_option(trim(path) // "/prognostic")) then

       complete_field_path=trim(path) // "/prognostic"

    else if (have_option(trim(path) // "/diagnostic")) then

       complete_field_path=trim(path) // "/diagnostic"

    else if (have_option(trim(path) // "/prescribed")) then

       complete_field_path=trim(path) // "/prescribed"

    else

      if (present(stat)) then
        stat = 1
        complete_field_path=trim(path)
      else
        ewrite(0,*) "Error completing field path given the option path:", trim(path)
        if (present(name)) then
          ewrite(0,*) "for field name:", trim(name)
        else
          ewrite(0,*) "Note field name not available.  Modify the call to include this for a more informative error message."
        end if
        FLAbort("Error: unknown field type or wrong field option path.")
      end if
    end if

  end function complete_field_path

  function make_coordinate_field(state, target_mesh) result (to_position)
  !!< Creates a coordinate field interpolated on the target_mesh
  !!< If we're on the sphere and the super parametric option is selected
  !!< the coordinate field is "bended" onto the sphere. This routine
  !!< always creates a field with a new reference (even if we simply return
  !!< a coordinate field that is already in state), so it has to be deallocated
  !!< after its use.
    type(vector_field):: to_position
    type(state_type), intent(inout):: state
    type(mesh_type), intent(in):: target_mesh


    integer, parameter:: kloc(1:6)=(/ 2, 4, 5, 7, 8, 9 /)
    integer, parameter:: iloc(1:6)=(/ 1, 1, 3, 1, 3, 6 /)
    integer, parameter:: jloc(1:6)=(/ 3, 6, 6, 10, 10, 10 /)

    type(vector_field), pointer:: position
    type(element_type), pointer:: target_shape
    real radi, radj
    integer ele, k

    position => extract_vector_field(state, "Coordinate")
    if (position%mesh==target_mesh) then
       to_position=position
       ! make this a new reference
       call incref(to_position)
    else
       ! if position is not on the same mesh as the target_mesh
       call allocate(to_position, position%dim, target_mesh, &
          name=trim(target_mesh%name)//'Coordinate')
       call remap_field(position, to_position)
    end if

    target_shape => ele_shape(target_mesh, 1)
    if (target_shape%degree==2 .and. have_option( &
       '/geometry/spherical_earth/quadratic_superparametric_mapping')) then


       do ele=1, element_count(to_position)
          do k=1, 6
             radi=sqrt(sum(ele_val(to_position, iloc(k))**2))
             radj=sqrt(sum(ele_val(to_position, jloc(k))**2))
          end do
       end do

    end if

  end function make_coordinate_field

  subroutine adaptivity_bounds(state, minch, maxch, name)
    type(state_type), intent(inout) :: state
    real, intent(in) :: minch, maxch
    type(tensor_field) :: min_eigen, max_eigen
    type(mesh_type), pointer :: mesh
    character(len=*), intent(in), optional :: name
    integer :: dim

    if (present(name)) then
      mesh => extract_mesh(state, trim(name))
    else
      mesh => extract_mesh(state, "Mesh")
    end if
    dim = mesh_dim(mesh)
    call allocate(min_eigen, mesh, "MinMetricEigenbound", field_type=FIELD_TYPE_CONSTANT)
    call allocate(max_eigen, mesh, "MaxMetricEigenbound", field_type=FIELD_TYPE_CONSTANT)

    call set(min_eigen, get_matrix_identity(dim) * eigenvalue_from_edge_length(maxch))
    call set(max_eigen, get_matrix_identity(dim) * eigenvalue_from_edge_length(minch))

    call insert(state, min_eigen, "MinMetricEigenbound")
    call insert(state, max_eigen, "MaxMetricEigenbound")

    call deallocate(min_eigen)
    call deallocate(max_eigen)
  end subroutine adaptivity_bounds

  subroutine adaptivity_options_scalar(state, field, weight, relative, min_psi)
    type(state_type), intent(inout) :: state
    type(scalar_field), intent(inout) :: field
    real, intent(in) :: weight
    logical, intent(in) :: relative
    type(scalar_field) :: adaptivity_weight
    real, intent(in), optional :: min_psi
    integer :: stat

    field%option_path = "/material_phase[0]/" // field%name
    if (relative) then
      call add_option(trim(field%option_path) // "/virtual/adaptivity_options/relative_measure", stat=stat)
      call set_option(trim(field%option_path) // "/virtual/adaptivity_option&
           &s/relative_measure/tolerance", min_psi, stat=stat)
    else
      call add_option(trim(field%option_path) // "/virtual/adaptivity_options/absolute_measure", stat=stat)
    end if

    call allocate(adaptivity_weight, field%mesh, trim(field%name) // "InterpolationErrorBound", FIELD_TYPE_CONSTANT)
    call set(adaptivity_weight, weight)
    call insert(state, adaptivity_weight, trim(field%name) // "InterpolationErrorBound")
    call deallocate(adaptivity_weight)

  end subroutine adaptivity_options_scalar

  subroutine adaptivity_options_vector(state, field, weight, relative, min_psi)
    type(state_type), intent(inout) :: state
    type(vector_field), intent(inout) :: field
    real, intent(in), dimension(:) :: weight
    logical, intent(in) :: relative
    type(vector_field) :: adaptivity_weight
    real, intent(in), optional, dimension(:) :: min_psi
    integer :: stat

    field%option_path = "/material_phase[0]/" // field%name
    if (relative) then
      call add_option(trim(field%option_path) // "/virtual/adaptivity_options/relative_measure", stat=stat)
      call set_option(trim(field%option_path) // "/virtual/adaptivity_options/relative_measure/tolerance", min_psi, stat=stat)
    else
      call add_option(trim(field%option_path) // "/virtual/adaptivity_options/absolute_measure", stat=stat)
    end if

    call allocate(adaptivity_weight, field%dim, field%mesh, trim(field%name) // "InterpolationErrorBound", FIELD_TYPE_CONSTANT)
    call set(adaptivity_weight, weight)
    call insert(state, adaptivity_weight, trim(field%name) // "InterpolationErrorBound")
    call deallocate(adaptivity_weight)

  end subroutine adaptivity_options_vector

  function get_external_mesh(states, external_mesh_name) result(mesh)
    type(state_type), dimension(:), intent(in) :: states
    type(mesh_type), pointer:: mesh
    integer :: nmeshes, i
    character(len=OPTION_PATH_LEN) :: mesh_path
    character(len=FIELD_NAME_LEN) linear_mesh_name
    character(len=FIELD_NAME_LEN), intent(in), optional :: external_mesh_name

    if (present(external_mesh_name)) then
      mesh => extract_mesh(states(1), trim(external_mesh_name))
    else
      nmeshes=option_count("/geometry/mesh")
      do i = 0, nmeshes-1
         mesh_path="/geometry/mesh["//int2str(i)//"]"
         if(have_option(trim(mesh_path)//"/from_file")) exit
      end do
      if (i>nmeshes-1) then
         FLExit("Options tree does not have external mesh")
      end if
      call get_option(trim(mesh_path)//"/name", linear_mesh_name)
      ! We'll assume this one is the linear mesh
      ewrite(2,*) "Assuming linear mesh is: " // trim(linear_mesh_name)
      mesh => extract_mesh(states(1), linear_mesh_name)
    end if

  end function get_external_mesh

  function get_external_coordinate_field(state, mesh) result (positions)
  !!< returns a coordinate field called trim(mesh%name)//"Coordinate"
  !!< pulled from state if present, otherwise it returns the "Coordinate" field
  type(state_type), intent(in):: state
  type(mesh_type), intent(in):: mesh
  type(vector_field), pointer :: positions

    if (has_vector_field(state, trim(mesh%name)//"Coordinate")) then
       positions=>extract_vector_field(state, trim(mesh%name)//"Coordinate")
    elseif (has_vector_field(state, "IteratedCoordinate")) then
       ! if the mesh is moving it's necessary to evaluate diagnostics on the most
       ! up to date coordinate.
       ! if the mesh is not moving this is just aliased to Coordinate anyway.
       positions=>extract_vector_field(state, "IteratedCoordinate")
    else
       positions=>extract_vector_field(state, "Coordinate")
    end if

  end function get_external_coordinate_field

  function get_diagnostic_coordinate_field(state, mesh) result (positions)
  !!< Returns a coordinate field suitable for finite element integration.
  !!< In most cases this will be the "Coordinate" field. In case of
  !!< external meshes with the exclude_from_mesh_adaptivity_option, or in
  !!< the case of the horizontal mesh in an extruded mesh setup, it will
  !!< return the positions field stored for that mesh in state.
  type(state_type), intent(in):: state
  type(mesh_type), intent(in):: mesh
  type(vector_field) positions

    type(vector_field), pointer:: coordinate_field
    type(mesh_type), pointer:: mesh_i
    character(len=FIELD_NAME_LEN):: mesh_name
    integer:: i

    if (have_option(trim(mesh%option_path)//"/exclude_from_mesh_adaptivity")) then
      positions=extract_vector_field(state, trim(mesh%name)//"Coordinate")
      call incref(positions)
    else
      coordinate_field => extract_vector_field(state, "Coordinate")
      if (mesh_dim(mesh)+1==mesh_dim(coordinate_field)) then
         ! coordinate is extruded, mesh is a horizontal mesh
         if (mesh_periodic(mesh)) then
            ! search for the unperiodic mesh derived from it
            do i=1, mesh_count(state)
              mesh_i => extract_mesh(state, i)
              if (have_option(trim(mesh_i%option_path)//"/from_mesh/mesh[0]/name") &
                .and. have_option(trim(mesh_i%option_path)// &
                "/from_mesh/periodic_boundary_conditions/remove_periodicity")) then
                 call get_option(trim(mesh_i%option_path)//"/name", mesh_name)
                 positions=extract_vector_field(state, trim(mesh%name)//"Coordinate")
                 call incref(positions)
                 return
              end if
            end do
            ! no mesh derived from it
            ewrite(0,*) "Can't find a suitable unperiodic coordinate field for periodic mesh ", &
              trim(mesh%name)
            ewrite(0,*) "This probably means the operation you want to do on this field "//&
               &"is not supported for a horizontal periodic mesh"
            FLExit("No suitable coordinate field found.")
         else
            positions=extract_vector_field(state, trim(mesh%name)//"Coordinate")
            call incref(positions)
         end if
      else if (element_count(coordinate_field)==element_count(mesh) .and. &
         mesh_dim(coordinate_field)==mesh_dim(mesh)) then
         positions=coordinate_field
         call incref(positions)
      else
         ewrite(0,*) "Can't find a suitable coordinate field for mesh ", &
             trim(mesh%name)
         ewrite(0,*) "This probably means the operation you want to do is not supported for this mesh"
         FLExit("No suitable coordinate field found.")
      end if
    end if

  end function get_diagnostic_coordinate_field

  function get_coordinate_field(state, mesh) result (positions)
  !!< Returns a coordinate field for the given mesh, that has the same
  !!< shape (and thus number of nodes) in each element. If the mesh is
  !!< discontinuous or periodic however, the positions%mesh is not
  !!< necessarily the same as 'mesh'. The returned positions field is
  !!< never periodic and thus provides valid coordinates for each element
  !!< individually.
  !!< As the polynomial degree of the returned coordinates is the same as
  !!< that of the input mesh, it won't return a superparametric
  !!< "Coordinate" field. This routine should therefore only be used if
  !!< there is an algorithmic reason to have the positions on the same
  !!< nodes as 'mesh' in each element.
  !!< NOTE: The returned vector_field should always be deallocated
  type(state_type), intent(in):: state
  type(mesh_type), intent(in):: mesh
  type(vector_field) positions

    type(vector_field), pointer:: coordinate_field
    type(mesh_type) :: unperiodic_mesh
    logical:: can_remap_from_coordinate_field

    if (have_option(trim(mesh%option_path)//"/exclude_from_mesh_adaptivity").and.&
        has_vector_field(state, trim(mesh%name)//"Coordinate")) then
      positions=extract_vector_field(state, trim(mesh%name)//"Coordinate")
      call incref(positions)
    else
      coordinate_field => extract_vector_field(state, "Coordinate")
      can_remap_from_coordinate_field = &
         element_count(coordinate_field)==element_count(mesh) .and. &
         mesh_dim(coordinate_field)==mesh_dim(mesh)
      if (mesh_dim(mesh)+1==mesh_dim(coordinate_field)) then
         ! coordinate is extruded, mesh is a horizontal mesh
         if (mesh_periodic(mesh)) call give_up
         positions=extract_vector_field(state, trim(mesh%name)//"Coordinate")
         call incref(positions)
      else if (mesh_periodic(mesh)) then
         ! never return periodic coordinates
         if((coordinate_field%mesh%shape==mesh%shape).and.&
            (coordinate_field%mesh%elements==mesh%elements).and.&
            (coordinate_field%mesh%continuity==mesh%continuity)) then
            ! meshes are sufficiently similar that you shouldn't
            ! need to remake the mesh (hopefully)
            positions=coordinate_field
            call incref(positions)
         else if (can_remap_from_coordinate_field) then
            ! meshes are different in an important way so
            ! remake the mesh using the shape and continuity
            ! of the desired mesh (but not periodic)
            unperiodic_mesh=make_mesh(coordinate_field%mesh, mesh%shape, mesh%continuity, &
                                      name="UnPeriodicCoordinateMesh")
            call allocate(positions, coordinate_field%dim, unperiodic_mesh, &
                          name="Coordinate")
            call remap_field(coordinate_field, positions)
            call deallocate(unperiodic_mesh)
         else
            call give_up()
         end if
      elseif (has_vector_field(state, trim(mesh%name)//"Coordinate")) then
         positions=extract_vector_field(state, trim(mesh%name)//"Coordinate")
         call incref(positions)
      elseif (coordinate_field%mesh==mesh) then
         positions=coordinate_field
         call incref(positions)
      else if (can_remap_from_coordinate_field) then
         call allocate(positions, coordinate_field%dim, mesh, name="Coordinate")
         call remap_field(coordinate_field, positions)
      else
         call give_up()
      end if
    end if

    contains

    subroutine give_up()

      ewrite(0,*) "Can't find a suitable coordinate field for mesh ", &
          trim(mesh%name)
      ewrite(0,*) "This probably means the operation you want to do is not "//&
           &"supported for this mesh"
      FLExit("No suitable coordinate field found.")

    end subroutine give_up

  end function get_coordinate_field

  function get_nodal_coordinate_field(state, mesh) result (positions)
  !!< Returns a coordinate field on the mesh provided. For periodic
  !!< meshes the nodes on the periodic boundary will get the original
  !!< position of the aliased nodes. The returned field is not suitable
  !!< for finite element integration.
  !!< NOTE: The returned vector_field should always be deallocated
  type(state_type), intent(in):: state
  type(mesh_type), intent(in):: mesh
  type(vector_field) positions

    type(vector_field), pointer:: coordinate_field
    integer:: stat

    if (has_vector_field(state, trim(mesh%name)//"Coordinate")) then
      positions=extract_vector_field(state, trim(mesh%name)//"Coordinate")
      call incref(positions)
    else
      coordinate_field => extract_vector_field(state, "Coordinate")
      if (coordinate_field%mesh==mesh) then
         positions=coordinate_field
         call incref(positions)
      else if (element_count(coordinate_field)==element_count(mesh) .and. &
         mesh_dim(coordinate_field)==mesh_dim(mesh)) then
         ! can remap from coordinate field
         call allocate(positions, coordinate_field%dim, mesh, name="Coordinate")

         if (mesh_periodic(mesh) .and. continuity(mesh)>=0) then
            call remap_field(coordinate_field, positions, stat=stat)
            if (stat/=REMAP_ERR_UNPERIODIC_PERIODIC) then
              FLAbort("Error remapping coordinate field")
            end if
            ! make sure the remapped coordinates adhere to the convention
            ! of storing the aliased from position in the periodic boundary nodes
            call postprocess_periodic_mesh(coordinate_field%mesh, coordinate_field, &
              mesh, positions)
         else
            call remap_field(coordinate_field, positions)
         end if
      else
         ewrite(0,*) "Can't find a suitable coordinate field for mesh ", &
             trim(mesh%name)
         ewrite(0,*) "This probably means the operation you want to do is not&
              &supported for this mesh"
         FLExit("No suitable coordinate field found.")
      end if
    end if

  end function get_nodal_coordinate_field

  function extract_pressure_mesh_from_state(state, stat)
    type(state_type), intent(in):: state
    type(mesh_type), pointer:: extract_pressure_mesh_from_state
    integer, optional, intent(out):: stat

    type(scalar_field), pointer:: p

    p => extract_scalar_field(state, "Pressure", stat=stat)
    if (associated(p)) then
      extract_pressure_mesh_from_state => p%mesh
    else
      nullify(extract_pressure_mesh_from_state)
    end if

  end function extract_pressure_mesh_from_state

  function extract_pressure_mesh_from_any_state(states, stat)
    type(state_type), dimension(:), intent(in):: states
    type(mesh_type), pointer:: extract_pressure_mesh_from_any_state
    integer, optional, intent(out):: stat

    type(scalar_field), pointer:: p

    p => extract_scalar_field(states, "Pressure", stat=stat)
    if (associated(p)) then
      extract_pressure_mesh_from_any_state => p%mesh
    else
      nullify(extract_pressure_mesh_from_any_state)
    end if

  end function extract_pressure_mesh_from_any_state

  function extract_velocity_mesh_from_state(state, stat)
    type(state_type), intent(in):: state
    type(mesh_type), pointer:: extract_velocity_mesh_from_state
    integer, optional, intent(out):: stat

    type(vector_field), pointer:: u

    u => extract_vector_field(state, "Velocity", stat=stat)
    if (associated(u)) then
      extract_velocity_mesh_from_state => u%mesh
    else
      nullify(extract_velocity_mesh_from_state)
    end if

  end function extract_velocity_mesh_from_state

  function extract_velocity_mesh_from_any_state(states, stat)
    type(state_type), dimension(:), intent(in):: states
    type(mesh_type), pointer:: extract_velocity_mesh_from_any_state
    integer, optional, intent(out):: stat

    type(vector_field), pointer:: u

    u => extract_vector_field(states, "Velocity", stat=stat)
    if (associated(u)) then
      extract_velocity_mesh_from_any_state => u%mesh
    else
      nullify(extract_velocity_mesh_from_any_state)
    end if

  end function extract_velocity_mesh_from_any_state

  subroutine select_fields_to_interpolate(state, interpolate_state, no_positions, &
    first_time_step)
    !!< This routine returns a state that is a selection of the fields
    !!< of the input state that should be interpolated after adaptivity
    !!< from the old to the new mesh.
    !!< Diagnostic are never interpolated, although this could
    !!< be useful for diagnostic purposes - again an option may be implemented.
    !!< Aliased fields are obviously excluded.

    type(state_type), intent(in):: state
    type(state_type), intent(out):: interpolate_state
    !! leave out the "Coordinate" field
    logical, optional, intent(in) :: no_positions
    !! exclude prognostic fields that can be re-initialised (not from_file)
    logical, optional, intent(in) :: first_time_step

    integer :: i
    type(mesh_type), pointer :: mesh
    type(scalar_field), pointer:: sfield => null()
    type(vector_field), pointer:: vfield => null()
    type(tensor_field), pointer:: tfield => null()
    integer :: stat

    interpolate_state%name = state%name

    do i=1, mesh_count(state)
      mesh => extract_mesh(state, i)
      call insert(interpolate_state, mesh, trim(mesh%name))

      if (.not. present_and_true(no_positions)) then
          vfield => extract_vector_field(state, trim(mesh%name) // 'Coordinate', stat=stat)
          if (stat == 0) then
            call insert(interpolate_state, vfield, trim(mesh%name) // 'Coordinate')
          end if
      end if
    end do

    ! Select all prognostic and prescribed scalar fields that do not have interpolation
    ! disabled. In addition, select diagnostic scalar fields that have interpolation enabled.
    do i = 1, scalar_field_count(state)
      sfield => extract_scalar_field(state, i)
      if (interpolate_field(sfield, first_time_step=first_time_step)) then
        ewrite(2,*) 'selecting to interpolate ', trim(sfield%name)
        call insert(interpolate_state, sfield, sfield%name)
      end if
    end do

    ! Select all prognostic and prescribed vector fields that do not have interpolation
    ! disabled. In addition, select diagnostic vector fields that have interpolation enabled.
    do i = 1, vector_field_count(state)
      vfield => extract_vector_field(state, i)
      if (interpolate_field(vfield, first_time_step=first_time_step)) then
        ewrite(2,*) 'selecting to interpolate ', trim(vfield%name)
        call insert(interpolate_state, vfield, vfield%name)
      end if
    end do

    ! Select all prognostic and prescribed tensor fields that do not have interpolation
    ! disabled. In addition, select diagnostic tensor fields that have interpolation enabled.
    do i = 1, tensor_field_count(state)
      tfield => extract_tensor_field(state, i)
      if (interpolate_field(tfield, first_time_step=first_time_step)) then
        ewrite(2,*) 'selecting to interpolate ', trim(tfield%name)
        call insert(interpolate_state, tfield, tfield%name)
      end if
    end do

    if (.not. present_and_true(no_positions)) then
      ! Need coordinate for interpolation
      vfield => extract_vector_field(state, "Coordinate")
      call insert(interpolate_state, vfield, vfield%name)
    end if

  end subroutine select_fields_to_interpolate

  function interpolate_field_scalar(field, first_time_step) result (interpolate_field)
    !!< Does this field want to be interpolated?
    logical :: interpolate_field
    type(scalar_field), intent(in) :: field
    !! at first_time_step skip those prognostice fields that can be reinitialised
    logical, optional, intent(in):: first_time_step

    interpolate_field = (.not.aliased(field)) .and. &
          interpolate_options(trim(field%option_path), first_time_step=first_time_step)

  end function interpolate_field_scalar

  function interpolate_field_vector(field, first_time_step) result (interpolate_field)
    !!< Does this field want to be interpolated?
    logical :: interpolate_field
    type(vector_field), intent(in) :: field
    !! at first_time_step skip those prognostice fields that can be reinitialised
    logical, optional, intent(in):: first_time_step

    interpolate_field = (.not.aliased(field)) .and. &
          interpolate_options(trim(field%option_path), first_time_step=first_time_step)

  end function interpolate_field_vector

  function interpolate_field_tensor(field, first_time_step) result (interpolate_field)
    !!< Does this field want to be interpolated?
    logical :: interpolate_field
    type(tensor_field), intent(in) :: field
    !! at first_time_step skip those prognostice fields that can be reinitialised
    logical, optional, intent(in):: first_time_step

    interpolate_field = (.not.aliased(field)) .and. &
          interpolate_options(trim(field%option_path), first_time_step=first_time_step)

  end function interpolate_field_tensor

  logical function interpolate_options(option_path, first_time_step)
    !!< Does this option path make the associated field want to be interpolated?
    character(len=*) :: option_path
    !! at first_time_step skip those prognostice fields that can be reinitialised
    logical, optional, intent(in):: first_time_step

    if (present(first_time_step)) then
      if (first_time_step) then
        ! if the field is prognostic/prescribed and needs its initial mesh
        ! it can't be reinitialised, so we need to interpolate
        interpolate_options = needs_initial_mesh_options(option_path) .and. &
           (have_option(trim(option_path) // "/prognostic") .or. &
            have_option(trim(option_path) // "/prescribed"))
        return
      end if
    end if

    interpolate_options = (have_option(trim(option_path) // "/prognostic") .and. &
          & .not. have_option(trim(option_path) // "/prognostic/no_interpolation")) &
          .or. have_option(trim(option_path)//"/prescribed/galerkin_projection") &
          .or. have_option(trim(option_path)//"/prescribed/consistent_interpolation") &
          .or. have_option(trim(option_path)//"/prescribed/pseudo_consistent_interpolation") &
          .or. have_option(trim(option_path)//"/prescribed/grandy_interpolation") &
          .or. have_option(trim(option_path)//"/diagnostic/galerkin_projection") &
          .or. have_option(trim(option_path)//"/diagnostic/consistent_interpolation") &
          .or. have_option(trim(option_path)//"/diagnostic/pseudo_consistent_interpolation") &
          .or. have_option(trim(option_path)//"/diagnostic/grandy_interpolation")

  end function interpolate_options

  function needs_initial_mesh_scalar(field) result (needs_initial_mesh)
    !!< Does the field need the initial mesh for (re)initialisation or prescribing
    !!< this is the case if any of its initial conditions are from_file
    logical :: needs_initial_mesh
    type(scalar_field), intent(in) :: field

    needs_initial_mesh = needs_initial_mesh_options(field%option_path)

  end function needs_initial_mesh_scalar

  function needs_initial_mesh_vector(field) result (needs_initial_mesh)
    !!< Does the field need the initial mesh for (re)initialisation or prescribing
    !!< this is the case if any of its initial conditions are from_file
    logical :: needs_initial_mesh
    type(vector_field), intent(in) :: field

    needs_initial_mesh = needs_initial_mesh_options(field%option_path)

  end function needs_initial_mesh_vector

  function needs_initial_mesh_tensor(field) result (needs_initial_mesh)
    !!< Does the field need the initial mesh for (re)initialisation or prescribing
    !!< this is the case if any of its initial conditions are from_file
    logical :: needs_initial_mesh
    type(tensor_field), intent(in) :: field

    needs_initial_mesh = needs_initial_mesh_options(field%option_path)

  end function needs_initial_mesh_tensor

  logical function needs_initial_mesh_options(option_path)
    !!< Does the field need the initial mesh for initialisation or prescribing
    !!< this is the case if any of its initial conditions are from_file
    character(len=*) :: option_path

    character(len=OPTION_PATH_LEN):: initialisation_path
    integer:: i

    if (have_option(trim(option_path)//'/prognostic')) then
      initialisation_path=trim(option_path)//'/prognostic/initial_condition'
    else if (have_option(trim(option_path)//'/prescribed')) then
      initialisation_path=trim(option_path)//'/prescribed/value'
    else
      ! diagnostic fields are not initialised/prescribed anyway
      needs_initial_mesh_options=.false.
      return
    end if

    ! return .true. if any of the regions are initialised from file
    do i=0, option_count(initialisation_path)-1
      if (have_option(trim(initialisation_path)// &
         '['//int2str(i)//']/from_file')) then
         needs_initial_mesh_options=.true.
         return
      end if
    end do

    needs_initial_mesh_options=.false.

  end function needs_initial_mesh_options

  logical function do_not_recalculate(option_path)
    !!< Does this option path tell us not to represcribe
    character(len=*) :: option_path

    integer :: stat

    do_not_recalculate = have_option(trim(complete_field_path(option_path, stat=stat))//"/do_not_recalculate")

  end function do_not_recalculate

  subroutine find_linear_parent_mesh(state, mesh, parent_mesh, stat)
  !!< Tries to find the parent mesh (possibly grand...parent mesh)
  !!< that is linear and continuous by trawling through
  !!< the options tree. It will have the same periodicity.
  type(state_type), intent(in):: state
  type(mesh_type), target, intent(in):: mesh
  type(mesh_type), pointer:: parent_mesh
  integer, optional, intent(out) :: stat

    character(len=FIELD_NAME_LEN) parent_name
    integer lstat

    if(present(stat)) stat = 0

    ! start with mesh itself:
    parent_mesh => mesh

    do
      ! this is what we're looking for:
      if (parent_mesh%shape%degree==1 .and. parent_mesh%continuity>=0) then
        return
      end if
      call get_option(trim(parent_mesh%option_path)// &
         '/from_mesh/mesh[0]/name', parent_name, stat=lstat)
      if (lstat/=0) then
        if (present(stat)) then
          stat = 1
          return
        else
          ! this fails if parent is not derived, i.e. external, and not
          ! meeting our criteria (currently not possible as higher order
          ! meshes are always derived directly from a P1 mesh - can't
          ! extrude or periodise higher order meshes)
          ewrite(-1,*) "Trying to find linear and continuous parent mesh of ", &
             trim(mesh%name)
          ewrite(-1,*) "External mesh ", trim(parent_mesh%name), &
              &" does not meet these criteria"
          FLExit("Linear input mesh required")
        end if
      end if
      parent_mesh => extract_mesh(state, parent_name)
    end do

  end subroutine find_linear_parent_mesh

  subroutine find_mesh_to_adapt(state, mesh)
  !!< Finds the external mesh used as basis for adaptivity
  !!< This has to be a continuous linear mesh. It has to be the
  !!< external mesh as all other meshes will be derived from it.
  type(state_type), intent(in):: state
  type(mesh_type), pointer:: mesh

    mesh => extract_mesh(state, adaptivity_mesh_name)

    if (mesh%shape%degree/=1 .or. mesh%continuity<0) then
      FLExit("For adaptivity external mesh needs to be linear and continuous.")
    end if

  end subroutine find_mesh_to_adapt

  function convergence_norm_integer(option_path) result(norm)
  !!< Return the integer index for the norm for the convergence index
  !!< defaults to the infinity norm if no option found
  integer :: norm
  character(len=*), intent(in) :: option_path

  if(have_option(trim(option_path)//"/l2_norm")) then
    norm = CONVERGENCE_L2_NORM
  else if(have_option(trim(option_path)//"/cv_l2_norm")) then
    norm = CONVERGENCE_CV_L2_NORM
  else
    norm = CONVERGENCE_INFINITY_NORM
  end if

  end function convergence_norm_integer

  integer function equation_type_index(option_path)

    character(len=*) :: option_path

    character(len=FIELD_NAME_LEN) :: equation_type

    ! find out equation type
    call get_option(trim(option_path)//'/prognostic/equation[0]/name', &
                    equation_type, default="Unknown")

    select case(trim(equation_type))
    case ("AdvectionDiffusion")
      equation_type_index = FIELD_EQUATION_ADVECTIONDIFFUSION
    case ("HeatTransfer")
      equation_type_index = FIELD_EQUATION_HEATTRANSFER
    case ("ConservationOfMass")
      equation_type_index = FIELD_EQUATION_CONSERVATIONOFMASS
    case ("ReducedConservationOfMass")
      equation_type_index = FIELD_EQUATION_REDUCEDCONSERVATIONOFMASS
    case ( "InternalEnergy" )
      equation_type_index = FIELD_EQUATION_INTERNALENERGY
    case ( "ElectricalPotential" )
      equation_type_index = FIELD_EQUATION_ELECTRICALPOTENTIAL
    case ( "KEpsilon" )
      equation_type_index = FIELD_EQUATION_KEPSILON
    case default
      equation_type_index = FIELD_EQUATION_UNKNOWN
    end select

  end function equation_type_index

  subroutine collect_fields_by_mesh(states, mesh_names, mesh_states)
    !! For each mesh_names(i) returns a mesh_states(i) that contains the corresponding mesh
    !! and all fields in all states defined on that mesh. To distinguish between different fields
    !! with the same name from different states, each field coming from states(j) gets inserted
    !! as "State<j>FieldName", the actual field name is not changed.
    type(state_type), dimension(:), intent(in) :: states
    character(len=*), dimension(:), intent(in) :: mesh_names
    type(state_type), dimension(:), intent(out) :: mesh_states

    type(integer_hash_table) :: refcount_to_meshno
    integer :: i, j, mesh_no
    type(mesh_type), pointer :: mesh
    type(scalar_field), pointer :: sfield
    type(vector_field), pointer :: vfield
    type(tensor_field), pointer :: tfield

    assert( size(mesh_names)==size(mesh_states) )

    call allocate(refcount_to_meshno)
    do i=1, size(mesh_names)
      mesh => extract_mesh(states(1), mesh_names(i))
      call insert(mesh_states(i), mesh, mesh_names(i))
      call insert(refcount_to_meshno, mesh%refcount%id, i)
    end do

    do i=1,size(states)
      do j=1,scalar_field_count(states(i))
        sfield => extract_scalar_field(states(i), j)
        if(has_key(refcount_to_meshno, sfield%mesh%refcount%id)) then
          mesh_no = fetch(refcount_to_meshno, sfield%mesh%refcount%id)
          call insert(mesh_states(mesh_no), sfield, "State" // int2str(i) // trim(sfield%name))
        end if
      end do
      sfield => null()

      do j=1,vector_field_count(states(i))
        vfield => extract_vector_field(states(i), j)
        if(has_key(refcount_to_meshno, vfield%mesh%refcount%id)) then
          mesh_no = fetch(refcount_to_meshno, vfield%mesh%refcount%id)
          call insert(mesh_states(mesh_no), vfield, "State" // int2str(i) // trim(vfield%name))
        end if
      end do
      vfield => null()

      do j=1,tensor_field_count(states(i))
        tfield => extract_tensor_field(states(i), j)
        if(has_key(refcount_to_meshno, tfield%mesh%refcount%id)) then
          mesh_no = fetch(refcount_to_meshno, tfield%mesh%refcount%id)
          call insert(mesh_states(mesh_no), tfield, "State" // int2str(i) // trim(tfield%name))
        end if
      end do
      tfield => null()
    end do

    call deallocate(refcount_to_meshno)

  end subroutine collect_fields_by_mesh

  function constant_field_scalar(field) result (constant)
    type(scalar_field), intent(in) :: field

    logical :: constant

    integer :: value_count, constant_count

    value_count = option_count(trim(field%option_path)//"/prescribed/value")
    constant_count = option_count(trim(field%option_path)//"/prescribed/value/constant")

    constant = (value_count==constant_count).and.(constant_count>0)

  end function constant_field_scalar

  function constant_field_vector(field) result (constant)
    type(vector_field), intent(in) :: field

    logical :: constant

    integer :: value_count, constant_count

    value_count = option_count(trim(field%option_path)//"/prescribed/value")
    constant_count = option_count(trim(field%option_path)//"/prescribed/value/constant")

    constant = (value_count==constant_count).and.(constant_count>0)

  end function constant_field_vector

  function constant_field_tensor(field) result (constant)
    type(tensor_field), intent(in) :: field

    logical :: constant

    integer :: value_count, constant_count

    value_count = option_count(trim(field%option_path)//"/prescribed/value")
    constant_count = option_count(trim(field%option_path)//"/prescribed/value/isotropic/constant")+&
                     option_count(trim(field%option_path)//"/prescribed/value/diagonal/constant")+&
                     option_count(trim(field%option_path)//"/prescribed/value/anistropic_symmetric/constant")+&
                     option_count(trim(field%option_path)//"/prescribed/value/anistropic_asymmetric/constant")

    constant = (value_count==constant_count).and.(constant_count>0)

  end function constant_field_tensor

  function isotropic_field_tensor(field) result (isotropic)
    type(tensor_field), intent(in) :: field

    logical :: isotropic

    integer :: value_count, isotropic_count

    isotropic_count = option_count(trim(field%option_path)//"/prescribed/value/isotropic")
    value_count = option_count(trim(field%option_path)//"/prescribed/value")

    isotropic = (value_count==isotropic_count).and.(value_count>0)

  end function isotropic_field_tensor

  function diagonal_field_tensor(field) result (diagonal)
    type(tensor_field), intent(in) :: field

    logical :: diagonal

    integer :: value_count, diagonal_count

    diagonal_count = option_count(trim(field%option_path)//"/prescribed/value/diagonal")
    value_count = option_count(trim(field%option_path)//"/prescribed/value")

    diagonal = (value_count==diagonal_count).and.(value_count>0)

  end function diagonal_field_tensor

  subroutine postprocess_periodic_mesh(external_mesh, external_positions, periodic_mesh, periodic_positions)
    type(mesh_type), intent(in) :: periodic_mesh
    type(mesh_type), intent(in) :: external_mesh
    type(vector_field), intent(inout) :: periodic_positions
    type(vector_field), intent(in) :: external_positions

    type(integer_set):: physical_bc_nodes
    integer :: periodic_bc, aliased_id, physical_id, j
    integer, dimension(2) :: shape_option
    integer, dimension(:), allocatable :: aliased_boundary_ids, physical_boundary_ids
    integer :: face
    integer :: ele
    real, dimension(ele_ngi(periodic_positions, 1)) :: detwei
    real :: vol

    ! Check for degenerate elements
    do ele=1,ele_count(periodic_positions)
      call transform_to_physical(periodic_positions, ele, detwei=detwei)
      vol = sum(detwei)
      assert(vol > 0)
    end do

    ! First we need to create a set of all nodes that are on physical periodic
    ! boundaries. If these nodes (in the non-periodic mesh) also appear on
    ! aliased boundaries (happens for double periodic), they should not be
    ! used to determine the position in the periodic coordinate field

    call allocate(physical_bc_nodes)
    do periodic_bc=0,option_count(trim(periodic_mesh%option_path) // '/from_mesh/periodic_boundary_conditions')-1
      shape_option = option_shape(trim(periodic_mesh%option_path) // '/from_mesh/periodic_boundary_conditions['//int2str(periodic_bc)//']/physical_boundary_ids')
      allocate(physical_boundary_ids(shape_option(1)))
      call get_option(trim(periodic_mesh%option_path) // '/from_mesh/periodic_boundary_conditions[' // int2str(periodic_bc) // ']/physical_boundary_ids', physical_boundary_ids)
      do j=1, shape_option(1)
        physical_id = physical_boundary_ids(j)
        ! Now we can finally loop over faces with this id and set appropriately
        do face=1, surface_element_count(external_mesh)
          if (surface_element_id(external_mesh, face) == physical_id) then
            call insert(physical_bc_nodes, face_global_nodes(external_mesh, face))
          end if
        end do
      end do
      deallocate(physical_boundary_ids)
    end do

    ! We need to loop through all aliased faces and set the periodic positions to
    ! the aliased values, so that we can recover them by applying the map later

    do periodic_bc=0,option_count(trim(periodic_mesh%option_path) // '/from_mesh/periodic_boundary_conditions')-1
      shape_option = option_shape(trim(periodic_mesh%option_path) // '/from_mesh/periodic_boundary_conditions['//int2str(periodic_bc)//']/aliased_boundary_ids')
      allocate(aliased_boundary_ids(shape_option(1)))
      call get_option(trim(periodic_mesh%option_path) // '/from_mesh/periodic_boundary_conditions[' // int2str(periodic_bc) // ']/aliased_boundary_ids', aliased_boundary_ids)
      do j=1,shape_option(1)
        aliased_id = aliased_boundary_ids(j)
        ! Now we can finally loop over faces with this id and set appropriately
        do face=1,surface_element_count(external_mesh)
          if (surface_element_id(external_mesh, face) == aliased_id) then
            call postprocess_periodic_mesh_face(periodic_positions, external_positions, face, physical_bc_nodes)
          end if
        end do
      end do
      deallocate(aliased_boundary_ids)

    end do

    call deallocate(physical_bc_nodes)

    ! Check for degenerate elements
    do ele=1,ele_count(periodic_positions)
      call transform_to_physical(periodic_positions, ele, detwei=detwei)
      vol = sum(detwei)
      assert(vol > 0)
    end do

  end subroutine postprocess_periodic_mesh

  subroutine postprocess_periodic_mesh_face(periodic_positions, external_positions, face, physical_bc_nodes)
    ! Write the positions of the nodes on this aliased face
    ! to the periodic_positions. The physical positions can later be
    ! recovered by applying the coordinate map. Nodes that are on both
    ! a physical and an aliased face (happens for double/triple periodic)
    ! should not be written, as they have another copy, going back
    ! applying the inverse of the coordinate map associated with the
    ! physical face - and its that position we want to write out.
    type(vector_field), intent(inout):: periodic_positions
    type(vector_field), intent(in):: external_positions
    integer, intent(in):: face
    type(integer_set), intent(in):: physical_bc_nodes

    integer, dimension(face_loc(external_positions, face)):: face_non_periodic_nodes, face_periodic_nodes
    integer:: k, np_node, p_node

    face_non_periodic_nodes=face_global_nodes(external_positions, face)
    face_periodic_nodes=face_global_nodes(periodic_positions, face)

    do k=1, size(face_non_periodic_nodes)
      np_node=face_non_periodic_nodes(k)
      p_node=face_periodic_nodes(k)
      if (.not. has_value(physical_bc_nodes, np_node)) then
        call set(periodic_positions, p_node, node_val(external_positions, np_node))
      end if
    end do

  end subroutine postprocess_periodic_mesh_face

  subroutine field_options_check_options
    integer :: nmat, nfield, m, f
    character(len=OPTION_PATH_LEN) :: mat_name, field_name
    integer :: equation_type
    logical :: cv_disc, cg_disc

    nmat = option_count("/material_phase")

    do m = 0, nmat-1
      call get_option("/material_phase["//int2str(m)//"]/name", mat_name)
      nfield = option_count("/material_phase["//int2str(m)//"]/scalar_field")
      do f = 0, nfield-1
        call get_option("/material_phase["//int2str(m)//"]/scalar_field["//int2str(f)//&
                        "]/name", field_name)

        cv_disc=have_option("/material_phase["//int2str(m)//"]/scalar_field["//int2str(f)//&
                        "]/prognostic/spatial_discretisation/control_volumes")
        cg_disc=have_option("/material_phase["//int2str(m)//"]/scalar_field["//int2str(f)//&
                        "]/prognostic/spatial_discretisation/continuous_galerkin")

        equation_type=equation_type_index(trim("/material_phase["//int2str(m)//"]/scalar_field["//int2str(f)//"]"))
        select case(equation_type)
        case(FIELD_EQUATION_CONSERVATIONOFMASS)
          if(.not.cv_disc) then
            ewrite(-1,*) "Options checking field "//&
                          trim(field_name)//" in material_phase "//&
                          trim(mat_name)//"."
            FLExit("Selected equation type only compatible with control volume spatial_discretisation")
          end if
        case(FIELD_EQUATION_REDUCEDCONSERVATIONOFMASS)
          if(.not.cv_disc) then
            ewrite(-1,*) "Options checking field "//&
                          trim(field_name)//" in material_phase "//&
                          trim(mat_name)//"."
            FLExit("Selected equation type only compatible with control volume spatial_discretisation")
          end if
        case(FIELD_EQUATION_INTERNALENERGY)
          if(.not.(cv_disc.or.cg_disc)) then
            ewrite(-1,*) "Options checking field "//&
                          trim(field_name)//" in material_phase "//&
                          trim(mat_name)//"."
            FLExit("Selected equation type only compatible with control volume or continuous galerkin spatial_discretisation")
          end if
        case(FIELD_EQUATION_HEATTRANSFER)
          if(.not.cv_disc) then
            ewrite(-1,*) "Options checking field "//&
                          trim(field_name)//" in material_phase "//&
                          trim(mat_name)//"."
            FLExit("Selected equation type only compatible with control volume spatial_discretisation")
          end if
        case(FIELD_EQUATION_KEPSILON)
          if(.not.cg_disc) then
            ewrite(-1,*) "Options checking field "//&
                          trim(field_name)//" in material_phase "//&
                          trim(mat_name)//"."
            FLExit("Selected equation type only compatible with continuous galerkin spatial_discretisation")
          end if
        end select
      end do
    end do

  end subroutine field_options_check_options

  function extract_prognostic_pressure(state, stat) result(sfield)

  type(state_type), dimension(:), intent(in) :: state
  type(scalar_field), pointer :: sfield
  integer, optional :: stat

  integer :: i, prognostic_count
  character(len=OPTION_PATH_LEN) :: thismaterial_phase

  ! subroutine to get the prognostic field out of an array of states
  ! intended to get the correct pointer to pressure or velocity

  if(present(stat)) stat = 0

  prognostic_count=option_count('/material_phase/scalar_field::Pressure/prognostic')

  if(prognostic_count==1) then
    do i = 1,size(state)
      write(thismaterial_phase, '(a)') "/material_phase::"//trim(state(i)%name)

      if(have_option(trim(thismaterial_phase)//'/scalar_field::Pressure/prognostic')) then
        sfield=>extract_scalar_field(state(i),'Pressure')
      end if

    end do
  else if(prognostic_count>1) then
    if(present(stat)) then
      stat = 2
    else
      FLExit("Multiple prognostic pressure fields.")
    end if
  else
    if(present(stat)) then
      stat = 1
    else
      FLExit("No prognostic pressure field found.")
    end if
  end if

  end function extract_prognostic_pressure

  function extract_prognostic_velocity(state, stat) result(vfield)

  type(state_type), dimension(:), intent(in) :: state
  type(vector_field), pointer :: vfield
  integer, optional :: stat

  integer :: i, prognostic_count
  character(len=OPTION_PATH_LEN) :: thismaterial_phase

  ! subroutine to get the prognostic field out of an array of states
  ! intended to get the correct pointer to pressure or velocity

  if(present(stat)) stat = 0

  prognostic_count=option_count('/material_phase/vector_field::Velocity/prognostic')

  if(prognostic_count==1) then

    do i = 1,size(state)
      write(thismaterial_phase, '(a)') "/material_phase::"//trim(state(i)%name)

      if(have_option(trim(thismaterial_phase)//'/vector_field::Velocity/prognostic')) then
        vfield=>extract_vector_field(state(i),'Velocity')
      end if

    end do
  else if(prognostic_count>1) then
    if(present(stat)) then
      stat = 2
    else
      FLExit("Multiple prognostic velocity fields.")
    end if
  else
    if(present(stat)) then
      stat = 1
    else
      FLExit("No prognostic velocity field found.")
    end if
  end if

  end function extract_prognostic_velocity

end module field_options
