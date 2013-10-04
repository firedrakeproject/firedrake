! Contains data output routines

#include "fdebug.h"

module write_state_module
  !!< Data output routines

  use embed_python
  use fields
  use FLDebug
  use spud
  use state_module
  use timers
  use Profiler
  use vtk_interfaces
  use field_options
  use futils
  use global_parameters, only: OPTION_PATH_LEN
  use halos

  implicit none

  private

  public :: initialise_write_state, do_write_state, write_state, write_state_module_check_options, &
            vtk_write_state_new_options

  ! Static variables set by update_dump_times and used by do_write_state
  logical, save :: last_times_initialised = .false.
  real, save :: last_dump_time
  real, save :: last_dump_cpu_time
  real, save :: last_dump_wall_time
  real, save :: real_dump_period
  integer, save :: int_dump_period

contains

  subroutine initialise_write_state
    !!< Initialises the write_state module (setting the last_write_state_*time
    !!< variables)

    call update_dump_times

  end subroutine initialise_write_state

  function do_write_state(current_time, timestep, adjoint)
    !!< Data output test routine. Test conditions listed under /io. Returns true
    !!< if these conditions are satisfied and false otherwise.

    real, intent(in) :: current_time
    integer, intent(in) :: timestep
    logical, intent(in), optional :: adjoint

    logical :: do_write_state

    character(len = OPTION_PATH_LEN) :: func

    integer ::  i, stat
    real :: current_cpu_time, current_wall_time

    do_write_state = .false.

    do i = 1, 5
      select case(i)
        case(1)
          if(.not. last_times_initialised) then
            ! if the last_dump*_time variables have not been initialised, assume write_state should be called
            do_write_state = .true.
            exit
          end if
        case(2)
          if(have_option("/io/dump_period")) then
            if(real_dump_period == 0.0 .or. dump_count_greater(current_time, last_dump_time, real_dump_period, adjoint=adjoint)) then
              if(have_option("/io/dump_period/constant")) then
                 call get_option("/io/dump_period/constant", real_dump_period)
              else if (have_option("/io/dump_period/python")) then
                 call get_option("/io/dump_period/python", func)
                 call real_from_python(func, current_time, real_dump_period)
              else
                 FLAbort("Unable to determine dump period type.")
              end if
              if(real_dump_period < 0.0) then
                 FLExit("Dump period cannot be negative.")
              end if
              do_write_state = .true.
              exit
            end if
          end if
        case(3)
          if(have_option("/io/dump_period_in_timesteps")) then
            if(int_dump_period == 0 .or. mod(timestep, int_dump_period) == 0) then
              if(have_option("/io/dump_period_in_timesteps/constant")) then
                call get_option("/io/dump_period_in_timesteps/constant", int_dump_period)
              else if (have_option("/io/dump_period_in_timesteps/python")) then
                call get_option("/io/dump_period_in_timesteps/python", func)
                call integer_from_python(func, current_time, int_dump_period)
              else
                 FLAbort("Unable to determine dump period type.")
              end if
              if(int_dump_period < 0) then
                FLExit("Dump period cannot be negative.")
              END if
              do_write_state = .true.
              exit
            end if
          end if
        case(4)
          call cpu_time(current_cpu_time)
          call allmax(current_cpu_time)
          call get_option("/io/cpu_dump_period", real_dump_period, stat)
          if(stat == SPUD_NO_ERROR) then
            if(real_dump_period == 0.0 .or. dump_count_greater(current_cpu_time, last_dump_cpu_time, real_dump_period)) then
              do_write_state = .true.
              exit
            end if
          end if
        case(5)
          current_wall_time = wall_time()
          call allmax(current_wall_time)
          call get_option("/io/wall_time_dump_period", real_dump_period, stat)
          if(stat == SPUD_NO_ERROR) then
            if(real_dump_period == 0.0 .or. dump_count_greater(current_wall_time, last_dump_wall_time, real_dump_period)) then
              do_write_state = .true.
              exit
            end if
          end if
        case default
          FLAbort("Invalid loop index.")
      end select
    end do

    if(do_write_state) then
      ewrite(2, *) "do_write_state returning .true."
    else
      ewrite(2, *) "do_write_state returning .false."
    end if

  contains

    pure function dump_count_greater(later_time, earlier_time, dump_period, adjoint)
      !!< Return if the total number of dumps at time later_time is greater
      !!< than the total number of dumps at time earlier_time.

      real, intent(in) :: later_time
      real, intent(in) :: earlier_time
      real, intent(in) :: dump_period
      logical, intent(in), optional :: adjoint

      logical :: dump_count_greater

      if (present_and_true(adjoint)) then
        dump_count_greater = (floor(earlier_time / dump_period) > floor(later_time / dump_period))
      else
        dump_count_greater = (floor(later_time / dump_period) > floor(earlier_time / dump_period))
      endif

    end function dump_count_greater

  end function do_write_state

  subroutine update_dump_times
    !!< Update the last_dump_*time variables.
    character(len = OPTION_PATH_LEN) :: func

    last_times_initialised = .true.
    real_dump_period = huge(0.0)
    int_dump_period = huge(0)

    call get_option("/timestepping/current_time", last_dump_time)
    call cpu_time(last_dump_cpu_time)
    call allmax(last_dump_cpu_time)
    last_dump_wall_time = wall_time()
    call allmax(last_dump_wall_time)
    if(have_option("/io/dump_period/constant")) then
       call get_option("/io/dump_period/constant", real_dump_period)
    else if (have_option("/io/dump_period/python")) then
       call get_option("/io/dump_period/python", func)
       call real_from_python(func, last_dump_time, real_dump_period)
       if(real_dump_period < 0.0) then
         FLExit("Dump period cannot be negative.")
       end if
    else if(have_option("/io/dump_period_in_timesteps/constant")) then
       call get_option("/io/dump_period_in_timesteps/constant", int_dump_period)
    else if (have_option("/io/dump_period_in_timesteps/python")) then
       call get_option("/io/dump_period_in_timesteps/python", func)
       call integer_from_python(func, last_dump_time, int_dump_period)
       if(int_dump_period < 0) then
         FLExit("Dump period cannot be negative.")
       end if
    else
      FLExit("Dump period must be specified (in either simulated time or timesteps).")
    end if

  end subroutine update_dump_times

  subroutine write_state(dump_no, state, adjoint)
    !!< Data output routine. Write output data.

    integer, intent(inout) :: dump_no
    type(state_type), dimension(:), intent(inout) :: state
    logical, intent(in), optional :: adjoint

    character(len = OPTION_PATH_LEN) :: dump_filename, dump_format
    integer :: max_dump_no, stat
    integer :: increment

    ewrite(1, *) "In write_state"
    call profiler_tic("I/O")

    call get_option("/simulation_name", dump_filename)
    call get_option("/io/max_dump_file_count", max_dump_no, stat, default = huge(0))

    dump_no = modulo(dump_no, max_dump_no)

    call get_option("/io/dump_format", dump_format)
    select case(trim(dump_format))
      case("vtk")
         ewrite(2, *) "Writing output " // int2str(dump_no) // " to vtu"
         call vtk_write_state_new_options(dump_filename, dump_no, state)
      case default
        FLAbort("Unrecognised dump file format.")
    end select

    if (present_and_true(adjoint)) then
      increment = -1
    else
      increment = 1
    end if

    dump_no = modulo(dump_no + increment, max_dump_no)
    call update_dump_times

    call profiler_toc("I/O")
    ewrite(1, *) "Exiting write_state"

  end subroutine write_state

  subroutine vtk_write_state_new_options(filename, index, state, write_region_ids)
    !!< Write the state variables out to a vtu file according to options
    !!< set in the options tree. Only fields present in the option tree
    !!< will be written, except for those disabled in the same options tree.
    !!<
    !!< All the fields will be promoted/reduced to the degree of the
    !!< chosen mesh.

    character(len=*), intent(in) :: filename  !! Base filename with no trailing _number.vtu
    integer, intent(in), optional :: index    !! Index number of dump for filename.
    type(state_type), dimension(:), intent(inout) :: state
    logical, intent(in), optional :: write_region_ids

    type(vector_field), pointer :: model_coordinate
    type(mesh_type), pointer :: model_mesh

    type(scalar_field), dimension(:), allocatable :: lsfields
    type(vector_field), dimension(:), allocatable :: lvfields
    type(tensor_field), dimension(:), allocatable :: ltfields
    character(len = FIELD_NAME_LEN) :: field_name, mesh_name
    integer :: i, f, counter
    logical :: multi_state

    ewrite(1, *) "In vtk_write_state_new_options"

    call get_option("/io/output_mesh[0]/name", mesh_name)
    model_mesh => extract_mesh(state(1), mesh_name)

    multi_state = size(state) > 1

    ! count number of scalar fields in output:
    counter = 0
    do i = 1, size(state)
      if (associated(state(i)%scalar_fields)) then
        do f = 1, size(state(i)%scalar_fields)
          field_name = state(i)%scalar_fields(f)%ptr%name
          if (include_scalar_field_in_vtu(state, i, field_name)) then
            counter = counter + 1
          end if
        end do
      end if
    end do

    ! collect scalar fields:
    allocate(lsfields(1:counter))
    counter = 0
    do i = 1, size(state)
      if (associated(state(i)%scalar_fields)) then
        do f = 1, size(state(i)%scalar_fields)
          field_name = state(i)%scalar_fields(f)%ptr%name
          if (include_scalar_field_in_vtu(state, i, field_name)) then
            counter = counter + 1
            lsfields(counter)=extract_scalar_field(state(i), field_name)
            if (multi_state) then
              lsfields(counter)%name = trim(state(i)%name)//'::'//trim(field_name)
            end if
          end if
        end do
      end if
    end do

    ! count number of vector fields in output:
    counter = 0
    do i = 1, size(state)
      if (associated(state(i)%vector_fields)) then
        do f = 1, size(state(i)%vector_fields)
          field_name = state(i)%vector_fields(f)%ptr%name
          if (include_vector_field_in_vtu(state, i, field_name)) then
            counter = counter + 1
          end if
        end do
      end if
    end do

    ! collect vector fields:
    allocate(lvfields(1:counter))
    counter = 0
    do i = 1, size(state)
      if (associated(state(i)%vector_fields)) then
        do f = 1, size(state(i)%vector_fields)
          field_name = state(i)%vector_fields(f)%ptr%name
          if (include_vector_field_in_vtu(state, i, field_name)) then
            counter = counter + 1
            lvfields(counter)=extract_vector_field(state(i), field_name)
            if (multi_state) then
              lvfields(counter)%name = trim(state(i)%name)//'::'//trim(field_name)
            end if
          end if
        end do
      end if
    end do

    ! count number of tensor fields in output:
    counter = 0
    do i = 1, size(state)
      if (associated(state(i)%tensor_fields)) then
        do f = 1, size(state(i)%tensor_fields)
          field_name = state(i)%tensor_fields(f)%ptr%name
          if (include_tensor_field_in_vtu(state, i, field_name)) then
            counter = counter + 1
          end if
        end do
      end if
    end do

    ! collect tensor fields:
    allocate(ltfields(1:counter))
    counter = 0
    do i = 1, size(state)
      if (associated(state(i)%tensor_fields)) then
        do f = 1, size(state(i)%tensor_fields)
          field_name = state(i)%tensor_fields(f)%ptr%name
          if (include_tensor_field_in_vtu(state, i, field_name)) then
            counter = counter + 1
            ltfields(counter)=extract_tensor_field(state(i), field_name)
            if (multi_state) then
              ltfields(counter)%name = trim(state(i)%name)//'::'//trim(field_name)
            end if
          end if
        end do
      end if
    end do

    ewrite(2, *) "Writing using mesh " // trim(mesh_name)
    ewrite(2, "(a,i0,a)") "Writing ", size(lsfields), " scalar field(s)"
    ewrite(2, "(a,i0,a)") "Writing ", size(lvfields), " vector field(s)"
    ewrite(2, "(a,i0,a)") "Writing ", size(ltfields), " tensor field(s)"

    model_coordinate=>get_external_coordinate_field(state(1), model_mesh)

    call vtk_write_fields(filename, index, &
         model_coordinate, &
         model_mesh,  &
         sfields=lsfields, &
         vfields=lvfields, &
         tfields=ltfields, &
         write_region_ids=write_region_ids)

    ewrite(1, *) "Exiting vtk_write_state_new_options"

  end subroutine vtk_write_state_new_options

  logical function include_scalar_field_in_vtu(state, istate, field_name)
    !!< function that uses optionpath and state number to work out
    !!< if a field should be written out (skipping aliased fields)

    type(state_type), dimension(:), intent(in):: state
    integer, intent(in):: istate
    character(len=*), intent(in):: field_name

    type(scalar_field), pointer:: field
    character(len=OPTION_PATH_LEN) output_option_path
    logical is_old_field, is_nonlinear_field, is_iterated_field

    integer :: stat

    if (field_name=='Time') then
      field => extract_scalar_field(state(istate), field_name)
      ! Time is special, always included (unless it's aliased)
      include_scalar_field_in_vtu=.not.aliased(field)
      return
    end if

    if (.not. has_scalar_field(state(istate), field_name)) then
      ! not even in state, so no
      include_scalar_field_in_vtu=.false.
      return
    end if

    is_old_field=.false.
    is_nonlinear_field=.false.
    is_iterated_field=.false.

    field => extract_scalar_field(state(istate), field_name)
    if (len_trim(field%option_path)==0) then
      ! fields without option paths
      if (starts_with(field_name, 'Old')) then
        is_old_field=.true.
        field => extract_scalar_field(state(istate), field_name(4:), stat=stat)
        if (stat /= 0) then
          include_scalar_field_in_vtu = .false.
          return
        end if
      else if (starts_with(field_name, 'Nonlinear')) then
        is_nonlinear_field=.true.
        field => extract_scalar_field(state(istate), field_name(10:), stat=stat)
        if (stat /= 0) then
          include_scalar_field_in_vtu = .false.
          return
        end if
      else if (starts_with(field_name, 'Iterated')) then
        is_iterated_field=.true.
        field => extract_scalar_field(state(istate), field_name(9:), stat=stat)
        if (stat /= 0) then
          include_scalar_field_in_vtu = .false.
          return
        end if
      else
        include_scalar_field_in_vtu=.false.
        return
      end if
    end if

    if (starts_with(field%option_path,'/material_phase[')) then
      if (aliased(field)) then
        ! option_path points to other material_phase
        ! must be an aliased field
        include_scalar_field_in_vtu=.false.
        return
      end if
    else
      ! fields outside any material_phase
      ! only output once for first state:
      if (istate/=1) then
        include_scalar_field_in_vtu=.false.
        return
      end if
    end if

    ! if we get here the field is not aliased and has an option_path
    ! now we let the user decide!

    output_option_path=trim(complete_field_path(field%option_path, name=trim(field_name)))//'/output'

    if (is_old_field) then
      include_scalar_field_in_vtu=have_option(trim(output_option_path)//'/include_previous_time_step')
    else if (is_nonlinear_field) then
      include_scalar_field_in_vtu=have_option(trim(output_option_path)//'/include_nonlinear_field')
    else if (is_iterated_field) then
      include_scalar_field_in_vtu=have_option(trim(output_option_path)//'/include_nonlinear_field')
    else
      include_scalar_field_in_vtu=.not. have_option(trim(output_option_path)//'/exclude_from_vtu')
    end if

  end function include_scalar_field_in_vtu

  logical function include_vector_field_in_vtu(state, istate, field_name)
    !!< function that uses optionpath and state number to work out
    !!< if a field should be written to vtu (skipping aliased fields)

    type(state_type), dimension(:), intent(in):: state
    integer, intent(in):: istate
    character(len=*), intent(in):: field_name

    type(vector_field), pointer:: field
    character(len=OPTION_PATH_LEN) output_option_path
    logical is_old_field, is_nonlinear_field, is_iterated_field

    integer :: stat

    if (.not. has_vector_field(state(istate), field_name)) then
      ! not even in state, so no
      include_vector_field_in_vtu=.false.
      return
    end if

    is_old_field=.false.
    is_nonlinear_field=.false.
    is_iterated_field=.false.

    field => extract_vector_field(state(istate), field_name)
    if (len_trim(field%option_path)==0) then
      ! fields without option paths
      if (field_name=="OldCoordinate") then
        include_vector_field_in_vtu=.false.
        return
      else if (field_name=="IteratedCoordinate") then
        include_vector_field_in_vtu=.false.
        return
      else if (field_name=="OldGridVelocity") then
        include_vector_field_in_vtu=.false.
        return
      else if (field_name=="IteratedGridVelocity") then
        include_vector_field_in_vtu=.false.
        return
      else if (starts_with(field_name, 'Old')) then
        is_old_field=.true.
        field => extract_vector_field(state(istate), field_name(4:), stat=stat)
        if (stat /= 0) then
          include_vector_field_in_vtu = .false.
          return
        end if
      else if (starts_with(field_name, 'Nonlinear')) then
        is_nonlinear_field=.true.
        field => extract_vector_field(state(istate), field_name(10:), stat=stat)
        if (stat /= 0) then
          include_vector_field_in_vtu = .false.
          return
        end if
      else if (starts_with(field_name, 'Iterated')) then
        is_iterated_field=.true.
        field => extract_vector_field(state(istate), field_name(9:), stat=stat)
        if (stat /= 0) then
          include_vector_field_in_vtu = .false.
          return
        end if
      else
        include_vector_field_in_vtu=.false.
        return
      end if
    end if

    if (starts_with(field%option_path,'/material_phase[')) then
      if (aliased(field)) then
        ! option_path points to other material_phase
        ! must be an aliased field
        include_vector_field_in_vtu=.false.
        return
      end if
    else
      ! fields outside any material_phase
      ! only output once for first state:
      if (istate/=1) then
        include_vector_field_in_vtu=.false.
        return
      end if
    end if

    ! if we get here the field is not aliased and has an option_path
    ! now we let the user decide!

    output_option_path=trim(complete_field_path(field%option_path))//'/output'

    if (is_old_field) then
     include_vector_field_in_vtu=have_option(trim(output_option_path)//'/include_previous_time_step')
    else if (is_nonlinear_field) then
      include_vector_field_in_vtu=have_option(trim(output_option_path)//'/include_nonlinear_field')
    else if (is_iterated_field) then
      include_vector_field_in_vtu=have_option(trim(output_option_path)//'/include_nonlinear_field')
    else
      include_vector_field_in_vtu=.not. have_option(trim(output_option_path)//'/exclude_from_vtu')
    end if

  end function include_vector_field_in_vtu

  logical function include_tensor_field_in_vtu(state, istate, field_name)
    !!< function that uses optionpath and state number to work out
    !!< if a field should be written to vtu (skipping aliased fields)

    type(state_type), dimension(:), intent(in):: state
    integer, intent(in):: istate
    character(len=*), intent(in):: field_name

    type(tensor_field), pointer:: field
    character(len=OPTION_PATH_LEN) output_option_path
    logical is_old_field, is_nonlinear_field, is_iterated_field

    integer :: stat

    if (.not. has_tensor_field(state(istate), field_name)) then
      ! not even in state, so no
      include_tensor_field_in_vtu=.false.
      return
    end if

    is_old_field=.false.
    is_nonlinear_field=.false.
    is_iterated_field=.false.

    field => extract_tensor_field(state(istate), field_name)
    if (len_trim(field%option_path)==0) then
      ! fields without option paths
      if (starts_with(field_name, 'Old')) then
        is_old_field=.true.
        field => extract_tensor_field(state(istate), field_name(4:), stat=stat)
        if (stat /= 0) then
          include_tensor_field_in_vtu = .false.
          return
        end if
      else if (starts_with(field_name, 'Nonlinear')) then
        is_nonlinear_field=.true.
        field => extract_tensor_field(state(istate), field_name(10:), stat=stat)
        if (stat /= 0) then
          include_tensor_field_in_vtu = .false.
          return
        end if
      else if (starts_with(field_name, 'Iterated')) then
        is_iterated_field=.true.
        field => extract_tensor_field(state(istate), field_name(9:), stat=stat)
        if (stat /= 0) then
          include_tensor_field_in_vtu = .false.
          return
        end if
      else
        include_tensor_field_in_vtu=.false.
        return
      end if
    end if

    if (starts_with(field%option_path,'/material_phase[')) then
      if (aliased(field)) then
        ! option_path points to other material_phase
        ! must be an aliased field
        include_tensor_field_in_vtu=.false.
        return
      end if
    else
      ! fields outside any material_phase
      ! only output once for first state:
      if (istate/=1) then
        include_tensor_field_in_vtu=.false.
        return
      end if
    end if

    ! if we get here the field is not aliased and has an option_path
    ! now we let the user decide!

    output_option_path=trim(complete_field_path(field%option_path))//'/output'

    if (is_old_field) then
      include_tensor_field_in_vtu=have_option(trim(output_option_path)//'/include_previous_time_step')
    else if (is_nonlinear_field) then
      include_tensor_field_in_vtu=have_option(trim(output_option_path)//'/include_nonlinear_field')
    else if (is_iterated_field) then
      include_tensor_field_in_vtu=have_option(trim(output_option_path)//'/include_nonlinear_field')
    else
      include_tensor_field_in_vtu=.not. have_option(trim(output_option_path)//'/exclude_from_vtu')
    end if

  end function include_tensor_field_in_vtu

  subroutine write_state_module_check_options
    !!< Check output related options

    character(len = OPTION_PATH_LEN) :: dump_format, output_mesh_name, func
    integer :: int_dump_period, max_dump_file_count, stat
    real :: real_dump_period, current_time

    ewrite(2, *) "Checking output options"

    call get_option("/timestepping/current_time", current_time)

    call get_option("/io/dump_format", dump_format, stat)
    if(stat == SPUD_NO_ERROR) then
      if(trim(dump_format) == "vtk") then
        call get_option("/io/output_mesh[0]/name", output_mesh_name, stat = stat)
        if(stat /= SPUD_NO_ERROR) then
          FLExit("An output mesh must be specified if using a VTK dump format.")
        else if(option_count("/geometry/mesh::" // output_mesh_name) == 0) then
          FLExit("Output mesh " // trim(output_mesh_name) // " is not defined.")
        end if
      else
        FLExit('Unrecognised dump format "' // trim(dump_format) // '"specified.')
      end if
    else
      FLExit("Dump format must be specified.")
    end if

    if(have_option("/io/dump_period/constant")) then
      call get_option("/io/dump_period/constant", real_dump_period, stat)
      if(stat == SPUD_NO_ERROR) then
        if(real_dump_period < 0.0) then
          FLExit("Dump period cannot be negative.")
        end if
      end if
    else if(have_option("/io/dump_period/python")) then
      call get_option("/io/dump_period/python", func)
      call real_from_python(func, current_time, real_dump_period, STAT)
      if(stat == SPUD_NO_ERROR) then
        if(real_dump_period < 0.0) then
          FLExit("Dump period cannot be negative.")
        end if
      end if
    else if(have_option("/io/dump_period_in_timesteps/constant")) then
      call get_option("/io/dump_period_in_timesteps/constant", int_dump_period, stat)
      if(stat == SPUD_NO_ERROR) then
        if(int_dump_period < 0) then
          FLExit("Dump period cannot be negative.")
        end if
      end if
    else if(have_option("/io/dump_period_in_timesteps/python")) then
      call get_option("/io/dump_period_in_timesteps/python", func)
      call integer_from_python(func, current_time, int_dump_period, stat)
      if(stat == SPUD_NO_ERROR) then
        if(int_dump_period < 0) then
          FLExit("Dump period cannot be negative.")
        end if
      end if
    else
      FLExit("Dump period must be specified (in either simulated time or timesteps).")
    end if

    call get_option("/io/cpu_dump_period", real_dump_period, stat)
    if(stat == SPUD_NO_ERROR) then
      if(real_dump_period < 0.0) then
        FLExit("CPU dump period cannot be negative.")
      end if
    end if

    call get_option("/io/wall_time_dump_period", real_dump_period, stat)
    if(stat == SPUD_NO_ERROR) then
      if(real_dump_period < 0.0) then
        FLExit("Wall time dump period cannot be negative.")
      end if
      if(.not. wall_time_supported()) then
        FLExit("Wall time dump period supplied, but wall time is not available.")
      end if
    end if

    call get_option("/io/max_dump_file_count", max_dump_file_count, stat)
    if(stat == SPUD_NO_ERROR) then
      if(max_dump_file_count <= 0) then
        FLExit("Max dump file count must be positive.")
      end if
    end if

    ewrite(2, *) "Finished checking output options."

  end subroutine write_state_module_check_options

end module write_state_module
