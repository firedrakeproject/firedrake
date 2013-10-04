program ballistics
  !!< This is a simple program which illustrates the use of spud to drive a
  !!< trivial simulation.
  use spud
  implicit none

  integer, parameter :: D=kind(0.0D0)

  type projectile_type
     character(len=256) :: name
     real(D), dimension(2) :: velocity
     real(D), dimension(2) :: position
     ! Whether this projectile is still airbourne.
     logical :: active = .true.
  end type projectile_type

  ! The list of projectiles to be evolved
  type(projectile_type), dimension(:), allocatable :: projectiles

  ! The acceleration due to gravity.
  real(D) :: gravity

  real(D) :: current_time, finish_time, dt

  character(len=1024) :: time_integration_scheme

  character(len=1024) :: filename

  integer, parameter :: output_unit=42

  integer :: i

  !------------------------------------------------------------------------
  ! Program starts here
  !------------------------------------------------------------------------

  ! Read the input file name from the command line.
  call read_command_line(filename)

  ! Load the input options into the dictionary.
  call load_options(filename)

  call setup_projectiles(projectiles)

  call setup_output_file(projectiles, output_unit)

  call get_option("/timestepping/finish_time", finish_time)
  call get_option("/timestepping/dt", dt)
  call get_option("/timestepping/time_integration_scheme",&
       & time_integration_scheme)
  call get_option("/gravity", gravity)

  current_time=0.0
  timeloop: do while (current_time<finish_time)
     current_time=current_time+dt

     projectileloop: do i=1,size(projectiles)

        if (projectiles(i)%active) then

           call move_projectile(projectiles(i), dt)

        end if

     end do projectileloop

     call output_projectile_positions(current_time, projectiles, output_unit)

  end do timeloop

  close(output_unit)

contains

  subroutine read_command_line(filename)
    ! Read the input filename on the command line.
    character(len=*), intent(out) :: filename
    integer :: status

    call get_command_argument(1, value=filename, status=status)

    select case(status)
    case(1:)
       call usage
       stop
    case(:-1)
       write(0,*) "Warning: truncating filename"
    end select

  end subroutine read_command_line

  subroutine usage

    write (0,*) "usage: ballistics <options_file_name>"

  end subroutine usage

  subroutine setup_projectiles(projectiles)
    ! Read in the starting positions and velocities of the projectiles.
    type(projectile_type), dimension(:), allocatable, intent(inout)&
         :: projectiles

    integer :: projectile_count, i
    character(len=1024) :: path

    projectile_count=option_count("/projectile")

    allocate(projectiles(projectile_count))

    do i=1,projectile_count

       write(path, '(a,i0,a)') "/projectile[",i-1,"]"

       call get_option(trim(path)//"/name", projectiles(i)%name)

       call get_option(trim(path)//"/initial_velocity", &
            projectiles(i)%velocity)

       ! Note that the launch position is measured along the x axis.
       call get_option(trim(path)//"/launch_position", &
            projectiles(i)%position(1:1))

    end do

  end subroutine setup_projectiles

  subroutine setup_output_file(projectiles, output_unit)
    ! Open the output file and populate the header line.
    type(projectile_type), dimension(:), intent(in) :: projectiles
    integer, intent(in) :: output_unit

    character(len=1024) :: output_filename
    integer :: i

    call get_option("/simulation_name", output_filename)

    open(unit=output_unit, file=trim(output_filename)//".csv", &
         action="write")

    write(output_unit, '(a)', advance="no") "time"

    do i=1,size(projectiles)
       write(output_unit, '(a)', advance="no") &
            ", "//trim(projectiles(i)%name)//"_x, "&
            //trim(projectiles(i)%name)//"_y"
    end do

    ! Finish the line.
    write(output_unit, '(a)') ""

  end subroutine setup_output_file

  subroutine output_projectile_positions(current_time, projectiles, output_unit)
    !!< Simply dump the time and the positions to the output file.
    real(D), intent(in) :: current_time
    type(projectile_type), dimension(:), intent(in) :: projectiles
    integer, intent(in) :: output_unit

    integer :: i

    write(output_unit, '(e15.8,",")', advance="no") current_time

    do i=1,size(projectiles)
       write(output_unit, '(2(e15.8,","))', advance="no") projectiles(i)%position
    end do

    ! Finish the line.
    write(output_unit, '(a)') ""

  end subroutine output_projectile_positions

  subroutine move_projectile(projectile, dt)
    !!< Move the current projectile by dt.
    type(projectile_type), intent(inout) :: projectile
    real(D), intent(in) :: dt

    select case(time_integration_scheme)

    case("explicit_euler")
       ! Move the projectile using the existing velocity.

       projectile%position=projectile%position + dt * projectile%velocity

    case("analytic")
       ! Move the projectile using s= u*dt + 0.5*a*dt**2

       projectile%position=projectile%position + dt * projectile%velocity &
            & + 0.5_D*dt**2*(/0.0_D, -gravity/)

    case default
       write(0,*) "Unknown time integration scheme"

    end select

    ! Calculate the new velocity.
    projectile%velocity=projectile%velocity + dt * (/0.0_D, -gravity/)

    ! Deactivate projectiles which touch the ground.
    if (projectile%position(2)<=0.0) then
       projectile%active=.false.
    end if

  end subroutine move_projectile


end program ballistics
