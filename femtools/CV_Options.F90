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
module cv_options
  !!< Module containing general tools for discretising Control Volume problems.
  use spud
  use fldebug
  use cvtools, only: complete_cv_field_path
  use field_options, only: complete_field_path
  use global_parameters, only: FIELD_NAME_LEN, OPTION_PATH_LEN
  use element_numbering, only: FAMILY_SIMPLEX, FAMILY_CUBE
  use futils

  implicit none

  integer, parameter, public :: CV_FACEVALUE_NONE=0, &
                                CV_FACEVALUE_FIRSTORDERUPWIND=1, &
                                CV_FACEVALUE_TRAPEZOIDAL=2, &
                                CV_FACEVALUE_FINITEELEMENT=3, &
                                CV_FACEVALUE_HYPERC=4, &
                                CV_FACEVALUE_ULTRAC=5, &
                                CV_FACEVALUE_POTENTIALULTRAC=6, &
                                CV_FACEVALUE_FIRSTORDERDOWNWIND=7

  integer, parameter, public :: CV_DIFFUSION_NONE=0, &
                                CV_DIFFUSION_BASSIREBAY=1, &
                                CV_DIFFUSION_ELEMENTGRADIENT=2

  integer, parameter, public :: CV_DOWNWIND_PROJECTION_NODE=1, &
                                CV_DONOR_PROJECTION_NODE=2

  integer, parameter, public :: CV_LIMITER_NONE=0, &
                                CV_LIMITER_SWEBY=1, &
                                CV_LIMITER_ULTIMATE=2

  integer, public, parameter :: CV_UPWINDVALUE_NONE=0, &
                                CV_UPWINDVALUE_PROJECT_POINT=1, &
                                CV_UPWINDVALUE_PROJECT_GRAD=2, &
                                CV_UPWINDVALUE_LOCAL=3, &
                                CV_UPWINDVALUE_STRUCTURED=4

   type cv_options_type
      ! this is a wrapper type to pass around all the options a control
      ! volume field needs to be advected
      ! creating this saves on the overhead of calling the options dictionary

      ! what CV_FACEVALUE_ to use
      integer :: facevalue
      ! what CV_DIFFUSIONSCHEME_ to use
      integer :: diffusionscheme
      ! whether to limit the face value spatially and temporally
      logical :: limit_facevalue, limit_theta
      ! what CV_LIMITER_ to use
      integer :: limiter
      ! time and conservation discretisation options
      real :: theta, ptheta, beta
      ! the slopes of the limiter being used (if Sweby)
      real, dimension(2) :: limiter_slopes
      ! the targets for UltraC and PotentialUltraC
      real :: target_max, target_min
      ! the targets for coupled_cv
      real :: sum_target_max, sum_target_min
      ! what do we do if the Potential isn't sufficient?
      logical :: hyperc_switch, potential_flux
      ! what upwind scheme are we using (if any)
      integer :: upwind_scheme

   end type cv_options_type

  private
  public :: cv_options_type, get_cv_options, &
            cv_projection_node, cv_facevalue_integer

contains

  function get_cv_options(option_path, element_family, dim, coefficient_field) result(cv_options)

    ! This function retrieves all current control volume
    ! discretisation options wrapped in a cv_options_type.

    ! Defaults deal with the case where nonprognostic fields
    ! are passed in.

    character(len=*), intent(in) :: option_path
    integer, intent(in) :: element_family, dim
    logical, intent(in), optional :: coefficient_field
    type(cv_options_type) :: cv_options

    character(len=FIELD_NAME_LEN) :: tmpstring
    integer :: stat

    ! spatial discretisation options
    call get_option(trim(complete_cv_field_path(option_path))//&
                        "/face_value[0]/name", &
                        tmpstring)
    cv_options%facevalue = cv_facevalue_integer(tmpstring)

    call get_option(trim(complete_cv_field_path(option_path))//&
                        "/diffusion_scheme[0]/name", &
                        tmpstring, default="None")
    cv_options%diffusionscheme = cv_diffusionscheme_integer(tmpstring)

    cv_options%limit_facevalue = have_option(trim(complete_cv_field_path(option_path))//&
                        "/face_value[0]/limit_face_value")

    call get_option(trim(complete_cv_field_path(option_path))//&
                        "/face_value[0]/limit_face_value/limiter[0]/name", &
                        tmpstring, default="None")
    cv_options%limiter = cv_limiter_integer(tmpstring)

    call get_option(trim(complete_cv_field_path(option_path))//&
                    '/face_value[0]/target_maximum', &
                    cv_options%target_max, default=1.0)
    call get_option(trim(complete_cv_field_path(option_path))//&
                    '/face_value[0]/target_minimum', &
                    cv_options%target_min, default=0.0)

    call get_option(trim(complete_cv_field_path(option_path))//&
                    '/parent_sum/target_maximum', &
                    cv_options%sum_target_max, default=1.0)
    call get_option(trim(complete_cv_field_path(option_path))//&
                    '/parent_sum/target_minimum', &
                    cv_options%sum_target_min, default=0.0)

    cv_options%hyperc_switch = have_option(trim(complete_cv_field_path(option_path))//&
                        "/face_value[0]/switch_to_hyperc")
    cv_options%potential_flux = have_option(trim(complete_cv_field_path(option_path))//&
                        "/face_value[0]/use_potential_flux")

    ! temporal discretisation options
    cv_options%limit_theta = have_option(trim(complete_field_path(option_path, stat=stat))//&
                        "/temporal_discretisation&
                        &/control_volumes/limit_theta")
    call get_option(trim(complete_field_path(option_path, stat=stat))//&
                        "/temporal_discretisation&
                        &/theta", cv_options%theta)
    if (cv_options%facevalue==CV_FACEVALUE_FIRSTORDERUPWIND) then
      call get_option(trim(complete_field_path(option_path, stat=stat))//&
                          "/temporal_discretisation&
                          &/control_volumes/pivot_theta", &
                          cv_options%ptheta, stat=stat)
      if(stat==0) then
        if(cv_options%ptheta/=cv_options%theta) then
          ewrite(-1,*) "Found a different pivot_theta and theta for the field with"
          ewrite(-1,*) "option_path: "//trim(option_path)
          ewrite(-1,*) "This field uses first order upwinding."
          ewrite(-1,*) "As the pivot is also first order upwinding theta and"
          ewrite(-1,*) "pivot theta should be the same."
          FLExit("Switch off pivot_theta or set it to be the same as theta.")
        end if
      else
        ! the pivot is a first order upwind value value too so
        ! the pivot theta should be the same
        cv_options%ptheta = cv_options%theta
      end if
    else
      call get_option(trim(complete_field_path(option_path, stat=stat))//&
                          "/temporal_discretisation&
                          &/control_volumes/pivot_theta", &
                          cv_options%ptheta, default=1.0)
    end if
    if(present_and_true(coefficient_field)) then
      ! if these options are for a field that just a coefficient to the main
      ! equation then this isn't need.
      ! initialise it to something insane to make sure it will be noticed if used.
      cv_options%beta = -666.0
    else
      call get_option(trim(complete_field_path(option_path, stat=stat))//&
                          "/spatial_discretisation&
                          &/conservative_advection", &
                          cv_options%beta)
    end if
    call get_option(trim(complete_cv_field_path(option_path))//&
                    '/face_value[0]/limit_face_value/limiter[0]/slopes&
                    &/lower', cv_options%limiter_slopes(1), default=1.0)
    call get_option(trim(complete_cv_field_path(option_path))//&
                    '/face_value[0]/limit_face_value/limiter[0]/slopes&
                    &/upper', cv_options%limiter_slopes(2), default=2.0)

    cv_options%upwind_scheme=cv_upwind_scheme(option_path, element_family, dim)

  end function get_cv_options

  integer function cv_facevalue_integer(face_discretisation)

    character(len=*) :: face_discretisation

    select case(trim(face_discretisation))
    case ("FirstOrderUpwind")
      cv_facevalue_integer = CV_FACEVALUE_FIRSTORDERUPWIND
    case ("Trapezoidal")
      cv_facevalue_integer = CV_FACEVALUE_TRAPEZOIDAL
    case ("FiniteElement")
      cv_facevalue_integer = CV_FACEVALUE_FINITEELEMENT
    case ( "HyperC" )
      cv_facevalue_integer = CV_FACEVALUE_HYPERC
    case ( "UltraC" )
      cv_facevalue_integer = CV_FACEVALUE_ULTRAC
    case ( "PotentialUltraC" )
      cv_facevalue_integer = CV_FACEVALUE_POTENTIALULTRAC
    case ("FirstOrderDownwind")
      cv_facevalue_integer = CV_FACEVALUE_FIRSTORDERDOWNWIND
    case ("None")
      cv_facevalue_integer = CV_FACEVALUE_NONE
    case default
      FLAbort("Unknown control volume face value scheme.")
    end select

  end function cv_facevalue_integer

  integer function cv_diffusionscheme_integer(face_discretisation)

    character(len=*) :: face_discretisation

    select case(trim(face_discretisation))
    case ("BassiRebay")
      cv_diffusionscheme_integer = CV_DIFFUSION_BASSIREBAY
    case ("ElementGradient")
      cv_diffusionscheme_integer = CV_DIFFUSION_ELEMENTGRADIENT
    case ("None")
      cv_diffusionscheme_integer = CV_DIFFUSION_NONE
    case default
      FLAbort("Unknown control volume diffusion scheme.")
    end select

  end function cv_diffusionscheme_integer

  integer function cv_limiter_integer(limiter_name)

    character(len=*) :: limiter_name

    select case(trim(limiter_name))
    case ("None")
      cv_limiter_integer = CV_LIMITER_NONE
    case ("Sweby")
      cv_limiter_integer = CV_LIMITER_SWEBY
    case ("Ultimate")
      cv_limiter_integer = CV_LIMITER_ULTIMATE
    case default
      FLAbort("Unknown control volume face value limiter option.")
    end select

  end function cv_limiter_integer

  integer function cv_projection_node(option_path)

     character(len=*) :: option_path

     if((have_option(trim(complete_cv_field_path(option_path))//&
                    &"/face_value[0]/limit_face_value/limiter[0]"//&
                    &"/project_upwind_value_from_gradient"//&
                    &"/project_from_downwind_value")).or.&
        (have_option(trim(complete_cv_field_path(option_path))//&
                    &"/face_value[0]"//&
                    &"/project_upwind_value_from_gradient"//&
                    &"/project_from_downwind_value"))) then

       cv_projection_node = CV_DOWNWIND_PROJECTION_NODE

     else if((have_option(trim(complete_cv_field_path(option_path))//&
                    &"/face_value[0]/limit_face_value/limiter[0]"//&
                    &"/project_upwind_value_from_gradient"//&
                    &"/project_from_donor_value")).or.&
             (have_option(trim(complete_cv_field_path(option_path))//&
                    &"/face_value[0]"//&
                    &"/project_upwind_value_from_gradient"//&
                    &"/project_from_donor_value"))) then

       cv_projection_node = CV_DONOR_PROJECTION_NODE

     else
       FLAbort("Unknown projection_node")
     end if

  end function cv_projection_node

  function cv_upwind_scheme(option_path, element_family, dim) result(upwind_scheme)

    character(len=*), intent(in) :: option_path
    integer, intent(in) :: element_family, dim
    integer :: upwind_scheme

    character(len=OPTION_PATH_LEN) :: spatial_discretisation_path, upwind_value_path
    logical :: project_point, project_grad, local, structured

    spatial_discretisation_path = trim(complete_cv_field_path(option_path))

    if(have_option(trim(spatial_discretisation_path)//"/face_value[0]/limit_face_value")) then
      upwind_value_path = trim(spatial_discretisation_path)//"/face_value[0]/limit_face_value/limiter[0]"
    else
      upwind_value_path = trim(spatial_discretisation_path)//"/face_value[0]"
    end if

    ! do we want to project to the upwind value from a point?
    project_point = have_option(trim(upwind_value_path)//&
                    '/project_upwind_value_from_point')

    ! do we want to project to the upwind value using the gradient?
    project_grad = have_option(trim(upwind_value_path)//&
                    '/project_upwind_value_from_gradient')

    ! do we want to use local values as the upwind value?
    local = have_option(trim(upwind_value_path)//&
                    '/locally_bound_upwind_value')

    ! do we want to use pseudo-structured values as the upwind value?
    structured = have_option(trim(upwind_value_path)//&
                    '/pseudo_structured_upwind_value')

    ! in case none (or both) selected default to family type selection
    select case(element_family)
    case (FAMILY_SIMPLEX) ! use projection except in 1d
      if((.not.project_point).and.(.not.local).and.(.not.project_grad).and.(.not.structured)) then
        if(dim==1) then
          local = .true.
        else
          project_point = .true.
        end if
      end if
    case (FAMILY_CUBE) ! use local
      if((.not.project_point).and.(.not.local).and.(.not.project_grad).and.(.not.structured)) then
        local=.true.
      end if
    case default
      FLAbort('Illegal element family')
    end select

    if(project_point) then
      upwind_scheme = CV_UPWINDVALUE_PROJECT_POINT
    else if(project_grad) then
      upwind_scheme = CV_UPWINDVALUE_PROJECT_GRAD
    else if(local) then
      upwind_scheme = CV_UPWINDVALUE_LOCAL
    else if(structured) then
      upwind_scheme = CV_UPWINDVALUE_STRUCTURED
    else
      upwind_scheme = CV_UPWINDVALUE_NONE
    end if

  end function cv_upwind_scheme

end module cv_options
