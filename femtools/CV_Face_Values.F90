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
module cv_face_values
  !!< Module containing general tools for discretising Control Volume problems.
  use spud
  use fields
  use sparse_tools
  use element_numbering
  use state_module
  use fldebug
  use cv_shape_functions
  use cv_faces
  use cv_options
  use vector_tools
  use transform_elements
  use field_derivatives, only: grad

  implicit none

  real, private, parameter :: tolerance=tiny(0.0)

  private
  public :: cv_facevalue_integer, &
            evaluate_face_val, &
            theta_val, &
            couple_face_value

contains

  subroutine evaluate_face_val(face_val, old_face_val, &
                      iloc, oloc, ggi, nodes, &
                      cvshape, &
                      field_ele, old_field_ele, &
                      upwind_values, old_upwind_values, &
                      inflow, cfl_ele, &
                      cv_options, save_pos)

    ! given a discretisation type calculate the face value for a field

    ! output:
    real, intent(out) :: face_val, old_face_val

    ! input:
    integer, intent(in) :: iloc, oloc, ggi
    integer, dimension(:) :: nodes
    type(element_type) :: cvshape
    real, dimension(:) :: field_ele, old_field_ele
    type(csr_matrix), intent(in) :: upwind_values, old_upwind_values
    logical, intent(in) :: inflow
    real, dimension(:), intent(in) :: cfl_ele
    type(cv_options_type), intent(in) :: cv_options ! a wrapper type to pass in all the options for control volumes
    integer, intent(inout), optional :: save_pos

    ! local memory:
    real :: upwind_val, donor_val, downwind_val
    real :: old_upwind_val, old_donor_val, old_downwind_val
    real :: cfl_donor
    real :: potential, old_potential
    real :: target_upwind, old_target_upwind, target_downwind, old_target_downwind
    real :: income=0.0
    integer :: l_save_pos

    if(present(save_pos)) then
      l_save_pos=save_pos ! an attempt at optimising the val calls by saving the matrix position
    else
      l_save_pos = 0
    end if

    select case(cv_options%facevalue)
    case (CV_FACEVALUE_FIRSTORDERUPWIND)

      income = merge(1.0,0.0,inflow)

      donor_val = income*field_ele(oloc) + (1.-income)*field_ele(iloc)
      old_donor_val = income*old_field_ele(oloc) + (1.-income)*old_field_ele(iloc)

      face_val = donor_val
      old_face_val = old_donor_val

    case (CV_FACEVALUE_TRAPEZOIDAL)

      face_val = 0.5*(field_ele(oloc) + field_ele(iloc))
      old_face_val = 0.5*(old_field_ele(oloc) + old_field_ele(iloc))

    case (CV_FACEVALUE_FINITEELEMENT)

      face_val = dot_product(cvshape%n(:,ggi), field_ele)
      old_face_val = dot_product(cvshape%n(:,ggi), old_field_ele)

    case (CV_FACEVALUE_HYPERC)

      income = merge(1.0,0.0,inflow)

      cfl_donor = income*cfl_ele(oloc) + (1.-income)*cfl_ele(iloc)

      downwind_val = income*field_ele(iloc) + (1.-income)*field_ele(oloc)
      donor_val = income*field_ele(oloc) + (1.-income)*field_ele(iloc)
      if(inflow) then
        upwind_val = val(upwind_values, nodes(oloc), nodes(iloc), save_pos=l_save_pos)
        ! save_pos should save the value of csr_pos in this call
      else
        upwind_val = val(upwind_values, nodes(iloc), nodes(oloc), save_pos=l_save_pos)
      end if

      old_downwind_val = income*old_field_ele(iloc) + (1.-income)*old_field_ele(oloc)
      old_donor_val = income*old_field_ele(oloc) + (1.-income)*old_field_ele(iloc)
      if(inflow) then
        old_upwind_val = val(old_upwind_values, nodes(oloc), nodes(iloc), save_pos=l_save_pos)
        ! and as inflow is the same it should reuse it in this call
        ! (similarly for all uses below)
      else
        old_upwind_val = val(old_upwind_values, nodes(iloc), nodes(oloc), save_pos=l_save_pos)
      end if

      face_val = hyperc_val(upwind_val, donor_val, downwind_val, cfl_donor)
      old_face_val = hyperc_val(old_upwind_val, old_donor_val, &
                                  old_downwind_val, cfl_donor)

    case (CV_FACEVALUE_ULTRAC)

      income = merge(1.0,0.0,inflow)

      cfl_donor = income*cfl_ele(oloc) + (1.-income)*cfl_ele(iloc)

      downwind_val = income*field_ele(iloc) + (1.-income)*field_ele(oloc)
      donor_val = income*field_ele(oloc) + (1.-income)*field_ele(iloc)
      if(inflow) then
        upwind_val = val(upwind_values, nodes(oloc), nodes(iloc), save_pos=l_save_pos)
      else
        upwind_val = val(upwind_values, nodes(iloc), nodes(oloc), save_pos=l_save_pos)
      end if

      old_downwind_val = income*old_field_ele(iloc) + (1.-income)*old_field_ele(oloc)
      old_donor_val = income*old_field_ele(oloc) + (1.-income)*old_field_ele(iloc)
      if(inflow) then
        old_upwind_val = val(old_upwind_values, nodes(oloc), nodes(iloc), save_pos=l_save_pos)
      else
        old_upwind_val = val(old_upwind_values, nodes(iloc), nodes(oloc), save_pos=l_save_pos)
      end if

      if(downwind_val<upwind_val) then
          target_upwind = cv_options%target_max
          target_downwind = cv_options%target_min
      else
          target_upwind = cv_options%target_min
          target_downwind = cv_options%target_max
      end if

      face_val = hyperc_val(upwind_val=target_upwind, donor_val=donor_val, &
                            downwind_val=target_downwind, cfl_donor=cfl_donor)

      if(old_downwind_val<old_upwind_val) then
          old_target_upwind = cv_options%target_max
          old_target_downwind = cv_options%target_min
      else
          old_target_upwind = cv_options%target_min
          old_target_downwind = cv_options%target_max
      end if

      old_face_val = hyperc_val(upwind_val=old_target_upwind, donor_val=old_donor_val, &
                                downwind_val=old_target_downwind, cfl_donor=cfl_donor)

    case (CV_FACEVALUE_POTENTIALULTRAC)

      income = merge(1.0,0.0,inflow)

      cfl_donor = income*cfl_ele(oloc) + (1.-income)*cfl_ele(iloc)

      downwind_val = income*field_ele(iloc) + (1.-income)*field_ele(oloc)
      donor_val = income*field_ele(oloc) + (1.-income)*field_ele(iloc)
      if(inflow) then
        upwind_val = val(upwind_values, nodes(oloc), nodes(iloc), save_pos=l_save_pos)
      else
        upwind_val = val(upwind_values, nodes(iloc), nodes(oloc), save_pos=l_save_pos)
      end if

      old_downwind_val = income*old_field_ele(iloc) + (1.-income)*old_field_ele(oloc)
      old_donor_val = income*old_field_ele(oloc) + (1.-income)*old_field_ele(iloc)
      if(inflow) then
        old_upwind_val = val(old_upwind_values, nodes(oloc), nodes(iloc), save_pos=l_save_pos)
      else
        old_upwind_val = val(old_upwind_values, nodes(iloc), nodes(oloc), save_pos=l_save_pos)
      end if

      potential = downwind_val + donor_val + upwind_val
      old_potential = old_downwind_val + old_donor_val + old_upwind_val

      target_upwind = upwind_val
      target_downwind = downwind_val

      if(cv_options%hyperc_switch) then
        if(potential>cv_options%target_max-tolerance) then
          if(downwind_val<upwind_val) then
              target_upwind = cv_options%target_max
              target_downwind = cv_options%target_min
          else
              target_upwind = cv_options%target_min
              target_downwind = cv_options%target_max
          end if
        end if
      end if

      if(cv_options%potential_flux) then
!         if(downwind_val-upwind_val<-tolerance) then
        if(downwind_val<upwind_val) then
            target_upwind = min(potential, cv_options%target_max)
            target_downwind = cv_options%target_min
        else
            target_upwind = cv_options%target_min
            target_downwind = min(potential, cv_options%target_max)
        end if
      end if

      face_val = hyperc_val(upwind_val=target_upwind, donor_val=donor_val, &
                            downwind_val=target_downwind, cfl_donor=cfl_donor)

      old_target_upwind = old_upwind_val
      old_target_downwind = old_downwind_val

      if(cv_options%hyperc_switch) then
        if(old_potential>cv_options%target_max-tolerance) then
          if(old_downwind_val<old_upwind_val) then
              old_target_upwind = cv_options%target_max
              old_target_downwind = cv_options%target_min
          else
              old_target_upwind = cv_options%target_min
              old_target_downwind = cv_options%target_max
          end if
        end if
      end if

      if(cv_options%potential_flux) then
!         if(old_downwind_val-old_upwind_val<-tolerance) then
        if(old_downwind_val<old_upwind_val) then
            old_target_upwind = min(old_potential, cv_options%target_max)
            old_target_downwind = cv_options%target_min
        else
            old_target_upwind = cv_options%target_min
            old_target_downwind = min(old_potential, cv_options%target_max)
        end if
      end if

      old_face_val = hyperc_val(upwind_val=old_target_upwind, donor_val=old_donor_val, &
                                downwind_val=old_target_downwind, cfl_donor=cfl_donor)

    case (CV_FACEVALUE_FIRSTORDERDOWNWIND)

      income = merge(1.0,0.0,inflow)

      donor_val = (1.-income)*field_ele(oloc) + income*field_ele(iloc)
      old_donor_val = (1.-income)*old_field_ele(oloc) + income*old_field_ele(iloc)

      face_val = donor_val
      old_face_val = old_donor_val

    end select

    if(cv_options%limit_facevalue) then

      income = merge(1.0,0.0,inflow)

      cfl_donor = income*cfl_ele(oloc) + (1.-income)*cfl_ele(iloc)

      downwind_val = income*field_ele(iloc) + (1.-income)*field_ele(oloc)
      donor_val = income*field_ele(oloc) + (1.-income)*field_ele(iloc)
      if(inflow) then
        upwind_val = val(upwind_values, nodes(oloc), nodes(iloc), save_pos=l_save_pos)
      else
        upwind_val = val(upwind_values, nodes(iloc), nodes(oloc), save_pos=l_save_pos)
      end if

      old_downwind_val = income*old_field_ele(iloc) + (1.-income)*old_field_ele(oloc)
      old_donor_val = income*old_field_ele(oloc) + (1.-income)*old_field_ele(iloc)
      if(inflow) then
        old_upwind_val = val(old_upwind_values, nodes(oloc), nodes(iloc), save_pos=l_save_pos)
      else
        old_upwind_val = val(old_upwind_values, nodes(iloc), nodes(oloc), save_pos=l_save_pos)
      end if

      face_val = limit_val(upwind_val, donor_val, downwind_val, face_val, &
                                cv_options%limiter, cfl_donor, cv_options%limiter_slopes)
      old_face_val = limit_val(old_upwind_val, old_donor_val, old_downwind_val, old_face_val, &
                                cv_options%limiter, cfl_donor, cv_options%limiter_slopes)

    end if

    if(present(save_pos)) then
      save_pos = l_save_pos
    end if

  end subroutine evaluate_face_val

  function hyperc_val(upwind_val, donor_val, downwind_val, cfl_donor)

      ! given an upwind, donor, cfl and downwind value calculate the face
      ! value using hyper-c

      real, intent(in) :: upwind_val, donor_val, downwind_val, cfl_donor

      real :: hyperc_val

      real :: nvd_denom, nvd_donor

      nvd_denom = sign(max(abs(downwind_val-upwind_val),tolerance),&
                       downwind_val-upwind_val)
      nvd_donor = (donor_val-upwind_val)/nvd_denom

      if(((nvd_donor)>0.0).and.((nvd_donor)<1.0)) then
         hyperc_val = upwind_val+&
                      (min(1.0, max((nvd_donor/cfl_donor), nvd_donor)))*nvd_denom
      else
         hyperc_val = donor_val
      end if

   end function hyperc_val

  function limit_val(upwind_val, donor_val, downwind_val, face_val, &
                               limiter, cfl_donor, limiter_slopes)

    ! given an upwind, downwind, donor and face value and the slopes of the limiter
    ! decide if the face value needs to be limited and do so if necessary

    real, intent(in) :: upwind_val, donor_val, downwind_val, face_val, cfl_donor
    integer, intent(in) :: limiter
    real, dimension(2), intent(in) :: limiter_slopes

    real :: limit_val

    real :: nvd_denom, nvd_donor, nvd_face

    limit_val = 0.0

    nvd_denom = sign(max(abs(downwind_val-upwind_val),tolerance),&
                      downwind_val-upwind_val)
    nvd_donor = (donor_val-upwind_val)/nvd_denom
    nvd_face = (face_val-upwind_val)/nvd_denom

    select case(limiter)
    case(CV_LIMITER_SWEBY)

      if(((nvd_donor)>0.0).and.((nvd_donor)<1.0)) then
        limit_val = upwind_val+&
                      (min(1.0, (limiter_slopes(2)*nvd_donor), &
                          max(nvd_face, limiter_slopes(1)*nvd_donor)))*nvd_denom
      else
        limit_val = donor_val
      end if

    case(CV_LIMITER_ULTIMATE)

      if(((nvd_donor)>0.0).and.((nvd_donor)<1.0)) then
        limit_val = upwind_val+&
                      (min(1.0, nvd_donor/cfl_donor, &
                          max(nvd_face, nvd_donor)))*nvd_denom
      else
        limit_val = donor_val
      end if

    end select

  end function limit_val

  subroutine couple_face_value(face_val, old_face_val, &
                               sibling_face_val, old_sibling_face_val, &
                               field_ele, old_field_ele, &
                               sibling_field_ele, old_sibling_field_ele, &
                               upwind_values, old_upwind_values, &
                               inflow, iloc, oloc, nodes, cfl_ele, &
                               cv_options, save_pos)

    real, intent(inout) :: face_val, old_face_val
    real, intent(in) :: sibling_face_val, old_sibling_face_val
    logical, intent(in) :: inflow
    integer, intent(in) :: iloc, oloc
    integer, dimension(:) :: nodes

    real, dimension(:), intent(in) :: cfl_ele, field_ele, old_field_ele, sibling_field_ele, old_sibling_field_ele
    type(csr_matrix), intent(in) :: upwind_values, old_upwind_values
    type(cv_options_type), intent(in) :: cv_options ! a wrapper type to pass in all the options for control volumes
    integer, intent(inout), optional :: save_pos

    ! local memory
    real :: income, cfl_donor
    real :: downwind_val, donor_val, upwind_val
    real :: old_downwind_val, old_donor_val, old_upwind_val
    real :: sibling_downwind_val, sibling_donor_val
    real :: old_sibling_downwind_val, old_sibling_donor_val
    real, dimension(2) :: parent_target_vals, old_parent_target_vals
    integer :: l_save_pos

    if(present(save_pos)) then
      l_save_pos = save_pos
    else
      l_save_pos = 0
    end if

    income = merge(1.0,0.0,inflow)

    cfl_donor = income*cfl_ele(oloc) + (1.-income)*cfl_ele(iloc)

    downwind_val = income*field_ele(iloc) + (1.-income)*field_ele(oloc)

    donor_val = income*field_ele(oloc) + (1.-income)*field_ele(iloc)

    sibling_downwind_val = income*sibling_field_ele(iloc) + (1.-income)*sibling_field_ele(oloc)

    sibling_donor_val = income*sibling_field_ele(oloc) + (1.-income)*sibling_field_ele(iloc)

    if(inflow) then
      upwind_val = val(upwind_values, nodes(oloc), nodes(iloc), save_pos=l_save_pos)
    else
      upwind_val = val(upwind_values, nodes(iloc), nodes(oloc), save_pos=l_save_pos)
    end if

    parent_target_vals = (/cv_options%sum_target_max, cv_options%sum_target_min/)

    face_val = limit_val_coupled(upwind_val, donor_val, downwind_val, face_val, &
                             sibling_donor_val, sibling_face_val, &
                             parent_target_vals, &
                             cv_options%limiter, cfl_donor, cv_options%limiter_slopes)

    old_downwind_val = income*old_field_ele(iloc) + (1.-income)*old_field_ele(oloc)

    old_donor_val = income*old_field_ele(oloc) + (1.-income)*old_field_ele(iloc)

    old_sibling_downwind_val = income*old_sibling_field_ele(iloc) + (1.-income)*old_sibling_field_ele(oloc)

    old_sibling_donor_val = income*old_sibling_field_ele(oloc) + (1.-income)*old_sibling_field_ele(iloc)

    if(inflow) then
      old_upwind_val = val(old_upwind_values, nodes(oloc), nodes(iloc), save_pos=l_save_pos)
    else
      old_upwind_val = val(old_upwind_values, nodes(iloc), nodes(oloc), save_pos=l_save_pos)
    end if

    old_parent_target_vals = (/cv_options%sum_target_max, cv_options%sum_target_min/)

    old_face_val = limit_val_coupled(old_upwind_val, old_donor_val, old_downwind_val, old_face_val, &
                             old_sibling_donor_val, old_sibling_face_val, &
                             old_parent_target_vals, &
                             cv_options%limiter, cfl_donor, cv_options%limiter_slopes)

    if(present(save_pos)) then
      save_pos = l_save_pos
    end if

  end subroutine couple_face_value

  function limit_val_coupled(upwind_val, donor_val, downwind_val, face_val, &
                             sibling_donor_val, sibling_face_val, &
                             parent_target_vals, &
                             limiter, cfl_donor, limiter_slopes)

    ! given an upwind, downwind, donor and face value of a field and its sibling
    ! decide if the field face value needs to be limited and do so if necessary

    real, intent(in) :: upwind_val, donor_val, downwind_val, face_val, cfl_donor
    real, intent(in) :: sibling_donor_val, sibling_face_val
    real, dimension(2), intent(in) :: parent_target_vals
    integer, intent(in) :: limiter
    real, dimension(2), intent(in) :: limiter_slopes

    real :: limit_val_coupled

    real :: nvd_denom, nvd_donor, nvd_face
    real :: nvd_sibling_donor, nvd_sibling_face
    real, dimension(2) :: nvd_parent_targets

    real, dimension(2) :: monotonicity_bounds
    real, dimension(2) :: downwind_lines, limited_lines
    real :: upwind_line

    limit_val_coupled = 0.0

    nvd_denom = sign(max(abs(downwind_val-upwind_val),tolerance),&
                      downwind_val-upwind_val)

    nvd_donor = (donor_val-upwind_val)/nvd_denom
    nvd_face = (face_val-upwind_val)/nvd_denom

    nvd_sibling_donor = (sibling_donor_val-upwind_val)/nvd_denom
    nvd_sibling_face = (sibling_face_val-upwind_val)/nvd_denom
    nvd_parent_targets = (parent_target_vals-2.*upwind_val)/nvd_denom

    downwind_lines = nvd_parent_targets - nvd_sibling_face
    monotonicity_bounds = nvd_parent_targets-nvd_sibling_donor
    upwind_line = nvd_donor + nvd_sibling_donor - nvd_sibling_face

    select case(limiter)
    case(CV_LIMITER_SWEBY)

      limited_lines = limiter_slopes(2)*(nvd_donor + nvd_sibling_donor - nvd_parent_targets) &
                     + nvd_parent_targets - nvd_sibling_face

    case default ! unless Sweby is specified use ULTIMATE (even if limiter type is NONE)

      limited_lines = (1./cfl_donor)*(nvd_donor + nvd_sibling_donor - nvd_parent_targets) &
                      + nvd_parent_targets - nvd_sibling_face
    end select

    if((nvd_donor>minval(monotonicity_bounds)).and.&
        (nvd_donor<maxval(monotonicity_bounds))) then
      limit_val_coupled = upwind_val+&
                  (min(maxval(downwind_lines), &
                        maxval(limited_lines), &
                        max(nvd_face, &
                            minval(downwind_lines), &
                            minval(limited_lines))))*nvd_denom
    else
      limit_val_coupled = upwind_val + upwind_line*nvd_denom
    end if

  end function limit_val_coupled

  function theta_val(iloc, oloc, &
                     face_val, old_face_val, &
                     theta, dt, udotn, &
                     x_ele, limit_theta, &
                     field_ele, old_field_ele, &
                     ftheta)

    ! if necessary limit the theta value and calculate the
    ! time discretised face value from the old and new
    ! face values

    integer, intent(in) :: iloc, oloc
    real, intent(in) :: face_val, old_face_val
    real, intent(in) :: theta, dt, udotn
    real, dimension(:,:), intent(in) :: x_ele
    real, dimension(:), intent(in) :: field_ele, old_field_ele
    real, intent(out), optional :: ftheta

    logical, intent(in) :: limit_theta

    real theta_val

    real :: gf, hdc, pinvth, qinvth, l_ftheta

    if(limit_theta) then
      gf = sign(max(abs(dt*udotn*(face_val-old_face_val)),tolerance), &
            dt*udotn*(face_val-old_face_val))
      hdc = sqrt(sum((x_ele(:,iloc)-x_ele(:,oloc))**2))
      pinvth = hdc*(field_ele(iloc)-old_field_ele(iloc))/gf
      qinvth = hdc*(field_ele(oloc)-old_field_ele(oloc))/gf
      l_ftheta = max(0.5, 1.-0.5*min(abs(pinvth),abs(qinvth)))
    else
      l_ftheta = theta
    end if

    theta_val = l_ftheta*face_val + (1.-l_ftheta)*old_face_val

    if(present(ftheta)) then
      ftheta = theta ! temporary hack to fix formulation
    end if

  end function

end module cv_face_values
