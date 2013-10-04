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
module cvtools
  !!< Module containing general tools for discretising Control Volume problems.
  use spud
  use state_module
  use fldebug
  use vector_tools
  use futils, only: int2str
  use field_options, only: complete_field_path
  use global_parameters, only: OPTION_PATH_LEN, FIELD_NAME_LEN

  implicit none

  interface clean_deferred_deletion
    module procedure clean_deferred_deletion_single_state, clean_deferred_deletion_multiple_states
  end interface

  private
  public :: orientate_cvsurf_normgi, &
            clean_deferred_deletion, &
            complete_cv_field_path

contains

  character(len=OPTION_PATH_LEN) function complete_cv_field_path(path)
    !!< Auxillary function to add  control_volumes/legacy_mixed_cv_cg/coupled_cv
    !!< to field option path.

    character(len=*), intent(in) :: path
    integer :: stat

    if (have_option(trim(complete_field_path(path, stat=stat)) // &
                 &"/spatial_discretisation/control_volumes")) then

       complete_cv_field_path=trim(complete_field_path(path, stat=stat)) // &
                 &"/spatial_discretisation/control_volumes"

    elseif (have_option(trim(complete_field_path(path, stat=stat)) // &
                 &"/spatial_discretisation/legacy_mixed_cv_cg")) then

       complete_cv_field_path=trim(complete_field_path(path, stat=stat)) // &
                 &"/spatial_discretisation/legacy_mixed_cv_cg"

    elseif (have_option(trim(complete_field_path(path, stat=stat)) // &
                 &"/spatial_discretisation/coupled_cv")) then

       complete_cv_field_path=trim(complete_field_path(path, stat=stat)) // &
                 &"/spatial_discretisation/coupled_cv"

    else

      complete_cv_field_path=path(1:len_trim(path))

    end if

  end function complete_cv_field_path

  function orientate_cvsurf_normgi(X, X_f, normgi) result(normgi_f)
    ! This subroutine corrects the orientation of a cv face normal
    ! relative to a set of nodal coordinates so that it points away from
    ! that node
    ! This is useful as there is no easy automatic orientation for cv face
    ! normals as they point in different directions when looking from different
    ! control volumes.

    ! unorientated normal
    real, dimension(:) :: normgi
    ! result
    real, dimension(size(normgi)) :: normgi_f
    ! node and gauss pt. locations respectively.
    real, dimension (:), intent(in) :: X, X_f
    ! Outward pointing not necessarily normal vector.
    real, dimension(3) :: outv

    integer :: ldim, i

    ldim = size(normgi)

    ! Outv is the vector from the gauss pt to the node.
    forall(i=1:ldim)
      outv(i) = X_f(i)-X(i)
    end forall

    ! Set correct orientation.
    normgi=normgi*dot_product(normgi, outv(1:ldim) )

    ! normalise
    normgi_f=normgi/sqrt(sum(normgi**2))

  end function orientate_cvsurf_normgi

  subroutine clean_deferred_deletion_single_state(state)
    type(state_type), intent(inout) :: state

    type(state_type), dimension(1) :: states

    states = (/state/)
    call clean_deferred_deletion(states)
    state = states(1)

  end subroutine clean_deferred_deletion_single_state

  subroutine clean_deferred_deletion_multiple_states(state)

    ! this subroutine cleans up state of any inserted control volume
    ! matrices whose deletion was deferred to speed up the subroutine

    type(state_type), dimension(:), intent(inout) :: state

    integer :: i, f, mesh, mesh_cnt, prefix, prefix_cnt, stat
    logical :: delete
    character(len=OPTION_PATH_LEN) :: option_path

    character(len=255), dimension(2), parameter :: &
        matrix_prefixes = (/ "         ", &
                             "Reflected" /)
    character(len=FIELD_NAME_LEN) :: mesh_name, mesh_name2

    mesh_cnt = option_count("/geometry/mesh")
    prefix_cnt = size(matrix_prefixes)

    do i = 1, size(state)

      do prefix = 1, prefix_cnt

        do mesh = 0, mesh_cnt-1

          call get_option("/geometry/mesh["//int2str(mesh)//"]/name", mesh_name)

          if (has_csr_matrix(state(i), &
              trim(matrix_prefixes(prefix))//trim(mesh_name)//"CVUpwindElements")) then

            delete = .true.
            field_loop: do f = 1, size(state(i)%scalar_fields)
              call get_option(trim(state(i)%scalar_fields(f)%ptr%option_path)//"/prognostic/mesh/name", mesh_name2, stat)

              if(stat==0) then
                if(trim(mesh_name2)==trim(mesh_name)) then

                  option_path = trim(complete_cv_field_path(state(i)%scalar_fields(f)%ptr%option_path))

                  if(have_option(trim(option_path)//"/face_value[0]/limit_face_value")) then
                    option_path = trim(option_path)//"/face_value[0]/limit_face_value/limiter[0]"
                  else
                    option_path = trim(option_path)//"/face_value[0]"
                  end if

                  if(have_option(trim(option_path)//"/project_upwind_from_gradient")) then
                    option_path = trim(option_path)//"/project_upwind_from_gradient/bound_projected_value_locally"
                  else
                    option_path = trim(option_path)//"/project_upwind_from_point"
                  end if

                  if(have_option(trim(option_path)//&
                                          "/store_upwind_elements")) then
                    if((trim(matrix_prefixes(prefix))=="Reflected")) then
                       if(have_option(trim(option_path)//&
                                          "/reflect_off_domain_boundaries")) then
                          delete = .false.
                          exit field_loop
                       end if
                    else
                       if(.not.have_option(trim(option_path)//&
                                          "/reflect_off_domain_boundaries")) then
                          delete = .false.
                          exit field_loop
                       end if
                    end if
                  end if
                end if
              end if
            end do field_loop

            if (delete) then

              call remove_csr_matrix(state(i), trim(matrix_prefixes(prefix))//trim(mesh_name)//"CVUpwindElements")

            end if

          end if

          if (has_block_csr_matrix(state(i), &
              trim(matrix_prefixes(prefix))//trim(mesh_name)//"CVUpwindQuadrature")) then
            delete = .true.
            field_loop2: do f = 1, size(state(i)%scalar_fields)

              call get_option(trim(state(i)%scalar_fields(f)%ptr%option_path)//"/prognostic/mesh/name", mesh_name2, stat)

              if(stat==0) then
                if(trim(mesh_name2)==trim(mesh_name)) then

                  option_path = trim(complete_cv_field_path(state(i)%scalar_fields(f)%ptr%option_path))

                  if(have_option(trim(option_path)//"/face_value[0]/limit_face_value")) then
                    option_path = trim(option_path)//"/face_value[0]/limit_face_value/limiter[0]"
                  else
                    option_path = trim(option_path)//"/face_value[0]"
                  end if

                  option_path = trim(option_path)//"/project_upwind_from_point"  ! not possible to store quadratures with gradient based method

                  if(have_option(trim(option_path)//&
                                          "/store_upwind_elements/store_upwind_quadrature")) then
                    if((trim(matrix_prefixes(prefix))=="Reflected")) then
                       if(have_option(trim(option_path)//&
                                          "/reflect_off_domain_boundaries")) then
                          delete = .false.
                          exit field_loop2
                       end if
                    else
                       if(.not.have_option(trim(option_path)//&
                                          "/reflect_off_domain_boundaries")) then
                          delete = .false.
                          exit field_loop2
                       end if
                    end if
                  end if
                end if
              end if
            end do field_loop2

            if (delete) then

              call remove_block_csr_matrix(state(i), trim(matrix_prefixes(prefix))//trim(mesh_name)//"CVUpwindQuadrature")

            end if

          end if

        end do

      end do

    end do

  end subroutine clean_deferred_deletion_multiple_states

end module cvtools
