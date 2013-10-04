#include "fdebug.h"

module node_boundary
!!< Does a node lie on a boundary? Surprisingly
!!< difficult question to answer.

  use surfacelabels
  use fields
  use linked_lists
  use eventcounter
  implicit none

  integer, dimension(:), pointer, save :: boundcount
  integer, save :: expected_boundcount
  logical, save :: boundcount_initialised = .false.
  integer, save :: pseudo2d_coord = 0

  interface node_boundary_count
    module procedure node_boundary_count_full, node_boundary_count_slim
  end interface

  interface node_lies_on_boundary
    module procedure node_lies_on_boundary_full, node_lies_on_boundary_slim
  end interface

  contains

  function one_to_n(n) result(arr)
    integer, intent(in) :: n
    integer, dimension(n) :: arr
    integer :: i

    do i=1,n
      arr(i) = i
    end do
  end function one_to_n

  function boundcount_is_initialised() result(init)
    logical :: init

    init = boundcount_initialised
  end function boundcount_is_initialised

  subroutine deallocate_boundcount
    if (associated(boundcount)) then
      deallocate(boundcount)
    end if

    boundcount_initialised = .false.
  end subroutine deallocate_boundcount

  subroutine initialise_boundcount(mesh, positions, out_boundcount)
    type(mesh_type), intent(inout) :: mesh
    type(vector_field), intent(in) :: positions
    integer, dimension(:), optional :: out_boundcount

    integer, save :: eventcount = 0
    integer :: latest_eventcount
    integer :: i, j
    integer, dimension(:), pointer :: surf_ids
    integer, dimension(:), pointer :: neighbours
    type(ilist), dimension(:), allocatable :: tags
    integer :: snloc, face
    real, dimension(mesh_dim(mesh)) :: dimlen
    integer, dimension(1) :: minloc_out

    integer :: count_zero_boundaries(1)
    type(element_type) :: element
    integer, dimension(:), allocatable :: face_glob_nod

    latest_eventcount = 0

    call GetEventCounter(EVENT_ADAPTIVITY, latest_eventcount)
    if (latest_eventcount > eventcount) then
      eventcount = latest_eventcount
      boundcount_initialised = .false.
      if (associated(boundcount)) then
        deallocate(boundcount)
      end if
    end if

    ! generate coplanar ids, if not already done
    call get_coplanar_ids(mesh, positions, surf_ids)

    if (boundcount_initialised .eqv. .false.) then
      element = ele_shape(mesh, 1)
      if (.not. has_faces(mesh)) then
        call add_faces(mesh)
      end if

      allocate(boundcount(node_count(mesh)))
      boundcount = 0
      boundcount_initialised = .true.

      allocate(tags(node_count(mesh)))
      snloc = face_loc(mesh, 1)
      allocate(face_glob_nod(snloc))

      do i=1,surface_element_count(mesh)
        face_glob_nod = face_global_nodes(mesh, i)
        do j=1,snloc
          call insert_ascending(tags(face_glob_nod(j)), surf_ids(i))
        end do
      end do

      do i=1,size(boundcount)
        boundcount(i) = tags(i)%length
      end do

      if (present(out_boundcount)) then
        out_boundcount = boundcount
      end if

      deallocate(face_glob_nod)

      if (allocated(tags)) then
        do i=1,size(tags)
          call deallocate(tags(i))
        end do
        deallocate(tags)
      end if

      expected_boundcount = 0
      if (mesh_dim(mesh) == 3 .and. domain_is_2d()) expected_boundcount = 1
      if (minval(boundcount) > 0) expected_boundcount = 1
      count_zero_boundaries = count(boundcount == 0)
      ! if only 20% of nodes are not on boundaries,
      ! assume it's pseudo2d
      if (mesh_dim(mesh) == 3) then
        if ((float(count_zero_boundaries(1)) / float(size(boundcount))) < 0.20) then
          expected_boundcount = 1
          if (pseudo2d_coord /= 0) then
            return
          else
            do i=1,mesh_dim(mesh)
              dimlen(i) = maxval(positions%val(i,:)) - minval(positions%val(i,:))
            end do
            minloc_out = minloc(dimlen)
            pseudo2d_coord = minloc_out(1)
          end if
          ewrite(1,*) 'WARNING: pseudo2D switched on'
        end if
      end if
    end if
  end subroutine initialise_boundcount

  function node_lies_on_boundary_full(mesh, positions, node, expected) result(on_bound)
    !!< Does the given node lie on the boundary?
    type(mesh_type), intent(inout) :: mesh
    type(vector_field), intent(in) :: positions
    integer, intent(in) :: node
    integer, intent(in), optional :: expected
    logical :: on_bound

    integer :: lexpected

    call initialise_boundcount(mesh, positions)

    if (present(expected)) then
      lexpected = expected
    else
      lexpected = expected_boundcount
    end if

    if (boundcount(node) > lexpected) then
      on_bound = .true.
    else
      on_bound = .false.
    end if

  end function node_lies_on_boundary_full

  function node_lies_on_boundary_slim(node, expected) result(on_bound)
    integer, intent(in) :: node
    integer, intent(in), optional :: expected
    integer :: lexpected
    logical :: on_bound

    if (.not. boundcount_is_initialised()) then
      FLAbort("You need to call initialise_boundcount before using this routine")
    end if

    if (present(expected)) then
      lexpected = expected
    else
      lexpected = expected_boundcount
    end if

    if (boundcount(node) > lexpected) then
      on_bound = .true.
    else
      on_bound = .false.
    end if

  end function node_lies_on_boundary_slim

  function node_boundary_count_full(mesh, positions, node) result(cnt)
    type(mesh_type), intent(inout) :: mesh
    type(vector_field), intent(in) :: positions
    integer, intent(in) :: node
    integer :: cnt

    call initialise_boundcount(mesh, positions)

    cnt = boundcount(node)
  end function node_boundary_count_full

  function node_boundary_count_slim(node) result(cnt)
    integer, intent(in) :: node
    integer :: cnt

    if (.not. boundcount_is_initialised()) then
      FLAbort("You need to have called initialise_boundcount before using this routine!")
    end if

    cnt = boundcount(node)
  end function node_boundary_count_slim

  function get_expected_boundcount() result(lexpected_boundcount)
    integer :: lexpected_boundcount

    lexpected_boundcount = expected_boundcount
  end function

  function domain_is_2d() result(bool)
    !!< Is the domain pseudo2d or not?
    logical :: bool

    bool = .false.
    if (pseudo2d_coord > 0 .and. pseudo2d_coord < 4) bool = .true.
  end function domain_is_2d

  function domain_is_2d_x() result(bool)
    !!< Is the domain pseudo2d in the x direction?
    logical :: bool
    bool = (pseudo2d_coord == 1)
  end function domain_is_2d_x

  function domain_is_2d_y() result(bool)
    !!< Is the domain pseudo2d in the y direction?
    logical :: bool
    bool = (pseudo2d_coord == 2)
  end function domain_is_2d_y

  function domain_is_2d_z() result(bool)
    !!< Is the domain pseudo2d in the z direction?
    logical :: bool
    bool = (pseudo2d_coord == 3)

  end function domain_is_2d_z

end module node_boundary
