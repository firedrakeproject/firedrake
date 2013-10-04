#define BUF_SIZE 150
#include "fdebug.h"

module tetrahedron_intersection_module

  use elements
  use vector_tools
  use fields_data_types
  use fields_base
  use fields_allocates
  use fields_manipulation
  use transform_elements
  implicit none

  type tet_type
    real, dimension(3, 4) :: V ! vertices of the tet
    integer, dimension(4) :: colours = -1 ! surface colours
  end type tet_type

  type plane_type
    real, dimension(3) :: normal
    real :: c
  end type plane_type

  type(tet_type), dimension(BUF_SIZE), save :: tet_array, tet_array_tmp
  integer :: tet_cnt = 0, tet_cnt_tmp = 0
  type(mesh_type), save :: intersection_mesh
  logical, save :: mesh_allocated = .false.

  public :: tet_type, plane_type, intersect_tets, get_planes, finalise_tet_intersector

  interface intersect_tets
    module procedure intersect_tets_dt
  end interface

  interface get_planes
    module procedure get_planes_tet, get_planes_hex
  end interface

  contains

  subroutine finalise_tet_intersector
    if (mesh_allocated) then
      call deallocate(intersection_mesh)
      mesh_allocated = .false.
    end if
  end subroutine finalise_tet_intersector

  subroutine intersect_tets_dt(tetA, planesB, shape, stat, output, surface_shape, surface_positions, surface_colours)
    type(tet_type), intent(in) :: tetA
    type(plane_type), dimension(:), intent(in) :: planesB
    type(element_type), intent(in) :: shape
    type(vector_field), intent(inout) :: output
    type(vector_field), intent(out), optional :: surface_positions
    type(scalar_field), intent(out), optional :: surface_colours
    type(element_type), intent(in), optional :: surface_shape
    integer :: ele
    integer, intent(out) :: stat

    integer :: i, j, k, l
    real :: vol
    real, dimension(3) :: vec_tmp
    integer, dimension(3) :: idx_tmp
    integer :: surface_eles, colour_tmp
    type(mesh_type) :: surface_mesh, pwc_surface_mesh

    if (present(surface_colours) .or. present(surface_positions) .or. present(surface_shape)) then
      assert(present(surface_positions))
      assert(present(surface_colours))
      assert(present(surface_shape))
    end if


    assert(shape%degree == 1)
    assert(cell_family(shape) == FAMILY_SIMPLEX)
    assert(shape%dim == 3)

    tet_cnt = 1
    tet_array(1) = tetA

    if (.not. mesh_allocated) then
      call allocate(intersection_mesh, BUF_SIZE * 4, BUF_SIZE, shape, name="IntersectionMesh")
      intersection_mesh%ndglno = (/ (i, i=1,BUF_SIZE*4) /)
      intersection_mesh%continuity = -1
      mesh_allocated = .true.
    end if

    do i=1,size(planesB)
      ! Clip the tet_array against the i'th plane
      tet_cnt_tmp = 0

      do j=1,tet_cnt
        call clip(planesB(i), tet_array(j))
      end do

      if (i /= size(planesB)) then
        tet_cnt = tet_cnt_tmp
        tet_array(1:tet_cnt) = tet_array_tmp(1:tet_cnt)
      else
        ! Copy the result if the volume is > epsilon
        tet_cnt = 0
        do j=1,tet_cnt_tmp
          vol = tet_volume(tet_array_tmp(j))
          if (vol < 0.0) then
            vec_tmp = tet_array_tmp(j)%V(:, 1)
            colour_tmp = tet_array_tmp(j)%colours(1)
            tet_array_tmp(j)%V(:, 1) = tet_array_tmp(j)%V(:, 2)
            tet_array_tmp(j)%colours(1) = tet_array_tmp(j)%colours(2)
            tet_array_tmp(j)%V(:, 2) = vec_tmp
            tet_array_tmp(j)%colours(2) = colour_tmp
            vol = -vol
          end if

          if (vol > epsilon(0.0)) then
            tet_cnt = tet_cnt + 1
            tet_array(tet_cnt) = tet_array_tmp(j)
          end if
        end do
      end if
    end do

    if (tet_cnt == 0) then
      stat=1
      return
    end if

    stat = 0
    intersection_mesh%nodes = tet_cnt*4
    intersection_mesh%elements = tet_cnt
    call allocate(output, 3, intersection_mesh, "IntersectionCoordinates")

    do ele=1,tet_cnt
      call set(output, ele_nodes(output, ele), tet_array(ele)%V)
    end do

    if (present(surface_positions)) then
      ! OK! Let's loop through all the tets we have and see which faces have positive
      ! colour. These are the ones we want to record in the mesh
      surface_eles = 0
      do ele=1,tet_cnt
        surface_eles = surface_eles + count(tet_array(ele)%colours > 0)
      end do

      call allocate(surface_mesh, surface_eles * 3, surface_eles, surface_shape, name="SurfaceMesh")
      surface_mesh%ndglno = (/ (i, i=1,surface_eles * 3) /)
      call allocate(surface_positions, 3, surface_mesh, "OutputSurfaceCoordinate")
      pwc_surface_mesh = piecewise_constant_mesh(surface_mesh, "PWCSurfaceMesh")
      call allocate(surface_colours, pwc_surface_mesh, "SurfaceColours")
      call deallocate(surface_mesh)
      call deallocate(pwc_surface_mesh)

      j = 1
      do ele=1,tet_cnt
        do i=1,4
          if (tet_array(ele)%colours(i) > 0) then

            ! In python, this is
            ! idx_tmp = [x for x in range(4) if x != i]
            ! Hopefully that will make it clearer
            k = 1
            do l=1,4
              if (l /= i) then
                idx_tmp(k) = l
                k = k + 1
              end if
            end do
            call set(surface_positions, ele_nodes(surface_positions, j), tet_array(ele)%V(:, idx_tmp))
            call set(surface_colours, j, float(tet_array(ele)%colours(i)))
            j = j + 1
          end if
        end do
      end do
    end if

  end subroutine intersect_tets_dt

  subroutine clip(plane, tet)
  ! Clip tet against the plane
  ! and append any output to tet_array_tmp.
    type(plane_type), intent(in) :: plane
    type(tet_type), intent(in) :: tet

    real, dimension(4) :: dists
    integer :: neg_cnt, pos_cnt, zer_cnt
    integer, dimension(4) :: neg_idx, pos_idx, zer_idx
    integer :: i

    real :: invdiff, w0, w1
    type(tet_type) :: tet_tmp

    ! Negative == inside
    ! Positive == outside

    neg_cnt = 0
    pos_cnt = 0
    zer_cnt = 0

    dists = distances_to_plane(plane, tet)
    do i=1,4
      if (abs(dists(i)) < epsilon(0.0)) then
        zer_cnt = zer_cnt + 1
        zer_idx(zer_cnt) = i
      else if (dists(i) < 0.0) then
        neg_cnt = neg_cnt + 1
        neg_idx(neg_cnt) = i
      else if (dists(i) > 0.0) then
        pos_cnt = pos_cnt + 1
        pos_idx(pos_cnt) = i
      end if
    end do

    if (neg_cnt == 0) then
      ! tet is completely on positive side of plane, full clip
      return
    end if

    if (pos_cnt == 0) then
      ! tet is completely on negative side of plane, no clip
      tet_cnt_tmp = tet_cnt_tmp + 1
      tet_array_tmp(tet_cnt_tmp) = tet
      return
    end if

    ! The tet is split by the plane, so we have more work to do.

    select case(pos_cnt)
    case(3)
      ! +++-
      tet_cnt_tmp = tet_cnt_tmp + 1
      tet_array_tmp(tet_cnt_tmp) = tet
      do i=1,pos_cnt
        invdiff = 1.0 / ( dists(pos_idx(i)) - dists(neg_idx(1)) )
        w0 = -dists(neg_idx(1)) * invdiff
        w1 =  dists(pos_idx(i)) * invdiff
        tet_array_tmp(tet_cnt_tmp)%V(:, pos_idx(i)) = &
           w0 * tet_array_tmp(tet_cnt_tmp)%V(:, pos_idx(i)) + &
           w1 * tet_array_tmp(tet_cnt_tmp)%V(:, neg_idx(1))
      end do
      ! The colours will have been inherited already; we just need to zero
      ! the one corresponding to the plane cut
      tet_array_tmp(tet_cnt_tmp)%colours(face_no(pos_idx(1), pos_idx(2), pos_idx(3))) = 0
    case(2)
      select case(neg_cnt)
      case(2)
        ! ++--
        do i=1,pos_cnt
          invdiff = 1.0 / ( dists(pos_idx(i)) - dists(neg_idx(1)) )
          w0 = -dists(neg_idx(1)) * invdiff
          w1 =  dists(pos_idx(i)) * invdiff
          tet_tmp%V(:, i) = w0 * tet%V(:, pos_idx(i)) + w1 * tet%V(:, neg_idx(1))
        end do
        do i=1,neg_cnt
          invdiff = 1.0 / ( dists(pos_idx(i)) - dists(neg_idx(2)) )
          w0 = -dists(neg_idx(2)) * invdiff
          w1 =  dists(pos_idx(i)) * invdiff
          tet_tmp%V(:, i+2) = w0 * tet%V(:, pos_idx(i)) + w1 * tet%V(:, neg_idx(2))
        end do

        tet_cnt_tmp = tet_cnt_tmp + 1
        tet_array_tmp(tet_cnt_tmp) = tet
        tet_array_tmp(tet_cnt_tmp)%V(:, pos_idx(1)) = tet_tmp%V(:, 3)
        tet_array_tmp(tet_cnt_tmp)%V(:, pos_idx(2)) = tet_tmp%V(:, 2)
        tet_array_tmp(tet_cnt_tmp)%colours(neg_idx(1)) = 0
        tet_array_tmp(tet_cnt_tmp)%colours(neg_idx(2)) = 0

        tet_cnt_tmp = tet_cnt_tmp + 1
        tet_array_tmp(tet_cnt_tmp)%V(:, 1) = tet%V(:, neg_idx(2))
        tet_array_tmp(tet_cnt_tmp)%colours(1) = 0
        tet_array_tmp(tet_cnt_tmp)%V(:, 2) = tet_tmp%V(:, 4)
        tet_array_tmp(tet_cnt_tmp)%colours(2) = 0
        tet_array_tmp(tet_cnt_tmp)%V(:, 3) = tet_tmp%V(:, 3)
        tet_array_tmp(tet_cnt_tmp)%colours(3) = tet%colours(pos_idx(1))
        tet_array_tmp(tet_cnt_tmp)%V(:, 4) = tet_tmp%V(:, 2)
        tet_array_tmp(tet_cnt_tmp)%colours(4) = tet%colours(neg_idx(1))

        tet_cnt_tmp = tet_cnt_tmp + 1
        tet_array_tmp(tet_cnt_tmp)%V(:, 1) = tet%V(:, neg_idx(1))
        tet_array_tmp(tet_cnt_tmp)%colours(1) = 0
        tet_array_tmp(tet_cnt_tmp)%V(:, 2) = tet_tmp%V(:, 1)
        tet_array_tmp(tet_cnt_tmp)%colours(2) = 0
        tet_array_tmp(tet_cnt_tmp)%V(:, 3) = tet_tmp%V(:, 2)
        tet_array_tmp(tet_cnt_tmp)%colours(3) = tet%colours(pos_idx(2))
        tet_array_tmp(tet_cnt_tmp)%V(:, 4) = tet_tmp%V(:, 3)
        tet_array_tmp(tet_cnt_tmp)%colours(4) = tet%colours(neg_idx(2))
      case(1)
        ! ++-0
        tet_cnt_tmp = tet_cnt_tmp + 1
        tet_array_tmp(tet_cnt_tmp) = tet
        do i=1,pos_cnt
          invdiff = 1.0 / ( dists(pos_idx(i)) - dists(neg_idx(1)) )
          w0 = -dists(neg_idx(1)) * invdiff
          w1 =  dists(pos_idx(i)) * invdiff
          tet_array_tmp(tet_cnt_tmp)%V(:, pos_idx(i)) = &
             w0 * tet_array_tmp(tet_cnt_tmp)%V(:, pos_idx(i)) + &
             w1 * tet_array_tmp(tet_cnt_tmp)%V(:, neg_idx(1))
        end do
        tet_array_tmp(tet_cnt_tmp)%colours(neg_idx(1)) = 0
      end select
    case(1)
      select case(neg_cnt)
      case(3)
        ! +---
        do i=1,neg_cnt
          invdiff = 1.0 / ( dists(pos_idx(1)) - dists(neg_idx(i)) )
          w0 = -dists(neg_idx(i)) * invdiff
          w1 =  dists(pos_idx(1)) * invdiff
          tet_tmp%V(:, i) = w0 * tet%V(:, pos_idx(1)) + w1 * tet%V(:, neg_idx(i))
        end do

        tet_cnt_tmp = tet_cnt_tmp + 1
        tet_array_tmp(tet_cnt_tmp) = tet
        tet_array_tmp(tet_cnt_tmp)%V(:, pos_idx(1)) = tet_tmp%V(:, 1)
        tet_array_tmp(tet_cnt_tmp)%colours(neg_idx(1)) = 0

        tet_cnt_tmp = tet_cnt_tmp + 1
        tet_array_tmp(tet_cnt_tmp)%V(:, 1) = tet_tmp%V(:, 1)
        tet_array_tmp(tet_cnt_tmp)%colours(1) = tet%colours(neg_idx(1))
        tet_array_tmp(tet_cnt_tmp)%V(:, 2) = tet%V(:, neg_idx(2))
        tet_array_tmp(tet_cnt_tmp)%colours(2) = 0
        tet_array_tmp(tet_cnt_tmp)%V(:, 3) = tet%V(:, neg_idx(3))
        tet_array_tmp(tet_cnt_tmp)%colours(3) = tet%colours(neg_idx(3))
        tet_array_tmp(tet_cnt_tmp)%V(:, 4) = tet_tmp%V(:, 2)
        tet_array_tmp(tet_cnt_tmp)%colours(4) = 0

        tet_cnt_tmp = tet_cnt_tmp + 1
        tet_array_tmp(tet_cnt_tmp)%V(:, 1) = tet%V(:, neg_idx(3))
        tet_array_tmp(tet_cnt_tmp)%colours(1) = 0
        tet_array_tmp(tet_cnt_tmp)%V(:, 2) = tet_tmp%V(:, 2)
        tet_array_tmp(tet_cnt_tmp)%colours(2) = tet%colours(neg_idx(2))
        tet_array_tmp(tet_cnt_tmp)%V(:, 3) = tet_tmp%V(:, 3)
        tet_array_tmp(tet_cnt_tmp)%colours(3) = 0
        tet_array_tmp(tet_cnt_tmp)%V(:, 4) = tet_tmp%V(:, 1)
        tet_array_tmp(tet_cnt_tmp)%colours(4) = tet%colours(neg_idx(1))
      case(2)
        ! +--0
        do i=1,neg_cnt
          invdiff = 1.0 / ( dists(pos_idx(1)) - dists(neg_idx(i)) )
          w0 = -dists(neg_idx(i)) * invdiff
          w1 =  dists(pos_idx(1)) * invdiff
          tet_tmp%V(:, i) = w0 * tet%V(:, pos_idx(1)) + w1 * tet%V(:, neg_idx(i))
        end do

        tet_cnt_tmp = tet_cnt_tmp + 1
        tet_array_tmp(tet_cnt_tmp) = tet
        tet_array_tmp(tet_cnt_tmp)%V(:, pos_idx(1)) = tet_tmp%V(:, 1)
        tet_array_tmp(tet_cnt_tmp)%colours(neg_idx(1)) = 0

        tet_cnt_tmp = tet_cnt_tmp + 1
        tet_array_tmp(tet_cnt_tmp)%V(:, 1) = tet_tmp%V(:, 2)
        tet_array_tmp(tet_cnt_tmp)%colours(1) = 0
        tet_array_tmp(tet_cnt_tmp)%V(:, 2) = tet%V(:, zer_idx(1))
        tet_array_tmp(tet_cnt_tmp)%colours(2) = tet%colours(zer_idx(1))
        tet_array_tmp(tet_cnt_tmp)%V(:, 3) = tet%V(:, neg_idx(2))
        tet_array_tmp(tet_cnt_tmp)%colours(3) = 0
        tet_array_tmp(tet_cnt_tmp)%V(:, 4) = tet_tmp%V(:, 1)
        tet_array_tmp(tet_cnt_tmp)%colours(4) = tet%colours(neg_idx(1))
      case(1)
        ! +-00
        invdiff = 1.0 / ( dists(pos_idx(1)) - dists(neg_idx(1)) )
        w0 = -dists(neg_idx(1)) * invdiff
        w1 =  dists(pos_idx(1)) * invdiff

        tet_cnt_tmp = tet_cnt_tmp + 1
        tet_array_tmp(tet_cnt_tmp) = tet
        tet_array_tmp(tet_cnt_tmp)%V(:, pos_idx(1)) = w0 * tet%V(:, pos_idx(1)) + w1 * tet%V(:, neg_idx(1))
        tet_array_tmp(tet_cnt_tmp)%colours(neg_idx(1)) = 0
      end select
    end select

  end subroutine clip

  pure function get_planes_tet(tet) result(plane)
    type(tet_type), intent(in) :: tet
    type(plane_type), dimension(4) :: plane

    real, dimension(3) :: edge10, edge20, edge30, edge21, edge31
    real :: det
    integer :: i

    edge10 = tet%V(:, 2) - tet%V(:, 1);
    edge20 = tet%V(:, 3) - tet%V(:, 1);
    edge30 = tet%V(:, 4) - tet%V(:, 1);
    edge21 = tet%V(:, 3) - tet%V(:, 2);
    edge31 = tet%V(:, 4) - tet%V(:, 2);

    plane(1)%normal = unit_cross(edge20, edge10)
    plane(2)%normal = unit_cross(edge10, edge30)
    plane(3)%normal = unit_cross(edge30, edge20)
    plane(4)%normal = unit_cross(edge21, edge31)

    det = dot_product(edge10, plane(4)%normal)
    if (det < 0) then
      do i=1,4
        plane(i)%normal = -plane(i)%normal
      end do
    end if

    ! And calibrate what is the zero of this plane by dotting with
    ! a point we know to be on it
    do i=1,4
      plane(i)%c = dot_product(tet%V(:, i), plane(i)%normal)
    end do

  end function get_planes_tet

  function get_planes_hex(positions, ele) result(plane)
    type(vector_field), intent(in) :: positions
    integer, intent(in) :: ele
    type(plane_type), dimension(6) :: plane
    integer, dimension(:), pointer :: faces
    integer :: i, face
    integer, dimension(4) :: fnodes
    real, dimension(positions%dim, face_ngi(positions, ele)) :: normals

    ! This could be done much more efficiently by exploiting
    ! more information about how we number faces and such on a hex

    assert(cell_family(positions%mesh%shape) == FAMILY_CUBE)
    assert(cell_family(positions%mesh%faces%shape) == FAMILY_CUBE)
    assert(positions%mesh%shape%degree == 1)
    assert(has_faces(positions%mesh))

    faces => ele_faces(positions, ele)
    assert(size(faces) == 6)

    do i=1,size(faces)
      face = faces(i)
      fnodes = face_global_nodes(positions, face)

      call transform_facet_to_physical(positions, face, normal=normals)
      plane(i)%normal = normals(:, 1)

      ! Now we calibrate the constant (setting the 'zero level' of the plane, as it were)
      ! with a node we know is on the face
      plane(i)%c = dot_product(plane(i)%normal, node_val(positions, fnodes(1)))

    end do
  end function get_planes_hex

  pure function unit_cross(vecA, vecB) result(cross)
    real, dimension(3), intent(in) :: vecA, vecB
    real, dimension(3) :: cross
    cross(1) = vecA(2) * vecB(3) - vecA(3) * vecB(2)
    cross(2) = vecA(3) * vecB(1) - vecA(1) * vecB(3)
    cross(3) = vecA(1) * vecB(2) - vecA(2) * vecB(1)

    cross = cross / norm2(cross)
  end function unit_cross

  pure function distances_to_plane(plane, tet) result(dists)
    type(plane_type), intent(in) :: plane
    type(tet_type), intent(in) :: tet
    real, dimension(4) :: dists
    integer :: i

    forall(i=1:4)
      dists(i) = dot_product(plane%normal, tet%V(:, i)) - plane%c
    end forall
  end function distances_to_plane

  pure function tet_volume(tet) result(vol)
    type(tet_type), intent(in) :: tet
    real :: vol
    real, dimension(3) :: cross, vecA, vecB, vecC

    vecA = tet%V(:, 1) - tet%V(:, 4)
    vecB = tet%V(:, 2) - tet%V(:, 4)
    vecC = tet%V(:, 3) - tet%V(:, 4)

    cross(1) = vecB(2) * vecC(3) - vecB(3) * vecC(2)
    cross(2) = vecB(3) * vecC(1) - vecB(1) * vecC(3)
    cross(3) = vecB(1) * vecC(2) - vecB(2) * vecC(1)

    vol = dot_product(vecA, cross) / 6.0
  end function tet_volume

  function face_no(i, j, k) result(face)
    ! Given three local node numbers, what is the face that they share?
    integer, intent(in) :: i, j, k
    integer :: face

    do face=1,4
      if (face /= i .and. face /= j .and. face /= k) return
    end do

  end function face_no

end module tetrahedron_intersection_module
