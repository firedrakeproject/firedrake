#include "fdebug.h"

module metric_tools

  use unittest_tools
  use fields_data_types
  use fields_base
  use fields_allocates
  use fields_manipulation
  use vector_tools
  use spud
  use fldebug
  implicit none

  interface edge_length_from_eigenvalue
    module procedure edge_length_from_eigenvalue_scalar, edge_length_from_eigenvalue_vector, &
                   & edge_length_from_eigenvalue_metric
  end interface

  interface eigenvalue_from_edge_length
    module procedure eigenvalue_from_edge_length_scalar, eigenvalue_from_edge_length_vector, &
                   & eigenvalue_from_edge_length_metric
  end interface

  interface aspect_ratio
    module procedure aspect_ratio_metric, aspect_ratio_eigenvalues
  end interface

  interface metric_isotropic
    module procedure metric_isotropic_metric, metric_isotropic_eigenvalues
  end interface

  interface metric_spheroid
    module procedure metric_spheroid_metric, metric_spheroid_eigenvalues
  end interface

  interface metric_ellipsoid
    module procedure metric_ellipsoid_metric, metric_ellipsoid_eigenvalues
  end interface

  interface get_adapt_opt
    module procedure get_adapt_opt_real_scalar, get_adapt_opt_real_vector
  end interface

  contains

  subroutine check_metric(metric)
    !!< This code checks if the metric has NaN's in it.
    type(tensor_field), intent(in) :: metric

    integer :: i, j, k

#ifdef DDEBUG

    do i=1,metric%mesh%nodes
      do j=1,metric%dim(1)
        do k=1,metric%dim(2)
          if (is_nan(metric%val(j, k, i))) then
            ewrite(-1,*) "Node == ", i, "; position (", j, ", ", k, ")"
            ewrite(-1,*) metric%val(:, :, i)
            FLAbort("Your metric has NaNs!")
          end if
        end do
      end do
      if (.not. mat_is_symmetric(metric%val(:, :, i))) then
        ewrite(-1,*) "Node == ", i
        ewrite(-1,*) metric%val(:, :, i)
        FLAbort("Your metric is not symmetric!")
      end if
    end do

#endif

  end subroutine check_metric

  subroutine check_basis(basis, stat)
    !!< This code checks if the matrix passed in represents
    !!< an orthonormal basis.
    real, dimension(:, :), intent(in) :: basis
    integer, intent(out), optional :: stat
    integer :: i, j
    real :: dot

    if (present(stat)) stat = 0

#ifdef DDEBUG

    do i=1,size(basis,2)
      do j=i+1,size(basis,2)
        dot = dot_product(basis(:, i), basis(:, j))
        if (dot .fne. 0.0) then
          if (.not. present(stat)) then
            call write_matrix(basis, "basis")
            FLAbort("basis is not orthogonal!")
          else
            stat = 1
          end if
        end if
      end do
    end do

    do i=1,size(basis,2)
      dot = dot_product(basis(:, i), basis(:, i))
      if (dot .fne. 1.0) then
        if (.not. present(stat)) then
          call write_matrix(basis, "basis")
          FLAbort("basis is not orthonormal!")
        else
          stat = 1
        end if
      end if
    end do

#endif

  end subroutine check_basis

  function edge_length_from_eigenvalue_scalar(evalue) result(edge_len)
    real, intent(in) :: evalue
    real :: edge_len

    if (evalue /= 0.0) then
       edge_len = 1.0/sqrt(abs(evalue))
    else
       edge_len = huge(evalue)
    end if

  end function edge_length_from_eigenvalue_scalar

  function edge_length_from_eigenvalue_vector(evalues) result(edge_lens)
    real, dimension(:), intent(in) :: evalues
    real, dimension(size(evalues)) :: edge_lens

    integer :: i

    do i=1,size(evalues)
      assert(evalues(i) /= 0.0)
      edge_lens(i) = 1.0/sqrt(abs(evalues(i)))
    end do
  end function edge_length_from_eigenvalue_vector

  function edge_length_from_eigenvalue_metric(metric) result(edge)
    real, dimension(:, :), intent(in) :: metric
    real, dimension(size(metric, 1), size(metric, 1)) :: edge, evecs
    real, dimension(size(metric, 1)) :: evals

    call eigendecomposition_symmetric(metric, evecs, evals)
    call eigenrecomposition(edge, evecs, edge_length_from_eigenvalue_vector(evals))
  end function edge_length_from_eigenvalue_metric

  function eigenvalue_from_edge_length_scalar(edge_len) result(evalue)
    real, intent(in) :: edge_len
    real :: evalue

    assert(edge_len /= 0.0)
    evalue = 1.0/(edge_len * edge_len)
  end function eigenvalue_from_edge_length_scalar

  function eigenvalue_from_edge_length_vector(edge_lens) result(evalues)
    real, dimension(:), intent(in) :: edge_lens
    real, dimension(size(edge_lens)) :: evalues

    integer :: i

    do i=1,size(edge_lens)
      if(edge_lens(i) == 0.0) then
        evalues(i) = huge(0.0)
      else
        evalues(i) = 1.0/(edge_lens(i) * edge_lens(i))
      end if
    end do
  end function eigenvalue_from_edge_length_vector

  function eigenvalue_from_edge_length_metric(edge) result(metric)
    real, dimension(:, :), intent(in) :: edge
    real, dimension(size(edge, 1), size(edge, 1)) :: metric, evecs
    real, dimension(size(edge, 1)) :: evals

    call eigendecomposition_symmetric(edge, evecs, evals)
    call eigenrecomposition(metric, evecs, eigenvalue_from_edge_length_vector(evals))
  end function eigenvalue_from_edge_length_metric

  function metric_isotropic_metric(metric) result(isotropic)
    !!< Is the metric isotropic, that is, all its eigenvalues are the same?
    real, dimension(:, :) :: metric

    real, dimension(size(metric,1), size(metric,1)) :: lvecs
    real, dimension(size(metric,1)) :: lvals

    integer :: i
    real :: maxv

    logical :: isotropic

    isotropic = .true.

    call eigendecomposition_symmetric(metric, lvecs, lvals)

    maxv = maxval(lvals)
    do i=1,size(metric,1)
      if (.not. fequals(lvals(i), maxv)) then
        isotropic = .false.
        return
      end if
    end do
  end function metric_isotropic_metric

  function metric_isotropic_eigenvalues(vals) result(isotropic)
    !!< Is the metric isotropic, that is, all its eigenvalues are the same?
    real, dimension(:) :: vals

    integer :: i
    real :: maxv

    logical :: isotropic

    isotropic = .true.

    maxv = maxval(vals)
    do i=1,size(vals)
      if (.not. fequals(vals(i), maxv)) then
        isotropic = .false.
        return
      end if
    end do
  end function metric_isotropic_eigenvalues

  function metric_spheroid_metric(metric) result(spheroid)
  !!< Is the metric a spheroid, that is, all but one eigenvalues the same?
    real, dimension(:, :) :: metric

    real, dimension(size(metric,1), size(metric,1)) :: lvecs
    real, dimension(size(metric,1)) :: lvals

    integer :: i, count
    real :: minv

    logical :: spheroid

    spheroid = .false.

    call eigendecomposition_symmetric(metric, lvecs, lvals)

    minv = minval(lvals)

    count = 0
    do i=1,size(metric,1)
      if (fequals(lvals(i), minv)) then
        count = count + 1
      end if
    end do

    if (count == (size(metric,1) - 1)) then
      spheroid = .true.
      return
    end if
  end function metric_spheroid_metric

  function metric_spheroid_eigenvalues(vals) result(spheroid)
  !!< Is the metric a spheroid, that is, all but one eigenvalues the same?
    real, dimension(:) :: vals

    real, dimension(size(vals)) :: lvals

    integer :: i, count
    real :: minv

    logical :: spheroid

    spheroid = .false.

    lvals = vals

    minv = minval(lvals)

    count = 0
    do i=1,size(vals)
      if (fequals(lvals(i), minv)) then
        count = count + 1
      end if
    end do

    if (count == (size(vals) - 1)) then
      spheroid = .true.
      return
    end if
  end function metric_spheroid_eigenvalues

  function metric_ellipsoid_metric(mat) result(ellipsoid)
    real, intent(in), dimension(:, :) :: mat
    logical :: ellipsoid

    ellipsoid = .false.
    if ((.not. metric_spheroid(mat)) .and. (.not. metric_isotropic(mat))) ellipsoid = .true.
  end function metric_ellipsoid_metric

  function metric_ellipsoid_eigenvalues(vals) result(ellipsoid)
    real, intent(in), dimension(:) :: vals
    logical :: ellipsoid

    ellipsoid = .false.
    if ((.not. metric_spheroid(vals)) .and. (.not. metric_isotropic(vals))) ellipsoid = .true.
  end function metric_ellipsoid_eigenvalues

  function get_spheroid_index(metric, vecs, vals) result(idx)
  !!< Is the metric a spheroid, that is, all but one eigenvalues the same?
    real, dimension(:, :) :: metric
    real, dimension(size(metric,1), size(metric,1)), optional :: vecs
    real, dimension(size(metric,1)), optional :: vals

    real, dimension(size(metric,1), size(metric,1)) :: lvecs
    real, dimension(size(metric,1)) :: lvals

    integer :: i, count
    real :: maxv, minv

    integer :: idx, fakeidx(1)

    idx = 0

    if (present(vecs) .and. present(vals)) then
      lvecs = vecs
      lvals = vals
    else
      call eigendecomposition_symmetric(metric, lvecs, lvals)
    end if

    maxv = maxval(lvals)
    minv = minval(lvals) ! have to try both

    count = 0
    do i=1,size(metric,1)
      if (fequals(lvals(i), maxv)) then
        count = count + 1
      end if
    end do

    if (count == (size(metric,1) - 1)) then
      fakeidx = maxloc(lvals)
      idx = fakeidx(1)
      return
    end if

    ! Wasn't maxval? Try again with minval.

    count = 0
    do i=1,size(metric,1)
      if (fequals(lvals(i), minv)) then
        count = count + 1
      end if
      end do

    if (count == (size(metric,1) - 1)) then
      fakeidx = minloc(lvals)
      idx = fakeidx(1)
      return
    end if
  end function get_spheroid_index

  function get_polar_index(vals) result(idx)
  !!< Is the metric a polar, that is, all but one eigenvalues the same?
    real, dimension(:) :: vals

    real, dimension(size(vals)) :: lvals

    integer :: i, count
    real :: maxv, minv

    integer :: idx, fakeidx(1)

    idx = 0

    lvals = vals

    maxv = maxval(lvals)
    minv = minval(lvals) ! have to try both

    count = 0
    do i=1,size(vals)
      if (fequals(lvals(i), maxv)) then
        count = count + 1
      end if
    end do

    if (count == (size(vals) - 1)) then
      fakeidx = minloc(lvals)
      idx = fakeidx(1)
      return
    end if

    ! Wasn't maxval? Try again with minval.

    count = 0
    do i=1,size(vals)
      if (fequals(lvals(i), minv)) then
        count = count + 1
      end if
      end do

    if (count == (size(vals) - 1)) then
      fakeidx = maxloc(lvals)
      idx = fakeidx(1)
      return
    end if
  end function get_polar_index

  function aspect_ratio_metric(metric) result(ratio)
  !!< Returns the aspect ratio: the largest edgelength over the smallest.
    real, dimension(:, :) :: metric

    real, dimension(size(metric,1), size(metric,1)) :: lvecs
    real, dimension(size(metric,1)) :: lvals

    real :: ratio

    call eigendecomposition_symmetric(metric, lvecs, lvals)
    assert( all(lvals >= 0) )

    if (minval(lvals) == 0) then
      ratio = huge(0.0)
    else
      ratio = sqrt(minval(lvals) / maxval(lvals))
    end if
  end function aspect_ratio_metric

  function aspect_ratio_eigenvalues(vals) result(ratio)
  !!< Returns the aspect ratio: the largest edgelength over the smallest.
    real, dimension(:) :: vals

    real :: ratio

    if (minval(vals) == 0) then
      ratio = huge(0.0)
    else
      ratio = sqrt(minval(vals) / maxval(vals))
    end if
  end function aspect_ratio_eigenvalues

  function norm(vec) result(r2norm)
    !!< R-2 norm. R2NORM has an unnecessarily long
    !!< interface (this doesn't need to scale to vectors distributed over CPUs).

    real, dimension(:), intent(in) :: vec
    real :: r2norm
    integer :: i

    r2norm = 0.0

    do i=1,size(vec)
      r2norm = r2norm + vec(i)**2
    end do

    r2norm = sqrt(r2norm)
  end function norm

  function get_angle(vecA, vecB) result(angle)
    !!< Return the angle between two vectors.
    !!< Computed with the dot product formula.
    !!< This just treats vectors in terms of direction,
    !!< i.e. vecA is considered the same as -vecA for my purposes here.
    !!< See also get_angle_2d, get_real_angle.
    real, dimension(:), intent(in) :: vecA, vecB
    real :: angle, pi

    pi = 4.0 * atan(1.0)

    if (vecA .feq. vecB) then
      angle = 0.0
      return
    end if

    if (vecA .feq. (-1 * vecB)) then
      angle = 0.0
      return
    end if

    angle = acos(dot_product(vecA, vecB)/ (norm(vecA) * norm(vecB)))
    if (angle > pi / 2.0) angle = pi - angle ! ignore sign, truncate to [0, Pi/2]
    if (is_nan(angle)) angle = 0.0
  end function get_angle

  function get_real_angle(vecA, vecB) result(angle)
    !!< Return the angle between two vectors.
    !!< Computed with the dot product formula.
    !!< See also get_angle_2d, get_angle.
    real, dimension(:), intent(in) :: vecA, vecB
    real :: angle

    if (vecA .feq. vecB) then
      angle = 0.0
      return
    end if

    angle = acos(dot_product(vecA, vecB)/ (norm(vecA) * norm(vecB)))
  end function get_real_angle

  function get_angle_2d(vecA, vecB) result(angle)
    !!< Return the angle between two vectors.
    !!< Computed with the arctan formula.
    real, dimension(:), intent(in) :: vecA, vecB
    real :: angle

    if (vecA .feq. vecB) then
      angle = 0.0
      return
    end if

    angle = atan2(vecB(2), vecB(1)) - atan2(vecA(2), vecA(1))
  end function get_angle_2d

  function get_rotation_matrix(A, B) result(mat)
    !!< Really, this should be done with an interface block.
    !!< But fortran is a stupid language! It doesn't work!
    real, dimension(:), intent(in) :: A, B
    real, dimension(size(A), size(A)) :: mat

    real, dimension(size(A)) :: normed_A, normed_B
    real :: norm_AplusB, norm_AminusB

    normed_A = A / norm(A)
    normed_B = B / norm(B)

    norm_AplusB  = norm(A + B)
    norm_AminusB = norm(A - B)

    !write(0,*) "norm_AminusB == ", norm_AminusB

    if (norm_AminusB < 1e-4) then
      mat = get_matrix_identity(size(A))
      return
    end if

    !write(0,*) "norm_AplusB == ", norm_AplusB

    if (norm_AplusB < 1e-4) then
      mat = -1 * get_matrix_identity(size(A))
      return
    end if

    if (size(A) == 2) mat = get_rotation_matrix_2d(normed_A, normed_B)
    if (size(A) == 3) mat = get_rotation_matrix_3d(normed_A, normed_B)
    call mat_clean(mat, epsilon(0.0))
  end function get_rotation_matrix

  function get_rotation_matrix_cross(vec, angle) result(mat)
    !!< Given a vector as the axis of rotation and the angle,
    !!< return the rotation matrix.
    real, dimension(3), intent(in) :: vec
    real, intent(in) :: angle
    real, dimension(3, 3) :: mat
    real, dimension(3) :: cross
    real :: x, y, z
    real :: c, s

    if (abs(angle) < 0.01) then
      mat = get_matrix_identity(3)
      if (angle < 0) mat = -1 * mat
      return
    end if

    c = cos(angle) ; s = sin(angle)

    cross = vec / norm(vec)

    x = cross(1) ; y = cross(2) ; z = cross(3)

    mat(1, 1) = c + (1 - c) * x * x
    mat(2, 2) = c + (1 - c) * y * y
    mat(3, 3) = c + (1 - c) * z * z

    mat(1, 2) = (1 - c) * x * y - s * z
    mat(1, 3) = (1 - c) * x * z + s * y

    mat(2, 1) = (1 - c) * y * x + s * z
    mat(2, 3) = (1 - c) * y * z - s * x

    mat(3, 1) = (1 - c) * z * x - s * y
    mat(3, 2) = (1 - c) * z * y + s * x
  end function get_rotation_matrix_cross

  function get_rotation_matrix_2d(A, B) result(mat)
    !!< Return the rotation matrix that would map A -> B.
    real, dimension(2), intent(in) :: A, B

    real :: angle
    real, dimension(2, 2) :: mat

    if (A .feq. B) then
      mat = get_matrix_identity(2)
      return
    end if

    if (A .feq. (-1 * B)) then
      mat = -1 * get_matrix_identity(2)
      return
    end if

    angle = get_angle_2d(A, B)
    mat(1, 1) = cos(angle) ; mat(1, 2) = -1 * sin(angle)
    mat(2, 1) = sin(angle) ; mat(2, 2) = cos(angle)
  end function get_rotation_matrix_2d

  function get_rotation_matrix_3d(A, B) result(mat)
    !!< Return the rotation matrix that would map A -> B.
    real, dimension(3), intent(in) :: A, B

    real :: angle, c, s, x, y, z
    real, dimension(3, 3) :: mat
    real, dimension(3) :: cross

    if (A .feq. B) then
      mat = get_matrix_identity(3)
      return
    end if

    if (A .feq. (-1 * B)) then
      mat = -1 * get_matrix_identity(3)
      return
    end if

    angle = get_real_angle(A, B)
    c = cos(angle) ; s = sin(angle)

    cross = cross_product(A, B)
    cross = cross / norm(cross)

    x = cross(1) ; y = cross(2) ; z = cross(3)

    mat(1, 1) = c + (1 - c) * x * x
    mat(2, 2) = c + (1 - c) * y * y
    mat(3, 3) = c + (1 - c) * z * z

    mat(1, 2) = (1 - c) * x * y - s * z
    mat(1, 3) = (1 - c) * x * z + s * y

    mat(2, 1) = (1 - c) * y * x + s * z
    mat(2, 3) = (1 - c) * y * z - s * x

    mat(3, 1) = (1 - c) * z * x - s * y
    mat(3, 2) = (1 - c) * z * y + s * x

  end function get_rotation_matrix_3d

  function project_to_subspace(vec, basis) result(proj)
    !!< Project the vector vec onto the subspace spanned
    !!< by the basis vectors basis.
    real, dimension(:), intent(in) :: vec
    real, dimension(:, :), intent(in) :: basis

    real, dimension(size(vec)) :: proj

    real, dimension(size(vec), size(vec)) :: proj_operator

    integer :: dim

    dim = size(vec)
    assert(size(basis, 1) == dim)

    proj_operator = matmul(basis, transpose(basis))
    proj = matmul(proj_operator, vec)

  end function project_to_subspace

  function dominant_eigenvector(vecs, vals) result(vec)
    !!< Return the dominant eigenvector (the
    !!< eigenvector corresponding to the largest eigenvalue).
    real, dimension(:, :), intent(in) :: vecs
    real, dimension(size(vecs, 1)), intent(in) :: vals
    real, dimension(size(vecs, 1)) :: vec
    integer :: i(1)

    i = maxloc(vals)

    vec = vecs(:, i(1))
  end function dominant_eigenvector

  function midval(vals) result(mid)
    !!< Return the middle eigenvalue, the one that's
    !!< not the max or the min.
    real, dimension(:), intent(in) :: vals
    real :: mid

    real :: maxv

    maxv = maxval(vals)
    mid = maxval(vals, mask=(vals /= maxv))
  end function midval

  subroutine get_node_field(mesh, field)
    !!< Return a field containing each node number.
    type(mesh_type), intent(in) :: mesh
    type(scalar_field), intent(inout) :: field
    integer :: i

    do i=1,mesh%nodes
      field%val(i) = float(i)
    end do
  end subroutine get_node_field

  function domain_length_scale(positions) result(scale)
    !!< Return the domain length scale.
    type(vector_field), intent(in) :: positions
    real :: scale
    real, dimension(positions%dim) :: domainwidth
    integer :: i

    do i=1,positions%dim
      domainwidth(i) = maxval(positions%val(i,:)) - minval(positions%val(i,:))
    end do

    scale = maxval(domainwidth)
  end function domain_length_scale

  subroutine check_perm(perm, stat)
    !!< Check if perm represents a permutation.
    integer, dimension(:), intent(in) :: perm
    integer, optional :: stat
#ifdef DDEBUG
    integer :: dim
#endif

    if (present(stat)) stat = 0

#ifdef DDEBUG
    dim = size(perm)
    if ((dim * (dim + 1)) / 2 /= sum(perm)) then
      if (present(stat)) then
        stat = 1
      else
        write(0,*) "perm == ", perm
        FLAbort("Not a permutation!")
      end if
    end if

    if (factorial(dim) /= product(perm)) then
      if (present(stat)) then
        stat = 1
      else
        write(0,*) "perm == ", perm
        FLAbort("Not a permutation!")
      end if
    end if
#endif
  end subroutine check_perm

  function factorial(n) result(fact)
    integer, intent(in) :: n
    integer :: fact
    integer :: i

    fact = 1
    do i=2,n
      fact = fact * i
    end do
  end function factorial

  function have_adapt_opt(path, ext)
    character(len=*), intent(in) :: path, ext
    logical :: have_adapt_opt
    have_adapt_opt = (have_option((path) // "/virtual" // (ext)) .or. &
                   &  have_option((path) // "/prescribed" // (ext)) .or. &
                   &  have_option((path) // "/prognostic" // (ext)) .or. &
                   &  have_option((path) // "/diagnostic" // (ext)))
  end function have_adapt_opt

  subroutine get_adapt_opt_real_scalar(path, ext, var)
    character(len=*), intent(in) :: path, ext
    real, intent(out) :: var
    integer :: stat

    call get_option(path // "/virtual" // ext, var, stat)
    if (stat == 0) return
    call get_option(path // "/prescribed" // ext, var, stat)
    if (stat == 0) return
    call get_option(path // "/prognostic" // ext, var, stat)
    if (stat == 0) return
    call get_option(path // "/diagnostic" // ext, var, stat)
    if (stat == 0) return

    ewrite(-1,*) "path == ", path
    ewrite(-1,*) "ext == ", ext
    FLAbort("no such variable")
  end subroutine get_adapt_opt_real_scalar

  subroutine get_adapt_opt_real_vector(path, ext, var)
    character(len=*), intent(in) :: path, ext
    real, dimension(:), intent(out) :: var
    integer :: stat

    call get_option(path // "/virtual" // ext, var, stat)
    if (stat == 0) return
    call get_option(path // "/prescribed" // ext, var, stat)
    if (stat == 0) return
    call get_option(path // "/prognostic" // ext, var, stat)
    if (stat == 0) return
    call get_option(path // "/diagnostic" // ext, var, stat)
    if (stat == 0) return

    ewrite(-1,*) "path == ", path
    ewrite(-1,*) "ext == ", ext
    !call print_children(path)
    FLAbort("no such variable")
  end subroutine get_adapt_opt_real_vector

  function error_bound_name(dep) result(ret)
    character(len=*), intent(in) :: dep
    character(len=len_trim(dep) + len("InterpolationErrorBound")) :: ret

    integer :: idx
    idx = index(dep, "%")
    if (idx == 0) then
      ret = dep // "InterpolationErrorBound"
    else
      ret = dep(1:idx-1) // "InterpolationErrorBound%" // dep(idx+1:len_trim(dep))
    end if
  end function error_bound_name

  function metric_from_edge_lengths(edgelen) result(metric)
    real, dimension(:, :), intent(in) :: edgelen
    real, dimension(size(edgelen, 1), size(edgelen, 1)) :: metric, evecs
    real, dimension(size(edgelen, 1)) :: evals

    call eigendecomposition_symmetric(edgelen, evecs, evals)
    evals = eigenvalue_from_edge_length(evals)
    call eigenrecomposition(metric, evecs, evals)
  end function metric_from_edge_lengths

  function edge_lengths_from_metric(metric) result(edgelen)
    real, dimension(:, :), intent(in) :: metric
    real, dimension(size(metric, 1), size(metric, 1)) :: edgelen, evecs
    real, dimension(size(metric, 1)) :: evals
    call eigendecomposition_symmetric(metric, evecs, evals)
    evals = edge_length_from_eigenvalue(evals)
    call eigenrecomposition(edgelen, evecs, evals)
  end function edge_lengths_from_metric

  function simplex_tensor(positions, ele, power) result(m)
    !!< Compute the metric tensor
    !!< associated with the element.
    !!< Note: this only works for linear position
    !!< (otherwise you have to do everything in integral form,
    !!< and I couldn't be bothered)

    !!< Note well: the units of the returned tensor
    !!< are L^-2 (if no power argument is given)

    type(vector_field), intent(in) :: positions
    integer, intent(in) :: ele
    real, intent(in), optional :: power

    real, dimension(positions%dim, positions%dim) :: m, evecs
    real, dimension(dof(positions%dim), dof(positions%dim)) :: A
    real, dimension(dof(positions%dim)) :: x
    real, dimension(positions%dim, ele_loc(positions, ele)) :: pos_ele
    real, dimension(positions%dim) :: diff, evals

    integer :: loc, i, j, k, l, p, n, dim, d

    loc = ele_loc(positions, ele)
    pos_ele = ele_val(positions, ele)
    dim = positions%dim
    d = dof(dim)

    ! Assemble
    n = 1
    do i=1,loc
      do j=i+1,loc
        diff = pos_ele(:, j) - pos_ele(:, i)
        do k=1,dim
          do l=1,dim
            p = idx(k, l, dim)
            A(n, p) = diff(k) * diff(l) * coeff(k, l)
          end do
        end do

        n = n + 1
      end do
   end do

    x = 1

    call solve(A, x)

    do i=1,dim
      do j=1,dim
        m(i, j) = x(idx(i, j, dim))
      end do
    end do

    if (present(power)) then
      call eigendecomposition_symmetric(m, evecs, evals)
      evals = evals**power
      call eigenrecomposition(m, evecs, evals)
    end if

    contains

    function idx(i, j, dim)
      integer, intent(in) :: i, j, dim
      integer :: idx, k, l

      k = min(i, j)
      l = max(i, j)

      if (k == 1) then
        idx = l
      else
        if (dim == 3) then
          idx = l + k
        else
          idx = l + k - 1
        end if
      end if
    end function idx

    function coeff(k, l)
      integer, intent(in) :: k, l
      integer :: coeff
      if (k == l) then
        coeff = 1
      else
        coeff = 2
      end if
    end function coeff

  end function simplex_tensor

  pure function dof(n)
    integer, intent(in) :: n
    integer :: dof

    dof = (n * (n+1)) / 2
  end function dof

  function apply_transform(pos, metric) result(new_pos)
    !! Given a metric and the positions of some points,
    !! map the points to their image under the transformation
    !! given by metric.
    real, dimension(:, :), intent(in) :: pos
    real, dimension(:, :), intent(in) :: metric
    real, dimension(size(pos, 1), size(pos, 2)) :: new_pos
    integer :: loc, i

    loc = size(pos, 2)
    do i=1,loc
      new_pos(:, i) = matmul(metric, pos(:, i))
    end do
  end function apply_transform

  subroutine form_anisotropic_metric_from_isotropic_metric(isotropic_metric, anisotropic_metric)
    type(scalar_field), intent(in) :: isotropic_metric
    type(tensor_field), intent(out) :: anisotropic_metric

    integer :: i
    real, dimension(mesh_dim(isotropic_metric)) :: eigenvals
    real, dimension(mesh_dim(isotropic_metric), mesh_dim(isotropic_metric)) :: eigenvecs, tensor

    assert(anisotropic_metric%dim(1)==anisotropic_metric%dim(2))

    call allocate(anisotropic_metric, isotropic_metric%mesh, isotropic_metric%name)

    eigenvecs = get_matrix_identity(anisotropic_metric%dim(1))

    call zero(anisotropic_metric)
    do i = 1, node_count(isotropic_metric)
      eigenvals = node_val(isotropic_metric, i)
      call eigenrecomposition(tensor, eigenvecs, eigenvals)
      call set(anisotropic_metric, i, tensor)
    end do

  end subroutine form_anisotropic_metric_from_isotropic_metric

  function absolutify_tensor(tens) result(absolute_tens)
    !! Given a tensor, map all its eigenvalues to their absolute values.
    real, dimension(:, :), intent(in) :: tens
    real, dimension(size(tens, 1), size(tens, 2)) :: absolute_tens, evecs
    real, dimension(size(tens, 1)) :: evals

    call eigendecomposition_symmetric(tens, evecs, evals)
    evals = abs(evals)
    call eigenrecomposition(absolute_tens, evecs, evals)
  end function absolutify_tensor

  function lipnikov_functional(ele, positions, metric) result(func)
    !!< Evaluate the Lipnikov functional for the supplied element

    integer, intent(in) :: ele
    type(vector_field), intent(in) :: positions
    type(tensor_field), intent(in) :: metric

    real :: func

    assert(cell_family(positions, ele) == FAMILY_SIMPLEX)

    select case(positions%dim)
      case(2)
        assert(ele_loc(positions, ele) == 3)
        func = lipnikov_functional_2d(ele, positions, metric)
      case(3)
        assert(ele_loc(positions, ele) == 4)
        func = lipnikov_functional_3d(ele, positions, metric)
      case default
        FLExit("The Lipnikov functional is only available in 2 or 3d.")
    end select

  end function lipnikov_functional

  function lipnikov_functional_2d(ele, positions, metric) result(func)
    !!< Evaluate the Lipnikov functional for the supplied 2d triangle. See:
    !!<   Yu. V. Vasileskii and K. N. Lipnikov, An Adaptive Algorithm for
    !!<   Quasioptimal Mesh Generation, Computational Mathematics and
    !!<   Mathematical Physics, Vol. 39, No. 9, 1999, pp. 1468 - 1486

    integer, intent(in) :: ele
    type(vector_field), intent(in) :: positions
    type(tensor_field), intent(in) :: metric

    real :: func

    integer, dimension(:), pointer :: element_nodes
    real :: edge_sum, vol
    real, dimension(2) :: tmp_pos
    real, dimension(2, 2) :: pos
    real, dimension(2, 2) :: m

    real :: scale_factor = 12.0 * sqrt(3.0)

    m = sum(ele_val(metric, ele), 3) / 3.0

    element_nodes => ele_nodes(positions, ele)
    tmp_pos = node_val(positions, element_nodes(1))
    pos(:, 1) = node_val(positions, element_nodes(2)) - tmp_pos
    pos(:, 2) = node_val(positions, element_nodes(3)) - tmp_pos

    edge_sum = metric_edge_length(pos(:, 1), m) + &
             & metric_edge_length(pos(:, 2), m) + &
             & metric_edge_length(pos(:, 2) - pos(:, 1), m)
    vol = abs(sqrt(det(m)) * det(pos) / 2.0)

    func = (scale_factor * vol / (edge_sum ** 2)) * F(edge_sum / 3.0)

  contains

    pure function F(x)
      real, intent(in) :: x

      real :: F

      real :: x1

      x1 = min(x, 1.0 / x)
      F = x1 * (2.0 - x1)
      F = F ** 3

    end function F

  end function lipnikov_functional_2d

  function lipnikov_functional_3d(ele, positions, metric) result(func)
    !!< Evaluate the Lipnikov functional for the supplied 3d tetrahedron. See:
    !!<   A. Agouzal, K Lipnikov, Yu. Vassilevski, Adaptive generation of
    !!<   quasi-optimal tetrahedral meshes, East-West J. Numer. Math., Vol. 7,
    !!<   No. 4, pp. 223-244 (1999)

    integer, intent(in) :: ele
    type(vector_field), intent(in) :: positions
    type(tensor_field), intent(in) :: metric

    real :: func

    integer, dimension(:), pointer :: element_nodes
    real :: edge_sum, vol
    real, dimension(3) :: tmp_pos
    real, dimension(3, 3) :: pos
    real, dimension(3, 3) :: m

    real :: scale_factor = (6.0 ** 4) * sqrt(2.0)

    m = sum(ele_val(metric, ele), 3) / 4.0

    element_nodes => ele_nodes(positions, ele)
    tmp_pos = node_val(positions, element_nodes(1))
    pos(:, 1) = node_val(positions, element_nodes(2)) - tmp_pos
    pos(:, 2) = node_val(positions, element_nodes(3)) - tmp_pos
    pos(:, 3) = node_val(positions, element_nodes(4)) - tmp_pos

    edge_sum = metric_edge_length(pos(:, 1), m) + &
             & metric_edge_length(pos(:, 2), m) + &
             & metric_edge_length(pos(:, 3), m) + &
             & metric_edge_length(pos(:, 3) - pos(:, 1), m) + &
             & metric_edge_length(pos(:, 3) - pos(:, 2), m) + &
             & metric_edge_length(pos(:, 2) - pos(:, 1), m)
    vol = abs(sqrt(det(m)) * det(pos) / 6.0)

    func = (scale_factor * vol / (edge_sum ** 3)) * F(edge_sum / 6.0)

  contains

    pure function F(x)
      real, intent(in) :: x

      real :: F

      real :: x1

      x1 = min(x, 1.0 / x)
      F = x1 * (2.0 - x1)
      F = F ** 3

    end function F

  end function lipnikov_functional_3d

  pure function metric_edge_length(x, m) result(length)
    !!< Return the length of an edge x in a metric space with metric m

    real, dimension(:), intent(in) :: x
    real, dimension(size(x), size(x)), intent(in) :: m

    real :: length

    length = sqrt(dot_product(x, matmul(m, x)))

  end function metric_edge_length

  subroutine element_quality_p0(positions, metric, quality)
    type(vector_field), intent(in) :: positions
    type(tensor_field), intent(in) :: metric
    type(scalar_field), intent(out) :: quality
    type(mesh_type) :: pwc_mesh
    integer :: ele

    pwc_mesh = piecewise_constant_mesh(positions%mesh, "PWCMesh")
    call allocate(quality, pwc_mesh, "ElementQuality")
    call deallocate(pwc_mesh)

    do ele=1,ele_count(positions)
      call set(quality, ele, lipnikov_functional(ele, positions, metric))
    end do
  end subroutine element_quality_p0

end module metric_tools
