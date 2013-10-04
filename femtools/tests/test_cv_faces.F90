subroutine test_cv_faces

  use cv_faces
  use elements
  use cv_shape_functions

  implicit none

  type(cv_faces_type) :: linear, quadratic, linear_tet
  type(element_type) :: cv_p1p1, cv_p1p2, cv_p2p1, cv_p2p2, cv_p1p1_tet
  type(element_type) :: cv_p1p1_bdy, cv_p1p2_bdy, cv_p2p1_bdy, cv_p2p2_bdy
  integer :: i, j

  linear = find_cv_faces(vertices=3, dimension=2, polydegree=1, quaddegree=1)
  linear_tet = find_cv_faces(vertices=4, dimension=3, polydegree=1, quaddegree=1)
  quadratic = find_cv_faces(vertices=3, dimension=2, polydegree=2, quaddegree=2)

  write(0,*) 'linear'
  do i = 1, size(linear%corners,1)
    write(0,*) 'face = ', i
    do j = 1, size(linear%corners,3)
      write(0,*) 'corner = ', j
      write(0,*) linear%corners(i,:,j)
    end do
    write(0,*) 'neiloc = ', linear%neiloc(:,i)
  end do
  do i = 1, size(linear%scorners,1)
    write(0,*) 'sface = ', i
    do j = 1, size(linear%scorners,3)
      write(0,*) 'scorner = ', j
      write(0,*) linear%scorners(i,:,j)
    end do
    write(0,*) 'sneiloc = ', linear%sneiloc(:,i)
  end do

  write(0,*) 'quadratic'
  do i = 1, size(quadratic%corners,1)
    write(0,*) 'face = ', i
    do j = 1, size(quadratic%corners,3)
      write(0,*) 'corner = ', j
      write(0,*) quadratic%corners(i,:,j)
    end do
    write(0,*) 'neiloc = ', quadratic%neiloc(:,i)
  end do
  do i = 1, size(quadratic%scorners,1)
    write(0,*) 'sface = ', i
    do j = 1, size(quadratic%scorners,3)
      write(0,*) 'scorner = ', j
      write(0,*) quadratic%scorners(i,:,j)
    end do
    write(0,*) 'sneiloc = ', quadratic%sneiloc(:,i)
  end do

  cv_p1p1=make_cv_element_shape(linear, 1)
  cv_p1p1_tet=make_cv_element_shape(linear_tet, 1)
  cv_p2p1=make_cv_element_shape(linear, 2)
  cv_p1p2=make_cv_element_shape(quadratic,1)
  cv_p2p2=make_cv_element_shape(quadratic,2)

  write(0,*) 'p1p1'
  do i = 1, size(cv_p1p1%n,2)
    write(0,*) 'gi = ', i
    write(0,*) 'cv_p1p1%n(:,gi) = ', cv_p1p1%n(:,i)
    do j = 1,size(cv_p1p1%dn,3)
      write(0,*) 'dim = ', j
      write(0,*) 'cv_p1p1%dn(:,gi,dim) = ', cv_p1p1%dn(:,i,j)
    end do
    write(0,*) 'cv_p1p1%quadrature%weight(gi) = ', cv_p1p1%quadrature%weight(i)
  end do

  write(0,*) 'p1p1_tet'
  do i = 1, size(cv_p1p1_tet%n,2)
    write(0,*) 'gi = ', i
    write(0,*) 'cv_p1p1_tet%n(:,gi) = ', cv_p1p1_tet%n(:,i)
    do j = 1,size(cv_p1p1_tet%dn,3)
      write(0,*) 'dim = ', j
      write(0,*) 'cv_p1p1_tet%dn(:,gi,dim) = ', cv_p1p1_tet%dn(:,i,j)
    end do
    write(0,*) 'cv_p1p1_tet%quadrature%weight(gi) = ', cv_p1p1_tet%quadrature%weight(i)
  end do

  write(0,*) 'p2p1'
  do i = 1, size(cv_p2p1%n,2)
    write(0,*) 'gi = ', i
    write(0,*) 'cv_p2p1%n(:,gi) = ', cv_p2p1%n(:,i)
    do j = 1,size(cv_p2p1%dn,3)
      write(0,*) 'dim = ', j
      write(0,*) 'cv_p2p1%dn(:,gi,dim) = ', cv_p2p1%dn(:,i,j)
    end do
  end do

  write(0,*) 'p1p2'
  do i = 1, size(cv_p1p2%n,2)
    write(0,*) 'gi = ', i
    write(0,*) 'cv_p1p2%n(:,gi) = ', cv_p1p2%n(:,i)
    do j = 1,size(cv_p1p2%dn,3)
      write(0,*) 'dim = ', j
      write(0,*) 'cv_p1p2%dn(:,gi,dim) = ', cv_p1p2%dn(:,i,j)
    end do
  end do

  write(0,*) 'p2p2'
  do i = 1, size(cv_p2p2%n,2)
    write(0,*) 'gi = ', i
    write(0,*) 'cv_p2p2%n(:,gi) = ', cv_p2p2%n(:,i)
    do j = 1,size(cv_p2p2%dn,3)
      write(0,*) 'dim = ', j
      write(0,*) 'cv_p2p2%dn(:,gi,dim) = ', cv_p2p2%dn(:,i,j)
    end do
  end do

  cv_p1p1_bdy=make_cvbdy_element_shape(linear, 1)
  cv_p2p1_bdy=make_cvbdy_element_shape(linear, 2)
  cv_p1p2_bdy=make_cvbdy_element_shape(quadratic,1)
  cv_p2p2_bdy=make_cvbdy_element_shape(quadratic,2)

  write(0,*) 'p1p1_bdy'
  do i = 1, size(cv_p1p1_bdy%n,2)
    write(0,*) 'gi = ', i
    write(0,*) 'cv_p1p1_bdy%n(:,gi) = ', cv_p1p1_bdy%n(:,i)
    do j = 1,size(cv_p1p1_bdy%dn,3)
      write(0,*) 'dim = ', j
      write(0,*) 'cv_p1p1_bdy%dn(:,gi,dim) = ', cv_p1p1_bdy%dn(:,i,j)
    end do
  end do

  write(0,*) 'p2p1_bdy'
  do i = 1, size(cv_p2p1_bdy%n,2)
    write(0,*) 'gi = ', i
    write(0,*) 'cv_p2p1_bdy%n(:,gi) = ', cv_p2p1_bdy%n(:,i)
    do j = 1,size(cv_p2p1_bdy%dn,3)
      write(0,*) 'dim = ', j
      write(0,*) 'cv_p2p1_bdy%dn(:,gi,dim) = ', cv_p2p1_bdy%dn(:,i,j)
    end do
  end do

  write(0,*) 'p1p2_bdy'
  do i = 1, size(cv_p1p2_bdy%n,2)
    write(0,*) 'gi = ', i
    write(0,*) 'cv_p1p2_bdy%n(:,gi) = ', cv_p1p2_bdy%n(:,i)
    do j = 1,size(cv_p1p2_bdy%dn,3)
      write(0,*) 'dim = ', j
      write(0,*) 'cv_p1p2_bdy%dn(:,gi,dim) = ', cv_p1p2_bdy%dn(:,i,j)
    end do
  end do

  write(0,*) 'p2p2_bdy'
  do i = 1, size(cv_p2p2_bdy%n,2)
    write(0,*) 'gi = ', i
    write(0,*) 'cv_p2p2_bdy%n(:,gi) = ', cv_p2p2_bdy%n(:,i)
    do j = 1,size(cv_p2p2_bdy%dn,3)
      write(0,*) 'dim = ', j
      write(0,*) 'cv_p2p2_bdy%dn(:,gi,dim) = ', cv_p2p2_bdy%dn(:,i,j)
    end do
  end do

  write(0,*) 'ending'

end subroutine test_cv_faces