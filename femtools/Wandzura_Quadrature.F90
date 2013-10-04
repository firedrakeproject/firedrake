module wandzura_quadrature

  use global_parameters, only : real_4, real_8

  implicit none

  interface wandzura_rule
    module procedure wandzura_rule_sp, wandzura_rule_orig
  end interface wandzura_rule

  contains
  function i4_wrap ( ival, ilo, ihi )

  !*****************************************************************************80
  !
  !! I4_WRAP forces an I4 to lie between given limits by wrapping.
  !
  !  Example:
  !
  !    ILO = 4, IHI = 8
  !
  !    I  Value
  !
  !    -2     8
  !    -1     4
  !     0     5
  !     1     6
  !     2     7
  !     3     8
  !     4     4
  !     5     5
  !     6     6
  !     7     7
  !     8     8
  !     9     4
  !    10     5
  !    11     6
  !    12     7
  !    13     8
  !    14     4
  !
  !  Licensing:
  !
  !    This code is distributed under the GNU LGPL license.
  !
  !  Modified:
  !
  !    19 August 2003
  !
  !  Author:
  !
  !    John Burkardt
  !
  !  Parameters:
  !
  !    Input, integer ( kind = 4 ) IVAL, an integer value.
  !
  !    Input, integer ( kind = 4 ) ILO, IHI, the desired bounds for the integer value.
  !
  !    Output, integer ( kind = 4 ) I4_WRAP, a "wrapped" version of IVAL.
  !
    implicit none

    integer ( kind = 4 ) i4_wrap
    integer ( kind = 4 ) ihi
    integer ( kind = 4 ) ilo
    integer ( kind = 4 ) ival
    integer ( kind = 4 ) jhi
    integer ( kind = 4 ) jlo
    integer ( kind = 4 ) value
    integer ( kind = 4 ) wide

    jlo = min ( ilo, ihi )
    jhi = max ( ilo, ihi )

    wide = jhi - jlo + 1

    if ( wide == 1 ) then
      value = jlo
    else
      value = jlo + i4_modp ( ival - jlo, wide )
    end if

    i4_wrap = value

    return
  end function
  subroutine file_name_inc ( file_name )

  !*****************************************************************************80
  !
  !! FILE_NAME_INC increments a partially numeric filename.
  !
  !  Discussion:
  !
  !    It is assumed that the digits in the name, whether scattered or
  !    connected, represent a number that is to be increased by 1 on
  !    each call.  If this number is all 9's on input, the output number
  !    is all 0's.  Non-numeric letters of the name are unaffected.
  !
  !    If the name is empty, then the routine stops.
  !
  !    If the name contains no digits, the empty string is returned.
  !
  !  Example:
  !
  !      Input            Output
  !      -----            ------
  !      'a7to11.txt'     'a7to12.txt'
  !      'a7to99.txt'     'a8to00.txt'
  !      'a9to99.txt'     'a0to00.txt'
  !      'cat.txt'        ' '
  !      ' '              STOP!
  !
  !  Licensing:
  !
  !    This code is distributed under the GNU LGPL license.
  !
  !  Modified:
  !
  !    14 September 2005
  !
  !  Author:
  !
  !    John Burkardt
  !
  !  Parameters:
  !
  !    Input/output, character ( len = * ) FILE_NAME.
  !    On input, a character string to be incremented.
  !    On output, the incremented string.
  !
    implicit none

    character c
    integer ( kind = 4 ) change
    integer ( kind = 4 ) digit
    character ( len = * ) file_name
    integer ( kind = 4 ) i
    integer ( kind = 4 ) lens

    lens = len_trim ( file_name )

    if ( lens <= 0 ) then
      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) 'FILE_NAME_INC - Fatal error!'
      write ( *, '(a)' ) '  The input string is empty.'
      stop
    end if

    change = 0

    do i = lens, 1, -1

      c = file_name(i:i)

      if ( lge ( c, '0' ) .and. lle ( c, '9' ) ) then

        change = change + 1

        digit = ichar ( c ) - 48
        digit = digit + 1

        if ( digit == 10 ) then
          digit = 0
        end if

        c = char ( digit + 48 )

        file_name(i:i) = c

        if ( c /= '0' ) then
          return
        end if

      end if

    end do

    if ( change == 0 ) then
      file_name = ' '
      return
    end if

    return
  end subroutine
  subroutine get_unit ( iunit )

  !*****************************************************************************80
  !
  !! GET_UNIT returns a free FORTRAN unit number.
  !
  !  Discussion:
  !
  !    A "free" FORTRAN unit number is an integer between 1 and 99 which
  !    is not currently associated with an I/O device.  A free FORTRAN unit
  !    number is needed in order to open a file with the OPEN command.
  !
  !    If IUNIT = 0, then no free FORTRAN unit could be found, although
  !    all 99 units were checked (except for units 5, 6 and 9, which
  !    are commonly reserved for console I/O).
  !
  !    Otherwise, IUNIT is an integer between 1 and 99, representing a
  !    free FORTRAN unit.  Note that GET_UNIT assumes that units 5 and 6
  !    are special, and will never return those values.
  !
  !  Licensing:
  !
  !    This code is distributed under the GNU LGPL license.
  !
  !  Modified:
  !
  !    18 September 2005
  !
  !  Author:
  !
  !    John Burkardt
  !
  !  Parameters:
  !
  !    Output, integer ( kind = 4 ) IUNIT, the free unit number.
  !
    implicit none

    integer ( kind = 4 ) i
    integer ( kind = 4 ) ios
    integer ( kind = 4 ) iunit
    logical lopen

    iunit = 0

    do i = 1, 99

      if ( i /= 5 .and. i /= 6 .and. i /= 9 ) then

        inquire ( unit = i, opened = lopen, iostat = ios )

        if ( ios == 0 ) then
          if ( .not. lopen ) then
            iunit = i
            return
          end if
        end if

      end if

    end do

    return
  end subroutine
  function i4_modp ( i, j )

  !*****************************************************************************80
  !
  !! I4_MODP returns the nonnegative remainder of I4 division.
  !
  !  Discussion:
  !
  !    If
  !      NREM = I4_MODP ( I, J )
  !      NMULT = ( I - NREM ) / J
  !    then
  !      I = J * NMULT + NREM
  !    where NREM is always nonnegative.
  !
  !    The MOD function computes a result with the same sign as the
  !    quantity being divided.  Thus, suppose you had an angle A,
  !    and you wanted to ensure that it was between 0 and 360.
  !    Then mod(A,360) would do, if A was positive, but if A
  !    was negative, your result would be between -360 and 0.
  !
  !    On the other hand, I4_MODP(A,360) is between 0 and 360, always.
  !
  !  Example:
  !
  !        I     J     MOD I4_MODP    Factorization
  !
  !      107    50       7       7    107 =  2 *  50 + 7
  !      107   -50       7       7    107 = -2 * -50 + 7
  !     -107    50      -7      43   -107 = -3 *  50 + 43
  !     -107   -50      -7      43   -107 =  3 * -50 + 43
  !
  !  Licensing:
  !
  !    This code is distributed under the GNU LGPL license.
  !
  !  Modified:
  !
  !    02 March 1999
  !
  !  Author:
  !
  !    John Burkardt
  !
  !  Parameters:
  !
  !    Input, integer ( kind = 4 ) I, the number to be divided.
  !
  !    Input, integer ( kind = 4 ) J, the number that divides I.
  !
  !    Output, integer ( kind = 4 ) I4_MODP, the nonnegative remainder when I is
  !    divided by J.
  !
    implicit none

    integer ( kind = 4 ) i
    integer ( kind = 4 ) i4_modp
    integer ( kind = 4 ) j
    integer ( kind = 4 ) value

    if ( j == 0 ) then
      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) 'I4_MODP - Fatal error!'
      write ( *, '(a,i8)' ) '  Illegal divisor J = ', j
      stop
    end if

    value = mod ( i, j )

    if ( value < 0 ) then
      value = value + abs ( j )
    end if

    i4_modp = value

    return
  end function
  subroutine reference_to_physical_t3 ( node_xy, n, ref, phy )

  !*****************************************************************************80
  !
  !! REFERENCE_TO_PHYSICAL_T3 maps T3 reference points to physical points.
  !
  !  Discussion:
  !
  !    Given the vertices of an order 3 physical triangle and a point
  !    (XSI,ETA) in the reference triangle, the routine computes the value
  !    of the corresponding image point (X,Y) in physical space.
  !
  !    This routine is also appropriate for an order 4 triangle,
  !    as long as the fourth node is the centroid of the triangle.
  !
  !    This routine may also be appropriate for an order 6
  !    triangle, if the mapping between reference and physical space
  !    is linear.  This implies, in particular, that the sides of the
  !    image triangle are straight and that the "midside" nodes in the
  !    physical triangle are literally halfway along the sides of
  !    the physical triangle.
  !
  !  Reference Element T3:
  !
  !    |
  !    1  3
  !    |  |\
  !    |  | \
  !    S  |  \
  !    |  |   \
  !    |  |    \
  !    0  1-----2
  !    |
  !    +--0--R--1-->
  !
  !  Licensing:
  !
  !    This code is distributed under the GNU LGPL license.
  !
  !  Modified:
  !
  !    10 May 2007
  !
  !  Author:
  !
  !    John Burkardt
  !
  !  Parameters:
  !
  !    Input, real ( kind = 8 ) NODE_XY(2,3), the coordinates of the vertices.
  !    The vertices are assumed to be the images of (0,0), (1,0) and
  !    (0,1) respectively.
  !
  !    Input, integer ( kind = 4 ) N, the number of objects to transform.
  !
  !    Input, real ( kind = 8 ) REF(2,N), points in the reference triangle.
  !
  !    Output, real ( kind = 8 ) PHY(2,N), corresponding points in the
  !    physical triangle.
  !
    implicit none

    integer ( kind = 4 ) n

    integer ( kind = 4 ) i
    real    ( kind = 8 ) node_xy(2,3)
    real    ( kind = 8 ) phy(2,n)
    real    ( kind = 8 ) ref(2,n)

    do i = 1, 2
      phy(i,1:n) = node_xy(i,1) * ( 1.0D+00 - ref(1,1:n) - ref(2,1:n) ) &
                 + node_xy(i,2) *             ref(1,1:n)                &
                 + node_xy(i,3) *                          ref(2,1:n)
    end do

    return
  end subroutine
  subroutine timestamp ( )

  !*****************************************************************************80
  !
  !! TIMESTAMP prints the current YMDHMS date as a time stamp.
  !
  !  Example:
  !
  !    31 May 2001   9:45:54.872 AM
  !
  !  Licensing:
  !
  !    This code is distributed under the GNU LGPL license.
  !
  !  Modified:
  !
  !    06 August 2005
  !
  !  Author:
  !
  !    John Burkardt
  !
  !  Parameters:
  !
  !    None
  !
    implicit none

    character ( len = 8 ) ampm
    integer ( kind = 4 ) d
    integer ( kind = 4 ) h
    integer ( kind = 4 ) m
    integer ( kind = 4 ) mm
    character ( len = 9 ), parameter, dimension(12) :: month = (/ &
      'January  ', 'February ', 'March    ', 'April    ', &
      'May      ', 'June     ', 'July     ', 'August   ', &
      'September', 'October  ', 'November ', 'December ' /)
    integer ( kind = 4 ) n
    integer ( kind = 4 ) s
    integer ( kind = 4 ) values(8)
    integer ( kind = 4 ) y

    call date_and_time ( values = values )

    y = values(1)
    m = values(2)
    d = values(3)
    h = values(5)
    n = values(6)
    s = values(7)
    mm = values(8)

    if ( h < 12 ) then
      ampm = 'AM'
    else if ( h == 12 ) then
      if ( n == 0 .and. s == 0 ) then
        ampm = 'Noon'
      else
        ampm = 'PM'
      end if
    else
      h = h - 12
      if ( h < 12 ) then
        ampm = 'PM'
      else if ( h == 12 ) then
        if ( n == 0 .and. s == 0 ) then
          ampm = 'Midnight'
        else
          ampm = 'AM'
        end if
      end if
    end if

    write ( *, '(i2,1x,a,1x,i4,2x,i2,a1,i2.2,a1,i2.2,a1,i3.3,1x,a)' ) &
      d, trim ( month(m) ), y, h, ':', n, ':', s, '.', mm, trim ( ampm )

    return
  end subroutine
  subroutine timestring ( string )

  !*****************************************************************************80
  !
  !! TIMESTRING writes the current YMDHMS date into a string.
  !
  !  Example:
  !
  !    STRING = '31 May 2001   9:45:54.872 AM'
  !
  !  Licensing:
  !
  !    This code is distributed under the GNU LGPL license.
  !
  !  Modified:
  !
  !    06 August 2005
  !
  !  Author:
  !
  !    John Burkardt
  !
  !  Parameters:
  !
  !    Output, character ( len = * ) STRING, the date information.
  !    A character length of 40 should always be sufficient.
  !
    implicit none

    character ( len = 8 ) ampm
    integer ( kind = 4 ) d
    integer ( kind = 4 ) h
    integer ( kind = 4 ) m
    integer ( kind = 4 ) mm
    character ( len = 9 ), parameter, dimension(12) :: month = (/ &
      'January  ', 'February ', 'March    ', 'April    ', &
      'May      ', 'June     ', 'July     ', 'August   ', &
      'September', 'October  ', 'November ', 'December ' /)
    integer ( kind = 4 ) n
    integer ( kind = 4 ) s
    character ( len = * ) string
    integer ( kind = 4 ) values(8)
    integer ( kind = 4 ) y

    call date_and_time ( values = values )

    y = values(1)
    m = values(2)
    d = values(3)
    h = values(5)
    n = values(6)
    s = values(7)
    mm = values(8)

    if ( h < 12 ) then
      ampm = 'AM'
    else if ( h == 12 ) then
      if ( n == 0 .and. s == 0 ) then
        ampm = 'Noon'
      else
        ampm = 'PM'
      end if
    else
      h = h - 12
      if ( h < 12 ) then
        ampm = 'PM'
      else if ( h == 12 ) then
        if ( n == 0 .and. s == 0 ) then
          ampm = 'Midnight'
        else
          ampm = 'AM'
        end if
      end if
    end if

    write ( string, '(i2,1x,a,1x,i4,2x,i2,a1,i2.2,a1,i2.2,a1,i3.3,1x,a)' ) &
      d, trim ( month(m) ), y, h, ':', n, ':', s, '.', mm, trim ( ampm )

    return
  end subroutine
  subroutine triangle_area ( node_xy, area )

  !*****************************************************************************80
  !
  !! TRIANGLE_AREA computes the area of a triangle.
  !
  !  Discussion:
  !
  !    If the triangle's vertices are given in counterclockwise order,
  !    the area will be positive.  If the triangle's vertices are given
  !    in clockwise order, the area will be negative!
  !
  !    If you cannot guarantee counterclockwise order, and you need to
  !    have the area positive, then you can simply take the absolute value
  !    of the result of this routine.
  !
  !    An earlier version of this routine always returned the absolute
  !    value of the computed area.  I am convinced now that that is
  !    a less useful result!  For instance, by returning the signed
  !    area of a triangle, it is possible to easily compute the area
  !    of a nonconvex polygon as the sum of the (possibly negative)
  !    areas of triangles formed by node 1 and successive pairs of vertices.
  !
  !  Licensing:
  !
  !    This code is distributed under the GNU LGPL license.
  !
  !  Modified:
  !
  !    17 October 2005
  !
  !  Author:
  !
  !    John Burkardt
  !
  !  Parameters:
  !
  !    Input, real ( kind = 8 ) NODE_XY(2,3), the triangle vertices.
  !
  !    Output, real ( kind = 8 ) AREA, the area of the triangle.
  !
    implicit none

    real    ( kind = 8 ) area
    real    ( kind = 8 ) node_xy(2,3)

    area = 0.5D+00 * ( &
        node_xy(1,1) * ( node_xy(2,2) - node_xy(2,3) ) &
      + node_xy(1,2) * ( node_xy(2,3) - node_xy(2,1) ) &
      + node_xy(1,3) * ( node_xy(2,1) - node_xy(2,2) ) )

    return
  end subroutine
  subroutine triangle_points_plot ( file_name, node_xy, node_show, point_num, &
    point_xy, point_show )

  !*****************************************************************************80
  !
  !! TRIANGLE_POINTS_PLOT plots a triangle and some points.
  !
  !  Licensing:
  !
  !    This code is distributed under the GNU LGPL license.
  !
  !  Modified:
  !
  !    03 October 2006
  !
  !  Author:
  !
  !    John Burkardt
  !
  !  Parameters:
  !
  !    Input, character ( len = * ) FILE_NAME, the name of the output file.
  !
  !    Input, real ( kind = 8 ) NODE_XY(2,3), the coordinates of the nodes
  !    of the triangle.
  !
  !    Input, integer ( kind = 4 ) NODE_SHOW,
  !   -1, do not show the triangle, or the nodes.
  !    0, show the triangle, do not show the nodes;
  !    1, show the triangle and the nodes;
  !    2, show the triangle, the nodes and number them.
  !
  !    Input, integer ( kind = 4 ) POINT_NUM, the number of points.
  !
  !    Input, real ( kind = 8 ) POINT_XY(2,POINT_NUM), the coordinates of the
  !    points.
  !
  !    Input, integer ( kind = 4 ) POINT_SHOW,
  !    0, do not show the points;
  !    1, show the points;
  !    2, show the points and number them.
  !
    implicit none

    integer ( kind = 4 ), parameter :: node_num = 3
    integer ( kind = 4 ) point_num

    character ( len = 40 ) date_time
    integer ( kind = 4 ) :: circle_size
    integer ( kind = 4 ) delta
    character ( len = * ) file_name
    integer ( kind = 4 ) file_unit
    integer ( kind = 4 ) i
    integer ( kind = 4 ), parameter :: i4_1 = 1
    integer ( kind = 4 ), parameter :: i4_3 = 3
    integer ( kind = 4 ) ios
    integer ( kind = 4 ) node
    integer ( kind = 4 ) node_show
    real    ( kind = 8 ) node_xy(2,node_num)
    integer ( kind = 4 ) point
    integer ( kind = 4 ) point_show
    real    ( kind = 8 ) point_xy(2,point_num)
    character ( len = 40 ) string
    real    ( kind = 8 ) x_max
    real    ( kind = 8 ) x_min
    integer ( kind = 4 ) x_ps
    integer ( kind = 4 ) :: x_ps_max = 576
    integer ( kind = 4 ) :: x_ps_max_clip = 594
    integer ( kind = 4 ) :: x_ps_min = 36
    integer ( kind = 4 ) :: x_ps_min_clip = 18
    real    ( kind = 8 ) x_scale
    real    ( kind = 8 ) y_max
    real    ( kind = 8 ) y_min
    integer ( kind = 4 ) y_ps
    integer ( kind = 4 ) :: y_ps_max = 666
    integer ( kind = 4 ) :: y_ps_max_clip = 684
    integer ( kind = 4 ) :: y_ps_min = 126
    integer ( kind = 4 ) :: y_ps_min_clip = 108
    real    ( kind = 8 ) y_scale

    call timestring ( date_time )
  !
  !  We need to do some figuring here, so that we can determine
  !  the range of the data, and hence the height and width
  !  of the piece of paper.
  !
    x_max = max ( maxval ( node_xy(1,1:node_num) ), &
                  maxval ( point_xy(1,1:point_num) ) )
    x_min = min ( minval ( node_xy(1,1:node_num) ), &
                  minval ( point_xy(1,1:point_num) ) )
    x_scale = x_max - x_min

    x_max = x_max + 0.05D+00 * x_scale
    x_min = x_min - 0.05D+00 * x_scale
    x_scale = x_max - x_min

    y_max = max ( maxval ( node_xy(2,1:node_num) ), &
                  maxval ( point_xy(2,1:point_num) ) )
    y_min = min ( minval ( node_xy(2,1:node_num) ), &
                  minval ( point_xy(2,1:point_num) ) )
    y_scale = y_max - y_min

    y_max = y_max + 0.05D+00 * y_scale
    y_min = y_min - 0.05D+00 * y_scale
    y_scale = y_max - y_min

    if ( x_scale < y_scale ) then

      delta = nint ( real ( x_ps_max - x_ps_min, kind = 8 ) &
        * ( y_scale - x_scale ) / ( 2.0D+00 * y_scale ) )

      x_ps_max = x_ps_max - delta
      x_ps_min = x_ps_min + delta

      x_ps_max_clip = x_ps_max_clip - delta
      x_ps_min_clip = x_ps_min_clip + delta

      x_scale = y_scale

    else if ( y_scale < x_scale ) then

      delta = nint ( real ( y_ps_max - y_ps_min, kind = 8 ) &
        * ( x_scale - y_scale ) / ( 2.0D+00 * x_scale ) )

      y_ps_max      = y_ps_max - delta
      y_ps_min      = y_ps_min + delta

      y_ps_max_clip = y_ps_max_clip - delta
      y_ps_min_clip = y_ps_min_clip + delta

      y_scale = x_scale

    end if

    call get_unit ( file_unit )

    open ( unit = file_unit, file = file_name, status = 'replace', &
      iostat = ios )

    if ( ios /= 0 ) then
      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) 'TRIANGLE_POINTS_PLOT - Fatal error!'
      write ( *, '(a)' ) '  Can not open output file.'
      return
    end if

    write ( file_unit, '(a)' ) '%!PS-Adobe-3.0 EPSF-3.0'
    write ( file_unit, '(a)' ) '%%Creator: triangulation_order3_plot.f90'
    write ( file_unit, '(a)' ) '%%Title: ' // trim ( file_name )
    write ( file_unit, '(a)' ) '%%CreationDate: ' // trim ( date_time )
    write ( file_unit, '(a)' ) '%%Pages: 1'
    write ( file_unit, '(a,i3,2x,i3,2x,i3,2x,i3)' ) '%%BoundingBox: ', &
      x_ps_min, y_ps_min, x_ps_max, y_ps_max
    write ( file_unit, '(a)' ) '%%Document-Fonts: Times-Roman'
    write ( file_unit, '(a)' ) '%%LanguageLevel: 1'
    write ( file_unit, '(a)' ) '%%EndComments'
    write ( file_unit, '(a)' ) '%%BeginProlog'
    write ( file_unit, '(a)' ) '/inch {72 mul} def'
    write ( file_unit, '(a)' ) '%%EndProlog'
    write ( file_unit, '(a)' ) '%%Page: 1 1'
    write ( file_unit, '(a)' ) 'save'
    write ( file_unit, '(a)' ) '%'
    write ( file_unit, '(a)' ) '%  Set the RGB line color to very light gray.'
    write ( file_unit, '(a)' ) '%'
    write ( file_unit, '(a)' ) '0.900  0.900  0.900 setrgbcolor'
    write ( file_unit, '(a)' ) '%'
    write ( file_unit, '(a)' ) '%  Draw a gray border around the page.'
    write ( file_unit, '(a)' ) '%'
    write ( file_unit, '(a)' ) 'newpath'
    write ( file_unit, '(a,i3,2x,i3,2x,a)' ) '  ', x_ps_min, y_ps_min, ' moveto'
    write ( file_unit, '(a,i3,2x,i3,2x,a)' ) '  ', x_ps_max, y_ps_min, ' lineto'
    write ( file_unit, '(a,i3,2x,i3,2x,a)' ) '  ', x_ps_max, y_ps_max, ' lineto'
    write ( file_unit, '(a,i3,2x,i3,2x,a)' ) '  ', x_ps_min, y_ps_max, ' lineto'
    write ( file_unit, '(a,i3,2x,i3,2x,a)' ) '  ', x_ps_min, y_ps_min, ' lineto'
    write ( file_unit, '(a)' ) 'stroke'
    write ( file_unit, '(a)' ) '%'
    write ( file_unit, '(a)' ) '%  Set the RGB color to black.'
    write ( file_unit, '(a)' ) '%'
    write ( file_unit, '(a)' ) '0.000  0.000  0.000 setrgbcolor'
    write ( file_unit, '(a)' ) '%'
    write ( file_unit, '(a)' ) '%  Set the font and its size.'
    write ( file_unit, '(a)' ) '%'
    write ( file_unit, '(a)' ) '/Times-Roman findfont'
    write ( file_unit, '(a)' ) '0.50 inch scalefont'
    write ( file_unit, '(a)' ) 'setfont'
    write ( file_unit, '(a)' ) '%'
    write ( file_unit, '(a)' ) '%  Print a title.'
    write ( file_unit, '(a)' ) '%'
    write ( file_unit, '(a)' ) '%  210  702  moveto'
    write ( file_unit, '(a)' ) '%  (Triangulation)  show'
    write ( file_unit, '(a)' ) '%'
    write ( file_unit, '(a)' ) '%  Define a clipping polygon.'
    write ( file_unit, '(a)' ) '%'
    write ( file_unit, '(a)' ) 'newpath'
    write ( file_unit, '(a,i3,2x,i3,2x,a)' ) '  ', &
      x_ps_min_clip, y_ps_min_clip, ' moveto'
    write ( file_unit, '(a,i3,2x,i3,2x,a)' ) '  ', &
      x_ps_max_clip, y_ps_min_clip, ' lineto'
    write ( file_unit, '(a,i3,2x,i3,2x,a)' ) '  ', &
      x_ps_max_clip, y_ps_max_clip, ' lineto'
    write ( file_unit, '(a,i3,2x,i3,2x,a)' ) '  ', &
      x_ps_min_clip, y_ps_max_clip, ' lineto'
    write ( file_unit, '(a,i3,2x,i3,2x,a)' ) '  ', &
      x_ps_min_clip, y_ps_min_clip, ' lineto'
    write ( file_unit, '(a)' ) 'clip newpath'
  !
  !  Draw the nodes.
  !
    if ( 1 <= node_show ) then

      circle_size = 5

      write ( file_unit, '(a)' ) '%'
      write ( file_unit, '(a)' ) '%  Draw filled dots at the nodes.'
      write ( file_unit, '(a)' ) '%'
      write ( file_unit, '(a)' ) '%  Set the RGB color to blue.'
      write ( file_unit, '(a)' ) '%'
      write ( file_unit, '(a)' ) '0.000  0.150  0.750 setrgbcolor'
      write ( file_unit, '(a)' ) '%'

      do node = 1, 3

        x_ps = int ( &
          ( ( x_max - node_xy(1,node)         ) * real ( x_ps_min, kind = 8 )   &
          + (         node_xy(1,node) - x_min ) * real ( x_ps_max, kind = 8 ) ) &
          / ( x_max                   - x_min ) )

        y_ps = int ( &
          ( ( y_max - node_xy(2,node)         ) * real ( y_ps_min, kind = 8 )   &
          + (         node_xy(2,node) - y_min ) * real ( y_ps_max, kind = 8 ) ) &
          / ( y_max                   - y_min ) )

        write ( file_unit, '(a,i4,2x,i4,2x,i4,2x,a)' ) 'newpath ', x_ps, y_ps, &
          circle_size, '0 360 arc closepath fill'

      end do

    end if
  !
  !  Label the nodes.
  !
    if ( 2 <= node_show ) then

      write ( file_unit, '(a)' ) '%'
      write ( file_unit, '(a)' ) '%  Label the nodes:'
      write ( file_unit, '(a)' ) '%'
      write ( file_unit, '(a)' ) '%  Set the RGB color to darker blue.'
      write ( file_unit, '(a)' ) '%'
      write ( file_unit, '(a)' ) '0.000  0.250  0.850 setrgbcolor'
      write ( file_unit, '(a)' ) '/Times-Roman findfont'
      write ( file_unit, '(a)' ) '0.20 inch scalefont'
      write ( file_unit, '(a)' ) 'setfont'
      write ( file_unit, '(a)' ) '%'

      do node = 1, node_num

        x_ps = int ( &
          ( ( x_max - node_xy(1,node)         ) * real ( x_ps_min, kind = 8 )   &
          + (       + node_xy(1,node) - x_min ) * real ( x_ps_max, kind = 8 ) ) &
          / ( x_max                   - x_min ) )

        y_ps = int ( &
          ( ( y_max - node_xy(2,node)         ) * real ( y_ps_min, kind = 8 )   &
          + (         node_xy(2,node) - y_min ) * real ( y_ps_max, kind = 8 ) ) &
          / ( y_max                   - y_min ) )

        write ( string, '(i4)' ) node
        string = adjustl ( string )

        write ( file_unit, '(i4,2x,i4,a)' ) x_ps, y_ps+5, &
          ' moveto (' // trim ( string ) // ') show'

      end do

    end if
  !
  !  Draw the points.
  !
    if ( point_num <= 200 ) then
      circle_size = 5
    else if ( point_num <= 500 ) then
      circle_size = 4
    else if ( point_num <= 1000 ) then
      circle_size = 3
    else if ( point_num <= 5000 ) then
      circle_size = 2
    else
      circle_size = 1
    end if

    if ( 1 <= point_show ) then
      write ( file_unit, '(a)' ) '%'
      write ( file_unit, '(a)' ) '%  Draw filled dots at the points.'
      write ( file_unit, '(a)' ) '%'
      write ( file_unit, '(a)' ) '%  Set the RGB color to green.'
      write ( file_unit, '(a)' ) '%'
      write ( file_unit, '(a)' ) '0.150  0.750  0.000 setrgbcolor'
      write ( file_unit, '(a)' ) '%'

      do point = 1, point_num

        x_ps = int ( &
          ( ( x_max - point_xy(1,point)         ) * real ( x_ps_min, kind = 8 )   &
          + (         point_xy(1,point) - x_min ) * real ( x_ps_max, kind = 8 ) ) &
          / ( x_max                     - x_min ) )

        y_ps = int ( &
          ( ( y_max - point_xy(2,point)         ) * real ( y_ps_min, kind = 8 )   &
          + (         point_xy(2,point) - y_min ) * real ( y_ps_max, kind = 8 ) ) &
          / ( y_max                     - y_min ) )

        write ( file_unit, '(a,i4,2x,i4,2x,i4,2x,a)' ) 'newpath ', x_ps, y_ps, &
          circle_size, '0 360 arc closepath fill'

      end do

    end if
  !
  !  Label the points.
  !
    if ( 2 <= point_show ) then

      write ( file_unit, '(a)' ) '%'
      write ( file_unit, '(a)' ) '%  Label the point:'
      write ( file_unit, '(a)' ) '%'
      write ( file_unit, '(a)' ) '%  Set the RGB color to darker green.'
      write ( file_unit, '(a)' ) '%'
      write ( file_unit, '(a)' ) '0.250  0.850  0.000 setrgbcolor'
      write ( file_unit, '(a)' ) '/Times-Roman findfont'
      write ( file_unit, '(a)' ) '0.20 inch scalefont'
      write ( file_unit, '(a)' ) 'setfont'
      write ( file_unit, '(a)' ) '%'

      do point = 1, point_num

        x_ps = int ( &
          ( ( x_max - point_xy(1,point)         ) * real ( x_ps_min, kind = 8 )   &
          + (       + point_xy(1,point) - x_min ) * real ( x_ps_max, kind = 8 ) ) &
          / ( x_max                     - x_min ) )

        y_ps = int ( &
          ( ( y_max - point_xy(2,point)         ) * real ( y_ps_min, kind = 8 )   &
          + (         point_xy(2,point) - y_min ) * real ( y_ps_max, kind = 8 ) ) &
          / ( y_max                     - y_min ) )

        write ( string, '(i4)' ) point
        string = adjustl ( string )

        write ( file_unit, '(i4,2x,i4,a)' ) x_ps, y_ps+5, &
          ' moveto (' // trim ( string ) // ') show'

      end do

    end if
  !
  !  Draw the triangle.
  !
    if ( 0 <= node_show ) then
      write ( file_unit, '(a)' ) '%'
      write ( file_unit, '(a)' ) '%  Set the RGB color to red.'
      write ( file_unit, '(a)' ) '%'
      write ( file_unit, '(a)' ) '0.900  0.200  0.100 setrgbcolor'
      write ( file_unit, '(a)' ) '%'
      write ( file_unit, '(a)' ) '%  Draw the triangle.'
      write ( file_unit, '(a)' ) '%'

      write ( file_unit, '(a)' ) 'newpath'

      do i = 1, 4

        node = i4_wrap ( i, i4_1, i4_3 )

        x_ps = int ( &
          ( ( x_max - node_xy(1,node)         ) * real ( x_ps_min, kind = 8 )   &
          + (         node_xy(1,node) - x_min ) * real ( x_ps_max, kind = 8 ) ) &
          / ( x_max                   - x_min ) )

        y_ps = int ( &
          ( ( y_max - node_xy(2,node)         ) * real ( y_ps_min, kind = 8 )   &
          + (         node_xy(2,node) - y_min ) * real ( y_ps_max, kind = 8 ) ) &
          / ( y_max                   - y_min ) )

        if ( i == 1 ) then
          write ( file_unit, '(i3,2x,i3,2x,a)' ) x_ps, y_ps, ' moveto'
        else
          write ( file_unit, '(i3,2x,i3,2x,a)' ) x_ps, y_ps, ' lineto'
        end if

      end do

      write ( file_unit, '(a)' ) 'stroke'

    end if

    write ( file_unit, '(a)' ) '%'
    write ( file_unit, '(a)' ) 'restore  showpage'
    write ( file_unit, '(a)' ) '%'
    write ( file_unit, '(a)' ) '%  End of page.'
    write ( file_unit, '(a)' ) '%'
    write ( file_unit, '(a)' ) '%%Trailer'
    write ( file_unit, '(a)' ) '%%EOF'
    close ( unit = file_unit )

    return
  end subroutine
  subroutine wandzura_degree ( rule, degree )

  !*****************************************************************************80
  !
  !! WANDZURA_DEGREE returns the degree of a given Wandzura rule for the triangle.
  !
  !  Licensing:
  !
  !    This code is distributed under the GNU LGPL license.
  !
  !  Modified:
  !
  !    10 December 2006
  !
  !  Author:
  !
  !    John Burkardt
  !
  !  Reference:
  !
  !    Stephen Wandzura, Hong Xiao,
  !    Symmetric Quadrature Rules on a Triangle,
  !    Computers and Mathematics with Applications,
  !    Volume 45, Number 12, June 2003, pages 1829-1840.
  !
  !  Parameters:
  !
  !    Input, integer ( kind = 4 ) RULE, the index of the rule.
  !
  !    Output, integer ( kind = 4 ) DEGREE, the polynomial degree of exactness of
  !    the rule.
  !
    implicit none

    integer ( kind = 4 ) degree
    integer ( kind = 4 ) rule

    if ( rule == 1 ) then
      degree = 5
    else if ( rule == 2 ) then
      degree = 10
    else if ( rule == 3 ) then
      degree = 15
    else if ( rule == 4 ) then
      degree = 20
    else if ( rule == 5 ) then
      degree = 25
    else if ( rule == 6 ) then
      degree = 30
    else

      degree = -1
      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) 'WANDZURA_DEGREE - Fatal error!'
      write ( *, '(a,i8)' ) '  Illegal RULE = ', rule
      stop

    end if

    return
  end subroutine
  subroutine wandzura_order_num ( rule, order_num )

  !*****************************************************************************80
  !
  !! WANDZURA_ORDER_NUM returns the order of a Wandzura rule for the triangle.
  !
  !  Licensing:
  !
  !    This code is distributed under the GNU LGPL license.
  !
  !  Modified:
  !
  !    10 December 2006
  !
  !  Author:
  !
  !    John Burkardt
  !
  !  Reference:
  !
  !    Stephen Wandzura, Hong Xiao,
  !    Symmetric Quadrature Rules on a Triangle,
  !    Computers and Mathematics with Applications,
  !    Volume 45, Number 12, June 2003, pages 1829-1840.
  !
  !  Parameters:
  !
  !    Input, integer ( kind = 4 ) RULE, the index of the rule.
  !
  !    Output, integer ( kind = 4 ) ORDER_NUM, the order (number of points) of the rule.
  !
    implicit none

    integer ( kind = 4 ) order_num
    integer ( kind = 4 ) rule
    integer ( kind = 4 ), allocatable, dimension ( : ) :: suborder
    integer ( kind = 4 ) suborder_num

    call wandzura_suborder_num ( rule, suborder_num )

    allocate ( suborder(1:suborder_num) )

    call wandzura_suborder ( rule, suborder_num, suborder )

    order_num = sum ( suborder(1:suborder_num) )

    deallocate ( suborder )

    return
  end subroutine

  subroutine wandzura_rule_sp(rule, order_num, xy, w)
    integer, intent(in) :: rule
    integer, intent(in) :: order_num
    real(kind = real_4), dimension(2, order_num), intent(out) :: xy
    real(kind = real_4), dimension(order_num), intent(out) :: w

    real(kind = real_8), dimension(2, order_num) :: lxy
    real(kind = real_8), dimension(order_num) :: lw

    call wandzura_rule(rule, order_num, lxy, lw)
    xy = lxy
    w = lw

  end subroutine wandzura_rule_sp

  subroutine wandzura_rule_orig ( rule, order_num, xy, w )

  !*****************************************************************************80
  !
  !! WANDZURA_RULE returns the points and weights of a Wandzura rule.
  !
  !  Licensing:
  !
  !    This code is distributed under the GNU LGPL license.
  !
  !  Modified:
  !
  !    10 December 2006
  !
  !  Author:
  !
  !    John Burkardt
  !
  !  Reference:
  !
  !    Stephen Wandzura, Hong Xiao,
  !    Symmetric Quadrature Rules on a Triangle,
  !    Computers and Mathematics with Applications,
  !    Volume 45, Number 12, June 2003, pages 1829-1840.
  !
  !  Parameters:
  !
  !    Input, integer ( kind = 4 ) RULE, the index of the rule.
  !
  !    Input, integer ( kind = 4 ) ORDER_NUM, the order (number of points)
  !    of the rule.
  !
  !    Output, real ( kind = 8 ) XY(2,ORDER_NUM), the points of the rule.
  !
  !    Output, real ( kind = 8 ) W(ORDER_NUM), the weights of the rule.
  !
    implicit none

    integer ( kind = 4 ) order_num

    integer ( kind = 4 ), parameter :: i4_1 = 1
    integer ( kind = 4 ), parameter :: i4_3 = 3
    integer ( kind = 4 ) k
    integer ( kind = 4 ) o
    integer ( kind = 4 ) rule
    integer ( kind = 4 ) s
    integer ( kind = 4 ), allocatable, dimension ( : ) :: suborder
    integer ( kind = 4 ) suborder_num
    real    ( kind = 8 ), allocatable, dimension ( : ) :: suborder_w
    real    ( kind = 8 ), allocatable, dimension ( :, : ) :: suborder_xyz
    real    ( kind = 8 ) w(order_num)
    real    ( kind = 8 ) xy(2,order_num)
  !
  !  Get the suborder information.
  !
    call wandzura_suborder_num ( rule, suborder_num )

    allocate ( suborder(suborder_num) )
    allocate ( suborder_xyz(3,suborder_num) )
    allocate ( suborder_w(suborder_num) )

    call wandzura_suborder ( rule, suborder_num, suborder )

    call wandzura_subrule ( rule, suborder_num, suborder_xyz, suborder_w )
  !
  !  Expand the suborder information to a full order rule.
  !
    o = 0

    do s = 1, suborder_num

      if ( suborder(s) == 1 ) then

        o = o + 1
        xy(1:2,o) = suborder_xyz(1:2,s)
        w(o) = 0.5D+00 * suborder_w(s)

      else if ( suborder(s) == 3 ) then

        do k = 1, 3
          o = o + 1
          xy(1,o) = suborder_xyz ( i4_wrap(k,     i4_1,i4_3), s )
          xy(2,o) = suborder_xyz ( i4_wrap(k+i4_1,i4_1,i4_3), s )
          w(o) = 0.5D+00 * suborder_w(s)
        end do

      else if ( suborder(s) == 6 ) then

        do k = 1, 3
          o = o + 1
          xy(1,o) = suborder_xyz ( i4_wrap(k,     i4_1,i4_3), s )
          xy(2,o) = suborder_xyz ( i4_wrap(k+i4_1,i4_1,i4_3), s )
          w(o) = 0.5D+00 * suborder_w(s)
        end do

        do k = 1, 3
          o = o + 1
          xy(1,o) = suborder_xyz ( i4_wrap(k+i4_1,i4_1,i4_3), s )
          xy(2,o) = suborder_xyz ( i4_wrap(k,     i4_1,i4_3), s )
          w(o) = 0.5D+00 * suborder_w(s)
        end do

      else

        write ( *, '(a)' ) ' '
        write ( *, '(a)' ) 'WANDZURA_RULE - Fatal error!'
        write ( *, '(a,i8,a,i8)' ) '  Illegal SUBORDER(', s, ') = ', suborder(s)
        stop

      end if

    end do

    deallocate ( suborder )
    deallocate ( suborder_xyz )
    deallocate ( suborder_w )

    return
  end subroutine
  subroutine wandzura_rule_num ( rule_num )

  !*****************************************************************************80
  !
  !! WANDZURA_RULE_NUM returns the number of Wandzura rules available.
  !
  !  Licensing:
  !
  !    This code is distributed under the GNU LGPL license.
  !
  !  Modified:
  !
  !    10 December 2006
  !
  !  Author:
  !
  !    John Burkardt
  !
  !  Reference:
  !
  !    Stephen Wandzura, Hong Xiao,
  !    Symmetric Quadrature Rules on a Triangle,
  !    Computers and Mathematics with Applications,
  !    Volume 45, Number 12, June 2003, pages 1829-1840.
  !
  !  Parameters:
  !
  !    Output, integer ( kind = 4 ) RULE_NUM, the number of rules available.
  !
    implicit none

    integer ( kind = 4 ) rule_num

    rule_num = 6

    return
  end subroutine
  subroutine wandzura_suborder ( rule, suborder_num, suborder )

  !*****************************************************************************80
  !
  !! WANDZURA_SUBORDER returns the suborders for a Wandzura rule.
  !
  !  Licensing:
  !
  !    This code is distributed under the GNU LGPL license.
  !
  !  Modified:
  !
  !    10 December 2006
  !
  !  Author:
  !
  !    John Burkardt
  !
  !  Reference:
  !
  !    Stephen Wandzura, Hong Xiao,
  !    Symmetric Quadrature Rules on a Triangle,
  !    Computers and Mathematics with Applications,
  !    Volume 45, Number 12, June 2003, pages 1829-1840.
  !
  !  Parameters:
  !
  !    Input, integer ( kind = 4 ) RULE, the index of the rule.
  !
  !    Input, integer ( kind = 4 ) SUBORDER_NUM, the number of suborders
  !    of the rule.
  !
  !    Output, integer ( kind = 4 ) SUBORDER(SUBORDER_NUM), the suborders
  !    of the rule.
  !
    implicit none

    integer ( kind = 4 ) suborder_num

    integer ( kind = 4 ) rule
    integer ( kind = 4 ) suborder(suborder_num)

    if ( rule == 1 ) then
      suborder(1:suborder_num) = (/ &
        1, 3, 3 /)
    else if ( rule == 2 ) then
      suborder(1:suborder_num) = (/ &
        1, 3, 3, 3, 3, 6, 6 /)
    else if ( rule == 3 ) then
      suborder(1:suborder_num) = (/ &
        3, 3, 3, 3, 3, 3, 6, 6, 6, 6, &
        6, 6 /)
    else if ( rule == 4 ) then
      suborder(1:suborder_num) = (/ &
        1, 3, 3, 3, 3, 3, 3, 3, 3, 6, &
        6, 6, 6, 6, 6, 6, 6, 6, 6 /)
    else if ( rule == 5 ) then
      suborder(1:suborder_num) = (/ &
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, &
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, &
        6, 6, 6, 6, 6, 6  /)
    else if ( rule == 6 ) then
      suborder(1:suborder_num) = (/ &
        1, 3, 3, 3, 3, 3, 3, 3, 3, 3, &
        3, 3, 3, 6, 6, 6, 6, 6, 6, 6, &
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, &
        6, 6, 6, 6, 6, 6 /)
    else

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) 'WANDZURA_SUBORDER - Fatal error!'
      write ( *, '(a,i8)' ) '  Illegal RULE = ', rule
      stop

    end if

    return
  end subroutine
  subroutine wandzura_suborder_num ( rule, suborder_num )

  !*****************************************************************************80
  !
  !! WANDZURA_SUBORDER_NUM returns the number of suborders for a Wandzura rule.
  !
  !  Licensing:
  !
  !    This code is distributed under the GNU LGPL license.
  !
  !  Modified:
  !
  !    10 December 2006
  !
  !  Author:
  !
  !    John Burkardt
  !
  !  Reference:
  !
  !    Stephen Wandzura, Hong Xiao,
  !    Symmetric Quadrature Rules on a Triangle,
  !    Computers and Mathematics with Applications,
  !    Volume 45, Number 12, June 2003, pages 1829-1840.
  !
  !  Parameters:
  !
  !    Input, integer ( kind = 4 ) RULE, the index of the rule.
  !
  !    Output, integer ( kind = 4 ) SUBORDER_NUM, the number of suborders of the rule.
  !
    implicit none

    integer ( kind = 4 ) rule
    integer ( kind = 4 ) suborder_num

    if ( rule == 1 ) then
      suborder_num = 3
    else if ( rule == 2 ) then
      suborder_num = 7
    else if ( rule == 3 ) then
      suborder_num = 12
    else if ( rule == 4 ) then
      suborder_num = 19
    else if ( rule == 5 ) then
      suborder_num = 26
    else if ( rule == 6 ) then
      suborder_num = 36
    else

      suborder_num = -1
      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) 'WANDZURA_SUBORDER_NUM - Fatal error!'
      write ( *, '(a,i8)' ) '  Illegal RULE = ', rule
      stop

    end if

    return
  end subroutine
  subroutine wandzura_subrule ( rule, suborder_num, suborder_xyz, suborder_w )

  !*****************************************************************************80
  !
  !! WANDZURA_SUBRULE returns a compressed Wandzura rule.
  !
  !  Licensing:
  !
  !    This code is distributed under the GNU LGPL license.
  !
  !  Modified:
  !
  !    10 May 2007
  !
  !  Author:
  !
  !    John Burkardt
  !
  !  Reference:
  !
  !    Stephen Wandzura, Hong Xiao,
  !    Symmetric Quadrature Rules on a Triangle,
  !    Computers and Mathematics with Applications,
  !    Volume 45, Number 12, June 2003, pages 1829-1840.
  !
  !  Parameters:
  !
  !    Input, integer ( kind = 4 ) RULE, the index of the rule.
  !
  !    Input, integer ( kind = 4 ) SUBORDER_NUM, the number of suborders
  !    of the rule.
  !
  !    Output, real ( kind = 8 ) SUBORDER_XYZ(3,SUBORDER_NUM),
  !    the barycentric coordinates of the abscissas.
  !
  !    Output, real ( kind = 8 ) SUBORDER_W(SUBORDER_NUM), the
  !    suborder weights.
  !
    implicit none

    integer ( kind = 4 ) suborder_num

    integer ( kind = 4 ), parameter :: i4_3 = 3
    integer ( kind = 4 ) rule
    real    ( kind = 8 ) suborder_w(suborder_num)
    real    ( kind = 8 ) suborder_xyz(3,suborder_num)

    if ( rule == 1 ) then

      suborder_xyz(1:3,1:suborder_num) = reshape ( (/ &
        0.33333333333333D+00, 0.33333333333333D+00, 0.33333333333333D+00, &
        0.05971587178977D+00, 0.47014206410512D+00, 0.47014206410512D+00, &
        0.79742698535309D+00, 0.10128650732346D+00, 0.10128650732346D+00  &
      /), (/ i4_3, suborder_num /) )

      suborder_w(1:suborder_num) = (/ &
        0.2250000000000000D+00, &
        0.1323941527885062D+00, &
        0.1259391805448271D+00  &
      /)

    else if ( rule == 2 ) then

      suborder_xyz(1:3,1:suborder_num) = reshape ( (/ &
        0.33333333333333D+00, 0.33333333333333D+00, 0.33333333333333D+00, &
        0.00426913409105D+00, 0.49786543295447D+00, 0.49786543295447D+00, &
        0.14397510054189D+00, 0.42801244972906D+00, 0.42801244972906D+00, &
        0.63048717451355D+00, 0.18475641274322D+00, 0.18475641274322D+00, &
        0.95903756285664D+00, 0.02048121857168D+00, 0.02048121857168D+00, &
        0.03500298989727D+00, 0.13657357625603D+00, 0.82842343384669D+00, &
        0.03754907025844D+00, 0.33274360058864D+00, 0.62970732915292D+00  &
      /), (/ i4_3, suborder_num /) )

      suborder_w(1:suborder_num) = (/ &
        0.8352339980519638D-01, &
        0.7229850592056743D-02, &
        0.7449217792098051D-01, &
        0.7864647340310853D-01, &
        0.6928323087107504D-02, &
        0.2951832033477940D-01, &
        0.3957936719606124D-01  &
      /)

    else if ( rule == 3 ) then

      suborder_xyz(1:3,1:suborder_num) = reshape ( (/ &
        0.08343840726175D+00, 0.45828079636912D+00, 0.45828079636913D+00, &
        0.19277907084174D+00, 0.40361046457913D+00, 0.40361046457913D+00, &
        0.41360566417395D+00, 0.29319716791303D+00, 0.29319716791303D+00, &
        0.70706442611445D+00, 0.14646778694277D+00, 0.14646778694277D+00, &
        0.88727426466879D+00, 0.05636286766560D+00, 0.05636286766560D+00, &
        0.96684974628326D+00, 0.01657512685837D+00, 0.01657512685837D+00, &
        0.00991220330923D+00, 0.23953455415479D+00, 0.75055324253598D+00, &
        0.01580377063023D+00, 0.40487880731834D+00, 0.57931742205143D+00, &
        0.00514360881697D+00, 0.09500211311304D+00, 0.89985427806998D+00, &
        0.04892232575299D+00, 0.14975310732227D+00, 0.80132456692474D+00, &
        0.06876874863252D+00, 0.28691961244133D+00, 0.64431163892615D+00, &
        0.16840441812470D+00, 0.28183566809908D+00, 0.54975991377622D+00  &
      /), (/ i4_3, suborder_num /) )

      suborder_w(1:suborder_num) = (/ &
        0.3266181884880529D-01, &
        0.2741281803136436D-01, &
        0.2651003659870330D-01, &
        0.2921596213648611D-01, &
        0.1058460806624399D-01, &
        0.3614643064092035D-02, &
        0.8527748101709436D-02, &
        0.1391617651669193D-01, &
        0.4291932940734835D-02, &
        0.1623532928177489D-01, &
        0.2560734092126239D-01, &
        0.3308819553164567D-01  &
      /)

    else if ( rule == 4 ) then

      suborder_xyz(1:3,1:suborder_num) = reshape ( (/ &
        0.33333333333333D+00, 0.33333333333333D+00, 0.33333333333333D+00, &
        0.00150064932443D+00, 0.49924967533779D+00, 0.49924967533779D+00, &
        0.09413975193895D+00, 0.45293012403052D+00, 0.45293012403052D+00, &
        0.20447212408953D+00, 0.39776393795524D+00, 0.39776393795524D+00, &
        0.47099959493443D+00, 0.26450020253279D+00, 0.26450020253279D+00, &
        0.57796207181585D+00, 0.21101896409208D+00, 0.21101896409208D+00, &
        0.78452878565746D+00, 0.10773560717127D+00, 0.10773560717127D+00, &
        0.92186182432439D+00, 0.03906908783780D+00, 0.03906908783780D+00, &
        0.97765124054134D+00, 0.01117437972933D+00, 0.01117437972933D+00, &
        0.00534961818734D+00, 0.06354966590835D+00, 0.93110071590431D+00, &
        0.00795481706620D+00, 0.15710691894071D+00, 0.83493826399309D+00, &
        0.01042239828126D+00, 0.39564211436437D+00, 0.59393548735436D+00, &
        0.01096441479612D+00, 0.27316757071291D+00, 0.71586801449097D+00, &
        0.03856671208546D+00, 0.10178538248502D+00, 0.85964790542952D+00, &
        0.03558050781722D+00, 0.44665854917641D+00, 0.51776094300637D+00, &
        0.04967081636276D+00, 0.19901079414950D+00, 0.75131838948773D+00, &
        0.05851972508433D+00, 0.32426118369228D+00, 0.61721909122339D+00, &
        0.12149778700439D+00, 0.20853136321013D+00, 0.66997084978547D+00, &
        0.14071084494394D+00, 0.32317056653626D+00, 0.53611858851980D+00  &
      /), (/ i4_3, suborder_num /) )

      suborder_w(1:suborder_num) = (/ &
        0.2761042699769952D-01, &
        0.1779029547326740D-02, &
        0.2011239811396117D-01, &
        0.2681784725933157D-01, &
        0.2452313380150201D-01, &
        0.1639457841069539D-01, &
        0.1479590739864960D-01, &
        0.4579282277704251D-02, &
        0.1651826515576217D-02, &
        0.2349170908575584D-02, &
        0.4465925754181793D-02, &
        0.6099566807907972D-02, &
        0.6891081327188203D-02, &
        0.7997475072478163D-02, &
        0.7386134285336024D-02, &
        0.1279933187864826D-01, &
        0.1725807117569655D-01, &
        0.1867294590293547D-01, &
        0.2281822405839526D-01  &
      /)

    else if ( rule == 5 ) then

      suborder_xyz(1:3,1:suborder_num) = reshape ( (/ &
        0.02794648307317D+00, 0.48602675846341D+00, 0.48602675846341D+00, &
        0.13117860132765D+00, 0.43441069933617D+00, 0.43441069933617D+00, &
        0.22022172951207D+00, 0.38988913524396D+00, 0.38988913524396D+00, &
        0.40311353196039D+00, 0.29844323401980D+00, 0.29844323401980D+00, &
        0.53191165532526D+00, 0.23404417233737D+00, 0.23404417233737D+00, &
        0.69706333078196D+00, 0.15146833460902D+00, 0.15146833460902D+00, &
        0.77453221290801D+00, 0.11273389354599D+00, 0.11273389354599D+00, &
        0.84456861581695D+00, 0.07771569209153D+00, 0.07771569209153D+00, &
        0.93021381277141D+00, 0.03489309361430D+00, 0.03489309361430D+00, &
        0.98548363075813D+00, 0.00725818462093D+00, 0.00725818462093D+00, &
        0.00129235270444D+00, 0.22721445215336D+00, 0.77149319514219D+00, &
        0.00539970127212D+00, 0.43501055485357D+00, 0.55958974387431D+00, &
        0.00638400303398D+00, 0.32030959927220D+00, 0.67330639769382D+00, &
        0.00502821150199D+00, 0.09175032228001D+00, 0.90322146621800D+00, &
        0.00682675862178D+00, 0.03801083585872D+00, 0.95516240551949D+00, &
        0.01001619963993D+00, 0.15742521848531D+00, 0.83255858187476D+00, &
        0.02575781317339D+00, 0.23988965977853D+00, 0.73435252704808D+00, &
        0.03022789811992D+00, 0.36194311812606D+00, 0.60782898375402D+00, &
        0.03050499010716D+00, 0.08355196095483D+00, 0.88594304893801D+00, &
        0.04595654736257D+00, 0.14844322073242D+00, 0.80560023190501D+00, &
        0.06744280054028D+00, 0.28373970872753D+00, 0.64881749073219D+00, &
        0.07004509141591D+00, 0.40689937511879D+00, 0.52305553346530D+00, &
        0.08391152464012D+00, 0.19411398702489D+00, 0.72197448833499D+00, &
        0.12037553567715D+00, 0.32413434700070D+00, 0.55549011732214D+00, &
        0.14806689915737D+00, 0.22927748355598D+00, 0.62265561728665D+00, &
        0.19177186586733D+00, 0.32561812259598D+00, 0.48261001153669D+00  &
      /), (/ i4_3, suborder_num /) )

      suborder_w(1:suborder_num) = (/ &
        0.8005581880020417D-02, &
        0.1594707683239050D-01, &
        0.1310914123079553D-01, &
        0.1958300096563562D-01, &
        0.1647088544153727D-01, &
        0.8547279074092100D-02, &
        0.8161885857226492D-02, &
        0.6121146539983779D-02, &
        0.2908498264936665D-02, &
        0.6922752456619963D-03, &
        0.1248289199277397D-02, &
        0.3404752908803022D-02, &
        0.3359654326064051D-02, &
        0.1716156539496754D-02, &
        0.1480856316715606D-02, &
        0.3511312610728685D-02, &
        0.7393550149706484D-02, &
        0.7983087477376558D-02, &
        0.4355962613158041D-02, &
        0.7365056701417832D-02, &
        0.1096357284641955D-01, &
        0.1174996174354112D-01, &
        0.1001560071379857D-01, &
        0.1330964078762868D-01, &
        0.1415444650522614D-01, &
        0.1488137956116801D-01  &
      /)

    else if ( rule == 6 ) then

      suborder_xyz(1:3,1:suborder_num) = reshape ( (/ &
        0.33333333333333D+00, 0.33333333333333D+00, 0.33333333333333D+00, &
        0.00733011643277D+00, 0.49633494178362D+00, 0.49633494178362D+00, &
        0.08299567580296D+00, 0.45850216209852D+00, 0.45850216209852D+00, &
        0.15098095612541D+00, 0.42450952193729D+00, 0.42450952193729D+00, &
        0.23590585989217D+00, 0.38204707005392D+00, 0.38204707005392D+00, &
        0.43802430840785D+00, 0.28098784579608D+00, 0.28098784579608D+00, &
        0.54530204829193D+00, 0.22734897585403D+00, 0.22734897585403D+00, &
        0.65088177698254D+00, 0.17455911150873D+00, 0.17455911150873D+00, &
        0.75348314559713D+00, 0.12325842720144D+00, 0.12325842720144D+00, &
        0.83983154221561D+00, 0.08008422889220D+00, 0.08008422889220D+00, &
        0.90445106518420D+00, 0.04777446740790D+00, 0.04777446740790D+00, &
        0.95655897063972D+00, 0.02172051468014D+00, 0.02172051468014D+00, &
        0.99047064476913D+00, 0.00476467761544D+00, 0.00476467761544D+00, &
        0.00092537119335D+00, 0.41529527091331D+00, 0.58377935789334D+00, &
        0.00138592585556D+00, 0.06118990978535D+00, 0.93742416435909D+00, &
        0.00368241545591D+00, 0.16490869013691D+00, 0.83140889440718D+00, &
        0.00390322342416D+00, 0.02503506223200D+00, 0.97106171434384D+00, &
        0.00323324815501D+00, 0.30606446515110D+00, 0.69070228669389D+00, &
        0.00646743211224D+00, 0.10707328373022D+00, 0.88645928415754D+00, &
        0.00324747549133D+00, 0.22995754934558D+00, 0.76679497516308D+00, &
        0.00867509080675D+00, 0.33703663330578D+00, 0.65428827588746D+00, &
        0.01559702646731D+00, 0.05625657618206D+00, 0.92814639735063D+00, &
        0.01797672125369D+00, 0.40245137521240D+00, 0.57957190353391D+00, &
        0.01712424535389D+00, 0.24365470201083D+00, 0.73922105263528D+00, &
        0.02288340534658D+00, 0.16538958561453D+00, 0.81172700903888D+00, &
        0.03273759728777D+00, 0.09930187449585D+00, 0.86796052821639D+00, &
        0.03382101234234D+00, 0.30847833306905D+00, 0.65770065458860D+00, &
        0.03554761446002D+00, 0.46066831859211D+00, 0.50378406694787D+00, &
        0.05053979030687D+00, 0.21881529945393D+00, 0.73064491023920D+00, &
        0.05701471491573D+00, 0.37920955156027D+00, 0.56377573352399D+00, &
        0.06415280642120D+00, 0.14296081941819D+00, 0.79288637416061D+00, &
        0.08050114828763D+00, 0.28373128210592D+00, 0.63576756960645D+00, &
        0.10436706813453D+00, 0.19673744100444D+00, 0.69889549086103D+00, &
        0.11384489442875D+00, 0.35588914121166D+00, 0.53026596435959D+00, &
        0.14536348771552D+00, 0.25981868535191D+00, 0.59481782693256D+00, &
        0.18994565282198D+00, 0.32192318123130D+00, 0.48813116594672D+00  &
      /), (/ i4_3, suborder_num /) )

      suborder_w(1:suborder_num) = (/ &
        0.1557996020289920D-01, &
        0.3177233700534134D-02, &
        0.1048342663573077D-01, &
        0.1320945957774363D-01, &
        0.1497500696627150D-01, &
        0.1498790444338419D-01, &
        0.1333886474102166D-01, &
        0.1088917111390201D-01, &
        0.8189440660893461D-02, &
        0.5575387588607785D-02, &
        0.3191216473411976D-02, &
        0.1296715144327045D-02, &
        0.2982628261349172D-03, &
        0.9989056850788964D-03, &
        0.4628508491732533D-03, &
        0.1234451336382413D-02, &
        0.5707198522432062D-03, &
        0.1126946125877624D-02, &
        0.1747866949407337D-02, &
        0.1182818815031657D-02, &
        0.1990839294675034D-02, &
        0.1900412795035980D-02, &
        0.4498365808817451D-02, &
        0.3478719460274719D-02, &
        0.4102399036723953D-02, &
        0.4021761549744162D-02, &
        0.6033164660795066D-02, &
        0.3946290302129598D-02, &
        0.6644044537680268D-02, &
        0.8254305856078458D-02, &
        0.6496056633406411D-02, &
        0.9252778144146602D-02, &
        0.9164920726294280D-02, &
        0.1156952462809767D-01, &
        0.1176111646760917D-01, &
        0.1382470218216540D-01  &
      /)

    else

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) 'WANDZURA_SUBRULE - Fatal error!'
      write ( *, '(a,i8)' ) '  Illegal RULE = ', rule
      stop

    end if

    return
  end subroutine
  subroutine wandzura_subrule2 ( rule, suborder_num, suborder_xy, suborder_w )

  !*****************************************************************************80
  !
  !! WANDZURA_SUBRULE2 returns a compressed Wandzura rule.
  !
  !  Discussion:
  !
  !    This version of the rules uses as reference the equilateral
  !    triangle whose vertices are (-1/2,-sqrt(3)/2), (1,0), (-1/2,sqrt(3)/2).
  !
  !    This, in fact, is the data as printed in the reference.
  !
  !    Currently, we don't use this routine at all.  The values of
  !    X and Y here could be converted to lie XSI and ETA in the
  !    standard (0,0), (1,0), (0,1) reference triangle by
  !
  !      XSI = ( 2/3) * X                 + 1/3
  !      ETA = (-1/3) * X + sqrt(3)/3 * Y + 1/3
  !
  !  Licensing:
  !
  !    This code is distributed under the GNU LGPL license.
  !
  !  Modified:
  !
  !    11 May 2007
  !
  !  Author:
  !
  !    John Burkardt
  !
  !  Reference:
  !
  !    Stephen Wandzura, Hong Xiao,
  !    Symmetric Quadrature Rules on a Triangle,
  !    Computers and Mathematics with Applications,
  !    Volume 45, Number 12, June 2003, pages 1829-1840.
  !
  !  Parameters:
  !
  !    Input, integer ( kind = 4 ) RULE, the index of the rule.
  !
  !    Input, integer ( kind = 4 ) SUBORDER_NUM, the number of suborders
  !    of the rule.
  !
  !    Output, real ( kind = 8 ) SUBORDER_XY(2,SUBORDER_NUM),
  !    the (X,Y) coordinates of the abscissas.
  !
  !    Output, real ( kind = 8 ) SUBORDER_W(SUBORDER_NUM), the
  !    suborder weights.
  !
    implicit none

    integer ( kind = 4 ) suborder_num

    integer ( kind = 4 ), parameter :: i4_2 = 2
    integer ( kind = 4 ) rule
    real    ( kind = 8 ) suborder_w(suborder_num)
    real    ( kind = 8 ) suborder_xy(3,suborder_num)

    if ( rule == 1 ) then

      suborder_xy(1:2,1:suborder_num) = reshape ( (/ &
         0.0000000000000000D+00,   0.0000000000000000D+00, &
        -0.4104261923153453D+00,   0.0000000000000000D+00, &
         0.6961404780296310D+00,   0.0000000000000000D+00  &
      /), (/ i4_2, suborder_num /) )

      suborder_w(1:suborder_num) = (/ &
         0.2250000000000000D+00, &
         0.1323941527885062D+00, &
         0.1259391805448271D+00  &
      /)

    else if ( rule == 2 ) then

      suborder_xy(1:2,1:suborder_num) = reshape ( (/ &
         0.0000000000000000D+00,   0.0000000000000000D+00, &
        -0.4935962988634245D+00,   0.0000000000000000D+00, &
        -0.2840373491871686D+00,   0.0000000000000000D+00, &
         0.4457307617703263D+00,   0.0000000000000000D+00, &
         0.9385563442849673D+00,   0.0000000000000000D+00, &
        -0.4474955151540920D+00,  -0.5991595522781586D+00, &
        -0.4436763946123360D+00,  -0.2571781329392130D+00  &
      /), (/ i4_2, suborder_num /) )

      suborder_w(1:suborder_num) = (/ &
         0.8352339980519638D-01, &
         0.7229850592056743D-02, &
         0.7449217792098051D-01, &
         0.7864647340310853D-01, &
         0.6928323087107504D-02, &
         0.2951832033477940D-01, &
         0.3957936719606124D-01  &
      /)

    else if ( rule == 3 ) then

      suborder_xy(1:2,1:suborder_num) = reshape ( (/ &
        -0.3748423891073751D+00,   0.0000000000000000D+00, &
        -0.2108313937373917D+00,   0.0000000000000000D+00, &
         0.1204084962609239D+00,   0.0000000000000000D+00, &
         0.5605966391716812D+00,   0.0000000000000000D+00, &
         0.8309113970031897D+00,   0.0000000000000000D+00, &
         0.9502746194248890D+00,   0.0000000000000000D+00, &
        -0.4851316950361628D+00,  -0.4425551659467111D+00, &
        -0.4762943440546580D+00,  -0.1510682717598242D+00, &
        -0.4922845867745440D+00,  -0.6970224211436132D+00, &
        -0.4266165113705168D+00,  -0.5642774363966393D+00, &
        -0.3968468770512212D+00,  -0.3095105740458471D+00, &
        -0.2473933728129512D+00,  -0.2320292030461791D+00  &
      /), (/ i4_2, suborder_num /) )

      suborder_w(1:suborder_num) = (/ &
         0.3266181884880529D-01, &
         0.2741281803136436D-01, &
         0.2651003659870330D-01, &
         0.2921596213648611D-01, &
         0.1058460806624399D-01, &
         0.3614643064092035D-02, &
         0.8527748101709436D-02, &
         0.1391617651669193D-01, &
         0.4291932940734835D-02, &
         0.1623532928177489D-01, &
         0.2560734092126239D-01, &
         0.3308819553164567D-01  &
      /)

    else if ( rule == 4 ) then

      suborder_xy(1:2,1:suborder_num) = reshape ( (/ &
         0.0000000000000000D+00,   0.0000000000000000D+00, &
        -0.4977490260133565D+00,   0.0000000000000000D+00, &
        -0.3587903720915737D+00,   0.0000000000000000D+00, &
        -0.1932918138657104D+00,   0.0000000000000000D+00, &
         0.2064993924016380D+00,   0.0000000000000000D+00, &
         0.3669431077237697D+00,   0.0000000000000000D+00, &
         0.6767931784861860D+00,   0.0000000000000000D+00, &
         0.8827927364865920D+00,   0.0000000000000000D+00, &
         0.9664768608120111D+00,   0.0000000000000000D+00, &
        -0.4919755727189941D+00,  -0.7513212483763635D+00, &
        -0.4880677744007016D+00,  -0.5870191642967427D+00, &
        -0.4843664025781043D+00,  -0.1717270984114328D+00, &
        -0.4835533778058150D+00,  -0.3833898305784408D+00, &
        -0.4421499318718065D+00,  -0.6563281974461070D+00, &
        -0.4466292382741727D+00,  -0.6157647932662624D-01, &
        -0.4254937754558538D+00,  -0.4783124082660027D+00, &
        -0.4122204123735024D+00,  -0.2537089901614676D+00, &
        -0.3177533194934086D+00,  -0.3996183176834929D+00, &
        -0.2889337325840919D+00,  -0.1844183967233982D+00  &
      /), (/ i4_2, suborder_num /) )

      suborder_w(1:suborder_num) = (/ &
         0.2761042699769952D-01, &
         0.1779029547326740D-02, &
         0.2011239811396117D-01, &
         0.2681784725933157D-01, &
         0.2452313380150201D-01, &
         0.1639457841069539D-01, &
         0.1479590739864960D-01, &
         0.4579282277704251D-02, &
         0.1651826515576217D-02, &
         0.2349170908575584D-02, &
         0.4465925754181793D-02, &
         0.6099566807907972D-02, &
         0.6891081327188203D-02, &
         0.7997475072478163D-02, &
         0.7386134285336024D-02, &
         0.1279933187864826D-01, &
         0.1725807117569655D-01, &
         0.1867294590293547D-01, &
         0.2281822405839526D-01  &
      /)

    else if ( rule == 5 ) then

      suborder_xy(1:2,1:suborder_num) = reshape ( (/ &
        -0.4580802753902387D+00,   0.0000000000000000D+00, &
        -0.3032320980085228D+00,   0.0000000000000000D+00, &
        -0.1696674057318916D+00,   0.0000000000000000D+00, &
         0.1046702979405866D+00,   0.0000000000000000D+00, &
         0.2978674829878846D+00,   0.0000000000000000D+00, &
         0.5455949961729473D+00,   0.0000000000000000D+00, &
         0.6617983193620190D+00,   0.0000000000000000D+00, &
         0.7668529237254211D+00,   0.0000000000000000D+00, &
         0.8953207191571090D+00,   0.0000000000000000D+00, &
         0.9782254461372029D+00,   0.0000000000000000D+00, &
        -0.4980614709433367D+00,  -0.4713592181681879D+00, &
        -0.4919004480918257D+00,  -0.1078887424748246D+00, &
        -0.4904239954490375D+00,  -0.3057041948876942D+00, &
        -0.4924576827470104D+00,  -0.7027546250883238D+00, &
        -0.4897598620673272D+00,  -0.7942765584469995D+00, &
        -0.4849757005401057D+00,  -0.5846826436376921D+00, &
        -0.4613632802399150D+00,  -0.4282174042835178D+00, &
        -0.4546581528201263D+00,  -0.2129434060653430D+00, &
        -0.4542425148392569D+00,  -0.6948910659636692D+00, &
        -0.4310651789561460D+00,  -0.5691146659505208D+00, &
        -0.3988357991895837D+00,  -0.3161666335733065D+00, &
        -0.3949323628761341D+00,  -0.1005941839340892D+00, &
        -0.3741327130398251D+00,  -0.4571406037889341D+00, &
        -0.3194366964842710D+00,  -0.2003599744104858D+00, &
        -0.2778996512639500D+00,  -0.3406754571040736D+00, &
        -0.2123422011990124D+00,  -0.1359589640107579D+00  &
      /), (/ i4_2, suborder_num /) )

      suborder_w(1:suborder_num) = (/ &
         0.8005581880020417D-02, &
         0.1594707683239050D-01, &
         0.1310914123079553D-01, &
         0.1958300096563562D-01, &
         0.1647088544153727D-01, &
         0.8547279074092100D-02, &
         0.8161885857226492D-02, &
         0.6121146539983779D-02, &
         0.2908498264936665D-02, &
         0.6922752456619963D-03, &
         0.1248289199277397D-02, &
         0.3404752908803022D-02, &
         0.3359654326064051D-02, &
         0.1716156539496754D-02, &
         0.1480856316715606D-02, &
         0.3511312610728685D-02, &
         0.7393550149706484D-02, &
         0.7983087477376558D-02, &
         0.4355962613158041D-02, &
         0.7365056701417832D-02, &
         0.1096357284641955D-01, &
         0.1174996174354112D-01, &
         0.1001560071379857D-01, &
         0.1330964078762868D-01, &
         0.1415444650522614D-01, &
         0.1488137956116801D-01  &
      /)

    else if ( rule == 6 ) then

      suborder_xy(1:2,1:suborder_num) = reshape ( (/ &
         0.0000000000000000D+00,   0.0000000000000000D+00, &
        -0.4890048253508517D+00,   0.0000000000000000D+00, &
        -0.3755064862955532D+00,   0.0000000000000000D+00, &
        -0.2735285658118844D+00,   0.0000000000000000D+00, &
        -0.1461412101617502D+00,   0.0000000000000000D+00, &
         0.1570364626117722D+00,   0.0000000000000000D+00, &
         0.3179530724378968D+00,   0.0000000000000000D+00, &
         0.4763226654738105D+00,   0.0000000000000000D+00, &
         0.6302247183956902D+00,   0.0000000000000000D+00, &
         0.7597473133234094D+00,   0.0000000000000000D+00, &
         0.8566765977763036D+00,   0.0000000000000000D+00, &
         0.9348384559595755D+00,   0.0000000000000000D+00, &
         0.9857059671536891D+00,   0.0000000000000000D+00, &
        -0.4986119432099803D+00,  -0.1459114994581331D+00, &
        -0.4979211112166541D+00,  -0.7588411241269780D+00, &
        -0.4944763768161339D+00,  -0.5772061085255766D+00, &
        -0.4941451648637610D+00,  -0.8192831133859931D+00, &
        -0.4951501277674842D+00,  -0.3331061247123685D+00, &
        -0.4902988518316453D+00,  -0.6749680757240147D+00, &
        -0.4951287867630010D+00,  -0.4649148484601980D+00, &
        -0.4869873637898693D+00,  -0.2747479818680760D+00, &
        -0.4766044602990292D+00,  -0.7550787344330482D+00, &
        -0.4730349181194722D+00,  -0.1533908770581512D+00, &
        -0.4743136319691660D+00,  -0.4291730489015232D+00, &
        -0.4656748919801272D+00,  -0.5597446281020688D+00, &
        -0.4508936040683500D+00,  -0.6656779209607333D+00, &
        -0.4492684814864886D+00,  -0.3024354020045064D+00, &
        -0.4466785783099771D+00,  -0.3733933337926417D-01, &
        -0.4241903145397002D+00,  -0.4432574453491491D+00, &
        -0.4144779276264017D+00,  -0.1598390022600824D+00, &
        -0.4037707903681949D+00,  -0.5628520409756346D+00, &
        -0.3792482775685616D+00,  -0.3048723680294163D+00, &
        -0.3434493977982042D+00,  -0.4348816278906578D+00, &
        -0.3292326583568731D+00,  -0.1510147586773290D+00, &
        -0.2819547684267144D+00,  -0.2901177668548256D+00, &
        -0.2150815207670319D+00,  -0.1439403370753732D+00  &
      /), (/ i4_2, suborder_num /) )

      suborder_w(1:suborder_num) = (/ &
         0.1557996020289920D-01, &
         0.3177233700534134D-02, &
         0.1048342663573077D-01, &
         0.1320945957774363D-01, &
         0.1497500696627150D-01, &
         0.1498790444338419D-01, &
         0.1333886474102166D-01, &
         0.1088917111390201D-01, &
         0.8189440660893461D-02, &
         0.5575387588607785D-02, &
         0.3191216473411976D-02, &
         0.1296715144327045D-02, &
         0.2982628261349172D-03, &
         0.9989056850788964D-03, &
         0.4628508491732533D-03, &
         0.1234451336382413D-02, &
         0.5707198522432062D-03, &
         0.1126946125877624D-02, &
         0.1747866949407337D-02, &
         0.1182818815031657D-02, &
         0.1990839294675034D-02, &
         0.1900412795035980D-02, &
         0.4498365808817451D-02, &
         0.3478719460274719D-02, &
         0.4102399036723953D-02, &
         0.4021761549744162D-02, &
         0.6033164660795066D-02, &
         0.3946290302129598D-02, &
         0.6644044537680268D-02, &
         0.8254305856078458D-02, &
         0.6496056633406411D-02, &
         0.9252778144146602D-02, &
         0.9164920726294280D-02, &
         0.1156952462809767D-01, &
         0.1176111646760917D-01, &
         0.1382470218216540D-01  &
      /)

    else

      write ( *, '(a)' ) ' '
      write ( *, '(a)' ) 'WANDZURA_SUBRULE2 - Fatal error!'
      write ( *, '(a,i8)' ) '  Illegal RULE = ', rule
      stop

    end if

    return
  end subroutine
end module wandzura_quadrature
