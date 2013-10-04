dnl @synopsis ACX_LAPACK([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
dnl
dnl This macro looks for a library that implements the LAPACK
dnl linear-algebra interface (see http://www.netlib.org/lapack/).
dnl On success, it sets the LAPACK_LIBS output variable to
dnl hold the requisite library linkages.
dnl
dnl To link with LAPACK, you should link with:
dnl
dnl 	$LAPACK_LIBS $BLAS_LIBS $LIBS
dnl
dnl in that order.  BLAS_LIBS is the output variable of the ACX_BLAS
dnl macro, called automatically.  FLIBS is the output variable of the
dnl AC_F77_LIBRARY_LDFLAGS macro (called if necessary by ACX_BLAS),
dnl and is sometimes necessary in order to link with F77 libraries.
dnl Users will also need to use AC_F77_DUMMY_MAIN (see the autoconf
dnl manual), for the same reason.
dnl
dnl The user may also use --with-lapack=<lib> in order to use some
dnl specific LAPACK library <lib>.  In order to link successfully,
dnl however, be aware that you will probably need to use the same
dnl Fortran compiler (which can be set via the F77 env. var.) as
dnl was used to compile the LAPACK and BLAS libraries.
dnl
dnl ACTION-IF-FOUND is a list of shell commands to run if a LAPACK
dnl library is found, and ACTION-IF-NOT-FOUND is a list of commands
dnl to run it if it is not found.  If ACTION-IF-FOUND is not specified,
dnl the default action will define HAVE_LAPACK.
dnl
dnl @version $Id: acx_lapack.m4,v 1.2 2003/04/01 09:18:55 juselius Exp $
dnl @author Steven G. Johnson <stevenj@alum.mit.edu>
dnl
AC_DEFUN([ACX_LAPACK], [
AC_REQUIRE([ACX_BLAS])
acx_lapack_ok=no
acx_lapack_save_LIBS="$LIBS"
acx_lapack_save_LDFLAGS="$LDFLAGS"
acx_lapack_save_FFLAGS="$FFLAGS"
acx_lapack_libs=""
acx_lapack_dir=""

AC_ARG_WITH(lapack,
	[AC_HELP_STRING([--with-lapack=<lib>], [use LAPACK library <lib>])])

case $with_lapack in
	yes | "") ;;
	no) acx_lapack_ok=disable ;;
	-l* | */* | *.a | *.so | *.so.* | *.o) acx_lapack_libs="$with_lapack" ;;
	*) acx_lapack_libs="-l$with_lapack" ;;
esac

AC_ARG_WITH(lapack_dir,
	[AC_HELP_STRING([--with-lapack-dir=<dir>], [look for LAPACK library in <dir>])])

case $with_lapack_dir in
      yes | no | "") ;;
     -L*) LDFLAGS="$LDFLAGS $with_lapack_dir"
	      acx_lapack_dir="$with_lapack_dir" ;;
      *) LDFLAGS="$LDFLAGS -L$with_lapack_dir"
	      acx_lapack_dir="-L$with_lapack_dir" ;;
esac

# We cannot use LAPACK if BLAS is not found
if test "x$acx_blas_ok" != xyes; then
	acx_lapack_ok=noblas
fi

# add BLAS to libs
LIBS="$BLAS_LIBS $LIBS"

# Are we linking from C?
case "$ac_ext" in
  f*|F*) dsyev="dsyev" ;;
  *)
   AC_FC_FUNC([dsyev])
   LIBS="$LIBS $FCLIBS"
   ;;
esac

# If --with-lapack is defined, then look for THIS AND ONLY THIS lapack lib
if test $acx_lapack_ok = no; then
case $with_lapack in
    ""|yes) ;;
	*) save_LIBS="$LIBS"; LIBS="$acx_lapack_libs $LIBS"
	AC_MSG_CHECKING([for $dsyev in $acx_lapack_libs])
	AC_TRY_LINK_FUNC($dsyev, [acx_lapack_ok=yes])
	AC_MSG_RESULT($acx_lapack_ok)
	LIBS="$save_LIBS"
	acx_lapack_ok=specific
	;;
esac
fi

# First, check LAPACK_LIBS environment variable
if test $acx_lapack_ok = no; then
if test "x$LAPACK_LIBS" != x; then
	save_LIBS="$LIBS"; LIBS="$LAPACK_LIBS $LIBS"
	AC_MSG_CHECKING([for $dsyev in $LAPACK_LIBS])
	AC_TRY_LINK_FUNC($dsyev, [acx_lapack_ok=yes;
	     acx_lapack_libs=$LAPACK_LIBS])
	AC_MSG_RESULT($acx_lapack_ok)
	LIBS="$save_LIBS"
fi
fi

# Intel MKL LAPACK?
if test $acx_lapack_ok = no; then
	AC_CHECK_LIB(mkl_lapack, $dsyev,
	[acx_lapack_ok=yes; acx_lapack_libs="-lmkl_lapack"],
	[],[])
fi

# LAPACK linked to by default?  (is sometimes included in BLAS lib)
if test $acx_lapack_ok = no; then
	AC_MSG_CHECKING([for $dsyev in BLAS library])
	AC_TRY_LINK_FUNC($dsyev, [acx_lapack_ok=yes; acx_lapack_libs=""])
	AC_MSG_RESULT($acx_lapack_ok)
fi

# Generic LAPACK library?
if test $acx_lapack_ok = no; then
	AC_CHECK_LIB(lapack, $dsyev,
		[acx_lapack_ok=yes; acx_lapack_libs="-llapack"], [], [])
fi

LAPACK_LIBS="$acx_lapack_libs"
LIBS="$acx_lapack_save_LIBS"
LDFLAGS="$acx_lapack_save_LDFLAGS $acx_lapack_dir"

AC_SUBST(LAPACK_LIBS)

# Finally, execute ACTION-IF-FOUND/ACTION-IF-NOT-FOUND:
if test x"$acx_lapack_ok" = xyes; then
        ifelse([$1],,AC_DEFINE(HAVE_LAPACK,1,[Define if you have LAPACK library.]),[$1])
        :
else
        acx_lapack_ok=no
        $2
fi
])dnl ACX_LAPACK
