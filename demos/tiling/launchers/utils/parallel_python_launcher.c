#include <Python.h>
#include <mpi.h>
#include <stdio.h>

#ifdef __FreeBSD__
#include <floatingpoint.h>
#endif

int main(int argc, char **argv)
{
  int return_code;

  /* 754 requires that FP exceptions run in "no stop" mode by default,
   * and until C vendors implement C99's ways to control FP exceptions,
   * Python requires non-stop mode.  Alas, some platforms enable FP
   * exceptions by default.  Here we disable them.
   */
#ifdef __FreeBSD__
  fp_except_t m;

  m = fpgetmask();
  fpsetmask(m & ~FP_X_OFL);
#endif

  MPI_Init(&argc, &argv);
  MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

  Py_Initialize();
  return_code = Py_Main(argc, argv);

  MPI_Finalize();
  return return_code;
}
