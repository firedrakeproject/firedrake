/*  Copyright (C) 2006 Imperial College London and others.

    Please see the AUTHORS file in the main source directory for a full list
    of copyright holders.

    Prof. C Pain
    Applied Modelling and Computation Group
    Department of Earth Science and Engineering
    Imperial College London

    amcgsoftware@imperial.ac.uk

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation,
    version 2.1 of the License.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307
    USA
*/

#include <stdlib.h>
#include <unistd.h>

#include "confdefs.h"
#include "fmangle.h"
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#ifdef HAVE_PETSC
#include "petsc.h"
#endif



extern "C" {

#ifdef HAVE_PYTHON
#include "python_statec.h"
#endif
  void TESTNAME();
  void set_global_debug_level_fc(int *val);
  void set_pseudo2d_domain_fc(int* val);
}

int main(int argc, char **argv)
{
  int val = 0;

  set_global_debug_level_fc(&val);
  set_pseudo2d_domain_fc(&val);
#ifdef HAVE_MPI
  MPI::Init(argc, argv);
  // Undo some MPI init shenanigans
  chdir(getenv("PWD"));
#endif
#ifdef HAVE_PETSC
  PetscInitialize(&argc, &argv, NULL, PETSC_NULL);
  // PetscInitializeFortran needs to be called when initialising PETSc from C, but calling it from Fortran
  // This sets all kinds of objects such as PETSC_NULL_OBJECT, PETSC_COMM_WORLD, etc., etc.
  PetscInitializeFortran();
#endif

#ifdef HAVE_PYTHON
  // Initialize the Python Interpreter
  python_init_();
#endif
  TESTNAME();

#ifdef HAVE_PYTHON
  // Finalize the Python Interpreter
  python_end_();
#endif
#ifdef HAVE_PETSC
  PetscFinalize();
#endif
#ifdef HAVE_MPI
  MPI::Finalize();
#endif

  return 0;

}
