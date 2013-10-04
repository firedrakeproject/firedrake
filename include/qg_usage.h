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

#ifndef COMMAND_LINE_OPTIONS_H
#define COMMAND_LINE_OPTIONS_H
#include "confdefs.h"
#include "version.h"

#include "Tokenize.h"

#ifdef _AIX
#include <unistd.h>
#else
#include <getopt.h>
#endif

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#ifdef HAVE_PETSC
#include <petsc.h>
#endif

#include <signal.h>

#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "fmangle.h"

extern std::map<std::string, std::string> fl_command_line_options;

void ParseArguments(int argc, char** argv);
void PetscInit(int argc, char** argv);
void print_version(std::ostream& stream = std::cerr);
void qg_usage(char *cmd);

extern "C" {
  void set_global_debug_level_fc(int *val);
}

#endif
