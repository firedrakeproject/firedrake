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
#include "confdefs.h"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include "fmangle.h"
#include "c++debug.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#ifdef _GNU_SOURCE
#include <execinfo.h>
#endif

using namespace std;

static std::map<int, double> totaltime;
static std::map<int, double> starttime;

void print_backtrace(){
#ifdef _GNU_SOURCE
  cerr<<"Backtrace will follow if it is available:\n";

  void *bt[40];
  size_t btsize = backtrace(bt, 40);

  char **symbls = backtrace_symbols (bt, btsize);
  if(symbls!=NULL){
    for(size_t i=0;i<btsize;i++){
      cerr<<symbls[i]<<endl;
    }
    free(symbls);
  }

  cerr<<"Use addr2line -e <binary> <address> to decipher.\n";
#endif
  return;
}

void FLAbort(const char *ErrorStr, const char *FromFile, int LineNumber){
  cerr<<"*** FLUIDITY ERROR ***\n"
      <<"Source location: ("<<FromFile<<", "<<LineNumber<<")\n"
      <<"Error message: "<<ErrorStr<<endl;

  print_backtrace();

  cerr<<"Error is terminal.";
#ifdef HAVE_MPI
  MPI::COMM_WORLD.Abort(MPI::ERR_OTHER);
#endif
}

void FLExit(const char *ErrorStr, const char *FromFile, int LineNumber){
  cerr<<"*** ERROR ***\n"
#ifndef NDEBUG
      <<"Source location: ("<<FromFile<<", "<<LineNumber<<")\n"
#endif
      <<"Error message: "<<ErrorStr<<endl;
#ifdef HAVE_MPI
  MPI::COMM_WORLD.Abort(MPI::ERR_OTHER);
#endif
}

extern "C"{
  void fprint_backtrace_fc(){
#ifdef _GNU_SOURCE
    void *bt[40];
    size_t btsize = backtrace(bt, 40);

    char **symbls = backtrace_symbols (bt, btsize);
    if(symbls!=NULL){
      for(size_t i=0;i<btsize;i++){
  cerr<<symbls[i]<<endl;
      }
      free(symbls);
    }
    return;
#endif
  }

}

