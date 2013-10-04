/* Copyright (C) 2010- Imperial College London and others.

   Please see the AUTHORS file in the main source directory for a full
   list of copyright holders.

   Dr Gerard J Gorman
   Applied Modelling and Computation Group
   Department of Earth Science and Engineering
   Imperial College London

   g.gorman@imperial.ac.uk

   This library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
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

#include <map>
#include <string>
#include <iostream>

#include "confdefs.h"

#ifdef HAVE_LIBNUMA
#include <numa.h>
#include <numaif.h>
#include <sys/resource.h>
#endif

#ifdef HAVE_MPI
#include <mpi.h>
#endif

class Profiler{
 public:
  Profiler();
  ~Profiler();

  double get(const std::string&) const;
  void print() const;
  void tic(const std::string&);
  void toc(const std::string&);
  void zero();
  void zero(const std::string&);
  int minorpagefaults();
  int majorpagefaults();
  int getresidence(void *ptr);

#ifdef HAVE_LIBNUMA
 public:
  struct rusage usage;
#endif

private:
  double wall_time() const;
  std::map< std::string, std::pair<double, double> > timings;

};

extern Profiler flprofiler;
