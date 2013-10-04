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

#include "Profiler.h"

using namespace std;

Profiler::Profiler(){
}

Profiler::~Profiler(){}

double Profiler::get(const std::string &key) const{
  double time = timings.find(key)->second.second;
  if(MPI::Is_initialized()){
    double gtime;
    MPI::COMM_WORLD.Reduce(&time, &gtime, 1, MPI::DOUBLE, MPI::MAX, 0);
    return gtime;
  }

  return time;
}

void Profiler::print() const{
  bool print;
  double val;
  print = !MPI::Is_initialized() || MPI::COMM_WORLD.Get_rank() == 0;
  for(map< string, pair<double, double> >::const_iterator it=timings.begin();it!=timings.end();++it){
    val = get(it->first);
    if ( print ) {
      cout<<it->first<<" :: "<<val<<endl;
    }
  }
}

void Profiler::tic(const std::string &key){
  timings[key].first = wall_time();
}

void Profiler::toc(const std::string &key){
  timings[key].second += wall_time() - timings[key].first;
}

double Profiler::wall_time() const{
#ifdef HAVE_MPI
  return MPI::Wtime();
#else
  return 0.0;
#endif
}

void Profiler::zero(){
  for(map< string, pair<double, double> >::iterator it=timings.begin();it!=timings.end();++it){
    it->second.second = 0.0;
  }
}

void Profiler::zero(const std::string &key){
  timings[key].second = 0.0;
}

int Profiler::minorpagefaults(){
  int faults = -99;
#ifdef HAVE_LIBNUMA
  getrusage(RUSAGE_SELF, &flprofiler.usage);
  faults = flprofiler.usage.ru_minflt;
#endif
  return faults;
}

int Profiler::majorpagefaults(){
  int faults = -99;
#ifdef HAVE_LIBNUMA
  getrusage(RUSAGE_SELF, &flprofiler.usage);
  faults = flprofiler.usage.ru_majflt;
#endif
  return faults;
}


int Profiler::getresidence(void *ptr){
  int residence=-99;
#ifdef HAVE_LIBNUMA
  int mode;
  size_t page_size = getpagesize();
  size_t page_id = (size_t)ptr/page_size;
  /* round memory address down to start of page */
  void *start_of_page =  (void *)(page_id*page_size);

  /* If flags  specifies  both MPOL_F_NODE and MPOL_F_ADDR,
   * get_mempolicy() will return the node ID of the node on
   * which the address of the start of the page is allocated
   * into the location pointed to by mode
   */
  unsigned long flags = MPOL_F_NODE|MPOL_F_ADDR;

  // if(get_mempolicy(&mode, NULL, 0, start_of_page, flags)){
  //   perror("get_mempolicy()");
  // }
  // residence = mode;
#endif
  return residence;
}

// Opaque instances of profiler.
Profiler flprofiler;

// Fortran interface
extern "C" {
#define cprofiler_get_fc F77_FUNC(cprofiler_get, CPROFILER_GET)
  void cprofiler_get_fc(const char *key, const int *key_len, double *time){
    *time = flprofiler.get(string(key, *key_len));
  }

#define cprofiler_tic_fc F77_FUNC(cprofiler_tic, CPROFILER_TIC)
  void cprofiler_tic_fc(const char *key, const int *key_len){
    flprofiler.tic(string(key, *key_len));
  }

#define cprofiler_toc_fc F77_FUNC(cprofiler_toc, CPROFILER_TOC)
  void cprofiler_toc_fc(const char *key, const int *key_len){
    flprofiler.toc(string(key, *key_len));
  }

#define cprofiler_zero_fc F77_FUNC(cprofiler_zero, CPROFILER_ZERO)
  void cprofiler_zero_fc(){
    flprofiler.zero();
  }

#define cprofiler_minorpagefaults_fc F77_FUNC(cprofiler_minorpagefaults, CPROFILER_MINORPAGEFAULTS)
  void cprofiler_minorpagefaults_fc(int *faults){
    *faults = flprofiler.minorpagefaults();
  }

#define cprofiler_majorpagefaults_fc F77_FUNC(cprofiler_majorpagefaults, CPROFILER_MAJORPAGEFAULTS)
  void cprofiler_majorpagefaults_fc(int *faults){
    *faults = flprofiler.majorpagefaults();
  }

#define cprofiler_getresidence_fc F77_FUNC(cprofiler_getresidence, CPROFILER_GETRESIDENCE)
  void cprofiler_getresidence_fc(void *ptr, int *residence){
    *residence = flprofiler.getresidence(ptr);
  }
}
