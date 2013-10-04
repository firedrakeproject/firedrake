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
#include <confdefs.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <cstdlib>
#include <cstdio>
#include <ctime>

#include <unistd.h>
#include <pthread.h>
#include <sys/stat.h>

using namespace std;

//#define MSG cout
#define MSG if(false) cout

typedef struct {
#ifdef DOUBLEP
  const double *x, *y, *z;
#else
  const float *x, *y, *z;
#endif
  int count;
  const char *input;
} w_args;

typedef struct {
  const char *generic, *input, *output;
  int stat;
} e_args;

typedef struct {
#ifdef DOUBLEP
  double *scalar;
#else
  float *scalar;
#endif
  int count;
  const char *output;
} scalar_r_args;

void *generic_writer(void *args){
  MSG<<"0: void *generic_writer(void *args)\n";

  w_args *a = (w_args *)args;

  MSG<<"0: Write input to pipe - "<<a->input<<endl;
  fstream in;
  in.open(a->input,ios_base::out);
  for(int i=0;i<a->count;i++){
    in<<a->x[i]<<" "<<a->y[i]<<" "<<a->z[i]<<endl;
  }
  in.close();

  MSG<<"0: Finished writing\n";
}

void *scalar_generic_reader(void *args){
  MSG<<"1: void *scalar_generic_reader(void *args)\n";
  scalar_r_args *a = (scalar_r_args *)args;

  MSG<<"1: Read output through pipe - "<<a->output<<endl;
  fstream out;
  out.open(a->output,ios_base::in);
  for(int i=0;i<a->count;i++){
    out>>a->scalar[i];
  }
  out.close();
  MSG<<"1: Finished reading\n";
}

void *generic_exec(void *args){
  MSG<<"2: void *generic_exec(void *args)\n";

  e_args *a = (e_args *)args;

  MSG<<"2: Run generic function:\n";
  char command[4098];
  sprintf(command, "%s < %s > %s", a->generic, a->input, a->output);
  MSG<<command<<endl;
  a->stat = system(command);

  MSG<<"Finished system()\n";
}

extern "C" {
#define set_from_external_function_scalar_fc F77_FUNC(set_from_external_function_scalar, SET_FROM_EXTERNAL_FUNCTION_SCALAR)

  void set_from_external_function_scalar_fc(const char *fgeneric, const int *generic_len, const int *count,
#ifdef DOUBLEP
                                            const double *x, const double *y, const double *z,
                                            double *scalar,
#else
                                            const float *x, const float *y, const float *z,
                                            float *scalar,
#endif
                                            int *stat){
    // Fix fortran string
    string generic(fgeneric, *generic_len);

    // Create a unique fifo's
    string output(tmpnam(NULL));
    mkfifo(output.c_str(), S_IWUSR|S_IRUSR);

    string input(tmpnam(NULL));
    mkfifo(input.c_str(), S_IWUSR|S_IRUSR);

    pthread_t wthread;
    w_args wargs;
    wargs.x = x;
    wargs.y = y;
    wargs.z = z;
    wargs.count = *count;
    wargs.input = input.c_str();
    pthread_create (&wthread, NULL, generic_writer, &wargs);

    pthread_t rthread;
    scalar_r_args rargs;
    rargs.scalar = scalar;
    rargs.count = *count;
    rargs.output = output.c_str();
    pthread_create (&rthread, NULL, scalar_generic_reader, &rargs);

    pthread_t ethread;
    e_args eargs;
    eargs.generic = generic.c_str();
    eargs.input = input.c_str();
    eargs.output = output.c_str();
    pthread_create (&ethread, NULL, generic_exec, &eargs);

    pthread_join (wthread, NULL);
    pthread_join (rthread, NULL);
    pthread_join (ethread, NULL);

    *stat = eargs.stat;

    // Delete fifo's
    remove(output.c_str());
    remove(input.c_str());
  }
}

#ifdef GENERIC_FUNCTIONS_MAIN
int main(int argc, char **argv){
  char *generic="/home/gormo/sims/chimney-v2/init_temp";
  double x[]={0, 1, 2, 3, 4};
  double y[]={5, 6, 7, 8, 9};
  double z[]={8, 7, 6, 5, 4};
  double scalar[5];

  int count=5;
  set_from_external_function_scalar_fc(generic, &count,
                                      x, y, z, scalar);

  for(size_t i=0;i<count;i++)
    cout<<x[i]<<", "<<y[i]<<", "<<z[i]<<", "<<scalar[i]<<endl;
}

#endif
