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


#include "fldecomp.h"



void usage(char *binary){
  cerr<<"Usage: "<<binary<<" [OPTIONS] -n nparts file\n"
      <<"\t-c,--cores <number of cores per node>\n\t\tApplies hierarchical partitioning.\n"
      <<"\t-d,--diagnostics\n\t\tPrint out partition diagnostics.\n"
      <<"\t-f,--file <file name>\n\t\tInput file (can alternatively specify as final "
      <<"argument)\n"
      <<"\t-h,--help\n\t\tPrints out this message\n"
      <<"\t-k.--kway\n\t\tPartition a graph into k equal-size parts using the "
      <<"multilevel k-way partitioning algorithm (METIS PartGraphKway). This "
      <<"is the default if the number of partitions is greater than 8.\n"
      <<"\t-n,--nparts <number of partitions>\n\t\tNumber of parts\n"
      <<"\t-r,--recursive\n\t\tPartition a graph into k equal-size parts using multilevel recursive "
      <<"bisection (METIS PartGraphRecursive). This is the default if num partitions "
      <<"is less or equal to 8.\n"
      <<"\t-t,--terreno [boundary id]\n\t\tRecognise as a Terreno output that is extruded in the Z direction. By default the boundary id of the extruded plane is 1.\n"
      <<"\t-s,--shell\n\t\tRecognise the input as a spherical shell mesh.\n"
      <<"\t-v,--verbose\n\t\tVerbose mode\n"
      <<"\t-m,--meshformat\n\t\t(optional) specify mesh format (eg. triangle, gmsh)\n";
  exit(-1);
}



// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------



int main(int argc, char **argv){
  // Get any command line arguments
  // reset optarg so we can detect changes
#ifndef _AIX
  struct option longOptions[] = {
    {"cores", 0, 0, 'c'},
    {"diagnostics", 0, 0, 'd'},
    {"file", 0, 0, 'f'},
    {"help", 0, 0, 'h'},
    {"kway", 0, 0, 'k'},
    {"nparts", 0, 0, 'n'},
    {"recursive", 0, 0, 'r'},
    {"terreno", optional_argument, 0, 't'},
    {"shell", 0, 0, 's'},
    {"verbose", 0, 0, 'v'},
    {"meshformat", 0, 0, 'm'},
    {0, 0, 0, 0}
  };
#endif

  // Always recommend using flredecomp
  cerr<<"\n"
      <<"\t*** fldecomp will be removed in a future release of Fluidity. ***\n"
      <<"\n"
      <<"\tflredecomp is the recommended tool for mesh decomposition.\n"
      <<"\n"
      <<"\tReplace the fldecomp workflow:\n"
      <<"\n"
      <<"\t<generate mesh>\n"
      <<"\tfldecomp -n <nparts> <other options> mesh\n"
      <<"\tmpiexec -n <nparts> fluidity <other options> foo.flml\n"
      <<"\n"
      <<"\twith the flredecomp workflow:\n"
      <<"\n"
      <<"\t<generate mesh>\n"
      <<"\tmpiexec -n <nparts> flredecomp -i 1 -o <nparts> foo foo_flredecomp\n"
      <<"\tmpiexec -n <nparts> fluidity <other options> foo_flredecomp.flml\n"
      <<"\n"
      <<"\tfldecomp is retained only to decompose Terreno meshes, which\n"
      <<"\tflredecomp cannot process.\n"
      <<"\n";

  int optionIndex = 0;

  optarg = NULL;
  char c;
  map<char, string> flArgs;
  while (true){
#ifndef _AIX
    c = getopt_long(argc, argv, "c:df:hkn:rt::s::vm:", longOptions, &optionIndex);
#else
    c = getopt(argc, argv, "c:df:hkn:rt::s::vm:");
#endif
    if (c == -1) break;

    if (c != '?'){
      if(c == 't')
        flArgs[c] = (optarg == NULL) ? "1" : optarg;
      else if (optarg == NULL){
        flArgs[c] = "true";
      }else{
        flArgs[c] = optarg;
      }
    }else{
      if (isprint(optopt)){
        cerr << "Unknown option " << optopt << endl;
      }else{
        cerr << "Unknown option " << hex << optopt << endl;
      }
      usage(argv[0]);
      exit(-1);
    }
  }

  // Help?
  if(flArgs.count('h')||(flArgs.count('n')==0)){
    usage(argv[0]);
    exit(-1);
  }

  // What to do with stdout?
  bool verbose=false;
  int val=3;
  if(flArgs.count('v')){
    verbose = true;
    cout<<"Verbose mode enabled.\n";
  }else{
    val = 0;
  }
  set_global_debug_level_fc(&val);

  if(!flArgs.count('f')){
    if(argc>optind+1){
      flArgs['f'] = argv[optind+1];
    }else if(argc==optind+1){
      flArgs['f'] = argv[optind];
    }
  }

  string filename = flArgs['f'];

  int nparts = atoi(flArgs['n'].c_str());
  if(nparts<2){
    cerr<<"ERROR: number of partitions requested is less than 2. Please check your usage and try again.\n";
    usage(argv[0]);
    exit(-1);
  }
  int ncores = 0;
  if(flArgs.count('c')){
    ncores = atoi(flArgs['c'].c_str());
    if(nparts%ncores){
      cerr<<"ERROR: The number of partitions must be some multiple of the number of cores\n";
      exit(-1);
    }
  }

  // Get mesh format (optional - defaults to triangle if not specified)

  string file_format;

  if(flArgs.count('m'))
      file_format = flArgs['m'].c_str();
  else {
    cout << "fldecomp: defaulting to triangle format\n";
    file_format="triangle";
  }

  // Read in the mesh
  if(verbose)
    cout<<"Reading in mesh file with base name "<<filename<<"\n";


  int exitVal;

  // Call either decomp or GMSH decomposition routines

  if( string(file_format) == "triangle" )
    exitVal = decomp_triangle( flArgs, verbose, filename, file_format,
                                       nparts, ncores );
  else if( string(file_format) == "gmsh" )
    exitVal = decomp_gmsh( flArgs, verbose, filename, file_format,
                                   nparts, ncores );
  else
    {
      cerr<<"ERROR: file format not supported\n";
      exitVal=1;
    }

  exit(0);
}


