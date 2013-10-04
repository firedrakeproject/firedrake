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

#ifndef HALOS_IO_H
#define HALOS_IO_H

#include "confdefs.h"
#include "Tokenize.h"

#ifdef HAVE_MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <deque>
#include <iostream>
#include <map>
#include <set>
#include <string.h>
#include <vector>

#include "tinyxml.h"

#ifndef DDEBUG
#ifdef assert
#undef assert
#endif
#define assert
#endif

namespace Fluidity{

  enum HaloReadError{
    HALO_READ_SUCCESS = 0,
    HALO_READ_FILE_NOT_FOUND = -1,
    HALO_READ_FILE_INVALID = -2,
  };

  //* Read halo information
  /** Read from a halo file.
    * \param filename Halo file name
    * \param process The process number
    * \param nprocs The number of processes
    * \param npnodes Number of private nodes, by tag
    * \param send Sends, by tag and process
    * \param recv Receives, by tag and process
    * \return 0 on success, non-zero on failure
    */
  HaloReadError ReadHalos(const std::string& filename, int& process, int& nprocs, std::map<int, int>& npnodes, std::map<int, std::vector<std::vector<int> > >& send, std::map<int, std::vector<std::vector<int> > >& recv);

  //* Write halo information.
  /** Write to a halo file.
    * \param filename Halo file name
    * \param npnodes Number of private nodes, by tag
    * \param send Sends, by tag and process
    * \param recv Receives, by tag and process
    * \return 0 on success, non-zero on failure
    */
  int WriteHalos(const std::string& filename, const unsigned int& process, const unsigned int& nprocs, const std::map<int, int>& npnodes, const std::map<int, std::vector<std::vector<int> > >& send, const std::map<int, std::vector<std::vector<int> > >& recv);

  struct HaloData{
      int process, nprocs;
      std::map<int, int> npnodes;
      std::map<int, std::vector<std::vector<int> > > send, recv;
  };
}

extern Fluidity::HaloData* readHaloData;
extern Fluidity::HaloData* writeHaloData;

extern "C"{
#define cHaloReaderReset F77_FUNC(chalo_reader_reset, CHALO_READER_RESET)
  void cHaloReaderReset();

#define cHaloReaderSetInput F77_FUNC(chalo_reader_set_input, CHALO_READER_SET_INPUT)
  int cHaloReaderSetInput(char* filename, int* filename_len, int* process, int* nprocs);

#define cHaloReaderQueryOutput F77_FUNC(chalo_reader_query_output, CHALO_READER_QUERY_OUTPUT)
  void cHaloReaderQueryOutput(int* level, int* nprocs, int* nsends, int* nreceives);

#define cHaloReaderGetOutput F77_FUNC(chalo_reader_get_output, CHALO_READER_GET_OUTPUT)
  void cHaloReaderGetOutput(int* level, int* nprocs, int* nsends, int* nreceives,
    int* npnodes, int* send, int* recv);

#define cHaloWriterReset F77_FUNC(chalo_writer_reset, CHALO_WRITER_RESET)
  void cHaloWriterReset();

#define cHaloWriterInitialise F77_FUNC(chalo_writer_initialise, CHALO_WRITER_INITIALISE)
  void cHaloWriterInitialise(int* process, int* nprocs);

#define cHaloWriterSetInput F77_FUNC(chalo_writer_set_input, CHALO_WRITER_SET_INPUT)
  void cHaloWriterSetInput(int* level, int* nprocs, int* nsends, int* nreceives,
    int* npnodes, int* send, int* recv);

#define cHaloWriterWrite F77_FUNC(chalo_writer_write, CHALO_WRITER_WRITE)
  int cHaloWriterWrite(char* filename, int* filename_len);
}

#endif
