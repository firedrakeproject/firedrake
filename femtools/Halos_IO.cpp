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

#include "Halos_IO.h"

using namespace std;

using namespace Fluidity;

HaloReadError Fluidity::ReadHalos(const string& filename, int& process, int& nprocs, map<int, int>& npnodes, map<int, vector<vector<int> > >& send, map<int, vector<vector<int> > >& recv){
  // Read the halo file
  TiXmlDocument doc(filename);
  if(!doc.LoadFile()){
    doc.ErrorDesc();
    return HALO_READ_FILE_NOT_FOUND;
  }

  const char* charBuffer;
  ostringstream buffer;

  // Extract the XML header
  TiXmlNode* header = doc.FirstChild();
  while(header != NULL and header->Type() != TiXmlNode::DECLARATION){
    header = header->NextSibling();
  }

  // Extract the root node
  TiXmlNode* rootNode = header->NextSiblingElement();
  if(rootNode == NULL){
    return HALO_READ_FILE_INVALID;
  }
  TiXmlElement* rootEle = rootNode->ToElement();

  // Extract process
  charBuffer = rootEle->Attribute("process");
  if(charBuffer == NULL){
    return HALO_READ_FILE_INVALID;
  }
  process = atoi(charBuffer);
  if(process < 0){
    return HALO_READ_FILE_INVALID;
  }

  // Extract nprocs
  charBuffer = rootEle->Attribute("nprocs");
  if(charBuffer == NULL){
    return HALO_READ_FILE_INVALID;
  }
  nprocs = atoi(charBuffer);
  if(process >= nprocs){
    return HALO_READ_FILE_INVALID;
  }

  // Extract halo data for each process for each level
  npnodes.clear();
  send.clear();
  recv.clear();
  // Find the next halo element
  for(TiXmlNode* haloNode = rootEle->FirstChildElement("halo");haloNode != NULL;haloNode = haloNode->NextSiblingElement("halo")){
    if(haloNode == NULL){
      break;
    }
    TiXmlElement* haloEle = haloNode->ToElement();

    // Extract the level
    charBuffer = haloEle->Attribute("level");
    if(charBuffer == NULL){
      // Backwards compatibility
      charBuffer = haloEle->Attribute("tag");
      if(charBuffer == NULL){
        return HALO_READ_FILE_INVALID;
      }
    }
    int level = atoi(charBuffer);
    send[level] = vector<vector<int> >(nprocs);
    recv[level] = vector<vector<int> >(nprocs);

    // Extract n_private_nodes
    charBuffer = haloEle->Attribute("n_private_nodes");
    if(charBuffer == NULL){
      return HALO_READ_FILE_INVALID;
    }
    npnodes[level] = atoi(charBuffer);
    if(npnodes[level] < 0){
      return HALO_READ_FILE_INVALID;
    }

    // Find the next halo_data element
    for(TiXmlNode* dataNode = haloEle->FirstChildElement("halo_data");dataNode != NULL;dataNode = dataNode->NextSiblingElement("halo_data")){
      if(dataNode == NULL){
        break;
      }
      TiXmlElement* dataEle = dataNode->ToElement();

      // Extract the process
      charBuffer = dataEle->Attribute("process");
      if(charBuffer == NULL){
        return HALO_READ_FILE_INVALID;
      }
      int proc = atoi(charBuffer);
      if(proc < 0 or proc >= nprocs){
        return HALO_READ_FILE_INVALID;
      }

      // Check that data for this level and process has not already been extracted
      if(send[level][proc].size() > 0 or recv[level][proc].size() > 0){
        return HALO_READ_FILE_INVALID;
      }

      // Permit empty send and receive data elements
      send[level][proc] = vector<int>();
      recv[level][proc] = vector<int>();

      // Extract the send data
      TiXmlNode* sendDataNode = dataEle->FirstChildElement("send");
      if(sendDataNode != NULL){
        TiXmlNode* sendDataTextNode = sendDataNode->FirstChild();
        while(sendDataTextNode != NULL and sendDataTextNode->Type() != TiXmlNode::TEXT){
          sendDataTextNode = sendDataTextNode->NextSibling();
        }
        if(sendDataTextNode != NULL){
          vector<string> tokens;
          Tokenize(sendDataTextNode->ValueStr(), tokens, " ");
          for(size_t i = 0;i < tokens.size();i++){
            send[level][proc].push_back(atoi(tokens[i].c_str()));
          }
        }
      }

      // Extract the receive data
      TiXmlNode* recvDataNode = dataEle->FirstChildElement("receive");
      if(recvDataNode != NULL){
      TiXmlNode* recvDataTextNode = recvDataNode->FirstChild();
        while(recvDataTextNode != NULL and recvDataTextNode->Type() != TiXmlNode::TEXT){
          recvDataTextNode = recvDataTextNode->NextSibling();
        }
        if(recvDataTextNode != NULL){
          vector<string> tokens;
          Tokenize(recvDataTextNode->ValueStr(), tokens, " ");
          for(size_t i = 0;i < tokens.size();i++){
            recv[level][proc].push_back(atoi(tokens[i].c_str()));
          }
        }
      }
    }
  }

  return HALO_READ_SUCCESS;
}

int Fluidity::WriteHalos(const string& filename, const unsigned int& process, const unsigned int& nprocs, const map<int, int>& npnodes, const map<int, vector<vector<int> > >& send, const map<int, vector<vector<int> > >& recv){
#ifdef DDEBUG
  // Input check
  assert(process < nprocs);
  assert(send.size() == recv.size());
  for(map<int, vector<vector<int> > >::const_iterator sendIter = send.begin(), recvIter = recv.begin();sendIter != send.end() and recvIter != recv.end(), recvIter != recv.end();sendIter++, recvIter++){
    assert(recv.count(sendIter->first) != 0);
    assert(npnodes.count(sendIter->first) != 0);
    assert(sendIter->second.size() == recvIter->second.size());
  }
#endif

  TiXmlDocument doc;

  ostringstream buffer;

  // XML header
  TiXmlDeclaration* header = new TiXmlDeclaration("1.0", "utf-8", "");
  doc.LinkEndChild(header);

  // Add root node
  TiXmlElement* rootEle = new TiXmlElement("halos");
  doc.LinkEndChild(rootEle);

  // Add process attribute to root node
  buffer << process;
  rootEle->SetAttribute("process", buffer.str());
  buffer.str("");

  // Add nprocs attribute to root node
  buffer << nprocs;
  rootEle->SetAttribute("nprocs", buffer.str());
  buffer.str("");

  // Add halo data for each level
  map<int, int>::const_iterator npnodesIter = npnodes.begin();
  for(map<int, vector<vector<int> > >::const_iterator sendLevelIter = send.begin(), recvLevelIter = recv.begin();sendLevelIter != send.end() and recvLevelIter != recv.end() and npnodesIter != npnodes.end();sendLevelIter++, recvLevelIter++, npnodesIter++){
    // Add halo element to root element
    TiXmlElement* haloEle = new TiXmlElement("halo");
    rootEle->LinkEndChild(haloEle);

    // Add level attribute to halo element
    buffer << sendLevelIter->first;
    haloEle->SetAttribute("level", buffer.str());
    buffer.str("");

    // Add n_private_nodes attribute to halo element
    buffer << npnodesIter->second;
    haloEle->SetAttribute("n_private_nodes", buffer.str());
    buffer.str("");

    // Add halo data for each process for each level
    int j = 0;
    for(vector<vector<int> >::const_iterator sendProcIter = sendLevelIter->second.begin(), recvProcIter = recvLevelIter->second.begin();sendProcIter != sendLevelIter->second.end() and recvProcIter != recvLevelIter->second.end();sendProcIter++, recvProcIter++, j++){
      if(j == (int)nprocs){
        break;
      }

      // Add halo_data element to halo element
      TiXmlElement* dataEle = new TiXmlElement("halo_data");
      haloEle->LinkEndChild(dataEle);

      // Add process attribute to data element
      buffer << j;
      dataEle->SetAttribute("process", buffer.str());
      buffer.str("");

      // Add send data to data element
      TiXmlElement* sendDataEle = new TiXmlElement("send");
      dataEle->LinkEndChild(sendDataEle);

      // Add data to send data element
      for(vector<int>::const_iterator sendDataIter = sendProcIter->begin();sendDataIter != sendProcIter->end();sendDataIter++){
        buffer << *sendDataIter << " ";
      }
      TiXmlText* sendData = new TiXmlText(buffer.str());
      sendDataEle->LinkEndChild(sendData);
      buffer.str("");

      // Add receive data to data element
      TiXmlElement* recvDataEle = new TiXmlElement("receive");
      dataEle->LinkEndChild(recvDataEle);

      // Add data to receive data element
      for(vector<int>::const_iterator recvDataIter = recvProcIter->begin();recvDataIter != recvProcIter->end();recvDataIter++){
        buffer << *recvDataIter << " ";
      }
      TiXmlText* recvData = new TiXmlText(buffer.str());
      recvDataEle->LinkEndChild(recvData);
      buffer.str("");
    }
  }

  return doc.SaveFile(filename) ? 0 : -1;
}

HaloData* readHaloData = NULL;
HaloData* writeHaloData = NULL;

extern "C"{
  void cHaloReaderReset(){
    if(readHaloData){
      delete readHaloData;
      readHaloData = NULL;
    }

    return;
  }

  int cHaloReaderSetInput(char* filename, int* filename_len, int* process, int* nprocs){
    if(readHaloData){
      delete readHaloData;
      readHaloData = NULL;
    }
    readHaloData = new HaloData();

    ostringstream buffer;
    buffer << string(filename, *filename_len) << "_" << *process << ".halo";
    HaloReadError ret = ReadHalos(buffer.str(),
      readHaloData->process, readHaloData->nprocs,
      readHaloData->npnodes, readHaloData->send, readHaloData->recv);

    int errorCount = 0;
    if(ret == HALO_READ_FILE_NOT_FOUND){
      if(*process == 0){
        cerr << "Error reading halo file " << buffer.str() << "\n"
             << "Zero process file not found" << endl;
        errorCount++;
      }else{
        readHaloData->process = *process;
        readHaloData->nprocs = *nprocs;
        readHaloData->npnodes.clear();
        readHaloData->send.clear();
        readHaloData->recv.clear();
      }
    }else if(ret != HALO_READ_SUCCESS){
      cerr << "Error reading halo file " << buffer.str() << "\n";
      switch(ret){
        case(HALO_READ_FILE_INVALID):
          cerr << "Invalid .halo file" << endl;
          break;
        // HALO_READ_FILE_NOT_FOUND case handled above
        default:
          cerr << "Unknown error" << endl;
          break;
      }
      errorCount++;
    }else if(readHaloData->process != *process){
      cerr << "Error reading halo file " << buffer.str() << "\n"
           << "Unexpected process number in .halo file" << endl;
      errorCount++;
    }else if(readHaloData->nprocs > *nprocs){
      cerr << "Error reading halo file " << buffer.str() << "\n"
           << "Number of processes in .halo file: " << readHaloData->nprocs << "\n"
           << "Number of running processes: " << *nprocs << "\n"
           << "Number of processes in .halo file exceeds number of running processes" << endl;
      errorCount++;
    }

    return errorCount;
  }

  void cHaloReaderQueryOutput(int* level, int* nprocs, int* nsends, int* nreceives){
    assert(readHaloData);
    assert(*nprocs >= readHaloData->nprocs);

    if(readHaloData->send.count(*level) == 0){
      assert(readHaloData->recv.count(*level) == 0);

      for(int i = 0;i < *nprocs;i++){
        nsends[i] = 0;
        nreceives[i] = 0;
      }
    }else{
      assert(readHaloData->recv.count(*level) > 0);

      for(int i = 0;i < readHaloData->nprocs;i++){
        nsends[i] = readHaloData->send[*level][i].size();
        nreceives[i] = readHaloData->recv[*level][i].size();
      }
      for(int i = readHaloData->nprocs;i < *nprocs;i++){
        nsends[i] = 0;
        nreceives[i] = 0;
      }
    }

    return;
  }

  void cHaloReaderGetOutput(int* level, int* nprocs, int* nsends, int* nreceives,
    int* npnodes, int* send, int* recv){

#ifdef DDEBUG
    assert(readHaloData);
    assert(*nprocs >= readHaloData->nprocs);
    int* lnsends = (int*)malloc(*nprocs * sizeof(int));
    assert(lnsends);
    int* lnreceives = (int*)malloc(*nprocs * sizeof(int));
    assert(lnreceives);
    cHaloReaderQueryOutput(level, nprocs, lnsends, lnreceives);
    for(int i = 0;i < *nprocs;i++){
      assert(nsends[i] == lnsends[i]);
      assert(nreceives[i] == lnreceives[i]);
    }
    free(lnsends);
    free(lnreceives);
#endif

    if(readHaloData->send.count(*level) == 0){
#ifdef DDEBUG
      assert(readHaloData->recv.count(*level) == 0);
      for(int i = 0;i < *nprocs;i++){
        assert(nsends[i] == 0);
        assert(nreceives[i] == 0);
      }
#endif
    }else{
      assert(readHaloData->recv.count(*level) > 0);

      int sendIndex = 0, recvIndex = 0;;
      for(int i = 0;i < readHaloData->nprocs;i++){
        for(int j = 0;j < nsends[i];j++){
          send[sendIndex] = readHaloData->send[*level][i][j];
          sendIndex++;
        }
        for(int j = 0;j < nreceives[i];j++){
          recv[recvIndex] = readHaloData->recv[*level][i][j];
          recvIndex++;
        }
      }
    }

    *npnodes = readHaloData->npnodes[*level];

    return;
    }

  void cHaloWriterReset(){
    if(writeHaloData){
      delete writeHaloData;
      writeHaloData = NULL;
    }

    return;
  }

  void cHaloWriterInitialise(int* process, int* nprocs){
    if(writeHaloData){
      delete writeHaloData;
      writeHaloData = NULL;
    }
    writeHaloData = new HaloData();

    writeHaloData->process = *process;
    writeHaloData->nprocs = *nprocs;

    return;
  }

  void cHaloWriterSetInput(int* level, int* nprocs, int* nsends, int* nreceives,
    int* npnodes, int* send, int* recv){

    assert(writeHaloData);
    assert(writeHaloData->nprocs == *nprocs);

    writeHaloData->send[*level].clear();  writeHaloData->send[*level].resize(*nprocs);
    writeHaloData->recv[*level].clear();  writeHaloData->recv[*level].resize(*nprocs);
    int send_index = 0, recv_index = 0;
    for(int i = 0;i < *nprocs;i++){
      writeHaloData->send[*level][i].resize(nsends[i]);
      for(int j = 0;j < nsends[i];j++){
        writeHaloData->send[*level][i][j] = send[send_index];
        send_index++;
      }
      writeHaloData->recv[*level][i].resize(nreceives[i]);
      for(int j = 0;j < nreceives[i];j++){
        writeHaloData->recv[*level][i][j] = recv[recv_index];
        recv_index++;
      }
    }

    writeHaloData->npnodes[*level] = *npnodes;

    return;
  }

  int cHaloWriterWrite(char* filename, int* filename_len){
    assert(writeHaloData);

    // Write out the halos
    ostringstream buffer;
    buffer << string(filename, *filename_len) << "_" << writeHaloData->process << ".halo";

    return WriteHalos(buffer.str(),
                      writeHaloData->process, writeHaloData->nprocs,
                      writeHaloData->npnodes, writeHaloData->send, writeHaloData->recv);
  }
}
