#define ALLOW_IMPORT_ARRAY
#include "python_statec.h"

int python_initialised = 0;

void python_init_(void){
#ifdef HAVE_PYTHON
  // Initialize the Python interpreter
  Py_Initialize();
  PyRun_SimpleString("import string");

  PyObject* m;
  m = Py_InitModule("spud_manager", NULL);
  assert(m != NULL);

#if PY_MINOR_VERSION > 6
  void* manager = spud_get_manager();
  PyObject* manager_capsule = PyCapsule_New(manager, "spud_manager._spud_manager", NULL);
  assert(manager_capsule != NULL);

  PyModule_AddObject(m, "_spud_manager", manager_capsule);
#endif

#endif
#ifdef HAVE_NUMPY
  // Enable use of NumPy arrays in C
  import_array();

  // Import the NumPy module in our Python interpreter
  if(PyRun_SimpleString("import numpy") == -1)
    fprintf(stderr,"Error: Importing the NumPy module failed.\n");

  // Initialize a persistent dictionary
  PyRun_SimpleString("persistent = {}");

  // Initialize Fluidity field caches
  PyRun_SimpleString("mesh_cache = {}");
  PyRun_SimpleString("halos_cache = {}");
  PyRun_SimpleString("scalar_field_cache = {}");
  PyRun_SimpleString("vector_field_cache = {}");
  PyRun_SimpleString("tensor_field_cache = {}");

  // Add the working directory to the module search path.
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append('.')");

  init_vars();
#endif

  python_initialised = 1;
}


void init_vars(void){
#ifdef HAVE_PYTHON
  if(PyRun_SimpleString("from fluidity.state_types import *") == -1){
    fprintf(stderr, "Warning: The 'state_types.py' module could not be loaded. Make sure the PYTHONPATH environment variable is set.\n");
    fprintf(stderr, "This is a problem if you have ocean biology or Python diagnostic fields.\n");
    fprintf(stderr, "It will not otherwise affect the running of Fluidity.\n");
    PyErr_Clear();
  } else {
    if (get_global_debug_level_() > 1) {
      printf("fluidity.state_types imported successfully; location: \n");
      PyRun_SimpleString("import fluidity.state_types; print fluidity.state_types.__file__");
    }
  }
  PyRun_SimpleString("states = dict()");
#endif
}


void python_reset_(void){
#ifdef HAVE_PYTHON
  if(Py_IsInitialized()){
    // Create a list of items to be kept
    PyRun_SimpleString("keep = ['keep', 'rem', '__builtins__', '__name__', '__doc__', 'string', 'numpy', 'persistent', 'mesh_cache', 'halos_cache', 'scalar_field_cache', 'vector_field_cache', 'tensor_field_cache', 'states']");

    // Create a list of items to  be removed
    PyRun_SimpleString("rem = []");
    PyRun_SimpleString("for i in globals().keys():\n if(not (i in keep)): rem.append(i)");

    // Delete every item except the ones we want to keep
    PyRun_SimpleString("for i in rem: del globals()[i]");
    PyRun_SimpleString("del globals()['keep'];del globals()['rem'];del globals()['i']");

    // Reinitialize the variables
    init_vars();

    // And run a garbage collection
    PyGC_Collect();
  }
#endif
}


void python_end_(void){
#ifdef HAVE_PYTHON
  if(Py_IsInitialized()){
    // Garbage collection
    PyGC_Collect();
    // Finalize the Python interpreter
    Py_Finalize();
  }
#endif
}


void python_run_stringc_(char *s,int *slen, int *stat){
#ifdef HAVE_PYTHON

  if (!python_initialised)
    python_init_();

  // Run a python command from Fortran
  char *c = fix_string(s,*slen);
  int tlen=8+*slen;
  char t[tlen];
  snprintf(t, tlen, "%s\n",c);
  *stat = PyRun_SimpleString(t);
  if(*stat != 0){
    PyErr_Print();
  }
  free(c);
#endif
}


void python_run_filec_(char *f,int *flen, int *stat){
#ifdef HAVE_PYTHON
  // Run a python file from Fortran
  char *filename = fix_string(f,*flen);
  FILE *pyfile;
  if ((pyfile = fopen(filename, "r")) == NULL){
    fprintf(stderr, "Error: cannot open '%s'. \n", filename);
    *stat = 1;
  }else {
    *stat = PyRun_SimpleFileExFlags(pyfile,filename,1,NULL);
    if(*stat != 0){
      PyErr_Print();
    }
  }
  free(filename);
#endif
}


char* fix_string(char *s,int len){
  char *ns = (char *)malloc(len+3);
  memcpy( ns, s, len );
  ns[len] = 0;
  return ns;
}


// Functions to add a state and fields: scalar, vector, tensor, mesh, quadrature, polynomial


void python_add_statec_(char *name,int *len){
#ifdef HAVE_PYTHON
  // Add a new state object to the Python environment
  char *n = fix_string(name,*len);
  int tlen=23+2*(*len);
  char t[tlen];
  // 'state' in Python will always be the last state added while the 'states' dictionary
  // includes all added states
  snprintf(t, tlen, "states[\"%s\"] = State(\"%s\")",n,n);
  PyRun_SimpleString(t);
  snprintf(t, tlen, "state = states[\"%s\"]",n);
  PyRun_SimpleString(t);

  free(n);
#endif
}


void python_add_scalar_(int *sx,double x[],char *name,int *nlen, int *field_type,
  char *option_path, int *oplen, char *state,int *slen,
  char *mesh_name, int *mesh_name_len, int *uid){
#ifdef HAVE_NUMPY
  // Add the Fortran scalar field to the dictionary of the Python interpreter
  PyObject *pMain = PyImport_AddModule("__main__");
  PyObject *pDict = PyModule_GetDict(pMain);
  // Fix the Fortran strings for C and Python
  char *namec = fix_string(name,*nlen);
  char *opc = fix_string(option_path,*oplen);
  char *meshc = fix_string(mesh_name,*mesh_name_len);

  // Create the array
  python_add_array_double_1d(x,sx,"s");

  PyObject *pname = PyString_FromString(namec);
  PyDict_SetItemString(pDict,"n",pname);
  PyObject *poptionp = PyString_FromString(opc);
  PyDict_SetItemString(pDict,"op",poptionp);
  PyObject *pft = PyInt_FromLong(*field_type);
  PyDict_SetItemString(pDict,"ft",pft);

  PyRun_SimpleString("n = string.strip(n)");
  PyRun_SimpleString("op = string.strip(op)");

  char *n = fix_string(state,*slen);
  int tlen=140+2**slen+*nlen+*mesh_name_len;
  char t[tlen];
  snprintf(t, tlen,
      "field = scalar_field_cache.setdefault(%d, ScalarField(n,s,ft,op,%d,states['%s'].meshes['%s']))",
      *uid, *uid, n, meshc);
  PyRun_SimpleString(t);
  snprintf(t, tlen, "states['%s'].scalar_fields['%s'] = field",n,namec);
  PyRun_SimpleString(t);

  // Clean up
  PyRun_SimpleString("del n; del op; del ft; del s; del field");
  free(namec);
  free(opc);
  free(n);
  free(meshc);
  Py_DECREF(pname);
  Py_DECREF(poptionp);
  Py_DECREF(pft);
#endif
}
enum valtype_t { CSR_REAL, CSR_INTEGER };

static void python_add_csr_matrix(int *valSize, void *val, int *col_indSize, int *col_ind, int *row_ptrSize,
                                  int *row_ptr, char *name, int *namelen, char *state, int *statelen, int *numCols,
                                  enum valtype_t valtype)
{
#ifdef HAVE_NUMPY
  // Add the Fortran csr matrix to the dictionary of the Python interpreter
  PyObject *pMain = PyImport_AddModule("__main__");
  PyObject *pDict = PyModule_GetDict(pMain);

  // Fix the Fortran strings for C and Python
  char *namefixed = fix_string(name,*namelen);
  PyObject *pnumCols = PyInt_FromLong(*numCols);
  PyDict_SetItemString(pDict,"numCols",pnumCols);
  PyObject *pnumRows = PyInt_FromLong((*row_ptrSize) - 1);
  PyDict_SetItemString(pDict,"numRows",pnumRows);

  // Create the array
  switch (valtype) {
  case CSR_REAL:
      python_add_array_double_1d((double *)val, valSize, "val");
      break;
  case CSR_INTEGER:
      python_add_array_integer_1d((int *)val, valSize, "val");
  }
  python_add_array_integer_1d(col_ind,col_indSize,"col_ind");
  python_add_array_integer_1d(row_ptr,row_ptrSize,"row_ptr");

  PyObject *pname = PyString_FromString(namefixed);
  PyDict_SetItemString(pDict,"name",pname);

  PyRun_SimpleString("name = string.strip(name)");

  char *statefixed = fix_string(state,*statelen);
  int tlen=150+*statelen;
  char t[tlen];

  snprintf(t, tlen, "matrix = CsrMatrix((val,col_ind - 1,row_ptr - 1), shape=(numRows,numCols)); states['%s'].csr_matrices['%s'] = matrix",statefixed,namefixed);
  PyRun_SimpleString(t);

  // Clean up
  PyRun_SimpleString("del val; del col_ind; del row_ptr; del numRows; del numCols; del matrix");
  free(namefixed);
  free(statefixed);

  Py_DECREF(pname);
  Py_DECREF(pnumCols);
  Py_DECREF(pnumRows);
#endif
}

void python_add_csr_matrix_real_(int *valSize, double val[], int *col_indSize, int col_ind [], int *row_ptrSize,
                            int row_ptr [], char *name, int *namelen, char *state, int *statelen, int *numCols)
{
    python_add_csr_matrix(valSize, val, col_indSize, col_ind, row_ptrSize,
                          row_ptr, name, namelen, state, statelen, numCols,
                          CSR_REAL);
}

void python_add_csr_matrix_integer_(int *valSize, int ival[], int *col_indSize, int col_ind [], int *row_ptrSize,
                                    int row_ptr [], char *name, int *namelen, char *state, int *statelen, int *numCols)
{
    python_add_csr_matrix(valSize, ival, col_indSize, col_ind, row_ptrSize,
                          row_ptr, name, namelen, state, statelen, numCols,
                          CSR_INTEGER);
}


void python_add_vector_(int *num_dim, int *s,
  double x[],
  char *name,int *nlen, int *field_type, char *option_path, int *oplen, char *state,int *slen,
  char *mesh_name, int *mesh_name_len, int *uid){
#ifdef HAVE_NUMPY
  // Make the Fortran vector field availabe to the Python interpreter
  PyObject *pMain = PyImport_AddModule("__main__");
  PyObject *pDict = PyModule_GetDict(pMain);

  python_add_array_double_2d(x,num_dim,s,"vector");
  PyRun_SimpleString("vector = vector.transpose(1, 0)");

  // Fix the Fortran strings for C and Python
  char *namec = fix_string(name,*nlen);
  char *opc = fix_string(option_path,*oplen);
  char *meshc = fix_string(mesh_name,*mesh_name_len);

  PyObject *pname = PyString_FromString(namec);
  PyDict_SetItemString(pDict,"n",pname);
  PyObject *poptionp = PyString_FromString(opc);
  PyDict_SetItemString(pDict,"op",poptionp);
  PyObject *pft = PyInt_FromLong(*field_type);
  PyDict_SetItemString(pDict,"ft",pft);
  PyObject *pnd = PyInt_FromLong(*num_dim);
  PyDict_SetItemString(pDict,"nd",pnd);

  PyRun_SimpleString("n = string.strip(n)");
  PyRun_SimpleString("op = string.strip(op)");

  char *n = fix_string(state,*slen);
  int tlen=150+2**slen+*nlen+*mesh_name_len;
  char t[tlen];
  snprintf(t, tlen,
      "field = vector_field_cache.setdefault(%d, VectorField(n,vector,ft,op,nd,%d,states['%s'].meshes['%s']))",
      *uid, *uid, n, meshc);
  PyRun_SimpleString(t);
  snprintf(t, tlen, "states['%s'].vector_fields['%s'] = field", n, namec);
  PyRun_SimpleString(t);

  // Clean up
  PyRun_SimpleString("del n; del op; del ft; del nd; del vector; del field");
  free(n);
  free(namec);
  free(opc);
  free(meshc);

  Py_DECREF(pname);
  Py_DECREF(poptionp);
  Py_DECREF(pft);
  Py_DECREF(pnd);
#endif
}


void python_add_tensor_(int *sx,int *sy,int *sz, double *x, int num_dim[],
  char *name,int *nlen, int *field_type, char *option_path, int *oplen, char *state,int *slen,
  char *mesh_name, int *mesh_name_len, int *uid){
#ifdef HAVE_NUMPY
  // Expose a Fortran tensor field to the Python interpreter
  PyObject *pMain = PyImport_AddModule("__main__");
  PyObject *pDict = PyModule_GetDict(pMain);

  python_add_array_double_3d(x,sx,sy,sz,"val");
  PyRun_SimpleString("val = val.transpose(2, 0, 1)");

  // Fix the Fortran strings for C and Python

  char *namec = fix_string(name,*nlen);
  char *opc = fix_string(option_path,*oplen);
  char *meshc = fix_string(mesh_name,*mesh_name_len);

  PyObject *pname = PyString_FromString(namec);
  PyDict_SetItemString(pDict,"n",pname);
  PyObject *poptionp = PyString_FromString(opc);
  PyDict_SetItemString(pDict,"op",poptionp);
  PyObject *pft = PyInt_FromLong(*field_type);
  PyDict_SetItemString(pDict,"ft",pft);
  PyObject *pnd0 = PyInt_FromLong(num_dim[0]);
  PyDict_SetItemString(pDict,"nd0",pnd0);
  PyObject *pnd1 = PyInt_FromLong(num_dim[1]);
  PyDict_SetItemString(pDict,"nd1",pnd1);

  PyRun_SimpleString("n = string.strip(n)");
  PyRun_SimpleString("op = string.strip(op)");

  char *n = fix_string(state,*slen);
  int tlen=160+2**slen+*nlen+*mesh_name_len;
  char t[tlen];
  snprintf(t, tlen, "field = tensor_field_cache.setdefault(%d, TensorField(n,val,ft,op,nd0,nd1,%d,states['%s'].meshes['%s']))", *uid, *uid, n, meshc);
  PyRun_SimpleString(t);
  snprintf(t, tlen, "states['%s'].tensor_fields['%s'] = field", n, namec);
  PyRun_SimpleString(t);

  // Clean up
  PyRun_SimpleString("del n; del op; del val; del field");
  free(n);
  free(namec);
  free(opc);
  free(meshc);

  Py_DECREF(pname);
  Py_DECREF(poptionp);
  Py_DECREF(pft);
  Py_DECREF(pnd0);
  Py_DECREF(pnd1);
#endif
}


void python_add_halo_(char *name, int *name_len, int *nprocs, int *unn_offset,
                      char *state_name, int *state_name_len, int *comm,
                      int *uid) {
    PyObject *pMain = PyImport_AddModule("__main__");
    PyObject *pDict = PyModule_GetDict(pMain);

    char *c_name = fix_string(name, *name_len);
    char *c_state_name = fix_string(state_name, *state_name_len);

    char *t;
    char *str;
    int tmp = *nprocs;
    int c_nprocs = *nprocs;
    int digits = 0;
    int maxlen;
    int offset;
    int i;
    assert(tmp > 0);
    while (tmp!=0) { tmp /= 10; ++digits; }

    /* sends = (send1,...,sendN) */
    maxlen = 11 + c_nprocs * (5 + digits);
    t = malloc(maxlen);
    offset = 0;
    offset += snprintf(t, maxlen, "sends = (");
    for ( i = 1; i < c_nprocs; i++ ) {
        offset += snprintf(t + offset, maxlen - offset, "send%d,", i);
    }
    offset += snprintf(t + offset, maxlen - offset, "send%d)", c_nprocs);
    PyRun_SimpleString(t);

    /* recvs = (recv1,...,recvN) */
    offset = 0;
    offset += snprintf(t, maxlen, "recvs = (");
    for ( i = 1; i < c_nprocs; i++ ) {
        offset += snprintf(t + offset, maxlen - offset, "recv%d,", i);
    }
    offset += snprintf(t + offset, maxlen - offset, "recv%d)", c_nprocs);
    PyRun_SimpleString(t);

    i = 250 + strlen(c_state_name) + strlen(c_name);
    str = malloc(i);
    snprintf(str, i, "states['%s'].halos['%s'] = "
        "halos_cache.setdefault(%d, Halo(sends=sends,receives=recvs,comm=%d, gnn2unn=receives_gnn2unn, unn_offset=%d))",
             c_state_name, c_name, *uid, *comm, *unn_offset);
    PyRun_SimpleString(str);
    free(str);
    PyRun_SimpleString("del sends, recvs, receives_gnn2unn");
    for ( i = 1; i <= c_nprocs; i++ ) {
        /* maxlen will always be long enough for this */
        snprintf(t, maxlen, "del send%d,recv%d", i, i);
        PyRun_SimpleString(t);
    }
    free(t);
    free(c_name);
    free(c_state_name);
}
void python_add_mesh_(int ndglno[],int *sndglno, int *elements,
                      int *element_classes, int *nodes,
                      int *node_classes, char *name, int *nlen,
                      char *parent_name, int *plen, char *option_path,
                      int *oplen, int *continuity, int region_ids[],
                      int *sregion_ids,
                      char *state_name, int *state_name_len,
                      char *halo_name, int *halo_name_len,
                      char *element_halo_name, int *element_halo_name_len,
                      int *uid){
#ifdef HAVE_NUMPY
  // Add the Mesh to the interpreter
  PyObject *pMain = PyImport_AddModule("__main__");
  PyObject *pDict = PyModule_GetDict(pMain);

  // Fix the Fortran strings for C and Python
  char *namec = fix_string(name,*nlen);
  char *tmp;
  char *pnamec = fix_string(parent_name,*plen);
  char *opc = fix_string(option_path,*oplen);

  python_add_array_integer_1d(ndglno, sndglno,"mesh_array");

  python_add_array_integer_1d(region_ids, sregion_ids,"region_ids");

  PyObject *pname = PyString_FromString(namec);
  PyDict_SetItemString(pDict,"n",pname);
  PyObject *poptionp = PyString_FromString(opc);
  PyDict_SetItemString(pDict,"op",poptionp);

  PyRun_SimpleString("n = string.strip(n)");
  PyRun_SimpleString("op = string.strip(op)");

  char *n = fix_string(state_name,*state_name_len);
  int tlen=250 + *state_name_len;
  char t[tlen];

  if (*plen > 1) {
    snprintf(t, tlen, "m = states['%s'].meshes.get('%s')", n, pnamec);
    PyRun_SimpleString(t);
  } else {
    snprintf(t, tlen, "m = None");
    PyRun_SimpleString(t);
  }
  if (*halo_name_len > 0) {
      tmp = fix_string(halo_name, *halo_name_len);
      snprintf(t, tlen, "halo = states['%s'].halos.get('%s')", n, tmp);
      PyRun_SimpleString(t);
      free(tmp);
  } else {
      PyRun_SimpleString("halo = None");
  }
  if (*element_halo_name_len > 0 ) {
      tmp = fix_string(element_halo_name, *element_halo_name_len);
      snprintf(t, tlen, "element_halo = states['%s'].halos.get('%s')", n, tmp);
      PyRun_SimpleString(t);
      free(tmp);
  } else {
      PyRun_SimpleString("element_halo = None");
  }
  snprintf(t, tlen, "element_classes = (%d, %d, %d, %d)",
           element_classes[0], element_classes[1],
           element_classes[2], element_classes[3]);
  PyRun_SimpleString(t);
  snprintf(t, tlen, "node_classes = (%d, %d, %d, %d)",
           node_classes[0], node_classes[1],
           node_classes[2], node_classes[3]);
  PyRun_SimpleString(t);
  snprintf(t, tlen,
      "states['%s'].meshes['%s'] = mesh_cache.setdefault(%d, Mesh(mesh_array,%d,element_classes,%d,node_classes,%d,n,m,op,region_ids,%d,node_halo=halo, element_halo=element_halo))",
      n, namec, *uid, *elements, *nodes, *continuity, *uid);
  PyRun_SimpleString(t);

  // Clean up
  PyRun_SimpleString("del n; del op; del m; del element_classes; del node_classes; del element_halo; del halo");
  free(namec);
  free(pnamec);
  free(n);
  free(opc);

  Py_DECREF(pname);
  Py_DECREF(poptionp);
#endif
}


void python_add_faces_(char *state_name, int *state_name_len,
                       char *mesh_name,  int *mesh_name_len,
                       char *surface_mesh_name, int *surface_mesh_name_len,
                       int surface_node_list[], int *ssurface_node_list,
                       int face_element_list[], int *sface_element_list,
                       int boundary_ids[], int *sboundary_ids)
{
#ifdef HAVE_NUMPY
  char *meshc = fix_string(mesh_name,*mesh_name_len);
  char *statec = fix_string(state_name,*state_name_len);
  char *surface_meshc = fix_string(surface_mesh_name,*surface_mesh_name_len);
  int tlen = 220+*mesh_name_len+*state_name_len+*surface_mesh_name_len;
  char t[tlen];

  python_add_array_integer_1d(surface_node_list, ssurface_node_list, "surface_node_list_array");
  python_add_array_integer_1d(face_element_list, sface_element_list, "face_element_list_array");
  python_add_array_integer_1d(boundary_ids, sboundary_ids, "boundary_ids_array");

  // If the faces data structure has already been built we have nothing to do
  snprintf(t, tlen, "faces = states['%s'].meshes['%s'].faces or Faces(surface_node_list_array, face_element_list_array, boundary_ids_array); states['%s'].meshes['%s'].faces = faces; faces.surface_mesh = states['%s'].meshes['%s']",
           statec, meshc, statec, meshc, statec, surface_meshc);
  PyRun_SimpleString(t);

  PyRun_SimpleString("del surface_node_list_array; del face_element_list_array; del boundary_ids_array");
#endif
}


void python_add_element_(int *dim, int *loc, int *ngi, int *degree,
  char *state_name, int *state_name_len, char *mesh_name, int *mesh_name_len,
  double *n,int *nx, int *ny, double *dn, int *dnx, int *dny, int *dnz,
  int *size_spoly_x,int *size_spoly_y,int *size_dspoly_x,int *size_dspoly_y,
  char* family_name, int* family_name_len,
  char* type_name, int* type_name_len,
  double* coords, int* size_coords_x, int* size_coords_y){
#ifdef HAVE_NUMPY
  // Fix the Fortran strings for C and Python
  char *meshc = fix_string(mesh_name,*mesh_name_len);
  char *statec = fix_string(state_name,*state_name_len);
  char *family = fix_string(family_name, *family_name_len);
  char *type = fix_string(type_name, *type_name_len);
  int tlen=400+*mesh_name_len+*state_name_len;
  char t[tlen];

  // Set n
  python_add_array_double_2d(n,nx,ny,"n_array");
  // Set dn
  python_add_array_double_3d(dn,dnx,dny,dnz,"dn_array");
  // Set coords
  python_add_array_double_2d(coords, size_coords_x, size_coords_y, "coords_array");

  // Add the element to the interpreter and make the element variable available
  // so the other attributes in Fortran can be passed in
  PyRun_SimpleString("import copy");
  snprintf(t, tlen, "element = Element(%d,%d,%d,%d,n_array,dn_array, copy.copy(coords_array), %d,%d,%d,%d, '%s', '%s'); states['%s'].meshes['%s'].shape = element",
    *dim,*loc,*ngi,*degree,
    *size_spoly_x,*size_spoly_y,*size_dspoly_x,*size_dspoly_y, family, type,
    statec, meshc
  );
  PyRun_SimpleString(t);

  PyRun_SimpleString("del n_array; del dn_array; del coords_array");
#endif
}


void python_add_quadrature_(int *dim,int *degree,int *loc, int *ngi,
  double *weight, int *weight_size, double *locations, int *l_size, int *is_surfacequadr){
  // Only being called right after an element has been added
#ifdef HAVE_NUMPY
  // Set weights
  python_add_array_double_1d(weight,weight_size,"weight");

  // Set locations
  python_add_array_double_1d(locations,l_size,"locations");

  char c[150];
  if(*is_surfacequadr == 1)
    snprintf(c, 150, "element.set_surface_quadrature(Quadrature(weight,locations,%d,%d,%d,%d))",
      *dim,*degree,*loc,*ngi);
  else if(*is_surfacequadr == 0)
    snprintf(c, 150, "element.set_quadrature(Quadrature(weight,locations,%d,%d,%d,%d))",
      *dim,*degree,*loc,*ngi);
  PyRun_SimpleString(c);

  PyRun_SimpleString("del weight; del locations");
#endif
}


void python_add_polynomial_(double *coefs,int *size,int *degree, int *x,int *y, int *spoly){
#ifdef HAVE_NUMPY
  // Add a polynomial to the latest element
  // Set the coefs array
  python_add_array_double_1d(coefs, size, "coefs");

  // Set the polynomial to the element
  char c[120];
  if(*spoly == 1)
    snprintf(c, 120, "element.set_polynomial_s(Polynomial(coefs,%d),%d,%d)",*degree,*x,*y);
  else if(*spoly == 0)
    snprintf(c, 120, "element.set_polynomial_ds(Polynomial(coefs,%d),%d,%d)",*degree,*x,*y);
  PyRun_SimpleString(c);
  PyRun_SimpleString("del coefs");
//   Py_DECREF(arr);
#endif
}



// Interface for adding arrays

void python_add_array_double_1d(double *arr, int *size, char *name){
#ifdef HAVE_NUMPY
  // Add an array in Python which will be availabe under the variable name 'name'
  PyObject *pMain = PyImport_AddModule("__main__");
  PyObject *pDict = PyModule_GetDict(pMain);

  // Create our NumPy matrix struct:
  // Arguments are:
  //  number of dimensions (int),
  //  size of each dimension (int[]),
  //  data type to determine the width of each element in memory (int)
  //  the actual data as a byte array(char*)

  // Set the array
  PyObject *a = PyArray_SimpleNewFromData(1, (npy_intp[]){*size}, PyArray_DOUBLE, (char*)arr);
  PyDict_SetItemString(pDict,name,a);
  Py_DECREF(a);
#endif
}

void python_add_array_double_2d(double *arr, int *sizex, int *sizey, char *name){
#ifdef HAVE_NUMPY
  // Add an array in Python which will be availabe under the variable name 'name'
  PyObject *pMain = PyImport_AddModule("__main__");
  PyObject *pDict = PyModule_GetDict(pMain);

  // Set the array
  npy_intp dims[] = {*sizey,*sizex};
  PyObject *a = PyArray_SimpleNewFromData(2, dims, PyArray_DOUBLE, (char*)arr);
  PyDict_SetItemString(pDict,name,a);
  char c[200];
  snprintf(c, 200, "%s = numpy.transpose(%s,(1,0))",name,name);
  PyRun_SimpleString(c);
  Py_DECREF(a);
#endif
}

void python_add_array_double_3d(double *arr, int *sizex, int *sizey, int *sizez, char *name){
#ifdef HAVE_NUMPY
  // Add an array in Python which will be availabe under the variable name 'name'
  PyObject *pMain = PyImport_AddModule("__main__");
  PyObject *pDict = PyModule_GetDict(pMain);

  // Set the array
  npy_intp dims[] = {*sizez,*sizey,*sizex};
  PyObject *a = PyArray_SimpleNewFromData(3, dims, PyArray_DOUBLE, (char*)arr);
  PyDict_SetItemString(pDict,name,a);
  char c[200];
  snprintf(c, 200, "%s = numpy.transpose(%s,(2,1,0))",name,name);
  PyRun_SimpleString(c);
  Py_DECREF(a);
#endif
}

void python_add_array_integer_1d(int *arr, int *size, char *name){
#ifdef HAVE_NUMPY
  // Add an array in Python which will be availabe under the variable name 'name'
  PyObject *pMain = PyImport_AddModule("__main__");
  PyObject *pDict = PyModule_GetDict(pMain);

  // Set the array
  PyObject *a = PyArray_SimpleNewFromData(1, (npy_intp[]){*size}, PyArray_INT, (char*)arr);
  PyDict_SetItemString(pDict,name,a);
  Py_DECREF(a);
#endif
}

void python_add_array_integer_2d(int *arr, int *sizex, int *sizey, char *name){
#ifdef HAVE_NUMPY
  // Add an array in Python which will be availabe under the variable name 'name'
  PyObject *pMain = PyImport_AddModule("__main__");
  PyObject *pDict = PyModule_GetDict(pMain);

  // Set the array
  npy_intp dims[] = {*sizey,*sizex};
  PyObject *a = PyArray_SimpleNewFromData(2, dims, PyArray_INT, (char*)arr);
  PyDict_SetItemString(pDict,name,a);
  char c[200];
  snprintf(c, 200, "%s = numpy.transpose(%s,(1,0))",name,name);
  PyRun_SimpleString(c);
  Py_DECREF(a);
#endif
}

void python_add_array_integer_3d(int *arr, int *sizex, int *sizey, int *sizez, char *name){
#ifdef HAVE_NUMPY
  // Add an array in Python which will be availabe under the variable name 'name'
  PyObject *pMain = PyImport_AddModule("__main__");
  PyObject *pDict = PyModule_GetDict(pMain);

  // Set the array
  npy_intp dims[] = {*sizez,*sizey,*sizex};
  PyObject *a = PyArray_SimpleNewFromData(3, dims, PyArray_INT, (char*)arr);
  PyDict_SetItemString(pDict,name,a);
  char c[200];
  snprintf(c, 200, "%s = numpy.transpose(%s,(2,1,0))",name,name);
  PyRun_SimpleString(c);
  Py_DECREF(a);
#endif
}




// Wrapper functions

void python_add_array_double_1d_(double *arr, int *size, char *name, int *name_len){
  // Called from Fortran
  char *namec = fix_string(name,*name_len);
  python_add_array_double_1d(arr, size, namec);
  free(namec);
}

void python_add_array_double_2d_(double *arr, int *sizex, int *sizey, char *name, int *name_len){
  // Called from Fortran
  char *namec = fix_string(name,*name_len);
  python_add_array_double_2d(arr, sizex,sizey, namec);
  free(namec);
}

void python_add_array_double_3d_(double *arr, int *sizex, int *sizey, int *sizez, char *name, int *name_len){
  // Called from Fortran
  char *namec = fix_string(name,*name_len);
  python_add_array_double_3d(arr, sizex,sizey,sizez, namec);
  free(namec);
}
void python_add_array_integer_1d_(int *arr, int *size, char *name, int *name_len){
  // Called from Fortran
  char *namec = fix_string(name,*name_len);
  python_add_array_integer_1d(arr, size, namec);
  free(namec);
}

void python_add_array_integer_2d_(int *arr, int *sizex, int *sizey, char *name, int *name_len){
  // Called from Fortran
  char *namec = fix_string(name,*name_len);
  python_add_array_integer_2d(arr, sizex,sizey, namec);
  free(namec);
}

void python_add_array_integer_3d_(int *arr, int *sizex, int *sizey, int *sizez, char *name, int *name_len){
  // Called from Fortran
  char *namec = fix_string(name,*name_len);
  python_add_array_integer_3d(arr, sizex,sizey,sizez, namec);
  free(namec);
}

#define python_fetch_real F77_FUNC(python_fetch_real_c, PYTHON_FETCH_REAL_C)
void python_fetch_real(char* varname, int* varname_len, double* output)
{
#ifdef HAVE_PYTHON
  PyObject *pMain = PyImport_AddModule("__main__");
  PyObject *pDict = PyModule_GetDict(pMain);

  char *c = fix_string(varname, *varname_len);
  PyObject* real = PyDict_GetItemString(pDict, c);
  if (real == NULL)
  {
    PyErr_Print();
  }
  free(c);

  *output = PyFloat_AsDouble(real);
#endif
}

void python_fetch_integer_c_(char* varname, int* varname_len, long* output)
{
#ifdef HAVE_PYTHON
  PyObject *pMain = PyImport_AddModule("__main__");
  PyObject *pDict = PyModule_GetDict(pMain);

  char *c = fix_string(varname, *varname_len);
  PyObject* integer = PyDict_GetItemString(pDict, c);
  if (integer == NULL)
  {
    PyErr_Print();
  }
  free(c);

  *output = PyInt_AsLong(integer);
#endif
}
