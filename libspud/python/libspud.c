#include <Python.h>
#include <string.h>
#include "spud.h"
#include <stdio.h>

#define MAXLENGTH   2048

static PyObject *SpudError;
static PyObject *SpudTypeError;
static PyObject *SpudKeyError;
static PyObject *SpudFileError;
static PyObject *SpudNewKeyWarning;
static PyObject *SpudAttrSetFailedWarning;
static PyObject *SpudShapeError;
static PyObject *SpudRankError;

void* manager;

static PyObject *
error_checking(int outcome, char *functionname)
{
    char errormessage [MAXLENGTH];

    if (outcome == SPUD_KEY_ERROR){
        snprintf(errormessage, MAXLENGTH, "Error: The specified option is not present \
                        in the dictionary in %s", functionname);
        PyErr_SetString(SpudKeyError, errormessage);
        return NULL;
    }
    if (outcome == SPUD_TYPE_ERROR){
        snprintf(errormessage, MAXLENGTH, "Error: The specified option has a different \
                        type from that of the option argument provided in %s", functionname);
        PyErr_SetString(SpudTypeError, errormessage);
        return NULL;
    }
    if (outcome == SPUD_NEW_KEY_WARNING){
        snprintf(errormessage, MAXLENGTH, "Warning: The option being inserted is not ]  \
                        already in the dictionary %s", functionname);
        PyErr_SetString(SpudNewKeyWarning, errormessage);
        return NULL;
    }
    if (outcome == SPUD_FILE_ERROR){
        snprintf(errormessage, MAXLENGTH, "Error: The specified options file cannot be  \
                        read or written to as the routine requires in %s", functionname);
        PyErr_SetString(SpudFileError, errormessage);
        return NULL;
    }
    if (outcome == SPUD_RANK_ERROR){
        snprintf(errormessage, MAXLENGTH, "Error: The specified option has a different rank from \
                      that of the option argument provided %s", functionname);
        PyErr_SetString(SpudRankError, errormessage);
        return NULL;
    }
    if (outcome == SPUD_SHAPE_ERROR){
        snprintf(errormessage, MAXLENGTH, "Error: The specified option has a different shape from \
                      that of the option argument provided in %s", functionname);
        PyErr_SetString(SpudShapeError, errormessage);
        return NULL;
    }
    if (outcome == SPUD_ATTR_SET_FAILED_WARNING){
        snprintf(errormessage, MAXLENGTH, "Warning: The option being set as an attribute can not be \
                      set as an attribute in %s", functionname);
        PyErr_SetString(SpudAttrSetFailedWarning, errormessage);
        return NULL;
    }
    if (outcome == SPUD_NO_ERROR){
        Py_RETURN_NONE;
    }

    PyErr_SetString(SpudError,"Error: error checking failed.");
    return NULL;
}

static PyObject *
libspud_load_options(PyObject *self, PyObject *args)
{
    const char *key;
    int key_len;
    int outcomeLoadOptions;

    if (!PyArg_ParseTuple(args, "s", &key))
        return NULL;
    key_len = strlen(key);
    outcomeLoadOptions = spud_load_options(key,key_len);

    return error_checking(outcomeLoadOptions, "load options");
}

static PyObject*
libspud_print_options(PyObject *self, PyObject *args)
{
    spud_print_options();

    Py_RETURN_NONE;
}

static PyObject*
libspud_clear_options(PyObject *self, PyObject *args)
{
    spud_clear_options();

    Py_RETURN_NONE;
}

static PyObject *
libspud_get_number_of_children(PyObject *self, PyObject *args)
{
    const char *key;
    int key_len;
    int child_count;
    int outcomeGetNumChildren;

    if (!PyArg_ParseTuple(args, "s", &key))
        return NULL;
    key_len = strlen(key);
    outcomeGetNumChildren = spud_get_number_of_children(key, key_len, &child_count);
    if (error_checking(outcomeGetNumChildren, "get number of children") == NULL){
        return NULL;
    }

    return Py_BuildValue("i", child_count);
}

static PyObject *
libspud_get_child_name(PyObject *self, PyObject *args)
{
    const char *key;
    int key_len;
    int index;
    char child_name [MAXLENGTH];
    int i;
    int outcomeGetChildName;

    for (i = 0; i < MAXLENGTH; i++){
        child_name[i] = '\0';
    }
    if (!PyArg_ParseTuple(args, "si", &key, &index)){
        return NULL;
    }
    key_len = strlen(key);
    outcomeGetChildName = spud_get_child_name(key, key_len, index, child_name, MAXLENGTH);
    if (error_checking(outcomeGetChildName, "get child name") == NULL){
        return NULL;
    }

    return Py_BuildValue("s", child_name);
}

static PyObject *
libspud_option_count(PyObject *self, PyObject *args)
{
    const char *key;
    int key_len;
    int numoptions;

    if (!PyArg_ParseTuple(args, "s", &key)){
        return NULL;
    }
    key_len = strlen(key);
    numoptions = spud_option_count(key, key_len);

    return Py_BuildValue("i", numoptions);
}

static PyObject *
libspud_have_option(PyObject *self, PyObject *args)
{
    const char *key;
    int key_len;
    int haveoption;

    if (!PyArg_ParseTuple(args, "s", &key)){
        return NULL;
    }
    key_len = strlen(key);
    haveoption = spud_have_option(key, key_len);

    if (haveoption == 0){
        Py_RETURN_FALSE;
    }
    else{
        Py_RETURN_TRUE;
    }
}

static PyObject *
libspud_add_option(PyObject *self, PyObject *args)
{
    const char *key;
    int key_len;
    int outcomeAddOption;

    if (!PyArg_ParseTuple(args, "s", &key)){
        return NULL;
    }
    key_len = strlen(key);
    outcomeAddOption = spud_add_option(key, key_len);
    return error_checking(outcomeAddOption, "add option");

}

static PyObject *
libspud_get_option_type(PyObject *self, PyObject *args)
{
    const char *key;
    int key_len;
    int type;
    int outcomeGetOptionType;

    if (!PyArg_ParseTuple(args, "s", &key)){
        return NULL;
    }
    key_len = strlen(key);
    outcomeGetOptionType = spud_get_option_type(key, key_len, &type);
    if (error_checking(outcomeGetOptionType, "get option type") == NULL){
        return NULL;
    }
    if (type == SPUD_DOUBLE){
        Py_INCREF(&PyFloat_Type);
        return (PyObject*) &PyFloat_Type;
    }
    else if (type == SPUD_INT){
        Py_INCREF(&PyInt_Type);
        return (PyObject*) &PyInt_Type;
    }
    else if (type == SPUD_NONE){
        Py_RETURN_NONE;
    }
    else if (type == SPUD_STRING){
        Py_INCREF(&PyString_Type);
        return (PyObject*) &PyString_Type;
    }

    PyErr_SetString(SpudError,"Error: Get option type function failed");
    return NULL;
}

static PyObject *
libspud_get_option_rank(PyObject *self, PyObject *args)
{
    const char *key;
    int key_len;
    int rank;
    int outcomeGetOptionRank;

    if (!PyArg_ParseTuple(args, "s", &key)){
        return NULL;
    }
    key_len = strlen(key);
    outcomeGetOptionRank = spud_get_option_rank(key, key_len, &rank);
    if (error_checking(outcomeGetOptionRank, "get option rank") == NULL){
        return NULL;
    }

    return Py_BuildValue("i", rank);
}

static PyObject *
libspud_get_option_shape(PyObject *self, PyObject *args)
{
    const char *key;
    int key_len;
    int shape[2];
    int outcomeGetOptionShape;

    if (!PyArg_ParseTuple(args, "s", &key)){
        return NULL;
    }
    key_len = strlen(key);
    outcomeGetOptionShape = spud_get_option_shape(key, key_len, shape);
    if (error_checking(outcomeGetOptionShape, "get option shape") == NULL){
        return NULL;
    }

    return Py_BuildValue("(i,i)", shape[0],shape[1]);
}

static PyObject*
spud_get_option_aux_list_ints(const char *key, int key_len, int type, int rank, int *shape)
{   // this function is for getting option when the option is of type a list of ints
    int outcomeGetOption;
    int size = shape[0];
    int val [size];
    int j;

    outcomeGetOption = spud_get_option(key, key_len, val);
    if (error_checking(outcomeGetOption, "get option aux list") == NULL){
        return NULL;
    }
    PyObject* pylist = PyList_New(size);
    if (pylist == NULL){
        printf("New list error.");
        return NULL;
    }
    for (j = 0; j < size; j++){
        PyObject* element = Py_BuildValue("i", val[j]);
        PyList_SetItem(pylist, j, element);
    }

    return pylist;
}

static PyObject*
spud_get_option_aux_list_doubles(const char *key, int key_len, int type, int rank, int *shape)
{   // this function is for getting option when the option is of type a list of doubles
    int outcomeGetOption;
    int size = shape[0];
    double val [size];
    int j;

    outcomeGetOption = spud_get_option(key, key_len, val);
    if (error_checking(outcomeGetOption, "get option aux list") == NULL){
        return NULL;
    }
    PyObject* pylist = PyList_New(size);
    if (pylist == NULL){
        printf("New list error.");
        return NULL;
    }
    for (j = 0; j < size; j++){
        PyObject* element = Py_BuildValue("d", val[j]);
        PyList_SetItem(pylist, j, element);
    }

    return pylist;
}

static PyObject *
spud_get_option_aux_scalar_or_string(const char *key, int key_len, int type, int rank, int *shape)
{   // this function is for getting option when the option is of type a scalar or string
    int outcomeGetOption;
    if (type == SPUD_DOUBLE){
        double val;
        outcomeGetOption = spud_get_option(key, key_len, &val);
        if (error_checking(outcomeGetOption, "get option aux scalar or string") == NULL){
            return NULL;
        }
        return Py_BuildValue("d", val);
    }
    else if (type == SPUD_INT){
        int val;
        outcomeGetOption = spud_get_option(key, key_len, &val);
        if (error_checking(outcomeGetOption, "get option aux scalar or string") == NULL){
            return NULL;
        }
        return Py_BuildValue("i", val);
    }
    else if (type == SPUD_STRING) {
        int size = shape[0];
        char val[size+1];
        int i;
        for (i = 0; i < size+1; i++)
          val[i] = '\0';

        outcomeGetOption = spud_get_option(key, key_len, val);
        if (error_checking(outcomeGetOption, "get option aux scalar or string") == NULL){
            return NULL;
        }
        return Py_BuildValue("s", val);
    }

    PyErr_SetString(SpudError,"Error: Get option aux scalar failed");
    return NULL;
}

static PyObject*
spud_get_option_aux_tensor_doubles(const char *key, int key_len, int type, int rank, int *shape)
{   // this function is for getting option when the option is of type a tensor o doubles
    int outcomeGetOption;
    int rowsize = shape[0];
    int colsize = shape[1];
    int size = rowsize*colsize;
    double val [size];
    int m;
    int n;
    int counter;

    outcomeGetOption = spud_get_option(key, key_len, val);
    if (error_checking(outcomeGetOption, "get option aux tensor") == NULL){
        return NULL;
    }
    PyObject* pylist = PyList_New(rowsize);
    if (pylist == NULL){
        printf("New list error");
        return NULL;
    }
    counter = 0;
    for (m = 0; m < rowsize; m++){
        PyObject* pysublist = PyList_New(colsize);
        if (pysublist == NULL){
            printf("New sublist error");
            return NULL;
        }
        for (n = 0; n < colsize; n++){
            PyObject* element = Py_BuildValue("d", val[counter]);
            PyList_SetItem(pysublist, n, element);
            counter++;
        }
        PyList_SetItem(pylist, m, pysublist);
    }

    return pylist;
}

static PyObject*
spud_get_option_aux_tensor_ints(const char *key, int key_len, int type, int rank, int *shape)
{   // this function is for getting option when the option is of type a tensor of ints
    int outcomeGetOption;
    int rowsize = shape[0];
    int colsize = shape[1];
    int size = rowsize*colsize;
    int val [size];
    int m;
    int n;
    int counter;

    outcomeGetOption = spud_get_option(key, key_len, val);
    if (error_checking(outcomeGetOption, "get option aux tensor") == NULL){
        return NULL;
    }
    PyObject* pylist = PyList_New(rowsize);
    if (pylist == NULL){
        printf("New list error");
        return NULL;
    }
    counter = 0;
    for (m = 0; m < rowsize; m++){
        PyObject* pysublist = PyList_New(colsize);
        if (pysublist == NULL){
            printf("New sublist error");
            return NULL;
        }
        for (n = 0; n < colsize; n++){
            PyObject* element = Py_BuildValue("i", val[counter]);
            PyList_SetItem(pysublist, n, element);
            counter++;
        }
        PyList_SetItem(pylist, m, pysublist);
    }

    return pylist;
}

static PyObject *
libspud_get_option(PyObject *self, PyObject *args)
{
    const char *key;
    int key_len;
    int type;
    int rank = 0;
    int shape[2];
    int outcomeGetOptionType;
    int outcomeGetOptionRank;
    int outcomeGetOptionShape;

    if(!PyArg_ParseTuple(args, "s", &key)){
        return NULL;
    }
    key_len = strlen(key);
    outcomeGetOptionRank = spud_get_option_rank(key, key_len, &rank);
    if (error_checking(outcomeGetOptionRank, "get option") == NULL){
        return NULL;
    }
    outcomeGetOptionType = spud_get_option_type(key, key_len, &type);
    if (error_checking(outcomeGetOptionType, "get option") == NULL){
        return NULL;
    }
    outcomeGetOptionShape = spud_get_option_shape(key, key_len, shape);
    if (error_checking(outcomeGetOptionShape, "get option") == NULL){
        return NULL;
    }

    if (rank == -1){ // type error
        char errormessage [MAXLENGTH];
        snprintf(errormessage, MAXLENGTH, "Error: The specified option has a different \
                        type from that of the option argument provided in %s", "get option");
        PyErr_SetString(SpudTypeError, errormessage);
        return NULL;
    }
    else if (rank == 0){ // scalar
        return spud_get_option_aux_scalar_or_string(key, key_len, type, rank, shape);
    }
    else if (rank == 1){ // list or string
        if (type == SPUD_INT){  //a list of ints
            return spud_get_option_aux_list_ints(key, key_len, type, rank, shape);
        }
        else if (type == SPUD_DOUBLE){  //a list of doubles
            return spud_get_option_aux_list_doubles(key, key_len, type, rank, shape);
        }
        else if (type == SPUD_STRING){  //string
            return spud_get_option_aux_scalar_or_string(key, key_len, type, rank, shape);
        }
    }
    else if (rank == 2){ // tensor
        if (type == SPUD_DOUBLE){  //a tensor of doubles
            return spud_get_option_aux_tensor_doubles(key, key_len, type, rank, shape);
        }
        else if (type == SPUD_INT){  //a tensor of ints
            return spud_get_option_aux_tensor_ints(key, key_len, type, rank, shape);
        }
    }

    PyErr_SetString(SpudError,"Error: Get option failed.");
    return NULL;
}
static PyObject*
set_option_aux_list_ints(PyObject *pylist, const char *key, int key_len, int type, int rank, int *shape)
{   // this function is for setting option when the second argument is of type a list of ints
    int j;
    int psize = PyList_Size(pylist);
    shape[0] = psize;
    int val[psize] ;
    int outcomeSetOption;
    int element;

    for (j = 0; j < psize; j++){
        element = -1;
        PyObject* pelement = PyList_GetItem(pylist, j);
        PyArg_Parse(pelement, "i", &element);
        val[j] = element;
    }
    outcomeSetOption = spud_set_option(key, key_len, val, type, rank, shape);
    if (error_checking(outcomeSetOption, "set option aux list ints") == NULL){
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyObject*
set_option_aux_list_doubles(PyObject *pylist, const char *key, int key_len, int type, int rank, int *shape)
{   // this function is for setting option when the second argument is of type a list of doubles
    int j;
    int psize = PyList_Size(pylist);
    shape[0] = psize;
    double val [psize];
    int outcomeSetOption;
    double element;

    for (j = 0; j < psize; j++){
        element = -1.0;
        PyObject* pelement = PyList_GetItem(pylist, j);
        element = PyFloat_AS_DOUBLE(pelement);
        val[j] = element;
    }
    outcomeSetOption = spud_set_option(key, key_len, val, type, rank, shape);
    if (error_checking(outcomeSetOption, "set option aux list ints") == NULL){
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject*
set_option_aux_string(PyObject *pystring, const char *key, int key_len, int type, int rank, int *shape)
{   // this function is for setting option when the second argument is of type string
    char *val = PyString_AsString(pystring);
    int outcomeSetOption = spud_set_option(key, key_len, val, type, rank, shape);
    return error_checking(outcomeSetOption, "set option aux string");
}

static PyObject*
libspud_set_option_attribute(PyObject *self, PyObject *args)
{
    const char*key;
    int key_len;
    PyObject* firstArg;
    PyObject* secondArg;
    const char*val;
    int val_len;
    int outcomeSetOption;

    firstArg = PyTuple_GetItem(args, 0);
    secondArg = PyTuple_GetItem(args, 1);
    PyArg_Parse(firstArg, "s", &key);
    key_len = strlen(key);
    PyArg_Parse(secondArg, "s", &val);
    val_len = strlen(val);
    outcomeSetOption = spud_set_option_attribute(key, key_len, val, val_len);
    return error_checking(outcomeSetOption, "set option attribute");
}

static PyObject*
libspud_delete_option(PyObject *self, PyObject *args)
{
    const char*key;
    int key_len;
    PyObject* firstArg;
    int outcomeDeleteOption;

    firstArg = PyTuple_GetItem(args, 0);
    PyArg_Parse(firstArg, "s", &key);
    key_len = strlen(key);
    outcomeDeleteOption = spud_delete_option(key, key_len);
    return error_checking(outcomeDeleteOption, "delete option");
}

static PyObject*
set_option_aux_tensor_doubles(PyObject *pylist, const char *key, int key_len, int type, int rank, int *shape)
{   // this function is for setting option when the second argument is of type a tensor of doubles
    int i;
    int j;
    int counter = 0;

    int outcomeSetOption;

    int size = shape[0]*shape[1];

    double element;
    double val [size];

    for (i = 0; i < shape[0]; i++){
        PyObject* pysublist = PyList_GetItem(pylist, i);
        for (j = 0; j < shape[1]; j++){
            PyObject* pysublistElement = PyList_GetItem(pysublist, j);
            element = PyFloat_AS_DOUBLE(pysublistElement);
            val[counter] = (double) element;
            counter ++;
        }
    }

    outcomeSetOption = spud_set_option(key, key_len, val, type, rank, shape);
    return error_checking(outcomeSetOption, "set option aux tensor doubles");
}

static PyObject*
set_option_aux_tensor_ints(PyObject *pylist, const char *key, int key_len, int type, int rank, int *shape)
{   // this function is for setting option when the second argument is of type a tensor of ints
    int i;
    int j;
    int counter = 0;
    int size = shape[0]*shape[1];
    int val [size];
    int outcomeSetOption;

    int element;

    for (i = 0; i < shape[0]; i++){
        PyObject* pysublist = PyList_GetItem(pylist, i);
        for (j = 0; j < shape[1]; j++){
            element = 1;
            PyObject* pysublistElement = PyList_GetItem(pysublist, j);
            PyArg_Parse(pysublistElement, "i", &element);
            val[counter] = element;
            counter ++;
        }
    }

    outcomeSetOption = spud_set_option(key, key_len, val, type, rank, shape);
    return error_checking(outcomeSetOption, "set option aux tensor ints");
}

static PyObject*
set_option_aux_scalar(PyObject *pyscalar, const char *key, int key_len, int type, int rank, int *shape)
{   // this function is for setting option when the second argument is of type scalar
    int outcomeSetOption = SPUD_NO_ERROR;

    if (type == SPUD_DOUBLE){ //scalar is double
        double val;
        PyArg_Parse(pyscalar, "d", &val);
        outcomeSetOption = spud_set_option(key, key_len, &val, type, rank, shape);
    }
    else if (type == SPUD_INT){
        int val;
        PyArg_Parse(pyscalar, "i", &val);
        outcomeSetOption = spud_set_option(key, key_len, &val, type, rank, shape);
    }

    return error_checking(outcomeSetOption, "set option aux scalar");
}


static PyObject*
libspud_set_option(PyObject *self, PyObject *args)
{
    const char *key;
    int key_len;
    int type=-1;
    int rank=-1;
    int shape[2];
    PyObject* firstArg;
    PyObject* secondArg;

    if(PyTuple_GET_SIZE(args)!=2){
        PyErr_SetString(SpudError,"Error: set_option takes exactly 2 arguments.");
        return NULL;
    }

    firstArg = PyTuple_GetItem(args, 0);
    secondArg = PyTuple_GetItem(args, 1);
    PyArg_Parse(firstArg, "s", &key);
    key_len = strlen(key);

    if (!spud_have_option(key, key_len)){ //option does not exist yet
        int outcomeAddOption = spud_add_option(key, key_len);
        error_checking(outcomeAddOption, "set option");
    }

    if (PyInt_Check(secondArg)){ //just an int
        type = SPUD_INT;
        rank = 0;
        shape[0] = -1;
        shape[1] = -1;

    }
    else if (PyString_Check(secondArg)){// a string
        type = SPUD_STRING;
        rank = 1;
        shape[0] = PyString_GET_SIZE(secondArg);
        shape[1] = -1;
    }
    else if (PyFloat_Check(secondArg)){// a double
        type = SPUD_DOUBLE;
        rank = 0;
        shape[0] = -1;
        shape[1] = -1;
    }
    else if (PyList_Check(secondArg)){
        PyObject* listElement = PyList_GetItem(secondArg, 0);
        if (PyInt_Check(listElement)){ //list of ints
            type = SPUD_INT;
            rank = 1;
            shape[0] = 1;
            shape[1] = -1;
        }
        else if (PyFloat_Check(listElement)){
            type = SPUD_DOUBLE; //list of doubles
            rank = 1;
            shape[0] = 1;
            shape[1] = -1;
        }
        else if (PyList_Check(listElement)){ //list of lists
            int pylistSize = PyList_GET_SIZE(secondArg);
            int pysublistSize = PyList_GET_SIZE(listElement);
            PyObject* sublistElement = PyList_GetItem(listElement, 0);
            if (PyInt_Check(sublistElement)){ //list of lists of ints
                type = SPUD_INT;
            }
            else if (PyFloat_Check(sublistElement)){//list of lists of doubles
                type = SPUD_DOUBLE;
            }
            rank = 2;
            shape[0] = pylistSize;
            shape[1] = pysublistSize;
        }
    }

    if (rank == 0){ // scalar
        set_option_aux_scalar(secondArg, key, key_len, type, rank, shape);
    }
    else if (rank == 1){ // list or string
        if (PyString_Check(secondArg)){ // pystring
            set_option_aux_string(secondArg, key, key_len, type, rank, shape);
        }
        else if (type == SPUD_INT) { // list of ints
            set_option_aux_list_ints(secondArg, key, key_len, type, rank, shape);
        }
        else if (type == SPUD_DOUBLE){ // list of doubles
            set_option_aux_list_doubles(secondArg, key, key_len, type, rank, shape);
        }
    }
    else if (rank == 2){ // tensor
        if (type == SPUD_DOUBLE) { // tensor of doubles
            set_option_aux_tensor_doubles(secondArg, key, key_len, type, rank, shape);
        }
        else if (type == SPUD_INT) { // tensor of ints
            set_option_aux_tensor_ints(secondArg, key, key_len, type, rank, shape);
        }
    }

    Py_RETURN_NONE;
}

static PyObject*
libspud_write_options(PyObject *self, PyObject *args)
{
    PyObject* firstArg;
    char *filename;
    int filename_len;
    int outcomeWriteOptions;

    firstArg = PyTuple_GetItem(args, 0);
    PyArg_Parse(firstArg, "s", &filename);
    filename_len = strlen(filename);
    outcomeWriteOptions = spud_write_options (filename, filename_len);
    return error_checking(outcomeWriteOptions, "write options");
}

static PyMethodDef libspudMethods[] = {
    {"load_options",  libspud_load_options, METH_VARARGS,
     PyDoc_STR("Reads the xml file into the options tree.")},
    {"print_options",  libspud_print_options, METH_VARARGS,
     PyDoc_STR("Print the entire options tree to standard output.")},
    {"clear_options",  libspud_clear_options, METH_VARARGS,
     PyDoc_STR("Clears the entire options tree.")},
    {"get_number_of_children",  libspud_get_number_of_children, METH_VARARGS,
     PyDoc_STR("get number of children under key.")},
    {"get_child_name",  libspud_get_child_name, METH_VARARGS,
     PyDoc_STR("Get name of the indexth child of key.")},
    {"option_count",  libspud_option_count, METH_VARARGS,
     PyDoc_STR("Return the number of options matching key.")},
    {"have_option",  libspud_have_option, METH_VARARGS,
     PyDoc_STR("Checks whether key is present in options dictionary.")},
    {"get_option_type",  libspud_get_option_type, METH_VARARGS,
     PyDoc_STR("Returns the type of option specified by key.")},
    {"get_option_rank",  libspud_get_option_rank, METH_VARARGS,
     PyDoc_STR("Return the rank of option specified by key.")},
    {"get_option_shape",  libspud_get_option_shape, METH_VARARGS,
     PyDoc_STR("Return the shape of option specified by key.")},
    {"get_option",  libspud_get_option, METH_VARARGS,
     PyDoc_STR("Retrives option values from the options dictionary.")},
    {"set_option",  libspud_set_option, METH_VARARGS,
     PyDoc_STR("Sets options in the options tree.")},
    {"write_options",  libspud_write_options, METH_VARARGS,
     PyDoc_STR("Write options tree out to the xml file specified by name.")},
    {"delete_option",  libspud_delete_option, METH_VARARGS,
     PyDoc_STR("Delete options at the specified key.")},
    {"set_option_attribute",  libspud_set_option_attribute, METH_VARARGS,
     PyDoc_STR("As set_option, but additionally attempts to mark the option at the \
     specified key as an attribute. Set_option_attribute accepts only string data for val.")},
    {"add_option",  libspud_add_option, METH_VARARGS,
     PyDoc_STR("Creates a new option at the supplied key.")},
    {NULL, NULL, 0, NULL},
            /* Sentinel */
};

PyMODINIT_FUNC
initlibspud(void)
{
    PyObject *m;

    m = Py_InitModule("libspud", libspudMethods);
    if (m == NULL)
        return;

    SpudError = PyErr_NewException("Spud.error", NULL, NULL);
    SpudNewKeyWarning = PyErr_NewException("SpudNewKey.warning", NULL, NULL);
    SpudKeyError = PyErr_NewException("SpudKey.error", NULL, NULL);
    SpudTypeError = PyErr_NewException("SpudType.error", NULL, NULL);
    SpudFileError = PyErr_NewException("SpudFile.warning", NULL, NULL);
    SpudAttrSetFailedWarning = PyErr_NewException("SpudAttrSetFailed.warning", NULL, NULL);
    SpudShapeError = PyErr_NewException("SpudShape.error", NULL, NULL);
    SpudRankError = PyErr_NewException("SpudRank.error", NULL, NULL);

    Py_INCREF(SpudError);
    Py_INCREF(SpudNewKeyWarning);
    Py_INCREF(SpudKeyError);
    Py_INCREF(SpudTypeError);
    Py_INCREF(SpudFileError);
    Py_INCREF(SpudRankError);
    Py_INCREF(SpudShapeError);
    Py_INCREF(SpudAttrSetFailedWarning);

    PyModule_AddObject(m, "SpudError", SpudError);
    PyModule_AddObject(m, "SpudNewKeyWarning", SpudNewKeyWarning);
    PyModule_AddObject(m, "SpudKeyError", SpudKeyError);
    PyModule_AddObject(m, "SpudTypeError", SpudTypeError);
    PyModule_AddObject(m, "SpudFileError", SpudFileError);
    PyModule_AddObject(m, "SpudAttrSetFailedWarning", SpudAttrSetFailedWarning);
    PyModule_AddObject(m, "SpudShapeError", SpudShapeError);
    PyModule_AddObject(m, "SpudRankError", SpudRankError);


#if PY_MINOR_VERSION > 6
    manager = PyCapsule_Import("spud_manager._spud_manager", 0);
    if (manager != NULL) spud_set_manager(manager);
    else PyErr_Clear();
#endif

}


