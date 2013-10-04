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

#ifdef DDEBUG
#include <assert.h>
#else
#define assert(x)
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

#define get_environment_variable_fc F77_FUNC(get_environment_variable_c, GET_ENVIRONMENT_VARIABLE_C)
void get_environment_variable_fc(char* name, int* name_len, char* val, int* val_len, int* stat)
{
   *stat = 0;

   char* lname;
   lname = malloc((*name_len + 1) * sizeof(char));
   assert(lname != NULL);

   memcpy(lname, name, *name_len * sizeof(char));
   lname[*name_len] = '\0';

   char* lval = getenv(lname);

   free(lname);

   if(lval == NULL){
     *stat = 1;
     return;
   }

   if(strlen(lval) > *val_len){
     fprintf(stderr, "In get_environment_variable_fc\n");
     fprintf(stderr, "Warning: Truncating returned string\n");
     fflush(stderr);
   }else{
     *val_len = strlen(lval);
   }
   memcpy(val, lval, *val_len * sizeof(char));

   return;
}

void F77_FUNC(memcpy,MEMCPY)(void* dest, void* src, int* bytes)
{
  memcpy(dest, src, *bytes);
}

bool compare_pointers(void* ptr1, void* ptr2)
{
  return (ptr1 == ptr2);
}
