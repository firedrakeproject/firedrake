/*  Copyright (C) 2006 Imperial College London and others.

    Please see the AUTHORS file in the main source directory for a full list
    of copyright holders.

    Prof. C Pain
    Applied Modelling and Computation Group
    Department of Earth Science and Engineering
    Imperial College London

    C.Pain@Imperial.ac.uk

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

#ifndef CSPUD_H
#define CSPUD_H

#include "spud_enums.h"

#ifdef __cplusplus
extern "C" {
#endif

  void spud_clear_options();
  void* spud_get_manager();
  void spud_set_manager(void* m);

  int spud_load_options(const char* filename, const int filename_len);
  int spud_write_options(const char* filename, const int filename_len);

  int spud_get_child_name(const char* key, const int key_len, const int index, char* child_name, const int child_name_len);

  int spud_get_number_of_children(const char* key, const int key_len, int* child_count);

  int spud_option_count(const char* key, const int key_len);

  int spud_have_option(const char* key, const int key_len);

  int spud_get_option_type(const char* key, const int key_len, int* type);
  int spud_get_option_rank(const char* key, const int key_len, int* rank);
  int spud_get_option_shape(const char* key, const int key_len, int* shape);

  int spud_get_option(const char* key, const int key_len, void* val);

  int spud_add_option(const char* key, const int key_len);

  int spud_set_option(const char* key, const int key_len, const void* val, const int type, const int rank, const int* shape);

  int spud_set_option_attribute(const char* key, const int key_len, const char* val, const int val_len);

  int spud_move_option(const char* key1, const int key1_len, const char* key2, const int key2_len);
  int spud_copy_option(const char* key1, const int key1_len, const char* key2, const int key2_len);

  int spud_delete_option(const char* key, const int key_len);

  void spud_print_options();

#ifdef __cplusplus
}
#endif

#endif
