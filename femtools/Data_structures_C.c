#include "Judy.h"
#include "confdefs.h"
#include "stdio.h"
#include "assert.h"

/* To understand these, read
   http://judy.sourceforge.net/doc/Judy1_3x.htm and
   http://judy.sourceforge.net/doc/JudyL_3x.htm */

void F77_FUNC_(integer_set_create_c,INTEGER_SET_CREATE_C)(Pvoid_t* i)
{
  *i = (Pvoid_t) NULL;
}

void F77_FUNC_(integer_set_delete_c,INTEGER_SET_DELETE_C)(Pvoid_t* i)
{
  Word_t mem_freed;
  Pvoid_t ptr = (Pvoid_t) *i;
  J1FA(mem_freed, ptr);
  *i = ptr;
}

void F77_FUNC_(integer_set_insert_c,INTEGER_SET_INSERT_C)(Pvoid_t* i, int* v, int* c)
{
  Word_t changed;
  Word_t index = *v;
  Pvoid_t ptr = (Pvoid_t) *i;
  J1S(changed, ptr, index);
  *c = changed;
  *i = ptr;
}

void F77_FUNC_(integer_set_length_c,INTEGER_SET_LENGTH_C)(Pvoid_t* i, int* l)
{
  Word_t len;
  Pvoid_t ptr = (Pvoid_t) *i;
  J1C(len, ptr, 0, -1);
  *l = len;
  *i = ptr;
}

void F77_FUNC_(integer_set_fetch_c,INTEGER_SET_FETCH_C)(Pvoid_t* i, int* idx, int* val)
{
  Word_t worked, index = *idx, value;
  Pvoid_t ptr = (Pvoid_t) *i;
  J1BC(worked, ptr, index, value);
  assert(worked == 1);
  *val = value;
  *i = ptr;
}

void F77_FUNC_(integer_set_remove_c,INTEGER_SET_REMOVE_C)(Pvoid_t* i, int* idx, int* status)
{
  Word_t index = *idx;
  int stat;
  Pvoid_t ptr = (Pvoid_t) *i;
  J1U(stat, ptr, index);
  *status = stat;
  *i = ptr;
}

void F77_FUNC_(integer_set_has_value_c,INTEGER_SET_HAS_VALUE_C)(Pvoid_t* i, int* val, int* present)
{
  Word_t wpresent, value = *val;
  Pvoid_t ptr = (Pvoid_t) *i;
  J1T(wpresent, ptr, value);
  *present = wpresent;
}

void F77_FUNC_(integer_hash_table_create_c,INTEGER_HASH_TABLE_CREATE_C)(Pvoid_t* i)
{
  assert(sizeof(int*) == sizeof(Pvoid_t));
  *i = (Pvoid_t) NULL;
}

void F77_FUNC_(integer_hash_table_delete_c,INTEGER_HASH_TABLE_DELETE_C)(Pvoid_t* i)
{
  Word_t mem_freed;
  Pvoid_t ptr = (Pvoid_t) *i;
  JLFA(mem_freed, ptr);
  *i = ptr;
}

void F77_FUNC_(integer_hash_table_insert_c,INTEGER_HASH_TABLE_INSERT_C)(Pvoid_t* i, int* k, int* v)
{
  Word_t key = *k;
  PWord_t pvalue;
  Pvoid_t ptr = (Pvoid_t) *i;
  JLI(pvalue, ptr, key);
  *pvalue = *v;
  *i = ptr;
}

void F77_FUNC_(integer_hash_table_length_c,INTEGER_HASH_TABLE_LENGTH_C)(Pvoid_t* i, int* l)
{
  Word_t len;
  Pvoid_t ptr = (Pvoid_t) *i;
  JLC(len, ptr, 0, -1);
  *l = len;
  *i = ptr;
}

void F77_FUNC_(integer_hash_table_fetch_c,INTEGER_HASH_TABLE_FETCH_C)(Pvoid_t* i, int* k, int* v)
{
  Word_t key = *k, value;
  PWord_t pvalue = &value;
  Pvoid_t ptr = (Pvoid_t) *i;
  JLG(pvalue, ptr, key);
  if (pvalue == NULL)
  {
    fprintf(stderr, "Error: hash table has no key %d\n", *k);
    assert(pvalue != NULL);
  }
  *v = *pvalue;
  *i = ptr;
}

void F77_FUNC_(integer_hash_table_remove_c,INTEGER_HASH_TABLE_REMOVE_C)(Pvoid_t* i, int* k, int* status)
{
  Word_t key = *k;
  int stat;
  Pvoid_t ptr = (Pvoid_t) *i;
  JLD(stat, ptr, key);
  *status = stat;
  *i = ptr;
}

void F77_FUNC_(integer_hash_table_has_key_c,INTEGER_HASH_TABLE_HAS_KEY_C)(Pvoid_t* i, int* k, int* present)
{
  Word_t key = *k, value;
  PWord_t pvalue = &value;
  Pvoid_t ptr = (Pvoid_t) *i;
  JLG(pvalue, ptr, key);
  *present = (pvalue != NULL);
  *i = ptr;
}

void F77_FUNC_(integer_hash_table_fetch_pair_c,INTEGER_HASH_TABLE_FETCH_PAIR_C)(Pvoid_t* i, int* idx, int* key, int* val)
{
  Word_t nth = *idx; /* what Judy calls nth is what I am calling index */
  Word_t index; /* what Judy calls index is what I am calling key */
  PWord_t pvalue;
  Pvoid_t ptr = (Pvoid_t) *i;
  JLBC(pvalue, ptr, nth, index);
  assert(pvalue != NULL);
  *key = index;
  *val = *pvalue;
  *i = ptr;
}
