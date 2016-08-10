#include "khash.h"

#ifndef __AC_HASH_H
#define __AC_HASH_H

KHASH_MAP_INIT_INT(32, int)

typedef khash_t(32) *hash_t;

#define kh_set_val(ht, iter, val) kh_val((ht), (iter)) = (val)
#endif
