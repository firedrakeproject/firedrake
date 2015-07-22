#ifndef _LOCATE_H
#define _LOCATE_H

struct Function;

typedef int (*inside_p)(void *data_, struct Function *f, int cell, double *x);

extern int locate_cell(struct Function *f, double *x,
		       inside_p try_candidate, void *data_);

#endif /* _LOCATE_H */
