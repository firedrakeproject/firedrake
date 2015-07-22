#ifndef _EVALUATE_H
#define _EVALUATE_H

struct Function;

extern int evaluate(struct Function *f, double *x, double *result);

#define inside(f, x) (!evaluate((f), (x), NULL))

#endif /* _EVALUATE_H */
