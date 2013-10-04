#include "math.h"
#include "stdio.h"

#include "confdefs.h"

#ifdef DOUBLEP
#define REAL double
#else
#define REAL float
#endif

int F77_FUNC_(c99_isnan,C99_ISNAN)(REAL *x)
{
  return isnan(*x);
}

#ifdef DEBUG_NAN
int main(void)
{
  REAL x, y;

  x = 0.0;
  y = x / x;

  printf("y == %lf\n", y);
  printf("isnan(&y) == %d\n", F77_FUNC_(c99_isnan,C99_ISNAN)(&y));
}
#endif
