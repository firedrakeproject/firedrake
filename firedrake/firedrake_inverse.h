#include "clapack.h"

void inverse(double* A, int n)
{
    int IPIV[n];
    int LWORK = n*n;
    double WORK[LWORK];
    int INFO;

    dgetrf_(&N,&N,A,&N,IPIV,&INFO);
    dgetri_(&N,A,&N,IPIV,WORK,&LWORK,&INFO);
}
