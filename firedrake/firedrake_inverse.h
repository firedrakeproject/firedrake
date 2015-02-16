
/* Subroutine */ int dgetrf_(int *m, int *n, double *a, int *
	lda, int *ipiv, int *info);

/* Subroutine */ int dgetri_(int *n, double *a, int *lda, int 
	*ipiv, double *work, int *lwork, int *info);

void inverse(double* A, int n)
{
    int ipiv[n];
    int lwork = n*n;
    double work[lwork];
    int info;

    dgetrf_(&n,&n,A,&n,ipiv,&info);
    dgetri_(&n,A,&n,ipiv,work,&lwork,&info);
}
