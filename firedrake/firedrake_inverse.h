void invert(double* A, int n)
{
    int ipiv[n];
    int lwork = n*n;
    double work[lwork];
    int info;
    int lda = 4;

    dgetrf_(&n,&n,A,&lda,ipiv,&info);
    dgetri_(&n,A,&lda,ipiv,work,&lwork,&info);
}
