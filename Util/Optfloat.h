

#ifdef DOUBLEFLOAT
typedef double OPTFLOAT;
#define MPI_OPTFLOAT MPI_DOUBLE
#define blas_symv cblas_dsymv
#define blas_dot cblas_ddot
#define blas_syr cblas_dsyr
#else
typedef float OPTFLOAT;
#define MPI_OPTFLOAT MPI_FLOAT
#define blas_symv cblas_ssymv
#define blas_dot cblas_sdot
#define blas_syr cblas_ssyr
#endif

#ifdef LONGEXP
typedef long double EXPFLOAT;
#else
typedef double EXPFLOAT;
#endif
