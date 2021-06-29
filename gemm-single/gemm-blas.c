#define GEMM sgemm_
extern void GEMM (char*, char*, int*, int*, int*, float*, float*, int*, float*, int*, float*, float*, int*); 

const char* gemm_desc = "Reference gemm.";

/* This routine performs a gemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values.    
 * This function wraps a call to the BLAS-3 routine GEMM, via the standard FORTRAN interface - hence the reference semantics. */
void square_gemm (int N, float* A, float* B, float* C)
{
  char TRANSA = 'N';
  char TRANSB = 'N';
  int M = N;
  int K = N;
  float ALPHA = 1.;
  float BETA = 1.;
  int LDA = N;
  int LDB = N;
  int LDC = N;
  GEMM(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
}   
