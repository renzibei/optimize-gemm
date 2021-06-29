#include <stdlib.h> // For: exit, drand48, malloc, free, NULL, EXIT_FAILURE
#include <stdio.h>  // For: perror
#include <string.h> // For: memset

#include <float.h>  // For: DBL_EPSILON
#include <math.h>   // For: fabs

#ifdef GETTIMEOFDAY
#include <sys/time.h> // For struct timeval, gettimeofday
#else
#include <time.h> // For struct timespec, clock_gettime, CLOCK_MONOTONIC
#endif

/* reference_gemm wraps a call to the BLAS-3 routine GEMM, via the standard FORTRAN interface - hence the reference semantics. */
#define GEMM sgemm_
extern void GEMM (char*, char*, int*, int*, int*, float*, float*, int*, float*, int*, float*, float*, int*);
void reference_gemm (int N, float ALPHA, float* A, float* B, float* C)
{
  char TRANSA = 'N';
  char TRANSB = 'N';
  int M = N;
  int K = N;
  float BETA = 1.;
  int LDA = N;
  int LDB = N;
  int LDC = N;
  GEMM(&TRANSA, &TRANSB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
}

/* Your function must have the following signature: */
extern const char* gemm_desc;
extern void square_gemm (int, float*, float*, float*);

double wall_time ()
{
#ifdef GETTIMEOFDAY
  struct timeval t;
  gettimeofday (&t, NULL);
  return 1.*t.tv_sec + 1.e-6*t.tv_usec;
#else
  struct timespec t;
  clock_gettime (CLOCK_MONOTONIC, &t);
  return 1.*t.tv_sec + 1.e-9*t.tv_nsec;
#endif
}

void die (const char* message)
{
  perror (message);
  exit (EXIT_FAILURE);
}

void fill (float* p, int n)
{
  for (int i = 0; i < n; ++i)
    p[i] = 2 * drand48() - 1; // Uniformly distributed over [-1, 1]
}

void absolute_value (float *p, int n)
{
  for (int i = 0; i < n; ++i)
    p[i] = fabs (p[i]);
}

/* The benchmarking program */
int main (int argc, char **argv)
{
  printf ("Description:\t%s\n\n", gemm_desc);

  /* Test sizes should highlight performance dips at multiples of certain powers-of-two */

  int test_sizes[] =

  /* A representative subset of the first list for initial test.  */
  { 31, 32, 96, 97, 127, 128, 129, 191, 192, 229, 255, 256, 257,
    319, 320, 321, 417, 479, 480, 511, 512, 639, 640};

  int nsizes = sizeof(test_sizes)/sizeof(test_sizes[0]);

  /* assume last size is also the largest size */
  int nmax = test_sizes[nsizes-1];

  /* allocate memory for all problems */
  float* buf = NULL;
  buf = (float*) malloc (3 * nmax * nmax * sizeof(float));
  if (buf == NULL) die ("failed to allocate largest problem size");

  /* For each test size */
  for (int isize = 0; isize < sizeof(test_sizes)/sizeof(test_sizes[0]); ++isize)
  {
    /* Create and fill 3 random matrices A,B,C*/
    int n = test_sizes[isize];

    float* A = buf + 0;
    float* B = A + nmax*nmax;
    float* C = B + nmax*nmax;

    fill (A, n*n);
    fill (B, n*n);
    fill (C, n*n);

    /* Measure performance (in Gflops/s). */

    /* Time a "sufficiently long" sequence of calls to reduce noise */
    double Gflops_s, seconds = -1.0;
    double timeout = 0.1; // "sufficiently long" := at least 1/10 second.
    int    n_iterations = 0;
    for (n_iterations = 1; seconds < timeout; n_iterations *= 2)
    {
      /* Warm-up */
      square_gemm (n, A, B, C);

      /* Benchmark n_iterations runs of square_gemm */
      seconds = -wall_time();
      for (int it = 0; it < n_iterations; ++it)
	     square_gemm (n, A, B, C);
      seconds += wall_time();

      /*  compute Mflop/s rate */
      Gflops_s = 2.e-9 * n_iterations * n * n * n / seconds;
    }
    printf ("Size: %d\tGflop/s: %.3g (%d iter, %.3f seconds)\n", n, Gflops_s, n_iterations, seconds);

    /* Ensure that error does not exceed the theoretical error bound. */

    /* C := A * B, computed with square_gemm */
    memset (C, 0, n * n * sizeof(float));
    square_gemm (n, A, B, C);
    /* Do not explicitly check that A and B were unmodified on square_gemm exit
     *  - if they were, the following will most likely detect it:
     * C := C - A * B, computed with reference_gemm */
    reference_gemm(n, -1., A, B, C);

    /* A := |A|, B := |B|, C := |C| */
    absolute_value (A, n * n);
    absolute_value (B, n * n);
    absolute_value (C, n * n);

    /* C := |C| - 3 * e_mach * n * |A| * |B|, computed with reference_gemm */
    reference_gemm (n, -3.*FLT_EPSILON*n, A, B, C);

    /* If any element in C is positive, then something went wrong in square_gemm */
    for (int i = 0; i < n * n; ++i)
      if (C[i] > 0)
	die("*** FAILURE *** Error in matrix multiply exceeds componentwise error bounds.\n" );
  }

  free (buf);

  return 0;
}
