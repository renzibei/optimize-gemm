#include "arm_neon.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define BLOCK_ROW 64
#define BLOCK_COL 128

#define min(a,b) (((a)<(b))?(a):(b))

const char* gemm_desc = "Naive, three-loop gemm.";

static void naive_transpose(int row, int col, float *A) {
  for(int i = 1; i < row; ++i) {
    for (int j = 0;j < i; ++j) {
      int ij_index = i + j * row, ji_index = j + i * row;
      float tmp = A[ij_index];
      A[ij_index] = A[ji_index];
      A[ji_index] = tmp;
    }
  }
}

void dot_mul(int n, int incX, float *A, float *B, float *C) {
  for (int i = 0; i < n; ++i) {
    *C += A[i * incX] * B[i];
  }
}

#define NANOSECONDS_IN_SECOND 1000000000LL

static inline int64_t wall_time_ns ()
{
#ifdef GETTIMEOFDAY
  struct timeval t;
  gettimeofday (&t, NULL);
  return t.tv_sec * NANOSECONDS_IN_SECOND + t.tv_usec * 1000LL;
#else
  struct timespec t;
  clock_gettime (CLOCK_MONOTONIC, &t);
  return t.tv_sec * NANOSECONDS_IN_SECOND + t.tv_nsec;
#endif
}

#define UNROLL_NUM 4

#define SIMD_UNROLL 32
#define SIMD_UNROLLD4 8

void dot_mul_unroll(int n, int lda, int ldb, int ldc, float *A, float *B, float *C, float *old_A, float *old_B, float *old_C) {
  // dot_mul(n, lda, A, B, C);
  // dot_mul(n, lda, A, B + ldb, C + ldc);
  // dot_mul(n, lda, A, B + 2 * ldb, C + 2 * ldc);
  // dot_mul(n, lda, A, B + 3 * ldb, C + 3 * ldc);
  register float c00, c01, c02, c03, c04, c05, c06, c07, a0i;
  c00 = 0.0, c01 = 0.0, c02 = 0.0, c03 = 0.0, c04 = 0.0, c05 = 0.0, c06 = 0.0, c07 = 0.0;
  register float *bi0_p, *bi1_p, *bi2_p, *bi3_p, *bi4_p, *bi5_p, *bi6_p, *bi7_p;
  bi0_p = B, bi1_p = B + ldb, bi2_p = B + 2 * ldb, bi3_p = B + 3 * ldb, bi4_p = B + 4 * ldb, bi5_p = B + 5 * ldb, bi6_p = B + 6 * ldb, bi7_p = B + 7 *ldb;

  for (int i = 0; i < n; ++i) {
    // for (int x = 0; x < UNROLL_NUM; ++x) {
    //   // if (old_C - C + x * ldc >= n * n) {
    //   //   printf("C too large, index: %d, n: %d\n", old_C - C + x * ldc, n);
    //   //   // return;
    //   // }
    //   // if (old_B - B + i + x * ldb >= n * n) {
    //   //   printf("B too large, index: %d, n: %d\n", old_B - B + i + x * ldb, n);
    //   //   // return;
    //   // }
    //   // if (old_A - A + i * lda >= n * n) {
    //   //   printf("A too large, index: %d, n: %d\n", old_A - A + i * lda, n);
    //   //   // return;
    //   // }
    //   C[x * ldc] += A[i * lda] * B[i + x * ldb];
    // }
    // int ilda = i * lda;
    // C[0] += A[ilda] * B[i];
    // C[ldc] += A[ilda] * B[i + ldb];
    // C[2 * ldc] += A[ilda] * B[i + 2 * ldb];
    // C[3 * ldc] += A[ilda] * B[i + 3 * ldb];
    // C[4 * ldc] += A[ilda] * B[i + 4 * ldb];
    // C[5 * ldc] += A[ilda] * B[i + 5 * ldb];
    // C[6 * ldc] += A[ilda] * B[i + 6 * ldb];
    // C[7 * ldc] += A[ilda] * B[i + 7 * ldb];
    a0i = A[i * lda];
    c00 += a0i * *bi0_p++;
    c01 += a0i * *bi1_p++;
    c02 += a0i * *bi2_p++;
    c03 += a0i * *bi3_p++;
    c04 += a0i * *bi4_p++;
    c05 += a0i * *bi5_p++;
    c06 += a0i * *bi6_p++;
    c07 += a0i * *bi7_p++;
  }
  C[0] = c00;
  C[ldc] = c01;
  C[2 * ldc] = c02;
  C[3 * ldc] = c03;
  C[4 * ldc] = c04;
  C[5 * ldc] = c05;
  C[6 * ldc] = c06;
  C[7 * ldc] = c07;
}


#define UNROLL_ROW (UNROLL_NUM / 4)

void dot_mul_square(int n, int lda, int ldb, int ldc, float *A, float *B, float *C) {
  // for (int y = 0; y < UNROLL_NUM; ++y) {
  //   for (int x = 0; x < UNROLL_NUM; ++x) {
  //     dot_mul(n, lda, A + y, B + x * ldb, C + y + x * ldc);
  //   }
  // }
  // dot_mul(n, lda, A, B, C);
  // dot_mul(n, lda, A, B + 1 * ldb, C + 1 * ldc);
  // dot_mul(n, lda, A, B + 2 * ldb, C + 2 * ldc);
  // dot_mul(n, lda, A, B + 3 * ldb, C + 3 * ldc);

  // dot_mul(n, lda, A + 1, B + 0 * ldb, C + 1 + 0 * ldc);
  // dot_mul(n, lda, A + 1, B + 1 * ldb, C + 1 + 1 * ldc);
  // dot_mul(n, lda, A + 1, B + 2 * ldb, C + 1 + 2 * ldc);
  // dot_mul(n, lda, A + 1, B + 3 * ldb, C + 1 + 3 * ldc);

  // dot_mul(n, lda, A + 2, B + 0 * ldb, C + 2 + 0 * ldc);
  // dot_mul(n, lda, A + 2, B + 1 * ldb, C + 2 + 1 * ldc);
  // dot_mul(n, lda, A + 2, B + 2 * ldb, C + 2 + 2 * ldc);
  // dot_mul(n, lda, A + 2, B + 3 * ldb, C + 2 + 3 * ldc);

  // dot_mul(n, lda, A + 3, B + 0 * ldb, C + 3 + 0 * ldc);
  // dot_mul(n, lda, A + 3, B + 1 * ldb, C + 3 + 1 * ldc);
  // dot_mul(n, lda, A + 3, B + 2 * ldb, C + 3 + 2 * ldc);
  // dot_mul(n, lda, A + 3, B + 3 * ldb, C + 3 + 3 * ldc);

  // register float c00 = 0, c01 = 0, c02 = 0, c03 = 0, c10 = 0, c11 = 0, 
  //   c12 = 0, c13 = 0, c20 = 0, c21 = 0, c22 = 0, c23 = 0, c30 = 0, c31 = 0, c32 = 0, c33 = 0;
  // register float a0i, a1i, a2i, a3i;
  // register float bi0, bi1, bi2, bi3;
  // printf("use micro kernel\n");
  // printf("init c_c3: ");
  // for(int i = 0; i < 4; ++i) {
  //   printf("%.2f ", C[3 * ldc + i]);
  // }
  // printf("\n\n");
  register float32x4_t c_c0, c_c1, c_c2, c_c3, a_c0_i0, a_c0_i1, a_c0_i2, a_c0_i3, b_vi0_0, b_vi1_0, b_vi2_0, b_vi3_0, temp_v0,  temp_v3, part1_c0, part1_c1, part1_c2, part1_c3;

#if SIMD_UNROLL == 32
  register float32x4_t part2_c0, part2_c1, part2_c2, part2_c3, part3_c0, part3_c1, part3_c2, part3_c3;
  register float32x4_t a_c0_i4, a_c0_i5, b_vi4_0, b_vi5_0;
  register float32x4_t a_c0_i6, a_c0_i7, b_vi6_0, b_vi7_0;
#endif

  register float32x4_t zero = {0};
  // float32x4_t c_ci[UNROLL_NUM * UNROLL_ROW], a_rxi[UNROLL_ROW], b_vi[UNROLL_NUM];
  // c_c0 = vmovq_n_f32(0.0), c_c1 = vmovq_n_f32(0.0), c_c2 = vmovq_n_f32(0.0), c_c3 = vmovq_n_f32(0.0);
  part1_c0 = zero; part1_c1 = zero; part1_c2 = zero; part1_c3 = zero;
#if SIMD_UNROLL == 32
  part2_c0 = zero; part2_c1 = zero; part2_c2 = zero; part2_c3 = zero;
  part3_c0 = zero; part3_c1 = zero; part3_c2 = zero; part3_c3 = zero;
#endif
  c_c0 = vld1q_f32(C + 0 * ldc); c_c1 = vld1q_f32(C + 1 * ldc);
  c_c2 = vld1q_f32(C + 2 * ldc); c_c3 = vld1q_f32(C + 3 * ldc);
  
  // for (int x = 0; x < UNROLL_ROW; ++x) {
  //   for (int y = 0; y < UNROLL_NUM; ++y) {
  //     c_ci[x * UNROLL_NUM + y] = vld1q_f32(C + x * 4 + y * ldc);
  //   }
  // }
  // register float *bi0_p, *bi1_p, *bi2_p, *bi3_p;
  // float *bix_p[UNROLL_NUM];

  // for (int x = 0; x < UNROLL_NUM; ++x) {
  //   bix_p[x] = B + x * ldb;
  // }
  // bi0_p = B; bi1_p = B + 1 * ldb; bi2_p = B + 2 * ldb; bi3_p = B + 3 * ldb;
  // printf("Packed A: ");
  // for (int j = 0; j < 20; ++j) {
  //   printf("%.2f ", A[j]);
  // }
  // printf("\n\n");
  // printf("\nPacked B: ");
  // for (int j = 0; j < 20; ++j) {
  //   printf("%.2f ", B[j]);
  // }
  // printf("\n\n");
  int i;
  for (i = 0; i + SIMD_UNROLLD4 <= n; i += SIMD_UNROLLD4) {
    // int ilda = i * lda;
    // for(int y = 0; y < UNROLL_NUM; ++y) {
    //   for(int x = 0; x < UNROLL_NUM; ++x) {
    //     C[y + x * ldc] += A[y + ilda] * B[i + x * ldb];
    //   }
    // }
    // for (int x = 0; x < UNROLL_ROW; ++ x) {
    //   a_rxi[x] = vld1q_f32(A + 4 * x + i * lda);
    // }
    a_c0_i0 = vld1q_f32(A + i * 4);
    a_c0_i1 = vld1q_f32(A + i * 4 + 4);
    b_vi0_0 = vld1q_f32(B + 0);
    b_vi1_0 = vld1q_f32(B + 4);
    temp_v0 = vfmaq_laneq_f32(c_c0, a_c0_i0, b_vi0_0, 0);
    // temp_v2 = vmulq_laneq_f32(a_c0_i1, b_vi1_0, 0);
    // c_c0 = vaddq_f32(temp_v0, temp_v2);
    c_c0 = vfmaq_laneq_f32(temp_v0, a_c0_i1, b_vi1_0, 0);

    temp_v3 = vfmaq_laneq_f32(c_c1, a_c0_i1, b_vi1_0, 1);
    // temp_v1 = vmulq_laneq_f32(a_c0_i0, b_vi0_0, 1);
    // c_c1 = vaddq_f32(temp_v1, temp_v3);
    c_c1 = vfmaq_laneq_f32(temp_v3, a_c0_i0, b_vi0_0, 1);

    temp_v0 = vfmaq_laneq_f32(c_c2, a_c0_i0, b_vi0_0, 2);
    // temp_v2 = vmulq_laneq_f32(a_c0_i1, b_vi1_0, 2);
    // c_c2 = vaddq_f32(temp_v0, temp_v2);
    c_c2 = vfmaq_laneq_f32(temp_v0, a_c0_i1, b_vi1_0, 2);

    temp_v3 = vfmaq_laneq_f32(c_c3, a_c0_i1, b_vi1_0, 3);
    // temp_v1 = vmulq_laneq_f32(a_c0_i0, b_vi0_0, 3);
    // c_c3 = vaddq_f32(temp_v1, temp_v3);
    c_c3 = vfmaq_laneq_f32(temp_v3, a_c0_i0, b_vi0_0, 3);
    

    a_c0_i2 = vld1q_f32(A + i * 4 + 8);
    a_c0_i3 = vld1q_f32(A + i * 4 + 12);
    b_vi2_0 = vld1q_f32(B + 8);
    b_vi3_0 = vld1q_f32(B + 12);
    temp_v0 = vfmaq_laneq_f32(part1_c0, a_c0_i2, b_vi2_0, 0);
    // temp_v2 = vmulq_laneq_f32(a_c0_i3, b_vi3_0, 0);
    // part1_c0 = vaddq_f32(temp_v0, temp_v2);
    part1_c0 = vfmaq_laneq_f32(temp_v0, a_c0_i3, b_vi3_0, 0);
    
    temp_v3 = vfmaq_laneq_f32(part1_c1, a_c0_i3, b_vi3_0, 1);
    // temp_v1 = vmulq_laneq_f32(a_c0_i2, b_vi2_0, 1);
    // part1_c1 = vaddq_f32(temp_v1, temp_v3);
    part1_c1 = vfmaq_laneq_f32(temp_v3, a_c0_i2, b_vi2_0, 1);

    temp_v0 = vfmaq_laneq_f32(part1_c2, a_c0_i2, b_vi2_0, 2);
    // temp_v2 = vmulq_laneq_f32(a_c0_i3, b_vi3_0, 2);
    // part1_c2 = vaddq_f32(temp_v0, temp_v2);
    part1_c2 = vfmaq_laneq_f32(temp_v0, a_c0_i3, b_vi3_0, 2);

    temp_v3 = vfmaq_laneq_f32(part1_c3, a_c0_i3, b_vi3_0, 3);
    // temp_v1 = vmulq_laneq_f32(a_c0_i2, b_vi2_0, 3);
    // part1_c3 = vaddq_f32(temp_v1, temp_v3);
    part1_c3 = vfmaq_laneq_f32(temp_v3, a_c0_i2, b_vi2_0, 3);

#if SIMD_UNROLL == 32
    a_c0_i4 = vld1q_f32(A + i * 4 + 16);
    a_c0_i5 = vld1q_f32(A + i * 4 + 20);
    b_vi4_0 = vld1q_f32(B + 16);
    b_vi5_0 = vld1q_f32(B + 20);
    temp_v0 = vfmaq_laneq_f32(part2_c0, a_c0_i4, b_vi4_0, 0);
    part2_c0 = vfmaq_laneq_f32(temp_v0, a_c0_i5, b_vi5_0, 0);
    
    temp_v3 = vfmaq_laneq_f32(part2_c1, a_c0_i5, b_vi5_0, 1);
    part2_c1 = vfmaq_laneq_f32(temp_v3, a_c0_i4, b_vi4_0, 1);

    temp_v0 = vfmaq_laneq_f32(part2_c2, a_c0_i4, b_vi4_0, 2);
    part2_c2 = vfmaq_laneq_f32(temp_v0, a_c0_i5, b_vi5_0, 2);

    temp_v3 = vfmaq_laneq_f32(part2_c3, a_c0_i5, b_vi5_0, 3);
    part2_c3 = vfmaq_laneq_f32(temp_v3, a_c0_i4, b_vi4_0, 3);


    a_c0_i6 = vld1q_f32(A + i * 4 + 24);
    a_c0_i7 = vld1q_f32(A + i * 4 + 28);
    b_vi6_0 = vld1q_f32(B + 24);
    b_vi7_0 = vld1q_f32(B + 28);
    temp_v0 = vfmaq_laneq_f32(part3_c0, a_c0_i6, b_vi6_0, 0);
    part3_c0 = vfmaq_laneq_f32(temp_v0, a_c0_i7, b_vi7_0, 0);
    
    temp_v3 = vfmaq_laneq_f32(part3_c1, a_c0_i7, b_vi7_0, 1);
    part3_c1 = vfmaq_laneq_f32(temp_v3, a_c0_i6, b_vi6_0, 1);

    temp_v0 = vfmaq_laneq_f32(part3_c2, a_c0_i6, b_vi6_0, 2);
    part3_c2 = vfmaq_laneq_f32(temp_v0, a_c0_i7, b_vi7_0, 2);

    temp_v3 = vfmaq_laneq_f32(part3_c3, a_c0_i7, b_vi7_0, 3);
    part3_c3 = vfmaq_laneq_f32(temp_v3, a_c0_i6, b_vi6_0, 3);

#endif

    // a_ri = vld1q_f32(A);
    // A += UNROLL_NUM;
    
    
    
    // b_vi0 = vld1q_dup_f32(B); b_vi1 = vld1q_dup_f32(B + 1);
    // b_vi2 = vld1q_dup_f32(B + 2); b_vi3 = vld1q_dup_f32(B + 3);
    
    // b_vi0 = vld1q_dup_f32(bi0_p); b_vi1 = vld1q_dup_f32(bi1_p); 
    // b_vi2 = vld1q_dup_f32(bi2_p); b_vi3 = vld1q_dup_f32(bi3_p);
    // for(int x = 0; x < UNROLL_NUM; ++x) {
    //   b_vi[x] = vld1q_dup_f32(bix_p[x]++);
    // }

    // a0i = A[i * lda]; a1i = A[1 + i * lda]; a2i = A[2 + i * lda]; a3i = A[3 + i * lda];
    // bi0 = *bi0_p++; bi1 = *bi1_p++; bi2 = *bi2_p++; bi3 = *bi3_p++;
     
    // c_c0 = vaddq_f32(vaddq_f32(vmulq_laneq_f32(a_c0_i0, b_vi0_0, 0), vmulq_laneq_f32(a_c0_i1, b_vi1_0, 0)) ,  vaddq_f32(vmulq_laneq_f32(a_c0_i2, b_vi2_0, 0), vfmaq_laneq_f32(c_c0, a_c0_i3, b_vi3_0, 0)));
    
    // c_c1 = vaddq_f32(vaddq_f32(vmulq_laneq_f32(a_c0_i0, b_vi0_0, 1), vmulq_laneq_f32(a_c0_i1, b_vi1_0, 1)) ,  vaddq_f32(vmulq_laneq_f32(a_c0_i2, b_vi2_0, 1), vfmaq_laneq_f32(c_c1, a_c0_i3, b_vi3_0, 1)));
    
    // c_c2 = vaddq_f32(vaddq_f32(vmulq_laneq_f32(a_c0_i0, b_vi0_0, 2), vmulq_laneq_f32(a_c0_i1, b_vi1_0, 2)) ,  vaddq_f32(vmulq_laneq_f32(a_c0_i2, b_vi2_0, 2), vfmaq_laneq_f32(c_c2, a_c0_i3, b_vi3_0, 2)));
    
    // c_c3 = vaddq_f32(vaddq_f32(vmulq_laneq_f32(a_c0_i0, b_vi0_0, 3), vmulq_laneq_f32(a_c0_i1, b_vi1_0, 3)) ,  vaddq_f32(vmulq_laneq_f32(a_c0_i2, b_vi2_0, 3), vfmaq_laneq_f32(c_c3, a_c0_i3, b_vi3_0, 3)));
    B += SIMD_UNROLL;
    // c_c0 = vmlaq_f32(c_c0, a_ri, b_vi0);
    // c_c1 = vmlaq_f32(c_c1, a_ri, b_vi1);
    // c_c2 = vmlaq_f32(c_c2, a_ri, b_vi2);
    // c_c3 = vmlaq_f32(c_c3, a_ri, b_vi3);

    

    // bi0_p += UNROLL_NUM; bi1_p += UNROLL_NUM; bi2_p += UNROLL_NUM; bi3_p += UNROLL_NUM;

    // c00 += a0i * bi0;
    // c10 += a1i * bi0;
    // c20 += a2i * bi0;
    // c30 += a3i * bi0;

    // c01 += a0i * bi1;
    // c11 += a1i * bi1;
    // c21 += a2i * bi1;
    // c31 += a3i * bi1;

    // c02 += a0i * bi2;
    // c12 += a1i * bi2;
    // c22 += a2i * bi2;
    // c32 += a3i * bi2;

    // c03 += a0i * bi3;
    // c13 += a1i * bi3;
    // c23 += a2i * bi3;
    // c33 += a3i * bi3;

    // C[0] += A[i * lda] * B[i];
    // C[ldc] += A[i * lda] * B[i + 1 * ldb];
    // C[2 * ldc] += A[i * lda] * B[i + 2 * ldb];
    // C[3 * ldc] += A[i * lda] * B[i + 3 * ldb];

    // C[1 + 0 * ldc] += A[1 + i * lda] * B[i + 0 * ldb];
    // C[1 + 1 * ldc] += A[1 + i * lda] * B[i + 1 * ldb];
    // C[1 + 2 * ldc] += A[1 + i * lda] * B[i + 2 * ldb];
    // C[1 + 3 * ldc] += A[1 + i * lda] * B[i + 3 * ldb];

    // C[2 + 0 * ldc] += A[2 + i * lda] * B[i + 0 * ldb];
    // C[2 + 1 * ldc] += A[2 + i * lda] * B[i + 1 * ldb];
    // C[2 + 2 * ldc] += A[2 + i * lda] * B[i + 2 * ldb];
    // C[2 + 3 * ldc] += A[2 + i * lda] * B[i + 3 * ldb];

    // C[3 + 0 * ldc] += A[3 + i * lda] * B[i + 0 * ldb];
    // C[3 + 1 * ldc] += A[3 + i * lda] * B[i + 1 * ldb];
    // C[3 + 2 * ldc] += A[3 + i * lda] * B[i + 2 * ldb];
    // C[3 + 3 * ldc] += A[3 + i * lda] * B[i + 3 * ldb];
  }

#if SIMD_UNROLL == 32
  c_c0 = c_c0 + part1_c0 + part2_c0 + part3_c0;
  c_c1 = c_c1 + part1_c1 + part2_c1 + part3_c1;
  c_c2 = c_c2 + part1_c2 + part2_c2 + part3_c2;
  c_c3 = c_c3 + part1_c3 + part2_c3 + part3_c3;
#else
  c_c0 = vaddq_f32(part1_c0, c_c0);
  c_c1 = vaddq_f32(part1_c1, c_c1);
  c_c2 = vaddq_f32(part1_c2, c_c2);
  c_c3 = vaddq_f32(part1_c3, c_c3);
#endif

  float32x4_t b_vi0, b_vi1, b_vi2, b_vi3;
  for(; i < n; ++i) {
    a_c0_i0 = vld1q_f32(A + i * 4);
    b_vi0 = vld1q_dup_f32(B); b_vi1 = vld1q_dup_f32(B + 1);
    b_vi2 = vld1q_dup_f32(B + 2); b_vi3 = vld1q_dup_f32(B + 3);
    B += 4;
    c_c0 = vmlaq_f32(c_c0, a_c0_i0, b_vi0);
    c_c1 = vmlaq_f32(c_c1, a_c0_i0, b_vi1);
    c_c2 = vmlaq_f32(c_c2, a_c0_i0, b_vi2);
    c_c3 = vmlaq_f32(c_c3, a_c0_i0, b_vi3);
  }
  // for (int x = 0; x < UNROLL_ROW; ++x) {
  //   for (int y = 0; y < UNROLL_NUM; ++y) {
  //     vst1q_f32(C + 4 * x + y * ldc, c_ci[x * UNROLL_NUM + y]);
  //   }
  // }
  
  vst1q_f32(C + 0 * ldc, c_c0); vst1q_f32(C + 1 * ldc, c_c1);
  vst1q_f32(C + 2 * ldc, c_c2); vst1q_f32(C + 3 * ldc, c_c3);
  // printf("c_c3: ");
  // for(int i = 0; i < 4; ++i) {
  //   printf("%.2f ", C[3 * ldc + i]);
  // }
  // printf("\n\n");
  
  // C[0 + 0 * ldc] += c00;  C[0 + 1 * ldc] += c01;  C[0 + 2 * ldc] += c02;  C[0 + 3 * ldc] += c03;
  // C[1 + 0 * ldc] += c10;  C[1 + 1 * ldc] += c11;  C[1 + 2 * ldc] += c12;  C[1 + 3 * ldc] += c13;
  // C[2 + 0 * ldc] += c20;  C[2 + 1 * ldc] += c21;  C[2 + 2 * ldc] += c22;  C[2 + 3 * ldc] += c23;
  // C[3 + 0 * ldc] += c30;  C[3 + 1 * ldc] += c31;  C[3 + 2 * ldc] += c32;  C[3 + 3 * ldc] += c33;
}

static void pack_left_A(int K, int lda, float *A, float *packed_A) {
  // double *dst = (double*)packed_A;
  float *dst = packed_A;
  // double *src = (double*)A;
  int k;
  for (k = 0; k + SIMD_UNROLLD4 <= K; k += SIMD_UNROLLD4) {
    // double *a0k_p = (double*)(A + k * lda);
    // *dst++ = *a0k_p;
    // *dst++ = *(a0k_p+1);
    float *a0_k0_p = A + k * lda, *a0_k1_p = A + (k + 1) * lda, *a0_k2_p = A + (k + 2) * lda, *a0_k3_p = A + (k + 3) * lda;
#if SIMD_UNROLL == 32
    float *a0_k4_p = A + (k + 4) * lda, *a0_k5_p =  A + (k + 5) * lda, *a0_k6_p = A + (k + 6) * lda, *a0_k7_p = A + (k + 7) * lda;
#endif
    // for (int x = 0; x < 4; ++ x) {
      *(dst + 0) = *(a0_k0_p + 0);
      *(dst + 1) = *(a0_k0_p + 1);
      *(dst + 2) = *(a0_k0_p + 2);
      *(dst + 3) = *(a0_k0_p + 3);
      *(dst + 4) = *(a0_k1_p + 0);
      *(dst + 5) = *(a0_k1_p + 1);
      *(dst + 6) = *(a0_k1_p + 2);
      *(dst + 7) = *(a0_k1_p + 3);
      *(dst + 8) = *(a0_k2_p + 0);
      *(dst + 9) = *(a0_k2_p + 1);
      *(dst + 10) = *(a0_k2_p + 2);
      *(dst + 11) = *(a0_k2_p + 3);
      *(dst + 12) = *(a0_k3_p + 0);
      *(dst + 13) = *(a0_k3_p + 1);
      *(dst + 14) = *(a0_k3_p + 2);
      *(dst + 15) = *(a0_k3_p + 3);
#if SIMD_UNROLL == 32
      *(dst + 16) = *(a0_k4_p + 0);
      *(dst + 17) = *(a0_k4_p + 1);
      *(dst + 18) = *(a0_k4_p + 2);
      *(dst + 19) = *(a0_k4_p + 3);
      *(dst + 20) = *(a0_k5_p + 0);
      *(dst + 21) = *(a0_k5_p + 1);
      *(dst + 22) = *(a0_k5_p + 2);
      *(dst + 23) = *(a0_k5_p + 3);
      *(dst + 24) = *(a0_k6_p + 0);
      *(dst + 25) = *(a0_k6_p + 1);
      *(dst + 26) = *(a0_k6_p + 2);
      *(dst + 27) = *(a0_k6_p + 3);
      *(dst + 28) = *(a0_k7_p + 0);
      *(dst + 29) = *(a0_k7_p + 1);
      *(dst + 30) = *(a0_k7_p + 2);
      *(dst + 31) = *(a0_k7_p + 3);
#endif
      dst += SIMD_UNROLL;
    // }
  }

  for (; k < K; k++) {
    *(dst + 0) = A[0 + k * lda];
    *(dst + 1) = A[1 + k * lda];
    *(dst + 2) = A[2 + k * lda];
    *(dst + 3) = A[3 + k * lda];
    dst += 4;
  }
}

static void pack_right_B(int K, int ldb, float *B, float *packed_B) {
  float *dst = packed_B;
  // float *bik_p = B;
  // float *bi0_0_p = B + 0, *bi0_1_p = B + 0 + 1 * ldb, * bi0_2_p = B + 0 + 2 * ldb, *bi0_3_p = B + 0 + 3 * ldb;
  // float *bi1_0_p = B + 1, *bi1_1_p = B + 1 + 1 * ldb, * bi1_2_p = B + 1 + 2 * ldb, *bi1_3_p = B + 1 + 3 * ldb;
  // float *bi2_0_p = B + 2, *bi2_1_p = B + 2 + 1 * ldb, * bi2_2_p = B + 2 + 2 * ldb, *bi2_3_p = B + 2 + 3 * ldb;
  // float *bi3_0_p = B + 3, *bi3_1_p = B + 3 + 1 * ldb, * bi3_2_p = B + 3 + 2 * ldb, *bi3_3_p = B + 3 + 3 * ldb;
  // float *bix_p[UNROLL_NUM];
  // for (int x = 0; x < UNROLL_NUM; ++x) {
  //   bix_p[x] = B + x * ldb;
  // }
  int k;
  for(k = 0; k + SIMD_UNROLLD4 <= K; k += SIMD_UNROLLD4) {
      // for (int x = 0; x < UNROLL_NUM; ++x) {
      //   *dst++ = *(bik_p + x * ldb);
      // }
      // bik_p++;
      // for (int x = 0; x < UNROLL_NUM; ++x) {
      //   *dst++ = *bix_p[x]++;
      // }
      *(dst + 0) = *(B + k + 0 + 0 * ldb);
      *(dst + 1) = *(B + k + 0 + 1 * ldb);
      *(dst + 2) = *(B + k + 0 + 2 * ldb);
      *(dst + 3) = *(B + k + 0 + 3 * ldb);
      *(dst + 4) = *(B + k + 1 + 0 * ldb);
      *(dst + 5) = *(B + k + 1 + 1 * ldb);
      *(dst + 6) = *(B + k + 1 + 2 * ldb);
      *(dst + 7) = *(B + k + 1 + 3 * ldb);
      *(dst + 8) = *(B + k + 2 + 0 * ldb);
      *(dst + 9) = *(B + k + 2 + 1 * ldb);
      *(dst + 10) = *(B + k + 2 + 2 * ldb);
      *(dst + 11) = *(B + k + 2 + 3 * ldb);
      *(dst + 12) = *(B + k + 3 + 0 * ldb);
      *(dst + 13) = *(B + k + 3 + 1 * ldb);
      *(dst + 14) = *(B + k + 3 + 2 * ldb);
      *(dst + 15) = *(B + k + 3 + 3 * ldb);
#if SIMD_UNROLL == 32
      *(dst + 16) = *(B + k + 4 + 0 * ldb);
      *(dst + 17) = *(B + k + 4 + 1 * ldb);
      *(dst + 18) = *(B + k + 4 + 2 * ldb);
      *(dst + 19) = *(B + k + 4 + 3 * ldb);
      *(dst + 20) = *(B + k + 5 + 0 * ldb);
      *(dst + 21) = *(B + k + 5 + 1 * ldb);
      *(dst + 22) = *(B + k + 5 + 2 * ldb);
      *(dst + 23) = *(B + k + 5 + 3 * ldb);
      *(dst + 24) = *(B + k + 6 + 0 * ldb);
      *(dst + 25) = *(B + k + 6 + 1 * ldb);
      *(dst + 26) = *(B + k + 6 + 2 * ldb);
      *(dst + 27) = *(B + k + 6 + 3 * ldb);
      *(dst + 28) = *(B + k + 7 + 0 * ldb);
      *(dst + 29) = *(B + k + 7 + 1 * ldb);
      *(dst + 30) = *(B + k + 7 + 2 * ldb);
      *(dst + 31) = *(B + k + 7 + 3 * ldb);
#endif
      dst += SIMD_UNROLL;
      // *(dst + 0) = *bi0_0_p++;
      // *(dst + 1) = *bi1_0_p++;
      // *(dst + 2) = *bi2_0_p++;
      // *(dst + 3) = *bi3_0_p++;
      // *(dst + 4) = *bi0_1_p++;
      // *(dst + 5) = *bi1_1_p++;
      // *(dst + 6) = *bi2_1_p++;
      // *(dst + 7) = *bi3_1_p++;
      // *(dst + 8) = *bi0_2_p++;
      // *(dst + 9) = *bi1_2_p++;
      // *(dst + 10) = *bi2_2_p++;
      // *(dst + 11) = *bi3_2_p++;
      // *(dst + 12) = *bi0_3_p++;
      // *(dst + 13) = *bi1_3_p++;
      // *(dst + 14) = *bi2_3_p++;
      // *(dst + 15) = *bi3_3_p++;
      
  }
  // printf("in Pack B, k: %d, K: %d, ldb: %d\n", k, K, ldb);
  for (; k < K; ++k) {
    *(dst + 0) = *(B + k + 0 * ldb);
    *(dst + 1) = *(B + k + 1 * ldb);
    *(dst + 2) = *(B + k + 2 * ldb);
    *(dst + 3) = *(B + k + 3 * ldb);
    dst += 4;
  }
}

static void do_block(int M, int N, int K, int lda,int ldb, int ldc, float *A, float *B, float *C, float *packed_A, float *packed_B, int should_pack_B) {
  int j;
  for (j = 0; j + UNROLL_NUM <= N; j += UNROLL_NUM) {
    if (should_pack_B) {
      pack_right_B(K, ldb, B + j * ldb, packed_B + j * K);
    }
    // int jn = j * ldb;
    int i;
    for (i = 0; i + UNROLL_NUM <= M; i += UNROLL_NUM) {
      
      // for(int x = 0; x < UNROLL_NUM; ++x) {
      //   int jxn = jn + x * n, ijxn = jn + x * n + i;
      //   dot_mul(n, n, A + i, B + jxn, C + ijxn);
      // }
      // dot_mul(n, n, A + i, B + jn, C + i + jn);
      // dot_mul(n, n, A + i, B + jn + n, C + i + jn + n);
      // dot_mul(n, n, A + i, B + jn + 2 * n, C + i + jn + 2 * n);
      // dot_mul(n, n, A + i, B + jn + 3 * n, C + i + jn + 3 * n);

      // dot_mul_unroll(n, n, n, n, A + i, B + j * n, C + i + j * n, A, B, C);
      if (j == 0) {
        pack_left_A(K, lda, A + i, packed_A + i * K);
      }
      // dot_mul_square(K, lda, ldb, ldc, A + i, B + j * ldb, C + i + j * ldc);
      dot_mul_square(K, SIMD_UNROLL, 1, ldc, packed_A + i * K, packed_B + j * K, C + i + j * ldc);
      // dot_mul(n, n, A + i, B + j * n, C + i + j * n);
    }
    // if (i < M && i + UNROLL_NUM > M) {
    //   // printf("use addtional do_mul0, i = %d, j = %d\n", i, j);
    //   for (int temp_j = j; temp_j < j + UNROLL_NUM; ++temp_j) {
    //     for (int temp_i = i; temp_i < M; ++temp_i) {
    //       dot_mul(K, lda, A + temp_i, B + temp_j * ldb, C + temp_i + temp_j * ldc);
    //     }
    //   }
    // }
    
    
  }
  // for (; j < N; ++j) {
  //   // printf("use addtional do_mul1, j = %d\n", j);
  //   for (int i = 0; i < M; ++i) {
  //     dot_mul(K, lda,  A + i, B + j * ldb, C + i + j * ldc);
  //   }
  // }
}

static void print_matrix(int row, int col, float *A) {
  for(int i = 0; i < row; ++i) {
    for(int j = 0; j < col; ++j) {
      printf("%0.2f ", A[i + j * row]);
    }
    printf("\n");
  }
}

int should_padding(int m, int n, int *new_m_ptr, int *new_n_ptr) {
  *new_m_ptr = (m + UNROLL_NUM - 1) / UNROLL_NUM * UNROLL_NUM;
  *new_n_ptr = (n + UNROLL_NUM - 1) / UNROLL_NUM * UNROLL_NUM;
  return m != *new_m_ptr || n != *new_n_ptr;
}

static int64_t tempDur0 = 0, tempDur1 = 0;

float *get_padding_matrix(int lda, int m, int n, int new_m, int new_n, const float *A) {
  int64_t tempT0 = wall_time_ns();
  float *new_A = NULL;
  int ret = posix_memalign((void**)&new_A, 4096, sizeof(float) * new_m * new_n);
  int64_t tempT1 = wall_time_ns();
  if (ret != 0) {
    fprintf(stderr, "Can not align malloc padding matrix!\n");
    exit(-1);
  }
  int j;
  for (j = 0; j < n; ++j) {
    memcpy(new_A + j * new_m, A + j * lda, sizeof(float) * m);
    memset(new_A + m + j * new_m, 0, sizeof(float) * (new_m - m));
  }
  for (;j < new_n; ++j) {
    memset(new_A + j * new_m, 0, sizeof(float) * new_m);
  }
  int64_t tempT2 = wall_time_ns();
  tempDur0 += tempT1 - tempT0;
  tempDur1 += tempT2 - tempT1;
  return new_A;
}

void back_padding(int lda, int m, int n, int new_m, int new_n, float *A, float *padding_A) {
  for (int j = 0; j < n; ++j) {
    memcpy(A + j * lda, padding_A + j * new_m, sizeof(float) * m);
  }
}

void square_gemm (int n, float* A, float* B, float* C) {
  // posix_memalign
  
  
  // float *packed_B = packed_A + BLOCK_COL * BLOCK_ROW;
  tempDur0 = 0; tempDur1 = 0;
  int64_t t0 = wall_time_ns();

  float *padding_A = A, *padding_B = B, *padding_C = C;
  int newM, newN, newK;
  int should_pad_A = should_padding(n, n, &newM, &newK);
  int should_pad_B = should_padding(n, n, &newK, &newN); 
  int should_pad_C = should_padding(n, n, &newM, &newN);

  // float *packed_A = malloc(BLOCK_COL * BLOCK_ROW * sizeof(float));
  // float *packed_B = malloc(BLOCK_COL * n * sizeof(float));
  
  float *packed_A, *packed_B;
  int tempRet1 = posix_memalign((void**)&packed_A, 4096, BLOCK_COL * BLOCK_ROW * sizeof(float));
  int tempRet2 = posix_memalign((void**)&packed_B, 4096, BLOCK_COL * newN * sizeof(float));

  int64_t t1 = wall_time_ns();

  if (tempRet1 != 0|| tempRet2 != 0) {
    fprintf(stderr, "Can not align malloc packed pool!\n");
    exit(-1);
  }
  
  if (should_pad_A) {
    padding_A = get_padding_matrix(n, n, n, newM, newK, A);
  }
  if (should_pad_B) {
    padding_B = get_padding_matrix(n, n, n, newK, newN, B);
  }
  if (should_pad_C) {
    padding_C= get_padding_matrix(n, n, n, newM, newN, C);
  }

  int64_t t2 = wall_time_ns();

  for (int k = 0; k < newK; k += BLOCK_COL) {
    int K = min(newK - k, BLOCK_COL);
    for (int i = 0; i < newM; i += BLOCK_ROW) {
      int M = min(newM - i, BLOCK_ROW);
      int N = newN;
      // printf("do block, M: %d, N: %d, K: %d, A(%d, %d), B(%d, %d), C(%d, %d)\n",
      //         M, N, K, i, k, k, 0, i, 0);
      
      do_block(M, N, K, newM, newK, newM, padding_A + i + k * newM, padding_B + k, padding_C + i, packed_A, packed_B, i == 0);
      // printf("C:\n");
      // print_matrix(n, n, C);
    }
  }

  int64_t t3 = wall_time_ns();

  free(packed_A);
  free(packed_B);

  int64_t t4 = wall_time_ns();

  if (should_pad_A)
    free(padding_A);
  if (should_pad_B)
    free(padding_B);

  int64_t t5 = wall_time_ns();

  if (should_pad_C) {
    back_padding(n, n, n, newM, newN, C, padding_C);
    free(padding_C);
  }

  int64_t t6 = wall_time_ns();

  double totalDur = (t6 - t0) / (1e+6), dur1 = (t1 - t0) / (1e+6), dur2 = (t2 - t1) / (1e+6), dur3 = (t3 - t2)/ (1e+6), dur4 = (t4 - t3) / (1e+6), dur5 = (t5 - t4) / (1e+6), dur6 = (t6 - t5) / (1e+6);
  double dur2_part1 = tempDur0 / (1e+6), dur2_part2 = tempDur1 / (1e+6);
  // printf("Total: %.4f ms,  dur1: %.4f ms, dur2: %.4f ms = (%.4f + %.4f) ms, dur3: %.4f ms, dur4: %.4f ms, dur5: %.4f ms, dur6: %.4f ms\n", totalDur,  dur1, dur2, dur2_part1, dur2_part2, dur3, dur4, dur5, dur6);

}

/* This routine performs a gemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */    
// void square_gemm (int n, float* A, float* B, float* C)
// {
//   /* For each row i of A */
  
//   // naive_transpose(n, n, A);
//   // naive_transpose(n, n, B);
  

//   int j;
//   for (j = 0; j + UNROLL_NUM <= n; j += UNROLL_NUM) {
//     int jn = j * n;
//     int i;
//     for (i = 0; i + UNROLL_NUM <= n; i += UNROLL_NUM) {
      
//       // for(int x = 0; x < UNROLL_NUM; ++x) {
//       //   int jxn = jn + x * n, ijxn = jn + x * n + i;
//       //   dot_mul(n, n, A + i, B + jxn, C + ijxn);
//       // }
//       // dot_mul(n, n, A + i, B + jn, C + i + jn);
//       // dot_mul(n, n, A + i, B + jn + n, C + i + jn + n);
//       // dot_mul(n, n, A + i, B + jn + 2 * n, C + i + jn + 2 * n);
//       // dot_mul(n, n, A + i, B + jn + 3 * n, C + i + jn + 3 * n);

//       // dot_mul_unroll(n, n, n, n, A + i, B + j * n, C + i + j * n, A, B, C);

//       dot_mul_square(n, n, n, n, A + i, B + j * n, C + i + j * n);

//       // dot_mul(n, n, A + i, B + j * n, C + i + j * n);
//     }
//     if (i < n && i + UNROLL_NUM > n) {
//       for (int temp_j = j; temp_j < j + UNROLL_NUM; ++temp_j) {
//         for (int temp_i = i; temp_i < n; ++temp_i) {
//           dot_mul(n, n, A + temp_i, B + temp_j * n, C + temp_i + temp_j * n);
//         }
//       }
//     }
    
//   }
//   for (; j < n; ++j) {
//     for (int i = 0; i < n; ++i) {
//       dot_mul(n, n,  A + i, B + j * n, C + i + j * n);
//     }
//   }

//   // for (int i = 0; i < n; ++i) {
//   //   /* For each column j of B */
//   //   int in = i * n;
//   //   for (int j = 0; j < n; ++j) 
//   //   {
//   //     /* Compute C(i,j) */
//   //     // float cij = C[i+j*n];
//   //     int jn = j * n;
//   //     // for( int k = 0; k < n; k++ )
// 	//     //   // cij += A[i+k*n] * B[k+j*n];
//   //     //   cij += A[k + i * n] * B[k + j * n];
//   //     //   // cij += A[i + k * n] * B[j + k * n];
//   //     // C[i+j*n] = cij;

//   //     // float32x4_t buf[UNROLL_NUM] = {0};
//   //     // int k;
//   //     // for (k = 0; k < ((n) & (~7)); k += 4 * UNROLL_NUM) {
//   //     //   int kin = k + in, kjn = k + jn;
//   //     //   for (int x = 0; x < UNROLL_NUM; ++x) {
//   //     //     float32x4_t v1 = vld1q_f32(A + x * 4 + kin);
//   //     //     float32x4_t v2 = vld1q_f32(B + x * 4 + kjn);
//   //     //     // float32x4_t r1 = vmulq_f32(v1, v2);
//   //     //     // buf[x] = vaddq_f32(buf[x], r1);
//   //     //     buf[x] = vmlaq_f32(buf[x], v1, v2);
//   //     //   }
        
//   //     // }
//   //     // float temp = 0;
//   //     // for (k; k < n; ++k) {
//   //     //   temp += A[k + in] * B[k + jn];
//   //     // }
//   //     // float res[4];
      
      
//   //     // for (int x = 0; x < UNROLL_NUM; ++x) {
//   //     //   vst1q_f32(res, buf[x]);
//   //     //   temp += res[0] + res[1] + res[2] + res[3];
//   //     // }
//   //     // C[i + j * n] += temp; 

//   //   //   float cij = C[i * n + j];
//   //   //   for (int k = 0; k < n; ++k) {
//   //   //     cij += A[i * n + k] * B[k * n + j];
//   //   //   }
//   //   //   C[i * n + j] = cij;
//   //   }
//   // }
//     // naive_transpose(n, n, A);
// }
