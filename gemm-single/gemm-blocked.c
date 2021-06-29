#include "arm_neon.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

const char* gemm_desc = "Simple blocked gemm.";

#define BLOCK_ROW 64
#define BLOCK_COL 128

#define UNROLL_NUM 4

#define SIMD_UNROLL 32
#define SIMD_UNROLLD4 8

static void print_matrix(int row, int col, float *A) {
  for(int i = 0; i < row; ++i) {
    for(int j = 0; j < col; ++j) {
      printf("%0.2f ", A[i + j * row]);
    }
    printf("\n");
  }
}

#if !defined(BLOCK_SIZE)
#define BLOCK_SIZE 96
#endif

#define min(a,b) (((a)<(b))?(a):(b))


void dot_mul(int n, int incX, float *A, float *B, float *C) {
  for (int i = 0; i < n; ++i) {
    *C += A[i * incX] * B[i];
  }
}

#define UNROLL_NUM 4

#define UNROLL_ROW (UNROLL_NUM / 4)

void dot_mul_square(int n, int lda, int ldb, int ldc, float *A, float *B, float *C) {
  register float32x4_t c_c0, c_c1, c_c2, c_c3, a_c0_i0, a_c0_i1, a_c0_i2, a_c0_i3, b_vi0_0, b_vi1_0, b_vi2_0, b_vi3_0, temp_v0,  temp_v3, part1_c0, part1_c1, part1_c2, part1_c3;

#if SIMD_UNROLL == 32
  register float32x4_t part2_c0, part2_c1, part2_c2, part2_c3, part3_c0, part3_c1, part3_c2, part3_c3;
  register float32x4_t a_c0_i4, a_c0_i5, b_vi4_0, b_vi5_0;
  register float32x4_t a_c0_i6, a_c0_i7, b_vi6_0, b_vi7_0;
#endif

  register float32x4_t zero = {0};
  part1_c0 = zero; part1_c1 = zero; part1_c2 = zero; part1_c3 = zero;
#if SIMD_UNROLL == 32
  part2_c0 = zero; part2_c1 = zero; part2_c2 = zero; part2_c3 = zero;
  part3_c0 = zero; part3_c1 = zero; part3_c2 = zero; part3_c3 = zero;
#endif
  c_c0 = vld1q_f32(C + 0 * ldc); c_c1 = vld1q_f32(C + 1 * ldc);
  c_c2 = vld1q_f32(C + 2 * ldc); c_c3 = vld1q_f32(C + 3 * ldc);

  int i;
  for (i = 0; i + SIMD_UNROLLD4 <= n; i += SIMD_UNROLLD4) {
    a_c0_i0 = vld1q_f32(A + i * 4);
    a_c0_i1 = vld1q_f32(A + i * 4 + 4);
    b_vi0_0 = vld1q_f32(B + 0);
    b_vi1_0 = vld1q_f32(B + 4);
    temp_v0 = vfmaq_laneq_f32(c_c0, a_c0_i0, b_vi0_0, 0);
    c_c0 = vfmaq_laneq_f32(temp_v0, a_c0_i1, b_vi1_0, 0);

    temp_v3 = vfmaq_laneq_f32(c_c1, a_c0_i1, b_vi1_0, 1);
    c_c1 = vfmaq_laneq_f32(temp_v3, a_c0_i0, b_vi0_0, 1);

    temp_v0 = vfmaq_laneq_f32(c_c2, a_c0_i0, b_vi0_0, 2);
    c_c2 = vfmaq_laneq_f32(temp_v0, a_c0_i1, b_vi1_0, 2);

    temp_v3 = vfmaq_laneq_f32(c_c3, a_c0_i1, b_vi1_0, 3);
    c_c3 = vfmaq_laneq_f32(temp_v3, a_c0_i0, b_vi0_0, 3);
    

    a_c0_i2 = vld1q_f32(A + i * 4 + 8);
    a_c0_i3 = vld1q_f32(A + i * 4 + 12);
    b_vi2_0 = vld1q_f32(B + 8);
    b_vi3_0 = vld1q_f32(B + 12);
    temp_v0 = vfmaq_laneq_f32(part1_c0, a_c0_i2, b_vi2_0, 0);
    part1_c0 = vfmaq_laneq_f32(temp_v0, a_c0_i3, b_vi3_0, 0);
    
    temp_v3 = vfmaq_laneq_f32(part1_c1, a_c0_i3, b_vi3_0, 1);
    part1_c1 = vfmaq_laneq_f32(temp_v3, a_c0_i2, b_vi2_0, 1);

    temp_v0 = vfmaq_laneq_f32(part1_c2, a_c0_i2, b_vi2_0, 2);
    part1_c2 = vfmaq_laneq_f32(temp_v0, a_c0_i3, b_vi3_0, 2);

    temp_v3 = vfmaq_laneq_f32(part1_c3, a_c0_i3, b_vi3_0, 3);
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
    B += SIMD_UNROLL;
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
  
  vst1q_f32(C + 0 * ldc, c_c0); vst1q_f32(C + 1 * ldc, c_c1);
  vst1q_f32(C + 2 * ldc, c_c2); vst1q_f32(C + 3 * ldc, c_c3);
}

static void pack_left_A(int K, int lda, float *A, float *packed_A) {
  float *dst = packed_A;
  int k;
  for (k = 0; k + SIMD_UNROLLD4 <= K; k += SIMD_UNROLLD4) {
    float *a0_k0_p = A + k * lda, *a0_k1_p = A + (k + 1) * lda, *a0_k2_p = A + (k + 2) * lda, *a0_k3_p = A + (k + 3) * lda;
#if SIMD_UNROLL == 32
    float *a0_k4_p = A + (k + 4) * lda, *a0_k5_p =  A + (k + 5) * lda, *a0_k6_p = A + (k + 6) * lda, *a0_k7_p = A + (k + 7) * lda;
#endif
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
  int k;
  for(k = 0; k + SIMD_UNROLLD4 <= K; k += SIMD_UNROLLD4) {
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
      
  }
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
    int i;
    for (i = 0; i + UNROLL_NUM <= M; i += UNROLL_NUM) {
      if (j == 0) {
        pack_left_A(K, lda, A + i, packed_A + i * K);
      }
      dot_mul_square(K, SIMD_UNROLL, 1, ldc, packed_A + i * K, packed_B + j * K, C + i + j * ldc);
    }
    
    
  }
}


int should_padding(int m, int n, int *new_m_ptr, int *new_n_ptr) {
  *new_m_ptr = (m + UNROLL_NUM - 1) / UNROLL_NUM * UNROLL_NUM;
  *new_n_ptr = (n + UNROLL_NUM - 1) / UNROLL_NUM * UNROLL_NUM;
  return m != *new_m_ptr || n != *new_n_ptr;
}

static int64_t tempDur0 = 0, tempDur1 = 0;

float *get_padding_matrix(int lda, int m, int n, int new_m, int new_n, const float *A) {
  float *new_A = NULL;
  int ret = posix_memalign((void**)&new_A, 4096, sizeof(float) * new_m * new_n);
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
  return new_A;
}

void back_padding(int lda, int m, int n, int new_m, int new_n, float *A, float *padding_A) {
  for (int j = 0; j < n; ++j) {
    memcpy(A + j * lda, padding_A + j * new_m, sizeof(float) * m);
  }
}



void square_gemm (int n, float* A, float* B, float* C) {

  float *padding_A = A, *padding_B = B, *padding_C = C;
  int newM, newN, newK;
  int should_pad_A = should_padding(n, n, &newM, &newK);
  int should_pad_B = should_padding(n, n, &newK, &newN); 
  int should_pad_C = should_padding(n, n, &newM, &newN);

  
  float *packed_A, *packed_B;
  int tempRet1 = posix_memalign((void**)&packed_A, 4096, BLOCK_COL * BLOCK_ROW * sizeof(float));
  int tempRet2 = posix_memalign((void**)&packed_B, 4096, BLOCK_COL * newN * sizeof(float));


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


  for (int k = 0; k < newK; k += BLOCK_COL) {
    int K = min(newK - k, BLOCK_COL);
    for (int i = 0; i < newM; i += BLOCK_ROW) {
      int M = min(newM - i, BLOCK_ROW);
      int N = newN;
      
      do_block(M, N, K, newM, newK, newM, padding_A + i + k * newM, padding_B + k, padding_C + i, packed_A, packed_B, i == 0);
    }
  }


  free(packed_A);
  free(packed_B);


  if (should_pad_A)
    free(padding_A);
  if (should_pad_B)
    free(padding_B);


  if (should_pad_C) {
    back_padding(n, n, n, newM, newN, C, padding_C);
    free(padding_C);
  }

}