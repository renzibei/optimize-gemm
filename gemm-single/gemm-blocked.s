
gemm-blocked.o:     file format elf64-littleaarch64


Disassembly of section .text:

0000000000000000 <square_gemm>:
   0:	2a0003eb 	mov	w11, w0
   4:	7100001f 	cmp	w0, #0x0
   8:	aa0203ef 	mov	x15, x2
   c:	5400092d 	b.le	130 <square_gemm+0x130>
  10:	51000400 	sub	w0, w0, #0x1
  14:	531f7967 	lsl	w7, w11, #1
  18:	11000566 	add	w6, w11, #0x1
  1c:	121f7800 	and	w0, w0, #0xfffffffe
  20:	51000968 	sub	w8, w11, #0x2
  24:	aa0103ec 	mov	x12, x1
  28:	aa0303ed 	mov	x13, x3
  2c:	4b000108 	sub	w8, w8, w0
  30:	937e7ce7 	sbfiz	x7, x7, #2, #32
  34:	93407cc6 	sxtw	x6, w6
  38:	2a0b03e9 	mov	w9, w11
  3c:	93407d64 	sxtw	x4, w11
  40:	d280000e 	mov	x14, #0x0                   	// #0
  44:	d503201f 	nop
  48:	7100013f 	cmp	w9, #0x0
  4c:	5400066d 	b.le	118 <square_gemm+0x118>
  50:	aa0f03ea 	mov	x10, x15
  54:	aa0d03e0 	mov	x0, x13
  58:	2a0b03e5 	mov	w5, w11
  5c:	d503201f 	nop
  60:	aa0a03e2 	mov	x2, x10
  64:	aa0c03e1 	mov	x1, x12
  68:	2a0b03e3 	mov	w3, w11
  6c:	d503201f 	nop
  70:	710000bf 	cmp	w5, #0x0
  74:	540003ed 	b.le	f0 <square_gemm+0xf0>
  78:	bd400021 	ldr	s1, [x1]
  7c:	7100047f 	cmp	w3, #0x1
  80:	bd400042 	ldr	s2, [x2]
  84:	bd400000 	ldr	s0, [x0]
  88:	1f010040 	fmadd	s0, s2, s1, s0
  8c:	5400054c 	b.gt	134 <square_gemm+0x134>
  90:	bd000000 	str	s0, [x0]
  94:	710004bf 	cmp	w5, #0x1
  98:	540000cd 	b.le	b0 <square_gemm+0xb0>
  9c:	bc647842 	ldr	s2, [x2, x4, lsl #2]
  a0:	bd400021 	ldr	s1, [x1]
  a4:	bc647800 	ldr	s0, [x0, x4, lsl #2]
  a8:	1f010040 	fmadd	s0, s2, s1, s0
  ac:	bc247800 	str	s0, [x0, x4, lsl #2]
  b0:	7100053f 	cmp	w9, #0x1
  b4:	540001ed 	b.le	f0 <square_gemm+0xf0>
  b8:	bd400042 	ldr	s2, [x2]
  bc:	7100047f 	cmp	w3, #0x1
  c0:	bd400421 	ldr	s1, [x1, #4]
  c4:	bd400400 	ldr	s0, [x0, #4]
  c8:	1f010040 	fmadd	s0, s2, s1, s0
  cc:	5400050c 	b.gt	16c <square_gemm+0x16c>
  d0:	bd000400 	str	s0, [x0, #4]
  d4:	710004bf 	cmp	w5, #0x1
  d8:	540000cd 	b.le	f0 <square_gemm+0xf0>
  dc:	bc647841 	ldr	s1, [x2, x4, lsl #2]
  e0:	bd400422 	ldr	s2, [x1, #4]
  e4:	bc667800 	ldr	s0, [x0, x6, lsl #2]
  e8:	1f010040 	fmadd	s0, s2, s1, s0
  ec:	bc267800 	str	s0, [x0, x6, lsl #2]
  f0:	51000863 	sub	w3, w3, #0x2
  f4:	8b070021 	add	x1, x1, x7
  f8:	6b08007f 	cmp	w3, w8
  fc:	91002042 	add	x2, x2, #0x8
 100:	54fffb81 	b.ne	70 <square_gemm+0x70>  // b.any
 104:	510008a5 	sub	w5, w5, #0x2
 108:	8b07014a 	add	x10, x10, x7
 10c:	6b0800bf 	cmp	w5, w8
 110:	8b070000 	add	x0, x0, x7
 114:	54fffa61 	b.ne	60 <square_gemm+0x60>  // b.any
 118:	910009ce 	add	x14, x14, #0x2
 11c:	51000929 	sub	w9, w9, #0x2
 120:	6b0e017f 	cmp	w11, w14
 124:	9100218c 	add	x12, x12, #0x8
 128:	910021ad 	add	x13, x13, #0x8
 12c:	54fff8ec 	b.gt	48 <square_gemm+0x48>
 130:	d65f03c0 	ret
 134:	bc647822 	ldr	s2, [x1, x4, lsl #2]
 138:	710004bf 	cmp	w5, #0x1
 13c:	bd400441 	ldr	s1, [x2, #4]
 140:	1f010040 	fmadd	s0, s2, s1, s0
 144:	bd000000 	str	s0, [x0]
 148:	54fffb4d 	b.le	b0 <square_gemm+0xb0>
 14c:	bc647841 	ldr	s1, [x2, x4, lsl #2]
 150:	bd400023 	ldr	s3, [x1]
 154:	bc647800 	ldr	s0, [x0, x4, lsl #2]
 158:	bc667842 	ldr	s2, [x2, x6, lsl #2]
 15c:	1f010060 	fmadd	s0, s3, s1, s0
 160:	bc647821 	ldr	s1, [x1, x4, lsl #2]
 164:	1f010040 	fmadd	s0, s2, s1, s0
 168:	17ffffd1 	b	ac <square_gemm+0xac>
 16c:	bc667821 	ldr	s1, [x1, x6, lsl #2]
 170:	710004bf 	cmp	w5, #0x1
 174:	bd400442 	ldr	s2, [x2, #4]
 178:	1f010040 	fmadd	s0, s2, s1, s0
 17c:	bd000400 	str	s0, [x0, #4]
 180:	54fffb8d 	b.le	f0 <square_gemm+0xf0>
 184:	bc647843 	ldr	s3, [x2, x4, lsl #2]
 188:	bd400421 	ldr	s1, [x1, #4]
 18c:	bc667800 	ldr	s0, [x0, x6, lsl #2]
 190:	bc667822 	ldr	s2, [x1, x6, lsl #2]
 194:	1f010060 	fmadd	s0, s3, s1, s0
 198:	bc667841 	ldr	s1, [x2, x6, lsl #2]
 19c:	1f010040 	fmadd	s0, s2, s1, s0
 1a0:	17ffffd3 	b	ec <square_gemm+0xec>
