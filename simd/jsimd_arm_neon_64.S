 /*
  * ARMv8 NEON optimizations for libjpeg-turbo
  * This file is a copy of the armv7 neon version but ported to armv8.
  *
  * Copyright (C) 2009-2011 Nokia Corporation and/or it's subsidiary(-ies).
  * All rights reserved.
  * Author Siarhei Siamashka <siarhei.siamashka@nokia.com>
  * Copyright (C) 2013, Linaro Limited
  * Author: Ragesh Radhakrishnan  <ragesh.r@linaro.org>
  *
  * This software is provided 'as-is', without any express or implied
  * warranty.  In no event will the authors be held liable for any damages
  * arising from the use of this software.
  *
  * Permission is granted to anyone to use this software for any purpose,
  * including commercial applications, and to alter it and redistribute it
  * freely, subject to the following restrictions:
  *
  * 1. The origin of this software must not be misrepresented; you must not
  *    claim that you wrote the original software. If you use this software
  *    in a product, an acknowledgment in the product documentation would be
  *    appreciated but is not required.
  * 2. Altered source versions must be plainly marked as such, and must not be
  *    misrepresented as being the original software.
  * 3. This notice may not be removed or altered from any source distribution.
  */

#if defined(__linux__) && defined(__ELF__)
.section .note.GNU-stack,"",%progbits /* mark stack as non-executable */
#endif

.text

.arch armv8-a+fp+simd
#define RESPECT_STRICT_ALIGNMENT 1

#define RTSM_SQSHRN_SIM_ISSUE
/*****************************************************************************/

/* Supplementary macro for setting function attributes */
.macro asm_function fname
#ifdef __APPLE__
    .func _\fname
    .globl _\fname
_\fname:
#else
    .func \fname
    .global \fname
#ifdef __ELF__
    .hidden \fname
    .type \fname, %function
#endif
\fname:
#endif
.endm

/* Transpose elements of single 128 bit registers */
.macro transpose_single x0,x1,xi,xilen,literal
    ins  \xi\xilen[0],	\x0\xilen[0]
    ins  \x1\xilen[0],  \x0\xilen[1]
    trn1 \x0\literal,	\x0\literal, \x1\literal
    trn2 \x1\literal,	\xi\literal, \x1\literal
.endm


/* Transpose elements of 2 differnet registers */
.macro transpose x0,x1,xi,xilen,literal
    mov  \xi\xilen,	\x0\xilen
    trn1 \x0\literal,	\x0\literal, \x1\literal
    trn2 \x1\literal,	\xi\literal, \x1\literal
.endm
/* Transpose a block of 4x4 coefficients in four 64-bit registers */

.macro transpose_4x4_32 x0,x0len x1,x1len x2,x2len x3,x3len,xi,xilen
    mov  \xi\xilen, \x0\xilen
    trn1 \x0\x0len, \x0\x0len, \x2\x2len
    trn2 \x2\x2len, \xi\x0len, \x2\x2len
    mov  \xi\xilen, \x1\xilen
    trn1 \x1\x1len, \x1\x1len, \x3\x3len
    trn2 \x3\x3len, \xi\x1len, \x3\x3len
.endm

.macro transpose_4x4_16 x0,x0len x1,x1len, x2,x2len, x3,x3len,xi,xilen
    mov  \xi\xilen, \x0\xilen
    trn1 \x0\x0len, \x0\x0len, \x1\x1len
    trn2 \x1\x2len, \xi\x0len, \x1\x2len
    mov  \xi\xilen, \x2\xilen
    trn1 \x2\x2len, \x2\x2len, \x3\x3len
    trn2 \x3\x2len, \xi\x1len, \x3\x3len 
.endm

.macro transpose_4x4 x0, x1, x2, x3,x5
    transpose_4x4_16 \x0,.4h, \x1,.4h, \x2,.4h,\x3,.4h,\x5,.16b
    transpose_4x4_32 \x0,.2s, \x1,.2s, \x2,.2s,\x3,.2s,\x5,.16b     
.endm




#define CENTERJSAMPLE 128

/*****************************************************************************/

/*
 * Perform dequantization and inverse DCT on one block of coefficients.
 *
 * GLOBAL(void)
 * jsimd_idct_islow_neon (void * dct_table, JCOEFPTR coef_block,
 *                        JSAMPARRAY output_buf, JDIMENSION output_col)
 */

#define FIX_0_298631336  (2446)
#define FIX_0_390180644  (3196)
#define FIX_0_541196100  (4433)
#define FIX_0_765366865  (6270)
#define FIX_0_899976223  (7373)
#define FIX_1_175875602  (9633)
#define FIX_1_501321110  (12299)
#define FIX_1_847759065  (15137)
#define FIX_1_961570560  (16069)
#define FIX_2_053119869  (16819)
#define FIX_2_562915447  (20995)
#define FIX_3_072711026  (25172)

#define FIX_1_175875602_MINUS_1_961570560 (FIX_1_175875602 - FIX_1_961570560)
#define FIX_1_175875602_MINUS_0_390180644 (FIX_1_175875602 - FIX_0_390180644)
#define FIX_0_541196100_MINUS_1_847759065 (FIX_0_541196100 - FIX_1_847759065)
#define FIX_3_072711026_MINUS_2_562915447 (FIX_3_072711026 - FIX_2_562915447)
#define FIX_0_298631336_MINUS_0_899976223 (FIX_0_298631336 - FIX_0_899976223)
#define FIX_1_501321110_MINUS_0_899976223 (FIX_1_501321110 - FIX_0_899976223)
#define FIX_2_053119869_MINUS_2_562915447 (FIX_2_053119869 - FIX_2_562915447)
#define FIX_0_541196100_PLUS_0_765366865  (FIX_0_541196100 + FIX_0_765366865)

/*
 * Reference SIMD-friendly 1-D ISLOW iDCT C implementation.
 * Uses some ideas from the comments in 'simd/jiss2int-64.asm'
 */
#define REF_1D_IDCT(xrow0, xrow1, xrow2, xrow3, xrow4, xrow5, xrow6, xrow7)   \
{                                                                             \
    DCTELEM row0, row1, row2, row3, row4, row5, row6, row7;                   \
    INT32   q1, q2, q3, q4, q5, q6, q7;                                       \
    INT32   tmp11_plus_tmp2, tmp11_minus_tmp2;                                \
                                                                              \
    /* 1-D iDCT input data */                                                 \
    row0 = xrow0;                                                             \
    row1 = xrow1;                                                             \
    row2 = xrow2;                                                             \
    row3 = xrow3;                                                             \
    row4 = xrow4;                                                             \
    row5 = xrow5;                                                             \
    row6 = xrow6;                                                             \
    row7 = xrow7;                                                             \
                                                                              \
    q5 = row7 + row3;                                                         \
    q4 = row5 + row1;                                                         \
    q6 = MULTIPLY(q5, FIX_1_175875602_MINUS_1_961570560) +                    \
         MULTIPLY(q4, FIX_1_175875602);                                       \
    q7 = MULTIPLY(q5, FIX_1_175875602) +                                      \
         MULTIPLY(q4, FIX_1_175875602_MINUS_0_390180644);                     \
    q2 = MULTIPLY(row2, FIX_0_541196100) +                                    \
         MULTIPLY(row6, FIX_0_541196100_MINUS_1_847759065);                   \
    q4 = q6;                                                                  \
    q3 = ((INT32) row0 - (INT32) row4) << 13;                                 \
    q6 += MULTIPLY(row5, -FIX_2_562915447) +                                  \
          MULTIPLY(row3, FIX_3_072711026_MINUS_2_562915447);                  \
    /* now we can use q1 (reloadable constants have been used up) */          \
    q1 = q3 + q2;                                                             \
    q4 += MULTIPLY(row7, FIX_0_298631336_MINUS_0_899976223) +                 \
          MULTIPLY(row1, -FIX_0_899976223);                                   \
    q5 = q7;                                                                  \
    q1 = q1 + q6;                                                             \
    q7 += MULTIPLY(row7, -FIX_0_899976223) +                                  \
          MULTIPLY(row1, FIX_1_501321110_MINUS_0_899976223);                  \
                                                                              \
    /* (tmp11 + tmp2) has been calculated (out_row1 before descale) */        \
    tmp11_plus_tmp2 = q1;                                                     \
    row1 = 0;                                                                 \
                                                                              \
    q1 = q1 - q6;                                                             \
    q5 += MULTIPLY(row5, FIX_2_053119869_MINUS_2_562915447) +                 \
          MULTIPLY(row3, -FIX_2_562915447);                                   \
    q1 = q1 - q6;                                                             \
    q6 = MULTIPLY(row2, FIX_0_541196100_PLUS_0_765366865) +                   \
         MULTIPLY(row6, FIX_0_541196100);                                     \
    q3 = q3 - q2;                                                             \
                                                                              \
    /* (tmp11 - tmp2) has been calculated (out_row6 before descale) */        \
    tmp11_minus_tmp2 = q1;                                                    \
                                                                              \
    q1 = ((INT32) row0 + (INT32) row4) << 13;                                 \
    q2 = q1 + q6;                                                             \
    q1 = q1 - q6;                                                             \
                                                                              \
    /* pick up the results */                                                 \
    tmp0  = q4;                                                               \
    tmp1  = q5;                                                               \
    tmp2  = (tmp11_plus_tmp2 - tmp11_minus_tmp2) / 2;                         \
    tmp3  = q7;                                                               \
    tmp10 = q2;                                                               \
    tmp11 = (tmp11_plus_tmp2 + tmp11_minus_tmp2) / 2;                         \
    tmp12 = q3;                                                               \
    tmp13 = q1;                                                               \
}

#define XFIX_0_899976223                    v0.4h[0]
#define XFIX_0_541196100                    v0.4h[1]
#define XFIX_2_562915447                    v0.4h[2]
#define XFIX_0_298631336_MINUS_0_899976223  v0.4h[3]
#define XFIX_1_501321110_MINUS_0_899976223  v1.4h[0]
#define XFIX_2_053119869_MINUS_2_562915447  v1.4h[1]
#define XFIX_0_541196100_PLUS_0_765366865   v1.4h[2]
#define XFIX_1_175875602                    v1.4h[3]
#define XFIX_1_175875602_MINUS_0_390180644  v2.4h[0]
#define XFIX_0_541196100_MINUS_1_847759065  v2.4h[1]
#define XFIX_3_072711026_MINUS_2_562915447  v2.4h[2]
#define XFIX_1_175875602_MINUS_1_961570560  v2.4h[3]

.balign 16
jsimd_idct_islow_neon_consts:
    .short FIX_0_899976223                    /* d0[0] */
    .short FIX_0_541196100                    /* d0[1] */
    .short FIX_2_562915447                    /* d0[2] */
    .short FIX_0_298631336_MINUS_0_899976223  /* d0[3] */
    .short FIX_1_501321110_MINUS_0_899976223  /* d1[0] */
    .short FIX_2_053119869_MINUS_2_562915447  /* d1[1] */
    .short FIX_0_541196100_PLUS_0_765366865   /* d1[2] */
    .short FIX_1_175875602                    /* d1[3] */
    /* reloadable constants */
    .short FIX_1_175875602_MINUS_0_390180644  /* d2[0] */
    .short FIX_0_541196100_MINUS_1_847759065  /* d2[1] */
    .short FIX_3_072711026_MINUS_2_562915447  /* d2[2] */
    .short FIX_1_175875602_MINUS_1_961570560  /* d2[3] */

/******************************************************************************
*
* jsimd_idct_islow_neon
*
*****************************************************************************/

asm_function jsimd_idct_islow_neon
    DCT_TABLE       .req x0
    COEF_BLOCK      .req x1
    OUTPUT_BUF      .req x2
    OUTPUT_COL      .req x3
    TMP1            .req x0
    TMP2            .req x1
    TMP3            .req x2
    TMP4            .req x15

    ROW0L           .req v16
    ROW0R           .req v17
    ROW1L           .req v18
    ROW1R           .req v19
    ROW2L           .req v20
    ROW2R           .req v21
    ROW3L           .req v22
    ROW3R           .req v23
    ROW4L           .req v24
    ROW4R           .req v25
    ROW5L           .req v26
    ROW5R           .req v27
    ROW6L           .req v28
    ROW6R           .req v29
    ROW7L           .req v30
    ROW7R           .req v31

    adr     x15, jsimd_idct_islow_neon_consts
    ld1    {v16.4h, v17.4h, v18.4h, v19.4h},   	[COEF_BLOCK],	32
    ld1    {v0.4h, v1.4h, v2.4h, v3.4h},   	[DCT_TABLE],	32
    ld1    {v20.4h, v21.4h, v22.4h, v23.4h}, 	[COEF_BLOCK],	32 
    mul    v16.4h,v16.4h,v0.4h
    mul    v17.4h,v17.4h,v1.4h
    ins    v16.2d[1],v17.2d[0]                  /* 128 bit q8 */
    ld1    {v4.4h,v5.4h,v6.4h,v7.4h},    	[DCT_TABLE],	32
    mul    v18.4h,v18.4h,v2.4h
    mul    v19.4h,v19.4h,v3.4h
    ins    v18.2d[1],v19.2d[0]                  /* 128 bit q9 */
    ld1    {v24.4h, v25.4h, v26.4h, v27.4h},  	[COEF_BLOCK],	32
    mul    v20.4h,v20.4h,v4.4h
    mul    v21.4h,v21.4h,v5.4h
    ins    v20.2d[1],v21.2d[0]                  /* 128 bit q10 */
    ld1    {v0.4h, v1.4h, v2.4h, v3.4h},   	[DCT_TABLE],  	32
    mul    v22.4h,v22.4h,v6.4h
    mul    v23.4h,v23.4h,v7.4h
    ins    v22.2d[1],v23.2d[0]                  /* 128 bit q11 */
    ld1    {v28.4h, v29.4h, v30.4h, v31.4h}, 	[COEF_BLOCK],	32
    mul    v24.4h, v24.4h, v0.4h
    mul    v25.4h, v25.4h, v1.4h
    ins    v24.2d[1],v25.2d[0]                  /* 128 bit q12 */ 
    ld1    {v4.4h, v5.4h, v6.4h, v7.4h}, 	[DCT_TABLE],	32
    mul    v28.4h, v28.4h, v4.4h
    mul    v29.4h, v29.4h, v5.4h
    ins    v28.2d[1],v29.2d[0]                  /* 128 bit q14 */
    mul    v26.4h, v26.4h, v2.4h
    mul    v27.4h, v27.4h, v3.4h
    ins    v26.2d[1],v27.2d[0]                  /* 128 bit q13 */
    ld1    {v0.4h,v1.4h, v2.4h,v3.4h}, 		[x15] /* load constants */
    add    x15, x15, #16
    mul    v30.4h, v30.4h, v6.4h
    mul    v31.4h, v31.4h, v7.4h
    ins    v30.2d[1],v31.2d[0]                  /* 128 bit q15 */
    sub    sp,					sp,		#32
    st1    {v8.4h-v11.4h}, 			[sp]/* save NEON registers */
    sub    sp,					sp,		#32
    st1    {v12.4h-v15.4h}, 			[sp]
    /* 1-D IDCT, pass 1, left 4x8 half */
    add	   v4.4h,				ROW7L.4h, 	ROW3L.4h
    add	   v5.4h,				ROW5L.4h, 	ROW1L.4h
    smull  v12.4s,				v4.4h,		XFIX_1_175875602_MINUS_1_961570560
    smlal  v12.4s,				v5.4h,		XFIX_1_175875602
    smull  v14.4s,				v4.4h,		XFIX_1_175875602
    /* check for the zero coeffecients in the right 4*8 half */
    /*push       {x4, x5}*/ /*--------> need to be fixed */
    stp    x4,x5,[sp,-16]!
    mov    x5, #0
    smlal  v14.4s,    				v5.4h,    	XFIX_1_175875602_MINUS_0_390180644
    ssubl  v6.4s,    				ROW0L.4h, 	ROW4L.4h
    ldr    x4,    				[COEF_BLOCK, #(-96 + 2 * (4 + 1 * 8))]
    smull  v4.4s, 				ROW2L.4h, 	XFIX_0_541196100
    smlal  v4.4s,  				ROW6L.4h, 	XFIX_0_541196100_MINUS_1_847759065
    orr    x0,    				x4,    		x5
    mov    v8.16b, 				v12.16b
    smlsl  v12.4s,				ROW5L.4h, 	XFIX_2_562915447
    ldr    x4,    				[COEF_BLOCK, 	#(-96 + 2 * (4 + 2 * 8))]
    smlal  v12.4s,    				ROW3L.4h, 	XFIX_3_072711026_MINUS_2_562915447
    shl    v6.4s,    				v6.4s,    	#13
    orr    x0,    				x0,   		x4
    smlsl  v8.4s,    				ROW1L.4h, 	XFIX_0_899976223
    orr    x0,    				x0 ,   		x5
    add    v2.4s, 				v6.4s, 		v4.4s
    ldr    x4,    				[COEF_BLOCK, #(-96 + 2 * (4 + 3 * 8))]
    mov    v10.16b,    				v14.16b
    add    v2.4s, 				v2.4s, 		v12.4s
    orr    x0,    				x0,    		x4
    smlsl  v14.4s,    				ROW7L.4h, 	XFIX_0_899976223
    orr    x0,    				x0,    		x5
    smlal  v14.4s, 				ROW1L.4h, 	XFIX_1_501321110_MINUS_0_899976223
    rshrn  ROW1L.4h, 				v2.4s,    	#11
    ldr    x4,    				[COEF_BLOCK, #(-96 + 2 * (4 + 4 * 8))]
    sub    v2.4s,    				v2.4s,    	v12.4s
    smlal  v10.4s,    				ROW5L.4h, 	XFIX_2_053119869_MINUS_2_562915447
    orr    x0,    				x0,    		x4
    smlsl  v10.4s,				ROW3L.4h, 	XFIX_2_562915447
    orr    x0,    				x0,    		x5
    sub    v2.4s, 				v2.4s, 		v12.4s
    smull  v12.4s,				ROW2L.4h, 	XFIX_0_541196100_PLUS_0_765366865
    ldr    x4,    				[COEF_BLOCK, #(-96 + 2 * (4 + 5 * 8))]
    smlal  v12.4s,    				ROW6L.4h, 	XFIX_0_541196100
    sub    v6.4s,    				v6.4s,    	v4.4s
    orr    x0,    				x0,    		x4
    rshrn  ROW6L.4h, 				v2.4s,    	#11
    orr    x0,    				x0,    		x5
    add    v2.4s, 				v6.4s,  	v10.4s
    ldr    x4,    				[COEF_BLOCK, #(-96 + 2 * (4 + 6 * 8))]
    sub    v6.4s, 				v6.4s, 		v10.4s
    saddl  v10.4s,    				ROW0L.4h, 	ROW4L.4h
    orr    x0,    				x0,    		x4
    rshrn  ROW2L.4h, 				v2.4s, 		#11
    orr    x0,    				x0,    		x5
    rshrn  ROW5L.4h, 				v6.4s, 		#11
    ldr    x4,    				[COEF_BLOCK, 	#(-96 + 2 * (4 + 7 * 8))]
    shl    v10.4s, 				v10.4s, 	#13
    smlal  v8.4s,    				ROW7L.4h, 	XFIX_0_298631336_MINUS_0_899976223
    orr    x0,    				x0,    		x4
    add    v4.4s, 				v10.4s,		v12.4s
    orr    x0,    				x0,    		x5
    sub    v2.4s, 				v10.4s,		v12.4s
    add    v12.4s,				v4.4s, 		v14.4s
    ldr    x4,    				[COEF_BLOCK, #(-96 + 2 * (4 + 0 * 8))]
    sub    v4.4s, 				v4.4s,  	v14.4s
    add    v10.4s,				v2.4s,  	v8.4s
    orr    x0,    				x4,    		x5
    sub    v6.4s, 				v2.4s,  	v8.4s
    /*pop         {x4, x5} */ 
    ldp    x4,					x5, 		[sp],	16 
    rshrn  ROW7L.4h, 				v4.4s,    	#11
    rshrn  ROW3L.4h, 				v10.4s,   	#11
    rshrn  ROW0L.4h, 				v12.4s,   	#11
    rshrn  ROW4L.4h, 				v6.4s,    	#11
    cmp    x0, #0 /*orrs instruction removed*/
    beq    3f /* Go to do some special handling for the sparse right 4x8 half */


	/* 1-D IDCT, pass 1, right 4x8 half */
    ld1        {v2.4h},				[x15]    /* reload constants */
    add        v10.4h,  			ROW7R.4h, 	ROW3R.4h
    add        v8.4h,   			ROW5R.4h, 	ROW1R.4h
    /* Transpose ROW6L <-> ROW7L   (v3 avliable free register)*/
    transpose  ROW6L,ROW7L,v3,.16b,.4h
    smull      v12.4s,  			v10.4h,		XFIX_1_175875602_MINUS_1_961570560
    smlal      v12.4s,  			v8.4h,  	XFIX_1_175875602
    /* Transpose ROW2L <-> ROW3L   (v3 avliable free register)*/
    transpose  ROW2L,ROW3L,v3,.16b,.4h
    smull      v14.4s,				v10.4h, 	XFIX_1_175875602
    smlal      v14.4s,  			v8.4h,  	XFIX_1_175875602_MINUS_0_390180644
    /* Transpose ROW0L <-> ROW1L   (v3 avliable free register)*/
    transpose  ROW0L,ROW1L,v3,.16b,.4h
    ssubl      v6.4s,   			ROW0R.4h,	ROW4R.4h
    smull      v4.4s,   			ROW2R.4h,	XFIX_0_541196100
    smlal      v4.4s,   			ROW6R.4h,	XFIX_0_541196100_MINUS_1_847759065
    /* Transpose ROW4L <-> ROW5L   (v3 avliable free register)*/
    transpose  ROW4L,ROW5L,v3,.16b,.4h
    mov        v8.16b,   			v12.16b
    smlsl      v12.4s, 				ROW5R.4h,	XFIX_2_562915447
    smlal      v12.4s,  			ROW3R.4h,	XFIX_3_072711026_MINUS_2_562915447
    /* Transpose ROW1L <-> ROW3L   (v3 avliable free register)*/
    transpose  ROW1L,ROW3L,v3,.16b,.2s
    shl        v6.4s,   			v6.4s,    	#13
    smlsl      v8.4s,   			ROW1R.4h,	XFIX_0_899976223
    /* Trannnnnose ROW4L <-> ROW6L   (v3 avliable free register)*/
    transpose  ROW4L,ROW6L,v3,.16b,.2s
    add        v2.4s,   			v6.4s,  	v4.4s
    mov        v10.16b,  			v14.16b
    add        v2.4s,   			v2.4s,  	v12.4s
    /* Transpose ROW0L <-> ROW2L   (v3 avliable free register)*/
    transpose  ROW0L,ROW2L,v3,.16b,.2s
    smlsl      v14.4s,  			ROW7R.4h,	XFIX_0_899976223
    smlal      v14.4s,  			ROW1R.4h,	XFIX_1_501321110_MINUS_0_899976223
    rshrn      ROW1R.4h,			v2.4s,  	#11
    /* Transpose ROW5L <-> ROW7L   (v3 avliable free register)*/
    transpose  ROW5L,ROW7L,v3,.16b,.2s
    sub        v2.4s,   			v2.4s,  	v12.4s
    smlal      v10.4s,  			ROW5R.4h,	XFIX_2_053119869_MINUS_2_562915447
    smlsl      v10.4s,  			ROW3R.4h,	XFIX_2_562915447
    sub        v2.4s,   			v2.4s,  	v12.4s
    smull      v12.4s,  			ROW2R.4h,	XFIX_0_541196100_PLUS_0_765366865
    smlal      v12.4s,  			ROW6R.4h,	XFIX_0_541196100
    sub        v6.4s,   			v6.4s,  	v4.4s
    rshrn      ROW6R.4h,			v2.4s,  	#11
    add        v2.4s,   			v6.4s,  	v10.4s
    sub        v6.4s,   			v6.4s,  	v10.4s
    saddl      v10.4s,  			ROW0R.4h,	ROW4R.4h
    rshrn      ROW2R.4h,			v2.4s,  	#11
    rshrn      ROW5R.4h,			v6.4s,  	#11
    shl        v10.4s,  			v10.4s, 	#13
    smlal      v8.4s,   			ROW7R.4h,	XFIX_0_298631336_MINUS_0_899976223
    add        v4.4s,   			v10.4s, 	v12.4s
    sub        v2.4s,   			v10.4s, 	v12.4s
    add        v12.4s,  			v4.4s,  	v14.4s
    sub        v4.4s,   			v4.4s,  	v14.4s
    add        v10.4s,  			v2.4s,  	v8.4s
    sub        v12.4s,  			v2.4s,  	v8.4s
    rshrn      ROW7R.4h,			v4.4s,  	#11
    rshrn      ROW3R.4h,			v10.4s, 	#11
    rshrn      ROW0R.4h,			v12.4s, 	#11
    rshrn      ROW4R.4h,			v6.4s,  	#11
/* Transpose right 4x8 half */
    transpose  ROW6R, ROW7R,v3,.16b,.4h
    transpose  ROW2R, ROW3R,v3,.16b,.4h
    transpose  ROW0R, ROW1R,v3,.16b,.4h
    transpose  ROW4R, ROW5R,v3,.16b,.4h
    transpose  ROW1R, ROW3R,v3,.16b,.2s
    transpose  ROW4R, ROW6R,v3,.16b,.2s
    transpose  ROW0R, ROW2R,v3,.16b,.2s
    transpose  ROW5R, ROW7R,v3,.16b,.2s

	
1:  /* 1-D IDCT, pass 2 (normal variant), left 4x8 half */
    ld1         {v2.4h},  			[x15]    /* reload constants */
    smull	v12.4S,   			ROW1R.4h, 	XFIX_1_175875602 /* ROW5L.4h <-> ROW1R.4h */
    smlal	v12.4s,    			ROW1L.4h, 	XFIX_1_175875602
    smlal	v12.4s,    			ROW3R.4h, 	XFIX_1_175875602_MINUS_1_961570560 /* ROW7L.4h <-> ROW3R.4h */
    smlal	v12.4s,    			ROW3L.4h, 	XFIX_1_175875602_MINUS_1_961570560
    smull	v14.4s,    			ROW3R.4h, 	XFIX_1_175875602 /* ROW7L.4h <-> ROW3R.4h */
    smlal	v14.4s,    			ROW3L.4h, 	XFIX_1_175875602
    smlal	v14.4s,    			ROW1R.4h, 	XFIX_1_175875602_MINUS_0_390180644 /* ROW5L.4h <-> ROW1R.4h */
    smlal	v14.4s,    			ROW1L.4h, 	XFIX_1_175875602_MINUS_0_390180644
    ssubl	v6.4s,     			ROW0L.4h, 	ROW0R.4h /* ROW4L.4h <-> ROW0R.4h */
    smull	v4.4s,     			ROW2L.4h, 	XFIX_0_541196100
    smlal	v4.4s,     			ROW2R.4h, 	XFIX_0_541196100_MINUS_1_847759065 /* ROW6L.4h <-> ROW2R.4h */
    mov         v8.16b,     			v12.16b
    smlsl	v12.4s,    			ROW1R.4h, 	XFIX_2_562915447 /* ROW5L.4h <-> ROW1R.4h */
    smlal	v12.4s,    			ROW3L.4h, 	XFIX_3_072711026_MINUS_2_562915447
    shl		v6.4s,     			v6.4s,    	#13
    smlsl	v8.4s,     			ROW1L.4h, 	XFIX_0_899976223
    add		v2.4s,     			v6.4s,    	v4.4s
    mov         v10.16b,    			v14.16b
    add		v2.4s,     			v2.4s,    	v12.4s
    smlsl	v14.4s,    			ROW3R.4h, 	XFIX_0_899976223 /* ROW7L.4h <-> ROW3R.4h */
    smlal	v14.4s,    			ROW1L.4h, 	XFIX_1_501321110_MINUS_0_899976223
    shrn	ROW1L.4h,	    		v2.4s,    	#16
    sub		v2.4s,    			v2.4s,    	v12.4s
    smlal	v10.4s,   			ROW1R.4h, 	XFIX_2_053119869_MINUS_2_562915447 /* ROW5L.4h <-> ROW1R.4h */
    smlsl	v10.4s,   			ROW3L.4h, 	XFIX_2_562915447
    sub		v2.4s,    			v2.4s,    	v12.4s
    smull	v12.4s,   			ROW2L.4h, 	XFIX_0_541196100_PLUS_0_765366865
    smlal	v12.4s,   			ROW2R.4h, 	XFIX_0_541196100 /* ROW6L.4h <-> ROW2R.4h */
    sub		v6.4s,    			v6.4s,    	v4.4s
    shrn	ROW2R.4h,    			v2.4s,    	#16 /* ROW6L.4h <-> ROW2R.4h */
    add		v2.4s,    			v6.4s,   	v10.4s
    sub		v6.4s,    			v6.4s,    	v10.4s
    saddl	v10.4s,   			ROW0L.4h, 	ROW0R.4h /* ROW4L.4h <-> ROW0R.4h */
    shrn	ROW2L.4h,    			v2.4s,    	#16
    shrn	ROW1R.4h,    			v6.4s,    	#16 /* ROW5L.4h <-> ROW1R.4h */
    shl		v10.4s,   			v10.4s,   	#13
    smlal	v8.4s,    			ROW3R.4h, 	XFIX_0_298631336_MINUS_0_899976223 /* ROW7L.4h <-> ROW3R.4h */
    add		v4.4s,    			v10.4s,   	v12.4s
    sub		v2.4s,    			v10.4s,   	v12.4s
    add	    	v12.4s,   			v4.4s,    	v14.4s
    sub	        v4.4s,    			v4.4s,    	v14.4s
    add	        v10.4s,   			v2.4s,    	v8.4s
    sub	        v6.4s,    			v2.4s,    	v8.4s
    shrn	ROW3R.4h,    			v4.4s,    	#16 /* ROW7L.4h <-> ROW3R.4h */
    shrn	ROW3L.4h,    			v10.4s,   	#16
    shrn	ROW0L.4h,    			v12.4s,   	#16
    shrn	ROW0R.4h,    			v6.4s,    	#16 /* ROW4L.4h <-> ROW0R.4h */
    /* 1-D IDCT,pass 2, right 4x8 half */
    ld1         {v2.4h},  			[x15]    /* reload constants */
    smull       v12.4s,    			ROW5R.4h, 	XFIX_1_175875602
    smlal       v12.4s,    			ROW5L.4h, 	XFIX_1_175875602 /* ROW5L.4h <-> ROW1R.4h */
    smlal       v12.4s,    			ROW7R.4h, 	XFIX_1_175875602_MINUS_1_961570560
    smlal       v12.4s,    			ROW7L.4h, 	XFIX_1_175875602_MINUS_1_961570560 /* ROW7L.4h <-> ROW3R.4h */
    smull       v14.4s,    			ROW7R.4h, 	XFIX_1_175875602
    smlal       v14.4s,    			ROW7L.4h, 	XFIX_1_175875602 /* ROW7L.4h <-> ROW3R.4h */
    smlal       v14.4s,    			ROW5R.4h, 	XFIX_1_175875602_MINUS_0_390180644
    smlal       v14.4s,    			ROW5L.4h, 	XFIX_1_175875602_MINUS_0_390180644 /* ROW5L.4h <-> ROW1R.4h */
    ssubl       v6.4s,     			ROW4L.4h, 	ROW4R.4h /* ROW4L.4h <-> ROW0R.4h */
    smull       v4.4s,     			ROW6L.4h, 	XFIX_0_541196100 /* ROW6L.4h <-> ROW2R.4h */
    smlal       v4.4s,     			ROW6R.4h, 	XFIX_0_541196100_MINUS_1_847759065
    mov         v8.16b,     			v12.16b
    smlsl       v12.4s,    			ROW5R.4h, 	XFIX_2_562915447
    smlal       v12.4s,    			ROW7L.4h, 	XFIX_3_072711026_MINUS_2_562915447 /* ROW7L.4h <-> ROW3R.4h */
    shl         v6.4s,     			v6.4s,    	#13
    smlsl       v8.4s,     			ROW5L.4h, 	XFIX_0_899976223 /* ROW5L.4h <-> ROW1R.4h */
    add         v2.4s,     			v6.4s,    	v4.4s
    mov         v10.16b,    			v14.16b
    add         v2.4s,     			v2.4s, 		v12.4s
    smlsl       v14.4s,    			ROW7R.4h, 	XFIX_0_899976223
    smlal       v14.4s,    			ROW5L.4h, 	XFIX_1_501321110_MINUS_0_899976223 /* ROW5L.4h <-> ROW1R.4h */
    shrn        ROW5L.4h,    	 		v2.4s, #16 /* ROW5L.4h <-> ROW1R.4h */
    sub         v2.4s,     			v2.4s, v12.4s
    smlal       v10.4s,    			ROW5R.4h, 	XFIX_2_053119869_MINUS_2_562915447
    smlsl       v10.4s,    			ROW7L.4h, 	XFIX_2_562915447 /* ROW7L.4h <-> ROW3R.4h */
    sub         v2.4s,     			v2.4s, v12.4s
    smull       v12.4s,    			ROW6L.4h, 	XFIX_0_541196100_PLUS_0_765366865 /* ROW6L.4h <-> ROW2R.4h */
    smlal       v12.4s,    			ROW6R.4h, 	XFIX_0_541196100
    sub         v6.4s,    			v6.4s,  	v4.4s
    shrn        ROW6R.4h,    			v2.4s,  	#16
    add         v2.4s,    			v6.4s,  	v10.4s
    sub         v6.4s,    			v6.4s,  	v10.4s
    saddl       v10.4s,   			ROW4L.4h,  	ROW4R.4h /* ROW4L.4h <-> ROW0R.4h */
    shrn        ROW6L.4h,    			v2.4s,  	#16 /* ROW6L.4h <-> ROW2R.4h */
    shrn        ROW5R.4h,    			v6.4s,  	#16
    shl         v10.4s,   			v10.4s, 	#13
    smlal       v8.4s,    			ROW7R.4h,  	XFIX_0_298631336_MINUS_0_899976223
    add         v4.4s,    			v10.4s, 	v12.4s
    sub         v2.4s,    			v10.4s, 	v12.4s
    add         v12.4s,   			v4.4s,  	v14.4s
    sub         v4.4s,    			v4.4s,  	v14.4s
    add         v10.4s,   			v2.4s,  	v8.4s
    sub         v6.4s,    			v2.4s,  	v8.4s
    shrn        ROW7R.4h,    			v4.4s,  	#16
    shrn        ROW7L.4h,    			v10.4s, 	#16 /* ROW7L.4h <-> ROW3R.4h */
    shrn        ROW4L.4h,    			v12.4s, 	#16 /* ROW4L.4h <-> ROW0R.4h */
    shrn        ROW4R.4h,    			v6.4s,  	#16

2:  /* Descale to 8-bit and range limit */
    ins         v16.2d[1],			v17.2d[0]
    ins         v18.2d[1],			v19.2d[0]
    ins         v20.2d[1],			v21.2d[0]
    ins         v22.2d[1],			v23.2d[0]
#ifdef RTSM_SQSHRN_SIM_ISSUE
    sqrshrn     v16.8b,   			v16.8h,   	#2
    sqrshrn2    v16.16b,   			v18.8h,   	#2
    sqrshrn     v18.8b,   			v20.8h,   	#2
    sqrshrn2    v18.16b,   			v22.8h,   	#2
#else 
    sqrshrn     v16.4h,   			v16.4s,   	#2
    sqrshrn2    v16.8h,   			v18.4s,   	#2
    sqrshrn     v18.4h,   			v20.4s,   	#2
    sqrshrn2    v18.8h,   			v22.4s,   	#2
#endif    
    /*vpop        {v8.4h-d15.4h} *//* restore NEON registers */
    
    ld1        {v12.4h-v15.4h},			[sp],		32
    ld1        {v8.4h-v11.4h},			[sp],		32
    ins        v24.2d[1],			v25.2d[0]
#ifdef RTSM_SQSHRN_SIM_ISSUE
    
    sqrshrn    v20.8b,   			v24.8h,   	#2
#else
    
    sqrshrn    v20.4h,   			v24.4s,   	#2
#endif
      /* Transpose the final 8-bit samples and do signed->unsigned conversion */
    /*trn1       v16.8h, 				v16.8h,  	v18.8h*/
    transpose  v16,v18,v3,.16b,.8h 
    ins        v26.2d[1],			v27.2d[0]
    ins        v28.2d[1],			v29.2d[0]
    ins        v30.2d[1],			v31.2d[0]
#ifdef RTSM_SQSHRN_SIM_ISSUE
    sqrshrn2   v20.16b,   			v26.8h,   	#2
    sqrshrn    v22.8b,   			v28.8h,   	#2
#else
    sqrshrn2   v20.8h,   			v26.4s,   	#2
    sqrshrn    v22.4h,   			v28.4s,   	#2
#endif
    movi       v0.16b,    			#(CENTERJSAMPLE)
#ifdef RTSM_SQSHRN_SIM_ISSUE
    sqrshrn2   v22.16b,   			v30.8h,   	#2
#else
    sqrshrn2   v22.8h,   			v30.4s,   	#2
#endif
    transpose_single  v16,v17,v3,.2d,.8b
    transpose_single  v18,v19,v3,.2d,.8b
    add        v16.8b,  			v16.8b,   	v0.8b
    add        v17.8b,  			v17.8b,   	v0.8b
    add        v18.8b,  			v18.8b,   	v0.8b
    add        v19.8b,  			v19.8b,   	v0.8b
    transpose  v20,v22,v3,.16b,.8h
       /* Store results to the output buffer */
    
    ldp        TMP1,				TMP2,		[OUTPUT_BUF],16
    add        TMP1, 				TMP1, 		OUTPUT_COL
    add        TMP2, 				TMP2, 		OUTPUT_COL
    st1        {v16.8b}, 			[TMP1]
    transpose_single   v20,v21,v3,.2d,.8b
    st1        {v17.8b}, 			[TMP2]
    ldp        TMP1, 				TMP2,           [OUTPUT_BUF],16
    add        TMP1, 				TMP1, 		OUTPUT_COL
    add        TMP2, 				TMP2, 		OUTPUT_COL
    st1        {v18.8b}, 			[TMP1]
    add        v20.8b,  			v20.8b,  	v0.8b
    add        v21.8b,				v21.8b,		v0.8b
    st1        {v19.8b}, 			[TMP2]
    ldp        TMP1, 				TMP2,		[OUTPUT_BUF],16  
    ldp        TMP3, 				TMP4,		[OUTPUT_BUF]
    add        TMP1, 				TMP1, 		OUTPUT_COL
    add        TMP2, 				TMP2, 		OUTPUT_COL
    add        TMP3, 				TMP3, 		OUTPUT_COL
    add        TMP4, 				TMP4, 		OUTPUT_COL
    transpose_single v22, v23, v3, .2d,.8b
    st1        {v20.8b}, 			[TMP1]
    add        v22.8b,  			v22.8b, 	v0.8b
    add        v23.8b,  			v23.8b, 	v0.8b
    st1        {v21.8b}, 			[TMP2]
    st1        {v22.8b}, 			[TMP3]
    st1        {v23.8b}, 			[TMP4]
    blr        x30

3:  /* Left 4x8 half is done, right 4x8 half contains mostly zeros */

    /* Transpose left 4x8 half */
    transpose  ROW6L,ROW7L,v3,.16b,.4h
    transpose  ROW2L,ROW3L,v3,.16b,.4h
    transpose  ROW0L,ROW1L,v3,.16b,.4h
    transpose  ROW4L,ROW5L,v3,.16b,.4h
    shl        ROW0R.4h,			ROW0R.4h,  	#2 /* PASS1_BITS */
    transpose  ROW1L,ROW3L,v3,.16b,.2s
    transpose  ROW4L,ROW6L,v3,.16b,.2s
    transpose  ROW0L,ROW2L,v3,.16b,.2s
    transpose  ROW5L,ROW7L,v3,.16b,.2s
    cmp        x0, 				#0
    beq        4f /* Right 4x8 half has all zeros, go to 'sparse' second pass */

    /* Only row 0 is non-zero for the right 4x8 half  */
    dup        ROW1R.4h, 			ROW0R.4h[1]
    dup        ROW2R.4h,			ROW0R.4h[2]
    dup        ROW3R.4h, 			ROW0R.4h[3]
    dup        ROW4R.4h, 			ROW0R.4h[0]
    dup        ROW5R.4h, 			ROW0R.4h[1]
    dup        ROW6R.4h, 			ROW0R.4h[2]
    dup        ROW7R.4h, 			ROW0R.4h[3]
    dup        ROW0R.4h, 			ROW0R.4h[0]
    b               1b /* Go to 'normal' second pass */

4:  /* 1-D IDCT, pass 2 (sparse variant with zero rows 4-7), left 4x8 half */
    ld1        {v2.4h},  			[x15]    /* reload constants */
    smull      v12.4s,   			ROW1L.4h, 	XFIX_1_175875602
    smlal      v12.4s,   			ROW3L.4h, 	XFIX_1_175875602_MINUS_1_961570560
    smull      v14.4s,   			ROW3L.4h, 	XFIX_1_175875602
    smlal      v14.4s,   			ROW1L.4h, 	XFIX_1_175875602_MINUS_0_390180644
    smull      v4.4s,    			ROW2L.4h, 	XFIX_0_541196100
    sshll      v6.4s,    			ROW0L.4h, 	#13
    mov        v8.16b,    			v12.16b
    smlal      v12.4s,   			ROW3L.4h, 	XFIX_3_072711026_MINUS_2_562915447
    smlsl      v8.4s,    			ROW1L.4h, 	XFIX_0_899976223
    add        v2.4s,    			v6.4s,    	v4.4s
    mov        v10.16b,   			v14.16b
    smlal      v14.4s,   			ROW1L.4h, 	XFIX_1_501321110_MINUS_0_899976223
    add        v2.4s,    			v2.4s,    	v12.4s
    add        v12.4s,   			v12.4s,    	v12.4s
    smlsl      v10.4s,   			ROW3L.4h, 	XFIX_2_562915447
    shrn       ROW1L.4h,    			v2.4s,    	#16
    sub        v2.4s,    			v2.4s,    	v12.4s
    smull      v12.4s,   			ROW2L.4h, 	XFIX_0_541196100_PLUS_0_765366865
    sub        v6.4s,    			v6.4s,    	v4.4s
    shrn       ROW2R.4h,    			v2.4s,    	#16 /* ROW6L.4h <-> ROW2R.4h */
    add        v2.4s,    			v6.4s,    	v10.4s
    sub        v6.4s,    			v6.4s,    	v10.4s
    sshll      v10.4s,   			ROW0L.4h, 	#13
    shrn       ROW2L.4h,    			v2.4s,    	#16
    shrn       ROW1R.4h,    			v6.4s,    	#16 /* ROW5L.4h <-> ROW1R.4h */
    add        v4.4s,    			v10.4s,    	v12.4s
    sub        v2.4s,    			v10.4s,    	v12.4s
    add        v12.4s,   			v4.4s,    	v14.4s
    sub        v4.4s,    			v4.4s,    	v14.4s
    add        v10.4s,   			v2.4s,    	v8.4s
    sub        v6.4s,    			v2.4s,    	v8.4s
    shrn       ROW3R.4h,    			v4.4s,    	#16 /* ROW7L.4h <-> ROW3R.4h */
    shrn       ROW3L.4h,    			v10.4s,    	#16
    shrn       ROW0L.4h,    			v12.4s,    	#16
    shrn       ROW0R.4h,    			v6.4s,    	#16 /* ROW4L.4h <-> ROW0R.4h */
    /* 1-D IDCT, pass 2 (sparse variant with zero rows 4-7), right 4x8 half */
    ld1        {v2.4h},  			[x15]    /* reload constants */
    smull      v12.4s,   			ROW5L.4h, 	XFIX_1_175875602
    smlal      v12.4s,   			ROW7L.4h,	 XFIX_1_175875602_MINUS_1_961570560
    smull      v14.4s,   			ROW7L.4h, 	XFIX_1_175875602
    smlal      v14.4s,   			ROW5L.4h, 	XFIX_1_175875602_MINUS_0_390180644
    smull      v4.4s,    			ROW6L.4h, 	XFIX_0_541196100
    sshll      v6.4s,    			ROW4L.4h, 	#13
    mov        v8.16b,    			v12.16b
    smlal      v12.4s,   			ROW7L.4h, 	XFIX_3_072711026_MINUS_2_562915447
    smlsl      v8.4s,    			ROW5L.4h, 	XFIX_0_899976223
    add        v2.4s,    			v6.4s,    	v4.4s
    mov        v10.16b,   			v14.16b
    smlal      v14.4s,   			ROW5L.4h, 	XFIX_1_501321110_MINUS_0_899976223
    add        v2.4s,    			v2.4s,    	v12.4s
    add        v12.4s,   			v12.4s,    	v12.4s
    smlsl      v10.4s,   			ROW7L.4h, 	XFIX_2_562915447
    shrn       ROW5L.4h,    			v2.4s,    	#16 /* ROW5L.4h <-> ROW1R.4h */
    sub        v2.4s,    			v2.4s,    	v12.4s
    smull      v12.4s,   			ROW6L.4h, 	XFIX_0_541196100_PLUS_0_765366865
    sub        v6.4s,    			v6.4s,    	v4.4s
    shrn       ROW6R.4h,    			v2.4s,    	#16
    add        v2.4s,    			v6.4s,    	v10.4s
    sub        v6.4s,    			v6.4s,    	v10.4s
    sshll      v10.4s,   			ROW4L.4h, 	#13
    shrn       ROW6L.4h,    			v2.4s,    	#16 /* ROW6L.4h <-> ROW2R.4h */
    shrn       ROW5R.4h,    			v6.4s,    	#16
    add        v4.4s,    			v10.4s,    	v12.4s
    sub        v2.4s,    			v10.4s,    	v12.4s
    add        v12.4s,   			v4.4s,    	v14.4s
    sub        v4.4s,    			v4.4s,    	v14.4s
    add        v10.4s,   			v2.4s,    	v8.4s
    sub        v6.4s,    			v2.4s,    	v8.4s
    shrn       ROW7R.4h,    			v4.4s,    	#16
    shrn       ROW7L.4h,    			v10.4s,    	#16 /* ROW7L.4h <-> ROW3R.4h */
    shrn       ROW4L.4h,    			v12.4s,    	#16 /* ROW4L.4h <-> ROW0R.4h */
    shrn       ROW4R.4h,    			v6.4s,    	#16
    b               2b /* Go to epilogue */



 
    .unreq          DCT_TABLE
    .unreq          COEF_BLOCK
    .unreq          OUTPUT_BUF
    .unreq          OUTPUT_COL
    .unreq          TMP1
    .unreq          TMP2
    .unreq          TMP3
    .unreq          TMP4

    .unreq          ROW0L
    .unreq          ROW0R
    .unreq          ROW1L
    .unreq          ROW1R
    .unreq          ROW2L
    .unreq          ROW2R
    .unreq          ROW3L
    .unreq          ROW3R
    .unreq          ROW4L
    .unreq          ROW4R
    .unreq          ROW5L
    .unreq          ROW5R
    .unreq          ROW6L
    .unreq          ROW6R
    .unreq          ROW7L
    .unreq          ROW7R
.endfunc
