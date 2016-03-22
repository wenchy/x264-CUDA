
#ifndef __X264_CUDA_COMMON_H__
#define __X264_CUDA_COMMON_H__

typedef signed char int8_t;
typedef unsigned char   uint8_t;
typedef short  int16_t;
typedef unsigned short  uint16_t;
typedef int  int32_t;
typedef unsigned   uint32_t;

typedef uint8_t  pixel;
typedef long int intptr_t;


/* best Motion Vector and Cost */
typedef struct x264_cuda_mvc_t
{
	int16_t mv[2];
	int cost;
} x264_cuda_mvc_t;

/* ME for a MB */
typedef struct x264_cuda_me_t
{
	x264_cuda_mvc_t mvc16x16;
	x264_cuda_mvc_t mvc16x8[2];
	x264_cuda_mvc_t mvc8x16[2];
	x264_cuda_mvc_t mvc8x8[4];
	x264_cuda_mvc_t mvc8x4[8];
	x264_cuda_mvc_t mvc4x8[8];
	x264_cuda_mvc_t mvc4x4[16];
} x264_cuda_me_t;

typedef struct x264_cuda_t
{
	int i_frame;
	int i_me_range;

	int i_mb_x;
	int i_mb_y;

	int i_mb_width;
	int i_mb_height;

	// block width, block height
	int i_pixel;
	int bw;
	int bh;

	int mv_min_x;
	int mv_min_y;
	int mv_max_x;
	int mv_max_y;

	pixel *fenc_buf;
	pixel *fref_buf;
	// with PADV and PADH for me
	pixel *dev_fenc_buf;
	pixel *dev_fref_buf;
	int stride_buf;

	uint16_t *p_cost_mvx;
	uint16_t *p_cost_mvy;

	// each MB have a me(x264_cuda_me_t)
	x264_cuda_me_t *me;
} x264_cuda_t;



#endif  // __X264_CUDA_COMMON_H__

