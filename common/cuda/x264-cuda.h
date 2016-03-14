
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

typedef struct x264_cuda_t
{
	int i_me_range;

	int i_mb_x;
	int i_mb_y;

	int i_mb_width;
	int i_mb_height;

	int bw;
	int bh;

	int mv_min_x;
	int mv_min_y;
	int mv_max_x;
	int mv_max_y;

	pixel *fenc_buf;
	pixel *fref_buf;

	pixel *dev_fenc_buf;
	pixel *dev_fref_buf;
	int stride_buf;

	uint16_t *p_cost_mvx;
	uint16_t *p_cost_mvy;
} x264_cuda_t;

/* Motion Vector and Cost */
typedef struct x264_mvc_t
{
	int mx;
	int my;
	int cost;
} x264_mvc_t;

#endif  // __X264_CUDA_COMMON_H__

