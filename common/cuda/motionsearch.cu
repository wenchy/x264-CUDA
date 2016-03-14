#include "x264-cuda.h"

__global__ void me( pixel *dev_mb_enc, pixel *dev_frame_ref, int *dev_result, int stride_ref , int mb_x, int mb_y, int bmx, int bmy);
__device__ int pixel_sad_16x16( pixel *pix_enc, int stride_pix_enc, pixel *pix_ref, int stride_pix_ref );


/****************************************************************************
 * cuda_pixel_sad_WxH
 ****************************************************************************/
#define CUDA_PIXEL_SAD_C( name, lx, ly ) \
__device__ int name( pixel *pix1, intptr_t i_stride_pix1,  \
                 pixel *pix2, intptr_t i_stride_pix2 ) \
{                                                   \
    int i_sum = 0;                                  \
    for( int y = 0; y < ly; y++ )                   \
    {                                               \
        for( int x = 0; x < lx; x++ )               \
        {                                           \
            i_sum += abs( pix1[x] - pix2[x] );      \
        }                                           \
        pix1 += i_stride_pix1;                      \
        pix2 += i_stride_pix2;                      \
    }                                               \
    return i_sum;                                   \
}


CUDA_PIXEL_SAD_C( x264_CUDA_pixel_sad_16x16, 16, 16 )
CUDA_PIXEL_SAD_C( x264_CUDA_pixel_sad_16x8,  16,  8 )
CUDA_PIXEL_SAD_C( x264_CUDA_pixel_sad_8x16,   8, 16 )
CUDA_PIXEL_SAD_C( x264_CUDA_pixel_sad_8x8,    8,  8 )
CUDA_PIXEL_SAD_C( x264_CUDA_pixel_sad_8x4,    8,  4 )
CUDA_PIXEL_SAD_C( x264_CUDA_pixel_sad_4x16,   4, 16 )
CUDA_PIXEL_SAD_C( x264_CUDA_pixel_sad_4x8,    4,  8 )
CUDA_PIXEL_SAD_C( x264_CUDA_pixel_sad_4x4,    4,  4 )

extern "C" void cuda_me( x264_cuda_t *c, int *p_bmx, int *p_bmy, int *p_bcost ) {

	int i_me_range = c->i_me_range;
	int mb_x = c->i_mb_x;
	int mb_y = c->i_mb_y;

	pixel *fref_buf = c->fref_buf;
	pixel *mb_enc =  c->mb_enc;
	// int mb_stride = 16;

	int stride_ref = c->stride_ref;

	const uint16_t *p_cost_mvx = c->p_cost_mvx;
	const uint16_t *p_cost_mvy = c->p_cost_mvy;

	pixel *dev_mb_enc, *dev_fref_buf;
	int *dev_result;

	int *result = (int *)malloc( i_me_range*3 * i_me_range*3 * sizeof(int) );
	// allocate the memory on the GPU
	HANDLE_ERROR( cudaMalloc( (void**)&dev_mb_enc, 16*16 * sizeof(pixel) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_fref_buf, (352+PADH*2) * (288 + PADV*2) * sizeof(pixel) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_result,  i_me_range*3 * i_me_range*3 * sizeof(int) ) );

	// copy the arrays 'mb_enc' and 'frame_ref' to the GPU
	HANDLE_ERROR( cudaMemcpy( dev_mb_enc, mb_enc, (16 * 16) * sizeof(pixel), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( dev_fref_buf, fref_buf, (352+PADH*2) * (288 + PADV*2) * sizeof(pixel), cudaMemcpyHostToDevice ) );

//	printf("in bmx: %d\n", (*p_bmx));
//	printf("in bmy: %d\n", (*p_bmy));
//	printf("in bcost: %d\n", (*p_bcost));

	dim3 grid(16*3,16*3);

	me<<<grid,1>>>( dev_mb_enc, dev_fref_buf, dev_result, stride_ref, mb_x, mb_y, *p_bmx, *p_bmy);
	// copy the array 'dev_result' back from the GPU to the CPU
	HANDLE_ERROR( cudaMemcpy( result, dev_result, i_me_range*3 * i_me_range*3 * sizeof(int),
							  cudaMemcpyDeviceToHost ) );
	int *p_result = result;
//	for( int y = 0; y < 16*3*16*3; y++ )
//	{
//		printf("result[%d]: %d\n", y, p_result[y]);
//	}
	// process the results
	for( int y = 0; y < i_me_range*3; y++ )
	{
		for( int x = 0; x < i_me_range*3; x++ )
		{
			int mx = x + (*p_bmx - i_me_range);
			int my = y + (*p_bmy - i_me_range);
//			if(mx < c->mv_min_x || mx > c->mv_max_x || my < c->mv_min_y || my > c->mv_max_y)
//				continue;
			//printf("S: y: %d	x: %d    result: %d\n", y, x, p_result[x]);
			int cost =  p_result[x];// + (p_cost_mvx[(mx)<<2] + p_cost_mvy[(my)<<2]);
			if((cost)<(*p_bcost))
			{
				(*p_bcost)=(cost);
				(*p_bmx)=(mx);
				(*p_bmy)=(my);
			}
			//printf("E: y: %d	x: %d    result: %d\n", y, x, p_result[x]);
		}
		p_result += i_me_range*3;
	}
//	printf("out bmx: %d\n", (*p_bmx));
//	printf("out bmy: %d\n", (*p_bmy));
//	printf("out bcost: %d\n", (*p_bcost));

	// free the memory allocated on the CPU
	free(result);

	// free the memory allocated on the GPU
	HANDLE_ERROR( cudaFree( dev_mb_enc ) );
	HANDLE_ERROR( cudaFree( dev_fref_buf ) );
	HANDLE_ERROR( cudaFree( dev_result ) );

	return;
}

__global__ void me( pixel *dev_mb_enc, pixel *dev_fref_buf, int *dev_result, int stride_ref , int mb_x, int mb_y, int bmx, int bmy) {
	// map from blockIdx to pixel position
	pixel *p_fenc_plane = dev_fref_buf + stride_ref * PADV + PADH;
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y * gridDim.x;
	pixel *p_fref = p_fenc_plane + ( 16 * (mb_x - 1) + x ) +( 16 *  (mb_y - 1) + y )* stride_ref;
	p_fref += bmx +bmy * stride_ref;

	dev_result[offset] = x264_CUDA_pixel_sad_16x16(dev_mb_enc, 16, p_fref, stride_ref);
}

//__global__ void cost_cmp( int *dev_result) {
//	// map from blockIdx to pixel position
//	int x = blockIdx.x;
//	int y = blockIdx.y;
//	int offset = x + y * gridDim.x;
//	dev_result[offset]
//}

/* One thread performs 16x16 SAD */
__device__ int pixel_sad_16x16( pixel *pix_enc, int stride_pix_enc,
					 pixel *pix_ref, int stride_pix_ref )
{
	int cost = 0;
	for( int y = 0; y < 16; y++ )
	{
		for( int x = 0; x < 16; x++ )
		{
			cost += abs( pix_enc[x] - pix_ref[x] );
		}
		pix_enc += stride_pix_enc;
		pix_ref += stride_pix_ref;
	}
	return cost;

}

