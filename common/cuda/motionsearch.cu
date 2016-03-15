#include "x264-cuda.h"
#include "motionsearch.h"

__global__ void me( pixel *dev_mb_enc, pixel *dev_fref_buf, int *dev_sads, int me_range, int stride_ref , int mb_x, int mb_y, int bmx, int bmy);
__global__ void cmp(int *dev_sads, x264_mvc_t *dev_mvc, int me_range);
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


CUDA_PIXEL_SAD_C( x264_cuda_pixel_sad_16x16, 16, 16 )
CUDA_PIXEL_SAD_C( x264_cuda_pixel_sad_16x8,  16,  8 )
CUDA_PIXEL_SAD_C( x264_cuda_pixel_sad_8x16,   8, 16 )
CUDA_PIXEL_SAD_C( x264_cuda_pixel_sad_8x8,    8,  8 )
CUDA_PIXEL_SAD_C( x264_cuda_pixel_sad_8x4,    8,  4 )
CUDA_PIXEL_SAD_C( x264_cuda_pixel_sad_4x16,   4, 16 )
CUDA_PIXEL_SAD_C( x264_cuda_pixel_sad_4x8,    4,  8 )
CUDA_PIXEL_SAD_C( x264_cuda_pixel_sad_4x4,    4,  4 )

extern "C" void cuda_me_init( x264_cuda_t *c) {
	int mb_width = c->i_mb_width;
	int mb_height = c->i_mb_height;
	HANDLE_ERROR( cudaMalloc( (void**)&(c->dev_fenc_buf), (mb_width+PADH*2) * (mb_height + PADV*2) * sizeof(pixel) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&(c->dev_fref_buf), (mb_width+PADH*2) * (mb_height + PADV*2) * sizeof(pixel) ) );
}

extern "C" void cuda_me_end( x264_cuda_t *c) {
	HANDLE_ERROR( cudaFree( c->dev_fenc_buf ) );
	HANDLE_ERROR( cudaFree( c->dev_fref_buf ) );
}

extern "C" void cuda_me_fref_prefetch( x264_cuda_t *c) {
	int mb_width = c->i_mb_width;
	int mb_height = c->i_mb_height;
	// copy 'fenc_buf' and 'fref_buf'  to the GPU memory
	HANDLE_ERROR( cudaMemcpy( c->dev_fenc_buf, c->fenc_buf, (mb_width+PADH*2) * (mb_height + PADV*2) * sizeof(pixel), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( c->dev_fref_buf, c->fref_buf, (mb_width+PADH*2) * (mb_height + PADV*2) * sizeof(pixel), cudaMemcpyHostToDevice ) );
}

extern "C" void cuda_me( x264_cuda_t *c, int *p_bmx, int *p_bmy, int *p_bcost ) {

	int me_range = c->i_me_range + 8;
	int mb_width = c->i_mb_width;
	int mb_height = c->i_mb_height;
	int mb_x = c->i_mb_x;
	int mb_y = c->i_mb_y;

	int stride_buf = c->stride_buf;

	pixel *dev_mb_enc;
	int *dev_sads;
	x264_mvc_t *mvc; // CUDA Unified Memory

	// allocate the memory on the GPU
	HANDLE_ERROR( cudaMalloc( (void**)&dev_mb_enc, 16*16 * sizeof(pixel) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&dev_sads, me_range*2 * me_range*2 * sizeof( int) ) );
	// CUDA Unified Memory
	HANDLE_ERROR( cudaMallocManaged( (void**)&mvc, sizeof( x264_mvc_t) ) );

	mvc->mx = *p_bmx;
	mvc->my = *p_bmy;
	mvc->cost = *p_bcost;

	dim3 grid_sad(me_range*2, me_range*2);
	me<<<grid_sad, 1>>>( c->dev_fenc_buf, c->dev_fref_buf, dev_sads, me_range, stride_buf, mb_x, mb_y, *p_bmx, *p_bmy);

	dim3 grid_cmp(1, 1);
	cmp<<<grid_cmp, 1>>>( dev_sads, mvc, me_range);
	cudaDeviceSynchronize();

	(*p_bcost)= mvc->cost;
	(*p_bmx)= mvc->mx;
	(*p_bmy)=mvc->my;

	// free the memory allocated on the GPU
	HANDLE_ERROR( cudaFree( dev_mb_enc ) );
	HANDLE_ERROR( cudaFree( dev_sads ) );
	HANDLE_ERROR( cudaFree( mvc ) );

	return;
}

__global__ void me( pixel *dev_fenc_buf, pixel *dev_fref_buf, int *dev_sads, int me_range, int stride_buf , int mb_x, int mb_y, int bmx, int bmy) {
	// map from blockIdx to pixel position
	pixel *p_fenc_plane = dev_fenc_buf + stride_buf * PADV + PADH;
	pixel *p_fref_plane = dev_fref_buf + stride_buf * PADV + PADH;
	int x = blockIdx.x;
	int y = blockIdx.y;
	int offset = x + y * gridDim.x;
	pixel *p_fenc = p_fenc_plane + ( 16 * mb_x + x ) +( 16 * mb_y + y )* stride_buf;
	pixel *p_fref = p_fref_plane + ( 16 * mb_x + x - me_range) +( 16 * mb_y + y - me_range)* stride_buf;
	p_fref += bmx +bmy * stride_buf;

	dev_sads[offset] = x264_cuda_pixel_sad_16x16(p_fenc, stride_buf, p_fref, stride_buf);
}

__global__ void cmp(int *dev_sads, x264_mvc_t *mvc, int me_range){
	// map from blockIdx to pixel position
	int bmx = mvc->mx;
	int bmy = mvc->my;
	int *p_sad = dev_sads;
	for( int y = 0; y < me_range*2; y++ )
	{
		for( int x = 0; x < me_range*2; x++ )
		{
			int mx = x + (bmx - me_range);
			int my = y + (bmy - me_range);
			int cost =  p_sad[x];
			if((cost)<(mvc->cost))
			{
				mvc->cost = cost;
				mvc->mx = mx;
				mvc->my = my;
			}
		}
		p_sad += me_range*2;
	}
}
