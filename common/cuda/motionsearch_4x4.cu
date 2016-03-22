#include "x264-cuda.h"
#include "motionsearch.h"

__global__ void me( int i_pixel, pixel *dev_fenc_buf, pixel *dev_fref_buf, x264_cuda_me_t *me, int me_range, int stride_buf);

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
CUDA_PIXEL_SAD_C( x264_cuda_pixel_sad_4x8,    4,  8 )
CUDA_PIXEL_SAD_C( x264_cuda_pixel_sad_4x4,    4,  4 )


extern "C" void cuda_me_init( x264_cuda_t *c) {
	int buf_width = 16 * c->i_mb_width + PADH*2;
	int buf_height = 16 * c->i_mb_height + PADV*2;
	HANDLE_ERROR( cudaMalloc( (void**)&(c->dev_fenc_buf), buf_width * buf_height * sizeof(pixel) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&(c->dev_fref_buf), buf_width * buf_height * sizeof(pixel) ) );

	// CUDA Unified Memory: mb mvc
	HANDLE_ERROR( cudaMallocManaged( (void**)&(c->me), (c->i_mb_width * c->i_mb_height) * sizeof(x264_cuda_me_t) ) );

	printf("*****cuda_me_init***** %lu x 41 = %lu\n", sizeof(x264_cuda_mvc_t), sizeof(x264_cuda_me_t));
}

extern "C" void cuda_me_end( x264_cuda_t *c) {
	HANDLE_ERROR( cudaFree( c->dev_fenc_buf ) );
	HANDLE_ERROR( cudaFree( c->dev_fref_buf ) );
	HANDLE_ERROR( cudaFree( c->me ) );
	printf("*****cuda_me_end*****\n");
}

extern "C" void cuda_me_prefetch( x264_cuda_t *c) {
	int buf_width = 16 * c->i_mb_width + PADH*2;
	int buf_height = 16 * c->i_mb_height + PADV*2;
	// copy 'fenc_buf' and 'fref_buf'  to the GPU memory
	HANDLE_ERROR( cudaMemcpy( c->dev_fenc_buf, c->fenc_buf, buf_width * buf_height * sizeof(pixel), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( c->dev_fref_buf, c->fref_buf, buf_width * buf_height * sizeof(pixel), cudaMemcpyHostToDevice ) );
	//printf("*****cuda_me_prefetch*****\n");
}

extern "C" void cuda_me( x264_cuda_t *c) {

	int me_range = c->i_me_range;
	int mb_width = c->i_mb_width;
	int mb_height = c->i_mb_height;
	int stride_buf = c->stride_buf;


	dim3    blocks(mb_width, mb_height);
//	dim3    threads(me_range*2, me_range*2);

	me<<<blocks, THREADS_PER_BLOCK>>>( c->i_pixel, c->dev_fenc_buf, c->dev_fref_buf, c->me, me_range, stride_buf);
	HANDLE_ERROR( cudaPeekAtLastError() );
	HANDLE_ERROR( cudaDeviceSynchronize() );


	return;
}

/* *****************************************************************
 * 1. 4x4 SAD merging into 16x16
 * 2. find least SAD and corresponding MV
 * *****************************************************************/
__device__ void me_merge_16x16( int16_t sadCache[4][THREADS_PER_BLOCK], int sadMerge[THREADS_PER_BLOCK], int index[THREADS_PER_BLOCK], x264_cuda_me_t *me, int me_range )
{
	const int offset = threadIdx.x;
	int mb_x = blockIdx.x;
	int mb_y = blockIdx.y;
	int mb_width = gridDim.x;

	for(int i16x16 = 0; i16x16 < 1; i16x16++)
	{
		// set the sads values
		sadMerge[offset] = 0;
		for(int i4x4 = 0; i4x4 < 16; i4x4++)
		{
			sadMerge[offset] += sadCache[i4x4][offset];
		}
		index[offset] = offset;

		// synchronize threads in this block
		__syncthreads();

		// for reductions, THREADS_PER_BLOCK must be a power of 2
		// because of the following code： find least SAD
		int i = blockDim.x/2;
		while (i != 0) {
			if (offset < i)
			{
				if (sadMerge[ index[offset] ] > sadMerge[ index[offset + i] ])
				{
					index[offset] = index[offset + i];
				}
			}
			__syncthreads();
			i /= 2;
		}

		if (offset == 0)
		{
			me[mb_x + mb_y*mb_width].mvc16x16.mv[0] = index[0] % ( me_range*2 ) - me_range;
			me[mb_x + mb_y*mb_width].mvc16x16.mv[1] = index[0] / ( me_range*2 ) - me_range;
			me[mb_x + mb_y*mb_width].mvc16x16.cost = sadMerge[ index[0] ];
		}
		__syncthreads();
	}
}

/* *****************************************************************
 * 1. 4x4 SAD merging into 16x8
 * 2. find least SAD and corresponding MV
 * *****************************************************************/
__device__ void me_merge_16x8( int16_t sadCache[4][THREADS_PER_BLOCK], int sadMerge[THREADS_PER_BLOCK], int index[THREADS_PER_BLOCK], x264_cuda_me_t *me, int me_range )
{
	const int offset = threadIdx.x;
	int mb_x = blockIdx.x;
	int mb_y = blockIdx.y;
	int mb_width = gridDim.x;

	for(int i16x8 = 0; i16x8 < 2; i16x8++)
	{
		// set the sads values
		sadMerge[offset] = 0;
		for(int i4x4_y = i16x8 * 2; i4x4_y < i16x8 * 2 + 2; i4x4_y++)
		{
			for(int i4x4_x = 0; i4x4_x < 4; i4x4_x++)
			{
				sadMerge[offset] += sadCache[i4x4_x + i4x4_y * 4][offset];
			}
		}
		index[offset] = offset;

		// synchronize threads in this block
		__syncthreads();

		// for reductions, THREADS_PER_BLOCK must be a power of 2
		// because of the following code： find least SAD
		int i = blockDim.x/2;
		while (i != 0) {
			if (offset < i)
			{
				if (sadMerge[ index[offset] ] > sadMerge[ index[offset + i] ])
				{
					index[offset] = index[offset + i];
				}
			}
			__syncthreads();
			i /= 2;
		}

		if (offset == 0)
		{
			me[mb_x + mb_y*mb_width].mvc16x8[i16x8].mv[0] = index[0] % ( me_range*2 ) - me_range;
			me[mb_x + mb_y*mb_width].mvc16x8[i16x8].mv[1] = index[0] / ( me_range*2 ) - me_range;
			me[mb_x + mb_y*mb_width].mvc16x8[i16x8].cost = sadMerge[ index[0] ];
		}
		__syncthreads();
	}
}

/* *****************************************************************
 * 1. 4x4 SAD merging into 8x16
 * 2. find least SAD and corresponding MV
 * *****************************************************************/
__device__ void me_merge_8x16( int16_t sadCache[4][THREADS_PER_BLOCK], int sadMerge[THREADS_PER_BLOCK], int index[THREADS_PER_BLOCK], x264_cuda_me_t *me, int me_range )
{
	const int offset = threadIdx.x;
	int mb_x = blockIdx.x;
	int mb_y = blockIdx.y;
	int mb_width = gridDim.x;

	for(int i8x16 = 0; i8x16 < 2; i8x16++)
	{
		// set the sads values
		sadMerge[offset] = 0;
		for(int i4x4_y = 0; i4x4_y < 4; i4x4_y++)
		{
			for(int i4x4_x = i8x16 * 2; i4x4_x < i8x16 * 2 + 2; i4x4_x++)
			{
				sadMerge[offset] += sadCache[i4x4_x + i4x4_y * 4][offset];
			}
		}
		index[offset] = offset;

		// synchronize threads in this block
		__syncthreads();

		// for reductions, THREADS_PER_BLOCK must be a power of 2
		// because of the following code： find least SAD
		int i = blockDim.x/2;
		while (i != 0) {
			if (offset < i)
			{
				if (sadMerge[ index[offset] ] > sadMerge[ index[offset + i] ])
				{
					index[offset] = index[offset + i];
				}
			}
			__syncthreads();
			i /= 2;
		}

		if (offset == 0)
		{
			me[mb_x + mb_y*mb_width].mvc8x16[i8x16].mv[0] = index[0] % ( me_range*2 ) - me_range;
			me[mb_x + mb_y*mb_width].mvc8x16[i8x16].mv[1] = index[0] / ( me_range*2 ) - me_range;
			me[mb_x + mb_y*mb_width].mvc8x16[i8x16].cost = sadMerge[ index[0] ];
		}
		__syncthreads();
	}
}

/* *****************************************************************
 * 1. 4x4 SAD merging into 8x8
 * 2. find least SAD and corresponding MV
 * *****************************************************************/
__device__ void me_merge_8x8( int16_t sadCache[4][THREADS_PER_BLOCK], int sadMerge[THREADS_PER_BLOCK], int index[THREADS_PER_BLOCK], x264_cuda_me_t *me, int me_range )
{
	const int offset = threadIdx.x;
	int mb_x = blockIdx.x;
	int mb_y = blockIdx.y;
	int mb_width = gridDim.x;

	for(int i8x8_y = 0; i8x8_y < 2; i8x8_y++)
	{
		for(int i8x8_x = 0; i8x8_x < 2; i8x8_x++)
		{
			// set the sads values
			sadMerge[offset] = 0;
			for(int i4x4_y = i8x8_y * 2; i4x4_y < i8x8_y * 2 + 2; i4x4_y++)
			{
				for(int i4x4_x = i8x8_x * 2; i4x4_x < i8x8_x * 2 + 2; i4x4_x++)
				{
					sadMerge[offset] += sadCache[i4x4_x + i4x4_y * 4][offset];
				}
			}
			index[offset] = offset;

			// synchronize threads in this block
			__syncthreads();

			// for reductions, THREADS_PER_BLOCK must be a power of 2
			// because of the following code： find least SAD
			int i = blockDim.x/2;
			while (i != 0) {
				if (offset < i)
				{
					if (sadMerge[ index[offset] ] > sadMerge[ index[offset + i] ])
					{
						index[offset] = index[offset + i];
					}
				}
				__syncthreads();
				i /= 2;
			}

			if (offset == 0)
			{
				me[mb_x + mb_y*mb_width].mvc8x8[i8x8_x + i8x8_y * 2].mv[0] = index[0] % ( me_range*2 ) - me_range;
				me[mb_x + mb_y*mb_width].mvc8x8[i8x8_x + i8x8_y * 2].mv[1] = index[0] / ( me_range*2 ) - me_range;
				me[mb_x + mb_y*mb_width].mvc8x8[i8x8_x + i8x8_y * 2].cost = sadMerge[ index[0] ];
			}
			__syncthreads();
		}
	}
}

/* *****************************************************************
 * 1. 4x4 SAD merging into 8x4
 * 2. find least SAD and corresponding MV
 * *****************************************************************/
__device__ void me_merge_8x4( int16_t sadCache[4][THREADS_PER_BLOCK], int sadMerge[THREADS_PER_BLOCK], int index[THREADS_PER_BLOCK], x264_cuda_me_t *me, int me_range )
{
	const int offset = threadIdx.x;
	int mb_x = blockIdx.x;
	int mb_y = blockIdx.y;
	int mb_width = gridDim.x;

	for(int i8x4_y = 0; i8x4_y < 4; i8x4_y++)
	{
		for(int i8x4_x = 0; i8x4_x < 2; i8x4_x++)
		{
			// set the sads values
			sadMerge[offset] = 0;
			for(int i4x4_y = i8x4_y; i4x4_y < i8x4_y + 1; i4x4_y++)
			{
				for(int i4x4_x = i8x4_x * 2; i4x4_x < i8x4_x * 2 + 2; i4x4_x++)
				{
					sadMerge[offset] += sadCache[i4x4_x + i4x4_y * 4][offset];
				}
			}
			index[offset] = offset;

			// synchronize threads in this block
			__syncthreads();

			// for reductions, THREADS_PER_BLOCK must be a power of 2
			// because of the following code： find least SAD
			int i = blockDim.x/2;
			while (i != 0) {
				if (offset < i)
				{
					if (sadMerge[ index[offset] ] > sadMerge[ index[offset + i] ])
					{
						index[offset] = index[offset + i];
					}
				}
				__syncthreads();
				i /= 2;
			}

			if (offset == 0)
			{
				me[mb_x + mb_y*mb_width].mvc8x4[i8x4_x + i8x4_y * 2].mv[0] = index[0] % ( me_range*2 ) - me_range;
				me[mb_x + mb_y*mb_width].mvc8x4[i8x4_x + i8x4_y * 2].mv[1] = index[0] / ( me_range*2 ) - me_range;
				me[mb_x + mb_y*mb_width].mvc8x4[i8x4_x + i8x4_y * 2].cost = sadMerge[ index[0] ];
			}
			__syncthreads();
		}
	}
}

/* *****************************************************************
 * 1. 4x4 SAD merging into 4x8
 * 2. find least SAD and corresponding MV
 * *****************************************************************/
__device__ void me_merge_4x8( int16_t sadCache[4][THREADS_PER_BLOCK], int sadMerge[THREADS_PER_BLOCK], int index[THREADS_PER_BLOCK], x264_cuda_me_t *me, int me_range )
{
	const int offset = threadIdx.x;
	int mb_x = blockIdx.x;
	int mb_y = blockIdx.y;
	int mb_width = gridDim.x;

	for(int i4x8_y = 0; i4x8_y < 2; i4x8_y++)
	{
		for(int i4x8_x = 0; i4x8_x < 4; i4x8_x++)
		{
			// set the sads values
			sadMerge[offset] = 0;
			for(int i4x4_y = i4x8_y * 2; i4x4_y < i4x8_y * 2 + 2; i4x4_y++)
			{
				for(int i4x4_x = i4x8_x; i4x4_x < i4x8_x + 1; i4x4_x++)
				{
					sadMerge[offset] += sadCache[i4x4_x + i4x4_y * 4][offset];
				}
			}
			index[offset] = offset;

			// synchronize threads in this block
			__syncthreads();

			// for reductions, THREADS_PER_BLOCK must be a power of 2
			// because of the following code： find least SAD
			int i = blockDim.x/2;
			while (i != 0) {
				if (offset < i)
				{
					if (sadMerge[ index[offset] ] > sadMerge[ index[offset + i] ])
					{
						index[offset] = index[offset + i];
					}
				}
				__syncthreads();
				i /= 2;
			}

			if (offset == 0)
			{
				me[mb_x + mb_y*mb_width].mvc4x8[i4x8_x + i4x8_y * 2].mv[0] = index[0] % ( me_range*2 ) - me_range;
				me[mb_x + mb_y*mb_width].mvc4x8[i4x8_x + i4x8_y * 2].mv[1] = index[0] / ( me_range*2 ) - me_range;
				me[mb_x + mb_y*mb_width].mvc4x8[i4x8_x + i4x8_y * 2].cost = sadMerge[ index[0] ];
			}
			__syncthreads();
		}
	}
}

__global__ void me( int i_pixel, pixel *dev_fenc_buf, pixel *dev_fref_buf, x264_cuda_me_t *me, int me_range, int stride_buf) {
	 __shared__ int16_t sadCache[16][THREADS_PER_BLOCK]; // 32k
	 __shared__ int sadMerge[THREADS_PER_BLOCK]; // 4k
	 __shared__ int index[THREADS_PER_BLOCK]; // 4k

	const int offset = threadIdx.x;
	int ox = threadIdx.x % ( me_range*2 );
	int oy = threadIdx.x / ( me_range*2 );

	int mb_x = blockIdx.x;
	int mb_y = blockIdx.y;
	int mb_width = gridDim.x;

	pixel *p_fenc_plane = dev_fenc_buf + stride_buf * PADV + PADH;
	pixel *p_fref_plane = dev_fref_buf + stride_buf * PADV + PADH;

	pixel *p_fenc = p_fenc_plane + ( 16 * mb_x) +( 16 * mb_y)* stride_buf;
	pixel *p_fref = p_fref_plane + ( 16 * mb_x- me_range) +( 16 * mb_y  - me_range)* stride_buf;

	for(int i4x4 = 0; i4x4 < 16; i4x4++)
	{
		int i4x4_x = i4x4 % 4;
		int i4x4_y = i4x4 / 4;

		pixel *p_enc = p_fenc + ( i4x4_x * 4 + (i4x4_y * 4) * stride_buf );
		pixel *p_ref = p_fref + ox+ oy * stride_buf + ( i4x4_x * 4 + (i4x4_y * 4) * stride_buf );

		// set the sads values
		sadCache[i4x4][offset] = x264_cuda_pixel_sad_4x4(p_enc, stride_buf, p_ref, stride_buf);
		index[offset] = offset;

		// synchronize threads in this block
		__syncthreads();

		// for reductions, THREADS_PER_BLOCK must be a power of 2
		// because of the following code： find least SAD
		int i = blockDim.x/2;
		while (i != 0) {
			if (offset < i)
			{
				if (sadCache[i4x4][ index[offset] ] > sadCache[i4x4][ index[offset + i] ])
				{
					index[offset] = index[offset + i];
				}
			}
			__syncthreads();
			i /= 2;
		}

		if (offset == 0)
		{
			me[mb_x + mb_y*mb_width].mvc4x4[i4x4].mv[0] = index[0] % ( me_range*2 ) - me_range;
			me[mb_x + mb_y*mb_width].mvc4x4[i4x4].mv[1] = index[0] / ( me_range*2 ) - me_range;
			me[mb_x + mb_y*mb_width].mvc4x4[i4x4].cost = sadCache[i4x4][ index[0] ];
		}
		__syncthreads();
	}

	me_merge_16x16(sadCache, sadMerge, index, me, me_range);
	me_merge_16x8(sadCache, sadMerge, index, me, me_range);
	me_merge_8x16(sadCache, sadMerge, index, me, me_range);
	me_merge_8x8(sadCache, sadMerge, index, me, me_range);
	me_merge_8x4(sadCache, sadMerge, index, me, me_range);
	me_merge_4x8(sadCache, sadMerge, index, me, me_range);

}
