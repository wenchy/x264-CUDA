/*
 * 1. 8x8 SAD merge
 * 2. Two step and Square search Algorithm: 8*8 = (16*2)/4 * (16*2)/4, and  (4+1)*(4+1)
 */
#include "x264-cuda.h"
#include "motionsearch.h"

__global__ void me(pixel *dev_fenc_buf, pixel *dev_fref_buf, x264_cuda_me_t *me, int me_range, int stride_buf);

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

	// mb mvc
	c->me = (x264_cuda_me_t *)malloc( (c->i_mb_width * c->i_mb_height) * sizeof(x264_cuda_me_t) );
	HANDLE_ERROR( cudaMalloc( (void**)&(c->dev_me), (c->i_mb_width * c->i_mb_height) * sizeof(x264_cuda_me_t) ) );
	// CUDA Unified Memory
	//HANDLE_ERROR( cudaMallocManaged( (void**)&(c->dev_me), (c->i_mb_width * c->i_mb_height) * sizeof(x264_cuda_me_t) ) );
}

extern "C" void cuda_me_end( x264_cuda_t *c) {
	HANDLE_ERROR( cudaFree( c->dev_fenc_buf ) );
	HANDLE_ERROR( cudaFree( c->dev_fref_buf ) );
	HANDLE_ERROR( cudaFree( c->dev_me ) );
	free( c->me );
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

extern "C" void cuda_me0( x264_cuda_t *c) {

	int me_range = c->i_me_range;
	int mb_width = c->i_mb_width;
	int mb_height = c->i_mb_height;
	int stride_buf = c->stride_buf;

//	cudaEvent_t start, stop;
//	cudaEventCreate(&start);
//	cudaEventCreate(&stop);
//	cudaEventRecord( start, 0 );

	int buf_width = 16 * c->i_mb_width + PADH*2;
	int buf_height = 16 * c->i_mb_height + PADV*2;
	// copy 'fenc_buf' and 'fref_buf'  to the GPU memory
	HANDLE_ERROR( cudaMemcpy( c->dev_fenc_buf, c->fenc_buf, buf_width * buf_height * sizeof(pixel), cudaMemcpyHostToDevice ) );
	HANDLE_ERROR( cudaMemcpy( c->dev_fref_buf, c->fref_buf, buf_width * buf_height * sizeof(pixel), cudaMemcpyHostToDevice ) );
	//printf("*****cuda_me_prefetch*****\n");



	dim3    blocks(mb_width, mb_height);


	me<<<blocks, QQ_THREADS_PER_BLOCK>>>(c->dev_fenc_buf, c->dev_fref_buf, c->dev_me, me_range, stride_buf);
//	HANDLE_ERROR( cudaPeekAtLastError() );
//	HANDLE_ERROR( cudaDeviceSynchronize() );



	// copy the 'me' back from the GPU to the CPU
	 HANDLE_ERROR( cudaMemcpy( c->me, c->dev_me, (c->i_mb_width * c->i_mb_height) * sizeof(x264_cuda_me_t), cudaMemcpyDeviceToHost ) );

//	cudaEventRecord( stop, 0 );
//	cudaEventSynchronize( stop );
//	float elapsedTime;
//	HANDLE_ERROR( cudaEventElapsedTime( &elapsedTime, start, stop ) );
//	printf( "Time to generate: %3.1f ms\n", elapsedTime );
//	HANDLE_ERROR( cudaEventDestroy( start ) );
//	HANDLE_ERROR( cudaEventDestroy( stop ) );

	return;
}

extern "C" void *cuda_me( void *arg ) {

	x264_cuda_t *c= (x264_cuda_t *)arg;

	int me_range = c->i_me_range;
	int mb_width = c->i_mb_width;
	int mb_height = c->i_mb_height;
	int stride_buf = c->stride_buf;

	dim3    blocks(mb_width, mb_height);


	me<<<blocks, QQ_THREADS_PER_BLOCK>>>( c->dev_fenc_buf, c->dev_fref_buf, c->dev_me, me_range, stride_buf);
//	HANDLE_ERROR( cudaPeekAtLastError() );
//	HANDLE_ERROR( cudaDeviceSynchronize() );



	// copy the 'me' back from the GPU to the CPU
	 HANDLE_ERROR( cudaMemcpy( c->me, c->dev_me, (c->i_mb_width * c->i_mb_height) * sizeof(x264_cuda_me_t), cudaMemcpyDeviceToHost ) );
	 return(NULL);
}

/* *****************************************************************
 * 1. 8x8 SAD merging into 16x16
 * 2. find least SAD and corresponding MV
 * *****************************************************************/
__device__ void me_merge_16x16( int sadCache[4][QQ_THREADS_PER_BLOCK], int sadMerge[QQ_THREADS_PER_BLOCK], int index[THREADS_PER_BLOCK], x264_cuda_me_t *me, int me_range, pixel *p_fenc, pixel *p_fref, int stride_buf)
{
	 __shared__ int sadSquare[QQ_THREADS_PER_BLOCK];
	 __shared__ int sindex[QQ_THREADS_PER_BLOCK];
	 int sub_range = me_range/2;

	const int offset = threadIdx.x;
	int mb_x = blockIdx.x;
	int mb_y = blockIdx.y;
	int mb_width = gridDim.x;

	for(int i16x16 = 0; i16x16 < 1; i16x16++)
	{
		// set the sads values
		sadMerge[offset] = 0;
		for(int i8x8 = 0; i8x8 < 4; i8x8++)
		{
			sadMerge[offset] += sadCache[i8x8][offset];
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

//		if (offset == 0)
//		{
////			me[mb_x + mb_y*mb_width].mvc16x16.mv[0] = index[0] % ( me_range*2 ) - me_range;
////			me[mb_x + mb_y*mb_width].mvc16x16.mv[1] = index[0] / ( me_range*2 ) - me_range;
//			me[mb_x + mb_y*mb_width].mvc16x16.mv[0] = index[0] % me_range * 2 - me_range;
//			me[mb_x + mb_y*mb_width].mvc16x16.mv[1] = index[0] / me_range * 2 - me_range;
//			me[mb_x + mb_y*mb_width].mvc16x16.cost = sadMerge[ index[0] ];
//		}
//		__syncthreads();

		sadSquare[offset] = MAX_INT;
		sindex[offset] = offset;
		if (offset < 25)
		{

			int ox = index[0] % sub_range * 4 - 2 + offset%5;
			int oy = index[0] / sub_range * 4 - 2 + offset/5;

			pixel *p_ref = p_fref + ox+ oy * stride_buf;

			// set the sads values
			sadSquare[offset] = x264_cuda_pixel_sad_16x16(p_fenc, stride_buf, p_ref, stride_buf);

		}

		// synchronize threads in this block
		__syncthreads();

		// for reductions, QQ_THREADS_PER_BLOCK must be a power of 2
		// because of the following code： find least SAD
		i = blockDim.x/4;
		while (i != 0) {
			if (offset < i)
			{
				if (sadSquare[ sindex[offset] ] > sadSquare[ sindex[offset + i] ])
				{
					sindex[offset] = sindex[offset + i];
				}
			}
			__syncthreads();
			i /= 2;
		}

		if (offset == 0)
		{
			me[mb_x + mb_y*mb_width].mvc16x16.mv[0] = index[0] % sub_range * 4 - me_range + sindex[0]%5 - 2;
			me[mb_x + mb_y*mb_width].mvc16x16.mv[1] = index[0] / sub_range * 4 - me_range + sindex[0]/5 - 2;
			me[mb_x + mb_y*mb_width].mvc16x16.cost = sadSquare[ sindex[0] ];
		}
		__syncthreads();
	}
}

/* *****************************************************************
 * 1. 8x8 SAD merging into 16x8
 * 2. find least SAD and corresponding MV
 * *****************************************************************/
__device__ void me_merge_16x8( int sadCache[4][QQ_THREADS_PER_BLOCK], int sadMerge[QQ_THREADS_PER_BLOCK], int index[QQ_THREADS_PER_BLOCK], x264_cuda_me_t *me, int me_range, pixel *p_fenc, pixel *p_fref, int stride_buf )
{
	 __shared__ int sadSquare[QQ_THREADS_PER_BLOCK];
	 __shared__ int sindex[QQ_THREADS_PER_BLOCK];
	 int sub_range = me_range/2;

	const int offset = threadIdx.x;
	int mb_x = blockIdx.x;
	int mb_y = blockIdx.y;
	int mb_width = gridDim.x;

	for(int i16x8 = 0; i16x8 < 2; i16x8++)
	{

		// set the sads values
		sadMerge[offset] = sadCache[i16x8*2][offset] + sadCache[i16x8*2 + 1][offset];
		index[offset] = offset;

		// synchronize threads in this block
		__syncthreads();

		// for reductions, QQ_THREADS_PER_BLOCK must be a power of 2
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

//		if (offset == 0)
//		{
////			me[mb_x + mb_y*mb_width].mvc16x8[i16x8].mv[0] = index[0] % ( me_range*2 ) - me_range;
////			me[mb_x + mb_y*mb_width].mvc16x8[i16x8].mv[1] = index[0] / ( me_range*2 ) - me_range;
//			me[mb_x + mb_y*mb_width].mvc16x8[i16x8].mv[0] = index[0] % me_range * 2 - me_range;
//			me[mb_x + mb_y*mb_width].mvc16x8[i16x8].mv[1] = index[0] / me_range * 2 - me_range;
//			me[mb_x + mb_y*mb_width].mvc16x8[i16x8].cost = sadMerge[ index[0] ];
//		}
//		__syncthreads();

		sadSquare[offset] = MAX_INT;
		sindex[offset] = offset;
		if (offset < 25)
		{

			int ox = index[0] % sub_range * 4 - 2 + offset%5;
			int oy = index[0] / sub_range * 4 - 2 + offset/5;

			pixel *p_ref = p_fref + ox+ oy * stride_buf + ( i16x8 * 8 ) * stride_buf;

			// set the sads values
			sadSquare[offset] = x264_cuda_pixel_sad_16x8(p_fenc + ( i16x8 * 8 ) * stride_buf, stride_buf, p_ref, stride_buf);

		}

		// synchronize threads in this block
		__syncthreads();

		// for reductions, QQ_THREADS_PER_BLOCK must be a power of 2
		// because of the following code： find least SAD
		i = blockDim.x/4;
		while (i != 0) {
			if (offset < i)
			{
				if (sadSquare[ sindex[offset] ] > sadSquare[ sindex[offset + i] ])
				{
					sindex[offset] = sindex[offset + i];
				}
			}
			__syncthreads();
			i /= 2;
		}

		if (offset == 0)
		{
			me[mb_x + mb_y*mb_width].mvc16x8[i16x8].mv[0] = index[0] % sub_range * 4 - me_range + sindex[0]%5 - 2;
			me[mb_x + mb_y*mb_width].mvc16x8[i16x8].mv[1] = index[0] / sub_range * 4 - me_range + sindex[0]/5 - 2;
			me[mb_x + mb_y*mb_width].mvc16x8[i16x8].cost = sadSquare[ sindex[0] ];
		}
		__syncthreads();
	}
}

/* *****************************************************************
 * 1. 8x8 SAD merging into 8x16
 * 2. find least SAD and corresponding MV
 * *****************************************************************/
__device__ void me_merge_8x16( int sadCache[4][QQ_THREADS_PER_BLOCK], int sadMerge[QQ_THREADS_PER_BLOCK], int index[QQ_THREADS_PER_BLOCK], x264_cuda_me_t *me, int me_range, pixel *p_fenc, pixel *p_fref, int stride_buf )
{
	 __shared__ int sadSquare[QQ_THREADS_PER_BLOCK];
	 __shared__ int sindex[QQ_THREADS_PER_BLOCK];
	 int sub_range = me_range/2;

	const int offset = threadIdx.x;
	int mb_x = blockIdx.x;
	int mb_y = blockIdx.y;
	int mb_width = gridDim.x;

	for(int i8x16 = 0; i8x16 < 2; i8x16++)
	{
		// set the sads values
		sadMerge[offset] = sadCache[i8x16][offset] + sadCache[i8x16 + 2][offset];
		index[offset] = offset;

		// synchronize threads in this block
		__syncthreads();

		// for reductions, QQ_THREADS_PER_BLOCK must be a power of 2
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

//		if (offset == 0)
//		{
//			me[mb_x + mb_y*mb_width].mvc8x16[i8x16].mv[0] = index[0] % me_range * 2 - me_range;
//			me[mb_x + mb_y*mb_width].mvc8x16[i8x16].mv[1] = index[0] / me_range * 2 - me_range;
//			me[mb_x + mb_y*mb_width].mvc8x16[i8x16].cost = sadMerge[ index[0] ];
//		}
//		__syncthreads();

		sadSquare[offset] = MAX_INT;
		sindex[offset] = offset;
		if (offset < 25)
		{

			int ox = index[0] % sub_range * 4 - 2 + offset%5;
			int oy = index[0] / sub_range * 4 - 2 + offset/5;

			pixel *p_ref = p_fref + ox+ oy * stride_buf + ( i8x16 * 8);

			// set the sads values
			sadSquare[offset] = x264_cuda_pixel_sad_8x16(p_fenc + i8x16 * 8, stride_buf, p_ref, stride_buf);

		}

		// synchronize threads in this block
		__syncthreads();

		// for reductions, QQ_THREADS_PER_BLOCK must be a power of 2
		// because of the following code： find least SAD
		i = blockDim.x/4;
		while (i != 0) {
			if (offset < i)
			{
				if (sadSquare[ sindex[offset] ] > sadSquare[ sindex[offset + i] ])
				{
					sindex[offset] = sindex[offset + i];
				}
			}
			__syncthreads();
			i /= 2;
		}

		if (offset == 0)
		{
			me[mb_x + mb_y*mb_width].mvc8x16[i8x16].mv[0] = index[0] % sub_range * 4 - me_range + sindex[0]%5 - 2;
			me[mb_x + mb_y*mb_width].mvc8x16[i8x16].mv[1] = index[0] / sub_range * 4 - me_range + sindex[0]/5 - 2;
			me[mb_x + mb_y*mb_width].mvc8x16[i8x16].cost = sadSquare[ sindex[0] ];
		}
		__syncthreads();
	}
}

__global__ void me(pixel *dev_fenc_buf, pixel *dev_fref_buf, x264_cuda_me_t *me, int me_range, int stride_buf) {
	 __shared__ int sadCache[4][QQ_THREADS_PER_BLOCK];
	 __shared__ int sadMerge[QQ_THREADS_PER_BLOCK];
	 __shared__ int index[QQ_THREADS_PER_BLOCK];

	 __shared__ int sadSquare[QQ_THREADS_PER_BLOCK];
	 __shared__ int sindex[QQ_THREADS_PER_BLOCK];

	int sub_range = me_range/2;

	const int offset = threadIdx.x;
//	int ox = threadIdx.x % ( me_range*2 );
//	int oy = threadIdx.x / ( me_range*2 );

	int ox = threadIdx.x % sub_range * 4;
	int oy = threadIdx.x / sub_range * 4;

	int mb_x = blockIdx.x;
	int mb_y = blockIdx.y;
	int mb_width = gridDim.x;

	pixel *p_fenc_plane = dev_fenc_buf + stride_buf * PADV + PADH;
	pixel *p_fref_plane = dev_fref_buf + stride_buf * PADV + PADH;

	pixel *p_fenc = p_fenc_plane + ( 16 * mb_x) +( 16 * mb_y)* stride_buf;
	pixel *p_fref = p_fref_plane + ( 16 * mb_x- me_range) +( 16 * mb_y  - me_range)* stride_buf;

	for(int i8x8 = 0; i8x8 < 4; i8x8++)
	{
		int i8x8_x = i8x8 % 2;
		int i8x8_y = i8x8 / 2;

		pixel *p_enc = p_fenc + ( i8x8_x * 8 + (i8x8_y * 8) * stride_buf );
		pixel *p_ref = p_fref + ox+ oy * stride_buf + ( i8x8_x * 8 + (i8x8_y * 8) * stride_buf );

		int temp = MAX_INT;
		temp = x264_cuda_pixel_sad_8x8(p_enc, stride_buf, p_ref, stride_buf);

		// set the sads values
		sadCache[i8x8][offset] = temp;
		index[offset] = offset;

		// synchronize threads in this block
		__syncthreads();

		// for reductions, QQ_THREADS_PER_BLOCK must be a power of 2
		// because of the following code： find least SAD
		int i = blockDim.x/2;
		while (i != 0) {
			if (offset < i)
			{
				if (sadCache[i8x8][ index[offset] ] > sadCache[i8x8][ index[offset + i] ])
				{
					index[offset] = index[offset + i];
				}
			}
			__syncthreads();
			i /= 2;
		}


//		if (offset == 0)
//		{
//			me[mb_x + mb_y*mb_width].mvc8x8[i8x8].mv[0] = index[0] % ( me_range*2 ) - me_range;
//			me[mb_x + mb_y*mb_width].mvc8x8[i8x8].mv[1] = index[0] / ( me_range*2 ) - me_range;
//			me[mb_x + mb_y*mb_width].mvc8x8[i8x8].cost = sadCache[i8x8][ index[0] ];
//		}
//		__syncthreads();


		sadSquare[offset] = MAX_INT;
		sindex[offset] = offset;
		if (offset < 25)
		{

			ox = index[0] % sub_range * 4 - 2 + offset%5;
			oy = index[0] / sub_range * 4 - 2 + offset/5;

			p_ref = p_fref + ox+ oy * stride_buf + ( i8x8_x * 8 + (i8x8_y * 8) * stride_buf );

			// set the sads values
			sadSquare[offset] = x264_cuda_pixel_sad_8x8(p_enc, stride_buf, p_ref, stride_buf);
		}

		// synchronize threads in this block
		__syncthreads();

		// for reductions, QQ_THREADS_PER_BLOCK must be a power of 2
		// because of the following code： find least SAD
		i = blockDim.x/4;
		while (i != 0) {
			if (offset < i)
			{
				if (sadSquare[ sindex[offset] ] > sadSquare[ sindex[offset + i] ])
				{
					sindex[offset] = sindex[offset + i];
				}
			}
			__syncthreads();
			i /= 2;
		}

		if (offset == 0)
		{
//			me[mb_x + mb_y*mb_width].mvc8x8[i8x8].mv[0] = index[0] % ( me_range*2 ) - me_range;
//			me[mb_x + mb_y*mb_width].mvc8x8[i8x8].mv[1] = index[0] / ( me_range*2 ) - me_range;
//			me[mb_x + mb_y*mb_width].mvc8x8[i8x8].cost = sadCache[i8x8][ index[0] ];
			me[mb_x + mb_y*mb_width].mvc8x8[i8x8].mv[0] = index[0] % sub_range * 4 - me_range + sindex[0]%5 - 2;
			me[mb_x + mb_y*mb_width].mvc8x8[i8x8].mv[1] = index[0] / sub_range * 4 - me_range + sindex[0]/5 - 2;
			me[mb_x + mb_y*mb_width].mvc8x8[i8x8].cost = sadSquare[ sindex[0] ];
		}
		__syncthreads();
	}

	me_merge_16x16(sadCache, sadMerge, index, me, me_range, p_fenc, p_fref, stride_buf);
	me_merge_16x8(sadCache, sadMerge, index, me, me_range, p_fenc, p_fref, stride_buf);
	me_merge_8x16(sadCache, sadMerge, index, me, me_range, p_fenc, p_fref, stride_buf);

}
