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






void save_frame(pixel *plane, int stride, int i_frame)
{
    FILE *pFile;
    char szFilename[32];
    int  x, y;
    pixel zero = 0;

    int width =  352;
    int height = 288;


    // Open file
    sprintf(szFilename, "dev_frame_%d.ppm", i_frame);

    pFile = fopen(szFilename, "wb");

    if(pFile == NULL) {
        return;
    }
    // Write header
	fprintf(pFile, "P6\n%d %d\n255\n", width+PADH*2, height+PADV*2);

	// Write pixel data
	for(y = 0; y < height+PADV*2; y++) {
		for(x = 0; x < width+PADH*2; x++) {
		   fwrite(&(plane[x+stride*y]), 1, 1, pFile);
		   fwrite(&zero, 1, 1, pFile);
		   fwrite(&zero, 1, 1, pFile);
		}
	}
    // Close file
    fclose(pFile);
}
extern "C" void cuda_me_init( x264_cuda_t *c) {
	int buf_width = 16 * c->i_mb_width + PADH*2;
	int buf_height = 16 * c->i_mb_height + PADV*2;
	HANDLE_ERROR( cudaMalloc( (void**)&(c->dev_fenc_buf), buf_width * buf_height * sizeof(pixel) ) );
	HANDLE_ERROR( cudaMalloc( (void**)&(c->dev_fref_buf), buf_width * buf_height * sizeof(pixel) ) );
//	c->dev_fenc_plane = c->dev_fenc_buf + c->stride_buf * PADV + PADH;
//	c->dev_fref_plane = c->dev_fref_buf + c->stride_buf * PADV + PADH;
//	printf("%p	%p\n", c->dev_fenc_buf, c->dev_fenc_plane);

	// mb mvc
	// CUDA Unified Memory
//	HANDLE_ERROR( cudaMallocManaged( (void**)&(c->p_mvc16x16), (c->i_mb_width * c->i_mb_height) * sizeof(x264_cuda_mvc_t) ) );
	HANDLE_ERROR( cudaMallocManaged( (void**)&(c->me), (c->i_mb_width * c->i_mb_height) * sizeof(x264_cuda_me_t) ) );

	printf("*****cuda_me_init***** %lu x 41 = %lu\n", sizeof(x264_cuda_mvc_t), sizeof(x264_cuda_me_t));
}

extern "C" void cuda_me_end( x264_cuda_t *c) {
	HANDLE_ERROR( cudaFree( c->dev_fenc_buf ) );
	HANDLE_ERROR( cudaFree( c->dev_fref_buf ) );
//	HANDLE_ERROR( cudaFree( c->p_mvc16x16 ) );
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

//	if((*p_bmx) !=0 && (*p_bmy) != 0)
//		printf("i_pixel: %d mx: %d	my: %d\n", c->i_pixel, mvc->mx, mvc->my);


	return;
}

__global__ void me( int i_pixel, pixel *dev_fenc_buf, pixel *dev_fref_buf, x264_cuda_me_t *me, int me_range, int stride_buf) {
	 __shared__ int sadCache[4][THREADS_PER_BLOCK]; // 16k
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

		// for reductions, THREADS_PER_BLOCK must be a power of 2
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

		if (offset == 0)
		{
			me[mb_x + mb_y*mb_width].mvc8x8[i8x8].mv[0] = index[0] % ( me_range*2 ) - me_range;
			me[mb_x + mb_y*mb_width].mvc8x8[i8x8].mv[1] = index[0] / ( me_range*2 ) - me_range;
			me[mb_x + mb_y*mb_width].mvc8x8[i8x8].cost = sadCache[i8x8][ index[0] ];
		}
		__syncthreads();
	}

	/* 1. 8x8 SAD merging into 16x16, 16x8, 8x16
	 * 2. find least SAD and corresponding MV
	 */
	for(int i16x16 = 0; i16x16 < 1; i16x16++)
	{
//		int x = offset % ( me_range*2 );
//		int y = offset / ( me_range*2 );

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

		if (offset == 0)
		{
			me[mb_x + mb_y*mb_width].mvc16x16.mv[0] = index[0] % ( me_range*2 ) - me_range;
			me[mb_x + mb_y*mb_width].mvc16x16.mv[1] = index[0] / ( me_range*2 ) - me_range;
			me[mb_x + mb_y*mb_width].mvc16x16.cost = sadMerge[ index[0] ];
		}
		__syncthreads();
	}

}

__global__ void me0( int i_pixel, pixel *dev_fenc_buf, pixel *dev_fref_buf, x264_cuda_mvc_t *p_mvc16x16, int me_range, int stride_buf) {
	 __shared__ int sadCache[THREADS_PER_BLOCK]; // 4k
	 __shared__ int index[THREADS_PER_BLOCK]; // 4k

	// map from blockIdx to pixel position
//	int x = blockIdx.x;
//	int y = blockIdx.y;
//	int offset = x + y * gridDim.x;

	int offset = threadIdx.x;
	int x = threadIdx.x % ( me_range*2 );
	int y = threadIdx.x / ( me_range*2 );

	int mb_x = blockIdx.x;
	int mb_y = blockIdx.y;
	int mb_width = gridDim.x;

	pixel *p_fenc_plane = dev_fenc_buf + stride_buf * PADV + PADH;
	pixel *p_fref_plane = dev_fref_buf + stride_buf * PADV + PADH;

	pixel *p_fenc = p_fenc_plane + ( 16 * mb_x) +( 16 * mb_y)* stride_buf;
	pixel *p_fref = p_fref_plane + ( 16 * mb_x + x - me_range) +( 16 * mb_y + y - me_range)* stride_buf;
	//p_fref += bmx +bmy * stride_buf;

	int temp = MAX_INT;
	switch(i_pixel)
	{
		case PIXEL_16x16:
			temp = x264_cuda_pixel_sad_16x16(p_fenc, stride_buf, p_fref, stride_buf);
			break;
		case PIXEL_16x8:
			temp = x264_cuda_pixel_sad_16x8(p_fenc, stride_buf, p_fref, stride_buf);
			break;
		case PIXEL_8x16:
			temp = x264_cuda_pixel_sad_8x16(p_fenc, stride_buf, p_fref, stride_buf);
			break;
		case PIXEL_8x8:
			temp = x264_cuda_pixel_sad_8x8(p_fenc, stride_buf, p_fref, stride_buf);
			break;
		case PIXEL_8x4:
			temp = x264_cuda_pixel_sad_8x4(p_fenc, stride_buf, p_fref, stride_buf);
			break;
		case PIXEL_4x8:
			temp = x264_cuda_pixel_sad_4x8(p_fenc, stride_buf, p_fref, stride_buf);
			break;
		case PIXEL_4x4:
			temp = x264_cuda_pixel_sad_4x4(p_fenc, stride_buf, p_fref, stride_buf);
			break;
		default:
			break;

	}
	//temp = cudafpelcmp[i_pixel](p_fenc, stride_buf, p_fref, stride_buf);
	//temp = x264_cuda_pixel_sad_16x16(p_fenc, stride_buf, p_fref, stride_buf);

	// set the sads values
	sadCache[offset] = temp;
	index[offset] = offset;

	// synchronize threads in this block
	__syncthreads();

	// for reductions, THREADS_PER_BLOCK must be a power of 2
	// because of the following code： find least SAD
	int i = blockDim.x/2;
	while (i != 0) {
		if (offset < i)
		{
			if (sadCache[ index[offset] ] > sadCache[ index[offset + i] ])
			{
				index[offset] = index[offset + i];
			}
		}
		__syncthreads();
		i /= 2;
	}

	if (offset == 0)
	{
		p_mvc16x16[mb_x + mb_y*mb_width].mv[0] = index[0] % ( me_range*2 ) - me_range;
		p_mvc16x16[mb_x + mb_y*mb_width].mv[1] = index[0] / ( me_range*2 ) - me_range;
		p_mvc16x16[mb_x + mb_y*mb_width].cost = sadCache[ index[0] ];

//		mvc->cost = sadCache[ index[0] ];
//		mvc->mx = index[0] % ( me_range*2 ) - me_range;
//		mvc->my = index[0] / ( me_range*2 ) - me_range;
	}
}

// with shared memory cache of mb_enc and mb_ref
__global__ void me_shared_cache( int i_pixel, pixel *dev_fenc_buf, pixel *dev_fref_buf, x264_cuda_mvc_t *p_mvc16x16, int me_range, int stride_buf) {
	 __shared__ int sadCache[THREADS_PER_BLOCK]; // 4k
	 __shared__ int index[THREADS_PER_BLOCK];	// 4k

	 __shared__ pixel mb_enc[16*16];
	 __shared__ pixel mb_ref[16*3 * 16*3]; //3k
	 int stride_enc = 16;
	 int stride_ref = 16*3;

	int offset = threadIdx.x;


	int mb_x = blockIdx.x;
	int mb_y = blockIdx.y;
	int mb_width = gridDim.x;

	pixel *p_fenc_plane = dev_fenc_buf + stride_buf * PADV + PADH;
	pixel *p_fref_plane = dev_fref_buf + stride_buf * PADV + PADH;

	int x = threadIdx.x % ( me_range*2 );
	int y = threadIdx.x / ( me_range*2 );
	pixel *p_mb_ref = &(mb_ref[x + y*stride_ref]);

	pixel *p_fenc = p_fenc_plane + ( 16 * mb_x) +( 16 * mb_y)* stride_buf;
	pixel *p_fref = p_fref_plane + ( 16 * mb_x - me_range) +( 16 * mb_y - me_range)* stride_buf;

	if(offset < 16*16)
	{
		int x = offset % 16;
		int y = offset / 16;
		mb_enc[x + y*stride_enc] = p_fenc[x + y*stride_buf];
	}

	x = offset % (16*3);
	y = offset / (16*3);
	mb_ref[x + y*stride_ref] = p_fref[x + y*stride_buf];
	x = (offset + THREADS_PER_BLOCK) % (16*3);
	y = (offset + THREADS_PER_BLOCK) / (16*3);
	mb_ref[x + y*stride_ref] = p_fref[x + y*stride_buf];

	if(offset + 2*THREADS_PER_BLOCK < 16*3 * 16*3)
	{
		x = (offset + 2*THREADS_PER_BLOCK) % (16*3);
		y = (offset + 2*THREADS_PER_BLOCK) / (16*3);
		mb_ref[x + y*stride_ref] = p_fref[x + y*stride_buf];
	}

	// synchronize threads in this block: loading 'shared' memory mb_enc and mb_ref from 'global' memory
	__syncthreads();

	int temp = MAX_INT;
	switch(i_pixel)
	{
		case PIXEL_16x16:
			temp = x264_cuda_pixel_sad_16x16(mb_enc, stride_enc, p_mb_ref, stride_ref);
			break;
		case PIXEL_16x8:
			temp = x264_cuda_pixel_sad_16x8(mb_enc, stride_enc, p_mb_ref, stride_ref);
			break;
		case PIXEL_8x16:
			temp = x264_cuda_pixel_sad_8x16(mb_enc, stride_enc, p_mb_ref, stride_ref);
			break;
		case PIXEL_8x8:
			temp = x264_cuda_pixel_sad_8x8(mb_enc, stride_enc, p_mb_ref, stride_ref);
			break;
		case PIXEL_8x4:
			temp = x264_cuda_pixel_sad_8x4(mb_enc, stride_enc, p_mb_ref, stride_ref);
			break;
		case PIXEL_4x8:
			temp = x264_cuda_pixel_sad_4x8(mb_enc, stride_enc, p_mb_ref, stride_ref);
			break;
		case PIXEL_4x4:
			temp = x264_cuda_pixel_sad_4x4(mb_enc, stride_enc, p_mb_ref, stride_ref);
			break;
		default:
			break;

	}
	//temp = cudafpelcmp[i_pixel](p_fenc, stride_buf, p_fref, stride_buf);
	//temp = x264_cuda_pixel_sad_16x16(p_fenc, stride_buf, p_fref, stride_buf);

	// set the sads values
	sadCache[offset] = temp;
	index[offset] = offset;

	// synchronize threads in this block
	__syncthreads();

	// for reductions, THREADS_PER_BLOCK must be a power of 2
	// because of the following code： find least SAD
	int i = blockDim.x/2;
	while (i != 0) {
		if (offset < i)
		{
			if (sadCache[ index[offset] ] > sadCache[ index[offset + i] ])
			{
				index[offset] = index[offset + i];
			}
		}
		__syncthreads();
		i /= 2;
	}

	if (offset == 0)
	{
		p_mvc16x16[mb_x + mb_y*mb_width].mv[0] = index[0] % ( me_range*2 ) - me_range;
		p_mvc16x16[mb_x + mb_y*mb_width].mv[1] = index[0] / ( me_range*2 ) - me_range;
		p_mvc16x16[mb_x + mb_y*mb_width].cost = sadCache[ index[0] ];

//		mvc->cost = sadCache[ index[0] ];
//		mvc->mx = index[0] % ( me_range*2 ) - me_range;
//		mvc->my = index[0] / ( me_range*2 ) - me_range;
	}
}


// two encoding MB process in one thread block
__global__ void me_two_mb( int i_pixel, pixel *dev_fenc_buf, pixel *dev_fref_buf, x264_cuda_mvc_t *p_mvc16x16, int me_range, int stride_buf) {
	 __shared__ int sadCache[THREADS_PER_BLOCK]; // 4k
	 __shared__ int index[THREADS_PER_BLOCK];	// 4k

	 __shared__ pixel mb_enc[16*2 *16];
	 __shared__ pixel mb_ref[16*4 * 16*3]; //3k
	 int stride_enc = 16*2;
	 int stride_ref = 16*4;

	int offset = threadIdx.x;


	int mb_x = blockIdx.x;
	int mb_y = blockIdx.y;
	int mb_width = gridDim.x;

	pixel *p_fenc_plane = dev_fenc_buf + stride_buf * PADV + PADH;
	pixel *p_fref_plane = dev_fref_buf + stride_buf * PADV + PADH;

	int ox = threadIdx.x % ( me_range*2 );
	int oy = threadIdx.x / ( me_range*2 );


	pixel *p_fenc = p_fenc_plane + ( 16 * mb_x) +( 16 * mb_y)* stride_buf;
	pixel *p_fref = p_fref_plane + ( 16 * mb_x - me_range) +( 16 * mb_y - me_range)* stride_buf;

	if(offset < 16*16 * 2)
	{
		int x = offset % stride_enc;
		int y = offset / stride_enc;
		mb_enc[x + y*stride_enc] = p_fenc[x + y*stride_buf];
	}

	int x = offset % stride_ref;
	int y = offset / stride_ref;
	mb_ref[x + y*stride_ref] = p_fref[x + y*stride_buf];
	x = (offset + THREADS_PER_BLOCK) % stride_ref;
	y = (offset + THREADS_PER_BLOCK) / stride_ref;
	mb_ref[x + y*stride_ref] = p_fref[x + y*stride_buf];

	x = (offset + 2*THREADS_PER_BLOCK) % stride_ref;
	y = (offset + 2*THREADS_PER_BLOCK) / stride_ref;
	mb_ref[x + y*stride_ref] = p_fref[x + y*stride_buf];


	// synchronize threads in this block: loading 'shared' memory mb_enc and mb_ref from 'global' memory
	__syncthreads();

	for(int i_mb = 0; i_mb < 2; i_mb++)
	{
		int temp = MAX_INT;
		pixel *p_mb_enc = &(mb_enc[i_mb*16]);
		pixel *p_mb_ref = &(mb_ref[i_mb*16 + ox + oy*stride_ref]);
		switch(i_pixel)
		{
			case PIXEL_16x16:
				temp = x264_cuda_pixel_sad_16x16(p_mb_enc, stride_enc, p_mb_ref, stride_ref);
				break;
			case PIXEL_16x8:
				temp = x264_cuda_pixel_sad_16x8(p_mb_enc, stride_enc, p_mb_ref, stride_ref);
				break;
			case PIXEL_8x16:
				temp = x264_cuda_pixel_sad_8x16(p_mb_enc, stride_enc, p_mb_ref, stride_ref);
				break;
			case PIXEL_8x8:
				temp = x264_cuda_pixel_sad_8x8(p_mb_enc, stride_enc, p_mb_ref, stride_ref);
				break;
			case PIXEL_8x4:
				temp = x264_cuda_pixel_sad_8x4(p_mb_enc, stride_enc, p_mb_ref, stride_ref);
				break;
			case PIXEL_4x8:
				temp = x264_cuda_pixel_sad_4x8(p_mb_enc, stride_enc, p_mb_ref, stride_ref);
				break;
			case PIXEL_4x4:
				temp = x264_cuda_pixel_sad_4x4(p_mb_enc, stride_enc, p_mb_ref, stride_ref);
				break;
			default:
				break;

		}

		// set the sads values
		sadCache[offset] = temp;
		index[offset] = offset;

		// synchronize threads in this block
		__syncthreads();

		// for reductions, THREADS_PER_BLOCK must be a power of 2
		// because of the following code： find least SAD
		int i = blockDim.x/2;
		while (i != 0) {
			if (offset < i)
			{
				if (sadCache[ index[offset] ] > sadCache[ index[offset + i] ])
				{
					index[offset] = index[offset + i];
				}
			}
			__syncthreads();
			i /= 2;
		}

		if (offset == 0)
		{
			p_mvc16x16[i_mb + mb_x + mb_y*mb_width].mv[0] = index[0] % ( me_range*2 ) - me_range;
			p_mvc16x16[i_mb + mb_x + mb_y*mb_width].mv[1] = index[0] / ( me_range*2 ) - me_range;
			p_mvc16x16[i_mb + mb_x + mb_y*mb_width].cost = sadCache[ index[0] ];
		}
		__syncthreads();
	}
}
