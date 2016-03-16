#include "x264-cuda.h"
#include "motionsearch.h"

__global__ void me( int i_pixel, pixel *dev_fenc_buf, pixel *dev_fref_buf, x264_cuda_mvc_t *p_mvc16x16, int me_range, int stride_buf);
__global__ void cmp(int *dev_sads, x264_cuda_mvc_t *dev_mvc, int me_range);

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
	// CUDA Unified Memory
//	HANDLE_ERROR( cudaMallocManaged( (void**)&(c->cudafpelcmp), 7 * sizeof( x264_cuda_pixel_cmp_t) ) );
//	c->cudafpelcmp[PIXEL_16x16] = &x264_cuda_pixel_sad_16x16;
//	c->cudafpelcmp[PIXEL_16x8]	= &x264_cuda_pixel_sad_16x8;
//	c->cudafpelcmp[PIXEL_8x16]  = &x264_cuda_pixel_sad_8x16;
//	c->cudafpelcmp[PIXEL_8x8]   = &x264_cuda_pixel_sad_8x8;
//	c->cudafpelcmp[PIXEL_8x4]   = &x264_cuda_pixel_sad_8x4;
//	c->cudafpelcmp[PIXEL_4x8]   = &x264_cuda_pixel_sad_4x8;
//	c->cudafpelcmp[PIXEL_4x4]   = &x264_cuda_pixel_sad_4x4;

	// mb mvc
	// CUDA Unified Memory
	HANDLE_ERROR( cudaMallocManaged( (void**)&(c->p_mvc16x16), (c->i_mb_width * c->i_mb_height) * sizeof(x264_cuda_mvc_t) ) );
	printf("*****cuda_me_init*****\n");
}

extern "C" void cuda_me_end( x264_cuda_t *c) {
	HANDLE_ERROR( cudaFree( c->dev_fenc_buf ) );
	HANDLE_ERROR( cudaFree( c->dev_fref_buf ) );
	HANDLE_ERROR( cudaFree( c->p_mvc16x16 ) );
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
	//printf("me_range: %d\n", me_range);
	int mb_width = c->i_mb_width;
	int mb_height = c->i_mb_height;
//	int mb_x = c->i_mb_x;
//	int mb_y = c->i_mb_y;

	int stride_buf = c->stride_buf;


	dim3    blocks(mb_width, mb_height);
	//	dim3    threads(me_range*2, me_range*2);
	dim3 grid_sad(me_range*2 * me_range*2);

	me<<<blocks, THREADS_PER_BLOCK>>>( c->i_pixel, c->dev_fenc_buf, c->dev_fref_buf, c->p_mvc16x16, me_range, stride_buf);
	HANDLE_ERROR( cudaPeekAtLastError() );
	HANDLE_ERROR( cudaDeviceSynchronize() );

//	(*p_bcost)= mvc->cost;
//	(*p_bmx)= mvc->mx;
//	(*p_bmy)=mvc->my;



//	if((*p_bmx) !=0 && (*p_bmy) != 0)
//		printf("i_pixel: %d mx: %d	my: %d\n", c->i_pixel, mvc->mx, mvc->my);


	return;
}

__global__ void me( int i_pixel, pixel *dev_fenc_buf, pixel *dev_fref_buf, x264_cuda_mvc_t *p_mvc16x16, int me_range, int stride_buf) {
	 __shared__ int sadCache[THREADS_PER_BLOCK];
	 __shared__ int index[THREADS_PER_BLOCK];

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
		p_mvc16x16[mb_x + mb_y*mb_width].mx = index[0] % ( me_range*2 ) - me_range;
		p_mvc16x16[mb_x + mb_y*mb_width].my = index[0] / ( me_range*2 ) - me_range;
		p_mvc16x16[mb_x + mb_y*mb_width].cost = sadCache[ index[0] ];

//		mvc->cost = sadCache[ index[0] ];
//		mvc->mx = index[0] % ( me_range*2 ) - me_range;
//		mvc->my = index[0] / ( me_range*2 ) - me_range;
	}
}

__global__ void cmp(int *dev_sads, x264_cuda_mvc_t *mvc, int me_range){
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

