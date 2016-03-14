
#ifndef __X264_CUDA_COMMON_H__
#define __X264_CUDA_COMMON_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef signed char int8_t;
typedef unsigned char   uint8_t;
typedef short  int16_t;
typedef unsigned short  uint16_t;
typedef int  int32_t;
typedef unsigned   uint32_t;

typedef uint8_t  pixel;
typedef long int intptr_t;

/* number of pixels past the edge of the frame, for motion estimation/compensation */
#define PADH 32
#define PADV 32

#ifndef __X264_CUDA_T__
#define __X264_CUDA_T__
typedef struct x264_cuda_t
{
	int i_me_range;
	int i_mb_x;
	int i_mb_y;
	int bw;
	int bh;
	int mv_min_x;
	int mv_min_y;
	int mv_max_x;
	int mv_max_y;
	pixel *fref_buf;
	pixel *mb_enc;
	int stride_ref;
	uint16_t *p_cost_mvx;
	uint16_t *p_cost_mvy;
} x264_cuda_t;
#endif  // __X264_CUDA_T__

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

#endif  // __X264_CUDA_COMMON_H__

