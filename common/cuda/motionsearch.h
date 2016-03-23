
#ifndef __X264_CUDA_MOTIONSEARCH_H__
#define __X264_CUDA_MOTIONSEARCH_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
/* number of pixels past the edge of the frame, for motion estimation/compensation */
#define PADH 32
#define PADV 32

#define MAX_INT 65535
// 1024 = (16*2 * 16*2), also 2^10, also Max threads per block:  1024
#define THREADS_PER_BLOCK 1024
// Quarter of THREADS_PER_BLOCK: 256 = 1024/4
#define Q_THREADS_PER_BLOCK 256
// Quarter Quarter of THREADS_PER_BLOCK: 64 = 1024/4/4
#define QQ_THREADS_PER_BLOCK 64

enum
{
    PIXEL_16x16 = 0,
    PIXEL_16x8  = 1,
    PIXEL_8x16  = 2,
    PIXEL_8x8   = 3,
    PIXEL_8x4   = 4,
    PIXEL_4x8   = 5,
    PIXEL_4x4   = 6
};

#define HANDLE_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"CUDAassert: %s in %s at line %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
//#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL( a ) {if (a == NULL) { \
                            printf( "Host memory failed in %s at line %d\n", \
                                    __FILE__, __LINE__ ); \
                            exit( EXIT_FAILURE );}}

#endif  // __X264_CUDA_MOTIONSEARCH_H__

