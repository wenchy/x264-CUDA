
#ifndef __X264_CUDA_MOTIONSEARCH_H__
#define __X264_CUDA_MOTIONSEARCH_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* number of pixels past the edge of the frame, for motion estimation/compensation */
#define PADH 32
#define PADV 32


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

#endif  // __X264_CUDA_MOTIONSEARCH_H__

