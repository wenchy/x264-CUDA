# x264-CUDA

Parallel Optimization of Motion Estimation (ME) module based on CUDA
> source code base: `x264-snapshot-20151223-2245`

## cmd

`./x264 -o [filename].mkv --frames 50  --no-asm --no-cabac --trellis 0 --nr 0 --slices 1 --threads 1 --bframes 0 --ref 1 --no-mixed-refs --no-chroma-me --merange 16 --me cuda_esa --subme 1 --weightp 0  [filepathname].y4m`

## configuration

1. config.h:    `#define HAVE_CUDA 1`
2. config.mak:  `HAVE_CUDA=yes`
