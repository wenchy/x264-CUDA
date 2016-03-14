SRCPATH=.
prefix=/usr/local
exec_prefix=${prefix}
bindir=${exec_prefix}/bin
libdir=${exec_prefix}/lib
cudalibdir=/usr/local/cuda-7.5/lib64
includedir=${prefix}/include
SYS_ARCH=X86_64
SYS=LINUX
CC=gcc
#CFLAGS= -Wno-maybe-uninitialized -Wshadow -O3 -ffast-math -m64  -Wall -I. -I$(SRCPATH) -std=gnu99 -mpreferred-stack-boundary=5 -fomit-frame-pointer -fno-tree-vectorize
CFLAGS= -g -DHAVE_CUDA -Wno-maybe-uninitialized -Wshadow -O0 -ffast-math -m64  -Wall -I. -I$(SRCPATH) -std=gnu99 -mpreferred-stack-boundary=5 -fomit-frame-pointer -fno-tree-vectorize
COMPILER=GNU
COMPILER_STYLE=GNU
DEPMM=-MM -g0
DEPMT=-MT
LD=gcc -o 
LDFLAGS=-m64  -lm -lpthread -ldl -pg -lcudart -L${cudalibdir} -Wl,-rpath,${cudalibdir} -Wl,--as-needed
LIBX264=libx264.a
AR=ar rc 
RANLIB=ranlib
STRIP=strip
INSTALL=install
AS=yasm
ASFLAGS= -I. -I$(SRCPATH) -DARCH_X86_64=1 -I$(SRCPATH)/common/x86/ -f elf64 -Worphan-labels -DSTACK_ALIGNMENT=32 -DHIGH_BIT_DEPTH=0 -DBIT_DEPTH=8
RC=
RCFLAGS=
EXE=
HAVE_GETOPT_LONG=1
DEVNULL=/dev/null
PROF_GEN_CC=-fprofile-generate
PROF_GEN_LD=-fprofile-generate
PROF_USE_CC=-fprofile-use
PROF_USE_LD=-fprofile-use
HAVE_OPENCL=yes
# added by Wenchy 2016-03-12: compile .cu file
HAVE_CUDA=yes
default: cli
install: install-cli
LDFLAGSCLI = -ldl -lgpac_static -lz 
CLI_LIBX264 = $(LIBX264)
