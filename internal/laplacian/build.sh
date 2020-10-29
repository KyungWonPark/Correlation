#!/bin/bash

gcc -O3 -fPIC -DNDEBUG -DADD_ -Wall -fopenmp -DMAGMA_ILP64 -DHAVE_CUBLAS -DMIN_CUDA_ARCH=700 -I/usr/local/cuda/include -I./include -I./testing -c -o laplacian.o laplacian.c
ar -r liblaplacian.a laplacian.o