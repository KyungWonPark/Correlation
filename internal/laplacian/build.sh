#!/bin/bash

g++ -O3 -fPIC -DNDEBUG -DADD_ -Wall -fopenmp -DMAGMA_ILP64 -std=c++11 -DHAVE_CUBLAS -DMIN_CUDA_ARCH=700 -I/usr/local/cuda/include -I./files/include -I./files/testing -c -o files/laplacian.o files/laplacian.cpp
ar -r files/liblaplacian.a files/laplacian.o