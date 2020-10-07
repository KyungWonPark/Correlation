package main

import (
	"unsafe"

	"github.com/gonum/matrix/mat64"
)

func mat64tocArr(matrix *mat64.Dense, pArr unsafe.Pointer) {
	rows, cols := matrix.Dims()

	stride := uintptr(unsafe.Sizeof(float64(0)))

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			index := uintptr(i*cols + j)
			addr := (*float64)(unsafe.Pointer(uintptr(pArr) + index*stride))

			*addr = matrix.At(i, j)
		}
	}

	return
}

func cArrtomat64(matrix *mat64.Dense, pArr unsafe.Pointer) {
	rows, cols := matrix.Dims()

	stride := uintptr(unsafe.Sizeof(float64(0)))

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			index := uintptr(i)
			addr := (*float64)(unsafe.Pointer(uintptr(pArr) + index*stride))

			matrix.Set(i, j, *addr)
		}
	}

	return
}
