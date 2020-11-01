package io

import (
	"log"

	"github.com/gonum/matrix/mat64"
	"github.com/kshedden/gonpy"
)

// Mat64toNpy writes mat64 matrix to Python numpy npy binary file
func Mat64toNpy(path string, matrix *mat64.Dense) {
	rows, cols := matrix.Dims()
	rawMat := matrix.RawMatrix()

	w, err := gonpy.NewFileWriter(path)
	if err != nil {
		log.Fatalf("[Mat64toNpy] Failed to open file: %v\n", err)
	}
	w.Shape = []int{rows, cols}
	w.Version = 2
	err = w.WriteFloat64(rawMat.Data)
	if err != nil {
		log.Fatalf("[Mat64toNpy] Failed to write file: %v\n", err)
	}

	return
}

// NpytoMat64 reads Python numpy npy binary file as mat64 matrix
func NpytoMat64(path string) *mat64.Dense {
	r, err := gonpy.NewFileReader(path)
	if err != nil {
		log.Fatalf("[NpytoMat64] Failed to open file: %v\n", err)
	}

	rows := r.Shape[0]
	cols := r.Shape[1]
	data, err := r.GetFloat64()
	if err != nil {
		log.Fatalf("[NpytoMat64] Failed to read file: %v\n", err)
	}

	matrix := mat64.NewDense(rows, cols, data)
	return matrix
}
