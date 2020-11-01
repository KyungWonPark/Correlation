package main

import (
	"fmt"

	"github.com/gonum/matrix/mat64"
	"github.com/kshedden/gonpy"
)

func main() {
	fmt.Println("Creating a matrix")
	arrm := make([]float64, 9)
	m := mat64.NewDense(3, 3, arrm)

	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			m.Set(i, j, float64(i*j))
		}
	}

	fmt.Println("Matrix: ")
	printMat(m)

	fmt.Println("Writing matrix as npy...")
	w, _ := gonpy.NewFileWriter("matrix.npy")
	w.Shape = []int{3, 3}
	w.Version = 2

	_ = w.WriteFloat64(arrm)

	fmt.Println("Reading npy matrix...")
	r, _ := gonpy.NewFileReader("matrix.npy")
	data, _ := r.GetFloat64()

	rows := r.Shape[0]
	cols := r.Shape[0]

	m2 := mat64.NewDense(rows, cols, data)
	fmt.Println("Read matrix.npy: ")
	printMat(m2)

	rM := m2.RawMatrix()
	fmt.Println("Printing raw matrix slice:")
	fmt.Println(rM.Data)

	return
}

func printMat(matrix *mat64.Dense) {
	rows, cols := matrix.Dims()

	for i := 0; i < rows; i++ {
		fmt.Printf("[")
		for j := 0; j < cols; j++ {
			fmt.Printf("%f, ", matrix.At(i, j))
		}
		fmt.Printf("]\n")
	}

	return
}
