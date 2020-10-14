package main

import (
	"fmt"
	"log"
	"os/exec"

	"github.com/ghetzel/shmtool/shm"
	"github.com/gonum/matrix/mat64"
)

func main() {
	mat := []float64{
		3, 1, 1,
		1, 2, 2,
		1, 2, 2,
	}

	problemMat := mat64.NewDense(3, 3, mat)

	matBufferShm, err := shm.Create(3 * 3 * 8)
	if err != nil {
		log.Fatalf("Failed to create shared memory region: %s\n", err)
	}

	eigValShm, err := shm.Create(3 * 1 * 8)
	if err != nil {
		log.Fatalf("Failed to create shared memory region: %s\n", err)
	}

	pMatBuffer, err := matBufferShm.Attach()
	if err != nil {
		log.Fatalf("Failed to attach shared memory region: %s\n", err)
	}

	pEigVal, err := eigValShm.Attach()
	if err != nil {
		log.Fatalf("Failed to attach shared memory region: %s\n", err)
	}

	eigVal := mat64.NewDense(3, 1, nil)
	eigVec := mat64.NewDense(3, 3, nil)

	mat64tocArr(problemMat, pMatBuffer)

	// Call MAGMA
	cmd := exec.Command("files/magma", "3", fmt.Sprintf("%d", matBufferShm.Id), fmt.Sprintf("%d", eigValShm.Id))
	err = cmd.Run()
	if err != nil {
		log.Fatalf("MAGMA execution has failed: %s\n", err)
	}

	cArrtomat64(eigVal, pEigVal)
	cArrtomat64(eigVec, pMatBuffer)

	// Calculation is finished.
	matPrint("Original", problemMat)
	matPrint("EigenValue", eigVal)
	matPrint("U^T", eigVec)

	eigValMat := mat64.NewDense(3, 3, nil)
	for i := 0; i < 3; i++ {
		eigValMat.Set(i, i, eigVal.At(i, 0))
	}

	// U^T * A
	result0 := mat64.NewDense(3, 3, nil)
	result0.Mul(eigVec, problemMat)
	matPrint("U^T * A", result0)

	// S * U^T
	result1 := mat64.NewDense(3, 3, nil)
	result1.Mul(eigValMat, eigVec)
	matPrint("S * U^T", result1)

	isSame := mat64.EqualApprox(result0, result1, 0.000001)
	if !isSame {
		fmt.Printf("Eigenproblem failed!!\n")
		diff := mat64.NewDense(3, 3, nil)
		diff.Sub(result0, result1)
		fmt.Printf("Diff: %g\n", mat64.Max(diff))
	}

	matBufferShm.Detach(pMatBuffer)
	matBufferShm.Destroy()

	eigValShm.Detach(pEigVal)
	eigValShm.Destroy()

	return
}

func matPrint(name string, mat *mat64.Dense) {
	fmt.Printf("/-------- %s Matrix --------/\n", name)
	fmt.Println("")

	rows, cols := mat.Dims()

	for i := 0; i < rows; i++ {
		fmt.Printf("[ ")
		for j := 0; j < cols; j++ {
			fmt.Printf("%f ", mat.At(i, j))
		}
		fmt.Printf("]\n")
	}

	fmt.Println("")
	fmt.Printf("/-----------------")
	for i := 0; i < len(name); i++ {
		fmt.Printf("-")
	}
	fmt.Printf("--------/\n")

	return
}
