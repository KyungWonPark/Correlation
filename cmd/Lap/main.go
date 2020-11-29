package main

import (
	"fmt"
	"log"
	"os/exec"

	"github.com/KyungWonPark/Correlation/internal/calc"
	"github.com/KyungWonPark/Correlation/internal/io"
	"github.com/ghetzel/shmtool/shm"
	"github.com/gonum/matrix/mat64"
	// "gonum.org/v1/gonum/blas/blas64"
	// blas_netlib "gonum.org/v1/netlib/blas/netlib"
)

func main() { // thrStart thrEnd thrItv gamma timeStart timeEnd postFix
	// blas64.Use(blas_netlib.Implementation{})
	fmt.Println("Loading C2")
	// io.Mat64toCSV(RESULTDIR+"/c2-tilda.csv", avgedMat)
	c2 := io.NpytoMat64("bin/c2.npy")

	matBufferShm, err := shm.Create(13362 * 13362 * 8)
	if err != nil {
		log.Fatalf("Failed to create shared memory region: %s\n", err)
	}

	eigValShm, err := shm.Create(13362 * 1 * 8)
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

	//arrThredMat := make([]float64, 13362*13362)
	//thredMat := mat64.NewDense(13362, 13362, arrThredMat)
	arrEigVal := make([]float64, 13362*1)
	eigVal := mat64.NewDense(13362, 1, arrEigVal)
	arrEigVec := make([]float64, 13362*13362)
	eigVec := mat64.NewDense(13362, 13362, arrEigVec)

	pl := calc.Init(4, false)
	pl.Laplacian(c2) // Now c2 is a Laplacian matrix

	mat64tocArr(c2, pMatBuffer) // Copy thresholded matrix to MAGMA matrix buffer

	fmt.Printf("Diagonalizing...")
	// Call MAGMA
	cmd := exec.Command("magma", "13362", fmt.Sprintf("%d", matBufferShm.Id), fmt.Sprintf("%d", eigValShm.Id))
	err = cmd.Run()
	if err != nil {
		log.Fatalf("MAGMA execution has failed: %s\n", err)
	}

	// Copy result from MAGMA to eigVal and eigVec
	cArrtomat64(eigVal, pEigVal)
	cArrtomat64(eigVec, pMatBuffer) // eigVec rows are eigen vectors; eigVec[0] <- first eigen vector

	fmt.Println("Writing Eigen value")
	io.Mat64toNpy("bin/eigVal.npy", eigVal)
	fmt.Println("Writing Eigen vector")
	// io.Mat64toCSV(RESULTDIR+"/eigen-vector-thr-"+fmt.Sprintf("%f", thr)+".csv", eigVec)
	io.Mat64toNpy("bin/eigVec.npy", eigVec)

	matBufferShm.Detach(pMatBuffer)
	matBufferShm.Destroy()

	eigValShm.Detach(pEigVal)
	eigValShm.Destroy()

	return
}

func checkLaplacian(mat *mat64.Dense) bool {
	inputRows, inputCols := mat.Dims()

	isSym := true
	// Check Symmetry
	for i := 0; i < inputRows; i++ {
		for j := 0; j < inputCols; j++ {
			val0 := mat.At(i, j)
			val1 := mat.At(j, i)
			if val0 != val1 {
				isSym = false
			}
		}
	}

	return isSym
}
