package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"

	"github.com/KyungWonPark/Correlation/internal/calc"
	"github.com/KyungWonPark/Correlation/internal/io"
	"github.com/ghetzel/shmtool/shm"
	"github.com/gonum/matrix/mat64"
	// "gonum.org/v1/gonum/blas/blas64"
	// blas_netlib "gonum.org/v1/netlib/blas/netlib"
)

func main() { // thrStart thrEnd thrItv isDebugMode
	// blas64.Use(blas_netlib.Implementation{})
	RESULTDIR := os.Getenv("RESULT")

	numQueueSize := 8
	ringBuffer := make([]*mat64.Dense, numQueueSize)
	for i := 0; i < numQueueSize; i++ {
		ringBuffer[i] = mat64.NewDense(13362, 600, nil)
	}

	pl := calc.Init(numQueueSize, false)

	fmt.Println("Loading C2-tilda")
	c2 := mat64.NewDense(13362, 13362, nil)
	io.CSVtoMat64(RESULTDIR+"/c2-tilda.csv", c2)

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

	arrThredMat := make([]float64, 13362*13362)
	thredMat := mat64.NewDense(13362, 13362, arrThredMat)
	arrEigVal := make([]float64, 13362*1)
	eigVal := mat64.NewDense(13362, 1, arrEigVal)
	arrEigVec := make([]float64, 13362*13362)
	eigVec := mat64.NewDense(13362, 13362, arrEigVec)

	fmt.Println("Thresholding with thr: 0.87 / gamma: 0.001")
	pl.Threshold(c2, thredMat, 0.87, 0.001)
	fmt.Println("Constructing a Laplacian matrix")
	pl.Laplacian(thredMat) // Now thredMat is a Laplacian matrix
	fmt.Println("Checking constructed Laplacian matrix")
	isLaplacian := calc.CheckLaplacian(thredMat, 0.00000001)
	if !isLaplacian {
		log.Fatalf("Failed to pass laplacian check!\n")
	}
	fmt.Println("Copying constructed Laplacian matrix to MAGMA memory")
	mat64tocArr(thredMat, pMatBuffer) // Copy thresholded matrix to MAGMA matrix buffer

	fmt.Println("Diagonalizing...")
	cmd := exec.Command("files/magma", "13362", fmt.Sprintf("%d", matBufferShm.Id), fmt.Sprintf("%d", eigValShm.Id))
	txt, err := cmd.Output()
	if err != nil {
		log.Fatalf("MAGMA execution has failed: %s\n", txt)
	}

	// Copy result from MAGMA to eigVal and eigVec
	fmt.Println("Copying results back from MAGMA memory")
	cArrtomat64(eigVal, pEigVal)
	cArrtomat64(eigVec, pMatBuffer) // eigVec rows are eigen vectors; eigVec[0] <- first eigen vector

	fmt.Println("Checking eigen-decomposition quality")
	isEigenOK := calc.CheckEigenQuality(thredMat, eigVal, eigVec, 0.00000001)
	if !isEigenOK {
		log.Fatalf("Failed to pass eigen decomposition quality check!")
	}

	fmt.Println("Writing Eigen values")
	io.Mat64toCSV(RESULTDIR+"/eigen-value-thr-0.87-gamma-0.001.csv", eigVal)
	fmt.Println("Writing Eigen vectors")
	io.Mat64toCSV(RESULTDIR+"/eigen-vector-thr-0.87-gamma-0.001.csv", eigVec)

	matBufferShm.Detach(pMatBuffer)
	matBufferShm.Destroy()

	eigValShm.Detach(pEigVal)
	eigValShm.Destroy()

	return
}
