package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"strconv"

	"github.com/KyungWonPark/Correlation/internal/calc"
	"github.com/KyungWonPark/Correlation/internal/io"
	"github.com/KyungWonPark/shmtool/shm"
	"github.com/gonum/matrix/mat64"
	// "gonum.org/v1/gonum/blas/blas64"
	// blas_netlib "gonum.org/v1/netlib/blas/netlib"
)

func main() { // SUBJ TIMESTART TIMEEND anti-parallel GAMMA
	thr, _ := strconv.ParseFloat(os.Args[1], 64)
	gamma, _ := strconv.ParseFloat(os.Args[2], 64)
	// blas64.Use(blas_netlib.Implementation{})
	pl := calc.Init(4, false)

	fmt.Println("Reading C2...")

	// io.Mat64toCSV(RESULTDIR+"/c2-tilda.csv", avgedMat)
	c2 := io.NpytoMat64("bin/c2.npy")
	inputRows, inputCols := c2.Dims()

	var _inputRows uint64
	_inputRows = uint64(inputRows)
	var _inputCols uint64
	_inputCols = uint64(inputCols)

	matBufferShm, err := shm.Create(_inputRows * _inputCols * 8)
	if err != nil {
		log.Fatalf("Failed to create shared memory region: %s\n", err)
	}

	eigValShm, err := shm.Create(_inputRows * 1 * 8)
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

	arrEigVal := make([]float64, inputRows*1)
	eigVal := mat64.NewDense(inputRows, 1, arrEigVal)
	arrEigVec := make([]float64, inputRows*inputCols)
	eigVec := mat64.NewDense(inputRows, inputCols, arrEigVec)

	thredMat := mat64.NewDense(inputRows, inputCols, nil)

	fmt.Printf("Processing: thr - %.2f\n", thr)
	pl.Threshold(c2, thredMat, thr, gamma)

	pl.Laplacian(thredMat) // Now thredMat is a Laplacian matrix

	mat64tocArr(thredMat, pMatBuffer) // Copy thresholded matrix to MAGMA matrix buffer

	fmt.Printf("Diagonalizing...\n")
	// Call MAGMA

	cmd := exec.Command("/home/iksoochang2/kw-park/.root/usr/local/bin/magma", fmt.Sprintf("%d", inputRows), fmt.Sprintf("%d", matBufferShm.Id), fmt.Sprintf("%d", eigValShm.Id))
	err = cmd.Run()
	if err != nil {
		log.Fatalf("[main.go] MAGMA execution has failed: %s\n", err)
	}

	// Copy result from MAGMA to eigVal and eigVec
	cArrtomat64(eigVal, pEigVal)
	cArrtomat64(eigVec, pMatBuffer) // eigVec rows are eigen vectors; eigVec[0] <- first eigen vector

	fmt.Println("Writing Eigen value")
	io.Mat64toNpy("bin/eigVal.npy", eigVal)
	fmt.Println("Writing Eigen vector")
	io.Mat64toNpy("bin/eigVec.npy", eigVec)

	matBufferShm.Detach(pMatBuffer)
	matBufferShm.Destroy()

	eigValShm.Detach(pEigVal)
	eigValShm.Destroy()

	fmt.Println("Success.")
	return
}
