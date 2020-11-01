package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"strconv"

	"github.com/KyungWonPark/Correlation/internal/calc"
	"github.com/KyungWonPark/Correlation/internal/io"
	"github.com/ghetzel/shmtool/shm"
	"github.com/gonum/matrix/mat64"
	// "gonum.org/v1/gonum/blas/blas64"
	// blas_netlib "gonum.org/v1/netlib/blas/netlib"
)

func main() { // thrStart thrEnd thrItv
	// blas64.Use(blas_netlib.Implementation{})
	thrStart, _ := strconv.ParseFloat(os.Args[1], 64)
	thrEnd, _ := strconv.ParseFloat(os.Args[2], 64)
	thrItv, _ := strconv.ParseFloat(os.Args[3], 64)
	var gamma float64
	gamma = 0.001

	var isDebugMode bool
	isDebugMode = false

	DATADIR := os.Getenv("DATA")
	RESULTDIR := os.Getenv("RESULT")
	RESULTDIR = RESULTDIR + "/gamma-0.001"

	numQueueSize := 8
	ringBuffer := make([]*mat64.Dense, numQueueSize)
	for i := 0; i < numQueueSize; i++ {
		ringBuffer[i] = mat64.NewDense(13362, 600, nil)
	}

	pl := calc.Init(numQueueSize, isDebugMode)

	go func() {
		for _, file := range fileList {
			dest := pl.Malloc()
			doSampling(DATADIR+"/fMRI-Smoothed/"+file, ringBuffer[dest], pl.GetNP())
			pl.Push(dest)
		}

		pl.Close()
		pl.StopScheduler()

		return
	}()

	avgedMat := mat64.NewDense(13362, 13362, nil)

	{
		accedMat := mat64.NewDense(13362, 13362, nil)
		{
			temp0 := mat64.NewDense(13362, 600, nil)
			temp1 := mat64.NewDense(13362, 600, nil)
			temp2 := mat64.NewDense(13362, 13362, nil)

			for {
				job, ok := pl.Pop()
				if ok {
					pl.ZScoring(ringBuffer[job], temp0)
					pl.Sigmoid(temp0, temp1)
					pl.Pearson(temp1, temp2)
					pl.Acc(temp2, accedMat)

					pl.Free(job)
				} else {
					break
				}
			}
		}
		pl.Avg(accedMat, avgedMat, float64(len(fileList)))
	}

	fmt.Println("Writing C2-tilda")
	// io.Mat64toCSV(RESULTDIR+"/c2-tilda.csv", avgedMat)
	io.Mat64toNpy(RESULTDIR+"/c2-tilda.csv", avgedMat)

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

	var thr float64
	for thr = thrStart; thr < thrEnd; thr += thrItv {
		fmt.Printf("Processing: thr - %f\n", thr)
		pl.Threshold(avgedMat, thredMat, thr, gamma)
		pl.Laplacian(thredMat) // Now thredMat is a Laplacian matrix

		mat64tocArr(thredMat, pMatBuffer) // Copy thresholded matrix to MAGMA matrix buffer

		fmt.Printf("Diagonalizing...")
		// Call MAGMA
		cmd := exec.Command("files/magma", "13362", fmt.Sprintf("%d", matBufferShm.Id), fmt.Sprintf("%d", eigValShm.Id))
		err := cmd.Run()
		if err != nil {
			log.Fatalf("MAGMA execution has failed: %s\n", err)
		}

		// Copy result from MAGMA to eigVal and eigVec
		cArrtomat64(eigVal, pEigVal)
		cArrtomat64(eigVec, pMatBuffer) // eigVec rows are eigen vectors; eigVec[0] <- first eigen vector

		fmt.Println("Writing Eigen value")
		// io.Mat64toCSV(RESULTDIR+"/eigen-value-thr-"+fmt.Sprintf("%f", thr)+".csv", eigVal)
		io.Mat64toCSV(RESULTDIR+"/eigVal-thr-"+fmt.Sprintf("%f", thr)+"-gamma-"+fmt.Sprintf("%f", gamma)+".csv", eigVal)
		io.Mat64toNpy(RESULTDIR+"/eigVal-thr-"+fmt.Sprintf("%f", thr)+"-gamma-"+fmt.Sprintf("%f", gamma)+".npy", eigVal)
		fmt.Println("Writing Eigen vector")
		// io.Mat64toCSV(RESULTDIR+"/eigen-vector-thr-"+fmt.Sprintf("%f", thr)+".csv", eigVec)
		io.Mat64toNpy(RESULTDIR+"/eigVec-thr-"+fmt.Sprintf("%f", thr)+"-gamma-"+fmt.Sprintf("%f", gamma)+".npy", eigVec)

		fmt.Println("---- ---- ---- ---- ---- ---- ---- ----")
	}

	matBufferShm.Detach(pMatBuffer)
	matBufferShm.Destroy()

	eigValShm.Detach(pEigVal)
	eigValShm.Destroy()

	return
}
