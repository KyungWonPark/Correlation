package main

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"sort"

	"github.com/KyungWonPark/Correlation/internal/anal"
	"github.com/KyungWonPark/Correlation/internal/calc"
	"github.com/KyungWonPark/Correlation/internal/io"
	"github.com/ghetzel/shmtool/shm"
	"github.com/gonum/matrix/mat64"
)

func main() {
	DATADIR := os.Getenv("DATA")
	RESULTDIR := os.Getenv("RESULT")

	numQueueSize := 8
	ringBuffer := make([]*mat64.Dense, numQueueSize)
	for i := 0; i < numQueueSize; i++ {
		ringBuffer[i] = mat64.NewDense(13362, 600, nil)
	}

	pl := calc.Init(numQueueSize, false)

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

	thredMat := mat64.NewDense(13362, 13362, nil)

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

	eigVal := mat64.NewDense(13362, 1, nil)
	eigVec := mat64.NewDense(13362, 13362, nil)

	var thr float64
	for thr = 0; thr < 0.015; thr += 0.05 {
		pl.Threshold(avgedMat, thredMat, thr)
		pl.Laplacian(thredMat)

		mat64tocArr(thredMat, pMatBuffer)

		// Call MAGMA
		cmd := exec.Command("files/magma", "13362", fmt.Sprintf("%d", matBufferShm.Id), fmt.Sprintf("%d", eigValShm.Id))
		err := cmd.Run()
		if err != nil {
			log.Fatalf("MAGMA execution has failed: %s\n", err)
		}

		cArrtomat64(eigVal, pEigVal)
		cArrtomat64(eigVec, pMatBuffer) // eigVec == U^T

		fmt.Printf("Threshold: %f Checking results...\n", thr)

		eigValMat := mat64.NewDense(13362, 13362, nil)
		for i := 0; i < 13362; i++ {
			eigValMat.Set(i, i, eigVal.At(i, 0))
		}

		// A * U
		result0 := mat64.NewDense(13362, 13362, nil)
		result0.Mul(eigVec, thredMat)

		// U * S
		result1 := mat64.NewDense(13362, 13362, nil)
		result1.Mul(eigValMat, eigVec)

		isSame := mat64.EqualApprox(result0, result1, 0.000001)
		if !isSame {
			fmt.Printf("Thr: %f / Eigenproblem failed!!\n", thr)
			diff := mat64.NewDense(13362, 13362, nil)
			diff.Sub(result0, result1)
			fmt.Printf("Diff: %g\n", mat64.Max(diff))
		}

		nZSEigVal, nZSEigValIdx := anal.GetNonZeroSmallestEigVal(eigVal)

		tmp := make([]float64, 13362)
		eigVecStrip := mat64.NewDense(13362, 1, tmp)

		for i := 0; i < 13362; i++ {
			eigVecStrip.Set(i, 0, eigVec.At(nZSEigValIdx, i))
		}

		sort.Float64s(tmp)
		fmt.Printf("Smallest Non-Zero EigenValue: %g\n", nZSEigVal)

		io.Mat64toCSV(RESULTDIR+"/clustering-thr-"+fmt.Sprintf("%f", thr)+".csv", eigVecStrip)
	}

	matBufferShm.Detach(pMatBuffer)
	matBufferShm.Destroy()

	eigValShm.Detach(pEigVal)
	eigValShm.Destroy()

	return
}
