package main

import (
	"fmt"
	"os"

	"github.com/KyungWonPark/Correlation/internal/calc"
	"github.com/KyungWonPark/Correlation/internal/io"
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

	var thr float64
	for thr = 0; thr < 1; thr += 0.05 {
		pl.Threshold(avgedMat, thredMat, thr)

		fmt.Printf("Threshold: %f Writing results...\n", thr)
		io.Mat64toCSV(RESULTDIR+"/c2-thr-"+fmt.Sprintf("%f", thr)+".csv", thredMat)
	}

	return
}
