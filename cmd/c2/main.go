package main

import (
	"os"

	"github.com/KyungWonPark/Correlation/internal/calc"
	"github.com/KyungWonPark/Correlation/internal/io"
	"github.com/gonum/matrix/mat64"
)

func main() {

	DATADIR := os.Getenv("DATA")
	RESULTDIR := os.Getenv("RESULT")

	numQueueSize := 4
	ringBuffer := make([]*mat64.Dense, numQueueSize)
	for i := 0; i < numQueueSize; i++ {
		ringBuffer[i] = mat64.NewDense(13362, 13362, nil)
	}

	pl := calc.Init(numQueueSize, true)

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

	accMat := mat64.NewDense(13362, 13362, nil)
	finalMat := mat64.NewDense(13362, 13362, nil)

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
				pl.Acc(temp2, accMat)

				pl.Free(job)
			} else {
				break
			}
		}
	}

	pl.Avg(accMat, finalMat, float64(len(fileList)))
	io.Mat64toCSV(RESULTDIR+"/C2.csv", finalMat)

	return
}
