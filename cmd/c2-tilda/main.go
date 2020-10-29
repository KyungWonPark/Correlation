package main

import (
	"os"

	"github.com/KyungWonPark/Correlation/internal/calc"
	"github.com/KyungWonPark/Correlation/internal/io"
	"github.com/gonum/matrix/mat64"
)

func main() { // thrStart thrEnd thrItv isDebugMode
	DATADIR := os.Getenv("DATA")
	RESULTDIR := os.Getenv("RESULT")
	RESULTDIR = RESULTDIR + "/dump"

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

	{
		{
			temp0 := mat64.NewDense(13362, 600, nil)
			temp1 := mat64.NewDense(13362, 600, nil)
			temp2 := mat64.NewDense(13362, 13362, nil)

			for {
				job, ok := pl.Pop()
				if ok {
					io.Mat64toCSV(RESULTDIR+"/100610-sampled.csv", ringBuffer[job])
					pl.ZScoring(ringBuffer[job], temp0)
					io.Mat64toCSV(RESULTDIR+"/100610-zscored.csv", temp0)
					pl.Sigmoid(temp0, temp1)
					io.Mat64toCSV(RESULTDIR+"/100610-sigmoided.csv", temp1)
					pl.Pearson(temp1, temp2)
					io.Mat64toCSV(RESULTDIR+"/100610-pearsoned.csv", temp2)

					pl.Free(job)
				} else {
					break
				}
			}
		}
	}

	return
}
