package calc

import (
	"log"
	"sync"

	"github.com/gonum/matrix/mat64"
)

func zScoring(inputMat *mat64.Dense, outputMat *mat64.Dense, stats []statistic, order <-chan int, wg *sync.WaitGroup) {
	_, inputCols := inputMat.Dims()

	for {
		index, ok := <-order
		if ok {
			for t := 0; t < inputCols; t++ {
				value := inputMat.At(index, t)
				newValue := (value - stats[index].avg) / stats[index].std
				outputMat.Set(index, t, newValue)
			}

			wg.Done()
		} else {
			break
		}
	}

	return
}

// ZScoring does z-scoring on each rows
func (p *PipeLine) ZScoring(inputMat *mat64.Dense, outputMat *mat64.Dense) {
	inputRows, inputCols := inputMat.Dims()
	outputRows, outputCols := outputMat.Dims()

	{ // Check input matrix and output matrix dimensions
		if outputRows != inputRows || outputCols != inputRows {
			log.Fatalf("[ERROR] Pearson: Input is %d by %d but output is %d by %d\n", inputRows, inputCols, outputRows, outputCols)
		}
	}

	stats := make([]statistic, inputRows)

	{ // Get statistics for each voxel timeseries
		order := make(chan int, p.numPoper)
		var wg sync.WaitGroup

		wg.Add(inputRows)

		for i := 0; i < p.numPoper; i++ {
			go getStat(inputMat, stats, order, &wg)
		}

		for i := 0; i < inputRows; i++ {
			order <- i
		}

		wg.Wait()
		close(order)
	}

	{ // Z-Scoring
		order := make(chan int, p.numPoper)
		var wg sync.WaitGroup

		wg.Add(inputRows)

		for i := 0; i < p.numPoper; i++ {
			go zScoring(inputMat, outputMat, stats, order, &wg)
		}

		for i := 0; i < inputRows; i++ {
			order <- i
		}

		wg.Wait()
		close(order)
	}

	return
}
