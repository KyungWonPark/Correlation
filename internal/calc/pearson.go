package calc

import (
	"log"
	"math"
	"sync"

	"github.com/gonum/matrix/mat64"
)

func pearson(timeSeriesMat *mat64.Dense, pearsonMat *mat64.Dense, stats []statistic, order <-chan int, wg *sync.WaitGroup) {
	inputRows, inputCols := timeSeriesMat.Dims()

	for {
		from, ok := <-order
		if ok {
			for to := from; to < inputRows; to++ {
				var accProd float64
				for t := 0; t < inputCols; t++ {
					accProd += timeSeriesMat.At(from, t) * timeSeriesMat.At(to, t)
				}

				cov := (accProd / float64(inputCols)) - (stats[from].avg * stats[to].avg)
				pearson := cov / (stats[from].std * stats[to].std)

				pearsonMat.Set(from, to, pearson)
				pearsonMat.Set(to, from, pearson)
			}

			wg.Done()
		} else {
			break
		}
	}

	return
}

func getStat(timeSeriesMat *mat64.Dense, stats []statistic, order <-chan int, wg *sync.WaitGroup) {
	_, numCols := timeSeriesMat.Dims()
	for {
		index, ok := <-order
		if ok {
			var accVal float64
			var accSqrVal float64

			for t := 0; t < numCols; t++ {
				value := timeSeriesMat.At(index, t)
				accVal += value
				accSqrVal += value * value
			}

			avgVal := accVal / float64(numCols)
			avgSqrVal := accSqrVal / float64(numCols)

			stats[index].avg = avgVal
			stats[index].std = math.Sqrt(avgSqrVal - (avgVal * avgVal))

			wg.Done()
		} else {
			break
		}
	}

	return
}

// Pearson does Pearson's correlation calculation
func (p *PipeLine) Pearson(timeSeriesMat *mat64.Dense, outputMat *mat64.Dense) {
	inputRows, inputCols := timeSeriesMat.Dims()
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
			go getStat(timeSeriesMat, stats, order, &wg)
		}

		for i := 0; i < inputRows; i++ {
			order <- i
		}

		wg.Wait()
		close(order)
	}

	{ // Calculate Pearson's correlation
		order := make(chan int, p.numPoper)
		var wg sync.WaitGroup

		wg.Add(inputRows)

		for i := 0; i < p.numPoper; i++ {
			go pearson(timeSeriesMat, outputMat, stats, order, &wg)
		}

		for i := 0; i < inputRows; i++ {
			order <- i
		}

		wg.Wait()
		close(order)
	}

	return
}
