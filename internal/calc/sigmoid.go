package calc

import (
	"log"
	"math"
	"sync"

	"github.com/gonum/matrix/mat64"
)

func sigmoid(timeSeriesMat *mat64.Dense, outputMat *mat64.Dense, order <-chan int, wg *sync.WaitGroup) {
	_, inputCols := timeSeriesMat.Dims()

	for {
		index, ok := <-order
		if ok {
			for t := 0; t < inputCols; t++ {
				value := timeSeriesMat.At(index, t)
				newValue := (2 / (1 + math.Exp(-value))) - 1
				outputMat.Set(index, t, newValue)
			}

			wg.Done()
		} else {
			break
		}
	}

	return
}

// Sigmoid does sigmoid calculation
func (p *PipeLine) Sigmoid(timeSeriesMat *mat64.Dense, outputMat *mat64.Dense) {
	inputRows, inputCols := timeSeriesMat.Dims()
	outputRows, outputCols := outputMat.Dims()

	if inputRows != outputRows || inputCols != outputCols {
		log.Fatalf("[ERROR] Sigmoid: input dims: %d by %d when output dims: %d by %d\n", inputRows, inputCols, outputRows, outputCols)
	}

	order := make(chan int, p.numPoper)
	var wg sync.WaitGroup

	wg.Add(inputRows)

	for i := 0; i < p.numPoper; i++ {
		go sigmoid(timeSeriesMat, outputMat, order, &wg)
	}

	for i := 0; i < inputRows; i++ {
		order <- i
	}

	wg.Wait()
	close(order)
	return
}
