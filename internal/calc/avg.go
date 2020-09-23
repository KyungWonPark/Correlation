package calc

import (
	"log"
	"sync"

	"github.com/gonum/matrix/mat64"
)

func avg(inputMat *mat64.Dense, outputMat *mat64.Dense, div float64, order <-chan int, wg *sync.WaitGroup) {
	_, inputCols := inputMat.Dims()

	for {
		index, ok := <-order
		if ok {
			for t := 0; t < inputCols; t++ {
				value := inputMat.At(index, t) / div
				outputMat.Set(index, t, value)
			}

			wg.Done()
		} else {
			break
		}
	}

	return
}

// Avg does averaging
func (p *PipeLine) Avg(inputMat *mat64.Dense, outputMat *mat64.Dense, div float64) {
	inputRows, inputCols := inputMat.Dims()
	outputRows, outputCols := outputMat.Dims()

	if inputRows != outputRows || inputCols != outputCols {
		log.Fatalf("[ERROR] Sigmoid: input dims: %d by %d when output dims: %d by %d\n", inputRows, inputCols, outputRows, outputCols)
	}

	order := make(chan int, p.numPoper)
	var wg sync.WaitGroup

	wg.Add(inputRows)

	for i := 0; i < p.numPoper; i++ {
		go avg(inputMat, outputMat, div, order, &wg)
	}

	for i := 0; i < inputRows; i++ {
		order <- i
	}

	wg.Wait()
	close(order)
	return
}
