package calc

import (
	"log"
	"sync"

	"github.com/gonum/matrix/mat64"
)

func threshold(inputMat *mat64.Dense, outputMat *mat64.Dense, thr float64, sub float64, order <-chan int, wg *sync.WaitGroup) {
	_, inputCols := inputMat.Dims()

	for {
		index, ok := <-order
		if ok {
			for t := 0; t < inputCols; t++ {
				value := inputMat.At(index, t)
				if thr >= value {
					value = sub
				}

				outputMat.Set(index, t, value)
			}

			wg.Done()
		} else {
			break
		}
	}

	return
}

// Threshold does thresholding
func (p *PipeLine) Threshold(inputMat *mat64.Dense, outputMat *mat64.Dense, thr float64, sub float64) {
	inputRows, inputCols := inputMat.Dims()
	outputRows, outputCols := outputMat.Dims()

	if inputRows != outputRows || inputCols != outputCols {
		log.Fatalf("[ERROR] Threshold: input dims: %d by %d when output dims: %d by %d\n", inputRows, inputCols, outputRows, outputCols)
	}

	order := make(chan int, p.numPoper)
	var wg sync.WaitGroup

	wg.Add(inputRows)

	for i := 0; i < p.numPoper; i++ {
		go threshold(inputMat, outputMat, thr, sub, order, &wg)
	}

	for i := 0; i < inputRows; i++ {
		order <- i
	}

	wg.Wait()
	close(order)
	return
}
