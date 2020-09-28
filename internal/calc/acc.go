package calc

import (
	"log"
	"sync"

	"github.com/gonum/matrix/mat64"
)

func acc(inputMat *mat64.Dense, outputMat *mat64.Dense, order <-chan int, wg *sync.WaitGroup) {
	_, inputCols := inputMat.Dims()

	for {
		index, ok := <-order
		if ok {
			for t := 0; t < inputCols; t++ {
				value := outputMat.At(index, t) + inputMat.At(index, t)
				outputMat.Set(index, t, value)
			}

			wg.Done()
		} else {
			break
		}
	}

	return
}

// Acc does accumulation
func (p *PipeLine) Acc(inputMat *mat64.Dense, outputMat *mat64.Dense) {
	inputRows, inputCols := inputMat.Dims()
	outputRows, outputCols := outputMat.Dims()

	if inputRows != outputRows || inputCols != outputCols {
		log.Fatalf("[ERROR] Acc: input dims: %d by %d when output dims: %d by %d\n", inputRows, inputCols, outputRows, outputCols)
	}

	order := make(chan int, p.numPoper)
	var wg sync.WaitGroup

	wg.Add(inputRows)

	for i := 0; i < p.numPoper; i++ {
		go acc(inputMat, outputMat, order, &wg)
	}

	for i := 0; i < inputRows; i++ {
		order <- i
	}

	wg.Wait()
	close(order)
	return
}
